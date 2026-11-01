#!/usr/bin/env python
"""Benchmark the training validation path on a fixed subset, without training."""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time


def benchmark_options(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bench_samples", type=int, default=100,
                        help="Number of objects, not captions; 0 evaluates the full validation set.")
    parser.add_argument("--bench_warmup", type=int, default=3)
    parser.add_argument("--bench_batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--bench_repeats", type=int, default=1)
    parser.add_argument("--bench_json", type=Path, default=None)
    options, remaining = parser.parse_known_args(argv)
    if options.bench_samples < 0 or options.bench_warmup < 0:
        parser.error("bench_samples and bench_warmup must be nonnegative")
    if options.bench_repeats < 1 or any(value < 1 for value in options.bench_batches):
        parser.error("bench_repeats and bench_batches must be positive")
    return options, remaining


def main():
    options, remaining = benchmark_options(sys.argv[1:])
    from args import get_parser

    parser = get_parser()
    parser.description = (__doc__ + " Benchmark flags: --bench_samples 100 "
                          "--bench_batches 1 4 --bench_warmup 3 --bench_repeats 1 "
                          "--bench_json results.json. Other flags match train_with_val.py.")
    args = parser.parse_args(remaining)
    if not args.resume or not Path(args.resume).is_file():
        parser.error("--resume must point to an existing checkpoint")

    import torch
    from torch.utils.data import DataLoader, Subset
    from config import config2model_path
    from model.SAM2_RIS import SAM2_RIS
    from sam2.build_sam import build_sam2
    from train_with_val import build_refcoco_val_loader, get_transform, set_seed, validate_refcoco

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU.")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    saved_config = checkpoint.get("experiment_config")
    if saved_config:
        # Use the checkpoint's architecture; do not silently benchmark another ablation.
        for key in ("ablation", "model_cfg", "blip2_vit_model", "image_size"):
            setattr(args, key, saved_config[key])
        if saved_config.get("tmfm_output") is not None:
            args.tmfm_output = saved_config["tmfm_output"]

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("validation-benchmark")
    set_seed(args.seed)
    backbone = build_sam2(args.model_cfg, config2model_path[args.model_cfg],
                          device=device, apply_postprocessing=True)
    model = SAM2_RIS(backbone, args, device=device).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.unexpected_keys:
        logger.warning("Unexpected checkpoint keys: %s", incompatible.unexpected_keys)
    if not saved_config:
        logger.warning("Legacy checkpoint: architecture must be specified manually; "
                       "missing historical parameters may use pretrained initialization.")
    del checkpoint
    model.eval()

    transforms, cnn_transforms = get_transform(args)
    # The legacy dataset module parses sys.argv at import; remove benchmark-only flags.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        dataset = build_refcoco_val_loader(args, transforms, cnn_transforms).dataset
    finally:
        sys.argv = original_argv
    full_size = len(dataset)
    sample_count = min(options.bench_samples or full_size, full_size)
    if sample_count == 0:
        raise SystemExit("Validation dataset is empty.")

    def loader(count):
        return DataLoader(Subset(dataset, range(count)), batch_size=1, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    selected = Subset(dataset, range(sample_count))
    caption_count = sum(len(dataset.captions[index]) for index in selected.indices)
    metadata = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "checkpoint": str(Path(args.resume).resolve()),
        "ablation": args.ablation,
        "model_cfg": args.model_cfg,
        "image_size": args.image_size,
        "dataset": args.val_dataset or (args.dataset if args.dataset != "all" else "refcoco"),
        "amp": args.val_amp,
        "objects": sample_count,
        "captions": caption_count,
        "full_validation_objects": full_size,
    }
    logger.info("Benchmark settings: %s", json.dumps(metadata, ensure_ascii=False))
    results = []
    for batch in options.bench_batches:
        if options.bench_warmup:
            validate_refcoco(model, loader(min(options.bench_warmup, sample_count)), device,
                             logger, 0, 0, batch_size=batch, amp=args.val_amp)
        for repeat in range(options.bench_repeats):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            metrics = validate_refcoco(model, loader(sample_count), device, logger, 0, 0,
                                       batch_size=batch, amp=args.val_amp)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            result = {
                "batch_size": batch, "repeat": repeat + 1, "seconds": seconds,
                "captions_per_second": caption_count / seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3,
                "metrics": metrics,
            }
            results.append(result)
            logger.info("batch=%d repeat=%d: %.2fs, %.3f captions/s, %.2f GiB peak",
                        batch, repeat + 1, seconds, result["captions_per_second"],
                        result["peak_allocated_gib"])

    reference = next((result for result in results if result["batch_size"] == 1), results[0])
    for result in results:
        result["speedup_vs_reference"] = reference["seconds"] / result["seconds"]
        result["metric_deltas"] = {
            key: value - reference["metrics"][key] for key, value in result["metrics"].items()
        }
    report = {"settings": metadata, "reference_batch": reference["batch_size"], "results": results}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if options.bench_json:
        options.bench_json.parent.mkdir(parents=True, exist_ok=True)
        options.bench_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Match the training launcher's working directory and relative model resources.
    os.chdir(Path(__file__).resolve().parent)
    main()
