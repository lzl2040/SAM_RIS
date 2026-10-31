import ast
import copy
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.utils.data import ConcatDataset, Dataset


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, names, namespace):
    """Load real definitions without importing GPU/data/visualization dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = [copy.deepcopy(node) for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


class Tokenized(SimpleNamespace):
    def to(self, device):
        return self


class Bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 768)
        self.calls = 0

    def forward(self, input_ids, query_embeds, **kwargs):
        self.calls += 1
        text = self.embedding(input_ids)
        query = query_embeds + text.mean(dim=1, keepdim=True)
        if "encoder_hidden_states" in kwargs:
            query = query + kwargs["encoder_hidden_states"].mean(dim=1, keepdim=True)
        text = text + query.mean(dim=1, keepdim=True)
        return SimpleNamespace(last_hidden_state=torch.cat([query, text], dim=1))


def qformer_class():
    namespace = load_definitions(
        "lavis/models/blip2_models/blip2_qformer.py", ["Blip2Qformer"],
        {"Blip2Base": nn.Module, "registry": SimpleNamespace(register_model=lambda _: lambda cls: cls),
         "torch": torch, "BlipOutputFeatures": lambda **kwargs: SimpleNamespace(**kwargs)},
    )
    cls = namespace["Blip2Qformer"]
    # Keep the real feature-extraction methods; replace only heavyweight initialization.
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.visual_encoder = nn.Linear(4, 768)
    model.ln_vision = nn.Identity()
    model.Qformer = nn.Module()
    model.Qformer.bert = Bert()
    model.query_tokens = nn.Parameter(torch.randn(1, 4, 768))
    model.device = "cpu"
    model.maybe_autocast = lambda: torch.autocast("cpu", enabled=False)
    model.tokenizer = lambda *args, **kwargs: Tokenized(
        input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
    return model


class ControlledAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.parser = runpy.run_path(str(ROOT / "args.py"))["get_parser"]()

    def model(self, mode, output="text"):
        qformer = qformer_class()
        sam = nn.Module()
        sam.sam_prompt_encoder = nn.Module()
        sam.sam_prompt_encoder.no_mask_embed = nn.Embedding(1, 256)
        sam.sam_mask_decoder = nn.Linear(256, 1)
        sam.image_encoder = nn.Module()
        sam.image_encoder.fusion = nn.Linear(768, 768)
        sam.image_encoder.backbone = nn.Linear(4, 4)
        namespace = load_definitions(
            "model/SAM2_RIS.py", ["FeatureResizer", "SAM2_RIS"],
            {"nn": nn, "torch": torch, "os": os, "load_blip2_qformer": lambda *args: qformer},
        )
        args = self.parser.parse_args(["--ablation", mode, "--tmfm_output", output])
        return namespace["SAM2_RIS"](sam, args, device="cpu")

    def test_four_modes_keep_qformer_and_route_gradients(self):
        for mode, mma, tmfm in [
            ("baseline", False, False), ("mma_only", True, False),
            ("tmfm_only", False, True), ("full", True, True),
        ]:
            with self.subTest(mode=mode):
                model = self.model(mode)
                query, prompts, _, _ = model.generate_queries(torch.randn(1, 2, 4), ["object"])
                self.assertEqual(query is not None, mma)
                self.assertEqual(model.blip2_qformer.Qformer.bert.calls, 2 if tmfm else 1)
                self.assertEqual(prompts.shape, (1, 3 if tmfm else 4, 256))
                self.assertEqual(model.sam2_model.image_encoder.fusion.weight.requires_grad, mma)
                self.assertFalse(model.sam2_model.image_encoder.backbone.weight.requires_grad)
                prompts.square().mean().backward()
                self.assertIsNotNone(model.blip2_qformer.query_tokens.grad)
                self.assertIsNotNone(model.blip2_qformer.Qformer.bert.embedding.weight.grad)
                self.assertIsNone(model.blip2_qformer.visual_encoder.weight.grad)
                with torch.no_grad():
                    self.assertFalse(model.generate_queries(torch.randn(1, 2, 4), ["object"])[1].requires_grad)

    def test_tmfm_tokens_and_legacy_no_grad_api(self):
        for output, length in [("text", 3), ("query", 4), ("all", 7), ("cls", 1)]:
            with self.subTest(output=output):
                model = self.model("full", output)
                self.assertEqual(model.generate_queries(torch.randn(1, 2, 4), ["object"])[1].shape[1], length)
        model = qformer_class()
        result, _ = model.extract_features({"image": torch.randn(1, 2, 4), "text_input": ["object"]})
        self.assertFalse(result.multimodal_embeds.requires_grad)

    def test_checkpoint_retains_trainable_embeddings_and_rejects_wrong_mode(self):
        namespace = load_definitions(
            "train_with_val.py", ["make_checkpoint", "load_resume_checkpoint"],
            {"torch": torch, "os": os, "move_optimizer_to_device": Mock()},
        )
        model = self.model("baseline")
        state = Mock(state_dict=lambda: {})
        checkpoint, saved = namespace["make_checkpoint"](model, state, state, state, 0, .5, 1.)
        self.assertIn("blip2_qformer.query_tokens", saved)
        self.assertIn("sam2_model.sam_prompt_encoder.no_mask_embed.weight", saved)
        self.assertEqual(checkpoint["experiment_config"]["ablation"], "baseline")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save(checkpoint, path)
            restored = self.model("baseline")
            namespace["load_resume_checkpoint"](path, restored, state, state, state, "cpu", Mock())
            torch.testing.assert_close(restored.blip2_qformer.query_tokens,
                                       model.blip2_qformer.query_tokens)
            for other in [self.model("full"), self.model("mma_only")]:
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    namespace["load_resume_checkpoint"](path, other, state, state, state, "cpu", Mock())
            checkpoint.pop("experiment_config")
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                namespace["load_resume_checkpoint"](path, model, state, state, state, "cpu", Mock())

    def test_mixed_filter_removes_cross_dataset_overlap_and_updates_lengths(self):
        class ReferDataset(Dataset):
            def __init__(self, args, **kwargs):
                ids = {"refcoco": [1, 2], "refcoco+": [2, 3], "refcocog": [3, 4]}[args.dataset]
                self.refer = SimpleNamespace(Refs={
                    1: {"image_id": ids[0], "split": "train"},
                    2: {"image_id": ids[1], "split": "train"},
                    3: {"image_id": 1 if args.dataset == "refcocog" else 9, "split": "val"},
                })
                self.ref_ids = [1, 2]
                self.captions = [[str(value)] for value in ids]

            def __len__(self):
                return len(self.ref_ids)

        namespace = load_definitions("train_with_val.py", ["get_dataset"], {"ConcatDataset": ConcatDataset})
        args = self.parser.parse_args(["--exclude_eval_images"])
        with patch.dict("sys.modules", {"data.dataset_refer_bert": SimpleNamespace(ReferDataset=ReferDataset)}):
            dataset = namespace["get_dataset"]("train", None, None, args)
        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset.datasets[0].ref_ids, [2])
        self.assertEqual(dataset.datasets[0].captions, [["2"]])
        self.assertEqual(args.dataset, "all")

    def test_validation_follows_single_dataset_and_restores_args(self):
        args = self.parser.parse_args(["--dataset", "refcocog", "--splitBy", "umd"])
        captured = []

        def get_dataset(*unused, **kwargs):
            captured.append((kwargs["args"].dataset, kwargs["args"].splitBy))
            return []

        namespace = load_definitions(
            "train_with_val.py", ["build_refcoco_val_loader"],
            {"get_dataset": get_dataset, "torch": torch, "DataLoader": lambda **kwargs: kwargs},
        )
        # DataLoader receives the dataset as its first positional argument.
        namespace["DataLoader"] = lambda dataset, **kwargs: dataset
        namespace["build_refcoco_val_loader"](args, None, None)
        self.assertEqual(captured, [("refcocog", "umd")])
        self.assertEqual((args.dataset, args.splitBy), ("refcocog", "umd"))
        namespace["get_dataset"] = Mock(side_effect=RuntimeError("missing dataset"))
        with self.assertRaises(RuntimeError):
            namespace["build_refcoco_val_loader"](args, None, None)
        self.assertEqual((args.dataset, args.splitBy), ("refcocog", "umd"))

    def test_controlled_visuals_need_no_previous_caption_and_save_probability(self):
        import numpy as np
        from PIL import Image

        tree = ast.parse((ROOT / "test.py").read_text())
        node = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "args.controlled_visuals"
                    and any(isinstance(child, ast.Name) and child.id == "probabilities"
                            for child in ast.walk(node)))
        with tempfile.TemporaryDirectory() as directory:
            namespace = {
                "args": SimpleNamespace(controlled_visuals=True),
                "Image": Image, "np": np, "os": os, "img_save_path": directory,
                "probabilities": np.array([[[[.25, .75]]]]),
                "sam2_pred": np.array([[[0, 1]]], dtype=np.uint8),
                "obj_cap": "the object", "this_iou": .5,
            }
            exec(compile(ast.Module(body=[node], type_ignores=[]), "test.py", "exec"), namespace)
            self.assertEqual(np.array(Image.open(Path(directory) / "probability.png")).ravel().tolist(), [63, 191])
            self.assertEqual(np.array(Image.open(Path(directory) / "mask.png")).ravel().tolist(), [0, 255])
            self.assertIn("IoU:0.5", (Path(directory) / "caption.txt").read_text())


if __name__ == "__main__":
    unittest.main()
