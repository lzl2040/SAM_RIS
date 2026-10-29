#!/usr/bin/env bash
set -euo pipefail

# Format: https://github.com/jiaqihuang01/DETRIS/blob/main/tools/prepare_datasets.md
# Usage: bash prepare.sh [all|data|lmdb|weights]
MODE="${1:-all}"
case "$MODE" in
    all|data|lmdb|weights) ;;
    *) echo "Usage: bash prepare.sh [all|data|lmdb|weights]" >&2; exit 2 ;;
esac

REFER_DATA_ROOT="${REFER_DATA_ROOT:-/home/v-zuoleili/Data/coco}"
PRETRAIN_WEIGHT_ROOT="${PRETRAIN_WEIGHT_ROOT:-/home/v-zuoleili/Pretrain}"
PYTHON="${PYTHON:-python}"

download() {
    if ! wget -q -c "$1"; then
        echo "Download failed: $1 (rerun to resume)" >&2
        return 1
    fi
}

case "$REFER_DATA_ROOT" in
    /*) ;;
    *) REFER_DATA_ROOT="$PWD/$REFER_DATA_ROOT" ;;
esac
case "$PRETRAIN_WEIGHT_ROOT" in
    /*) ;;
    *) PRETRAIN_WEIGHT_ROOT="$PWD/$PRETRAIN_WEIGHT_ROOT" ;;
esac

if [[ "$MODE" == all || "$MODE" == data ]]; then
    mkdir -p "$REFER_DATA_ROOT"
    cd "$REFER_DATA_ROOT"
    if [[ ! -d train2014 && ! -d images/train2014 ]]; then
        download http://images.cocodataset.org/zips/train2014.zip
        "$PYTHON" -m zipfile -e train2014.zip .
        rm train2014.zip
    fi
    for dataset in refcoco 'refcoco+' refcocog; do
        case "$dataset" in
            refcoco) timestamp=20220413011718 ;;
            refcoco+) timestamp=20220413011656 ;;
            refcocog) timestamp=20220413012904 ;;
        esac
        if [[ ! -f "$dataset/instances.json" || ! -f "$dataset/refs(unc).p" && "$dataset" != refcocog ||
              "$dataset" == refcocog && ( ! -f "$dataset/refs(umd).p" || ! -f "$dataset/refs(google).p" ) ]]; then
            download "https://web.archive.org/web/$timestamp/https://bvisionweb1.cs.unc.edu/licheng/referit/data/$dataset.zip"
            "$PYTHON" -m zipfile -e "$dataset.zip" .
            rm "$dataset.zip"
        fi
    done
fi

if [[ "$MODE" == all || "$MODE" == data || "$MODE" == lmdb ]]; then
    "$PYTHON" - <<'PY'
try:
    import cv2
    import lmdb
    import matplotlib
    import numpy as np
    import pyarrow as pa
    import pycocotools.mask
    import skimage
    import tqdm
    assert hasattr(pa, "serialize") and hasattr(pa, "deserialize")
except (ImportError, AssertionError) as error:
    raise SystemExit(
        "DETRIS conversion requires its legacy PyArrow serialization (not pickle).\n"
        "Use Python 3.10/3.11 and install in that environment:\n"
        "python -m pip install 'numpy<2' 'pyarrow==11.0.0' lmdb pycocotools "
        "'opencv-python-headless<4.12' matplotlib scikit-image tqdm\n"
        "Select that interpreter with PYTHON=/path/to/python."
    ) from error
PY

    # Pin upstream tools so later DETRIS changes cannot alter the on-disk format.
    DETRIS_REV=c94b1e1e5253f51eb3a924fe944a455ffe1d75d6
    DETRIS_TOOLS_DIR="${DETRIS_TOOLS_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/sam-ris/DETRIS/$DETRIS_REV/tools}"
    mkdir -p "$DETRIS_TOOLS_DIR"
    DETRIS_TOOLS_DIR="$(cd "$DETRIS_TOOLS_DIR" && pwd)"
    for tool in data_process.py folder2lmdb.py refer.py; do
        if [[ ! -s "$DETRIS_TOOLS_DIR/$tool" ]]; then
            curl --silent --show-error --fail --location --retry 3 \
                "https://raw.githubusercontent.com/jiaqihuang01/DETRIS/$DETRIS_REV/tools/$tool" \
                --output "$DETRIS_TOOLS_DIR/$tool.part"
            mv "$DETRIS_TOOLS_DIR/$tool.part" "$DETRIS_TOOLS_DIR/$tool"
        fi
    done

    "$PYTHON" - "$REFER_DATA_ROOT" "$DETRIS_TOOLS_DIR" <<'PY'
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import lmdb
import pyarrow as pa

root = Path(sys.argv[1])
tools = Path(sys.argv[2])
images = root / "train2014"
if not images.is_dir():
    images = root / "images/train2014"
if not images.is_dir():
    raise SystemExit(f"Missing COCO images under {root}; run prepare.sh data first.")

settings = [
    ("refcoco", "unc", "refcoco", ("train", "val", "testA", "testB")),
    ("refcoco+", "unc", "refcoco+", ("train", "val", "testA", "testB")),
    ("refcocog", "umd", "refcocog_u", ("train", "val", "test")),
    ("refcocog", "google", "refcocog_g", ("train", "val")),
]


def validate_database(path, length):
    with lmdb.open(str(path), subdir=False, readonly=True, lock=False) as database:
        with database.begin() as txn:
            keys = pa.deserialize(txn.get(b"__keys__"))
            if pa.deserialize(txn.get(b"__len__")) != length or len(keys) != length:
                raise ValueError(f"Incorrect LMDB length: {path}")
            if length and (txn.get(keys[0]) is None or txn.get(keys[-1]) is None):
                raise ValueError(f"Incomplete LMDB: {path}")


for dataset, split_by, output_name, splits in settings:
    annotations = root / "anns" / output_name
    masks = root / "masks" / output_name
    if not (all((annotations / f"{split}.json").is_file() for split in splits)
            and masks.is_dir()):
        # Stage each split separately; never rename/delete the original REFER data.
        with tempfile.TemporaryDirectory(prefix=f".prepare-{output_name}-", dir=root) as staging:
            subprocess.run([
                # Upstream uses np.alen, removed in NumPy 1.24; ref_ids is a list.
                sys.executable, "-c",
                "import os, runpy, sys, numpy as np; np.alen = len; "
                "sys.argv = sys.argv[1:]; sys.path.insert(0, os.path.dirname(sys.argv[0])); "
                "runpy.run_path(sys.argv[0], run_name='__main__')",
                str(tools / "data_process.py"),
                "--data_root", str(root), "--output_dir", staging,
                "--dataset", dataset, "--split", split_by, "--generate_mask",
            ], check=True)
            annotations.mkdir(parents=True, exist_ok=True)
            masks.mkdir(parents=True, exist_ok=True)
            for split in splits:
                shutil.copyfile(Path(staging) / "anns" / dataset / f"{split}.json",
                                annotations / f"{split}.json")
            for mask in (Path(staging) / "masks" / dataset).iterdir():
                shutil.copyfile(mask, masks / mask.name)

    output = root / "lmdb" / output_name
    output.mkdir(parents=True, exist_ok=True)
    for split in splits:
        annotation = annotations / f"{split}.json"
        with annotation.open() as handle:
            length = len(json.load(handle))
        if not length:
            raise ValueError(f"Refusing to generate an empty split: {annotation}")
        destination = output / f"{split}.lmdb"
        if destination.exists():
            validate_database(destination, length)
            print(f"Keeping existing valid LMDB: {destination}", flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix=f".{split}-", dir=output) as staging:
            subprocess.run([
                sys.executable, str(tools / "folder2lmdb.py"),
                "-j", str(annotation), "-i", str(images),
                "-m", str(masks), "-o", staging,
            ], check=True)
            generated = Path(staging) / f"{split}.lmdb"
            validate_database(generated, length)
            os.replace(generated, destination)
    print(f"Prepared {output_name}: {', '.join(splits)}", flush=True)

print(f"Dataset ready: {root}/{{train2014,refcoco,refcoco+,refcocog,anns,masks,lmdb}}")
PY
fi

if [[ "$MODE" == all || "$MODE" == weights ]]; then
    mkdir -p "$PRETRAIN_WEIGHT_ROOT"
    cd "$PRETRAIN_WEIGHT_ROOT"
    for model in small base_plus large; do
        filename="sam2.1_hiera_$model.pt"
        if [[ ! -s "$filename" ]]; then
            download "https://dl.fbaipublicfiles.com/segment_anything_2/092824/$filename"
        fi
    done
    for entry in BLIP2/eva_vit_g.pth InstructBLIP/instruct_blip_vicuna7b_trimmed.pth; do
        if [[ ! -s "${entry##*/}" ]]; then
            download "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/$entry"
        fi
    done
fi
