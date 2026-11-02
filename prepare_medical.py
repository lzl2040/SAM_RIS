#!/usr/bin/env python
"""Download original medical data plus LViT texts/splits; create SAM_RIS manifests.

python prepare_medical.py --data_root /data2/lizuolei/data
python prepare_medical.py --data_root /data2/lizuolei/data --skip_download
Sources: https://github.com/kzxl0721/MedMARS
         https://github.com/HUANGLIZI/LViT#1-data-preparation
No captions or random train/val splits are synthesized.
"""

import argparse
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile


DATASETS = {
    "qata": {
        "folder": "QaTa-COV19",
        "kaggle": "aysendegerli/qatacov19-dataset",
        "texts": {
            "Train_Val_text_for_Covid19.xlsx": "14RWeP3e4sKmgkzeGxKzFKc3C9XFyQ2Lr",
            "Train_ID_for_Covid19.xlsx": "1YbJoozbbqWFnJgsoqPw-Pvqyi0col1HL",
            "Val_ID_for_Covid19.xlsx": "1EURgnb5dRF43gxcIEXlpp7el_rK2f6PP",
            "Test_text_for_Covid19.xlsx": "1XaD7U3ZJAnVdg7WO9AFUAkBrRqDEamub",
        },
    },
    "mosmed": {
        "folder": "MosMedDataPlus",
        "kaggle": "maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset",
        "texts": {
            "Train_text_MosMedData.xlsx": "1e9BJq4lcWfDUAeXpdbCnZE80Rx6fQC8N",
            "Val_text_MosMedData.xlsx": "12sUxVenEHYRAAZGNIwuvUAsn19ntK5nd",
            "Test_text_MosMedData+.xlsx": "16-8OttzEcjJO_0uzz6Z5i5swChLRH65Z",
        },
    },
}


def download(url, destination):
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if not zipfile.is_zipfile(partial):
        raise ValueError(f"Not a ZIP/XLSX download: {url}; partial file kept at {partial}")
    partial.replace(destination)


def spreadsheet(path, descriptions=True):
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    result = {}
    try:
        for row in workbook.active.iter_rows(min_row=2, values_only=True):
            if not row[0]:
                continue
            name = Path(str(row[0]).strip()).name
            if name in result:
                raise ValueError(f"Duplicate annotation: {path}: {name}")
            text = str(row[1]).strip() if descriptions and len(row) > 1 and row[1] else ""
            if descriptions and not text:
                raise ValueError(f"Missing description: {path}: {name}")
            result[name] = text
    finally:
        workbook.close()
    return result


def make_manifests(root, key):
    from PIL import Image

    annotation_dir = root / "annotations"
    if key == "qata":
        texts = spreadsheet(annotation_dir / "Train_Val_text_for_Covid19.xlsx")
        splits = {}
        for split, prefix in [("train", "Train"), ("val", "Val")]:
            ids = spreadsheet(annotation_dir / f"{prefix}_ID_for_Covid19.xlsx", descriptions=False)
            splits[split] = {name: texts[name] for name in ids}
        if set(splits["train"]) | set(splits["val"]) != set(texts):
            raise ValueError("QaTa train/val IDs do not cover the text annotations")
        splits["test"] = spreadsheet(annotation_dir / "Test_text_for_Covid19.xlsx")
    else:
        splits = {
            split: spreadsheet(annotation_dir / name)
            for split, name in [
                ("train", "Train_text_MosMedData.xlsx"), ("val", "Val_text_MosMedData.xlsx"),
                ("test", "Test_text_MosMedData+.xlsx"),
            ]
        }
    files = {}
    image_root = root / "raw"
    qata_v2 = image_root / "QaTa-COV19" / "QaTa-COV19-v2"
    if key == "qata" and qata_v2.is_dir():
        image_root = qata_v2
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            files.setdefault(path.name.lower(), []).append(path)
    seen = set()
    manifests = {}
    for split, descriptions in splits.items():
        records = []
        for name, caption in descriptions.items():
            if name in seen:
                raise ValueError(f"Image occurs in multiple splits: {name}")
            seen.add(name)
            image_name = name.removeprefix("mask_") if key == "qata" else name
            mask_candidates = files.get(name.lower(), [])
            image_candidates = files.get(image_name.lower(), [])
            if key == "mosmed":
                mask_candidates = [path for path in mask_candidates
                                   if any("mask" in part.lower() or "label" in part.lower()
                                          for part in path.relative_to(root / "raw").parts[:-1])]
                image_candidates = [path for path in image_candidates if path not in mask_candidates]
            if len(mask_candidates) != 1 or len(image_candidates) != 1:
                raise ValueError(
                    f"Ambiguous/missing image-mask pair for {name}: "
                    f"images={image_candidates}, masks={mask_candidates}")
            with Image.open(image_candidates[0]) as image, Image.open(mask_candidates[0]) as mask:
                if image.size != mask.size:
                    raise ValueError(f"Image/mask dimensions differ: {name}")
            records.append({
                "image": str(image_candidates[0].relative_to(root)),
                "mask": str(mask_candidates[0].relative_to(root)), "captions": [caption],
            })
        manifests[split] = records
    (root / "anns").mkdir(exist_ok=True)
    for split, records in manifests.items():
        destination = root / "anns" / f"{split}.json"
        temporary = destination.with_suffix(".json.part")
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(destination)
        print(f"{root.name} {split}: {len(records)} image/mask/text triples")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=Path, default=Path("/data2/lizuolei/data"))
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()
    for key in DATASETS if args.dataset == "all" else [args.dataset]:
        config = DATASETS[key]
        root = args.data_root / config["folder"]
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "dataset.zip"
        if not args.skip_download:
            download(f"https://www.kaggle.com/api/v1/datasets/download/{config['kaggle']}", archive)
            for name, file_id in config["texts"].items():
                download(f"https://drive.google.com/uc?export=download&id={file_id}",
                         root / "annotations" / name)
        marker = root / "raw" / ".extracted"
        if not marker.exists():
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    target = (root / "raw" / member.filename).resolve()
                    if not target.is_relative_to((root / "raw").resolve()):
                        raise ValueError(f"Unsafe archive entry: {member.filename}")
                source.extractall(root / "raw")
            marker.touch()
        make_manifests(root, key)


if __name__ == "__main__":
    main()
