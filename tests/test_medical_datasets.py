import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from openpyxl import Workbook
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_medical", "prepare_medical.py")
dataset_module = load("dataset_medical", "data/dataset_medical.py")


def sheet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    book = Workbook()
    for row in rows:
        book.active.append(row)
    book.save(path)
    book.close()


class MedicalTests(unittest.TestCase):
    def test_mosmed_official_split_pairing_and_overlap_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, prefix in [("train", "Train"), ("val", "Val"), ("test", "Test")]:
                name = f"{split}.png"
                xlsx = f"{prefix}_text_MosMedData{'+' if split == 'test' else ''}.xlsx"
                sheet(root / "annotations" / xlsx, [("Image", "text"), (name, f"{split} lesion")])
                for kind in ["images", "masks"]:
                    path = root / "raw" / kind / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("L", (4, 4), 255).save(path)
            prepare.make_manifests(root, "mosmed")
            record = json.loads((root / "anns/train.json").read_text())[0]
            self.assertEqual(record["captions"], ["train lesion"])
            self.assertEqual(record["image"], "raw/images/train.png")
            self.assertEqual(record["mask"], "raw/masks/train.png")
            sheet(root / "annotations/Val_text_MosMedData.xlsx",
                  [("Image", "text"), ("train.png", "duplicate lesion")])
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                prepare.make_manifests(root, "mosmed")

    def test_qata_id_sheets_are_used_without_random_resplitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sheet(root / "annotations/Train_Val_text_for_Covid19.xlsx",
                  [("Image", "Description"), ("mask_a.png", "left lesion"),
                   ("mask_b.png", "right lesion")])
            sheet(root / "annotations/Train_ID_for_Covid19.xlsx", [("Image",), ("mask_b.png",)])
            sheet(root / "annotations/Val_ID_for_Covid19.xlsx", [("Image",), ("mask_a.png",)])
            sheet(root / "annotations/Test_text_for_Covid19.xlsx",
                  [("Image", "Description"), ("mask_c.png", "test lesion")])
            for version in ["QaTa-COV19-v1", "QaTa-COV19-v2"]:
                raw = root / "raw" / "QaTa-COV19" / version
                raw.mkdir(parents=True)
                for name in ["a", "b", "c"]:
                    Image.new("L", (4, 4), 100).save(raw / f"{name}.png")
                    Image.new("L", (4, 4), 255).save(raw / f"mask_{name}.png")
            prepare.make_manifests(root, "qata")
            train_record = json.loads((root / "anns/train.json").read_text())[0]
            self.assertEqual(train_record["captions"], ["right lesion"])
            self.assertIn("QaTa-COV19-v2", train_record["image"])
            self.assertEqual(json.loads((root / "anns/val.json").read_text())[0]["captions"],
                             ["left lesion"])

    def test_adapter_returns_training_validation_batch_contract(self):
        def transform(image, mask):
            return torch.zeros(3, 4, 4), torch.tensor(np.array(mask), dtype=torch.long)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "QaTa-COV19"
            (root / "anns").mkdir(parents=True)
            Image.new("L", (4, 4), 123).save(root / "image.png")
            Image.fromarray(np.array([[0, 0, 255, 255]] * 4, dtype=np.uint8)).save(root / "mask.png")
            records = [{"image": "image.png", "mask": "mask.png", "captions": ["left opacity"]}]
            (root / "anns/train.json").write_text(json.dumps(records))
            args = SimpleNamespace(medical_data_root=directory, dataset="qata", image_size=4)
            stub = SimpleNamespace(BlipImageTrainProcessor=lambda **kwargs: lambda image: torch.zeros(3, 224, 224))
            with patch.dict(sys.modules, {"data.dataset_refer_bert": stub}):
                for evaluating in [False, True]:
                    dataset = dataset_module.MedicalReferDataset(
                        args, "train", transform, transform, eval_mode=evaluating)
                    item = dataset[0]
                    self.assertEqual(len(item), 7)
                    self.assertEqual(item[3], "left opacity")
                    self.assertEqual(item[4].shape, (3, 224, 224))
                    self.assertEqual(set(item[2].unique().tolist()), {0, 1})
                    if evaluating:
                        self.assertEqual(item[6]["raw_imgs"].shape, (4, 4, 3))
                        self.assertEqual(item[6]["org_size"].tolist(), [4, 4])
                    else:
                        self.assertEqual(item[6], {})


if __name__ == "__main__":
    unittest.main()
