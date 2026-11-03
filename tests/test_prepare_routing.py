import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PrepareRoutingTests(unittest.TestCase):
    def test_medical_routes_options_without_touching_coco_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "record-python"
            python.write_text(
                f"#!{sys.executable}\n"
                "import json, sys\n"
                "print(json.dumps(sys.argv[1:]))\n")
            python.chmod(0o755)
            env = dict(os.environ, PYTHON=str(python),
                       MEDICAL_DATA_ROOT=str(root / "medical data"),
                       REFER_DATA_ROOT=str(root / "coco"),
                       PRETRAIN_WEIGHT_ROOT=str(root / "weights"))
            for options in [[], ["--dataset", "qata"],
                            ["--dataset", "mosmed", "--skip_download"]]:
                with self.subTest(options=options):
                    result = subprocess.run(
                        ["bash", str(ROOT / "prepare.sh"), "medical", *options],
                        cwd=root, env=env, capture_output=True, text=True, check=True)
                    self.assertEqual(json.loads(result.stdout), [
                        str(ROOT / "prepare_medical.py"),
                        "--data_root", str(root / "medical data"), *options,
                    ])
                    self.assertFalse((root / "coco").exists())
                    self.assertFalse((root / "weights").exists())

    def test_weights_mode_does_not_prepare_any_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights"
            weights.mkdir()
            for filename in ["sam2.1_hiera_small.pt", "sam2.1_hiera_base_plus.pt",
                             "sam2.1_hiera_large.pt", "eva_vit_g.pth",
                             "instruct_blip_vicuna7b_trimmed.pth"]:
                (weights / filename).write_bytes(b"cached weight")
            env = dict(os.environ, PYTHON="/nonexistent/python",
                       REFER_DATA_ROOT=str(root / "coco"),
                       MEDICAL_DATA_ROOT=str(root / "medical"),
                       PRETRAIN_WEIGHT_ROOT=str(weights))
            subprocess.run(["bash", str(ROOT / "prepare.sh"), "weights"],
                           cwd=root, env=env, capture_output=True, text=True, check=True)
            self.assertFalse((root / "coco").exists())
            self.assertFalse((root / "medical").exists())


if __name__ == "__main__":
    unittest.main()
