import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class MedicalReferDataset(Dataset):
    """LViT image/mask/text triples, with the same batch interface as REFER."""

    def __init__(self, args, split, image_transforms, cnn_transform, eval_mode=False):
        from data.dataset_refer_bert import BlipImageTrainProcessor

        folder = {"qata": "QaTa-COV19", "mosmed": "MosMedDataPlus"}[args.dataset]
        self.root = Path(args.medical_data_root) / folder
        manifest = self.root / "anns" / f"{split}.json"
        with manifest.open() as handle:
            self.records = json.load(handle)
        if not self.records:
            raise ValueError(f"Empty medical split: {manifest}")
        self.captions = [record["captions"] for record in self.records]
        self.image_transforms = image_transforms
        self.cnn_transform = cnn_transform
        self.blip_transform = BlipImageTrainProcessor(image_size=224)
        self.image_size = args.image_size
        self.eval_mode = eval_mode

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["image"]) as source:
            image = source.convert("RGB")
        with Image.open(self.root / record["mask"]) as source:
            mask = (np.array(source.convert("L")) > 0).astype(np.uint8)
        if image.size != (mask.shape[1], mask.shape[0]):
            raise ValueError(f"Image/mask dimensions differ: {record['image']}")
        annotation = Image.fromarray(mask, mode="P")
        blip_image = self.blip_transform(image)
        cnn_image, _ = self.cnn_transform(image, annotation)
        transformed, target = self.image_transforms(image, annotation)
        captions = self.captions[index]
        caption = "#####".join(captions) if self.eval_mode else captions[np.random.randint(len(captions))]
        info = {}
        if self.eval_mode:
            info = {"org_size": torch.tensor(image.size), "raw_masks": mask,
                    "raw_imgs": np.array(image)}
        raw_resized = np.array(image.resize((self.image_size, self.image_size)))
        return transformed, cnn_image, target, caption, blip_image, raw_resized, info
