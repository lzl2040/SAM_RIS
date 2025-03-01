import os
import sys
import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from data.refer import REFER
import data.transforms as T

from args import get_parser
from lavis.processors.blip_processors import BlipImageBaseProcessor

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()


class BlipImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=364, mean=None, std=None, min_scale=0.5, max_scale=1.0, eval = False
    ):
        super().__init__(mean=mean, std=std)
        if eval:
            # transforms.RandomAffine
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    self.normalize
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )

    def __call__(self, item):
        return self.transform(item)

class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)
        self.blip2_transforms = BlipImageTrainProcessor(image_size=224)
        self.resize_transform = transforms.Resize((1024, 1024))

        if args.ref_image_combined_pretrain:
            self.max_tokens = 22
        else:
            self.max_tokens = 20

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.captions = []
        self.attention_masks = []

        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []
            captions_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                captions_for_ref.append(sentence_raw)

            self.captions.append(captions_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        # print(len(self.ref_ids), len(self.captions))
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")
        raw_img = np.array(self.resize_transform(img))
        blip2_img = self.blip2_transforms(img)

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img, target = self.image_transforms(img, annot)

        if self.eval_mode:
            captions = self.captions[index]
        else:

            caption_id = np.random.choice(len(self.captions[index]))
            captions = self.captions[index][caption_id]

        # return img, target, tensor_embeddings, attention_mask
        return img, target, captions, blip2_img, raw_img

if __name__ == '__main__':
    tr = T.Compose([T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ])
    dataset = ReferDataset(args, image_transforms=tr, eval_mode=False)
    loaders = DataLoader(dataset, batch_size=2)
    for img, target, captions, blip2_img, raw_img in loaders:
        print(captions)
        print(blip2_img.shape)
        print(img.shape)
        print(target.shape)
        print(raw_img.shape)
        break
