import os
import sys
import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from data.refer import REFER
# https://github.com/mydcxiao/ReferSAM/blob/main/utils/transforms.py
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
                 cnn_transform = None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.cnn_transform = cnn_transform
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)
        self.blip2_transforms = BlipImageTrainProcessor(image_size=224)
        self.resize_transform = transforms.Resize((args.image_size, args.image_size))
        # self.resize_transform = T.Compose([T.ResizeLongestSide(args.image_size // 4),
        #                  T.ToTensor(),
        #                  T.Pad(args.image_size // 4)
        #                  ])

        # if args.ref_image_combined_pretrain:
        #     self.max_tokens = 22
        # else:
        #     self.max_tokens = 20

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
        np_img = np.array(img)
        org_size = img.size # w h
        imgs_wo_trans = np.array(self.resize_transform(img)) # for visualize
        # imgs_wo_trans = self.resize_transform(np_img).permute(1, 2, 0).numpy() # from referSAM
        blip2_img = self.blip2_transforms(img)

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")
        raw_target = np.array(annot)

        if self.cnn_transform is not None:
            cnn_img, _ = self.cnn_transform(img, annot)

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, annot)

        # from refer SAM #
        # if self.image_transforms is not None:
        #     img = self.image_transforms(np_img)
        #
        # if self.target_transforms is not None:
        #     target = self.target_transforms(target)
        # from refer SAM #

        if self.eval_mode:
            captions_list = self.captions[index]
            captions = "#####".join(captions_list)
            # captions = self.captions[index]
        else:

            caption_id = np.random.choice(len(self.captions[index]))
            captions = self.captions[index][caption_id]

        # return img, target, tensor_embeddings, attention_mask
        info = {}
        if self.eval_mode:
            info["org_size"] = torch.as_tensor(org_size)
            info["raw_masks"] = raw_target
            info["raw_imgs"] = np_img
        return img, cnn_img, target, captions, blip2_img, imgs_wo_trans, info

if __name__ == '__main__':
    # from referSAM #
    # im_tr = T.Compose([T.ResizeLongestSide(args.image_size),
    #                     T.Normalize(),
    #                     T.ToTensor(),
    #                     T.Pad(args.image_size)
    #                     ])
    # target_tr = T.Compose([T.ResizeLongestSide(args.image_size // 4),
    #                      T.ToTensor(),
    #                      T.Pad(args.image_size // 4)
    #                      ])
    # from referSAM #
    im_tr = T.Compose([T.Resize(args.image_size, args.image_size),
                      T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # only normalize image
                 ])
    target_tr = None
    dataset = ReferDataset(args, image_transforms=im_tr, target_transforms=target_tr, eval_mode=True)
    loaders = DataLoader(dataset, batch_size=1)
    for img, target, captions, blip2_img, raw_img, info in loaders:
        # print(captions)
        for caption in captions:
            object_caption = caption.split("#####")
            print(object_caption)
        print(blip2_img.shape)
        print(img.shape)
        print(target.shape)
        print(raw_img.shape)
        raw_img = raw_img.detach().numpy()
        print(info["raw_masks"].shape)
        break
