
from os.path import splitext
from os import listdir



import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf
import random
import numpy
import cv2
import albumentations as A
import pandas as pd
import sys
import os
from albumentations.pytorch.transforms import ToTensorV2

class MetaDatasetfs1000(Dataset):
    # def __init__(self, imgs_dir, masks_dir, scale=1, data_type='train'):
    def __init__(self, args, fileroots, mode='train'):
        self.args = args
        self.tasks = fileroots
        self.mode = mode

        self.relative_path = '/data1/wangjingtao/workplace/python/data/seg/meta'
        
        self.transform_train = A.Compose([
            A.Resize(width=self.args.dl_resize, height=self.args.dl_resize, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate((-25, 25), p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1,
            #                  num_flare_circles_lower=1, num_flare_circles_upper=2,
            #                  src_radius=160, src_color=(255, 255, 255), always_apply=False, p=0.3),
            # A.RGBShift(r_shift_limit=10, g_shift_limit=10,
            #            b_shift_limit=10, always_apply=False, p=0.2),
            # A.ElasticTransform(alpha=2, sigma=15, alpha_affine=25, interpolation=1,
            #                    border_mode=4, value=None, mask_value=None,
            #                    always_apply=False, approximate=False, p=0.3),
            ToTensorV2(),
        ])
        
        self.transform_val = A.Compose([
            A.Resize(width=self.args.meta_resize, height=self.args.meta_resize, p=1.0),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.tasks)


    def __getitem__(self, idx):
        task = self.tasks[idx]

        img_list, msk_list = [], []
        for each_path in task:
            img_path = os.path.join(self.relative_path, each_path[0])
            mask_path = os.path.join(self.relative_path, each_path[1])

            if self.args.n_channels == 1:
                image = cv2.imread(img_path, 0)  # [height,width,channel]
            else:
                 image = cv2.imread(img_path)

            mask = cv2.imread(mask_path, 0)  # [height, width]
            
            augmented = self.transform_val(image=image, mask=mask)
            image = augmented["image"] /255.0 # [c, h, w]
            mask = augmented["mask"] /255 # [h,w]
            mask = mask[None] # [c,h,w]

            img_list.append(image)
            msk_list.append(mask)

        img_tensor = torch.stack(img_list)  # 沿指定维度拼接,会多一维 [shot, channel, height, width]
        msk_tensor = torch.stack(msk_list)  # [shot, c, height, width]

        return [img_tensor, msk_tensor]



    