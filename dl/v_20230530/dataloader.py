from os.path import splitext
from os import listdir
import os
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


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, data_type='train'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.data_type = data_type
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.mask_file = glob(os.path.join(self.masks_dir, "**", "*.png"), recursive=True)
        self.img_file = [elm.replace(self.masks_dir, self.imgs_dir).replace("_label", "")[:-4] + ".*" for elm in
                         self.mask_file]
        for i in range(len(self.img_file)):
            img_file_i = glob(self.img_file[i])
            assert len(img_file_i) == 1, "multi img_file_i : %s" % self.img_file
            img_file_i = img_file_i[0]
            assert img_file_i.endswith("jpg") or img_file_i.endswith("png"), "img_file_i type error : %s" % img_file_i
            self.img_file[i] = img_file_i

        logging.info(f'Creating dataset with {len(self.mask_file)} examples')

    def __len__(self):
        return len(self.mask_file)

    def transform(self, image, mask):
        original_height = 512
        original_width = 512
        image5 = A.Compose([
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.OneOf([
            #     # A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
            #     A.GaussNoise(),    # 将高斯噪声应用于输入图像。
            # ], p=0.7),   # 应用选定变换的概率
            A.OneOf([
                A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
            ], p=0.7),
            # A.ElasticTransform(p=0.5, alpha=120*0.1, sigma=120*0.1 * 0.05, alpha_affine=120*0.1 * 0.03),
            # A.GridDistortion(p=0.5, num_steps=50),
            A.ShiftScaleRotate(shift_limit=0.0625 / 2, scale_limit=0.2 / 2, rotate_limit=15, p=0.5),
            # 随机应用仿射变换：平移，缩放和旋转输入
            # A.CLAHE(p=0.3), # 自适应直方图均衡化
            A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
        ])(image=image, mask=mask)
        return image5["image"], image5["mask"]

    def __getitem__(self, i):
        mask = cv2.imread(self.mask_file[i], 0)
        img = cv2.imread(self.img_file[i], 0)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)  # [heigth,wideth]
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)  # [heigth,wideth]

        # if self.data_type == 'train':
        #     img, mask = self.transform(img, mask)
        img = img[None]  # [ , heigth, wideth]
        # mask = mask[None]
        # print(img.dtype, mask.dtype)
        # print(img.max(), img.min())
        img = torch.from_numpy(img) / 255.0
        mask = torch.from_numpy(mask) / 255
        # print(img.size(), mask.size())

        return {'image': img, 'mask': mask}


class CSVBasicDataset(Dataset):
    def __init__(self, csv_dir='', dataframe=None):
        self.csv_dir = csv_dir
        if self.csv_dir == '' and dataframe is not None:
            self.dataframe = dataframe
        else:
            self.dataframe = pd.read_csv(self.csv_dir)

        self.img_file = list(self.dataframe["Image_path"])
        self.mask_file = list(self.dataframe["Label_path"])


        logging.info(f'Creating dataset with {len(self.mask_file)} examples')

    def __len__(self):
        return len(self.mask_file)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_file[idx], 0)  # [height,width,channel]
        mask = cv2.imread(self.mask_file[idx], 0)  # [height, width]
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
        # augmented = self.transform(image=image, mask=mask)
        # image = augmented["image"]
        # mask = augmented["mask"]

        image = image[None]
        # mask = mask[None] # [channel, height,width] # 输出是one_channel的时候才会用到
        mask = torch.from_numpy(mask) / 255
        image = torch.from_numpy(image) / 255.0
        # image = image.permute(2, 0, 1)  # 从[height,width,channel]变成[channel, height,width]
        return {'image': image, 'mask': mask}

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import cv2
    from tqdm import tqdm
    import torch.nn.functional as F
#     dir_img = r"../../data/BV1000_Segmentation/merge_onhlayer/test/image"
#     dir_mask = r"../../data/BV1000_Segmentation/merge_onhlayer/test/label"
#     img_scale = 1
#     dataset = BasicDataset(dir_img, dir_mask, img_scale)
#     train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
#     print(len(dataset))
#     # batch = dataset.__getitem__(500)
#     #
#     # imgs = batch['image']
#     # true_masks = batch['mask']
#     # print(true_masks.shape)
#     # cv2.imwrite("image.png", imgs[0].numpy()*255)
#     # cv2.imwrite("true_masks.png", (true_masks[0].numpy()*255).astype('uint8'))
#
#     for batch in train_loader:
#         print(batch["image"].shape, batch["mask"].shape)
#     #     # cv2.imwrite("image.png", batch["image"][0,0].numpy()*255)
#     #     # cv2.imwrite("mask.png", batch["mask"][0,0].numpy()*255)
# #
    train_set = CSVBasicDataset(csv_dir='data/train_data.csv')
    loader_args = dict(batch_size=2, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    n_train = len(train_set)
    with tqdm(total=n_train, desc=f'test', unit='img') as pbar:
        for batch in train_loader:
            images = batch['image']  # [batch,channel,height,wideth]
            true_masks = batch['mask']  # [batch,height,wideth]
            true_masks = true_masks.to(device='cpu', dtype=torch.long)
            tmp = F.one_hot(true_masks, 2)
            print('nihao')