
from os.path import splitext
from os import listdir
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader


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



class DriveDataset(Dataset):

    def __init__(self, args, csv_dir = '', fileroots=[],dataframe=None, mode='train', ):
        self.csv_dir = csv_dir
        self.args = args
        self.mode = mode
        self.img_file =[]
        self.mask_file=[]
        if len(fileroots) !=0:
             for i in range(len(fileroots)):
                 self.img_file.append(fileroots[i][0])
                 self.mask_file.append(fileroots[i][1])
        else:
            if self.csv_dir == '' and dataframe is not None:
                self.dataframe = dataframe
            else:
                self.dataframe = pd.read_csv(self.csv_dir)

            self.img_file = list(self.dataframe["Image_path"])
            self.mask_file = list(self.dataframe["Label_path"])

        print(f'Creating dataset with {len(self.mask_file)} examples')

        self.transform_train = A.Compose([
            A.Resize(width=self.args.dl_resize, height=self.args.dl_resize, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate((-25, 25), p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.5),
            ToTensorV2(),
        ])
        self.transform_val = A.Compose([
            A.Resize(width=self.args.dl_resize, height=self.args.dl_resize, p=1.0),
            ToTensorV2(),
        ])


    def __len__(self):
        return len(self.mask_file)
    
    def to_one_hot(self, mask):
        """
        Transform a mask to one hot
        change a mask to n * h* w   n is the class
        Args:
            mask:
            n_class: number of class for segmentation
        Returns:
            y_one_hot: one hot mask
        """
        y_one_hot = torch.zeros((self.args.n_classes + 1, mask.shape[1], mask.shape[2]))
        y_one_hot = y_one_hot.scatter(0, torch.LongTensor(mask.numpy()) , 1).long()

        return y_one_hot[1:,...]



    def __getitem__(self, idx):
        relative_path = '/data1/wangjingtao/workplace/python/data/rp_data/drive_eye/zhuanhuan'
        image = cv2.imread(os.path.join(relative_path, 'x',self.img_file[idx]))  # [height,width]
        mask = cv2.imread(os.path.join(relative_path, 'y_2',self.mask_file[idx]), 0)  # [height, width]

        mask[mask == 255] = 1
        mask[mask == 128] = 2
        mask[mask == 64] = 1
        

        if  self.mode=='train':
            augmented = self.transform_train(image=image, mask=mask)
        else:
            augmented = self.transform_val(image=image, mask=mask)

        image = augmented["image"] /255.0

        mask = augmented["mask"]

        
        mask = mask[None]
        mask = self.to_one_hot(mask)

        return {'image': image, 'mask': mask}
    
def parse_args():
    import argparse

    parser = argparse.ArgumentParser('Gradient-Based Meta-Learning Algorithms')
    parser.add_argument('--dl_resize', type=int, default=256)
    
    # base settings
    parser.add_argument('--description_name', type=str, default='description')    
    args = parser.parse_args()

    return args
    
if __name__ == '__main__':
    csv_dir = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/data/drive_eye/data.csv'

    args = parse_args()

    dataset = DriveDataset(args, csv_dir=csv_dir)


    train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=4,
                              drop_last=True)

    pd = tqdm(train_loader)

    for idx, data in enumerate(pd):
        in_img = data['image'].to('cpu')
        real_img = data['mask'].to('cpu')