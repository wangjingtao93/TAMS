from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os

class BV_CNV_MetaDataset(Dataset):
    def __init__(self, args, fileroots, mode='train'):
        self.args = args
        self.mode = mode
        self.tasks = fileroots
        if args.is_gsnet:
            self.relative_path = self.args.synthetic_data_csv.replace('sys.csv', 'imp_net')
        else:
            self.relative_path = self.args.synthetic_data_csv.replace('sys.csv', 'no_imp')
        
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
            A.Resize(width=self.args.dl_resize, height=self.args.dl_resize, p=1.0),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.tasks)

    def toTwoClass(self, mask):
        pixel_value = 5 * 8
        index = np.where(mask != pixel_value)
        mask[index] = 0
        mask = mask / pixel_value * 255
        return mask

    def __getitem__(self, idx):
        task = self.tasks[idx]

        img_list, msk_list = [], []
        for each_path in task:
            
            img_path = os.path.join(self.relative_path, each_path[0])
            mask_path = os.path.join(self.relative_path, each_path[1])

            image = cv2.imread(img_path, 0)  # [h,w]
            mask = cv2.imread(mask_path, 0)  # [h, w]
            mask = self.toTwoClass(mask)
            if self.args.is_base_agu:
                augmented = self.transform_train(image=image, mask=mask)
            else:
                augmented = self.transform_val(image=image, mask=mask)
            image = augmented["image"] /255.0 # [c, h, w]
            mask = augmented["mask"] /255 # [h,w]

            mask = mask[None] # [c,h,w]
            
            img_list.append(image)
            msk_list.append(mask)

        img_tensor = torch.stack(img_list)  # 沿指定维度拼接,会多一维 [shot, channel, height, width]
        msk_tensor = torch.stack(msk_list)  # [shot, height, width]

        return [img_tensor, msk_tensor]
    
class BV1000_OCT_CNV(Dataset):
    def __init__(self, args, csv_dir='',fileroots=[], dataframe=None, mode='train'):
        self.args = args
        self.csv_dir = csv_dir
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
            A.Resize(width=self.args.dl_resize, height=self.args.dl_resize, p=1.0),
            ToTensorV2(),
        ])


    def __len__(self):
        return len(self.mask_file)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_file[idx].replace('/x','/x_d'), 0)
        mask = cv2.imread(self.mask_file[idx], 0)
    
        if  self.mode=='train' and self.args.is_base_agu:
            augmented = self.transform_train(image=image, mask=mask)
        else:
            augmented = self.transform_val(image=image, mask=mask)
        image = augmented["image"] /255.0
        mask = augmented["mask"] /255
        mask = mask[None]

        return {'image': image, 'mask': mask}
    
class BV1000_OCT_CNV_Sys(Dataset):
    def __init__(self, args, csv_dir='',fileroots=[], dataframe=None, mode='train'):
        self.args = args
        self.csv_dir = csv_dir
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
            A.Resize(width=self.args.dl_resize, height=self.args.dl_resize, p=1.0),
            ToTensorV2(),
        ])


    def __len__(self):
        return len(self.mask_file)
    
    def toTwoClass(self, mask):
        pixel_value = 5 * 8
        index = np.where(mask != pixel_value)
        mask[index] = 0
        mask = mask / pixel_value * 255
        return mask

    def __getitem__(self, idx):
        relative_path = self.args.synthetic_data_csv.replace('sys.csv', 'imp_net')
        img_path = os.path.join(relative_path, self.img_file[idx])
        mask_path = os.path.join(relative_path, self.mask_file[idx])
        image = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)
        mask = self.toTwoClass(mask)

        
        if  self.mode=='train' and self.args.is_base_agu:
            augmented = self.transform_train(image=image, mask=mask)
        else:
            augmented = self.transform_val(image=image, mask=mask)
        image = augmented["image"] /255.0
        mask = augmented["mask"] /255
        mask = mask[None]

        return {'image': image, 'mask': mask}

def denoise(tempImg, brightness=25):  # brightness取值 0-50
    '''
    from oysk，oct去噪,传入单通道图
    '''
    (Mean, std) = cv2.meanStdDev(tempImg)
    tempImg = tempImg.astype('float')
    tempImg = tempImg * 1.1 - (np.around(Mean[0][0]) + brightness)  # 默认条件是contrastRatio：1.1，brightness：25
    tempImg[tempImg < 0] = 0
    tempImg = tempImg.astype(np.uint8)
    out = np.zeros(tempImg.shape, np.uint8)
    cv2.normalize(tempImg, out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    out = clahe.apply(out)
    return out

import argparse
def parse_args():
    parser = argparse.ArgumentParser('Gradient-Based Meta-Learning Algorithms')
    parser.add_argument('--dl_resize', type=int, default=256)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    dframe= pd.read_csv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/data/bv1000_ocv_cnv/real_data/cnv/all_data.csv')

    args = parse_args()
    datasets = BV1000_OCT_CNV(args, '', [], dframe)

    train_loader = DataLoader(dataset=datasets, batch_size=16, shuffle=True, num_workers=4,
                              drop_last=True)

    pd = tqdm(train_loader)

    for idx, data in enumerate(pd):
        img = data['image'].to('cpu')
        mask = data['mask'].to('cpu')


        pd.desc = 'train_{} G_loss: {} D_loss: {}'.format(1, 2, 3)



