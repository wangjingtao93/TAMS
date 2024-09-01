
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

class RIPSMetaDataset(Dataset):
    # def __init__(self, imgs_dir, masks_dir, scale=1, data_type='train'):
    def __init__(self, args, fileroots, mode='train'):
        self.args = args
        self.tasks = fileroots
        self.mode = mode

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

           
            image = cv2.imread(img_path)  # [height,width,channel]
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

class RIPSDataset(Dataset):

    def __init__(self, args, csv_dir = '', fileroots=[],dataframe=None, mode='train', is_sys=False):
        self.csv_dir = csv_dir
        self.args = args
        self.mode = mode
        self.is_sys = is_sys# 是否是合成数据

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

        logging.info(f'Creating dataset with {len(self.mask_file)} examples')

        self.transform_train = A.Compose([
            A.Resize(width=self.args.dl_resize, height=self.args.dl_resize, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate((-25, 25), p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2(),
        ])
        self.transform_val = A.Compose([
            A.Resize(width=self.args.dl_resize, height=self.args.dl_resize, p=1.0),
            ToTensorV2(),
        ])


    def __len__(self):
        return len(self.mask_file)

    def __getitem__(self, idx):
        if not self.is_sys:
            image = cv2.imread(self.img_file[idx])  # [h,w,c]
            mask = cv2.imread(self.mask_file[idx], 0)  # [h, w]
        else:
            relative_path = self.args.synthetic_data_csv.replace('sys.csv', 'imp_net')
            img_path = os.path.join(relative_path, self.img_file[idx])
            mask_path = os.path.join(relative_path, self.mask_file[idx])
            image = cv2.imread(img_path)  # [c,h,w]
            mask = cv2.imread(mask_path, 0)  # [h, w]

        if  self.mode=='train' and self.args.is_base_agu:
            augmented = self.transform_train(image=image, mask=mask)
        else:
            augmented = self.transform_val(image=image, mask=mask)

        image = augmented["image"] /255.0
        mask = augmented["mask"] /255
        mask = mask[None]

        return {'image': image, 'mask': mask}



def tichu_black(image_l, mask_l):
    new_mask = []
    new_image = []
    len_l = len(image_l)
    for i in range(len_l):
        arr = cv2.imread(mask_l[i])
        if np.sum(arr) != 0:
            new_mask.append(mask_l[i])
            new_image.append(image_l[i])
        
    return new_image, new_mask

def tichu_black_for_patch(dataframe):
    id_arr = np.unique(dataframe['ID'])
    
    dataframe_new = pd.DataFrame()
    for id in id_arr:
        df_one_picture = dataframe[dataframe['ID'] == id]
        mask_l = list(df_one_picture['Label_path'])

        pix_sum = 0
        for mask in mask_l:
            arr = cv2.imread(mask)
            pix_sum += np.sum(arr)
        

        if pix_sum != 0:
            dataframe_new=pd.concat([dataframe_new, df_one_picture], ignore_index=True)

    return dataframe_new

        

class CropDataset(Dataset):

    def __init__(self, csv_dir = '',dataframe=None, mode='train', args=None):
        self.csv_dir = csv_dir
        if self.csv_dir == '' and dataframe is not None:
            if mode != 'train' and args != None and args.if_tichu:
                self.dataframe = tichu_black_for_patch(dataframe)
            else:
                self.dataframe = dataframe
        else:
            self.dataframe = pd.read_csv(self.csv_dir)
        

        self.img_file = list(self.dataframe["Image_path"])
        self.mask_file = list(self.dataframe["Label_path"])

        self.mode = mode
        
        logging.info(f'Creating dataset with {len(self.mask_file)} examples')


    def __len__(self):
        return len(self.mask_file)
 
    def __getitem__(self, idx):
        image = cv2.imread(self.img_file[idx]) 
        mask = cv2.imread(self.mask_file[idx], 0)  # [height, width]

        mask = torch.from_numpy(mask) / 255
        
        image = torch.from_numpy(image) / 255.0
        image = image.permute(2, 0, 1)  

        return {'image': image, 'mask': mask}



def test_CropDataset():
    train_set = CropDataset(csv_dir='../TL/data/crop/train_data.csv')
    loader_args = dict(batch_size=32, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=False, **loader_args)
    n_train = len(train_set)
    num_crops = 4
    step_height = 128
    step_width = 256
    count = 0
    with tqdm(total=n_train, desc=f'test', unit='img') as pbar:
        for batch in train_loader:
            images = batch['image']  # [batch,channel,height,wideth]
            true_masks = batch['mask']  # [batch,height,wideth]
            image_huifu = np.zeros((512, 1024, 3))
            mask_huifu = np.zeros((512, 1024))
            
            for i in range(2):
                # 每个batch里包含的图片
                for k in range(num_crops):
                    # k 即 宽step_index
                    for j in range(num_crops):
                        image_huifu[(j * step_height):((j + 1) * step_height),
                                    (k * step_width):((k + 1) * step_width),
                                    0] = images[i*16 + k * num_crops + j, :, :, 0]
                        image_huifu[(j * step_height):((j + 1) * step_height),
                                    (k * step_width):((k + 1) * step_width),
                                    1] = images[i*16 + k * num_crops + j, :, :, 1]
                        image_huifu[(j * step_height):((j + 1) * step_height),
                                    (k * step_width):((k + 1) * step_width),
                                    2] = images[i*16 + k * num_crops + j, :, :, 2]

                        mask_huifu[(j * step_height):((j + 1) * step_height),
                                (k * step_width):(
                                    (k + 1) *
                                    step_width)] = true_masks[i*16 + k * num_crops + j, :, :]
                        # cv2.imwrite('tmp.png', numpy.array(images[i*16 + k * num_crops + j, ...])) 
                        
                store_path_x = 'tmp/image_huiifu_' + str(count) + '.png'
                store_path_y = 'tmp/mask_huifu' + str(count) + '.png' 
                cv2.imwrite(store_path_x, image_huifu)
                cv2.imwrite(store_path_y, mask_huifu)
                
                count += 1

            # true_masks = true_masks.to(device='cpu', dtype=torch.long)
            # tmp = F.one_hot(true_masks, 2)
            # print('nihao')
import argparse
def parse_args():
    parser = argparse.ArgumentParser('Gradient-Based Meta-Learning Algorithms')
    parser.add_argument('--dl_resize', type=int, default=256)
    args = parser.parse_args()

    return args

def test_RPDataset():
    dframe= pd.read_csv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/data/rips_fundus_rp/real_data/all_data.csv')

    args = parse_args()
    datasets = RIPSDataset(args, '', [], dframe)

    train_loader = DataLoader(dataset=datasets, batch_size=16, shuffle=True, num_workers=4,
                              drop_last=True)

    td = tqdm(train_loader)

    for idx, data in enumerate(td):
        img = data['image'].to('cpu')
        mask = data['mask'].to('cpu')


        td.desc = 'train_{} G_loss: {} D_loss: {}'.format(1, 2, 3)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import cv2
    from tqdm import tqdm
    import torch.nn.functional as F
    from tqdm import tqdm
    
    # test_CropDataset()
    test_RPDataset()


    