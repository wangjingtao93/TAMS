
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

class RPMetaDataset(Dataset):
    # def __init__(self, imgs_dir, masks_dir, scale=1, data_type='train'):
    def __init__(self, fileroots, mode='train'):
        self.tasks = fileroots
        self.mode = mode

        # self.transform = A.Compose([
        #     # A.Resize(width=128, height=128, p=1.0),
        #     A.HorizontalFlip(p=0.5),
        #     # A.VerticalFlip(p=0.5),
        #     A.Rotate((-5, 5), p=0.5),
        #     # A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1,
        #     #                  num_flare_circles_lower=1, num_flare_circles_upper=2,
        #     #                  src_radius=160, src_color=(255, 255, 255), always_apply=False, p=0.3),
        #     # A.RGBShift(r_shift_limit=10, g_shift_limit=10,
        #     #            b_shift_limit=10, always_apply=False, p=0.2),
        #     A.ElasticTransform(alpha=2, sigma=15, alpha_affine=25, interpolation=1,
        #                        border_mode=4, value=None, mask_value=None,
        #                        always_apply=False, approximate=False, p=0.3),
        #     # A.Normalize(p=1.0),
        #     # ToTensor(),
        # ])

    def transform(self, image, mask):
        argu = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # # A.OneOf([
            # #     # A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
            # #     A.GaussNoise(),    # 将高斯噪声应用于输入图像。
            # # ], p=0.7),   # 应用选定变换的概率
            # A.OneOf(
            #     [
            #         A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
            #         A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
            #         A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
            #     ],
            #     p=0.7),
            # # A.ElasticTransform(p=0.5, alpha=120*0.1, sigma=120*0.1 * 0.05, alpha_affine=120*0.1 * 0.03),
            # # A.GridDistortion(p=0.5, num_steps=50),
            A.ShiftScaleRotate(shift_limit=0.0625,
                               scale_limit=0.1,
                               rotate_limit=45,
                               p=0.5),
            # # 随机应用仿射变换：平移，缩放和旋转输入
            # A.CLAHE(p=0.3), # 自适应直方图均衡化
            # A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
        ])(image=image, mask=mask)
        return argu
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
            # image = np.array(Image.open(each_path[0]))
            # mask = np.array(Image.open(each_path[1]).convert('L'))

            # image = cv2.imread(each_path[0], 1)  # [height,width,channel]
            image = cv2.imread(each_path[0])  # [height,width,channel]
            mask = cv2.imread(each_path[1], 0)  # [height, width]
            
            # mask = self.toTwoClass(mask)
            # if mask is None:
            #     print("wjt+++++", each_path[1])
            # mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            # image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
            # augmented = self.transform(image=image, mask=mask)
            # image = augmented["image"]
            # mask = augmented["mask"]
            # _, _, image = cv2.split(image)# 只用红色通道
            
            if self.mode == 'train':
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            
            # image = image[None]  # [channel, height,width] #彩图不需要，灰图需要
            # mask = mask[None] # [channel, height,width]
            mask = torch.from_numpy(mask) / 255
            image = torch.from_numpy(image) / 255.0
            image = image.permute(2, 0, 1)  # 从[height,width,channel]变成[channel, height,width],彩图需要


            img_list.append(image)
            msk_list.append(mask)

        img_tensor = torch.stack(img_list)  # 沿指定维度拼接,会多一维 [shot, channel, height, width]
        msk_tensor = torch.stack(msk_list)  # [shot, height, width]

        return [img_tensor, msk_tensor]

class RPDataset(Dataset):

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

        # better not delete black images
        if mode != 'train' and args != None and args.if_tichu:
                self.img_file, self.mask_file = tichu_black(self.img_file, self.mask_file)

        logging.info(f'Creating dataset with {len(self.mask_file)} examples')

    def __len__(self):
        return len(self.mask_file)
    
    def transform(self, image, mask):
        argu = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # # A.OneOf([
            # #     # A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
            # #     A.GaussNoise(),    # 将高斯噪声应用于输入图像。
            # # ], p=0.7),   # 应用选定变换的概率
            # A.OneOf(
            #     [
            #         A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
            #         A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
            #         A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
            #     ],
            #     p=0.7),
            # # A.ElasticTransform(p=0.5, alpha=120*0.1, sigma=120*0.1 * 0.05, alpha_affine=120*0.1 * 0.03),
            # # A.GridDistortion(p=0.5, num_steps=50),
            A.ShiftScaleRotate(shift_limit=0.0625,
                               scale_limit=0.1,
                               rotate_limit=45,
                               p=0.5),
            # # 随机应用仿射变换：平移，缩放和旋转输入
            # A.CLAHE(p=0.3), # 自适应直方图均衡化
            # A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
        ])(image=image, mask=mask)
        return argu

    def __getitem__(self, idx):
        image = cv2.imread(self.img_file[idx])  # [height,width]
        mask = cv2.imread(self.mask_file[idx], 0)  # [height, width]

        # if image.empty():
        #     print('+++++++++++++++', self.img_file[idx])
            
        # _, _, image = cv2.split(image)

        image = cv2.resize(image, (self.args.dl_resize, self.args.dl_resize), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (self.args.dl_resize, self.args.dl_resize), interpolation=cv2.INTER_NEAREST)

        # 训练数据是否增强
        # if self.mode == 'train':
        #     augmented = self.transform(image=image, mask=mask)
        #     image = augmented["image"]
        #     mask = augmented["mask"]

        # image = image[None]  #彩图不需要，灰图需要
        # mask = mask[None] # [channel, height,width] # 输出是one_channel的时候才会用到
        mask = torch.from_numpy(mask) / 255
        image = torch.from_numpy(image) / 255.0
        image = image.permute(2, 0, 1)  # 从[height,width,channel]变成[channel, height,width],彩图需要
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

    def transform(self, image, mask):
        argu = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # # A.OneOf([
            # #     # A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
            # #     A.GaussNoise(),    # 将高斯噪声应用于输入图像。
            # # ], p=0.7),   # 应用选定变换的概率
            # A.OneOf(
            #     [
            #         A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
            #         A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
            #         A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
            #     ],
            #     p=0.7),
            # # A.ElasticTransform(p=0.5, alpha=120*0.1, sigma=120*0.1 * 0.05, alpha_affine=120*0.1 * 0.03),
            # # A.GridDistortion(p=0.5, num_steps=50),
            A.ShiftScaleRotate(shift_limit=0.0625,
                               scale_limit=0.1,
                               rotate_limit=45,
                               p=0.5),
            # # 随机应用仿射变换：平移，缩放和旋转输入
            # A.CLAHE(p=0.3), # 自适应直方图均衡化
            # A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
        ])(image=image, mask=mask)
        return argu
    
    def __getitem__(self, idx):
        image = cv2.imread(self.img_file[idx]) 
        mask = cv2.imread(self.mask_file[idx], 0)  # [height, width]

        # if image is None:
        #     print('wjt+++++++++++_____________+++++++++++++', self.img_file[idx])
        # _, _, image = cv2.split(image)

        # if self.mode == 'train':
        #     augmented = self.transform(image=image, mask=mask)
        #     image = augmented["image"]
        #     mask = augmented["mask"]

        # image = image[None] #彩图不需要，灰图需要
        # mask = mask[None] # [channel, height,width] # 输出是one_channel的时候才会用到
        mask = torch.from_numpy(mask) / 255
        
        image = torch.from_numpy(image) / 255.0
        image = image.permute(2, 0, 1)  # 从[height,width,channel]变成[channel, height,width], 彩图需要，灰图不需要
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