from torch.utils.data.dataset import Dataset
import torchvision.transforms as transform
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import os
import torchvision
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import torch
import albumentations as A

class CreateDatasets(Dataset):
    def __init__(self, ori_imglist,img_size):
        self.ori_imglist = ori_imglist
        self.transform = transform.Compose([
            transform.ToTensor(),
            transform.Resize((img_size, img_size)),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.ori_imglist)

    def __getitem__(self, item):
        ori_img = cv2.imread(self.ori_imglist[item])
        ori_img = ori_img[:, :, ::-1]
        real_img = Image.open(self.ori_imglist[item].replace('.png', '.jpg'))
        ori_img = self.transform(ori_img.copy())
        real_img = self.transform(real_img)
        return ori_img, real_img

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


class CreateDatasets_CNV(Dataset):
    def __init__(self, csv_root,img_size):
        self.dataframe = pd.read_csv(csv_root)

        self.real_imglist = self.dataframe['Image_path']
        self.ori_imglist = self.dataframe['Label_path']
        self.transform = transform.Compose([
            transform.ToTensor(),
            transform.Resize((img_size, img_size)),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.ori_imglist)

    def __getitem__(self, item):
        ori_img = (cv2.imread(self.ori_imglist[item], 0) / 255.0).astype(np.float32)
        real_img = Image.open(self.real_imglist[item].replace('/x/', '/x_d/'))

        lesion = np.where(ori_img, real_img, 0)
        random_matrix = np.random.randint(-20, 21, ori_img.shape)
        lesion = lesion + random_matrix
        lesion = np.clip(lesion, 0, 255).astype(np.float32)  # note: 不能缺少，否则图像会糊

        ori_img = 1 - ori_img
        koutu = np.where(ori_img, real_img, 0)
        ori_img = lesion + koutu
        ori_img = np.clip(ori_img, 0, 255) # note: 不能缺少，否则图像会糊

        ori_img = Image.fromarray(ori_img.astype(np.uint8)).convert('RGB')
        real_img = real_img.convert('RGB')

        ori_img = self.transform(ori_img.copy()) # hw-->chw
        real_img = self.transform(real_img) # hw-->chw
        return ori_img, real_img


class CreateDatasets_RIPS(Dataset):
    def __init__(self, csv_root,img_size):
        self.dataframe = pd.read_csv(csv_root)
        self.img_size = img_size

        self.real_imglist = self.dataframe['Image_path']
        self.ori_imglist = self.dataframe['Label_path']

        self.transform = transform.Compose([
            transform.ToTensor(),
            # transform.Resize((img_size, img_size)),
            # transform.RandomHorizontalFlip(p=1),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.lesion_trans =  A.Compose([
            # A.ColorJitter(always_apply=True),
            A.HueSaturationValue(always_apply=True)
                            ])

    def __len__(self):
        return len(self.ori_imglist)

    def __getitem__(self, item):
        # ori_img = cv2.imread(self.ori_imglist[item]).astype(np.float32)
        # ori_img = ori_img[:, :, ::-1]
    
        mask = cv2.imread(self.ori_imglist[item])
        # ori_img = ori_img[:, :, ::-1]# BGR-->RGB

        # real_img = Image.open(self.real_imglist[item])
        real_img = cv2.imread(self.real_imglist[item])
        # real_img = ori_img[:, :, ::-1]# BGR-->RGB

        # if real_img.shape != mask.shape:
        real_img = cv2.resize(real_img,(self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST) # interpolation不能少
        mask = cv2.resize(mask,(self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        mask_fan = 255 - mask
        lesion = np.where(mask, real_img, 0)

        # lesion tranform
        lesion =  self.lesion_trans(image=lesion)['image']

        ground = np.where(mask_fan, real_img, 0)
        ori_img = lesion + ground
        # ori_img = cv2.add(lesion, ground) # 似乎这个要比直接相加好，直接相加会出现异常像素

        ori_img = ori_img[:, :, ::-1]
        real_img = real_img[:, :, ::-1]


        ori_img = self.transform(ori_img.copy()) # 因为使用了付索引[:, :, ::-1]，所以必须用copy
        real_img = self.transform(real_img.copy())
        return ori_img, real_img


class CreateDatasets_CNV_Trans(Dataset):
    def __init__(self, csv_root,img_size):
        self.dataframe = pd.read_csv(csv_root)

        self.real_imglist = self.dataframe['Image_path']
        self.ori_imglist = self.dataframe['Label_path']
        self.transform = transform.Compose([
            transform.ToTensor(),
            transform.Resize((img_size, img_size)),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.ori_imglist)

    def __getitem__(self, item):
        ori_img = (cv2.imread(self.ori_imglist[item], 0) / 255.0).astype(np.float32)
        real_img = Image.open(self.real_imglist[item].replace('/x/', '/x_d/'))

        lesion = np.where(ori_img, real_img, 0)
        random_matrix = np.random.randint(-20, 21, ori_img.shape)
        lesion = lesion + random_matrix
        # lesion = np.clip(lesion, 0, 255).astype(np.float32)

        ori_img = 1 - ori_img
        koutu = np.where(ori_img, real_img, 0)
        ori_img = lesion + koutu
        # ori_img = np.clip(ori_img, 0, 255)

        ori_img = Image.fromarray(ori_img.astype(np.uint8)).convert('RGB')
        real_img = real_img.convert('RGB')

        ori_img = self.transform(ori_img.copy()) # hw-->chw
        real_img = self.transform(real_img) # hw-->chw

        latent_feature = torch.tensor(np.load(self.real_imglist[item].replace('/x/', '/x_retf/').replace('.jpg', '.npy')))

        return ori_img, real_img, latent_feature

if __name__ == '__main__':
    object_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation'

    # CNV+++++++++++++ 
    # train_path = os.path.join(object_path, 'data/bv1000_ocv_cnv/real_data/cnv/all_data.csv')
    # # train_datasets = CreateDatasets_CNV_Trans(train_path,224)
    # train_datasets = CreateDatasets_CNV(train_path, 256)


    
    # train_loader = DataLoader(dataset=train_datasets, batch_size=16, shuffle=True, num_workers=4,
    #                           drop_last=True)

    # pd = tqdm(train_loader)

    # for idx, data in enumerate(pd):
    #     in_img = data[0].to('cpu')
    #     real_img = data[1].to('cpu')
    #     pd.desc = 'train_{} G_loss: {} D_loss: {}'.format(1, 2, 3)

    # CNV----------------

    # RIPS++++++++++++++++++++++++++
    train_path = os.path.join(object_path, 'data/rips_fundus_rp/real_data/all_data.csv')
    train_datasets = CreateDatasets_RIPS(train_path,224)

    train_loader = DataLoader(dataset=train_datasets, batch_size=16, shuffle=True, num_workers=4,
                              drop_last=True)

    pd = tqdm(train_loader)

    for idx, data in enumerate(pd):
        in_img = data[0].to('cpu')
        real_img = data[1].to('cpu')



        if idx == 10:
            break


        pd.desc = 'train_{} G_loss: {} D_loss: {}'.format(1, 2, 3)
    
    # RPIS-----------------------------


