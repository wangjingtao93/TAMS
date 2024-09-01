from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import random

import cv2
import albumentations as A


class MetaDataset(Dataset):
    def __init__(self, fileroots):
        self.tasks = fileroots

        self.transform = A.Compose([
            # A.Resize(width=128, height=128, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate((-5, 5), p=0.5),
            # A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1,
            #                  num_flare_circles_lower=1, num_flare_circles_upper=2,
            #                  src_radius=160, src_color=(255, 255, 255), always_apply=False, p=0.3),
            # A.RGBShift(r_shift_limit=10, g_shift_limit=10,
            #            b_shift_limit=10, always_apply=False, p=0.2),
            # A.ElasticTransform(alpha=2, sigma=15, alpha_affine=25, interpolation=1,
                            #    border_mode=4, value=None, mask_value=None,
                            #    always_apply=False, approximate=False, p=0.3),
            # A.Normalize(p=1.0),
            # ToTensor(),
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
            # image = np.array(Image.open(each_path[0]))
            # mask = np.array(Image.open(each_path[1]).convert('L'))

            # image = cv2.imread(each_path[0], 1)  # [height,width,channel]
            image = cv2.imread(each_path[0], 0)  # [height,width]
            mask = cv2.imread(each_path[1], 0)  # [height, width]
            mask = self.toTwoClass(mask)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
            # augmented = self.transform(image=image, mask=mask)
            # image = augmented["image"]
            # mask = augmented["mask"]

            image = image[None]  # [channel, height,width]
            # mask = mask[None] # [channel, height,width]
            mask = torch.from_numpy(mask) / 255
            image = torch.from_numpy(image) / 255.0
            # image = image.permute(2, 0, 1) #从[height,width,channel]变成[channel, height,width]


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
        
        self.transform = A.Compose([
            # A.Resize(width=128, height=128, p=1.0),
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
            # A.Normalize(p=1.0),
            # ToTensor(),
        ])

    def __len__(self):
        return len(self.mask_file)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_file[idx], 0)  # [height,width,channel]
        mask = cv2.imread(self.mask_file[idx], 0)  # [height, width]
        mask = cv2.resize(mask, (self.args.dl_resize, self.args.dl_resize), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (self.args.dl_resize, self.args.dl_resize), interpolation=cv2.INTER_NEAREST)

        if self.args.if_agu and self.mode=='train':
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image[None]
        # mask = mask[None] # [channel, height,width] # 输出是one_channel的时候才会用到
        mask = torch.from_numpy(mask) / 255
        image = torch.from_numpy(image) / 255.0
        # image = image.permute(2, 0, 1)  # 从[height,width,channel]变成[channel, height,width]
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

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import cv2
    from tqdm import tqdm
    import torch.nn.functional as F
#     dir_img = r"../../data/BV1000_Segmentation/merge_onhlayer/test/image"
#     dir_mask = r"../../data/BV1000_Segmentation/merge_onhlayer/test/label"
#     img_scale = 1
#     dataset = MetaDataset(dir_img, dir_mask, img_scale)
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
    train_set = CSVMetaDataset(csv_dir='data/train_data.csv')
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


