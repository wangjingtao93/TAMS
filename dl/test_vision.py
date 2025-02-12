import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')

import torch
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
import os
import numpy as np
import torchvision
from model.unet import UNet
import albumentations as A
import pandas as pd
import sys
import os
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn.functional as F
from torchvision.utils import save_image
import glob


def test_drive_eye():
    palette = {0:0, 1:128, 2:255, 3:15}

    # img_ls = glob.glob('/data1/wangjingtao/workplace/python/data/rp_data/drive_eye/zhuanhuan/x/*')
    img_ls = glob.glob('/data1/wangjingtao/workplace/python/data/rp_data/heshi/normal/normal/*')
    for img_path in img_ls[:50]:
        store_path = os.path.join(project_path,'dl/analysis',img_path.split('/')[-1])
        mask_path = img_path.replace('x', 'y_2').replace('.jpg', '.png')
        img = cv2.imread(img_path)
        img = cv2.resize(np.array(img), (256,256), interpolation=cv2.INTER_NEAREST)
        store_img = img

        # mask = cv2.imread(mask_path)
        # mask = cv2.resize(np.array(mask), (256,256), interpolation=cv2.INTER_NEAREST)

        transform_val = A.Compose([
            A.Resize(width=256, height=256, p=1.0),
            ToTensorV2(),
        ])
        augmented = transform_val(image=img)

        img = augmented["image"] /255.0
        img = img[None].to('cuda') # [1,3,128,128]


        out = net(img)
        n_classes = out.shape[1]
        
        output = torch.sigmoid(out.data.cpu()).numpy()
        output[output>=0.5]=1
        output[output<0.5]=0
        output = output[0]

        # _mask = np.argmax(output, axis=0).astype(np.uint8)
        store_out = np.zeros((img.shape[2], img.shape[3]))

        for c in range(n_classes):
            tmp = (output[c] * 255).astype('uint8')
            store_img = np.concatenate((store_img, cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)), axis=1)

            store_out[output[c]==1] = np.array(palette[c+1])
        
        store_img = np.concatenate((store_img, cv2.cvtColor(store_out.astype(np.uint8), cv2.COLOR_GRAY2BGR)), axis=1)

        if store_path == None:
            cv2.imwrite('out.jpg', store_img)
        else:
            cv2.imwrite(store_path, store_img)

        # plt.figure()
        # plt.imshow(out)
        # plt.show()

if __name__ == '__main__':
    project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/'

    # 实例化网络
    net = UNet(n_channels=3, n_classes=2, bilinear=True).to('cuda')

    # 加载预训练权重
    ckpt = torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/result/result_20240408/drive-eye/dl/unet/0_fold/2024-04-25-21-42-51/meta_epoch/taskid_0/best_epoch_for_val_meta_epoch_0.pth')

    net.load_state_dict(ckpt)

    net.eval()

    test_drive_eye()

