import random

import cv2
import os
import numpy as np
from albumentations import (
    ElasticTransform,
)
import glob
import csv
import albumentations as A

import pandas as pd
import common.utils as utils
from task_aug.synthetic.simulate_lesion import SimulateLesions 
from skimage.metrics import structural_similarity as ssim
import json

from model.gen_net.mygen.mygenerator import lesion_G_256
import torch
import torchvision.transforms as transform
from model.unet import UNet
from albumentations.pytorch.transforms import ToTensorV2

class SyntheticCNV():
    def __init__(self, args):
        self.args = args
        
        self.sim = SimulateLesions(args)
        # self.init_mask_net()
        self.init_imp_net()

        # real data with focus, extractor focus features from these images.
        csv_dir = os.path.join(self.args.project_path, self.args.real_data_csv)
        self.img1, self.mask1, self.index_list = self.read_csv(csv_dir)  # all images info(path, label etc)

        # for test
        # self.img1 = self.img1[:3]
        # self.mask1 = self.mask1[:3]
        # self.index_list = self.index_list[:3]
        # self.img1 = ['/data1/wangjingtao/workplace/python/data/meta-oct/seg/CNV/x/72_2.jpg']
        # self.mask1 = ['/data1/wangjingtao/workplace/python/data/meta-oct/seg/CNV/y/75_2.png']
        # self.index_list = [145]

        # real data without foucus (normal images)
        # normal_csv_dir = 'D:/workplace/python/metaLearning/meta-learning-seg/dataPro/data/ML_csv/train_data.csv'
        normal_data_csv = os.path.join(self.args.project_path, self.args.normal_data_csv)
        # self.gen_mask()# 分层mask以生成并保存至本地，无需再次生成Layer masks to generate and save locally without having to generate them again
        self.img2, self.mask2,self.nor_idx = self.read_csv(normal_data_csv)  # simulate focus on these images, about 7500 images,在这些OCT图像上进行病灶伪造,应该有7500多张, 分层mask已生成并保存至本地，无需再次生成

    def init_mask_net(self):
        self.gen_mask_net =  UNet(n_channels=3, n_classes=5, bilinear=True).to('cuda')

        # 加载预训练权重
        ckpt = torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/result/result_20240408/bv1000/dl/unet/0_fold/2024-03-25-21-42-51/meta_epoch/taskid_0/best_epoch_for_val_meta_epoch_0.pth')

        self.gen_mask_net.load_state_dict(ckpt)

        self.gen_mask_net.eval()

    def gen_mask(self):
        transform_val = A.Compose([
                A.Resize(width=256, height=256, p=1.0),
                ToTensorV2(),
            ])
        
        self.mask2={}
        for i in self.img2:
            store_dir = i.replace('x/', 'y').replace('.jpg', '.png')
            
            augmented = transform_val(image=img)

            img = augmented["image"] /255.0
            img = img[None].to('cuda') # [1,3,128,128]


            out = self.gen_mask_net(img)
            n_classes = out.shape[1]
            
            output = torch.sigmoid(out.data.cpu()).numpy()
            output[output>=0.5]=1
            output[output<0.5]=0
            output = output[0]

            # _mask = np.argmax(output, axis=0).astype(np.uint8)
            store_out = np.zeros((img.shape[2], img.shape[3]))

            for c in range(n_classes):
                tmp = (output[c] * 255).astype('uint8')
                store_out[output[c]==1] = c+1

            self.mask2[store_dir] = (store_out.astype(np.uint8))



    def init_imp_net(self):
        # 实例化网络
        self.G = lesion_G_256().to('cuda')
        # 加载预训练权重
        ckpt = torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-segmentation/model/gen_net/results/result_20240416/bv1000_cnv/my_gen/2024-04-18-09-45-31/pix2pix_256.pth')
        self.G.load_state_dict(ckpt['G_model'], strict=False)

        self.G.eval()

    # extract features such contour, and augmentation
    def extract_features(self, image_path):
        count = 0
        mask = cv2.imread(image_path, 0)
        index = np.where(mask != 255)
        mask[index] = 0
        CNV_imageErZhi = np.array(255 * (mask == 255), dtype=np.uint8)
        if np.sum(CNV_imageErZhi) != 0:
            cnts, hierarchy = cv2.findContours(CNV_imageErZhi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                count += 1
                # Find boundary coordinates. 找到边界坐标
                x, y, w, h = cv2.boundingRect(c)  # Calculate the outermost rectangular boundary of the point set, 计算点集最外面的矩形边界
                if w < 10 or h < 10:
                    continue

                # store_path = 'data/tmp/' + 'real_cnv_' + str(count) + '.png'
                # cv2.imwrite(store_path, mask[y:(y + h), x:(x + w)])

                new_mask = mask[y:(y + h), x:(x + w)]
                new_mask_pad = cv2.copyMakeBorder(new_mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0))
                ela_Count = 0
                mask_bingzao_list = []
                while ela_Count < self.args.expand_n:
                    ela_Count += 1
                    a = 100
                    aug = ElasticTransform(p=1,
                                        alpha=a,
                                        sigma=a * 0.096,
                                        alpha_affine=a * 0.000000000001,
                                        interpolation=cv2.INTER_NEAREST,
                                        )

                    augmented = aug(image=new_mask_pad)
                    image_elastic = augmented['image']

                    cnv_index = np.where(image_elastic == 255)
                    min_cnv_index_h = min(cnv_index[0])
                    max_cnv_index_h = max(cnv_index[0])
                    min_cnv_index_w = min(cnv_index[1])
                    max_cnv_index_w = max(cnv_index[1])

                    # store_elastic_path = str(count) + '_' + str(ela_Count) + '.png'
                    # cv2.imwrite(store_elastic_path,
                    #             image_elastic[min_cnv_index_h:max_cnv_index_h, min_cnv_index_w:max_cnv_index_w] * 4)
                    
                    mask_bingzao_list.append(
                        image_elastic[min_cnv_index_h:max_cnv_index_h, min_cnv_index_w:max_cnv_index_w])
                
                # make sure juet extracte one focus from per real images with focus, 保证一张图只取一个病灶
                return mask_bingzao_list
            
    def read_csv(self, csv_dir):
        # csv_dir = 'D:/workplace/python/metaLearning/meta-learning-seg/MLCNV/data/train_data.csv'
        dataframe = pd.read_csv(csv_dir)

        img_file = dataframe["Image_path"]
        mask_file = dataframe["Label_path"]
        idx = dataframe['ID']
        return list(img_file), list(mask_file), list(idx)
    

    def improve_lesion(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        transforms = transform.Compose([
            transform.ToTensor(),
            # transform.Resize((256, 256)),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = transforms(img.copy())
        img = img[None].to('cuda')  # [1,3,128,128]
        out = self.G(img)[0]
        out = out.permute(1,2,0)
        out = (0.5 * (out + 1)).cpu().detach().numpy() * 255 #RGB
        out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)

        return out

    
    # 合成数据信息写成csv文件
    def gen_sys_csv(self):
        # real_data_csvfile = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/data/bv1000/real_data/cnv/all_data.csv'
        real_data_csvfile = os.path.join(self.args.project_path, self.args.real_data_csv)
        df_realdata = pd.read_csv(real_data_csvfile)

        img_files = df_realdata["Image_path"].tolist()
        mask_files = df_realdata["Label_path"].tolist()
        ids = df_realdata['ID'].tolist()
        eyes = df_realdata['Eye'].tolist()

        # relative_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/task_aug/result_sys/20240419/bv1000-oct-cnv/2024-04-26-10-56-48/imp_net/'
        relative_path = os.path.join(self.args.store_dir, 'imp_net/')
        
        store_csvfile = os.path.join(relative_path.replace('imp_net', ''), 'sys.csv')

        with open(store_csvfile, 'w', newline='') as csvfile:
            fields = ['ID', 'Eye', 'Image_path', 'Label_path']
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(fields)

            for i in range(len(img_files)):
                img_name = mask_files[i].split('/')[-1].replace('.png', '')
                fold_name = str(ids[i]) + '_' + img_name

                sys_img_path = os.path.join(relative_path,fold_name)

                sys_imgs = glob.glob(os.path.join(sys_img_path, 'x', '*'))

                sys_imgs.sort(key=lambda element: int(element.split('_')[-1].replace('.jpg', '')))

                for image_path in sys_imgs:
                    mask_path = image_path.replace('x', 'y').replace('.jpg', '.png')

                    csvwriter.writerow([ids[i], eyes[i], image_path.replace(relative_path, ''), mask_path.replace(relative_path, '')])

    def denoise(self, tempImg, brightness=25):  # brightness取值 0-50
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

    def forward(self):
        count_step = random.randint(0, len(self.img2))
        for i, koutu_mask_path in enumerate(self.mask1, 1):
            # index += 86
            name = koutu_mask_path.split('/')[-1].replace('.png', '')
            index = self.index_list[i-1]

            mask_bingzao_list = self.extract_features(koutu_mask_path)
            store_path_x = os.path.join(self.args.store_dir, 'imp_net',str(index) + '_' + name, 'x')
            store_path_y = os.path.join(self.args.store_dir, 'imp_net',str(index) + '_' + name, 'y')
            utils.mkdir(store_path_y)
            utils.mkdir(store_path_x)

            store_path_x_noimp = os.path.join(self.args.store_dir, 'no_imp',str(index) + '_' + name, 'x')
            store_path_y_no_imp = os.path.join(self.args.store_dir, 'no_imp',str(index) + '_' + name, 'y')
            utils.mkdir(store_path_y_no_imp) # imp 
            utils.mkdir(store_path_x_noimp)

            for idx, bingzao in enumerate(mask_bingzao_list, 1): # 10个
                bingzao = cv2.resize(bingzao,(int(bingzao.shape[1] / 2), bingzao.shape[0]), interpolation=cv2.INTER_NEAREST)

                flag = True
                while flag:

                    # if not os.path.exists(self.img2[count_step]) or not os.path.exists(self.mask2[count_step]):
                    #     print(f'waring+++{self.img2[count_step]} or {self.mask2[count_step]} is not exist ')
                    #     continue

                    out_list = self.sim.sim_use_real_focus(self.img2[count_step], self.mask2[count_step],
                                                                bingzao_data=bingzao/255 * 40)
                    count_step += 1
                    if count_step > len(self.img2) - 1:
                        count_step = 0
                    if len(out_list) == 0:
                        continue

                    x_name = 'faker' + "_" + str(count_step) + '_' + name + '_' + str(idx) + '.jpg'
                    y_name = 'faker' + "_" + str(count_step) + '_' + name + '_' + str(idx) + '.png'

                    out_list[0] = self.denoise(out_list[0])
                    cv2.imwrite(store_path_x_noimp +  '/' + x_name, out_list[0])
                    cv2.imwrite(store_path_y_no_imp + '/' + y_name, out_list[1])

                    x_imp = cv2.resize(out_list[0], (256,256), cv2.INTER_NEAREST)
                    y_imp = cv2.resize(out_list[1], (256,256), cv2.INTER_NEAREST)

                    x_imp = self.improve_lesion(x_imp)
                    cv2.imwrite(store_path_x +  '/' + x_name, x_imp)
                    cv2.imwrite(store_path_y + '/' + y_name, y_imp)


                    flag = False

        self.gen_sys_csv()