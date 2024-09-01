import random

import cv2
import os
import numpy as np
from albumentations import (
    ElasticTransform,
)
import glob
import torch
import albumentations as A

import pandas as pd
import common.utils as utils
from skimage.metrics import structural_similarity as ssim
import json
import task_aug.synthetic.utils as task_ut
from model.unet import UNet
from model.gen_net.mygen.mygenerator import lesion_G_256

class SyntheticRIPS():
    def __init__(self, args):
        self.args = args
        dataframe = pd.read_csv(os.path.join(self.args.project_path,self.args.real_data_csv))

        self.img_files = list(dataframe["Image_path"])
        self.mask_files = list(dataframe["Label_path"])

        # self.normal_images_path = os.path.join('/data1/wangjingtao/workplace/python/data/rp_data/heshi/normal/normal')
        self.normal_images_path = glob.glob('/data1/wangjingtao/workplace/python/data/ODIR-5K/train_normal/*')
        
        self.init_mask_net()
        self.init_imporve_lesion_net()

        self._init_acc()
        
    def init_mask_net(self):
        self.gen_mask_net =  UNet(n_channels=3, n_classes=2, bilinear=True).to('cuda')

        # 加载预训练权重
        ckpt = torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-segmentation/result/result_20240408/drive-eye/dl/unet/0_fold/2024-04-25-21-42-51/meta_epoch/taskid_0/best_epoch_for_val_meta_epoch_0.pth')

        self.gen_mask_net.load_state_dict(ckpt)

        self.gen_mask_net.eval()

    def init_imporve_lesion_net(self):
        # 实例化网络
        self.G = lesion_G_256().to('cuda')
        # note 注意
        # G = pix2pixG_256().to('cuda')
        # 加载预训练权重
        ckpt = torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-segmentation/model/gen_net/results/result_20240416/rips/my_gen/2024-04-24-15-53-51/gen_lesion_256.pth')
        self.G.load_state_dict(ckpt['G_model'], strict=False)

        self.G.eval()

    def _init_acc(self):
        self.trans = A.Compose([
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.OneOf([
            #     A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
            #     A.GaussNoise(),    # 将高斯噪声应用于输入图像。
            # ], p=0.2),   # 应用选定变换的概率
            # A.OneOf([
            #     A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
            #     A.MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
            #     A.Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像。
            # ], p=0.2),

            A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, interpolation=cv2.INTER_NEAREST),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            # A.ShiftScaleRotate(shift_limit=0.0625, rotate_limit=45, p=1),
            A.ShiftScaleRotate(scale_limit=0.20, rotate_limit=10, shift_limit=0.1, p=0.8, interpolation=cv2.INTER_NEAREST)
            # 随机应用仿射变换：平移，缩放和旋转输入
            # A.RandomBrightnessContrast(p=1),   # 随机明亮对比度

        ])

        
    def choose_similar_ssim(self, img1_path):

        # 已经计算好，直接从本地加载结果
        with open('/data1/wangjingtao/workplace/python/data/ODIR-5K/ssim/ssim.json', 'r') as f:
            loaded_dict = json.load(f)

        img1_name = img1_path.split('/')[-1].split('.')[0]
        return loaded_dict[img1_name]

        # 速度很
        image_A = cv2.imread(img1_path)
        image_A = cv2.resize(image_A, (565, 450), interpolation=cv2.INTER_NEAREST)
        top, bottom = 67, 67
        image_A = cv2.copyMakeBorder(image_A, top, bottom, 0, 0, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        image_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2GRAY)

        # 初始化存储最相似图片的列表
        most_similar_images = []

        for image_B_path in self.normal_images_path[:10]:
            image_B = cv2.imread(image_B_path)
            image_B = self.remove_the_blackborder(image_B)
            image_B = cv2.resize(image_B, (565, 584), interpolation=cv2.INTER_CUBIC)
            image_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2GRAY)

            # 计算结构相似度
            similarity = ssim(image_A, image_B, multichannel=True)
            # 添加到最相似图片列表
            most_similar_images.append((image_B_path, similarity))

        # 对最相似图片列表按相似度进行排序
        most_similar_images.sort(key=lambda x: x[1], reverse=True)

        # 获取前 20 张最相似的图片
        top_20_similar_images = most_similar_images[:20]
        # 打印或保存这些最相似的图片路径
        # for image_path, similarity in top_20_similar_images:
        #     print(f"Image A: {image_A_path}, Similar Image B: {image_path}, SSIM: {similarity}")

        return top_20_similar_images




    def aug(self, src):
        """图像亮度增强"""
        if self.get_lightness(src) > 130:
            # src = liner_trans(src, 0.9)
            print("图片亮度足够，不做增强")
            return src  
        # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
        # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内

    def get_lightness(self, src):
        # 计算亮度
        hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        lightness = hsv_image[:, :, 2].mean()

        return lightness
    
    # 裁去黑边
    def remove_the_blackborder(self, image):
        # image = cv2.imread(image)  # 读取图片
        img = cv2.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
        b = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
        binary_image = b[1]  # 二值图--具有三通道
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        # print(binary_image.shape)     #改为单通道

        edges_y, edges_x = np.where(binary_image == 255)  ##h, w
        bottom = min(edges_y)
        top = max(edges_y)
        height = top - bottom

        left = min(edges_x)
        right = max(edges_x)
        height = top - bottom
        width = right - left

        res_image = image[bottom:bottom + height, left:left + width]
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(image)
        # plt.subplot(1, 2, 2)
        # plt.imshow(res_image)
        # # plt.savefig(os.path.join("res_combine.jpg"))
        # plt.show()
        return res_image
    
    def liner_trans(self, img, gamma):#gamma大于1时图片变亮，小于1图片变暗
        img=np.float32(img)*gamma//1
        img[img>255]=255
        img=np.uint8(img)
        return img
    
    # 计算仿真病灶像素值
    # 合并lesions， 望在两张图像A和B中，对B中不为0的像素进行处理，合并到A上。当B的像素不为0时，如果A的对应像素也不为0，则取两者的平均值；如果A的对应像素为0，则直接用B的像素替换。
    def calculate_20240417(self, img1, mask,img2):
        # img1 = self.balance_lesion_pix(img1, img2)

        # 初始化仿真病灶
        sys_lesion_all = np.zeros_like(img1)
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for k, contour in enumerate(contours):
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)

            w  += 5
            h += 5
            # 矩形框像素个数
            pix_num_rec = h * w
            # 提取矩形框内的像素
            rect_img1 = img1[y:y+h, x:x+w]
            # 计算矩形框内像素的和
            rec_sum_pixels_values = np.sum(rect_img1)
            
            # 病灶轮廓内像素个数
            pix_num_lesion= np.count_nonzero(mask)
            # 计算病灶坐标和值
            tmp1 = np.zeros_like(img1)
            tmp1[y:y+h,x:x+w] = img1[y:y+h, x:x+w] 
            real_lesion = np.where(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), tmp1, 0)

            # 计算背景图像像素
            back_rect = img2[y:y+h, x:x+w]
            back_rec_sum_pix_values = np.sum(back_rect)
            
            # 等比例放缩
            sys_lesion = (real_lesion * (back_rec_sum_pix_values / rec_sum_pixels_values)).astype(np.uint8)
            cv2.imwrite('tmp' + str(k) + '.jpg', sys_lesion)

            # 合并lesions， 望在两张图像A和B中，对B中不为0的像素进行处理，合并到A上。当B的像素不为0时，如果A的对应像素也不为0，则取两者的平均值；如果A的对应像素为0，则直接用B的像素替换。
            # sys_lesion_all = cv2.add(sys_lesion_all, sys_lesion) # 不管不顾，直接取值
            mask_b_not_zero = np.any(sys_lesion > 0, axis=2)  # 找到B图像中非0的像素位置

            # 创建一个mask来检测A中对应位置也不为0的情况
            mask_a_not_zero = np.any(sys_lesion_all > 0, axis=2)  # 找到A图像中非0的像素位置

            # 创建一个合并mask，两个条件都满足（B不为0且A也不为0的位置）
            mask_both_not_zero = mask_b_not_zero & mask_a_not_zero

            # 对于同时不为0的位置，计算平均值
            for c in range(3):  # 遍历每一个颜色通道
                sys_lesion_all[:, :, c][mask_both_not_zero] = (
                    sys_lesion_all[:, :, c][mask_both_not_zero].astype(np.uint16) +
                    sys_lesion[:, :, c][mask_both_not_zero].astype(np.uint16)
                ) // 2

            # 对于B不为0但A为0的位置，直接使用B的像素值
            mask_a_zero = ~mask_a_not_zero  # A中为0的位置
            mask_b_to_a = mask_b_not_zero & mask_a_zero  # B不为0且A为0的位置

            sys_lesion_all[mask_b_to_a] = sys_lesion[mask_b_to_a]
                    
            # # 绘制矩形框
            # cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
        return sys_lesion_all
    
    # 计算仿真病灶像素值
    # 合并lesions， 望在两张图像A和B中，对B中不为0的像素进行处理，合并到A上。当B的像素不为0时，如果A的对应像素也不为0，则A不变；如果A的对应像素为0，则直接用B的像素替换。
    def calculate(self, img1, mask,img2):
        # img1 = self.balance_lesion_pix(img1, img2)

        # 初始化仿真病灶
        sys_lesion_all = np.zeros_like(img1)
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for k, contour in enumerate(contours):
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)

            w  += 5
            h += 5
            # 矩形框像素个数
            pix_num_rec = h * w
            # 提取矩形框内的像素
            rect_img1 = img1[y:y+h, x:x+w]
            # 计算矩形框内像素的和
            rec_sum_pixels_values = np.sum(rect_img1)
            
            # 病灶轮廓内像素个数
            pix_num_lesion= np.count_nonzero(mask)
            # 计算病灶坐标和值
            tmp1 = np.zeros_like(img1)
            tmp1[y:y+h,x:x+w] = img1[y:y+h, x:x+w] 
            real_lesion = np.where(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), tmp1, 0)

            # 计算背景图像像素
            back_rect = img2[y:y+h, x:x+w]
            back_rec_sum_pix_values = np.sum(back_rect)
            
            # 等比例放缩
            sys_lesion = (real_lesion * (back_rec_sum_pix_values / rec_sum_pixels_values)).astype(np.uint8)
            # cv2.imwrite('tmp' + str(k) + '.jpg', sys_lesion)

            # 合并lesions， 望在两张图像A和B中，对B中不为0的像素进行处理，合并到A上。当B的像素不为0时，如果A的对应像素也不为0，则取两者的平均值；如果A的对应像素为0，则直接用B的像素替换。
            # sys_lesion_all = cv2.add(sys_lesion_all, sys_lesion) # 不管不顾，直接取值
            mask_b_not_zero = np.any(sys_lesion > 0, axis=2)  # 找到B图像中非0的像素位置

            # 创建一个mask来检测A中对应位置也不为0的情况
            mask_a_not_zero = np.any(sys_lesion_all > 0, axis=2)  # 找到A图像中非0的像素位置

            # 创建一个合并mask，两个条件都满足（B不为0且A也不为0的位置）
            # mask_both_not_zero = mask_b_not_zero & mask_a_not_zero

            # 对于同时不为0的位置，计算平均值
            # for c in range(3):  # 遍历每一个颜色通道
            #     sys_lesion_all[:, :, c][mask_both_not_zero] = (
            #         sys_lesion_all[:, :, c][mask_both_not_zero].astype(np.uint16) +
            #         sys_lesion[:, :, c][mask_both_not_zero].astype(np.uint16)
            #     ) // 2
            # 对于同时不为0的位置，保持A不变
            # for c in range(3):  # 遍历每一个颜色通道
            #     sys_lesion_all[:, :, c][mask_both_not_zero] = (
            #         sys_lesion_all[:, :, c][mask_both_not_zero].astype(np.uint16) +
            #         sys_lesion[:, :, c][mask_both_not_zero].astype(np.uint16)
            #     ) // 2
            # sys_lesion_all[mask_both_not_zero] = sys_lesion_all[mask_both_not_zero]


            # 对于B不为0但A为0的位置，直接使用B的像素值
            mask_a_zero = ~mask_a_not_zero  # A中为0的位置
            mask_b_to_a = mask_b_not_zero & mask_a_zero  # B不为0且A为0的位置

            sys_lesion_all[mask_b_to_a] = sys_lesion[mask_b_to_a]
                    
            # # 绘制矩形框
            # cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
        return sys_lesion_all

    def balance_lesion_pix(self, img1, img2):
        # 病灶像素值均衡
        img1 = img1 * (img2[:, :, 2].mean() / img1[:, :, 2].mean())
        # 将 像素值 低于 值域区间[0, 255] 的 像素点 置0
        img1 *= (img1 > 0)
        # 将 像素值 高于 值域区间[0, 255] 的 像素点 置255
        img1 = img1 * (img1 <= 255) + 255 * (img1 > 255)
        # 将 dtype 转为图片的 dtype : uint8
        img1 = img1.astype(np.uint8)

    # 直方图均衡化
    def zhifangtu(self, image):
        # 将彩色图像转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 对灰度图像进行直方图均衡化
        equalized_image = cv2.equalizeHist(gray_image)

        # 将直方图均衡化后的灰度图像转换为彩色图像
        equalized_color_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

        # 保存直方图均衡化后的图像
        return equalized_color_image

    def forward_20240415(self):
        for i, img1_path in enumerate(self.img_files):
            name = img1_path.split('/')[-1]
            mask_path = img1_path.replace('.jpg', '.png').replace('/x/', '/y/')

            store_path_x = os.path.join(self.args.store_dir, name.split('.')[0], 'x')
            store_path_y = os.path.join(self.args.store_dir, name.split('.')[0], 'y')

            utils.mkdir(store_path_x)
            utils.mkdir(store_path_y)

            choose_img2_list = random.sample(self.normal_images_path, 2)
            for k in range(2):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(choose_img2_list[k])
                # img2 = self.aug(img2)
                img2 = self.remove_the_blackborder(img2)
                img2 = cv2.resize(img2, (565, 584), interpolation=cv2.INTER_CUBIC)

                # 用于溢出部分裁剪
                yichu_mask = np.zeros_like(img2)  # 创建与原始图像同形状的零数组
                # 将大于5的像素值设置为1，小于等于5的设置为0
                yichu_mask[img2 > 50] = 1
                yichu_mask[img2 <= 50] = 0

                img2 = self.liner_trans(img2,0.5)


                img2_name = choose_img2_list[k].split('/')[-1].split('.')[0]
                res_name_x = os.path.join(store_path_x, name.split('.')[0] + '_' + img2_name + '_' + str(k + 1) + '.jpg')
                res_name_y = os.path.join(store_path_y, name.split('.')[0] + '_' + img2_name + '_' + str(k + 1) + '.png')

                # resize images 缩小图片
                img1 = cv2.resize(img1, (565, 450), interpolation=cv2.INTER_NEAREST)
                mask = cv2.resize(mask, (565, 450), interpolation=cv2.INTER_NEAREST)

                # 填充边缘，与背景图像尺寸保持一致
                top, bottom = 67, 67
                img1 = cv2.copyMakeBorder(img1, top, bottom, 0, 0, cv2.BORDER_CONSTANT, None, (0, 0, 0))
                mask = cv2.copyMakeBorder(mask, top, bottom, 0, 0, cv2.BORDER_CONSTANT, None, (0, 0, 0))

                # img1 = self.aug(img1)
                img1 = self.liner_trans(img1, 0.8)

                # img1 = img1 * (img2[:, :, 2].mean() / img1[:, :, 2].mean())
                # 将 像素值 低于 值域区间[0, 255] 的 像素点 置0
                img1 *= (img1 > 0)
                # 将 像素值 高于 值域区间[0, 255] 的 像素点 置255
                img1 = img1 * (img1 <= 255) + 255 * (img1 > 255)
                # 将 dtype 转为图片的 dtype : uint8
                img1 = img1.astype(np.uint8)

                mask_1 = mask / 255.0
                img1[:, :, 0] = img1[:, :, 0] * mask_1
                img1[:, :, 1] = img1[:, :, 1] * mask_1
                img1[:, :, 2] = img1[:, :, 2] * mask_1

                # 弹性变换，位移
                # features aug
                pic_trans = self.trans(image=img1, mask=mask)
                img1 = pic_trans['image']
                mask = pic_trans['mask']

                scenic_mask = ~ mask
                scenic_mask = scenic_mask / 255.0
                img2[:, :, 0] = img2[:, :, 0] * scenic_mask
                img2[:, :, 1] = img2[:, :, 1] * scenic_mask
                img2[:, :, 2] = img2[:, :, 2] * scenic_mask


                # 合并 merge
                result = cv2.add(img2, img1)

                # 裁去溢出的病灶
                result = result * yichu_mask
                mask = mask * yichu_mask[:, :, 0]

                
                cv2.imwrite(res_name_x, result)
                cv2.imwrite(res_name_y, mask)
                # cv2.imshow('img3', result)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    # 选择ssim（结构相似）的图片
    def forward(self):
        for i, img1_path in enumerate(self.img_files):
            name = img1_path.split('/')[-1]
            mask_path = img1_path.replace('.jpg', '.png').replace('/x/', '/y/')

            store_path_x = os.path.join(self.args.store_dir, 'imp_net', name.split('.')[0], 'x')
            store_path_y = os.path.join(self.args.store_dir,'imp_net', name.split('.')[0], 'y')
            utils.mkdir(store_path_x)
            utils.mkdir(store_path_y)

            store_path_x_noimp = os.path.join(self.args.store_dir, 'no_imp', name.split('.')[0], 'x')
            store_path_y_noimp = os.path.join(self.args.store_dir,'no_imp', name.split('.')[0], 'y')
            utils.mkdir(store_path_x_noimp)
            utils.mkdir(store_path_y_noimp)



            # 查询最相似图片
            # ssim_similar_ls = self.choose_similar_ssim(img1_path)
            ssim_similar_ls = task_ut.choose_similar_ssim(img1_path, '/data1/wangjingtao/workplace/python/data/ssim/ssim_odir-5k.json')
            # ssim_similar_ls = random.sample(self.normal_images_path, 20)
            # for img2_path, similarity in ssim_similar_ls:
            for k, img2_name in enumerate(ssim_similar_ls):
                img2_path = os.path.join('/data1/wangjingtao/workplace/python/data/ODIR-5K/train_normal/', img2_name)
                
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                img1 = cv2.imread(img1_path)

                # resize images 缩小图片
                img1 = cv2.resize(img1, (565, 450), interpolation=cv2.INTER_NEAREST)
                mask = cv2.resize(mask, (565, 450), interpolation=cv2.INTER_NEAREST)

                # 填充边缘，与背景图像尺寸保持一致  [565, 450] -->[565, 584]
                top, bottom = 67, 67
                img1 = cv2.copyMakeBorder(img1, top, bottom, 0, 0, cv2.BORDER_CONSTANT, None, (0, 0, 0))
                mask = cv2.copyMakeBorder(mask, top, bottom, 0, 0, cv2.BORDER_CONSTANT, None, (0, 0, 0))
                # img1 = self.liner_trans(img1, 0.8) # 调整对比度
                

                img2 = cv2.imread(img2_path)
                # img2 = self.remove_the_blackborder(img2)
                img2 = task_ut.remove_the_blackborder(img2)
                img2_mask = task_ut.gen_mask(self.gen_mask_net, img2)
                img2 = cv2.resize(img2, (565, 584), interpolation=cv2.INTER_NEAREST)
                img2_mask = cv2.resize(img2_mask, (565, 584), interpolation=cv2.INTER_NEAREST)        
                # img2 = self.liner_trans(img2,0.5)


                img2_name = img2_path.split('/')[-1].split('.')[0]
                res_name_x = os.path.join(store_path_x, name.split('.')[0] + '_' + img2_name + '_' + str(k + 1) + '.jpg')
                res_name_y = os.path.join(store_path_y, name.split('.')[0] + '_' + img2_name + '_' + str(k + 1) + '.png')

                res_name_x_noimp = os.path.join(store_path_x_noimp, name.split('.')[0] + '_' + img2_name + '_' + str(k + 1) + '.jpg')
                res_name_y_noimp = os.path.join(store_path_y_noimp, name.split('.')[0] + '_' + img2_name + '_' + str(k + 1) + '.png')

                # 病灶弹性变换，位移，病灶特征增强
                # features aug
                pic_trans = self.trans(image=img1, mask=mask)
                img1 = pic_trans['image']
                mask = pic_trans['mask']

                # save middle process for analysis, 保留中间过程，便于分析和写论文

                utils.mkdir(store_path_x.replace('imp_net', 'mid_trans_aug'))
                utils.mkdir(store_path_y.replace('imp_net', 'mid_trans_aug'))
                cv2.imwrite(res_name_x.replace('imp_net', 'mid_trans_aug'), img1)
                cv2.imwrite(res_name_y.replace('imp_net', 'mid_trans_aug'), mask)



                # sys_lesion = self.calculate(img1, mask, img2)
                sys_lesion, mask = task_ut.calculate(img1, mask, img2, img2_mask)
                # sys_lesion = self.zhifangtu(sys_lesion)

                scenic_mask = ~ mask
                scenic_mask = scenic_mask / 255.0
                img2[:, :, 0] = img2[:, :, 0] * scenic_mask
                img2[:, :, 1] = img2[:, :, 1] * scenic_mask
                img2[:, :, 2] = img2[:, :, 2] * scenic_mask


                # 裁去位置不合适的病灶
                # result = result * yichu_mask
                # mask = mask * yichu_mask[:, :, 0]

                result = cv2.add(img2, sys_lesion)
                # result = img1

                result_imp = cv2.resize(result, (256, 256), interpolation=cv2.INTER_NEAREST)
                mask_imp = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

                result_imp = task_ut.improve_lesion(self.G, result_imp)

                cv2.imwrite(res_name_x, result_imp)
                cv2.imwrite(res_name_y, mask_imp)
                cv2.imwrite(res_name_x_noimp, result)
                cv2.imwrite(res_name_y_noimp, mask)
                # cv2.imshow('img3', result)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

        real_data_file = os.path.join(self.args.project_path,self.args.real_data_csv)
        reltive_path = os.path.join(self.args.store_dir, 'imp_net/')
        task_ut.gen_sys_csv(real_data_file, reltive_path)

