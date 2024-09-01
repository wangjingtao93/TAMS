import random

import cv2
import os
import numpy as np
from albumentations import (
    ElasticTransform,
)
import glob

import albumentations as A

import pandas as pd
import common.utils as utils
from task_aug.synthetic.simulate_lesion import SimulateLesions 
from skimage.metrics import structural_similarity as ssim
import json

class SyntheticCNV():
    def __init__(self, args):
        self.args = args
        
        self.sim = SimulateLesions(args)

        # real data with focus, extractor focus features from these images.
        csv_dir = os.path.join(self.args.project_path, self.args.real_data_csv)
        self.bingzao_img_file, self.bingzao_mask_file, self.index_list = self.read_csv(csv_dir)  # all images info(path, label etc)

        # real data without foucus (normal images)
        # normal_csv_dir = 'D:/workplace/python/metaLearning/meta-learning-seg/dataPro/data/ML_csv/train_data.csv'
        normal_data_csv = os.path.join(self.args.project_path, self.args.normal_data_csv)
        self.ori_img_file, self.ori_mask_file,self.nor_idx = self.read_csv(normal_data_csv)  # simulate focus on these images, about 7500 images,在这些OCT图像上进行病灶伪造,应该有7500多张
        

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
                while ela_Count < 10:
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
    
    def forward(self):
        count_step = random.randint(0, len(self.ori_img_file))
        for i, koutu_mask_path in enumerate(self.bingzao_mask_file, 1):
            # index += 86
            name = koutu_mask_path.split('/')[-1].replace('.png', '')
            index = self.index_list[i-1]

            mask_bingzao_list = self.extract_features(koutu_mask_path)
            store_path_x = os.path.join(self.args.store_dir, str(index) + '_' + name, 'x')
            store_path_y = os.path.join(self.args.store_dir, str(index) + '_' + name, 'y')
            utils.mkdir(store_path_y)
            utils.mkdir(store_path_x)

            for idx, bingzao in enumerate(mask_bingzao_list, 1): # 20个
                bingzao = cv2.resize(bingzao,(int(bingzao.shape[1] / 2), bingzao.shape[0]), interpolation=cv2.INTER_NEAREST)

                flag = True
                while flag:
                    out_list = self.sim.sim_use_real_focus(self.ori_img_file[count_step], self.ori_mask_file[count_step],
                                                                bingzao_data=bingzao/255 * 40)

                    count_step += 1
                    if count_step > len(self.ori_img_file) - 1:
                        count_step = 0
                    if len(out_list) == 0:
                        continue

                    x_name = 'faker' + "_" + str(count_step) + '_' + name + '_' + str(idx) + '.jpg'
                    y_name = 'faker' + "_" + str(count_step) + '_' + name + '_' + str(idx) + '.png'

                    cv2.imwrite(store_path_x +  '/' + x_name, out_list[0])
                    cv2.imwrite(store_path_y + '/' + y_name, out_list[1])
                    flag = False

class SyntheticDrusen():
    def __init__(self, args):
        self.args = args
        
        the_command = 'python3 drusen/generate_faker_Drusen.py'
        os.system(the_command)


class SyntheticHeShiRP():
    def __init__(self, args):
        self.args = args
        dataframe = pd.read_csv(os.path.join(self.args.project_path,self.args.real_data_csv))

        self.img_files = list(dataframe["Image_path"])
        self.mask_files = list(dataframe["Label_path"])

        # self.normal_images_path = os.path.join('/data1/wangjingtao/workplace/python/data/rp_data/heshi/normal/normal')
        self.normal_images_path = glob.glob('/data1/wangjingtao/workplace/python/data/rp_data/heshi/normal/normal/*')
        


        self._init_acc()

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

            store_path_x = os.path.join(self.args.store_dir, name.split('.')[0], 'x')
            store_path_y = os.path.join(self.args.store_dir, name.split('.')[0], 'y')
            utils.mkdir(store_path_x)
            utils.mkdir(store_path_y)

            # 查询最相似图片
            ssim_similar_ls = self.choose_similar_ssim(img1_path)
            k = 0
            for img2_path, similarity in ssim_similar_ls:
                img2_path = os.path.join('/data1/wangjingtao/workplace/python/data/rp_data/heshi/normal/normal', img2_path)
                k = k+1
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
                img2 = self.remove_the_blackborder(img2)
                img2 = cv2.resize(img2, (565, 584), interpolation=cv2.INTER_CUBIC)              
                # img2 = self.liner_trans(img2,0.5)


                img2_name = img2_path.split('/')[-1].split('.')[0]
                res_name_x = os.path.join(store_path_x, name.split('.')[0] + '_' + img2_name + '_' + str(k + 1) + '.jpg')
                res_name_y = os.path.join(store_path_y, name.split('.')[0] + '_' + img2_name + '_' + str(k + 1) + '.png')

                # 病灶弹性变换，位移，病灶特征增强
                # features aug
                pic_trans = self.trans(image=img1, mask=mask)
                img1 = pic_trans['image']
                mask = pic_trans['mask']


                sys_lesion = self.calculate(img1, mask, img2)

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
                cv2.imwrite(res_name_x, result)
                cv2.imwrite(res_name_y, mask)
                # cv2.imshow('img3', result)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        
    def choose_similar_ssim(self, img1_path):

        # 已经计算好，直接从本地加载结果
        with open('/data1/wangjingtao/workplace/python/data/rp_data/heshi/ssim.json', 'r') as f:
            loaded_dict = json.load(f)

        img1_name = img1_path.split('/')[-1].split('.')[0]
        return loaded_dict[img1_name]

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
    def calculate(self, img1, mask,img2):
        # img1 = self.balance_lesion_pix(img1, img2)

        # 初始化仿真病灶
        sys_lesion_all = np.zeros_like(img1)
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
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

            sys_lesion_all = cv2.add(sys_lesion_all, sys_lesion)

        # cv2.imwrite('tmp.jpg', sys_lesion_all)
            
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



class SyntheticRIPS():
    def __init__(self, args):
        self.args = args
        

def read_csv(csv_dir):
    dataframe = pd.read_csv(csv_dir)

    img_file = dataframe["Image_path"]
    mask_file = dataframe["Label_path"]

    return list(img_file), list(mask_file)