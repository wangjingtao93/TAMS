import json
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import torchvision.transforms as transform
import pandas as pd
import csv
import glob
import os

def choose_similar_ssim(img1_path, ssim_path, expand_n=20):

    # 已经计算好，直接从本地加载结果
    with open(ssim_path, 'r') as f:
        loaded_dict = json.load(f)

    img1_name = img1_path.split('/')[-1]
    return loaded_dict[img1_name]

    # 速度很慢，后续开发多线程
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
    top_20_similar_images = most_similar_images[:expand_n]
    # 打印或保存这些最相似的图片路径
    # for image_path, similarity in top_20_similar_images:
    #     print(f"Image A: {image_A_path}, Similar Image B: {image_path}, SSIM: {similarity}")

    return top_20_similar_images

# 裁去黑边
def remove_the_blackborder(image):
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


def calculate(img1, mask, img2, img2_mask):
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

    sys_lesion_all[img2_mask==0] = 0 # 去除边缘之外的病灶remove lesions beyond the edges
    sys_lesion_all[img2_mask==2] = 0 # optic disca
    mask[img2_mask==0] = 0 
    mask[img2_mask==2] = 0
    
    return sys_lesion_all, mask

def gen_mask(net, img):
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
        store_out[output[c]==1] = c+1

    return store_out.astype(np.uint8)

def improve_lesion(G, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transforms = transform.Compose([
        transform.ToTensor(),
        # transform.Resize((256, 256)),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transforms(img.copy())
    img = img[None].to('cuda')  # [1,3,128,128]

    out = G(img)[0]
    out = out.permute(1,2,0)
    out = (0.5 * (out + 1)).cpu().detach().numpy() * 255 #RGB

    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


# 合成数据信息写成csv文件
def gen_sys_csv(real_data_csvfile, relative_path):
    # real_data_csvfile = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/data/bv1000/real_data/cnv/all_data.csv'
    df_realdata = pd.read_csv(real_data_csvfile)

    img_files = df_realdata["Image_path"].tolist()
    mask_files = df_realdata["Label_path"].tolist()
    ids = df_realdata['ID'].tolist()
    eyes = df_realdata['Eye'].tolist()

    # relative_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/task_aug/result_sys/20240419/bv1000-oct-cnv/2024-04-26-10-56-48/imp_net/'

    store_csvfile = os.path.join(relative_path.replace('imp_net', ''), 'sys.csv')

    with open(store_csvfile, 'w', newline='') as csvfile:
        fields = ['ID', 'Eye', 'Image_path', 'Label_path']
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(fields)

        for i in range(len(img_files)):
            img_name = mask_files[i].split('/')[-1].replace('.png', '')
            fold_name = img_name

            sys_img_path = os.path.join(relative_path,fold_name)

            sys_imgs = glob.glob(os.path.join(sys_img_path, 'x', '*'))

            sys_imgs.sort(key=lambda element: int(element.split('_')[-1].replace('.jpg', '')))

            for image_path in sys_imgs:
                mask_path = image_path.replace('x', 'y').replace('.jpg', '.png')

                csvwriter.writerow([ids[i], eyes[i], image_path.replace(relative_path, ''), mask_path.replace(relative_path, '')])