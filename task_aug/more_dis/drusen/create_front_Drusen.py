import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob

from albumentations import (
    ElasticTransform,
)


def generate_triangle(output_image_name):
    image = np.ones((64, 64), dtype="uint8")
    image *= 0

    pt1 = (32, 3) # 顶点
    pt2 = (3, 64) # 左顶点
    pt3 = (61, 64)# 右顶点

    # cv2.circle(image, pt1, 2, 255, -1)
    # cv2.circle(image, pt2, 2, 255, -1)
    # cv2.circle(image, pt3, 2, 255, -1)
    triangle_cnt = np.array( [pt1, pt2, pt3] )

    cv2.drawContours(image, [triangle_cnt], 0, 255, -1)
    # cv2.imwrite(output_image_name,image)
    base_a = 60
    aug = ElasticTransform(p=1,
                           alpha=base_a,
                           sigma=base_a * 0.131,
                           alpha_affine=base_a * 0.0000365,
                           interpolation=cv2.INTER_NEAREST
                           )

    augmented = aug(image=image)
    image_elastic = augmented['image']
    cv2.imwrite(output_image_name, image_elastic)

def generate_ellipse(output_image_name):
    # 初始化一个空白的画布,背景色为黑色
    # image = np.ones((512, 300), dtype="uint8")
    # image *= 0
    # # 绘制椭圆，椭圆的中心点为（256， 256）， 椭圆的长轴长度为， 椭圆的短轴长度为， angle为椭圆旋转的角度，逆时针方向，
    # cv2.ellipse(img=image, center=(150, 256), axes=(150, 256), angle=0, startAngle=0, endAngle=360, color=(255, 255),
    #             thickness=-1)

    image = np.ones((256, 64), dtype="uint8")
    image *= 0
    # 绘制椭圆，椭圆的中心点为（256， 256）， 椭圆的长轴长度为， 椭圆的短轴长度为， angle为椭圆旋转的角度，逆时针方向，
    cv2.ellipse(img=image, center=(32, 128), axes=(32, 120), angle=0, startAngle=0, endAngle=360, color=(255, 255),
                thickness=-1)

    # cv2.imshow("image",image)
    # cv2.waitKey()
    a = 100
    aug = ElasticTransform(p=1,
                           alpha=a,
                           sigma=a * 0.096,
                           alpha_affine=a * 0.000000000001,
                           interpolation=cv2.INTER_NEAREST,
                           )

    augmented = aug(image=image)

    image_elastic = augmented['image']
    # temp = image_elastic[1:] - image_elastic[:-1]
    # plt.Figure(figsize=(10,10))
    # plt.imshow(temp)
    # hs = np.sum(temp == 255, 0)



    # cv2.imshow("image", image_elastic[:50])
    # cv2.waitKey()
    cv2.imwrite(output_image_name, image_elastic[:50])

#   去除不规则图形
def xiu_jian():
    image_elastic = cv2.imread("./data/Drusen/ellipse_mask_5.png", 0)

    temp = image_elastic[1:] - image_elastic[:-1]

    hs = np.sum(temp == 255, 0)

    indx = np.where(hs >= 2)[0]
    for i in indx:
        print(i)
        tmp_indx = np.where(temp[:, i] == 255)[0]
        print(tmp_indx)

        image_elastic[tmp_indx[-1]:, i] = 0

    plt.Figure(figsize=(10, 10))
    plt.imshow(image_elastic)
    cv2.imshow("tmp", image_elastic)
    cv2.imwrite("tmp.png", image_elastic)
    print("hello")

if __name__ == "__main__":

    for i in range(9500):
        ellipse_mask_name = "./data/Drusen/ellipse_mask_" + str(i) + ".png"
        generate_ellipse(ellipse_mask_name)
        if i == 200:
            time.sleep(2)
    for i in range(500):
        triangle_mask_name = "./data/Drusen/triangle_mask_" + str(i) + ".png"
        generate_triangle(triangle_mask_name)
        if i == 200:
            time.sleep(2)

