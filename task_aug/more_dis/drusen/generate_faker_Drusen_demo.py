
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy
import cv2
import os
import glob
import random
# import imutils
import math

def tmp_test(img1, mask2):
    cha = img1 - mask2
    tmp = cha.ravel()[numpy.flatnonzero(cha)]
    print(tmp)


raw_x_files_list = glob.glob("./data/bak/x/*.jpg")
raw_y_files_list = [raw_x_files_i.replace("/x\\", "/y\\") for raw_x_files_i in raw_x_files_list]

# mask_file = glob(os.path.join(backgroud_mask_dir, "**", "*.png"), recursive=True)
# img_file = [elm.replace(backgroud_mask_dir, backgroud_image_dir).replace("_label", "")[:-4] + ".*" for elm in
#                 mask_file]
mask_file_list = glob.glob("./data/Drusen/*.png")

x_raw = cv2.imread(raw_x_files_list[0], 0)
y_raw = cv2.imread(raw_y_files_list[0], 0)

# 玻璃膜优个数
dursen_num = 1
# dursen_num = random.randint(1, 4)

idx_begin = 250
idx_end =290
drusen_width_begin = 15
drusen_width_end = 28

idx_scope = idx_end - idx_begin
step = idx_scope // dursen_num

idx0_list = []
idx1_list = []
for i in range(dursen_num):
    weight_idx = random.randint(drusen_width_begin, drusen_width_end)
    tmp = idx_begin + step * i
    idx0_list.append(random.randint(tmp, tmp + step))
    idx1_list.append(idx0_list[i] + weight_idx)
    if idx1_list[i] > tmp + step:
        distance = idx1_list[i] - (tmp + step)
        idx0_list[i] = idx0_list[i] - distance
        idx1_list[i] = idx1_list[i] - distance

# idx0, idx1 = 150, 180
# idx0 = random.randint(180, 330)
#
# weight_idx = random.randint(16, 35)
#
# idx1 = idx0 + weight_idx



# x_fake = x_raw.copy()
# y_fake = y_raw.copy()
dim = (1024, 1024)
x_fake = cv2.resize(x_raw, dim, interpolation=cv2.INTER_NEAREST)
y_fake = cv2.resize(y_raw, dim, interpolation=cv2.INTER_NEAREST)
for k in range(dursen_num):
    idx0 = idx0_list[k]
    idx1 = idx1_list[k]
    temp = y_fake[1:] - y_fake[:-1]

    temp[:, :idx0] = 0
    temp[:, idx1:] = 0

    temp_29 = (temp > 0) * (y_fake[1:] == 112)
    temp_21 = (temp > 0) * (y_fake[1:] == 48)
    idx_up_29 = numpy.where(temp_29)
    idx_up_21 = numpy.where(temp_21)

    if len(idx_up_29) == 0 or len(idx_up_21) == 0:
        continue

    mask_file_path = random.choice(mask_file_list)
    front_PED = cv2.imread(mask_file_path, 0)
    # cv2.imshow("tmp1", front_PED)
    # (h, w) = front_PED.shape[:2]
    # (cX, cY) = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D((cX, cY), -45, 1.0)
    # front_PED = cv2.warpAffine(front_PED, M, (w, h))
    # cv2.imshow("tmp2",front_PED)


    front_PED = cv2.resize(front_PED, (idx1 - idx0, 30))

    hs = numpy.sum(front_PED == 255, 0)


    for i in range(min(len(idx_up_29[0]), len(idx_up_21[0]))):
        # idx_offset = random.randint(-5, 5)
        idx_offset = 5
        idx_up_29[0][i] = idx_up_29[0][i] + idx_offset
        for height_idx in range(idx_up_21[0][i], idx_up_29[0][i] - hs[i]):
            len_idx = idx_up_29[0][i] - hs[i] - idx_up_21[0][i]
            if hs[i] == 0:
                hs[i] = 1
            distance = int((height_idx - idx_up_21[0][i]) / (len_idx / hs[i]))
            x_fake[height_idx, idx_up_29[1][i]] = x_fake[height_idx + distance, idx_up_29[1][i]]
            y_fake[height_idx, idx_up_29[1][i]] = y_fake[height_idx + distance, idx_up_29[1][i]]

            # for height_idx in range(idx_up_12[0][i],idx_up[0][i]):
            #     len =  idx_up[0][i] - idx_up_12[0][i]
            #     distance = int((height_idx - idx_up_12[0][i]) / (len / hs[i]))
            #
            #     x_fake[height_idx, idx_up[1][i]] = x_fake[height_idx + distance, idx_up[1][i]]
            #     y_fake[height_idx, idx_up[1][i]] = y_fake[height_idx + distance, idx_up[1][i]]

        y_fake[idx_up_29[0][i] - idx_offset - hs[i]: idx_up_29[0][i] - idx_offset, idx_up_29[1][i]] = 6


cv2.imshow("x_fake.png", x_fake)
cv2.imshow("y_fake.png", y_fake)
cv2.waitKey()


def azimuthAngle( x1,  y1,  x2,  y2):
    angle = 0.0;
    dx = x2 - x1
    dy = y2 - y1
    if  x2 == x1:
        angle = math.pi / 2.0
        if  y2 == y1 :
            angle = 0.0
        elif y2 < y1 :
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif  x2 > x1 and  y2 < y1 :
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif  x2 < x1 and y2 < y1 :
        angle = math.pi + math.atan(dx / dy)
    elif  x2 < x1 and y2 > y1 :
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle * 180 / math.pi)
