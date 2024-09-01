# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import numpy
import cv2
import os
from glob import glob
import logging
import random
import math
import shutil
import numpy as np
from enum import Enum

import logutil

logger = logging.getLogger(__name__)


class FailReason(Enum):
    poor_quality = 1
    others = 2

def single_generate_fake_Drusen():
    pass

def batch_generate_fake_Drusen(x_image_path, y_image_path, mask_file_list, file_f):
    image_name = x_image_path.split("\\")[-1]

    dursen_num = 0
    if random.random() < 0.18:
        dursen_num = 1
    elif random.random() < 0.55:
        dursen_num = 2
    elif random.random() < 0.8:
        dursen_num = 3
    else:
        dursen_num = random.randint(4, 5)

    x_raw = cv2.imread(x_image_path, 0)
    y_raw = cv2.imread(y_image_path, 0)

    dim_raw = (x_raw.shape[1], x_raw.shape[0])

    idx_begin = 300
    idx_end = 750
    drusen_width_begin = 13
    drusen_width_end = 23
    if dim_raw[0] < 800:
        drusen_width_begin = 23
        drusen_width_end = 35
        idx_begin = 450
        idx_end = 650

    dim = (1024, 1024)
    x_fake = cv2.resize(x_raw, dim, interpolation=cv2.INTER_NEAREST)
    y_fake = cv2.resize(y_raw, dim, interpolation=cv2.INTER_NEAREST)

    mask_112 = np.where(y_fake == 112, 1, 0)
    mask_64 = np.where(y_fake == 64, 1, 0)
    mask_48 = np.where(y_fake == 48, 1, 0)
    if np.sum(mask_112) == 0 or np.sum(mask_64) == 0 or np.sum(mask_48) == 0:
        logger.warning(f'{y_image_path} 缺少脉络膜层, 无法伪造玻璃膜疣')
        save_image(False, x_raw, y_raw, x_image_path, y_image_path)
        return

    # line = y_fake[1:] - y_fake[:-1]
    # line_112 = (line > 0) * (y_fake[1:] == 112)
    # idx_112 = np.where(line_112)
    # min_idx = min(idx_112[0])
    # max_idx = max(idx_112[0])
    # angle_112 = azimuthAngle(idx_112[0][0], idx_112[1][0], idx_112[0][-1], idx_112[1][-1])
    # logger.info(angle_112)
    #
    # cv2.imshow("x_raw", x_fake)
    # cv2.imshow("y_raw", y_fake)
    # cv2.waitKey()

    quality_frame = x_fake.copy()
    quality_mask = y_fake.copy()
    all_image_quality = get_all_image_quality(quality_frame, quality_mask)
    logger.info(f'{x_image_path} 的--all_image_quality =  {all_image_quality}')
    if all_image_quality < all_image_quality_threshold:  # 28 32 35 50
        logger.warning(f'{x_image_path} 质量太差 = {all_image_quality} 无法伪造玻璃膜优')
        save_image(False, x_fake, y_fake, x_image_path, y_image_path, reason=FailReason.poor_quality)
        return

    idx0_list, idx1_list = idx_drusen(dursen_num, drusen_width_begin, drusen_width_end, idx_begin, idx_end)

    if_save_image = False
    k = 0
    tmp_count_2 = 0
    tmp_count_1 = 0
    tmp_count_3 = 0
    while k < dursen_num:
        idx0 = idx0_list[k]
        idx1 = idx1_list[k]
        k += 1

        drusen_file_path = random.choice(mask_file_list)
        if "triangle" in drusen_file_path:
            if dim_raw[0] < 800:
                if idx1 - idx0 > 25:
                    width = random.randint(14, 20)
                    idx1 = idx0 + width
            else:
                if idx1 - idx0 > 15:
                    width = random.randint(9, 12)

        temp = y_fake[1:] - y_fake[:-1]
        temp[:, :idx0] = 0
        temp[:, idx1:] = 0

        temp_112 = (temp > 0) * (y_fake[1:] == 112)
        temp_48 = (temp > 0) * (y_fake[1:] == 48)
        temp_64 = (temp > 0) * (y_fake[1:] == 64)

        idx_up_112 = numpy.where(temp_112.transpose())
        idx_up_48 = numpy.where(temp_48.transpose())
        idx_up_64 = numpy.where(temp_64.transpose())

        width_iter = min(len(idx_up_112[0]), len(idx_up_48[0]), len(idx_up_64[0]))

        if (temp_112 == 0).all() or (temp_48 == 0).all() or (temp_64 == 0).all() or width_iter < idx1 - idx0 - 4 or max(
                idx_up_112[1]) + 50 > 1024:
            if tmp_count_1 < 5:
                idx0_list, idx1_list = idx_drusen(dursen_num, drusen_width_begin, drusen_width_end, idx_begin, idx_end)
                k -= 1
                tmp_count_1 += 1
                if_save_image = False
                logger.warning(f"图片{image_name} 的位置 {idx0} {idx1} 无法定位脉络膜，尝试随机另一个位置")
                logger.warning(f"tmp_count_1 = {tmp_count_1}")
            else:
                if_save_image = False
                logger.warning(f"图片{image_name} 的位置 {idx0} {idx1} 无法在该位置成玻璃膜优")
                logger.warning(f"tmp_count_1**** = {tmp_count_1}")
            continue
        else:
            tmp_count_1 = 0
            if_save_image = True

        box_bolimoyoumin = [idx0 - 10, min(idx_up_48[1]), idx1 + 10, max(idx_up_112[1])]

        ret = check_idx(box_bolimoyoumin, image_name, dim_raw)
        if ret is False:
            if tmp_count_3 < 5:
                idx0_list, idx1_list = idx_drusen(dursen_num, drusen_width_begin, drusen_width_end, idx_begin, idx_end)
                k -= 1
                tmp_count_3 += 1
                if_save_image = False
                logger.warning(f"图片{image_name} 的位置 {idx0} {idx1} 有其他病灶，尝试随机另一个位置")
                logger.warning(f"tmp_count_3 = {tmp_count_3}")
            else:
                if_save_image = False
                logger.warning(f"图片{image_name} 的位置 {idx0} {idx1} 未随机到无病灶位置，无法在该位置生成玻璃膜优")
                logger.warning(f"tmp_count_1**** = {tmp_count_3}")
            continue
        else:
            tmp_count_3 = 0
            if_save_image = True

        quality_frame = x_fake.copy()
        quality_mask = y_fake.copy()
        part_image_quality = get_part_quality(quality_frame, quality_mask, idx0, idx1)
        logger.info(f"part_image_quality = {part_image_quality}")
        if part_image_quality < part_image_quality_threshold:  # 160 170 180
            if tmp_count_2 < 5:
                idx0_list, idx1_list = idx_drusen(dursen_num, drusen_width_begin, drusen_width_end, idx_begin, idx_end)
                k -= 1
                tmp_count_2 += 1
                if_save_image = False
                # wjt debug
                logger.warning(f"tmp_count_22222 =  {tmp_count_2}")
                logger.warning(
                    f"{x_image_path} 的[ {str(idx0)} 到 {str(idx1)} ] 位置质量太差 = {str(part_image_quality)}, 尝试随机另外一个位置")
            else:
                if_save_image = False
                # wjt debug
                logger.warning(f"tmp_count_222222 =  {tmp_count_1}")
                logger.warning(f"位置 {idx0_list[k - 1]}, {idx1_list[k - 1]} 质量太差，无法在该位置生成玻璃膜优")
            continue
        else:
            tmp_count_2 = 0
            if_save_image = True

        front_PED = cv2.imread(drusen_file_path, 0)

        temp_height_list = []
        for i in range(min(len(idx_up_112[1]), len(idx_up_64[1]))):
            temp_height_list.append(idx_up_112[1][i] - idx_up_64[1][i])

        max_height = max(temp_height_list)
        dursen_height_2 = max_height // 2
        front_PED_2 = cv2.resize(front_PED, (idx1 - idx0, dursen_height_2), cv2.INTER_NEAREST)
        hs_2 = numpy.sum(front_PED_2 == 255, 0)

        dursen_height_1 = 0
        # dursen_height_1 = max(temp_height_list) + random.randint(1, 3)
        probility_dursen_height_1 = random.random()
        if probility_dursen_height_1 < 0.28:
            dursen_height_1 = max_height + random.randint(dursen_height_2 // 3, dursen_height_2 // 2)
        elif probility_dursen_height_1 < 0.34:
            dursen_height_1 = max_height + random.randint(dursen_height_2 // 2, dursen_height_2 // 2 + 2)
        elif probility_dursen_height_1 < 0.5:
            dursen_height_1 = max_height + random.randint(0, dursen_height_2 // 5)
        else:
            dursen_height_1 = max_height + random.randint(dursen_height_2 // 5, dursen_height_2 // 4)

        front_PED = cv2.resize(front_PED, (idx1 - idx0, dursen_height_1), cv2.INTER_NEAREST)
        hs = numpy.sum(front_PED == 255, 0)

        idx_offset = 0
        if random.random() < 0.:
            idx_offset = random.randint(-2, 1)
        else:
            idx_offset = random.randint(4, 7)
        idx_up_6_list = []
        for i in range(width_iter):

            if hs[i] == 0:
                hs[i] = 1
            if hs_2[i] == 0:
                hs_2[i] = 1
            # width_average = idx_up_112[1][i] - idx_up_64[1][i]
            # zengliang = np.sum(
            #     x_fake[idx_up_112[1][i] - width_average: idx_up_112[1][i], idx_up_112[0][i] - 5:idx_up_112[0][i] + 5]) // (
            #                     width_average * 10)
            #
            # zengliang_tmp = x_fake[idx_up_112[1][i] - 1, idx_up_112[0][i]]
            #
            # # logger.info("增亮", width_average, zengliang_tmp, zengliang)

            # ave_light = x_fake[idx_up_112[1][i] - hs[i]:idx_up_112[1][i], idx_up_112[0][i]].mean()
            # max_light = max(x_fake[idx_up_112[1][i] - hs[i]:idx_up_112[1][i], idx_up_112[0][i]])
            # for height_idx in range(idx_up_112[1][i] - hs[i], idx_up_112[1][i] - 5):
            #     if x_fake[height_idx, idx_up_112[0][i]] < ave_light:
            #         if random.random() < 0.:
            #             x_fake[height_idx, idx_up_112[0][i]] = max_light
            #         else:
            #             x_fake[height_idx, idx_up_112[0][i]] = random.randint(max_light - 10, max_light)

            idx_up_112[1][i] = idx_up_112[1][i] + idx_offset
            tmp_idx_6_list = []
            for height_idx in range(idx_up_48[1][i], idx_up_112[1][i] - hs[i]):

                # if hs[i] - width_average < 3:
                #     hs[i] = hs[i] + random

                # len_idx = idx_up_112[1][i] - hs[i] - idx_up_48[1][i]
                # if hs[i] == 0:
                #     hs[i] = 1
                # distance = int((height_idx - idx_up_48[1][i]) / (len_idx / hs[i]))

                len_idx = idx_up_112[1][i] - hs[i] - idx_up_48[1][i]

                distance = int(abs(height_idx - idx_up_48[1][i]) / (len_idx / hs[i]))
                if height_idx > idx_up_64[1][i]:
                    # if distance == 0:
                    #     distance = 1
                    # temp_len_idx = idx_up_112[1][i] - idx_up_64[1][i]
                    # distance = int(abs(height_idx - idx_up_64[1][i]) / (temp_len_idx / distance))
                    # distance = distance // 3
                    #     # logger.info("distance = ", distance)
                    #     len_64 = idx_up_112[1][i] - idx_up_64[1][i]
                    #     if height_idx -  idx_up_64[1][i] <  len_64//2:
                    #         x_fake[height_idx, idx_up_112[0][i]] = x_fake[height_idx + len_64//2, idx_up_112[0][i]]
                    #     else:
                    #         x_fake[height_idx, idx_up_112[0][i]] = x_fake[height_idx + distance, idx_up_112[0][i]]
                    # else:
                    #     x_fake[height_idx, idx_up_112[0][i]] = x_fake[height_idx + distance, idx_up_112[0][i]]
                    # x_fake[height_idx, idx_up_112[0][i]] = max(x_fake[height_idx, idx_up_112[0][i]],
                    #                                           x_fake[height_idx + distance, idx_up_112[0][i]])
                    x_fake[height_idx, idx_up_112[0][i]] = x_fake[height_idx + distance, idx_up_112[0][i]]
                else:
                    x_fake[height_idx, idx_up_112[0][i]] = x_fake[height_idx + distance, idx_up_112[0][i]]

                if height_idx + distance > idx_up_64[1][i]:
                    y_fake[height_idx - idx_offset, idx_up_112[0][i]] = 6
                    tmp_idx_6_list.append(height_idx - idx_offset)
                else:
                    y_fake[height_idx, idx_up_112[0][i]] = y_fake[height_idx + distance, idx_up_112[0][i]]
            # if random.random() > 0.35:
            #     for height_idx in range(idx_up_112[1][i] - hs[i] - idx_offset, idx_up_112[1][i] - idx_offset):
            #         if x_fake[height_idx, idx_up_112[0][i]] < zengliang - 25:
            #             x_fake[height_idx, idx_up_112[0][i]] = random.randint(zengliang - 5, zengliang + 10)
            y_fake[idx_up_112[1][i] - idx_offset - hs[i]: idx_up_112[1][i] - idx_offset, idx_up_112[0][i]] = 6
            if len(tmp_idx_6_list) != 0:
                idx_up_6_list.append(min(idx_up_112[1][i] - idx_offset - hs[i], min(tmp_idx_6_list)))
            else:
                idx_up_6_list.append(idx_up_112[1][i] - idx_offset - hs[i])
            # y_fake[idx_up_112[1][i]  - hs[i]: idx_up_112[1][i], idx_up_112[0][i]] = 6

            # ave_light = x_fake[idx_up_112[1][i] - hs[i]:idx_up_112[1][i], idx_up_112[0][i]].mean()
            # max_light = max(x_fake[idx_up_112[1][i] - hs[i]:idx_up_112[1][i], idx_up_112[0][i]])
            # for height_idx in range(idx_up_112[1][i] - hs[i], idx_up_112[1][i] - idx_offset - 5):
            #     if x_fake[height_idx, idx_up_112[0][i]] < ave_light:
            #         if random.random() < 0.:
            #             x_fake[height_idx, idx_up_112[0][i]] = max_light
            #         else:
            #             x_fake[height_idx, idx_up_112[0][i]] = random.randint(max_light - 10, max_light)
            # len_idx = hs[i]
            # if hs_2[i] == 0:
            #     hs_2[i] = 1
            # distance = int(abs(height_idx - idx_up_112[1][i]) / (len_idx / hs_2[i]))

            for height_idx in range(idx_up_112[1][i] - hs[i] - 3, idx_up_112[1][i] - idx_offset):
                if height_idx - (idx_up_112[1][i] - hs[i]) < (idx_up_112[1][i] - idx_offset - idx_up_64[1][i]) // 2 - 5:
                    temp_2_len_idx = hs[i]
                    distance = int(abs(height_idx - idx_up_112[1][i]) / (temp_2_len_idx / hs_2[i]))
                    x_fake[height_idx, idx_up_112[0][i]] = x_fake[height_idx + distance, idx_up_112[0][i]]

        # cv2.imshow(x_image_path, x_fake)
        # cv2.imshow(y_image_path, y_fake)
        # cv2.waitKey()
        # plt.figure(figsize=(10,10), facecolor = "black")
        # plt.imshow(y_fake)

        min_idx_cols = min(idx_up_112[0]) - random.randint(3, 5)
        max_idx_cols = max(idx_up_112[0]) + random.randint(3, 5)
        # tmp = idx_up_112[1]-hs
        # min_idx_rows = min(tmp)
        min_idx_rows = min(idx_up_6_list) - random.randint(3, 5)
        max_idx_rows = max(idx_up_112[1]) + random.randint(3, 5)
        idx_kuang_fir = [min_idx_cols, min_idx_rows]
        idx_kuang_sec = [max_idx_cols, max_idx_rows]

        image_name = x_image_path.split("\\")[1]

        info = image_name + " " + str(idx_kuang_fir[0]) + " " + str(idx_kuang_fir[1]) + " " + str(
            idx_kuang_sec[0]) + " " + str(idx_kuang_sec[1]) + "\n"

        file_f.writelines(info)
    x_fake = cv2.resize(x_fake, dim_raw, interpolation=cv2.INTER_NEAREST)
    y_fake = cv2.resize(y_fake, dim_raw, interpolation=cv2.INTER_NEAREST)
    if if_save_image:
        save_image(True, x_fake, y_fake, x_image_path, y_image_path)
    else:
        logger.error(f'{image_name} 无法生成玻璃膜优')
        save_image(False, x_raw, y_raw, x_image_path, y_image_path)


# 随机玻璃膜位置
def idx_drusen(dursen_num, drusen_width_begin, drusen_width_end, idx_begin, idx_end):
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

    return idx0_list, idx1_list


def check_idx(box_bolimoyou, image_name, dim_raw):
    if image_name not in label_dict:
        return True
    resize_factor = 1024 / dim_raw[0]
    for info_list in label_dict[image_name]:
        box_info = []
        box_info.append(int(math.ceil(float(info_list[0]) * resize_factor)))
        box_info.append(int(info_list[1]))
        box_info.append(int(math.ceil(float(info_list[2]) * resize_factor)))
        box_info.append(int(info_list[3]))
        if overlap(box_bolimoyou, box_info):
            return False  # 矩形相交，此位置不行
    return True


def overlap(box1, box2):
    # 判断两个矩形是否相交
    # 思路来源于:https://www.cnblogs.com/avril/archive/2013/04/01/11293875.html
    # 然后把思路写成了代码
    minx1, miny1, maxx1, maxy1 = box1
    minx2, miny2, maxx2, maxy2 = box2
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx > maxx or miny > maxy:
        return False
    else:
        return True


# 计算脉络膜倾斜角度
def azimuthAngle(x1, y1, x2, y2):
    angle = 0.0;
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle * 180 / math.pi)


def save_image(if_save_image, image, mask, x_image_path, y_image_path, reason=FailReason.others):
    if if_save_image:
        save_x_dir = output_x_fake_dir
        save_y_dir = output_y_fake_dir
    else:
        if reason == FailReason.poor_quality:
            save_x_dir = fail_poor_quality_x_dir
            save_y_dir = fail_poor_quality_y_dir
        else:
            save_x_dir = fail_others_x_dir
            save_y_dir = fail_others_y_dir

    output_x_fake_image = save_x_dir + "/" + x_image_path.split("\\")[1]
    output_y_fake_image = save_y_dir + "/" + y_image_path.split("\\")[1]
    cv2.imwrite(output_x_fake_image, image)
    cv2.imwrite(output_y_fake_image, mask)
    if if_save_image is False:
        image_name = x_image_path.split("\\")[1]
        logging.info(f'Image {image_name} can not be created Drusen')


def check_huangban(y_fake):
    temp = y_fake[1:] - y_fake[:-1]

    plt.Figure(figsize=(10, 10))
    plt.imshow(temp)

    liner_112 = (temp > 0) * (y_fake[1:] == 112)
    plt.Figure(figsize=(10, 10))
    plt.imshow(liner_112)
    liner_64 = (temp > 0) * (y_fake[1:] == 64)
    liner_48 = (temp > 0) * (y_fake[1:] == 48)
    liner_32 = (temp > 0) * (y_fake[1:] == 32)
    liner_80 = (temp > 0) * (y_fake[1:] == 80)


def get_all_image_quality(frame, mask, threshold=150):
    img = frame.copy()
    mailuomo = numpy.where(mask == 112, 1, 0).astype("uint8")
    shibaipan = numpy.where(mask == 80, 1, 0).astype("uint8")
    mailuomo_shibaipan = mailuomo + shibaipan
    temp = numpy.zeros_like(mailuomo, dtype="uint8")
    pos = []
    distance_fix = True
    for i in range(temp.shape[1]):
        idx = numpy.where(mailuomo_shibaipan[:, i] > 0)[0]
        if len(idx) != 0:
            temp[:max(idx), i] = 1
            pos.append([max(idx), i])
    distance_list = []
    for i in range(1, len(pos)):
        distance_list.append(((pos[i - 1][0]) ** 2 + (pos[i - 1][1]) ** 2) ** 0.5)
        pass
    if distance_list != []:
        # logger.info(max(distance_list), (temp.shape[1] / 12))
        if max(distance_list) > (temp.shape[1] / 12):
            distamce_fix = False
    mask *= temp
    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(4, 4))
    img = clahe.apply(img)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    kernel = numpy.ones((5, 5), numpy.uint8)
    th3 = cv2.erode(th3, kernel) * (mask > 0)
    # th4 = th3 * (numpy.where(mask == 64, 1, 0) + numpy.where(mask == 48, 1, 0) + numpy.where(mask == 32, 1, 0))
    th4 = th3 * (numpy.where(mask == 64, 1, 0))
    # value1 = (th3.sum() + th4.sum() ) / th3.shape[0] / th3.shape[1] / 255 * 1000 / 1.5
    value1 = (th3.sum() + th4.sum()) / frame.sum() * 1000 / 1.5
    if distance_fix:
        value = value1
    else:
        value = value1 * 0.9
    return value


def get_part_quality_bak1(frame, mask, idx_begin, idx_end, threshold=150, ):
    mask[:, :idx_begin] = 0
    mask[:, idx_end:] = 0
    frame[:, :idx_begin] = 0
    frame[:, idx_end:] = 0
    img = frame.copy()
    mailuomo = numpy.where(mask == 112, 1, 0).astype("uint8")
    # shibaipan = numpy.where(mask == 80, 1, 0).astype("uint8")
    # mailuomo_shibaipan = mailuomo + shibaipan
    temp = numpy.zeros_like(mailuomo, dtype="uint8")
    pos = []
    distance_fix = True
    for i in range(temp.shape[1]):
        idx = numpy.where(mailuomo[:, i] > 0)[0]
        if len(idx) != 0:
            temp[:max(idx), i] = 1
            pos.append([max(idx), i])
    distance_list = []
    for i in range(1, len(pos)):
        distance_list.append(((pos[i - 1][0]) ** 2 + (pos[i - 1][1]) ** 2) ** 0.5)
        pass
    if distance_list != []:
        # logger.info(max(distance_list), (temp.shape[1] / 12))
        if max(distance_list) > (temp.shape[1] / 12):
            distamce_fix = False
    mask *= temp
    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(4, 4))
    img = clahe.apply(img)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    kernel = numpy.ones((5, 5), numpy.uint8)
    th3 = cv2.erode(th3, kernel) * (mask > 0)
    # th4 = th3 * (numpy.where(mask == 64, 1, 0) + numpy.where(mask == 48, 1, 0) + numpy.where(mask == 32, 1, 0))
    # value1 = (th3.sum() + th4.sum() * 3) / th3.shape[0] / (idx_end - idx_begin) / 255 * 1000 / 1.5
    th4 = th3 * (numpy.where(mask == 64, 1, 0))
    value1 = (th3.sum() + th4.sum()) / th3.shape[0] / (idx_end - idx_begin) / 255 * 1000 / 1.5
    if distance_fix:
        value = value1
    else:
        value = value1 * 0.9
    return value


def get_part_quality(frame, mask, idx_begin, idx_end, threshold=150, ):
    img = frame.copy()
    mask_cajian = mask.copy()
    mask_cajian[:, :idx_begin] = 0
    mask_cajian[:, idx_end:] = 0
    frame[:, :idx_begin] = 0
    frame[:, idx_end:] = 0

    mailuomo = numpy.where(mask == 112, 1, 0).astype("uint8")
    # shibaipan = numpy.where(mask == 80, 1, 0).astype("uint8")
    # mailuomo_shibaipan = mailuomo + shibaipan
    temp = numpy.zeros_like(mailuomo, dtype="uint8")
    pos = []
    distance_fix = True
    for i in range(temp.shape[1]):
        idx = numpy.where(mailuomo[:, i] > 0)[0]
        if len(idx) != 0:
            temp[:max(idx), i] = 1
            pos.append([max(idx), i])
    distance_list = []
    for i in range(1, len(pos)):
        distance_list.append(((pos[i - 1][0]) ** 2 + (pos[i - 1][1]) ** 2) ** 0.5)
        pass
    if distance_list != []:
        # logger.info(max(distance_list), (temp.shape[1] / 12))
        if max(distance_list) > (temp.shape[1] / 12):
            distamce_fix = False
    mask *= temp
    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(4, 4))
    img = clahe.apply(img)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    kernel = numpy.ones((5, 5), numpy.uint8)
    th3 = cv2.erode(th3, kernel) * (mask > 0)
    # th4 = th3 * (numpy.where(mask == 64, 1, 0) + numpy.where(mask == 48, 1, 0) + numpy.where(mask == 32, 1, 0))
    # value1 = (th3.sum() + th4.sum() * 3) / th3.shape[0] / (idx_end - idx_begin) / 255 * 1000 / 1.5
    th4_caijian = th3 * (numpy.where(mask_cajian == 64, 1, 0))
    th4_raw = th3 * (numpy.where(mask == 64, 1, 0))
    logger.info(f"th4.sum() = = =  {th4_caijian.sum()}, {th4_raw.sum()}")

    value1 = (th4_caijian.sum()) / th4_raw.sum() / ((idx_end - idx_begin) * 0.35) * 100000 / 2
    if distance_fix:
        value = value1
    else:
        value = value1 * 0.9
    return value


def generate_kuang_one_by_one(x_image_path, y_image_path):
    pass


def generate_kuang_all():
    mask_file = glob(os.path.join(output_y_fake_dir, "**", "*.png"), recursive=True)
    img_file = [elm.replace(output_y_fake_dir, output_x_fake_dir).replace("_label", "")[:-4] + ".*" for elm in
                mask_file]
    # mask_file_dir = "/192.168.128.161/temp/WJC/wjt_fake_dursen/result7/y"
    # img_file_dir = "/192.168.128.161/temp/WJC/wjt_fake_dursen/result7/x"
    # mask_file = glob(os.path.join(mask_file_dir, "**", "*.png"), recursive=True)
    # img_file = [elm.replace(mask_file_dir, img_file_dir).replace("_label", "")[:-4] + ".*" for elm in
    #             mask_file]
    for i in range(len(img_file)):
        img_file_i = glob(img_file[i])
        assert len(img_file_i) == 1, "multi img_file_i : %s" % img_file
        img_file_i = img_file_i[0]
        assert img_file_i.endswith("jpg") or img_file_i.endswith("png"), "img_file_i type error : %s" % img_file_i
        img_file[i] = img_file_i

    check_kuang_path = "./data/result/check_kuang.txt"
    # if os.path.isfile(check_kuang_path):  # 先检查文件是否存在
    #     os.remove(check_kuang_path)
    with open(check_kuang_path, "w", encoding="utf-8", ) as f:
        for i in range(len(mask_file)):
            image_name = img_file[i].split("\\")[1]
            y_raw = cv2.imread(mask_file[i], 0)
            # imageErZhi = np.array(255 * (y_raw == 6) + 255 * (y_raw == 176), dtype=np.uint8)
            imageErZhi = np.array(255 * (y_raw == 6), dtype=np.uint8)

            cnts, hierarchy = cv2.findContours(imageErZhi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                # 找到边界坐标
                x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
                idx_kuang_fir = [x - random.randint(3, 5), y - random.randint(3, 5)]
                idx_kuang_sec = [x + w + random.randint(3, 5), y + h + random.randint(3, 5)]
                info = image_name + " " + str(idx_kuang_fir[0]) + " " + str(idx_kuang_fir[1]) + " " + str(
                    idx_kuang_sec[0]) + " " + str(idx_kuang_sec[1]) + "\n"
                f.writelines(info)


def my_test_api_1():
    raw_x_files_list = glob("./data/bak/x/*.jpg")
    raw_y_files_list = [raw_x_files_i.replace("/x\\", "/y\\") for raw_x_files_i in raw_x_files_list]
    # generate_fake_Drusen_image(raw_x_files_list[0], raw_y_files_list[0], fake_Drusen_list)
    img = cv2.imread(raw_x_files_list[0], 0)
    mask = cv2.imread(raw_y_files_list[0], 0)
    mailuomo = numpy.where(mask == 112, 1, 0).astype("uint8")
    shibaipan = numpy.where(mask == 64, 1, 0).astype("uint8")

    img1 = img * mailuomo
    img2 = img * shibaipan
    cv2.imshow("112", img1)
    cv2.imshow("64", img2)
    cv2.waitKey()


# 测试函数： 获取图片质量
def my_test_api_2():
    mask_file = glob(os.path.join("./data/result/y", "**", "*.png"), recursive=True)
    img_file = [elm.replace("./data/result/y", "./data/result/x").replace("_label", "")[:-4] + ".*" for elm in
                mask_file]
    for i in range(len(img_file)):
        img_file_i = glob(img_file[i])
        assert len(img_file_i) == 1, "multi img_file_i : %s" % img_file
        img_file_i = img_file_i[0]
        assert img_file_i.endswith("jpg") or img_file_i.endswith("png"), "img_file_i type error : %s" % img_file_i
        img_file[i] = img_file_i

    for i in range(len(mask_file)):
        x_raw = cv2.imread(img_file[i], 0)
        y_raw = cv2.imread(mask_file[i], 0)
        # x_raw = cv2.imread("./data/result/x/20200324_121949_bscan_80.jpg", 0)
        # y_raw = cv2.imread("./data/result/y/20200324_121949_bscan_80_label.png", 0)
        logger.info(get_all_image_quality(x_raw, y_raw))
        #  logger.info(get_part_quality(x_raw, y_raw, 530,  580))
        cv2.imshow(img_file[i], x_raw)
        cv2.imshow(mask_file[i], y_raw)
        cv2.waitKey()


def my_test_api_3():
    mask_file = glob(os.path.join("./data/bak/y", "**", "*.png"), recursive=True)
    img_file = [elm.replace("./data/bak/y", "./data/bak/x").replace("_label", "")[:-4] + ".*" for elm in
                mask_file]
    for i in range(len(img_file)):
        img_file_i = glob(img_file[i])
        assert len(img_file_i) == 1, "multi img_file_i : %s" % img_file
        img_file_i = img_file_i[0]
        assert img_file_i.endswith("jpg") or img_file_i.endswith("png"), "img_file_i type error : %s" % img_file_i
        img_file[i] = img_file_i
    for i in range(len(mask_file)):
        x_raw = cv2.imread(img_file[i], 0)
        y_raw = cv2.imread(mask_file[i], 0)
        logger.info(get_all_image_quality(x_raw, y_raw))
        generate_fake_Drusen_image(img_file[i], mask_file[i], fake_Drusen_list)
        # cv2.imshow("x_raw", x_raw)
        # cv2.imshow("y_raw", y_raw)
        # cv2.waitKey()


def my_test_api_4():
    mask_file = glob(os.path.join("./data/nor_image/y", "**", "*.png"), recursive=True)
    img_file = [elm.replace("./data/nor_image/y", "./data/nor_image/x").replace("_label", "")[:-4] + ".*" for elm in
                mask_file]
    for i in range(len(img_file)):
        img_file_i = glob(img_file[i])
        assert len(img_file_i) == 1, "multi img_file_i : %s" % img_file
        img_file_i = img_file_i[0]
        assert img_file_i.endswith("jpg") or img_file_i.endswith("png"), "img_file_i type error : %s" % img_file_i
        img_file[i] = img_file_i

    clean_forward(dir_list=["./data/result/x", "./data/result/y"])
    for i in range(len(mask_file)):
        generate_fake_Drusen_image(img_file[i], mask_file[i], fake_Drusen_list)
        # cv2.imshow("x_raw", x_raw)
        # cv2.imshow("y_raw", y_raw)
        # cv2.waitKey()


# 测试矩形相交函数
def my_test_api_5():
    label_text_path = "D:/workplace/python/data-augmentation/argumentation_project/data/result/check_kuang.txt"
    label_dict_test = {}
    if os.path.isfile(label_text_path):  # 先检查文件是否存在
        with open(label_text_path, 'r') as f:
            for info in f.readlines():
                info_list = info.replace("\n", "").split(" ")
                key_image_name = info_list[0]
                value_list = info_list[1:]
                if key_image_name not in label_dict_test:
                    label_dict_test[key_image_name] = [value_list]
                else:
                    label_info = label_dict_test[key_image_name]
                    label_info.append(value_list)
                    label_dict_test[key_image_name] = label_info

    for image_name, bolimoyou_list in label_dict_test.items():
        if image_name not in label_dict:
            continue
        for bolimoyou in bolimoyou_list:
            box1 = [int(x) for x in bolimoyou]
            other_bingzao_list = label_dict[image_name]
            for other_bingzao in other_bingzao_list:
                box2 = [int(x) for x in other_bingzao]
                if overlap(box1, box2):
                    logger.error(f'{image_name} 选取的玻璃膜疣位置{box1}和其他病灶的位置{box2} 有重叠')


def read_labelTxt():
    label_text_path = "D:/workplace/python/data/BV1000_detaction/mini_train_label.txt"
    label_dict = {}
    if os.path.isfile(label_text_path):  # 先检查文件是否存在
        with open(label_text_path, 'r') as f:
            for info in f.readlines():
                info_list = info.replace("\n", "").split(" ")
                key_image_name = info_list[0]
                value_list = info_list[2:]
                if key_image_name not in label_dict:
                    label_dict[key_image_name] = [value_list]
                else:
                    label_info = label_dict[key_image_name]
                    label_info.append(value_list)
                    label_dict[key_image_name] = label_info

    return label_dict


def my_mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        logger.info(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print(path + ' 目录已存在')
        return False


def process_prepare():
    my_mkdir(output_x_fake_dir)
    my_mkdir(output_y_fake_dir)

    my_mkdir(fail_poor_quality_x_dir)
    my_mkdir(fail_poor_quality_y_dir)

    my_mkdir(fail_others_x_dir)
    my_mkdir(fail_others_y_dir)


def clean_forward(dir_list=[]):
    dir_list.append(output_x_fake_dir)
    dir_list.append(output_y_fake_dir)
    dir_list.append(fail_image_dir)

    for top in dir_list:
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            # for name in dirs:
            #     os.rmdir(os.path.join(root, name))


def store_every_result():
    houzhui = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    new_dir_name = "./data/store_result/store_" + houzhui
    os.makedirs(new_dir_name + "/result")
    # os.makedirs(new_dir_name + "/result/y")
    os.makedirs(new_dir_name + "/nor_image")
    # os.makedirs(new_dir_name + "/nor_image/y")
    shutil.copytree(output_x_fake_dir, new_dir_name + "/result/x")
    shutil.copytree(output_y_fake_dir, new_dir_name + "/result/y")
    # shutil.copytree(save_bad_quality_x_dir, new_dir_name + "/nor_image/x")
    # shutil.copytree(save_bad_quality_y_dir, new_dir_name + "/nor_image/y")


def main():
    clean_forward()
    process_prepare()
    f = open(check_kuang_path, "w", encoding="utf-8", )
    for i in range(len(img_file)):
        batch_generate_fake_Drusen(img_file[i], mask_file[i], fake_Drusen_list, f)
        if i % 100 == 0 and i != 0:
            time.sleep(2)
    f.close()


if __name__ == "__main__":

    logger = logutil.logs()
    backgroud_mask_dir = "./data/backgroud/detaction/y"
    backgroud_image_dir = "./data/backgroud/detaction/x"
    # backgroud_mask_dir = "D:/workplace/python/data/BV1000_detaction/mini_mask"
    # backgroud_image_dir = "D:/workplace/python/data/BV1000_detaction/mini_trainset"
    check_kuang_path = "./data/result/check_kuang.txt"

    output_x_fake_dir = "./data/result/succ_res/x"
    output_y_fake_dir = "./data/result/succ_res/y"

    fail_image_dir = "./data/result/fail_image"
    fail_poor_quality_x_dir = fail_image_dir + "/poor_quality/x"
    fail_poor_quality_y_dir = fail_image_dir + "/poor_quality/y"
    fail_others_x_dir = fail_image_dir + "/others/x"
    fail_others_y_dir = fail_image_dir + "/others/y"
    # fail_reson = ["pool_quality", "others"]

    label_dict = read_labelTxt()

    all_image_quality_threshold = 32
    part_image_quality_threshold = 150  # 160 170 180

    mask_file = glob(os.path.join(backgroud_mask_dir, "**", "*.png"), recursive=True)
    img_file = [elm.replace(backgroud_mask_dir, backgroud_image_dir).replace("_label", "")[:-4] + ".*" for elm in
                mask_file]
    fake_Drusen_list = glob("./data/Drusen/*.png")
    for i in range(len(img_file)):
        img_file_i = glob(img_file[i])
        assert len(img_file_i) == 1, "multi img_file_i : %s" % img_file
        img_file_i = img_file_i[0]
        assert img_file_i.endswith("jpg") or img_file_i.endswith("png"), "img_file_i type error : %s" % img_file_i
        img_file[i] = img_file_i

    logger.info(f'Creating dataset with {len(mask_file)} examples')

    # api
    main()
    # test_api_1()
    # my_test_api_2()
    # my_test_api_3()
    # my_test_api_4()
    # generate_kuang_all()
    # my_test_api_5()
