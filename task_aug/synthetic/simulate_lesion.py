# -*- coding: utf-8 -*-

import time

import numpy
import cv2
import os
from glob import glob
import logging
import random
import math
import numpy as np

import common.utils as utils
import common.logutil as logutil


logger = logging.getLogger(__name__)
logger = logutil.logs()

from enum import Enum

class FailReason(Enum):
    poor_quality = 1
    lack_layer = 2
    over_incline = 3
    others = 4



# for oct
class SimulateLesions():
    def __init__(self, args):
        self.args = args
        self._init_dir()

    def _init_dir(self):
        # backgroud_mask_dir = "./data/backgroud/detaction/y"
        # backgroud_image_dir = "./data/backgroud/detaction/x"
        self.backgroud_mask_dir = "D:/workplace/python/My_Pytorch_UNet/data/result/all/Layer5Line6/y"
        self.backgroud_image_dir = "D:/workplace/python/My_Pytorch_UNet/data/result/all/Layer5Line6/x"

        # self.output_x_fake_dir = os.path.join(self.args.project_path,"task_argu/result",self.args.datatype, "succ_res/x")
        # self.output_y_fake_dir = os.path.join(self.args.project_path,"task_argu/result",self.args.datatype,"succ_res/y")

        self.output_x_fake_dir = os.path.join(self.args.store_dir, "succ_res/x")
        self.output_y_fake_dir = os.path.join(self.args.store_dir, "succ_res/y")


        self.fail_image_dir = os.path.join(self.args.store_dir,"fail_image")
        self.fail_poor_quality_x_dir = os.path.join(self.fail_image_dir,"poor_quality/x")
        self.fail_poor_quality_y_dir = os.path.join(self.fail_image_dir, "poor_quality/y")
        self.lack_layer_x_dir = os.path.join(self.fail_image_dir,"lack_layer/x")
        self.lack_layer_y_dir = os.path.join(self.fail_image_dir, "lack_layer/y")
        self.over_incline_x_dir = os.path.join(self.fail_image_dir,"over_incline/x")
        self.over_incline_y_dir = os.path.join(self.fail_image_dir, "over_incline/y")
        self.fail_others_x_dir = os.path.join(self.fail_image_dir, "others/x")
        self.fail_others_y_dir = os.path.join(self.fail_image_dir, "others/y")

        self.clean_forward()
        self.process_prepare()

        #  optional:Obtain the location of all lesions on the real image to prevent fake lesions from overlapping with existing lesions and not meeting the real situation
        # 获取真实图像上，所有病灶的位置(可选)， 防止伪造病灶时，与已存在的病灶重叠，不符合真实情况
        self.label_dict = read_labelTxt(os.path.join(self.args.project_path, self.args.label_text_path))

    def process_prepare(self):
        my_mkdir(self.output_x_fake_dir)
        my_mkdir(self.output_y_fake_dir)

        my_mkdir(self.fail_poor_quality_x_dir)
        my_mkdir(self.fail_poor_quality_y_dir)

        my_mkdir(self.lack_layer_x_dir)
        my_mkdir(self.lack_layer_y_dir)

        my_mkdir(self.over_incline_x_dir)
        my_mkdir(self.over_incline_y_dir)

        my_mkdir(self.fail_others_x_dir)
        my_mkdir(self.fail_others_y_dir)


    def clean_forward(self, dir_list=[]):
        dir_list.append(self.output_x_fake_dir)
        dir_list.append(self.output_y_fake_dir)
        dir_list.append(self.fail_image_dir)

        for top in dir_list:
            for root, dirs, files in os.walk(top, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))


    # focus can be simulated also, but there just use real focus 
    def sim_use_real_focus(self, x_image_path, y_image_path, mask_file_list=None, bingzao_data=None):
        out_list=[]#返回 final return, output

        image_name = x_image_path.split("/")[-1]
        
        x_raw = cv2.imread(x_image_path, 0)
        y_raw = (cv2.imread(y_image_path, 0) / 8).astype(np.uint8)

        dim_raw = (x_raw.shape[1], x_raw.shape[0])
        setep = 1
        if x_raw.shape[1] != 1024:
            setep = 2

        dim = (1024, 1024)
        x_fake = cv2.resize(x_raw, dim, interpolation=cv2.INTER_NEAREST)
        y_fake = cv2.resize(y_raw, dim, interpolation=cv2.INTER_NEAREST)
        x_raw_resize = x_fake.copy()
        y_raw_resize = y_fake.copy()

        # layer position
        mask_24 = np.where(y_fake == 24, 1, 0)
        mask_23 = np.where(y_fake == 23, 1, 0)
        mask_22 = np.where(y_fake == 22, 1, 0)
        mask_21 = np.where(y_fake == 21, 1, 0)

        # check whether layer info  is correct, sometimes will missing layer information
        if np.sum(mask_24) == 0 or np.sum(mask_23) == 0 or np.sum(mask_22) == 0 or np.sum(mask_21) == 0:
            logger.warning(f'{y_image_path} 缺少层信息, 无法伪造CNV')
            self.save_image(False, x_raw, y_raw, x_image_path, y_image_path, FailReason.lack_layer)
            return out_list
        

        # computer suitable position for simulated begin ++++++++++++++++++++++
        # 计算适合伪造病灶的位置
        if mask_file_list != None:
            cnv_file_path = random.choice(mask_file_list)

            ######tmp别忘记改回来
            # real_cnv = cv2.imread(cnv_file_path, 0) / 4
            ######
        else:
            real_cnv = bingzao_data

        # real_cnv = cv2.imread(cnv_file_path, 0)
        cnv_dim = (real_cnv.shape[1], real_cnv.shape[0])
        width = cnv_dim[0] * setep
        
        idx_begin = 200
        idx_end = 800
        if width > 550: #
            real_cnv = cv2.resize(real_cnv, (550, cnv_dim[1]), interpolation=cv2.INTER_NEAREST)
            cnv_dim = (real_cnv.shape[1], real_cnv.shape[0])
            width = cnv_dim[0]

        idx0_list, idx1_list, fail_reason = self.random_CNV_idx(idx_begin, idx_end, [width], y_fake, image_name, setep = setep, bingzao_num=1)
        if len(idx0_list) == 0:
            logger.error(f'{y_image_path} 未随机到合适的位置, 无法伪造CNV')
            self.save_image(False, x_raw, y_raw, x_image_path, y_image_path, fail_reason)
            return out_list
        
        idx0 = idx0_list[0]
        idx1 = idx1_list[0]
        # computer suitable position for simulated end -------------------------------

        
        temp = y_fake[1:] - y_fake[:-1]
        temp[:, :idx0] = 0
        temp[:, idx1:] = 0
        # obtain layer line
        temp_24 = (temp > 0) * (y_fake[1:] == 24)
        temp_22 = (temp > 0) * (y_fake[1:] == 22)
        temp_23 = (temp > 0) * (y_fake[1:] == 23)
        temp_21 = (temp > 0) * (y_fake[1:] == 21)
        temp_28 = (temp > 0) * (y_fake[1:] == 28)

        idx_up_24 = numpy.where(temp_24.transpose())
        idx_up_22 = numpy.where(temp_22.transpose())
        idx_up_23 = numpy.where(temp_23.transpose())
        idx_up_21 = numpy.where(temp_21.transpose())
        idx_up_28 = numpy.where(temp_28.transpose())


        # begin++++++++++++++++++++++++++Recalculate the pixel values
        width_iter = min(len(idx_up_24[0]), len(idx_up_22[0]), len(idx_up_23[0]), len(idx_up_21[0]))
        if len(idx_up_28[1]) < width_iter:
            idx_up_28 = list(idx_up_28)
            idx_up_28[1] = np.append(idx_up_28[1], idx_up_21[1][len(idx_up_28[1]): width_iter])
            idx_up_28[0] = np.append(idx_up_28[0], idx_up_21[0][len(idx_up_28[1]): width_iter])
            idx_up_28 = tuple(idx_up_28)

        # calculate simulated foucus height
        temp_height_list_24_23 = []
        temp_height_list_23_22 = []
        temp_height_list_22_21 = []
        temp_height_list_21_28 = []
        for i in range(width_iter):
            temp_height_list_24_23.append(idx_up_24[1][i] - idx_up_23[1][i])
            temp_height_list_23_22.append(idx_up_23[1][i] - idx_up_22[1][i])
            temp_height_list_22_21.append(idx_up_22[1][i] - idx_up_21[1][i])
            temp_height_list_21_28.append(idx_up_21[1][i] - idx_up_28[1][i])

        height_average_24_23 = int(np.mean(temp_height_list_24_23))
        height_average_23_22 = int(np.mean(temp_height_list_23_22))
        height_average_22_21 = int(np.mean(temp_height_list_22_21))
        height_average_21_28 = int(np.mean(temp_height_list_21_28))
        rand_idx_offset = random.randint(4, 7)
        height_average_sum = height_average_24_23 + height_average_23_22 + height_average_22_21 + rand_idx_offset

        height_all = height_average_24_23 + height_average_23_22 + height_average_22_21 + height_average_21_28

        probility_cnv_height_sum = random.random()
        if height_average_sum > 70:
            if probility_cnv_height_sum < 0.6:
                cnv_height_sum = 45 + random.randint(10, 25)

            else:
                cnv_height_sum = 45 + random.randint(25, 40)
        elif height_average_sum < 40:
            logger.error(f'{y_image_path} height_average_sum 太薄, 无法伪造CNV')
            self.save_image(False, x_raw, y_raw, x_image_path, y_image_path, FailReason.others)
            return out_list
        else:
            cnv_height_sum = random.randint(40, height_average_sum)

        scope_idx0_idx1 = idx1 - idx0
        cnv_height_24_23 = height_average_24_23
        # kua_du =  int(scope_idx0_idx1 * random.uniform(0.04, 0.1))
        # if scope_idx0_idx1 - kua_du < 26 * setep:
        #     logger.error(f'{y_image_path} scope_idx0_idx1太窄, 无法伪造CNV')
        #     save_image(False, x_raw, y_raw, x_image_path, y_image_path, FailReason.others)
        #     return out_list
        
        try:
        # resize the focus contour again
        # need improve
            if  scope_idx0_idx1 < 60 * setep and scope_idx0_idx1 > 5 *setep:           
                faker_CNV_24_23 = cv2.resize(real_cnv, (scope_idx0_idx1 - 3, cnv_height_24_23), cv2.INTER_NEAREST)
                width_cha = 2 # 什么作用
            elif  scope_idx0_idx1 <= 5 *setep:
                faker_CNV_24_23 = cv2.resize(real_cnv, (scope_idx0_idx1 + random.randint(3, 15), cnv_height_24_23), cv2.INTER_NEAREST)
                width_cha = 2
            else:
                width_cha = int(scope_idx0_idx1 * random.uniform(0.04, 0.1))
                faker_CNV_24_23 = cv2.resize(real_cnv, (scope_idx0_idx1 - int((width_cha + random.uniform(0.2, 0.8))), cnv_height_24_23), cv2.INTER_NEAREST)
            hs_24_23 = numpy.sum(faker_CNV_24_23 == 40, 0)

            front_cnv_sum = cv2.resize(real_cnv, (idx1 - idx0 - 2 * width_cha, cnv_height_sum), cv2.INTER_NEAREST)
            hs_sum = numpy.sum(front_cnv_sum == 40, 0)

            front_cnv_all = cv2.resize(real_cnv, (idx1 - idx0 - 2 * width_cha, height_all + cnv_height_sum), cv2.INTER_NEAREST)
            hs_all = numpy.sum(front_cnv_all == 40, 0)

            height_width = random.randint(3, 10)
            front_cnv_sum_bigger = cv2.resize(real_cnv, (idx1 - idx0, cnv_height_sum + height_width), cv2.INTER_NEAREST)
            hs_sum_bigger = numpy.sum(front_cnv_sum_bigger == 40, 0)

            front_cnv_all_bigger = cv2.resize(real_cnv, (idx1 - idx0, height_all + cnv_height_sum + height_width), cv2.INTER_NEAREST)
            hs_all_bigger = numpy.sum(front_cnv_all_bigger == 40, 0)
        except Exception:
            logger.error(f'{y_image_path} something wrong')
            self.save_image(False, x_raw, y_raw, x_image_path, y_image_path, FailReason.others)
            return out_list

        try:
            for i in range(width_cha, width_iter - width_cha):
                # ##### the six th version 第六版begin 

                idx_all = idx_up_24[1][i]

                while idx_all > idx_up_24[1][i] - hs_sum[i - width_cha]:
                    distance = int((idx_all - idx_up_24[1][i] + hs_all[i - width_cha]) / (hs_all[i - width_cha] + 1) * hs_sum[
                        i - width_cha])

                    if idx_all + distance > 1023:
                        logger.error(f'{y_image_path} idx_all + distance > 1023, 无法伪造CNV')
                        self.save_image(False, x_raw, y_raw, x_image_path, y_image_path, FailReason.others)
                        return out_list

                    x_fake[idx_all, idx_up_24[0][i]] = x_raw_resize[idx_all + distance, idx_up_24[0][i]]

                    if distance != 0:
                        y_fake[idx_all, idx_up_24[0][i]] = 5

                    idx_all -= 1

                idx_24_1 = idx_up_24[1][i] + random.randint(3, 10)
                while idx_24_1 > idx_up_24[1][i] - random.randint(3, 10):
                    try:
                        distance = hs_24_23[i - width_cha]
                    except:
                        print('hellpo')
                    if idx_24_1 + distance > 1023:
                        logger.error(f'{y_image_path} idx_24_1 + distance > 1023, 无法伪造CNV')
                        self.save_image(False, x_raw, y_raw, x_image_path, y_image_path, FailReason.others)
                        return out_list

                    x_fake[idx_24_1, idx_up_24[0][i]] = x_raw_resize[idx_24_1 + distance, idx_up_24[0][i]]

                    idx_24_1 -= 1

            for i in range(width_iter):
                idx_21 = idx_up_24[1][i]
                while idx_21 > idx_up_24[1][i] - hs_sum_bigger[i]:
                    if y_fake[idx_21, idx_up_24[0][i]] != 5:
                        distance = int(
                            (idx_21 - idx_up_24[1][i] + hs_all_bigger[i]) / (hs_all_bigger[i] + 10) * hs_sum_bigger[i])

                        if idx_21 + distance > 1023:
                            logger.error(f'{y_image_path} idx_21 + distance > 1023, 无法伪造CNV')
                            self.save_image(False, x_raw, y_raw, x_image_path, y_image_path, FailReason.others)
                            return out_list

                        y_fake[idx_21, idx_up_24[0][i]] = y_raw_resize[idx_21 + distance, idx_up_24[0][i]]

                        # x_fake[idx_21, idx_up_24[0][i]] = x_raw_resize[idx_21 + distance, idx_up_24[0][i]]

                        if y_fake[idx_21, idx_up_24[0][i]] == 24:
                            y_fake[idx_21, idx_up_24[0][i]] = 5

                    idx_21 -= 1
            
            # end---------------------------Recalculate the pixel values
        except Exception:
            logger.error(f'{y_image_path} something wrong')
            self.save_image(False, x_raw, y_raw, x_image_path, y_image_path, FailReason.others)
            return out_list
        
        x_fake = cv2.resize(x_fake, dim_raw, interpolation=cv2.INTER_NEAREST)
        y_fake = cv2.resize(y_fake, dim_raw, interpolation=cv2.INTER_NEAREST)
        self.save_image(True, x_fake, y_fake * 8, x_image_path, y_image_path)
        out_list.append(x_fake)
        out_list.append(y_fake * 8)
        return out_list
        
    def random_CNV_idx(self, idx_begin, idx_end, weight_idx, mask, image_name, bingzao_num=1, setep = 1):
        idx_scope = idx_end - idx_begin
        step = idx_scope // bingzao_num
        idx0_list = []
        idx1_list = []
        fail_reason = FailReason.others
        for bingzao_idx in range(bingzao_num):
            cycle_count = 0
            all_line = mask[1:] - mask[:-1]
            bingzao_idx_begin = idx_begin + step * bingzao_idx

            while cycle_count < 50:
                cycle_count += 1
                all_line_copy = all_line.copy()

                idx0 = random.randint(bingzao_idx_begin, bingzao_idx_begin + step)

                idx1 = idx0 + weight_idx[bingzao_idx]

                if idx1 > bingzao_idx_begin + step:
                    distance = idx1 - (bingzao_idx_begin + step)
                    idx0 = idx0 - distance
                    idx1 = idx1 - distance

                all_line_copy[:, :idx0] = 0
                all_line_copy[:, idx1:] = 0

                temp_24 = (all_line_copy > 0) * (mask[1:] == 24)
                temp_22 = (all_line_copy > 0) * (mask[1:] == 22)
                temp_23 = (all_line_copy > 0) * (mask[1:] == 23)
                temp_21 = (all_line_copy > 0) * (mask[1:] == 21)

                idx_up_24 = numpy.where(temp_24.transpose())
                idx_up_22 = numpy.where(temp_22.transpose())
                idx_up_23 = numpy.where(temp_23.transpose())
                idx_up_21 = numpy.where(temp_21.transpose())
                width_iter = min(len(idx_up_24[0]), len(idx_up_22[0]), len(idx_up_23[0]), len(idx_up_21[0]))

                if (temp_24 == 0).all() or (temp_22 == 0).all() or (temp_23 == 0).all() or (
                        temp_21 == 0).all() or width_iter < idx1 - idx0 - 4 or max(idx_up_24[1]) + 50 > 1024:
                    fail_reason = FailReason.others
                    logger.warning(
                        f"在图片{image_name}的{bingzao_idx_begin}~{bingzao_idx_begin + step} 随机的位置 {idx0} {idx1} 无法定位脉络膜，尝试随机另一个位置")
                    continue
                if max(idx_up_24[1]) + 50 > 1023:
                    fail_reason = FailReason.others
                    logger.warning(f"图片{image_name} 的位置 {idx0} {idx1} 高度+50 超出1024，尝试随机另一个位置")
                    continue

                # 对于扣下来的真实的CNV没必要对宽度限制
                # for real focus(not simulated) from real cnv data, need not process width
                # if width_iter < 50 * setep:
                #     fail_reason = FailReason.others
                #     logger.warning(
                #         f"在图片{image_name}的{bingzao_idx_begin}~{bingzao_idx_begin + step} 随机的位置 {idx0} {idx1} winder={width_iter} 宽度太窄，尝试随机另一个位置")
                #     continue

                if width_iter == len(idx_up_24[0]):
                    min_len_temp = temp_24
                    min_len_idx_up = idx_up_24
                elif width_iter == len(idx_up_23[0]):
                    min_len_temp = temp_23
                    min_len_idx_up = idx_up_23
                elif width_iter == len(idx_up_22[0]):
                    min_len_temp = temp_22
                    min_len_idx_up = idx_up_22
                else:
                    min_len_temp = temp_21
                    min_len_idx_up = idx_up_21

                # begin+++++++++++++check the Angle of choroid tilt, whether is suitable for simulating
                y1 = min_len_idx_up[0][0]
                x1 = min_len_idx_up[1][0]
                y2 = min_len_idx_up[0][-1]
                x2 = min_len_idx_up[1][-1]
                angle = self.azimuthAngle(x1, y1, x2, y2)
                # box_cnv_min = [idx0 - 10, min(min_len_idx_up[1]), idx1 + 10, max(min_len_idx_up[1])]

                # if angle > 40.0 and angle < 140:
                #     logger.warning(f"图片{image_name} 的位置 {idx0} {idx1} 倾斜角度太大，尝试随机另一个位置")
                #     fail_reason = FailReason.over_incline
                #     continue
                # elif angle > 220 and angle < 210:
                #     logger.warning(f"图片{image_name} 的位置 {idx0} {idx1} 倾斜角度太大，尝试随机另一个位置")
                #     fail_reason = FailReason.over_incline
                #     continue

                # if check_idx(box_cnv_min, image_name) is False:
                #     fail_reason = FailReason.others
                #     logger.warning(f"图片{image_name} 的位置 {idx0} {idx1} 有其他病灶，尝试随机另一个位置")
                #     continue

                # end ----------------check the Angle of choroid tilt, whether is suitable for simulating


                temp_height_list_24_23 = []
                temp_height_list_23_22 = []
                temp_height_list_22_21 = []
                for i in range(width_iter):
                    temp_height_list_24_23.append(idx_up_24[1][i] - idx_up_23[1][i])
                    temp_height_list_23_22.append(idx_up_23[1][i] - idx_up_22[1][i])
                    temp_height_list_22_21.append(idx_up_22[1][i] - idx_up_21[1][i])

                height_average_24_23 = np.mean(temp_height_list_24_23)
                height_average_23_22 = np.mean(temp_height_list_23_22)
                height_average_22_21 = np.mean(temp_height_list_22_21)
                height_average_sum = height_average_24_23 + height_average_23_22 + height_average_22_21 + 7

                if height_average_sum < 30:
                    fail_reason = FailReason.others
                    logger.warning(f"图片{image_name} 的位置 {idx0} {idx1} 层太薄，尝试随机另一个位置")
                    continue
                
                # Obtain the location of all lesions on the real image (optional) to prevent fake lesions from overlapping with existing lesions and not meeting the real situation
                box_cnv_min = [idx0 - 10, min(min_len_idx_up[1]) - height_average_sum, idx1 + 10,  max(min_len_idx_up[1]) + 5]
                if self.check_idx(box_cnv_min, image_name) is False:
                    fail_reason = FailReason.others
                    logger.warning(f"图片{image_name} 的位置 {idx0} {idx1} 有其他病灶，尝试随机另一个位置")
                    continue

                idx0_list.append(idx0)
                idx1_list.append(idx1)
                # if idx1_list[i] > tmp + step:
                #     distance = idx1_list[i] - (tmp + step)
                #     idx0_list[i] = idx0_list[i] - distance
                #     idx1_list[i] = idx1_list[i] - distance
                break

        return idx0_list, idx1_list, fail_reason
    
    # Calculate the Angle of choroid tilt
    def azimuthAngle(self, x1, y1, x2, y2):
        angle = 0.0
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

    def save_image(self, if_save_image, image, mask, x_image_path, y_image_path, reason=FailReason.others):
        if if_save_image:
            save_x_dir = self.output_x_fake_dir
            save_y_dir = self.output_y_fake_dir
        else:
            if reason == FailReason.poor_quality:
                save_x_dir = self.fail_poor_quality_x_dir
                save_y_dir = self.fail_poor_quality_y_dir
            elif reason == FailReason.lack_layer:
                save_x_dir = self.lack_layer_x_dir
                save_y_dir = self.lack_layer_y_dir
            elif reason == FailReason.over_incline:
                save_x_dir = self.over_incline_x_dir
                save_y_dir = self.over_incline_y_dir
            else:
                save_x_dir = self.fail_others_x_dir
                save_y_dir = self.fail_others_y_dir

        output_x_fake_image = os.path.join(save_x_dir, x_image_path.split("/")[-1])
        output_y_fake_image = os.path.join(save_y_dir, y_image_path.split("/")[-1])
        cv2.imwrite(output_x_fake_image, image)
        cv2.imwrite(output_y_fake_image, mask)
        if if_save_image is False:
            image_name = x_image_path.split("/")[-1]
            logging.error(f'Image {image_name} can not be created CNV, resion is {reason}')

    def check_idx(self,box_bolimoyou, image_name):
        if image_name not in self.label_dict:
            return True
        # resize_factor = 1024 / dim_raw[0]
        resize_factor = 1
        for info_list in self.label_dict[image_name]:
            box_info = []
            box_info.append(int(math.ceil(float(info_list[0]) * resize_factor)))
            box_info.append(int(info_list[1]))
            box_info.append(int(math.ceil(float(info_list[2]) * resize_factor)))
            box_info.append(int(info_list[3]))
            if overlap(box_bolimoyou, box_info):
                return False  # 矩形相交，此位置不行
        return True
    

def read_labelTxt(label_text_path):
    label_dict = {}
    if os.path.isfile(label_text_path):  # 先检查文件是否存在
        with open(label_text_path, 'r') as f:
            for info in f.readlines():
                info_list = info.replace("\n", "").split(" ")
                key_image_name = info_list[0]
                tmp_list = info_list[2:]
                value_list = [int(i) for i in tmp_list]

                # 图像从512*1024 变为1024 * 1024
                # value_list[0] = 2 * value_list[0]
                # value_list[2] = 2 * value_list[2]

                if key_image_name not in label_dict:
                    label_dict[key_image_name] = [value_list]
                else:
                    label_info = label_dict[key_image_name]
                    label_info.append(value_list)
                    label_dict[key_image_name] = label_info

    return label_dict    
    
def overlap(box1, box2):
    # check whether two rectangles intersect
    # 判断两个矩形是否相交,
    # 思路来源于:https://www.cnblogs.com/avril/archive/2013/04/01/11293875.html
    minx1, miny1, maxx1, maxy1 = box1
    minx2, miny2, maxx2, maxy2 = box2
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx > maxx or miny > maxy:
        return False # 不相交
    else:
        return True # 相交
    
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
    

            

