# 图片像素换颜色+裁剪区域保存
from pylab import *
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import glob
import math
import os
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve, auc


# gts_path = 'data_1/analy/gt/'
gts_path = 'com'
# ress_path = 'data_1/result'
ress_path = 'com'
save_res_path = 'save/'
gt_list = os.listdir(ress_path)
print(gt_list)
ac = Io = Di = rec = pre = spe = F = 0

for name in gt_list:
    res_name = 'D:/BaiduSyncdisk/GitHub/imageProcess/paper/writePaper/my_first_meta_learning/pic/qualltative/cnv/fold/19_label_maml.png'

    # res_name = os.path.join(ress_path, name)
    # gts_name = os.path.join(gts_path, name.split('_se')[0] + '.jpg')
    # print(gts_name)
    # print(res_name)
    # res_name = os.path.join(ress_path, name.split('.')[0]+'_res.jpg')
    save_see_path = os.path.join(save_res_path, name.split('.')[0] + '_res.jpg')
    name_num = name.split('.')[0]   # 取.前面部分名字

    # res = cv2.imread(res_name)
    # gt = cv2.imread(gts_name)

    mask = cv2.imread(res_name)
    gt = mask[:,:512]
    res = mask[:,512:]

    # print(gt.shape)
    # print(res_name)

    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # res = res.reshape(1, res.shape[0], res.shape[1])
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    # res = cv2.resize(res, (3100, 2848), interpolation=cv2.INTER_AREA)
    # gt = gt.reshape(1, gt.shape[0], gt.shape[1])
    # print(res.shape[0])
    # cv2.imwrite(save_see_path, res)  # 保存修改像素点后的图片


    w, h = res.shape[0], res.shape[1]
    result = np.zeros((w, h, 3))
    res = (res >= 100)
    gt = (gt >= 100)

    TP = res * gt
    FP = res * (1 - gt)
    FN = (1 - res) * gt
    TN = (1 - res) * (1 - gt)

    # FN
    result[:, :, 0] = np.where(FN == 1, 0, result[:, :, 0])
    result[:, :, 1] = np.where(FN == 1, 0, result[:, :, 1])
    result[:, :, 2] = np.where(FN == 1, 255, result[:, :, 2])

    # FP
    result[:, :, 0] = np.where(FP == 1, 0, result[:, :, 0])
    result[:, :, 1] = np.where(FP == 1, 255, result[:, :, 1])
    result[:, :, 2] = np.where(FP == 1, 0, result[:, :, 2])

    # TP
    result[:, :, 0] = np.where(TP == 1, 255, result[:, :, 0])
    result[:, :, 1] = np.where(TP == 1, 255, result[:, :, 1])
    result[:, :, 2] = np.where(TP == 1, 255, result[:, :, 2])

    # TN
    result[:, :, 0] = np.where(TN == 1, 0, result[:, :, 0])
    result[:, :, 1] = np.where(TN == 1, 0, result[:, :, 1])
    result[:, :, 2] = np.where(TN == 1, 0, result[:, :, 2])


    cv2.imwrite(save_see_path,result)  # 保存修改像素点后的图片
