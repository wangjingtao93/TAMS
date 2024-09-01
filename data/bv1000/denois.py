import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')

import pandas as pd
import cv2
import numpy as np
import os
import glob
import pandas as pd
import common.utils as comm
def denoise(tempImg, brightness=25):  # brightness取值 0-50
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


project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation'
# image_ls = glob.glob(os.path.join('/data1/wangjingtao/workplace/python/data/meta-oct/seg/CNV/x/' + '/*.*'))
systhsis_df = pd.read_csv(os.path.join(project_path, 'data/bv1000_ocv_cnv/meta_data/synthetic_data.csv'))
image_ls = systhsis_df['Image_path']
for i in image_ls:
    img_arr = cv2.imread(i, 0)
    comm.mkdir(i.replace(i.split('/')[-1], '').replace('/x', '/x_d'))
    cv2.imwrite(i.replace('x', 'x_d'), denoise(img_arr))