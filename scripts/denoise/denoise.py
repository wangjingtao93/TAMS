
import cv2
import numpy as np
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')

import pandas as pd
import cv2
import numpy as np
import os
import glob
import pandas as pd
import common.utils as comm

def denoise(tempImg, brightness=50):  # brightness取值 0-50
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

i = '2.png'
img_arr = cv2.imread(i, 0)

cv2.imwrite(i.replace('.png', '_d.jpg'), denoise(img_arr))