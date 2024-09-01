import pandas as pd
from pathlib import Path,PurePosixPath
import csv
import os
import random
import glob

project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation'
store_path = os.path.join(project_path, 'data/drive_eye/')

with open(Path(store_path, 'data.csv'), 'w', newline='') as csvfile:
    fields = ['ID', 'Image_path', 'Label_path']
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(fields)


    img_name_ls_train = os.listdir("/data1/wangjingtao/workplace/python/data/rp_data/drive_eye/training/zhuanhuan/x/")
    img_name_ls_test = os.listdir("/data1/wangjingtao/workplace/python/data/rp_data/drive_eye/test/zhuanhuan/x/")
    img_name_ls = img_name_ls_train + img_name_ls_test

    img_name_ls.sort(key=lambda element: int(element.split('_')[0]))

    for img_name in img_name_ls:
        ID = int(img_name.split('_')[0])
        csvwriter.writerow([ID, img_name, img_name.replace('.jpg', '.png')])


