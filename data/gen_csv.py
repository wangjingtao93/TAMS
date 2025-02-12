'''
合成数据生成csv文件
'''


import pandas as pd
from pathlib import Path,PurePosixPath
import csv
import os
import random
import glob

def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

# 合成数据成csv文件
def gen_oct_sys_csv():
    real_data_csvfile = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/data/bv1000/real_data/cnv/all_data.csv'
    df_realdata = pd.read_csv(real_data_csvfile)

    img_files = df_realdata["Image_path"].tolist()
    mask_files = df_realdata["Label_path"].tolist()
    ids = df_realdata['ID'].tolist()
    eyes = df_realdata['Eye'].tolist()

    relative_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/task_aug/result_sys/20240419/bv1000-oct-cnv/2024-04-26-10-52-23/imp_net/'

    store_csvfile = os.path.join(relative_path.replace('imp_net', ''), 'sys.csv')

    with open(store_csvfile, 'w', newline='') as csvfile:
        fields = ['ID', 'Eye', 'Image_path', 'Label_path']
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(fields)

        for i in range(len(img_files)):
            img_name = mask_files[i].split('/')[-1].replace('.png', '')
            fold_name = str(ids[i]) + '_' + img_name

            sys_img_path = os.path.join(relative_path,fold_name)

            sys_imgs = glob.glob(os.path.join(sys_img_path, 'x', '*'))

            sys_imgs.sort(key=lambda element: int(element.split('_')[-1].replace('.jpg', '')))

            for image_path in sys_imgs:
                mask_path = image_path.replace('x', 'y').replace('.jpg', '.png')

                csvwriter.writerow([ids[i], eyes[i], image_path.replace(relative_path, ''), mask_path.replace(relative_path, '')])


def gen_fundus_sys_csv():
    real_data_csvfile = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/data/rips_fundus_rp/real_data/all_data.csv'
    df_realdata = pd.read_csv(real_data_csvfile)

    img_files = df_realdata["Image_path"].tolist()
    mask_files = df_realdata["Label_path"].tolist()
    ids = df_realdata['ID'].tolist()
    eyes = df_realdata['Eye'].tolist()

    relative_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/task_aug/result_sys/20240419/rips/2024-04-26-15-15-51/imp_net/'

    store_csvfile = os.path.join(relative_path.replace('imp_net', ''), 'sys.csv')

    with open(store_csvfile, 'w', newline='') as csvfile:
        fields = ['ID', 'Eye', 'Image_path', 'Label_path']
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(fields)

        for i in range(len(img_files)):
            img_name = mask_files[i].split('/')[-1].replace('.png', '')
            fold_name = img_name # 这个地方和oct不一样

            sys_img_path = os.path.join(relative_path,fold_name)

            sys_imgs = glob.glob(os.path.join(sys_img_path, 'x', '*'))

            sys_imgs.sort(key=lambda element: int(element.split('_')[-1].replace('.jpg', '')))

            for image_path in sys_imgs:
                mask_path = image_path.replace('x', 'y').replace('.jpg', '.png')

                csvwriter.writerow([ids[i], eyes[i], image_path.replace(relative_path, ''), mask_path.replace(relative_path, '')])

        check_path(store_csvfile, relative_path)
    
def check_path(csvfile, relative_path):
    df_realdata = pd.read_csv(csvfile)

    img_files = df_realdata["Image_path"].tolist()
    mask_files = df_realdata["Label_path"].tolist()

    for i, img in enumerate(img_files):
        if not os.path.exists(os.path.join(relative_path,img)):
            print('no files: ', os.path.join(relative_path,img))
            exit(-1)
        
        if not os.path.exists(os.path.join(relative_path,mask_files[i])):
            print('no files: ', os.path.join(relative_path,img))
            exit(-1)
        



if __name__ =='__main__':
    # gen_oct_sys_csv()
    gen_fundus_sys_csv()




    
    


