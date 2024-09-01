import pandas as pd

import csv
import os
import random
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/rp-project')
import os
os.chdir('/data1/wangjingtao/workplace/python/pycharm_remote/rp-project/data_synthesis/forge')


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


def write_forge_csv():
    root = '/data1/wangjingtao/workplace/python/pycharm_remote/rp-project/data_synthesis/forge/results_chosen'
    source_list = []
    for root, dirs, files in os.walk(root, topdown=True):
        source_list = dirs
        break
    split_index = len(source_list) - 1
    with open('data/train_data.csv', 'w', newline='') as csvfile:
        fields = ['ID', 'Image_path', 'Label_path', 'Class']
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(fields)
        for idx, each_set in enumerate(source_list[:split_index]):
            for image in os.listdir(root + '/' + each_set + '/' + 'y') and os.listdir(
                    root + '/' + each_set + '/' + 'x'):
                label_path = root + '/' + each_set + '/' + 'y' + '/' + image
                image_path = root + '/' + each_set + '/' + 'x' + '/' + image
                csvwriter.writerow([idx, image_path, label_path, each_set])

            
    with open('data/val_data.csv', 'w', newline='') as csvfile:
        fields = ['ID', 'Image_path', 'Label_path', 'Class']
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(fields)
        for idx, each_set in enumerate(source_list[split_index:]):
            for image in os.listdir(root + '/' + each_set + '/' + 'y') and os.listdir(
                    root + '/' + each_set + '/' + 'x'):
                label_path = root + '/' + each_set + '/' + 'y' + '/' + image
                image_path = root + '/' + each_set + '/' + 'x' + '/' + image
                csvwriter.writerow([idx + split_index, image_path, label_path, each_set])







if __name__ == '__main__':
    random.seed(1)
    print('exec func write_forge_csv')
    write_forge_csv()
    


    