import pandas as pd

import csv
import os
import random


def data_split(full_list, ratio, shuffle=True):
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


def write_rp_csv():
    source_list = ['private']

    root = '/data1/wangjingtao/workplace/python/data/rp_data'
    with open('train_data.csv', 'w', newline='') as traincsvfile, open('test_data.csv', 'w',
                                                                            newline='') as testcsvfile, open(
            'val_data.csv', 'w', newline='') as valcsvfile:
        fields = ['ID', 'Eye','Image_path', 'Label_path']
        traincsvwriter = csv.writer(traincsvfile, delimiter=',')
        traincsvwriter.writerow(fields)
        testcsvwriter = csv.writer(testcsvfile, delimiter=',')
        testcsvwriter.writerow(fields)
        valcsvwriter = csv.writer(valcsvfile, delimiter=',')
        valcsvwriter.writerow(fields)

        global_idx = 0
        for _, each_set in enumerate(source_list):

            images_path = []
            # for image in os.listdir(root + '/' + each_set + '/' + 'y') and os.listdir(
            #         root + '/' + each_set + '/' + 'x'):
            image_list =  os.listdir(root + '/' + each_set + '/' + 'y') and os.listdir(
                    root + '/' + each_set + '/' + 'x')
            image_list.sort(key = lambda element: int(element.replace('.png','').split('_')[0]))
     
            for image in image_list:
                # label_path = root + '/' + each_set + '/' + 'y' + '/' + image.replace('.jpg', '.png')
                images_path.append(root + '/' + each_set + '/' + 'x' + '/' + image)

            train_dataset, test_valdaset = data_split(images_path, ratio=0.6, shuffle=False)
            val_dataset, test_dataset= data_split(test_valdaset, ratio=0.5, shuffle=False)

            for image_path in train_dataset:
                global_idx += 1
                eye_idx = image_path.split('/')[-1].split('_')[0]
                label_path = image_path.replace('/x/', '/y/')
                traincsvwriter.writerow([global_idx, eye_idx, image_path, label_path])
            for image_path in val_dataset:
                global_idx += 1
                eye_idx = image_path.split('/')[-1].split('_')[0]
                label_path = image_path.replace('/x/', '/y/')
                valcsvwriter.writerow([global_idx, eye_idx, image_path, label_path])

            for image_path in test_dataset:
                global_idx += 1
                eye_idx = image_path.split('/')[-1].split('_')[0]
                label_path = image_path.replace('/x/', '/y/')
                testcsvwriter.writerow([global_idx, eye_idx, image_path, label_path])


if __name__ == '__main__':
    random.seed(1)
    write_rp_csv()
