import pandas as pd
from pathlib import Path,PurePosixPath
import csv
import os
import random

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

if __name__ == '__main__':
    csv_dir = 'D:/workplace/python/metaLearning/meta-learning-seg/DLCNV/data/win/train_data.csv'
    dataframe = pd.read_csv(csv_dir)

    percent = 0.8

    img_file = dataframe["Image_path"]
    mask_file = dataframe["Label_path"]
    id = dataframe['ID']
    eye = dataframe['Eye']
    img_file_list, mask_file_list, id_list, eye_list = list(img_file), list(mask_file), list(id), list(eye)

    prefix = 'D:/workplace/python/metaLearning/meta-learning-seg/'
    store_path = Path(prefix, 'meta_data')

    faker_data_prefix = 'D:/workplace/python/data/meta-oct/seg/faker_CNV_20230315/'

    with open(Path(store_path, 'train_data.csv'), 'w', newline='') as traincsvfile, open(Path(store_path, 'test_data.csv'), 'w', newline='') as testcsvfile:
        fields = ['ID', 'Eye', 'Image_path', 'Label_path']
        traincsvwriter = csv.writer(traincsvfile, delimiter=',')
        traincsvwriter.writerow(fields)
        testcsvwriter = csv.writer(testcsvfile, delimiter=',')
        testcsvwriter.writerow(fields)

        train_img_file_list, test_img_file_list = data_split(img_file_list, ratio=0.8, shuffle=False)
        train_mask_file_list, test_mask_file_list =data_split(mask_file_list, ratio=0.8, shuffle=False)
        train_id_list, test_id_list = data_split(id_list, ratio=0.8, shuffle=False)
        train_eye_list, test_eye_list = data_split(eye_list, ratio=0.8, shuffle=False)


        for i in range(len(train_img_file_list)):
            fold_name = str(train_id_list[i]) + '_' + train_mask_file_list[i].split('/')[-1].replace('.png', '')
            faker_data_path = Path(faker_data_prefix,fold_name)
            faker_image_list = os.listdir(Path(faker_data_path, 'x'))
            faker_image_list.sort(key=lambda element: int(element.split('_')[-1].replace('.jpg', '')))

            for image in faker_image_list:
                image_path = PurePosixPath(faker_data_path, 'x', image)
                mask_path = PurePosixPath(faker_data_path, 'y', image.replace('.jpg', '.png'))

                traincsvwriter.writerow([train_id_list[i], train_eye_list[i], image_path, mask_path])

        for i in range(len(test_img_file_list)):
            fold_name = str(test_id_list[i]) + '_' + test_mask_file_list[i].split('/')[-1].replace('.png', '')
            faker_data_path = Path(faker_data_prefix,fold_name)
            faker_image_list = os.listdir(Path(faker_data_path, 'x'))
            faker_image_list.sort(key=lambda element: int(element.split('_')[-1].replace('.jpg', '')))

            for image in faker_image_list:
                image_path = PurePosixPath(faker_data_path,'x', image)
                mask_path = PurePosixPath(faker_data_path, 'y', image.replace('.jpg', '.png'))
                testcsvwriter.writerow([test_id_list[i], test_eye_list[i], image_path, mask_path])
