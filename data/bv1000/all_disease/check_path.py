import pandas as pd
import os

dframe = pd.read_csv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/data/bv1000/all_disease/train_data.csv')

imgs = dframe['Image_path'].tolist()
masks = dframe['Label_path'].tolist()

for i, img in enumerate(imgs):
    if not os.path.exists(img):
        print('wrong: ', img)
        exit(-1)
    if not os.path.exists(masks[i]):
        print('wrong: ', masks[i])
        exit(-1)