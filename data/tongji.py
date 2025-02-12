import pandas as pd

dframe = pd.read_csv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/data/hsyk_fundus_rp/real_data/test.csv')


eye_ls = dframe['Eye'].unique()

print(len(dframe))
print(len(eye_ls))
