
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/rp-project')

import pandas as pd

import csv
import os
import random

import utils

df = pd.read_csv('/data1/wangjingtao/workplace/python/pycharm_remote/rp-project/DL/data_private/linux/all_data.csv')

id_l = list(df['ID'])

df_1 = df[df['ID'] <101]


indice = range(0, len(id_l) - 20)
train_indice,val_indice = utils.data_split(indice, 0.6)
val_indice, test_indice = utils.data_split(val_indice, 0.5)

train_df = df.iloc[train_indice]
val_df = df.iloc[val_indice]
test_df = df.iloc[test_indice]

indice_add = range(len(id_l) - 20,  len(id_l))
train_indice_add, val_indice_add = utils.data_split(indice_add, 0.6)
val_indice_add, test_indice_add = utils.data_split(val_indice_add, 0.5)

train_df = pd.concat([train_df, df.iloc[train_indice_add]])
val_df = pd.concat([val_df, df.iloc[val_indice_add]])
test_df = pd.concat([test_df, df.iloc[test_indice_add]])

path_script= os.path.split( os.path.realpath( sys.argv[0] ) )[0]
train_df.to_csv(os.path.join(path_script, 'train.csv') ,index=False)
val_df.to_csv(os.path.join(path_script, 'val.csv'), index=False)
test_df.to_csv(os.path.join(path_script, 'test.csv'), index=False)