import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')

import common.utils as utils

from sklearn.model_selection import KFold, GroupKFold
import pandas as pd


utils.set_seed(1)
k_fold_num = 5
# kf = KFold(n_splits=k_fold_num, random_state=None) # 4折
kf = GroupKFold(n_splits=k_fold_num)

train_dframe = pd.read_csv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/data/bv1000_ocv_cnv/real_data/cnv/acc_eyes/linux/train.csv')
val_dframe = pd.read_csv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/data/bv1000_ocv_cnv/real_data/cnv/acc_eyes/linux/val.csv')
dframe=pd.concat([train_dframe, val_dframe], ignore_index=True)

k_fold_count = 0
# 是否随机有待验证，不能随机
# dframe = dframe.sample(int(dframe.shape[0] * args.use_trainset_percent))
groups = list(dframe['Eye'])
# for train_index, test_index in kf.split(dframe):
eye_val=[]
eye_test =[]
for train_index, test_index in kf.split(dframe,groups=groups):
    k_fold_count += 1
    if k_fold_count == 5:
        train_df = dframe.iloc[train_index,:]
        val_df = dframe.iloc[test_index,:]
            
        # val_df.to_csv('val.csv',index=False)
        eye_val = list(val_df['Eye'])
    if k_fold_count == 4:
        train_df = dframe.iloc[train_index,:]
        test_df = dframe.iloc[test_index,:]
            
        # test_df.to_csv('test.csv',index=False)
        eye_test = list(test_df['Eye'])
  

        
train_df = dframe

for i in eye_val:
    train_df = train_df[train_df['Eye'] != i]
    
for i in eye_test:
    train_df = train_df[train_df['Eye'] != i]
# train_df.to_csv('train.csv',index=False)   


df_finanl = pd.concat([train_df, test_df, val_df], ignore_index=True).sort_values("Eye")

l_1 = list(dframe['Label_path'])
l_2 = list(df_finanl['Label_path'])

df_finanl.to_csv('tem.csv', index=False)

if l_1 == l_2:
    print("nihao")

