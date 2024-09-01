import os
import pandas as pd
from sklearn.model_selection import GroupKFold



# 用balance 之后的
relative_path= '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation'

def creat_multi_tasks():
    store_path = os.path.join(relative_path, 'data/bv1000_ocv_cnv/real_data/cnv/acc_eyes/linux/multi_tasks')
    os.mkdir(store_path)
    df_train = pd.read_csv(os.path.join(relative_path,'data/bv1000_ocv_cnv/real_data/cnv/acc_eyes/linux/train.csv'))
    df_val = pd.read_csv(os.path.join(relative_path,'data/bv1000_ocv_cnv/real_data/cnv/acc_eyes/linux/val.csv'))

    new_df = pd.DataFrame()
    new_df = pd.concat([df_train, df_val],ignore_index=True)

    groups = new_df['Eye']  # 作为GroupKFold的groups参数

    k_fold_num = 4
    kf = GroupKFold(n_splits=k_fold_num)
    k_fold_count = 0

    for fold, (train_index, val_index) in enumerate(kf.split(X=new_df, y=None, groups=groups), 1):
        print('\n{} of kfold {}'.format(k_fold_count,kf.n_splits))
        
        S_df = new_df.iloc[train_index]
        Q_df = new_df.iloc[val_index]

        S_df.to_csv(os.path.join(store_path, f't_{k_fold_count}_s.csv'), index=False)
        Q_df.to_csv(os.path.join(store_path, f't_{k_fold_count}_q.csv'),index=False)

        k_fold_count += 1
# end-------------------- use code/process to splide
        
    

# 测试 S和Q是否有重复的eye
def test_s_q():
    df_s = pd.read_csv(os.path.join(relative_path,'data/bv1000_ocv_cnv/real_data/cnv/acc_eyes/linux/multi_tasks/t_0_s.csv'))
    df_q = pd.read_csv(os.path.join(relative_path,'data/bv1000_ocv_cnv/real_data/cnv/acc_eyes/linux/multi_tasks/t_0_q.csv'))

    eye_ls_s = df_s['Eye'].unique()
    eye_ls_q = df_q['Eye'].unique()

    for eye in eye_ls_s:
        if eye in eye_ls_q:
            print('nonono', eye)
        else:
            print('yyyyy', eye)
        


# creat_multi_tasks()           
test_s_q()
