'''
将数据，按照多任务进行组织
区分不同的数据
Organize data according meta tasks
'''
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from torch.utils.data import DataLoader
from common.dataset.dataset_bv1000_cnv import BV_CNV_MetaDataset,BV1000_OCT_CNV, BV1000_OCT_CNV_Sys
from common.dataset.build_tasks import Task, DL_Task
from common.dataset.rips_organize import rips_organize, rips_organize_tl
from common.dataset.herp_organize import herp_organize, herp_organize_tl
from common.dataset.drive_eye_origanize import  drive_eye_organize 
from common.dataset.fs1000_origanize import  fs1000_organize 

def data_organize(args):
    if args.datatype == "bv1000-oct-cnv":
        if args.alg == 'pretrain':
            train_data_ls, val_data_ls, test_data_ls  = bv1000_oct_cnv_for_tl(args)
        else:
            train_data_ls, val_data_ls, test_data_ls = bv1000_oct_cnv(args)
    elif args.datatype == "rips":
        if args.alg == 'pretrain':
            train_data_ls, val_data_ls, test_data_ls  = rips_organize_tl(args)
        else:
            train_data_ls, val_data_ls, test_data_ls = rips_organize(args)

    elif args.datatype == "heshi-rp":
        if args.alg == 'pretrain':
            train_data_ls, val_data_ls, test_data_ls  = herp_organize_tl(args)
        else:
            train_data_ls, val_data_ls, test_data_ls = herp_organize(args)

    elif args.datatype == "drive-eye":
        train_data_ls, val_data_ls, test_data_ls = drive_eye_organize(args)
    elif 'fs1000' in args.datatype or 'polyp' in args.datatype:
        train_data_ls, val_data_ls, test_data_ls = fs1000_organize(args)
    
    else:
        exit('no data')

    return train_data_ls, val_data_ls, test_data_ls


def bv1000_oct_cnv(args):
    '''
    返回任务级别的数据，
    return tasks dataloader
    '''
    # synthetic_dframe 是所有的合成数据， synthetic_dframe synthetic data 
    synthetic_dframe = pd.read_csv(args.synthetic_data_csv)

    # real_dframe是所有的真实数据， real_dframe is all real data
    # real_dframe = pd.read_csv(os.path.join(args.project_path,'data/bv1000_ocv_cnv/real_data/all_data.csv'))
    real_dframe = pd.read_csv(os.path.join(args.project_path,args.real_data_csv))


    # not include synthetic data 不包括合成数据
    train_df_ori = pd.read_csv(os.path.join(args.project_path, args.train_csv, f't_{args.index_fold}_s.csv'))
    val_df_ori = pd.read_csv(os.path.join(args.project_path, args.val_csv, f't_{args.index_fold}_q.csv'))
    test_df_ori = pd.read_csv(os.path.join(args.project_path,args.test_csv))

    # 仅使用一定比例的数据如1或0.5
    # just use part of dataset for train and test
    if args.use_trainset_percent != 1.0 :
        train_df_ori = train_df_ori.sample(frac=args.use_trainset_percent,random_state=args.random_state)
        val_df_ori = val_df_ori.sample(frac=args.use_trainset_percent,random_state=args.random_state)
        # test_df_ori = test_df_ori.sample(frac=args.use_trainset_percent,random_state=args.random_state)

    # synthetic_dframe 是所有数据，包括合成数据， synthetic_dframe is all data, include synthetic data 
    # 根据原始的数据，获取选取的合成数据的dataframe, obtain choosed synthetic dataframe according origin data ID， include synthetic data
    meta_trainframe = synthetic_dframe[synthetic_dframe['ID'].isin(list(train_df_ori['ID']))]
    meta_valframe = synthetic_dframe[synthetic_dframe['ID'].isin(list(val_df_ori['ID']))]


    all_meta_train_classes = list(meta_trainframe["ID"].unique())
    all_meta_val_classes = list(meta_valframe["ID"].unique())

    
    meta_train_support_tasks, meta_train_query_tasks = [], []
    for each_task in range(args.n_train_tasks):  # num_train_task 训练任务的总数
        task = Task(all_meta_train_classes, args.n_way, args.k_shot, args.k_qry, meta_trainframe)
        meta_train_support_tasks.append(task.support_roots)
        meta_train_query_tasks.append(task.query_roots)

    meta_val_support_tasks, meta_val_query_tasks  = [], []

    for each_task in range(args.n_val_tasks):
        task = Task(all_meta_val_classes, args.n_way, args.k_shot, args.k_qry, meta_valframe)
        meta_val_query_tasks.append(task.support_roots)
        meta_val_support_tasks.append(task.query_roots)

    
    # final test, aka meta test, multi-tasks by samping data according to meta theory, 
    # but wo juet build one tasks using all data, according dl strategy
    # 根据元学习理论，元测试任务是多任务，每个任务训练 验证的样本量很少，但是我们这里按照深度学习的策略，只构建一个任务，将所有数据全用上
    test_support_tasks,test_query_tasks = [], []
    final_test_tasks = []#用于最终测试

    for each_task in range(args.n_test_tasks):
        task = DL_Task(args, [train_df_ori, val_df_ori, test_df_ori])

        train_set = BV1000_OCT_CNV(args, fileroots=task.support_roots)
        val_set = BV1000_OCT_CNV(args, fileroots=task.query_roots, mode='val')
        test_set = BV1000_OCT_CNV(args, fileroots=task.test_roots, mode='test')

        train_loader = DataLoader(train_set, shuffle=True, batch_size = args.batch_size_train, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size = args.batch_size_val, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=True, batch_size = args.batch_size_test, num_workers=args.num_workers, pin_memory=True)


        test_support_tasks.append(train_loader)
        test_query_tasks.append(val_loader)
        final_test_tasks.append(test_loader)

    
    meta_train_support_loader = DataLoader(BV_CNV_MetaDataset(args, meta_train_support_tasks), 
                                      batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)
    
    meta_train_query_loader = DataLoader(BV_CNV_MetaDataset(args,meta_train_query_tasks),
                                    batch_size=args.meta_size, shuffle=True, num_workers=0, pin_memory=True)
    

    meta_val_support_loader = DataLoader(BV_CNV_MetaDataset(args, meta_val_support_tasks),
                                     batch_size=args.meta_size, shuffle=True, num_workers=0, pin_memory=True)
    
    meta_val_query_loader = DataLoader(BV_CNV_MetaDataset(args, meta_val_query_tasks),
                                   batch_size=args.meta_size, shuffle=True, num_workers=0, pin_memory=True)
    

    
    
    return [meta_train_support_loader, meta_train_query_loader], [meta_val_support_loader, meta_val_query_loader], [test_support_tasks, test_query_tasks, final_test_tasks]

def bv1000_oct_cnv_for_tl(args):
    # synthetic_dframe 是所有的合成数据， synthetic_dframe synthetic data 
    synthetic_dframe = pd.read_csv(args.synthetic_data_csv)

    # real_dframe是所有的真实数据， real_dframe is all real data
    # real_dframe = pd.read_csv(os.path.join(args.project_path,'data/bv1000_ocv_cnv/real_data/all_data.csv'))
    # real_dframe = pd.read_csv(os.path.join(args.project_path,args.real_data_csv))
    train_df_ori = pd.read_csv(os.path.join(args.project_path, args.train_csv, f't_{args.index_fold}_s.csv'))
    val_df_ori = pd.read_csv(os.path.join(args.project_path, args.val_csv, f't_{args.index_fold}_q.csv'))
    test_df_ori = pd.read_csv(os.path.join(args.project_path,args.test_csv))

    # 仅使用一定比例的数据如1或0.5
    # just use part of dataset for train and test
    if args.use_trainset_percent != 1.0 :
        train_df_ori = train_df_ori.sample(frac=args.use_trainset_percent,random_state=args.random_state)
        val_df_ori = val_df_ori.sample(frac=args.use_trainset_percent,random_state=args.random_state)
        test_df_ori = test_df_ori.sample(frac=args.use_trainset_percent,random_state=args.random_state)

    # synthetic_dframe 是所有数据，包括合成数据， synthetic_dframe is all data, include synthetic data 
    # 根据原始的数据，获取选取的合成数据的dataframe, obtain choosed synthetic dataframe according origin data ID， include synthetic data
    meta_trainframe = synthetic_dframe[synthetic_dframe['ID'].isin(list(train_df_ori['ID']))]
    # meta_valframe = synthetic_dframe[synthetic_dframe['ID'].isin(list(val_df_ori['ID']))]
    meta_valframe = val_df_ori # 使用真实的验证集
    meta_testframe = synthetic_dframe[synthetic_dframe['ID'].isin(list(test_df_ori['ID']))]

    # final test, aka meta test, multi-tasks by samping data according to meta theory, 
    # but wo juet build one tasks using all data, according dl strategy
    # 根据元学习理论，元测试任务是多任务，每个任务训练 验证的样本量很少，但是我们这里按照深度学习的策略，只构建一个任务，将所有数据全用上
    test_support_tasks,test_query_tasks = [], []
    final_test_tasks = []#用于最终测试

    for each_task in range(args.n_test_tasks):
        task = DL_Task(args, [meta_trainframe, meta_valframe, meta_testframe])

        train_set = BV1000_OCT_CNV_Sys(args, fileroots=task.support_roots)
        # val_set = BV1000_OCT_CNV_Sys(args, fileroots=task.query_roots, mode='val')
        val_set = BV1000_OCT_CNV(args, fileroots=task.query_roots, mode='val') # 使用真实图片验证
        test_set = BV1000_OCT_CNV_Sys(args, fileroots=task.test_roots, mode='test')

        train_loader = DataLoader(train_set, shuffle=True, batch_size = args.batch_size_train, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size = args.batch_size_val, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=True, batch_size = args.batch_size_test, num_workers=args.num_workers, pin_memory=True)


        test_support_tasks.append(train_loader)
        test_query_tasks.append(val_loader)
        final_test_tasks.append(test_loader)
   
    return [], [], [test_support_tasks, test_query_tasks, final_test_tasks]

def bv1000_oct_cnv_k_fold(args, real_dframe):

    # begin+++++++++++++++++++ read from csv file directly
    train_df_ori = pd.read_csv(os.path.join(args.project_path, args.k_fold_csv, str(args.index_fold), "train.csv"))
    test_df_ori = pd.read_csv(os.path.join(args.project_path, args.k_fold_csv, str(args.index_fold), "test.csv"))
    
    return train_df_ori, test_df_ori
    # end--------------------- read from csv file directly

    # begin+++++++++++++++++++ use code/process to splite
    k_fold_num = args.k_fold_num
    # kf = KFold(n_splits=k_fold_num, random_state=None) # 4折
    kf = GroupKFold(n_splits=k_fold_num)
    # dframe = pd.read_csv(args.real_data_csv)
    k_fold_count = 0


    real_train_df_ls = []
    real_test_df_ls = []
    groups = list(real_dframe['Eye'])
    for train_index, test_index in kf.split(real_dframe,groups=groups):
        k_fold_count += 1
        print('\n{} of kfold {}'.format(k_fold_count,kf.n_splits))
        train_df_ori = real_dframe.iloc[train_index,:]
        test_df_ori = real_dframe.iloc[test_index,:]
        real_train_df_ls.append(train_df_ori)
        real_test_df_ls.append(test_df_ori)

     # end-------------------- use code/process to splide
     
    return real_train_df_ls[args.index_fold-1], real_test_df_ls[args.index_fold-1]


