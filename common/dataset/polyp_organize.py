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
from common.dataset.dataset_rips import RIPSDataset
from common.dataset.dataset_bv1000_cnv import BV1000_OCT_CNV
from common.dataset.build_tasks import Task, DL_Task
from .dataset_fs1000 import MetaDatasetfs1000




def polyp_organize(args):
    '''
    返回任务级别的数据，
    return tasks dataloader
    '''
    # synthetic_dframe 是所有的合成数据， synthetic_dframe synthetic data 
    # synthetic_dframe = pd.read_csv(os.path.join(args.project_path,'data/bv1000_ocv_cnv/meta_data/synthetic_data.csv'))
    fs1000_dframe = pd.read_csv(os.path.join(args.project_path, args.synthetic_data_csv))

    # real_dframe是所有的真实数据， real_dframe is all real data
    # real_dframe = pd.read_csv(os.path.join(args.project_path,'data/bv1000_ocv_cnv/real_data/all_data.csv'))
    # real_dframe = pd.read_csv(os.path.join(args.project_path,args.real_data_csv))

    # not include synthetic data 不包括合成数据 for test_tasks
    train_df_ori = pd.read_csv(os.path.join(args.project_path, args.train_csv, f't_{args.index_fold}_s.csv'))
    val_df_ori = pd.read_csv(os.path.join(args.project_path, args.val_csv, f't_{args.index_fold}_q.csv'))
    test_df_ori = pd.read_csv(os.path.join(args.project_path, args.test_csv))

    # 仅使用一定比例的数据如1或0.5
    # just use part of dataset for train and test
    if args.use_trainset_percent != 1.0 :
        train_df_ori = train_df_ori.sample(frac=args.use_trainset_percent,random_state=args.random_state)
        val_df_ori = val_df_ori.sample(frac=args.use_trainset_percent,random_state=args.random_state)
        # test_df_ori = test_df_ori.sample(frac=args.use_trainset_percent,random_state=args.random_state)

    
    meta_trainframe = fs1000_dframe
    # meta_valframe = synthetic_dframe[synthetic_dframe['ID'].isin(list(val_df_ori['ID']))]


    all_meta_train_classes = list(meta_trainframe["Class_ID"].unique())
    # all_meta_val_classes = list(meta_valframe["ID"].unique())

    # 将ID 替换为Class_ID
    meta_trainframe['ID'] = list(meta_trainframe["Class_ID"])
    
    meta_train_support_tasks, meta_train_query_tasks = [], []
    for each_task in range(args.n_train_tasks):  # num_train_task 训练任务的总数
        task = Task(all_meta_train_classes, args.n_way, args.k_shot, args.k_qry, meta_trainframe)
        meta_train_support_tasks.append(task.support_roots)
        meta_train_query_tasks.append(task.query_roots)

    meta_val_support_tasks, meta_val_query_tasks  = [], []

    # for each_task in range(args.n_val_tasks):
    #     task = Task(all_meta_val_classes, args.n_way, args.k_shot, args.k_qry, meta_valframe)
    #     meta_val_query_tasks.append(task.support_roots)
    #     meta_val_support_tasks.append(task.query_roots)

    
    # final test, aka meta test, multi-tasks by samping data according to meta theory, 
    # but wo juet build one tasks using all data, according dl strategy
    # 根据元学习理论，元测试任务是多任务，每个任务训练 验证的样本量很少，但是我们这里按照深度学习的策略，只构建一个任务，将所有数据全用上
    test_support_tasks,test_query_tasks = [], []
    final_test_tasks = []#用于最终测试

    if 'bv1000-oct-cnv' in args.datatype:
        dataset_func  = BV1000_OCT_CNV
    elif 'heshi_rp' in args.datatype:
        dataset_func= RIPSDataset
        

    for each_task in range(args.n_test_tasks):
        task = DL_Task(args, [train_df_ori, val_df_ori, test_df_ori])

        train_set = dataset_func(args, fileroots=task.support_roots)
        val_set = dataset_func(args, fileroots=task.query_roots, mode='val')
        test_set = dataset_func(args, fileroots=task.test_roots, mode='test') # test=val

        train_loader = DataLoader(train_set, shuffle=True, batch_size = args.batch_size_train, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size = args.batch_size_val, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=True, batch_size = args.batch_size_test, num_workers=args.num_workers, pin_memory=True)

        test_support_tasks.append(train_loader)
        test_query_tasks.append(val_loader)
        final_test_tasks.append(test_loader)
    
    meta_train_support_loader = DataLoader(MetaDatasetfs1000(args, meta_train_support_tasks), 
                                      batch_size=args.meta_size, num_workers=0, pin_memory=True, shuffle=True)
    
    meta_train_query_loader = DataLoader(MetaDatasetfs1000(args, meta_train_query_tasks),
                                    batch_size=args.meta_size, shuffle=True, num_workers=0, pin_memory=True)
    

    # meta_val_support_loader = DataLoader(MetaDatasetfs1000(args, meta_val_support_tasks),
    #                                  batch_size=args.meta_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # meta_val_query_loader = DataLoader(MetaDatasetfs1000(args, meta_val_query_tasks),
    #                                batch_size=args.meta_size, shuffle=True, num_workers=0, pin_memory=True)
    

    
    
    return [meta_train_support_loader, meta_train_query_loader], [[], []], [test_support_tasks, test_query_tasks, final_test_tasks]