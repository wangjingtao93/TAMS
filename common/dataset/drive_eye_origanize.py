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
from common.dataset.dataset_drive import DriveDataset
from common.dataset.build_tasks import Task, DL_Task


def drive_eye_organize(args):
    '''
    返回任务级别的数据，
    return tasks dataloader
    '''

    train_df_ori = pd.read_csv(os.path.join(args.project_path, args.train_csv))
    val_df_ori = pd.read_csv(os.path.join(args.project_path, args.val_csv))
    test_df_ori = pd.read_csv(os.path.join(args.project_path, args.val_csv))


    test_support_tasks,test_query_tasks = [], []
    final_test_tasks = []#用于最终测试

    for each_task in range(args.n_test_tasks):
        task = DL_Task(args, [train_df_ori, val_df_ori, test_df_ori])

        train_set = DriveDataset(args, fileroots=task.support_roots)
        val_set = DriveDataset(args, fileroots=task.query_roots, mode='val')
        test_set = DriveDataset(args, fileroots=task.test_roots, mode='test') # test=val

        train_loader = DataLoader(train_set, shuffle=True, batch_size = args.batch_size_train, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size = args.batch_size_val, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=True, batch_size = args.batch_size_test, num_workers=args.num_workers, pin_memory=True)

        print('++++++', len(train_loader))

        test_support_tasks.append(train_loader)
        test_query_tasks.append(val_loader)
        final_test_tasks.append(test_loader)
    

    
    # return [meta_train_support_loader, meta_train_query_loader], [meta_val_support_loader, meta_val_query_loader], [test_support_tasks, test_query_tasks, final_test_tasks]
    return [], [], [test_support_tasks, test_query_tasks, final_test_tasks]




