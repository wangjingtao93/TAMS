import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')

import os
import time
import json

import  common.dataset.data_organize as data_organize
import common.utils as utils
from  imaml.imaml_enter import imaml_enter
from maml.maml_enter import maml_enter
from dl.dl_enter import dl_enter
from dl.predict import predict
import shutil
import traceback
import run as dl_exe
import yaml
import torch


def main(args):

    # organize data
    train_dataloader_ls, val_dataloader_ls, test_data_ls=data_organize.data_organize(args)
    # train_dataloader_ls, val_dataloader_ls, test_data_ls = [],[],[[],[],[]]

        
    # choose alg
    if args.alg == 'dl' or args.alg == 'pretrain' or args.alg == 'meta_test_imaml' or args.alg == 'meta_test_maml':
        dl_enter(args, test_data_ls)
    elif 'predict' in args.alg:
        predict(args, test_data_ls)
    elif args.alg == 'imaml':
        imaml_enter(args, train_dataloader_ls, val_dataloader_ls, test_data_ls)
    elif args.alg == 'maml':
        maml_enter(args, train_dataloader_ls, val_dataloader_ls, test_data_ls)
    elif args.alg=='FOMAML':
        pass
    else:
        raise ValueError('Not implemented Meta-Learning Algorithm')
    
    print('over+++++++++++++++++++')

def parse_args():
    import argparse

    parser = argparse.ArgumentParser('Gradient-Based Meta-Learning Algorithms')
    
    # base settings
    parser.add_argument('--is_run_command', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--is_debug', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--description_name', type=str, default='description')
    parser.add_argument('--description', type=str, default='hh')
    parser.add_argument('--real_data_csv', type=str, default='')
    parser.add_argument('--synthetic_data_csv', type=str, default='')
    parser.add_argument('--train_csv', type=str, default='')
    parser.add_argument('--val_csv', type=str, default='')
    parser.add_argument('--test_csv', type=str, default='')
    parser.add_argument('--is_remove_exres', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--is_base_agu', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--is_meta_base_agu', type=lambda x: (str(x).lower() == 'true'), default=False)
    
    parser.add_argument('--data_path_prefix_sys', type=str, default='')
    parser.add_argument('--seed', type=int, default=1) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='0 = GPU.')
    parser.add_argument('--save_path', type=str, default='tmp')
    parser.add_argument('--n_classes', type=int, default=1)# Total number of fc out_class
    parser.add_argument('--n_channels', type=int, default=1)# Total number of in channels

    parser.add_argument('--prefix', type=str, default='debug') # The network architecture
    parser.add_argument('--is_load_imagenet', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--is_load_zk', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--is_load_imagenet_zk', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--is_load_st_sub', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--dl_resize', type=int, default=256)
    parser.add_argument('--meta_resize', type=int, default=256)

    parser.add_argument('--load',type=str, default="")

    parser.add_argument('--load_interrupt_path', type=str, default="", help='Load model from a .pth file for interrupt recover')
    parser.add_argument('--best_acc', type=float, default=0.0)
    parser.add_argument('--best_epoch', type=int, default=0)

    # for dataset
    parser.add_argument('--datatype', type=str, default='bv1000-oct-cnv')
    parser.add_argument('--use_trainset_percent', '-udp', metavar='UDP', type=float, default=1.0, help='use data percent')
    parser.add_argument('--random_state', type=int, default=42) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--is_gsnet', type=lambda x: (str(x).lower() == 'true'), default=True)
    # for RP
    
    # for predict
    parser.add_argument('--is_save_fig', type=lambda x: (str(x).lower() == 'true'), default=False)

    # for dl/
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--is_save_val_net', type=lambda x: (str(x).lower() == 'true'), default=True) # 是否保存验证集上最好的模型
    parser.add_argument('--batch_size_train', type=int, default=16)
    parser.add_argument('--batch_size_val', type=int, default=16)
    parser.add_argument('--batch_size_test', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading (default: 4).')
    parser.add_argument('--dl_lr', type=float, default=1e-2)
    parser.add_argument('--lr_mode', '-lm', type=str, default='poly')
    parser.add_argument('--is_mid_val', type=lambda x: (str(x).lower() == 'true'), default=False) # whether multi evl in per dl epoch, 每个epoch是否进行多次测试
    parser.add_argument('--n_mid_val', type=int, default=5, help='Number times of valdations per epcoh (default: 5).')
    
    # algorithm settings
    parser.add_argument('--alg', type=str, default='iMAML')
    parser.add_argument('--n_inner', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=1e-2)
    parser.add_argument('--inner_opt', type=str, default='SGD')
    parser.add_argument('--outer_lr', type=float, default=1e-3)
    parser.add_argument('--outer_opt', type=str, default='Adam')
    parser.add_argument('--lr_sched', type=lambda x: (str(x).lower() == 'true'), default=False)

    parser.add_argument('--n_train_tasks', type=int, default=1000)# Total number of trainng tasks
    parser.add_argument('--n_val_tasks', type=int, default=250)# Total number of trainng tasks
    parser.add_argument('--n_test_tasks', type=int, default=1)# Total number of testing tasks

    # meta training settings
    parser.add_argument('--n_meta_epoch', type=int, help='',default=50)
    parser.add_argument('--n_way', type=int, help='n way', default=2)
    parser.add_argument('--k_shot', type=int, help='k shot for support set', default=5)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    parser.add_argument('--meta_size', type=int, help='meta batch size, namely meta_size',default=4)
    parser.add_argument('--meta_learner_load', type=str, help='meta learner load path',default='')

    # test tasks
    parser.add_argument('--test_k_shot', type=int, help='k shot for support set', default=5)
    parser.add_argument('--test_k_qry', type=int, help='k shot for query set', default=5)
    # 默认值为1，最好不要改动 default 1, better not to chanage
    parser.add_argument('--test_meta_size', type=int, help='meta batch size, namely meta_size',default=1)


    # imaml specific settings
    parser.add_argument('--lambda', type=float, default=2.0)# 没有使用到
    parser.add_argument('--version', type=str, default='GD')
    parser.add_argument('--cg_steps', type=int, default=5)
    
    # network settings
    parser.add_argument('--net', type=str, default='resnet18')
    # parser.add_argument('--n_conv', type=int, default=4)
    # parser.add_argument('--n_dense', type=int, default=0)
    # parser.add_argument('--hidden_dim', type=int, default=64)
    # parser.add_argument('--in_channels', type=int, default=1)
    # parser.add_argument('--hidden_channels', type=int, default=64,
    #     help='Number of channels for each convolutional layer (default: 64).')
    # transformer 深度
    parser.add_argument('--trans_depth', type=int, default=12)

    # k-fold
    parser.add_argument('--k_fold_num', '-kf', type=int, default=4, help='k fold')
    parser.add_argument('--index_fold', type=int, default=0, help='the index fold.')


    args = parser.parse_args()

    if not args.is_run_command:
        args_dict = dl_exe.gen_args() 
        for key, value in args_dict.items():
            setattr(args, key, value)
    if args.is_debug:
        setattr(args, 'save_path', 'tmp')
        setattr(args, 'n_epoch', 2)
        setattr(args, 'n_meta_epoch', 2)
        setattr(args, 'n_train_tasks', 5)
        setattr(args, 'n_val_tasks', 2)
    return args


def create_store_dir(args):
    args.project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation'

    begin_time = time.time()
    time_name=time.strftime('%Y-%m-%d-%H-%M-%S')

    store_dir = os.path.join(args.save_path, args.datatype)
    if args.use_trainset_percent != 1.0:
        store_dir = os.path.join(store_dir, 'percent', str(args.use_trainset_percent))

    args.store_dir =os.path.join(store_dir, args.alg.lower(), args.net, str(args.index_fold) + '_fold', time_name)
    utils.mkdir(args.store_dir)
    
    # 创建一个记录测试任务的
    args.store_meta_test = os.path.join(args.store_dir,'meta_epoch')
    utils.mkdir(args.store_meta_test)

    

    # 创建一个说明文件
    description_file = os.path.join(args.store_dir,  args.description_name)
    with open(str(description_file), 'w') as f:
        f.write(args.description)



if __name__ == '__main__':
    args = parse_args()
    
    # utils.set_gpu(args.gpu)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.cuda.set_device(args.gpu[0])
    utils.set_seed(args.seed)

    # args is not deepcopy
    create_store_dir(args)
    utils.save_args_to_file(args, os.path.join(args.store_dir, 'args.json'))

    try:
        main(args)
    except Exception:
        print(traceback.print_exc())
        if args.is_remove_exres:
            shutil.rmtree(args.store_dir)
