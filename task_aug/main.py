import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')

import os
import time
import shutil
import traceback
import pandas as pd
import glob

import common.utils as utils
from task_aug.synthetic.synthetic_data import  SyntheticDrusen
from task_aug.synthetic.simulate_heshi import SyntheticHeShiRP
from task_aug.synthetic.simulate_oct import SyntheticCNV
from task_aug.synthetic.simulate_rips import SyntheticRIPS
from task_aug.synthetic.base_agu import BaseAgu
import task_aug.run_exe.run as dl_exe


def parse_args():
    import argparse

    parser = argparse.ArgumentParser('Gradient-Based Meta-Learning Algorithms')

    parser.add_argument('--is_run_command', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--is_debug', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--is_remove_exres', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--base_agu', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--project_path', type=str, default='')
    parser.add_argument('--real_data_csv', type=str, default='') # real data with focus
    parser.add_argument('--normal_data_csv', type=str, default='') # real data without foucus (normal images)
    parser.add_argument('--save_path', type=str, default='') # 
    parser.add_argument('--label_text_path', type=str, default='') # all lesions location infor

    parser.add_argument('--datatype', type=str, default='cnv_oct')
    parser.add_argument('--expand_n', type=int, default=20) # 生成数量 


    args = parser.parse_args()

    if not args.is_run_command:
        args_dict = dl_exe.gen_args() 
        for key, value in args_dict.items():
            setattr(args, key, value)

    if args.is_debug:
        setattr(args, 'save_path', 'tmp')
        

    return args

def exe_cnv_oct(args):
    synthetic = SyntheticCNV(args)
    synthetic.forward()
def exe_heshi_rp(args):
    synthetic = SyntheticHeShiRP(args)
    synthetic.forward()
def exe_drusen_oct(args):
    synthetic = SyntheticDrusen(args)
    synthetic.forward()
def exe_rips(args):
    synthetic = SyntheticRIPS(args)
    synthetic.forward()
def exe_base_agu(args):
    base_agu = BaseAgu(args)
    base_agu.forward()

def synthetic_enter(args):
    if args.base_agu:
        exe_base_agu(args)
    elif args.datatype == 'bv1000-oct-cnv' or args.datatype == 'demo-bv1000-oct-cnv':
        exe_cnv_oct(args)
    elif args.datatype == 'bv1000-oct-drusen':
        exe_drusen_oct(args)
    elif args.datatype == 'heshi-rp':
        exe_heshi_rp(args)
    elif args.datatype == 'rips':
        exe_rips(args)
    else:
        raise ValueError('no datatype implement')


def tmp():
    print('mnihao++++++++++++++')
    os.mkdir('pppp/pppp')

if __name__ == '__main__':
    args = parse_args()

    time_name=time.strftime('%Y-%m-%d-%H-%M-%S')
    # store synthetic data
    # self.store_dir = 'D:/workplace/python/metaLearning/meta-learning-seg/data_argumentation/data/result/'
    args.store_dir = os.path.join(args.save_path, args.datatype, time_name)
    utils.mkdir(args.store_dir)

    utils.save_args_to_file(args, os.path.join(args.store_dir, 'args.json'))
    
    try:
        synthetic_enter(args)
    except Exception:
        print(traceback.print_exc())
        if args.is_remove_exres:
            shutil.rmtree(args.store_dir)
