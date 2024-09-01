# 加载预训练参数，进行元训练
import sys
import yaml
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')

""" Generate commands for test. """
import os
import subprocess


def gen_args():
    description_name = 'nothing'
    project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation'
    relative_path = project_path.replace('pycharm_remote/', 'pycharm_remote/result/')
    save_path = os.path.join(relative_path, 'result/result_20240408/' )

    
    index_fold = 0 # 0,1,2,3,
    gpu = 0

    algs = {1:'dl', 2:'pretrain', 3:'imaml', 4:'maml', 5:'reptile', 6:'predict'}
    alg = algs[6]
    net = 'unet'

    datatypes = {0:'bv1000-oct-cnv', 1:'rips', 2:'heshi-rp', 3:'drive-eye'}
    datatype = datatypes[0]

    data_path_prefix_sys=""

    # baseline
    # load = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/result/result_20230530/bv1000-oct-cnv/dl/unet/2023-04-05-18-13-13/max_dsc_2023-04-05-18-13-13.pth'
    
    # tl
    # load = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/result/result_20230530/bv1000-oct-cnv/dl/unet/2023-04-05-19-22-08/max_dsc_2023-04-05-19-22-08.pth'
    
    # imaml
    # load = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/result/result_20230530/bv1000-oct-cnv/imaml/unet/2023-04-12-14-43-59/max_dsc_2023-04-12-14-43-59.pth'

    # tl + imaml
    # load = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/result/result_20230530/bv1000-oct-cnv/imaml/unet/2023-04-06-09-26-45/meta_epoch/2023-04-06-09-26-45/max_dsc_2023-04-06-09-26-45.pth'

    # batch_size_test = 1 # 1 和 4 测试结果会有差别，之前的版本用1，现在的版本用4

    load = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-segmentation/result/result_20240408/bv1000-oct-cnv/pretrain/unet/0_fold/2024-04-28-12-05-36/meta_epoch/taskid_0/best_epoch_for_val_meta_epoch_0.pth'
    batch_size_test = 16 # 1 和 4 测试结果会有差别，之前的版本用1，现在的版本用4，默认是4

    load_interrupt_path = ''

    
    with open(os.path.join(project_path,'run_exe/configs.yaml'), 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    args_dict = {}
    
    args_dict['gpu'] = [gpu]
    args_dict['save_path'] = save_path
    args_dict['dl_resize'] = 256
    args_dict['datatype'] = datatype
    args_dict['use_trainset_percent'] = 1.0
    args_dict['data_path_prefix_sys'] = data_path_prefix_sys
    args_dict['alg'] = alg
    args_dict['index_fold'] = index_fold
    args_dict['n_epoch'] = 100
    args_dict['net'] = net
    args_dict['n_classes'] = 1
    args_dict['n_channels'] = result[datatype]['n_channels']
    args_dict['dl_lr'] = 0.01
    args_dict['batch_size_train'] = 16
    args_dict['batch_size_val'] = 16
    args_dict['batch_size_test'] = batch_size_test
    args_dict['load'] = load
    args_dict['load_interrupt_path'] = load_interrupt_path
    args_dict['is_save_val_net'] = True # 最好不用用字符串，尤其是'false', 都会当做true
    args_dict['is_mid_val'] = False
    args_dict['n_mid_val'] = 5
    args_dict['trans_depth'] = 12
    args_dict['n_way'] = 2
    args_dict['n_inner'] = result[alg]['n_inner']
    args_dict['k_shot'] = 5
    args_dict['k_qry'] =  5
    args_dict['n_train_tasks'] = 1000
    args_dict['n_val_tasks'] = 200
    args_dict['n_test_tasks'] = 1
    args_dict['meta_size'] = 5
    args_dict['test_meta_size'] = 1
    args_dict['outer_lr'] = result[alg]['outer_lr']
    args_dict['inner_lr'] = result[alg]['inner_lr']
    args_dict['version'] = 'GD'
    args_dict['cg_steps'] = 5
    args_dict['outer_opt'] = 'Adam'
    args_dict['description_name'] = description_name
    args_dict['lr_sched'] = True
    args_dict['real_data_csv'] = result[datatype]['real_data_csv']
    args_dict['synthetic_data_csv'] = result[datatype]['synthetic_data_csv']
    args_dict['train_csv'] = result[datatype]['train_csv']
    args_dict['val_csv'] = result[datatype]['val_csv']
    args_dict['test_csv'] =  result[datatype]['test_csv']

    return args_dict

    
def run_command(args_dict):

    command_str = ''
    for key, values in args_dict.items():
        if key == 'gpu':
            command_str += ' --' + key + '=' + str(values[0])
        else:
            command_str += ' --' + key + '=' + str(values)

    the_command = 'python main.py --is_run_command=True' + command_str

    # the_command = 'python3 ../main.py --is_run_command=True'\
    #     + ' --gpu=' + str(args_dict['gpu'][0]) \
    #     + ' --save_path=' + args_dict['save_path'] \
    #     + ' --dl_resize=' + str(args_dict['dl_resize']) \
    #     + ' --datatype=' +  args_dict['datatype'] \
    #     + ' --use_trainset_percent=' + str(args_dict['use_trainset_percent']) \
    #     + ' --data_path_prefix=' + args_dict['data_path_prefix'] \
    #     + ' --alg=' + args_dict['alg'] \
    #     + ' --index_fold=' + str(args_dict['index_fold']) \
    #     + ' --n_epoch=' + str(args_dict['n_epoch'])\
    #     + ' --net=' + args_dict['net'] \
    #     + ' --n_channels=' +  str(args_dict['n_channels']) \
    #     + ' --dl_lr=' +  str(args_dict['dl_lr']) \
    #     + ' --batch_size_train=' +  str(args_dict['batch_size_train']) \
    #     + ' --batch_size_val=' +  str(args_dict['batch_size_val']) \
    #     + ' --batch_size_test=' + str(args_dict['batch_size_test']) \
    #     + ' --load=' + args_dict['load'] \
    #     + ' --load_interrupt_path=' + args_dict['load_interrupt_path'] \
    #     + ' --is_save_val_net=' +  str(args_dict['is_save_val_net'])\
    #     + ' --is_mid_val=' + str(args_dict['is_mid_val']) \
    #     + ' --n_mid_val=' + str(args_dict['n_mid_val']) \
    #     + ' --trans_depth=' + str(args_dict['trans_depth']) \
    #     + ' --n_way=' + str(args_dict['n_way']) \
    #     + ' --n_inner=' + str(args_dict['n_inner']) \
    #     + ' --k_shot=' + str(args_dict['k_shot']) \
    #     + ' --k_qry=' +  str(args_dict['k_qry']) \
    #     + ' --n_train_tasks=' + str(args_dict['n_train_tasks'])\
    #     + ' --n_val_tasks=' + str(args_dict['n_val_tasks']) \
    #     + ' --n_test_tasks=' + str(args_dict['n_test_tasks']) \
    #     + ' --meta_size=' + str(args_dict['meta_size']) \
    #     + ' --test_meta_size=' + str(args_dict['test_meta_size']) \
    #     + ' --outer_lr=' + str(args_dict['outer_lr']) \
    #     + ' --inner_lr=' + str(args_dict['inner_lr']) \
    #     + ' --version=' + args_dict['version']\
    #     + ' --cg_steps=' + str(args_dict['cg_steps']) \
    #     + ' --outer_opt=' +  str(args_dict['outer_opt']) \
    #     + ' --description_name=' + args_dict['description_name'] \
    #     + ' --lr_sched=' +  str(args_dict['lr_sched']) \
    #     + ' --real_data_csv=' +args_dict['real_data_csv'] \
    #     + ' --synthetic_data_csv=' + args_dict['real_data_csv'] \
    #     + ' --train_csv=' +  args_dict['train_csv'] \
    #     + ' --val_csv=' +  args_dict['val_csv'] \
    #     + ' --test_csv=' +   args_dict['test_csv'] \
    

    os.system(the_command)


if __name__ == '__main__':
    
    args_dict = gen_args()

    run_command(args_dict)
