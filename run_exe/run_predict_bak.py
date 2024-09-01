# 加载预训练参数，进行元训练
import sys
import yaml
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')

""" Generate commands for test. """
import os

def run_exp():
    description_name = 'predict' 
    save_path = 'tmp' 
    gpu = 0
    index_fold = 0

    alg = 'predict'
    net = 'unet'
    data_path_prefix=""

    datatypes = {0:'bv1000-oct-cnv', 1:'rips', 2:'heshi-rp', 3:'drive-eye'}
    datatype = datatypes[3]
    

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

    with open('configs.yaml', 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    the_command = 'python3 ../main.py --gpu=' + str(gpu) \
        + ' --save_path=' + save_path \
        + ' --dl_resize=' + str(256) \
        + ' --datatype=' +  datatype \
        + ' --use_trainset_percent=1.0' \
        + ' --data_path_prefix=' + str(data_path_prefix) \
        + ' --alg=' + alg \
        + ' --index_fold=' + str(index_fold) \
        + ' --n_epoch=' + str(100)\
        + ' --net=' + net \
        + ' --n_channels=' + str(result[datatype]['n_channels']) \
        + ' --dl_lr=0.01' \
        + ' --batch_size_train=8' \
        + ' --batch_size_val=8' \
        + ' --batch_size_test=8' \
        + ' --load=' + load \
        + ' --load_interrupt_path=' + load_interrupt_path \
        + ' --is_save_val_net=True' \
        + ' --is_mid_val=false' \
        + ' --n_mid_val=5' \
        + ' --trans_depth=12' \
        + ' --n_way=' + str(2) \
        + ' --n_inner=' + str(result[alg]['n_inner']) \
        + ' --k_shot=' + str(5) \
        + ' --k_qry=' + str(5) \
        + ' --n_train_task=' + str(1000) \
        + ' --n_val_tasks=' + str(200) \
        + ' --n_test_tasks=' + str(1) \
        + ' --meta_size=5' \
        + ' --test_meta_size=1' \
        + ' --outer_lr=' + str(result[alg]['outer_lr']) \
        + ' --inner_lr=' + str(result[alg]['inner_lr']) \
        + ' --version=GD'\
        + ' --cg_steps=' + str(5) \
        + ' --outer_opt=Adam' \
        + ' --description_name=' + description_name \
        + ' --lr_sched=true' \
        + ' --real_data_csv=' + result[datatype]['real_data_csv'] \
        + ' --synthetic_data_csv=' + result[datatype]['synthetic_data_csv'] \
        + ' --train_csv=' + result[datatype]['train_csv'] \
        + ' --val_csv=' +  result[datatype]['val_csv'] \
        + ' --test_csv=' +  result[datatype]['test_csv'] \

    os.system(the_command)



run_exp()