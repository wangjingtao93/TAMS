# 加载预训练参数，进行元训练
import sys
import yaml
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')

""" Generate commands for test. """
import os

def run_exp():
    description_name = 'task1000_meta_pretrain_poly_lr0.00001'
    # save_path = 'tmp' 
    save_path = 'result/result_20231130/'
    gpu = 1

    alg = 'imaml'
    net = 'unet'
    datatype = 'bv1000-oct-cnv'

    data_path_prefix=""

    load = ''
    load_interrupt_path = ''
    meta_learner_load = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/result/result_20230530/bv1000-oct-cnv/pretrain/unet/2023-04-05-18-15-44/max_dsc_2023-04-05-18-15-44.pth'

    with open('configs.yaml', 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

 
    the_command = 'python3 ../main.py --gpu=' + str(gpu) \
        + ' --datatype=' + datatype \
        + ' --data_path_prefix=' + str(data_path_prefix) \
        + ' --use_trainset_percent=' + str(1.0) \
        + ' --dl_resize=' + str(512) \
        + ' --meta_resize=' + str(256) \
        + ' --n_classes=1' \
        + ' --n_channels=1' \
        + ' --index_fold=' + str(1) \
        + ' --alg=' + alg \
        + ' --net=' + net \
        + ' --n_epoch=' + str(30) \
        + ' --dl_lr=0.01' \
        + ' --save_path=' + save_path \
        + ' --load=' + load \
        + ' --load_interrupt_path=' + load_interrupt_path \
        + ' --meta_learner_load=' + meta_learner_load \
        + ' --is_mid_val=true' \
        + ' --n_mid_val=5' \
        + ' --is_save_val_net=True' \
        + ' --n_meta_epoch=' + str(50) \
        + ' --n_way=' + str(2) \
        + ' --k_shot=' + str(4) \
        + ' --k_qry=' + str(4) \
        + ' --meta_size=' + str(4) \
        + ' --test_meta_size=' + str(1) \
        + ' --outer_lr=' + str(result[alg]['outer_lr']) \
        + ' --inner_lr=' + str(result[alg]['inner_lr']) \
        + ' --n_inner=' + str(result[alg]['n_inner']) \
        + ' --n_train_task=' + str(1000) \
        + ' --n_val_tasks=' + str(200) \
        + ' --n_test_task=' + str(1) \
        + ' --version=GD'\
        + ' --cg_steps=' + str(5) \
        + ' --outer_opt=Adam' \
        + ' --lambda=' + str(2) \
        + ' --description_name=' + description_name \
        + ' --is_save_val_net=True' \
        + ' --lr_sched=true' \
        + ' --real_data_csv=' + result[datatype]['real_data_csv'] \
        + ' --synthetic_data_csv=' + result[datatype]['synthetic_data_csv'] \
        + ' --train_csv=' + result[datatype]['train_csv'] \
        + ' --val_csv=' +  result[datatype]['val_csv'] \
        + ' --test_csv=' +  result[datatype]['test_csv'] \
        + ' --k_fold_csv=' +  result[datatype]['k_fold_csv'] \
               
    os.system(the_command)

run_exp()