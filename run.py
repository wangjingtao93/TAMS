# 加载预训练参数，进行元训练
import sys
import yaml
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')

""" Generate commands for test. """
import os



def gen_args():
    description_name = 'no_imp'
    project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation'
    relative_path = project_path.replace('pycharm_remote/', 'pycharm_remote/result/')
    save_path = os.path.join(relative_path, 'result/result_20240408/' )
    # save_path = os.path.join(relative_path, 'tmp/' )
    
    datatypes = {0:'bv1000-oct-cnv', 1:'rips', 2:'heshi-rp', 3:'drive-eye', 4:'fs1000/bv1000-oct-cnv', 5:'fs1000/heshi-rp', 6:'polyp/bv1000-oct-cnv', 7:'polyp/heshi-rp'}
    datatype = datatypes[0]
    
    index_fold = 3 # 0,1,2,3,
    # gpus={'bv1000-oct-cnv':0, 'rips':0, 'heshi-rp':0}
    # gpu = gpus[datatype]
    gpu = 0

    algs = {0:'dl', 1:'pretrain', 2:'itams', 3:'mtams', 4:'reptile', 5:'meta_test_itam', 6:'meta_test_mtams',7:'predict/dl', 8:'predict/itams', 9:'predict/mtams'}
    alg = algs[9]

    # note:rips, unet++和 maenet, lr=0.1, 否则不收敛
    nets = {0:'unet', 1:'unet++', 2:'transUNet', 3:'manet'}
    net = nets[0]

    data_path_prefix_sys=""


    with open(os.path.join(project_path,'configs.yaml'), 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    load = result['load_dict'][alg][datatype]
    load_interrupt_path = ''
    meta_learner_load = f'/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-segmentation/result/result_20240408/{datatype}/pretrain/unet/{index_fold}_fold/' + result[datatype]['meta_learner_load'][index_fold] + '/meta_epoch/taskid_0/best_epoch_for_val_meta_epoch_0.pth'
    
    
    args_dict = {}
    
    # base
    args_dict['gpu'] = [gpu]
    args_dict['save_path'] = save_path
    args_dict['description_name'] = description_name
    # args_dict['is_base_agu'] = False
    args_dict['is_meta_base_agu'] = True

    # dataset
    args_dict['dl_resize'] = 256
    args_dict['datatype'] = datatype
    args_dict['use_trainset_percent'] = 0.3
    args_dict['random_state'] = 42
    args_dict['data_path_prefix_sys'] = data_path_prefix_sys
    args_dict['alg'] = alg
    args_dict['index_fold'] = index_fold
    args_dict['is_gsnet'] = False

    # for predict
    args_dict['is_save_fig'] = True

    # dl
    args_dict['n_epoch'] = result[alg]['n_epoch']
    args_dict['net'] = net
    args_dict['n_classes'] = 1
    args_dict['n_channels'] = result[datatype]['n_channels']
    args_dict['dl_lr'] = 0.01
    args_dict['batch_size_train'] = 16
    args_dict['batch_size_val'] = 16
    
    if args_dict['is_meta_base_agu']:
        args_dict['batch_size_test'] = 1
    else:
        args_dict['batch_size_test'] = 16
    args_dict['load'] = load
    args_dict['load_interrupt_path'] = load_interrupt_path
    args_dict['is_save_val_net'] = True # 最好不用用字符串，尤其是'false', 都会当做true
    args_dict['n_mid_val'] = 1
    args_dict['is_mid_val'] = False
    # trans net
    args_dict['trans_depth'] = 12

    # meta
    args_dict['meta_learner_load'] = meta_learner_load
    args_dict['meta_resize'] = 256
    args_dict['n_way'] = 2
    args_dict['n_inner'] = result[alg]['n_inner']
    args_dict['k_shot'] = 5
    args_dict['k_qry'] =  5
    args_dict['n_train_tasks'] = 500
    args_dict['n_val_tasks'] = 50
    args_dict['n_test_tasks'] = 1
    args_dict['meta_size'] = 5
    args_dict['test_meta_size'] = 1
    args_dict['outer_lr'] = result[alg]['outer_lr']
    args_dict['inner_lr'] = result[alg]['inner_lr']
    args_dict['n_meta_epoch'] = 100
    args_dict['outer_opt'] = 'Adam'

    # imaml
    args_dict['version'] = 'GD'
    args_dict['cg_steps'] = 5
    args_dict['lr_sched'] = True

    
    # file path
    args_dict['real_data_csv'] = result[datatype]['real_data_csv']
    args_dict['synthetic_data_csv'] = result[datatype]['synthetic_data_csv']
    args_dict['train_csv'] = result[datatype]['train_csv']
    args_dict['val_csv'] = result[datatype]['val_csv']
    args_dict['test_csv'] =  result[datatype]['test_csv']

    check_error(args_dict)

    return args_dict

    
def run_command(args_dict):

    command_str = ''
    for key, values in args_dict.items():
        command_str += ' --' + key + '=' + str(values)

    the_command = 'python ../main.py --is_run_command=True' + command_str

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

def check_error(args_dict):

    # alg = predict
    if 'predict' in args_dict['alg']:
        if args_dict['load'] == '':
            raise ValueError('meta potential parameters must been exist')
               
        if f'{args_dict["index_fold"]}_fold' not in args_dict['load']:
            raise ValueError('load path index_fold is not corresponding')
        
        meta_alg = args_dict['alg'].split('/')[-1]
        key_val = {'dl':'dl', 'itams':'imaml', 'mtams':'maml'}
        if key_val[meta_alg] not in args_dict['load']:
            raise ValueError('load meta path is not corresponding')

        if args_dict['datatype'] not in args_dict['load']:
            raise ValueError('load meta path datatype is not corresponding')
      
    # alg == dl
    if args_dict['alg'] == 'dl':
        if args_dict['load'] != '':
            if args_dict['datatype'] not in args_dict['load']:
                raise ValueError('dl load path [datatype] is not corresponding')
            if args_dict['net'] not in args_dict['load']:
                raise ValueError('dl load path [net] is not corresponding')
            


if __name__ == '__main__':
    
    args_dict = run_command()

    

    run_command(args_dict)
