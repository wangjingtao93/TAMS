# 加载预训练参数，进行元训练
import sys
import yaml
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')

""" Generate commands for test. """
import os

def gen_args():
    project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation'
    relative_path = project_path.replace('pycharm_remote', 'pycharm_remote/result')
    save_path = os.path.join(relative_path, 'task_aug/result_sys/20240419/demo')
    datatype_ls = {1:'demo-bv1000-oct-cnv', 2:'bv1000-oct-cnv', 3:'heshi-rp', 4:'rips'}
    datatype = datatype_ls[1]
    

    yaml_path = os.path.join(project_path, 'task_aug/run_exe/configs.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    args_dict = {}
    args_dict['project_path'] = project_path
    args_dict['datatype'] = datatype
    args_dict['save_path'] = save_path
    args_dict['expand_n'] = 10
    args_dict['real_data_csv'] = str(result[datatype]['real_data_csv'])
    args_dict['normal_data_csv'] = str(result[datatype]['normal_data_csv'])
    args_dict['label_text_path'] = str(result[datatype]['label_text_path'])

    return args_dict

def run_command(args_dict):

    command_str = ''
    for key, values in args_dict.items():
        command_str += ' --' + key + '=' + str(values)

    the_command = 'python ../main.py --is_run_command=True' + command_str

    # project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation'
    # relative_path = project_path.replace('pycharm_remote', 'pycharm_remote/result')
    # save_path = os.path.join(relative_path, 'task_aug/result_sys/20240419/demo')
    # datatype_ls = {1:'demo-bv1000-oct-cnv', 2:'bv1000-oct-cnv', 3:'heshi-rp', 4:'rips'}
    # datatype = datatype_ls[4]
    

    # yaml_path = os.path.join(project_path, 'task_aug/run_exe/configs.yaml')
    # with open(yaml_path, 'r', encoding='utf-8') as f:
    #     result = yaml.load(f.read(), Loader=yaml.FullLoader)

    # main_path = os.path.join(project_path, 'task_aug/main.py')

    # the_command = 'python3 ' + main_path \
    #     + ' --project_path=' + project_path \
    #     + ' --datatype=' + datatype \
    #     + ' --save_path=' + save_path \
    #     + ' --expand_n=' + str(10) \
    #     + ' --real_data_csv=' + str(result[datatype]['real_data_csv']) \
    #     + ' --normal_data_csv=' + str(result[datatype]['normal_data_csv']) \
    #     + ' --label_text_path=' + str(result[datatype]['label_text_path']) \
  
    os.system(the_command)

if __name__ == '__main__':
    
    args_dict = run_command()

    

    run_command(args_dict)