import os
import csv
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import  DataLoader
from copy import deepcopy

from common.dl_comm import dl_comm
import common.utils as utils

def dl_enter(args,test_data_ls, meta_epoch=0):

    if args.alg == 'meta_test_imaml' or args.alg == 'meta_test_maml':
        meta_epoch =  int(args.load.split('/')[-1].split('_')[-1].replace('.pth', ''))
    
    return trainer(args,test_data_ls, meta_epoch)


def trainer(args,test_data_ls, meta_epoch):
    sppport_all_task = test_data_ls[0]
    query_all_task = test_data_ls[1]
    final_test_task = test_data_ls[2]

    # 多构建几个任务，结果会更准确, build more tasks will get more precise result
    task_num = len(sppport_all_task)

    metric_dir = os.path.join(args.store_meta_test,'metric_meta_epoch_' + str(meta_epoch) + '.csv')
    with open(str(metric_dir), 'w') as f:
        fields = ['task_idx', 'epoch', 'train_loss','train_dice', 'val_dice','val_iou','accuracy','f1_score','recall','precision','best_val_dice', 'best_epoch']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)  

    # 每个epoch内是否验证多次
    # Whether to verify multiple times per dl_epoch
    # if args.is_mid_val:
    # every task, middle epoch, best val pth
    args.middle_dir_meta_epoch = os.path.join(args.store_meta_test, 'middle_dl_epoch')
    utils.mkdir(args.middle_dir_meta_epoch)

    args.mid_dl_epoch_metric_dir = os.path.join(args.middle_dir_meta_epoch,'mid_dl_epoch_metric_meta_epoch_' + str(meta_epoch) + '.csv')
    with open(str(args.mid_dl_epoch_metric_dir), 'w') as f:
        fields = ['task_idx', 'epoch', 'train_loss','train_dice', 'val_dice','val_iou','accuracy','f1_score','recall','precision','best_val_dice', 'best_epoch']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)

    # 记录所有任务每个epoch的val 的pred, record every epoch val dice
    val_dice_all_epoch = [0] * args.n_epoch
    val_dice_all_epoch_mid = [0] * args.n_epoch

    # test_acc_all_epoch= []

    test_values_all_task = [] # record every test tasks, final test set pred
    test_values_all_task_mid = []
    for task_idx in range(task_num):
        # store per meta epoch,  per test task, per dl epoch(middle dl epoch), multi val_pth
        utils.mkdir(os.path.join(args.middle_dir_meta_epoch, f'taskid_{task_idx}'))
        args.taskpth_mid_epoch_store_dir = os.path.join(args.middle_dir_meta_epoch, f'taskid_{task_idx}', f'best_mid_dl_epoch_for_val_meta_epoch_{meta_epoch}.pth')
        args.task_idx=task_idx

        # 创建 网络 对象 create object network
        dl_ob = dl_comm(args)
        if args.alg == 'dl' and args.load != '':
            dl_ob.net.load_state_dict(torch.load(args.load))
        if args.alg == 'meta_test_imaml' or args.alg == 'meta_test_maml':
            dl_ob.net.load_state_dict(torch.load(args.load))
        elif args.alg == 'pretrain' and args.load != '':
            dl_ob.net.load_state_dict(torch.load(args.load))
        elif args.alg == 'imaml' or args.alg == 'maml' or  args.alg=='reptile':
            # meta的测试 meta test, use dl test strategy
            path = os.path.join(args.store_dir,'save_meta_pth', f'meta_epoch_{meta_epoch}.pth')
            dl_ob.net.load_state_dict(torch.load(path))

        train_loader = sppport_all_task[task_idx]
        val_loader = query_all_task[task_idx]
        test_loader = final_test_task[task_idx]


        # for val
        best_val = 0.0
        best_val_epoch = 0
        # 不用deepcopy是万万不行的，否则会随dl_ob.net发生变化
        # must use deepcopy, or dl_ob.net will chanage
        best_final_val_state_dict = deepcopy(dl_ob.net.state_dict())

        for epoch in range(args.n_epoch):
            dl_ob.adjust_learning_rate(epoch)
            res = run_epoch(epoch, dl_ob, train_loader, val_loader, test_loader, meta_epoch)

            val_dice_all_epoch[epoch] = val_dice_all_epoch[epoch] + res['val_dice']
            val_dice_all_epoch_mid[epoch] = val_dice_all_epoch_mid[epoch] + max(res['val_dice_mid_epoch'])

            is_val_best = res['val_dice'] > best_val
            best_val = max(best_val, res['val_dice'])
            if is_val_best:
                best_val_epoch = epoch
                best_final_val_state_dict = deepcopy(dl_ob.net.state_dict())

                if args.is_save_val_net:
                    taskpth_store_dir = os.path.join(args.store_meta_test, f'taskid_{task_idx}')
                    if not os.path.exists(taskpth_store_dir):
                        utils.mkdir(taskpth_store_dir)
                    torch.save(best_final_val_state_dict,os.path.join(taskpth_store_dir,f'best_epoch_for_val_meta_epoch_{meta_epoch}.pth' ))
                
            with open(str(metric_dir), 'a+') as f:
                csv_write = csv.writer(f, delimiter=',')
                data_row = [task_idx, epoch, res['train_loss'], res['train_dice'], res['val_dice'],  res['val_iou'], res['accuracy'],res['f1_score'], res['recall'],res['precision'], best_val, best_val_epoch]
                csv_write.writerow(data_row)

        # 进行两次test, 一次是最好的epoch, 一次是最好的middle_epoch(一次epoch,进行多次验证)
        # test twice, one is best epoch, one is middle_epoch( exe multi validation per dl_epoch)
        dl_ob.net.load_state_dict(best_final_val_state_dict)
        res_test = dl_ob.val(epoch, test_loader)

        test_values_all_task.append(list(res_test.values()) + [best_val, best_val_epoch])

        with open(str(metric_dir), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            data_row = ['finaltest', 'xx','xx','xx', res_test['val_dice'], res_test['val_iou'], res_test['accuracy'],res_test['f1_score'], res_test['recall'],res_test['precision'],best_val, best_val_epoch]
            csv_write.writerow(data_row)

        if args.is_mid_val and args.is_save_val_net:
            dl_ob.net.load_state_dict(torch.load(args.taskpth_mid_epoch_store_dir))
            res_test_mid = dl_ob.val(epoch, test_loader)

            test_values_all_task_mid.append(list(res_test_mid.values()) + [res['mid_dl_epoch_best_pred'], res['best_epoch_mid']])

            with open(metric_dir, 'a+') as f:
                csv_write = csv.writer(f, delimiter=',')
                data_row = ['finaltest_mid_epoch', 'xx','xx','xx', res_test_mid['val_dice'], res_test_mid['val_iou'], res_test_mid['accuracy'],res_test_mid['f1_score'], res_test_mid['recall'],res_test_mid['precision'], res['mid_dl_epoch_best_pred'], res['best_epoch_mid']]
                csv_write.writerow(data_row)
            with open(args.mid_dl_epoch_metric_dir, 'a+') as f:
                csv_write = csv.writer(f, delimiter=',')
                data_row = ['finaltest_mid_epoch', 'xx','xx','xx', res_test_mid['val_dice'], res_test_mid['val_iou'], res_test_mid['accuracy'],res_test_mid['f1_score'], res_test_mid['recall'],res_test_mid['precision'], res['mid_dl_epoch_best_pred'], res['best_epoch_mid']]
                csv_write.writerow(data_row)
            
    all_tasks_ave = np.around(np.mean(test_values_all_task, axis=0), 4).tolist()
    if args.is_mid_val:
        all_tasks_ave_mid = np.around(np.mean(test_values_all_task_mid, axis=0), 4).tolist()
    else:
        all_tasks_ave_mid = [0.0]

    with open(metric_dir, 'a+') as f:
        csv_write = csv.writer(f, delimiter=',')
        data_row = ['final_ave_test', 'xx','xx','xx'] + all_tasks_ave
        csv_write.writerow(data_row)

        data_row = ['mid_final_ave_test', 'xx','xx','xx'] + all_tasks_ave_mid
        csv_write.writerow(data_row)
 
    with open(args.mid_dl_epoch_metric_dir, 'a+') as f:
        csv_write = csv.writer(f, delimiter=',')
        csv_write = csv.writer(f, delimiter=',')
        data_row = ['final_ave_test', 'xx','xx','xx'] + all_tasks_ave
        csv_write.writerow(data_row)

        data_row = ['mid_final_ave_test', 'xx','xx','xx'] + all_tasks_ave_mid
        csv_write.writerow(data_row)

    return [round(item / task_num, 4) for item in val_dice_all_epoch], [round(item / task_num, 4) for item in val_dice_all_epoch_mid],all_tasks_ave, all_tasks_ave_mid
                
def run_epoch(epoch, model, train_loader, val_loader, test_loader, meta_epoch=0):
    res = {}
    res_train = model.train(epoch, train_loader, val_loader, test_loader, meta_epoch)

    res_val = model.val(epoch, val_loader)
    res.update(res_train)
    res.update(res_val)
    return res

def k_fold():
    pass



class Trainer():
    def __init__(self, args, test_data_ls, meta_epoch):
        pass
