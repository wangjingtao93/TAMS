import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')
import numpy as np
import torch
import time
from collections import OrderedDict
from tqdm import tqdm
import os
import csv
from torch.utils.tensorboard import SummaryWriter

from common.meta.maml_meta import MAML
import common.utils as utils
from dl.dl_enter import dl_enter
from common.record_result.record_meta_res import RecordMetaResult


def maml_enter(args ,train_dataloader_ls, val_dataloader_ls, test_data_ls):
        
    record_res = RecordMetaResult(args)

    model = MAML(args)

    for epoch in range(args.n_meta_epoch):
        res = run_epoch(epoch, args, model, zip(train_dataloader_ls[0], train_dataloader_ls[1]), zip(val_dataloader_ls[0], val_dataloader_ls[1]), test_data_ls)
        res['current_outer_lr'] =  model.outer_optimizer.state_dict()['param_groups'][0]['lr']

        record_res.write_res(res)

def run_epoch(epoch, args, model, train_loader, val_loader, test_data_ls):
    res = OrderedDict()
    print('Epoch {}'.format(epoch))

    train_loss, train_dice, = model.train(train_loader, epoch)

    val_loss, val_dice = model.val(val_loader, epoch)


    res['meta_epoch'] = epoch
    res['meta_train_loss'] = train_loss
    res['meta_train_dice'] = train_dice
    res['meta_train_grad'] = 0

    res['meta_val_loss'] = val_loss
    res['meta_val_dice'] = val_dice

    # 保存每一个epoch的元参数
    meta_epoch_model = os.path.join(args.store_dir, 'save_meta_pth',f'meta_epoch_{epoch}.pth')
    torch.save(model.net.state_dict(), meta_epoch_model)

    meta_test_Q_values_all_dl_epoch, meta_test_Q_values_all_dl_epoch_mid, meta_final_ave_tasks, meta_final_ave_tasks_mid=dl_enter(args, test_data_ls, epoch)


    res['meta_test_Q_values'] = meta_test_Q_values_all_dl_epoch
    res['meta_test_Q_values_mid'] = meta_test_Q_values_all_dl_epoch_mid
    res['meta_final_ave_tasks'] = meta_final_ave_tasks
    res['meta_final_ave_tasks_mid'] = meta_final_ave_tasks_mid

    return res
