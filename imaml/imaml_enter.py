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

from common.meta.meta_comm import iMAML
import common.utils as utils
from dl.dl_enter import dl_enter
from common.record_result.record_meta_res import RecordMetaResult


def imaml_enter(args ,train_dataloader_ls, val_dataloader_ls, test_data_ls):

    record_res = RecordMetaResult(args)

    model = iMAML(args)


    for epoch in range(args.n_meta_epoch):
        model.adjust_outer_learning_rate(epoch)
        train_loader_len = len(train_dataloader_ls[0])
        val_loader_len = len(val_dataloader_ls[0])
        res = run_epoch(epoch, args, model, train_loader_len, val_loader_len, zip(train_dataloader_ls[0], train_dataloader_ls[1]), zip(val_dataloader_ls[0], val_dataloader_ls[1]), test_data_ls)

        res['current_outer_lr'] =  model.outer_optimizer.state_dict()['param_groups'][0]['lr']

        record_res.write_res(res)

def run_epoch(epoch, args, model, train_loder_len, test_loader_len, train_loader, val_loader, test_data_ls):
    res = OrderedDict()
    print('Epoch {}'.format(epoch))
    train_loss, train_dice, train_grad = train(args, model, train_loader, train_loder_len)
    val_loss, val_dice = valid(args, model, val_loader, test_loader_len)

    res['meta_epoch'] = epoch
    res['meta_train_loss'] = train_loss
    res['meta_train_dice'] = train_dice
    res['meta_train_grad'] = train_grad

    res['meta_val_loss'] = val_loss
    res['meta_val_dice'] = val_dice


    # store every meta epoch, meta parameters
    meta_epoch_model = os.path.join(args.store_dir, 'save_meta_pth',f'meta_epoch_{epoch}.pth')
    torch.save(model.net.state_dict(), meta_epoch_model)

    meta_test_Q_values_all_dl_epoch, meta_test_Q_values_all_dl_epoch_mid, meta_final_ave_tasks, meta_final_ave_tasks_mid=dl_enter(args, test_data_ls, epoch)

    res['meta_test_Q_values'] = meta_test_Q_values_all_dl_epoch
    res['meta_test_Q_values_mid'] = meta_test_Q_values_all_dl_epoch_mid
    res['meta_final_ave_tasks'] = meta_final_ave_tasks
    res['meta_final_ave_tasks_mid'] = meta_final_ave_tasks_mid
    return res


def train(args, model, dataloader, loader_len):
    loss_list = []
    dice_list = []
    grad_list = []

    with tqdm(dataloader, total=loader_len) as pbar:

        for batch_idx, batch in enumerate(pbar):
            loss_log, dice_log, grad_log = model.outer_loop(batch, is_train=True)

            loss_list.append(loss_log)
            dice_list.append(dice_log)
            grad_list.append(grad_log)
            pbar.set_description(
                'batch_idx = {:.0f} || loss = {:.4f} || tdice={:.4f} || grad={:.4f}'.format(batch_idx,
                                                                                            np.mean(loss_list),
                                                                                            np.mean(dice_list),
                                                                                            np.mean(grad_list)))

    # 一个list包含所有task_batch的值
    loss = np.round(np.mean(loss_list), 4)
    dice = np.round(np.mean(dice_list), 4)
    grad = np.round(np.mean(grad_list), 4)

    return loss, dice, grad


@torch.no_grad()
def valid(args, model, dataloader,loader_len):
    loss_list = []
    dice_list = []
    with tqdm(dataloader, total=loader_len) as pbar:
        for batch_idx, batch in enumerate(pbar):
            loss_log, dice_log = model.outer_loop(batch, is_train=False)

            loss_list.append(loss_log)
            dice_list.append(dice_log)
            pbar.set_description('loss = {:.4f} || vdice={:.4f}'.format(np.mean(loss_list), np.mean(dice_list)))
            # if batch_idx >= args.num_valid_batches:
            #     break

    loss = np.round(np.mean(loss_list), 4)
    dice = np.round(np.mean(dice_list), 4)

    return loss, dice