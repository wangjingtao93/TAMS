import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.evl.dice_score import multiclass_dice_coeff, dice_coeff

import torch.nn as nn
from torch import Tensor


def evaluate(net, dataloader, device, flag='train'):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    if flag == 'train':
        desc_name = 'Train round'
    elif 'val' in flag:
        desc_name = 'Val round'
        
    else:
        desc_name = 'Test round'

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc=desc_name, unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        
        if torch.sum(mask_true) == 0 and flag != 'train' and batch.shape[0] == 1:
            num_val_batches -= 1
            continue
        
        # val 去黑图
        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred[:, 0, ...], mask_true.float(), reduce_batch_first=False)
            else:
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                            reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


# 计算单个batch的dsc
def single_batch_dsc(logits, label, n_classes = 0):
    assert n_classes != 0
    with torch.no_grad():
        if n_classes == 1:
            mask_pred = (torch.sigmoid(logits.data.cpu()) > 0.5).float()
            # compute the Dice score
            dice_score = dice_coeff(mask_pred[:, 0, ...], label.data.cpu().float(), reduce_batch_first=False)
        else:
            mask_true = F.one_hot(label.data.cpu(), n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(logits.data.cpu().argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
    return dice_score

# 和上一个好像一样
# 计算一个batch的dice
def dice_one_batch(mask_pred: Tensor, mask_true: Tensor, class_num = 1):
    # convert to one-hot format
    dice_score = 0.0
    if class_num == 1:
        mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
        # compute the Dice score
        dice_score += dice_coeff(mask_pred[:, 0, ...], mask_true.float(), reduce_batch_first=False)
    else:
        mask_true = F.one_hot(mask_true[:,0,...], class_num).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), class_num).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                    reduce_batch_first=False)
        
    return round(dice_score.cpu().item(),5)

# 可以计算std
# can calculate std
def test_evl(net, dataloader, device, flag='test'):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0.0
    count = 0

    if flag == 'train':
        desc_name = 'Train round'
    elif 'val' in flag:
        desc_name = 'Val round'
        
    else:
        desc_name = 'Test round'

    # iterate over the validation set
    dice_list =[] # record every batch val dice
    for batch in tqdm(dataloader, total=num_val_batches, desc=desc_name, unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        count += 1
        
        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score_single_batch = dice_coeff(mask_pred[:, 0, ...], mask_true.float(), reduce_batch_first=False)
                dice_list.append(dice_score_single_batch.cpu().item())
                dice_score += dice_score_single_batch.cpu().item()
            else:
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score_single_batch = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                            reduce_batch_first=False)
                dice_list.append(dice_score_single_batch.cpu().item())
                dice_score += dice_score_single_batch
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return round(dice_score / num_val_batches, 5), dice_list
