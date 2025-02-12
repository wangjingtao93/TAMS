import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/')

import os
import argparse
import logging

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import pandas as pd
from model.unet import UNet
from dl.v_20230530.dataloader import CSVBasicDataset
from common.evl.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff
from common.evl.evaluate import evaluate,test_evl, dice_one_batch
import common.evl.metrics as smp


from torch.utils.tensorboard import SummaryWriter
import common.utils as utils
from tqdm import tqdm
import time

import torch.nn.functional as F
from sklearn.model_selection import KFold, GroupKFold

def split_kfold(test_fold = 1):
    kf = GroupKFold(n_splits=4)
    dframe = pd.read_csv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-seg/DLCNV/data/linux/all_data.csv')

    k_fold_count = 0
    # 是否随机有待验证，不能随机
    # dframe = dframe.sample(int(dframe.shape[0] * args.use_trainset_percent))
    groups = list(dframe['Eye'])
    # for train_index, test_index in kf.split(dframe):
    for train_index, test_index in kf.split(dframe,groups=groups):
        k_fold_count += 1
        # 防止test_index_fold为空
        
        print('\n{} of kfold {} begin to run'.format(k_fold_count,kf.n_splits))

        train_df = dframe.iloc[train_index,:]
        test_df = dframe.iloc[test_index,:]
        val_df = test_df

        if test_fold == k_fold_count:
            return test_df

def metric_compu(mask_true, output):
    target = mask_true.unsqueeze(dim=1).round().long()
    # first compute statistics for true positives, false positives, false negative and
    # true negative "pixels"
    res = {}
    tp, fp, fn, tn = smp.get_stats(output, target, mode='multilabel', threshold=0.5)
    res['iou_score'] = smp.iou_score(tp, fp, fn, tn, reduction="micro")
    res['f1_score'] = smp.f1_score(tp, fp, fn, tn, reduction="micro")
    res['accuracy'] = smp.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
    res['recall'] = smp.recall(tp, fp, fn, tn, reduction="micro-imagewise")
    res['precision'] = smp.precision(tp, fp, fn, tn, reduction="micro")

    return res

def test():
    store_path = './res_tmp'
    f=open(os.path.join(store_path,"res.txt"),"w")
    net = UNet(n_channels=1, n_classes=1, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    net.eval()
    
    work_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/'

    test_set = CSVBasicDataset(csv_dir= work_path + 'data/bv1000_ocv_cnv/real_data/acc_eyes/linux/test.csv')

    # use_trainset_percent = 0.3
    # if use_trainset_percent != 1.0:     
    #     chosen_indices = range(len(test_set))
    #     chosen_indices,_ = utils.data_split(chosen_indices,use_trainset_percent)
    #     test_set = torch.utils.data.Subset(test_set, chosen_indices)
        

    # test_df = split_kfold(3)
    # test_set =  CSVBasicDataset(dataframe=test_df)


    n_test = len(test_set)

    # 3. Create data loaders batch_size 会影响测试结果
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, batch_size = 1, num_workers=4, pin_memory=True)
    dice_score = 0
    dice_list =[]
    count = 0
    
    res = {}
    res['iou_score'] = 0.0
    res['f1_score'] = 0.0
    res['accuracy'] = 0.0
    res['recall'] = 0.0
    res['precision'] = 0.0

    for batch in tqdm(test_loader, total=len(test_loader), desc='predict', unit='batch', leave=False):
        count += 1
        image, mask_true = batch['image'], batch['mask']
        cv2.imwrite(os.path.join(store_path,str(count)+'.jpg'),  np.asarray(image[0][0]*255.0))

        store_true = np.asarray(mask_true[0] * 255)
        
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        
        
        
        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)


            # res_tmp = metric_compu(mask_true.cpu(), mask_pred.cpu())
            # for key in res:
                # res[key] += res_tmp[key]


            store_pred = torch.sigmoid(mask_pred) > 0.5  
            store_pred = np.asarray(store_pred[0][0].cpu()*255)
            
            store_mask = np.concatenate([store_true, store_pred],axis=1)
            cv2.imwrite(os.path.join(store_path,str(count)+'_label.png'),  store_mask)
            
            # store_image = np.concatenate([image[0].permute(1,2,0)*255.0, mask_true[None] * 255 ], axis=1)
            # cv2.imwrite(store_image)
            
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score_single_batch = dice_coeff(mask_pred[:, 0, ...], mask_true.float(), reduce_batch_first=False)
                dice_list.append(np.round(np.asarray(dice_score_single_batch.cpu()), 3))
                dice_score += np.round(np.asarray(dice_score_single_batch.cpu()),3)
            else:
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score_single_batch = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                            reduce_batch_first=False)
                dice_list.append(round(np.asarray(dice_score_single_batch.cpu()), 3))
                dice_score += round(np.asarray(dice_score_single_batch.cpu()),3)
    
            # store_image = images[0, ...]
            # store_true = 

   
    for i, value in enumerate(dice_list):
        # f.write(str(i) + ' ' + str(value))
        f.write(str(value))
        f.write('\n')
    f.write (str(dice_score/len(test_loader)))
    f.close()

    # logging.info('Testing Dice score: {} ± {}'.format(test_score, std_value))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    


    parser.add_argument('--gpu', type=int, nargs='+', default=[4], help='2 = GPU.')
    parser.add_argument('--seed', type=int, default=1) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')

    parser.add_argument('--load', '-f', type=str,
                        default="/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/result/result_20230530/bv1000-oct-cnv/dl/unet/2023-04-05-18-13-13/max_dsc_2023-04-05-18-13-13.pth",
                        help='Load model from a .pth file')

    parser.add_argument('--save_path', '-st', type=str, default='tmp')
    parser.add_argument('--prefix', type=str, default='tmp')
    parser.add_argument('--remark', type=str, default='')
    parser.add_argument('--if_tichu', type=int, default=0, choices=[0, 1])
    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_args()
    utils.set_gpu(args.gpu)
    utils.set_seed(args.seed)
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    

    test()