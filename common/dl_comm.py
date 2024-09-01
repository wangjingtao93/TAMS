import os
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision
import segmentation_models_pytorch as seg

from tqdm import tqdm
from copy import deepcopy
# from miseval import evaluate as mis_evl
# from net.Att_Unet import Att_Unet
from model.unet import UNet
from model.transUNet.networks.get_trans_net import get_trans
from common.evl.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff
from common.evl.evaluate import evaluate,test_evl, dice_one_batch
import common.evl.metrics as smp
import common.evl.csdn_metrics as csdn_metric

import common.utils as utils
from  common.evl.calculate_metrics import calculate_metrics
from common.evl.csdn_dice_loss import BCEDiceLoss

class dl_comm():
    def __init__(self, args):
        self.args = args
        self.global_step = 0
        self._init_net()
        self._init_opt()
        self._init_criterion()
        
        # record mid_dl epcoh best_pred
        self.mid_dl_epoch_best_pred = 0.0
        self.mid_dl_epoch_best_epoch = 0



    def _init_net(self):

        if self.args.net == 'unet':
            self.get_unet()

        elif self.args.net == 'unet++':
            self.net = seg.UnetPlusPlus(
                encoder_name='resnet34',
                encoder_weights=None,
                in_channels=self.args.n_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.args.n_classes,                      # model output channels (number of classes in your dataset)
            )
        elif self.args.net == 'manet':
            self.net = seg.MAnet(
                encoder_name='resnet34',
                encoder_weights=None,
                in_channels=self.args.n_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.args.n_classes,                      # model output channels (number of classes in your dataset)
            )
        elif self.args.net == 'transUNet':
            self.net = get_trans(self.args)
        
        else:
            raise ValueError('Not implemented Net type')
    
        
        self.net.train()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        return None

    def _init_opt(self):
        self.modellrnew = self.args.dl_lr
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.dl_lr, momentum=0.9, weight_decay=1e-4)

    def _init_criterion(self):
        # if self.args.n_classes == 1:
        #     self.criterion = nn.BCEWithLogitsLoss()
        # else:
        #     self.criterion = nn.CrossEntropyLoss()

        self.criterion = BCEDiceLoss()

    def get_unet(self):
        self.net = UNet(n_channels=self.args.n_channels, n_classes=self.args.n_classes, bilinear=True)
        if self.args.load != '':
            self.net.load_state_dict(torch.load(self.args.load))
            print(f'++++++++++load {self.args.load}---------')

    # 所有的epoch都在里面
    def train(self, epoch, train_loader, val_loader, test_loader, meta_epoch):
        self.net.train()
        n_train = len(train_loader)

        sum_loss = 0.0
        train_dice_ls =[]
        val_dice_ls = [] # record every batch evl. 每个epoch会进行多次验证
        res = {}
        with tqdm(total=n_train, desc=f'Task_ID {self.args.index_fold}, Epoch {epoch}/{self.args.n_epoch}, lr {self.modellrnew}', unit='img') as pbar:
            for batch in train_loader:
                image, mask = batch['image'], batch['mask']
                image = image.to(device=self.device, dtype=torch.float32)
                mask = mask.to(device=self.device, dtype=torch.float32)

                output = self.net(image)
                loss = self.criterion(output, mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                sum_loss += loss.item()

                pbar.update(1)
                self.global_step += 1
                pbar.set_postfix(**{'loss (batch)': loss.item()}) 
                division_step = (n_train // self.args.n_mid_val) # every division step, evaluation 
                # if division_step > 0:
                
                if self.global_step % division_step == 0 and self.args.is_mid_val:
                    res_val = self.val(epoch, val_loader)
                    train_score = csdn_metric.dice_coef(output, mask)
                    print('Validation Dice score of all batch: {}'.format(res_val['val_dice']))
                    print(f'Train Dice score of one batch: {train_score}')
                    train_dice_ls.append(train_score)
                    val_dice_ls.append(res_val['val_dice'])

                    is_best = res_val['val_dice'] > self.mid_dl_epoch_best_pred
                    self.mid_dl_epoch_best_pred = max(self.mid_dl_epoch_best_pred, res_val['val_dice'])
                    if is_best:
                        self.mid_dl_epoch_best_epoch = epoch
                        if self.args.is_save_val_net:
                            torch.save(self.net.state_dict(),self.args.taskpth_mid_epoch_store_dir)

                    with open(str(self.args.mid_dl_epoch_metric_dir), 'a+') as f:
                        csv_write = csv.writer(f, delimiter=',')
                        data_row = [self.args.task_idx, epoch, 'xx', train_score, res_val['val_dice'], res_val['val_iou'], res_val['accuracy'],res_val['f1_score'], res_val['recall'],res_val['precision'], self.mid_dl_epoch_best_pred, self.mid_dl_epoch_best_epoch]
                        csv_write.writerow(data_row)
                if not self.args.is_mid_val:
                    train_dice_ls.append(csdn_metric.dice_coef(output, mask))
                    val_dice_ls.append(0.0)

        res['epoch'] = epoch
        res['train_loss'] = round((sum_loss / n_train), 5)
        res['train_dice'] = round(np.array(train_dice_ls).mean(), 5)
        res['val_dice_mid_epoch'] = val_dice_ls
        res['best_epoch_mid'] = self.mid_dl_epoch_best_epoch
        res['mid_dl_epoch_best_pred'] = self.mid_dl_epoch_best_pred
        
        return res

    def val(self, epoch, val_loader):
        self.net.eval()
        # val_score, val_dice_list = test_evl(self.net, val_loader, self.device, flag='val')

        n_val_batches = len(val_loader)
        val_score = 0.0
        iou_score = 0.0
        f1_score = 0.0
        accuracy = 0.0
        recall = 0.0
        precision = 0.0
        res = {}

        for batch in tqdm(val_loader, total=n_val_batches, desc=f'Epoch {epoch}/{self.args.n_epoch}, lr {self.modellrnew}', unit='batch', leave=False):
            
            image, mask = batch['image'], batch['mask']
            image = image.to(device=self.device, dtype=torch.float32)
            mask = mask.to(device=self.device, dtype=torch.long)

            with torch.no_grad():
                output = self.net(image)

                iou, dice= csdn_metric.iou_score(mask, output)
                iou_score += iou
                val_score += dice

                res_smp = self.smp_computer(output, mask)
                accuracy += res_smp['accuracy']
                f1_score += res_smp['f1_score']
                recall += res_smp['recall']
                precision += res_smp['precision']

         

                # # iou_score_1 += calculate_metrics(output[0,0,...], target[0,0,...], 'iou')
                # iou_score_1 += mis_evl(target, (torch.sigmoid(output) > 0.5).float(), metric="IoU")
                # f1_score_1 += calculate_metrics(output, target, 'f1')
                # # accuracy_1 += calculate_metrics(output, target, 'accuracy')
                # accuracy_1 += mis_evl(np.array(target), np.array((torch.sigmoid(output) > 0.5).float()), metric="ACC")
                # recall_1 += calculate_metrics(output, target, 'recall')
                # precision_1 += calculate_metrics(output, target, 'precision')
                # # val_score_1 += calculate_metrics(output, target, 'dice_coefficient')
                # val_score_1 += mis_evl(target, (torch.sigmoid(output) > 0.5).float(), metric="DSC")
                
                

        res['val_dice'] = round((val_score / n_val_batches).item(), 5)
        res['val_iou'] = round((iou_score / n_val_batches).item(), 5)
        res['accuracy'] = round((accuracy / n_val_batches).item(), 5)
        res['f1_score'] =  round((f1_score/ n_val_batches).item(), 5)
        res['recall'] = round((recall / n_val_batches).item(), 5)
        res['precision'] = round((precision / n_val_batches).item(), 5)

        self.net.train()

        return res
 
    def smp_computer(self, output, target):
        
        res = {}
        if self.args.n_classes == 1:
            metric_mod = 'binary'
        else:
            metric_mod = 'multilabel'

        # first compute statistics for true positives, false positives, false negative and
        # true negative "pixels"
        tp, fp, fn, tn = smp.get_stats(output, target.long(), mode=metric_mod, threshold=0.5)
        # iou_score += smp.iou_score(tp, fp, fn, tn, reduction="micro")
        res['f1_score'] = smp.f1_score(tp, fp, fn, tn, reduction="micro")
        res['accuracy'] = smp.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
        res['recall'] = smp.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        res['precision'] = smp.precision(tp, fp, fn, tn, reduction="micro")

        return res
    
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""

        if self.args.lr_mode == 'step':
            self.modellrnew = self.args.dl_lr * (0.1 ** (epoch // 10))# ** 乘方
        elif self.args.lr_mode == 'poly':
            self.modellrnew = self.args.dl_lr * (1 - epoch / self.args.n_epoch) ** 0.9
        else:
            raise ValueError('Unknown lr mode {}'.format(self.args.lr_mode))    
        
        print("lr:", self.modellrnew)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.modellrnew


            