import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
from tqdm import tqdm
from copy import deepcopy

from common.meta.gbml import GBML
from common.evl.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff
from common.evl.csdn_dice_loss import BCEDiceLoss
import common.evl.csdn_metrics as csdn_metric

class MAML(GBML):
    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self._init_opt()
        self.args=args

    def train(self, db, epoch):
        self.net.train()
        criterion = BCEDiceLoss()

        qry_dice_all_tasks = []# 所有batch tasks的query set 的dice
        qry_loss_all_tasks = []

        tqdm_train = tqdm(db)

        for batch_idx, batch in enumerate(tqdm_train, 1):
            support_x, support_y, query_x, query_y = self.unpack_batch(batch)
            task_num, setsz, c_, h, w = support_x.size()
            querysz = query_x.size(1)

            # Initialize the inner optimizer to adapt the parameters to
            # the support set.
            # 每一个epoch都实例化一个，不用从上一个epoch开始
            # 为什么呢
            inner_opt = torch.optim.SGD(self.net.parameters(), lr=self.args.inner_lr)

            qry_losses = []
            qry_dice_list=[] # 存储一个batch task，query set的dice
            self.outer_optimizer.zero_grad()

            for i in range(task_num):
                with higher.innerloop_ctx(self.net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                    # Optimize the likelihood of the support set by taking
                    # gradient steps w.r.t. the model's parameters.
                    # This adapts the model's meta-parameters to the task.
                    # higher is able to automatically keep copies of
                    # your network's parameters as they are being updated.
                    for _ in range(self.args.n_inner):
                        spt_logits = fnet(support_x[i])
                        spt_loss = criterion(spt_logits, support_y[i])

                        diffopt.step(spt_loss)
                    # The final set of adapted parameters will induce some
                    # final loss and diceuracy on the query dataset.
                    # These will be used to update the model's meta-parameters.
                    qry_logits = fnet(query_x[i])
                    qry_loss = criterion(qry_logits, query_y[i])
                    qry_losses.append(qry_loss.detach().cpu())
                
                    with torch.no_grad():
                        dice = csdn_metric.dice_coef(qry_logits, query_y[i])      
                    qry_dice_list.append(dice)
                    
                    # Update the model's meta-parameters to optimize the query
                    # losses across all of the tasks sampled in this batch.
                    # This unrolls through the gradient steps.
                    qry_loss.backward()  # 为什么要在这呢

            
            self.outer_optimizer.step()
            qry_losses = sum(qry_losses) / task_num

            # qry_dscs = 100. * sum(qry_dscs) / task_num # .* 和*有什么区别吗？没有吧
            dice_ave = sum(qry_dice_list) / task_num

            qry_dice_all_tasks.append(dice_ave)
            qry_loss_all_tasks.append(qry_losses)

            tqdm_train.set_description('Meta Training_Tasks Epoch {}, batch_idx {}, dice={:.4f}, , Loss={:.4f}'.format(epoch, batch_idx, dice_ave.item(), qry_losses))

        ave_qry_dice_all_tasks = round(np.array(qry_dice_all_tasks).mean(), 4) 
        ave_qry_loss_all_tasks = round(np.array(qry_loss_all_tasks).mean(), 4) 

        # 返回：1.所有batch tasks的query set 的dice的均值.2.所有任务平均loss
        return ave_qry_loss_all_tasks, ave_qry_dice_all_tasks
    

    def val(self, db, epoch):
        val_net = deepcopy(self.net)
        val_net.train()     
        criterion = BCEDiceLoss()

        qry_losses = []
        qry_dice_list=[] # 所有测试任务
        qry_dice_inner = [0 for i in range(self.args.n_inner)] # 任务内循环每一次梯度更新，均值

        tqdm_val = tqdm(db)
        for batch_idx, batch in enumerate(tqdm_val, 1):
            support_x, support_y, query_x, query_y = self.unpack_batch(batch)
            task_num, setsz, c_, h, w = support_x.size()
            querysz = query_x.size(1) 

            n_inner_iter = self.args.n_inner
            inner_opt = torch.optim.SGD(val_net.parameters(), lr=self.args.inner_lr)

             # track_higher_grads=False,随着innerloop加深，不会增加内存使用
            for i in range(task_num):
                with higher.innerloop_ctx(val_net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                    # Optimize the likelihood of the support set by taking
                    # gradient steps w.r.t. the model's parameters.
                    # This adapts the model's meta-parameters to the task.
                    for k in range(n_inner_iter):
                        spt_logits = fnet(support_x[i])
                        spt_loss = criterion(spt_logits, support_y[i])
                        diffopt.step(spt_loss)
                        
                        qry_logits = fnet(query_x[i])
                        
                        with torch.no_grad():
                            dice = csdn_metric.dice_coef(qry_logits, query_y[i])
                        
                        qry_dice_inner[k] += dice
                        
                    # The query loss and acc induced by these parameters.
                    # qry_logits = fnet(query_x[i]).detach() #放到循环里，记录每一次梯度更新
                    qry_loss = criterion(qry_logits, query_y[i])
                    qry_losses.append(qry_loss.detach().cpu()) 
                    qry_dice_list.append(dice)

            del val_net

            loss = round(np.mean(np.array(qry_losses)), 4)
            dice_ave = round(np.array(qry_dice_list).mean(),4)
            std = np.array(qry_dice_list).std()

            tqdm_val.set_description('Val_Tasks Epoch {}, dice={:.4f}, queryloss {:.4f}'.format(epoch, dice_ave, loss))
            
            # return [dice_ave, std, list(map(lambda x: round(x/self.args.n_val_tasks, 4),qry_dice_inner))]

            return loss, dice_ave