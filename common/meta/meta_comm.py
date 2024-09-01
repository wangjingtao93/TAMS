import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher

from common.meta.gbml import GBML
from common.evl.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff
from common.evl.csdn_dice_loss import BCEDiceLoss
import common.evl.csdn_metrics as csdn_metric
class iMAML(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self._init_opt()
        self.lamb = 100
        self.n_cg = 1
        # if self.args.n_classes == 1:
        #     self.criterion = nn.BCEWithLogitsLoss()
        # else:
        #     self.criterion = nn.CrossEntropyLoss()
        # return None

        self.criterion = BCEDiceLoss()

    @torch.enable_grad()
    def inner_loop(self, fmodel, diffopt, img, mask, step):

        output = fmodel(img)
        inner_loss = self.criterion(output, mask)
        diffopt.step(inner_loss)

        # self.writer.add_scalar('inner_loss', inner_loss, step, walltime=None)

        return None

    @torch.no_grad()
    def cg(self, in_grad, outer_grad, params):
        x = outer_grad.clone().detach()
        r = outer_grad.clone().detach() - self.hv_prod(in_grad, x, params)
        p = r.clone().detach()
        for i in range(self.n_cg):
            Ap = self.hv_prod(in_grad, p, params)
            alpha = (r @ r) / (p @ Ap)
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = (r_new @ r_new) / (r @ r)
            p = r_new + beta * p
            r = r_new.clone().detach()
        return self.vec_to_grad(x)

    def vec_to_grad(self, vec):
        pointer = 0
        res = []
        for param in self.net.parameters():
            num_param = param.numel()
            res.append(vec[pointer:pointer + num_param].view_as(param).data)
            pointer += num_param
        return res

    @torch.enable_grad()
    def hv_prod(self, in_grad, x, params):
        hv = torch.autograd.grad(in_grad, params, retain_graph=True, grad_outputs=x)
        vec = []
        for param in hv:
            vec.append(param.reshape(-1))
        hv = torch.cat(vec).detach()
        # hv = torch.nn.utils.parameters_to_vector(hv).detach()
        # precondition with identity matrixparameters
        return hv / self.lamb + x

    def outer_loop(self, batch, is_train):
        # 一个batch的tasks送进去[meta_size=4, image_batch_size=ways*shot/query, channel, height, weidth]=[4,2*4,1,512,512]
        S_imgs, S_masks, Q_imgs, Q_masks = self.unpack_batch(batch)

        loss_log = 0
        dice_log = 0
        grad_list = []
        loss_list = []

        for (S_img, S_mask, Q_img, Q_mask) in zip(S_imgs, S_masks, Q_imgs, Q_masks):
            
            with higher.innerloop_ctx(self.net, self.inner_optimizer, track_higher_grads=False) as (
                    fmodel, diffopt):

                # support set
                for step in range(self.args.n_inner):  # n_inner=150
                    self.inner_loop(fmodel, diffopt, S_img, S_mask, step)

                S_out = fmodel(S_img)
                in_loss = self.criterion(S_out, S_mask)
                             
                # query set
                Q_out = fmodel(Q_img)
                outer_loss =  self.criterion(Q_out, Q_mask)
                loss_log += outer_loss.item() / self.meta_size

                with torch.no_grad():
                    dice_log += csdn_metric.dice_coef(Q_out, Q_mask) / self.meta_size
                    
                if is_train:
                    params = list(fmodel.parameters(time=-1))
                    in_grad = torch.nn.utils.parameters_to_vector(
                        torch.autograd.grad(in_loss, params, create_graph=True))
                    outer_grad = torch.nn.utils.parameters_to_vector(torch.autograd.grad(outer_loss, params))
                    implicit_grad = self.cg(in_grad, outer_grad, params)
                    grad_list.append(implicit_grad)
                    loss_list.append(outer_loss.item())

        if is_train:
            self.outer_optimizer.zero_grad()
            weight = torch.ones(len(grad_list))
            weight = weight / torch.sum(weight)
            grad = mix_grad(grad_list, weight)
            grad_log = apply_grad(self.net, grad)
            self.outer_optimizer.step()

            return loss_log, dice_log, grad_log
        else:
            return loss_log, dice_log
        
def apply_grad(model, grad):
    '''
    assign gradient to model(nn.Module) instance. return the norm of gradient
    '''
    grad_norm = 0
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
        grad_norm += torch.sum(g ** 2)
    grad_norm = grad_norm ** (1 / 2)
    return grad_norm.item()


def mix_grad(grad_list, weight_list):
    '''
    calc weighted average of gradient
    '''
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    return mixed_grad