import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher

from common.meta.gbml import GBML
from common.evl.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff


class iMAML(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self._init_opt()
        self.lamb = 100
        self.n_cg = 1
        if self.args.n_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        return None

    @torch.enable_grad()
    def inner_loop(self, fmodel, diffopt, train_input, train_target, step):

        train_logit = fmodel(train_input)
        # inner_loss = bce_dice_loss(train_logit, train_target)
        # inner_loss = self.criterion(train_logit, train_target) \
        #              + dice_loss(F.softmax(train_logit, dim=1).float(),
        #                          F.one_hot(train_target, fmodel.n_classes).permute(0, 3, 1, 2).float(),
        #                          multiclass=True)

        # cv2.imwrite('train_target.png', train_target[0,...].float().cpu().numpy() * 255)
        # cv2.imwrite('train_input.png', train_input[0,0,...].float().cpu().numpy() * 255)

        if self.net.n_classes == 1:
            train_target_unsqu = train_target.unsqueeze(dim=1)
            dice_loss_score = dice_loss(torch.sigmoid(train_logit).float(), train_target_unsqu.float(),
                                        multiclass=False)
            inner_loss = self.criterion(train_logit, train_target_unsqu.float()) + dice_loss_score
        else:
            dice_loss_score = dice_loss(F.softmax(train_logit, dim=1).float(),
                                        F.one_hot(train_target[:, 0, ...], self.net.n_classes).permute(0, 3, 1,
                                                                                                           2).float(),
                                        multiclass=True)
            inner_loss = self.criterion(train_logit, train_target[:, 0, ...]) + dice_loss_score

        # self.writer.add_scalar('inner_loss', inner_loss, step, walltime=None)

        diffopt.step(inner_loss)

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
        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(batch)

        loss_log = 0
        dice_log = 0
        grad_list = []
        loss_list = []

        for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs,
                                                                        test_targets):
            
            with higher.innerloop_ctx(self.net, self.inner_optimizer, track_higher_grads=False) as (
                    fmodel, diffopt):

                # support set
                for step in range(self.args.n_inner):  # n_inner=150
                    self.inner_loop(fmodel, diffopt, train_input, train_target, step)

                train_logit = fmodel(train_input)
                # in_loss = bce_dice_loss(train_logit, train_target)
                # in_loss = self.criterion(train_logit, train_target) \
                #           + dice_loss(F.softmax(train_logit, dim=1).float(),
                #                       F.one_hot(train_target, fmodel.n_classes).permute(0, 3, 1, 2).float(),
                #                       multiclass=True)
                if self.net.n_classes == 1:
                    train_target_unsqu = train_target.unsqueeze(dim=1)
                    dice_loss_score = dice_loss(torch.sigmoid(train_logit).float(), train_target_unsqu.float(),
                                                multiclass=False)
                    in_loss = self.criterion(train_logit, train_target_unsqu.float()) + dice_loss_score
                else:
                    train_target_unsqu = train_target.unsqueeze(dim=1)
                    dice_loss_score = dice_loss(F.softmax(train_logit, dim=1).float(),
                                                F.one_hot(train_target[:, 0, ...], self.net.n_classes).permute(0, 3,
                                                                                                                   1,
                                                                                                                   2).float(),
                                                multiclass=True)
                    in_loss = self.criterion(train_logit, train_target[:, 0, ...]) + dice_loss_score
                
                
                # query set
                test_logit = fmodel(test_input)

                # outer_loss = bce_dice_loss(test_logit, test_target)
                # outer_loss = self.criterion(test_logit, test_target) \
                #              + dice_loss(F.softmax(test_logit, dim=1).float(),
                #                          F.one_hot(test_target, fmodel.n_classes).permute(0, 3, 1, 2).float(),
                #                          multiclass=True)
                if self.net.n_classes == 1:
                    test_target_unsqu = test_target.unsqueeze(dim=1)
                    dice_loss_score = dice_loss(torch.sigmoid(test_logit).float(), test_target_unsqu.float(),
                                                multiclass=False)
                    outer_loss = self.criterion(test_logit, test_target_unsqu.float()) + dice_loss_score
                else:
                    dice_loss_score = dice_loss(F.softmax(test_logit, dim=1).float(),
                                                F.one_hot(test_target[:, 0, ...], self.net.n_classes).permute(0, 3,
                                                                                                                  1,
                                                                                                                  2).float(),
                                                multiclass=True)
                    outer_loss = self.criterion(test_logit, test_target[:, 0, ...]) + dice_loss_score
               
                loss_log += outer_loss.item() / self.meta_size

                # out_cut = np.copy(test_logit.data.cpu().numpy())
                # out_cut[np.nonzero(out_cut < 0.3)] = 0.0  # threshold
                # out_cut[np.nonzero(out_cut >= 0.3)] = 1.0

                with torch.no_grad():
                    # dice_log += dice_coef(out_cut, test_target.data.cpu().numpy()).item() / self.meta_size
                    # masks_pred = F.one_hot(test_logit.data.cpu().argmax(dim=1), fmodel.n_classes).permute(0, 3, 1,
                    #                                                                                       2).float()
                    # # compute the Dice score, ignoring background
                    # masks_true = F.one_hot(test_target.data.cpu(), fmodel.n_classes).permute(0, 3, 1, 2).float()
                    # dice_log = multiclass_dice_coeff(masks_pred[:, 1:, ...], masks_true[:, 1:, ...],
                    #                                  reduce_batch_first=False)
                    # torch.save(self.net.state_dict(), './result/iMAML/test_model_150updates_pth')
                    # # for i in range(masks_pred.shape[0]):
                    # masks_pred_save = torch.softmax(masks_pred, dim=0).argmax(dim=1).float().cpu().numpy() * 255
                    #
                    # masks_true_save = torch.softmax(masks_true, dim=0).argmax(dim=1).float().cpu().numpy() * 255
                    # image_ori = test_input.permute(0, 2, 3, 1).float().cpu().numpy() * 255
                    # for i in range(masks_pred_save.shape[0]):
                    #
                    #     image_save =  np.concatenate((masks_true_save[i],masks_pred_save[i]), axis=1)
                    #     save_path = 'result/iMAML/oct_layer/' + str(i)+'.png'
                    #     cv2.imwrite(save_path, image_save)
                    #
                    #     image_ori_save_path = 'result/iMAML/oct_layer/image_' + str(i)+'.png'
                    #     cv2.imwrite(image_ori_save_path, image_ori[i])

                    if self.net.n_classes == 1:
                        mask_pred = (torch.sigmoid(test_logit.data.cpu()) > 0.5).float()
                        # compute the Dice score
                        dice_log += dice_coeff(mask_pred[:, 0, ...], test_target.data.cpu().float(),
                                               reduce_batch_first=False) / self.meta_size
                    else:
                        mask_true = F.one_hot(test_target.data.cpu(), self.net.n_classes).permute(0, 3, 1,
                                                                                                      2).float()
                        mask_pred = F.one_hot(test_logit.data.cpu().argmax(dim=1), self.net.n_classes).permute(0, 3,
                                                                                                                   1,
                                                                                                                   2).float()
                        # compute the Dice score, ignoring background
                        dice_log += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                          reduce_batch_first=False) / self.meta_size
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