import numpy as np
import torch
import torch.nn as nn
import os
from monai.networks.nets import SwinUNETR
# from net.Att_Unet import Att_Unet
from model.unet import UNet
import segmentation_models_pytorch as seg
from model.transUNet.networks.get_trans_net import get_trans
from model.transUNet.networks.get_trans_net import get_trans_for_meta

class GBML:
    '''
    Gradient-Based Meta-Learning
    '''

    def __init__(self, args):

        self.args = args
        self.meta_size = self.args.meta_size
        return None

    def _init_net(self):
        if self.args.net == 'unet':
            self.get_unet()
        elif self.args.net =='swinunetr':
             self.net = SwinUNETR(
                img_size=(self.args.dl_resize, self.args.dl_resize),
                in_channels=self.args.n_channels,
                out_channels=self.args.n_classes,
                # feature_size=48,
                # use_checkpoint=args.use_checkpoint,
                spatial_dims=2
            )
        elif self.args.net == 'manet':
            self.net = seg.MAnet(
                encoder_name='resnet34',
                encoder_weights=None,
                in_channels=self.args.n_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.args.n_classes,                      # model output channels (number of classes in your dataset)
            )
        elif self.args.net == 'unet++':
            self.net = seg.UnetPlusPlus(
                encoder_name='resnet34',
                encoder_weights=None,
                in_channels=self.args.n_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.args.n_classes,                      # model output channels (number of classes in your dataset)
            )
        elif self.args.net == 'transUNet':
            self.net_enc, self.net = get_trans_for_meta(self.args)

            for param in  self.net_enc.parameters():
                param.requires_grad = False
            self.net_enc.eval()
            self.net_enc.cuda()

        if self.args.meta_learner_load != '':
            if os.path.isfile(self.args.meta_learner_load):
                if self.args.net == 'transUNet':
                    self.transunet_copy_weights()
                else:
                    self.net.load_state_dict(torch.load(self.args.meta_learner_load))
                    print(f'++++++++++meta learner load {self.args.meta_learner_load}---------')
            else:
                print(f'waring -----+meta learner load is not exist {self.args.meta_learner_load}++++')

        self.net.train()
        self.net.cuda()
        return None

    def _init_opt(self):
        if self.args.inner_opt == 'SGD':  # 使用
            self.inner_optimizer = torch.optim.SGD(self.net.parameters(), weight_decay=1e-5, lr=self.args.inner_lr)
        elif self.args.inner_opt == 'Adam':
            self.inner_optimizer = torch.optim.Adam(self.net.parameters(), weight_decay=1e-5, lr=self.args.inner_lr,
                                                    betas=(0.0, 0.9))
        else:
            raise ValueError('Not supported inner optimizer.')
        if self.args.outer_opt == 'SGD':
            self.outer_optimizer = torch.optim.SGD(self.net.parameters(), weight_decay=1e-5, lr=self.args.outer_lr,
                                                   nesterov=True, momentum=0.9)
        elif self.args.outer_opt == 'Adam':  # 使用
            self.outer_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.outer_lr, weight_decay=1e-5)
        else:
            raise ValueError('Not supported outer optimizer.')

        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.outer_optimizer, step_size=10, gamma=0.5)

        return None

    def get_unet(self):
        self.net = UNet(n_channels=self.args.n_channels, n_classes=self.args.n_classes, bilinear=True)

    def unpack_batch(self, batch):

        device = torch.device('cuda')
        train_inputs, train_targets = batch[0]
        train_inputs = train_inputs.to(device=device, dtype=torch.float32)
        train_targets = train_targets.to(device=device, dtype=torch.float32)

        test_inputs, test_targets = batch[1]
        test_inputs = test_inputs.to(device=device, dtype=torch.float32)
        test_targets = test_targets.to(device=device, dtype=torch.float32)

        return train_inputs, train_targets, test_inputs, test_targets

    def inner_loop(self):
        raise NotImplementedError

    def outer_loop(self):
        raise NotImplementedError

    def lr_sched(self):
        self.lr_scheduler.step()
        return None

    # def load(self):
    #     path = self.args.load_path
    #     # path = "./result/iMAML/xirou_withouTrans/best_model.pth'"
    #     # self.net.load_state_dict(torch.load(path), strict=False)
    #     self.net.load_state_dict(torch.load(path))
    #     print('加载模型=', path)

    # def load_interrupt(self):
    #     path = self.args.load_interrupt_path
    #     checkpoint = torch.load(path)
    #     self.net.load_state_dict(checkpoint['model'])
    #     self.outer_optimizer.load_state_dict(checkpoint['out_optimizer'])
    #     self.inner_optimizer.load_state_dict(checkpoint['inner_optimizer'])
    #     begin_epoch = checkpoint['epoch']
    #     print('恢复中断模型=',path)
    #     return begin_epoch
    
    def save(self):
        path = os.path.join(self.store_dir, self.args.model_save_path)
        torch.save(self.net.state_dict(), path)

    def interrupt_save(self, epoch):
        state = {'model':self.net.state_dict(), 'out_optimizer':self.outer_optimizer.state_dict(), 'inner_optimizer':self.inner_optimizer.state_dict(), 'epoch':epoch}     
        path = os.path.join(self.store_dir, 'last_epoch_state.pth')
        torch.save(state, path)

    def adjust_outer_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""

        if self.args.lr_mode == 'step':
            self.modellrnew = self.args.outer_lr * (0.1 ** (epoch // 10))# ** 乘方
        elif self.args.lr_mode == 'poly':
            self.modellrnew = self.args.outer_lr * (1 - epoch / self.args.n_meta_epoch) ** 0.9
        else:
            raise ValueError('Unknown lr mode {}'.format(self.args.lr_mode))    
        
        print("lr:", self.modellrnew)
        for param_group in self.outer_optimizer.param_groups:
            param_group['lr'] = self.modellrnew    

    def transunet_copy_weights(self):
        state_dict = torch.load(self.args.meta_learner_load)
        # enc_state_dict = {k[len('transformer.'):]: v for k, v in state_dict.items() if k.startswith('transformer.')}
        dec_state_dict = {k[len('decoder.'):]: v for k, v in state_dict.items() if k.startswith('decoder.')}

        # 加载参数到net2
        missing_enc, unexpected_enc = self.net_enc.load_state_dict(state_dict, strict=False)
        if missing_enc:
            print("net_enc 缺少以下键:", missing_enc)

        if unexpected_enc:
            print("net_enc 存在以下意外键:", unexpected_enc)

        # 加载参数到net3
        missing_dec, unexpected_dec = self.net.load_state_dict(dec_state_dict, strict=False)
        if missing_dec:
            print("net 缺少以下键:", missing_dec)
        if unexpected_dec:
            print("net 存在以下意外键:", unexpected_dec)
