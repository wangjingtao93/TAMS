
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')
import torch.nn as nn
from torchsummary import summary
import torch
from collections import OrderedDict
import cv2
from torchvision.transforms import Resize 
import torch.nn.functional as F
from functools import partial
import model.gen_net.mygen.transformer_part as transformer_part
# 定义降采样部分
class downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downsample, self).__init__()
        self.down = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.down(x)

# 定义上采样部分
class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out=False):
        super(upsample, self).__init__()
        self.up = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5) if drop_out else nn.Identity()
        )

    def forward(self, x):
        return self.up(x)


# 256*256
class lesionD_256(nn.Module):
    def __init__(self):
        super(lesionD_256, self).__init__()

        # 定义基本的卷积\bn\relu
        def base_Conv_bn_lkrl(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )

        D_dic = OrderedDict()
        in_channels = 6
        out_channels = 64
        for i in range(4):
            if i < 3:
                D_dic.update({'layer_{}'.format(i + 1): base_Conv_bn_lkrl(in_channels, out_channels, 2)})
            else:
                D_dic.update({'layer_{}'.format(i + 1): base_Conv_bn_lkrl(in_channels, out_channels, 1)})
            in_channels = out_channels
            out_channels *= 2
        D_dic.update({'last_layer': nn.Conv2d(512, 1, 4, 1, 1)})  # [batch,1,30,30]
        self.D_model = nn.Sequential(D_dic)

    def forward(self, x1, x2):
        in_x = torch.cat([x1, x2], dim=1)
        return self.D_model(in_x)

# ---------------------------------------------------------------------------------
# 256*256
class lesion_G_256(nn.Module):
    def __init__(self):
        super(lesion_G_256, self).__init__()
        # down sample
        self.trans_part = MAE_Feature()
        
        self.down_1 = nn.Conv2d(3, 64, 4, 2, 1)  # [batch,3,256,256]=>[batch,64,128,128]
        self.tcu_1 = ToC(196, 64, 128)
        for i in range(7):
            if i == 0:
                self.down_2 = downsample(64, 128)  # [batch,64,128,128]=>[batch,128,64,64]
                self.tcu_2 = ToC(196,128,64) 
                self.down_3 = downsample(128, 256)  # [batch,128,64,64]=>[batch,256,32,32]
                self.tcu_3 = ToC(196,256,32) 
                self.down_4 = downsample(256, 512)  # [batch,256,32,32]=>[batch,512,16,16]
                self.tcu_4 = ToC(196,512,16) 
                self.down_5 = downsample(512, 512)  # [batch,512,16,16]=>[batch,512,8,8]
                self.tcu_5 = ToC(196,512,8) 
                self.down_6 = downsample(512, 512)  # [batch,512,8,8]=>[batch,512,4,4]
                self.tcu_6 = ToC(196,512,4) 
                self.down_7 = downsample(512, 512)  # [batch,512,4,4]=>[batch,512,2,2]
                self.tcu_7 = ToC(196,512,2) 
                self.down_8 = downsample(512, 512)  # [batch,512,2,2]=>[batch,512,1,1]
                self.tcu_8 = ToC(196,512,1)

        for i in range(7):
            if i == 0:
                self.up_1 = upsample(512, 512)  # [batch,512,1,1]=>[batch,512,2,2]
                self.up_2 = upsample(1024, 512, drop_out=True)  # [batch,1024,2,2]=>[batch,512,4,4]
                self.up_3 = upsample(1024, 512, drop_out=True)  # [batch,1024,4,4]=>[batch,512,8,8]
                self.up_4 = upsample(1024, 512)  # [batch,1024,8,8]=>[batch,512,16,16]
                self.up_5 = upsample(1024, 256)  # [batch,1024,16,16]=>[batch,256,32,32]
                self.up_6 = upsample(512, 128)  # [batch,512,32,32]=>[batch,128,64,64]
                self.up_7 = upsample(256, 64)  # [batch,256,64,64]=>[batch,64,128,128]

        self.last_Conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.init_weight()

    def init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_in')
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)

    # 固定参数，直接加载latent_feature
    def forward_with_trans(self, x, trans_feature):
        
        # torch_resize = Resize([224,224]) # 定义Resize类对象
        # im1_resize = torch_resize(x)
        # trans_feature = self.trans_part.forward_features(im1_resize) #  [N, 197,1024]

        # down
        # down_1 = (self.down_1(x)  + self.tcu_1(trans_feature)) / 2
        # down_2 = (self.down_2(down_1) + self.tcu_2(trans_feature)) / 2
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = (self.down_3(down_2) + self.tcu_3(trans_feature)) / 2
        # down_3 = self.down_3(down_2)
        # down_4 = (self.down_4(down_3) + self.tcu_4(trans_feature)) / 2
        # down_5 = (self.down_5(down_4) + self.tcu_5(trans_feature)) / 2
        # down_6 = (self.down_6(down_5) + self.tcu_6(trans_feature)) / 2
        # down_7 = (self.down_7(down_6) + self.tcu_7(trans_feature)) / 2
        # down_8 = (self.down_8(down_7) + self.tcu_8(trans_feature)) / 2
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        down_7 = self.down_7(down_6)
        down_6 = self.down_6(down_5) 
        down_8 = self.down_8(down_7)

        # up
        up_1 = self.up_1(down_8)
        up_2 = self.up_2(torch.cat([up_1, down_7], dim=1))
        up_3 = self.up_3(torch.cat([up_2, down_6], dim=1))
        up_4 = self.up_4(torch.cat([up_3, down_5], dim=1))
        up_5 = self.up_5(torch.cat([up_4, down_4], dim=1))
        up_6 = self.up_6(torch.cat([up_5, down_3], dim=1))
        up_7 = self.up_7(torch.cat([up_6, down_2], dim=1))
        out = self.last_Conv(torch.cat([up_7, down_1], dim=1))
        return out
    
    #不固定参数，mae耶参与训练,直接相加取平均
    def forward(self, x):
        
        trans_feature = self.trans_part(x) #  [N, 197,1024]

        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        # down_1 = (self.down_1(x)  + self.tcu_1(trans_feature)) / 2
        # down_2 = (self.down_2(down_1) + self.tcu_2(trans_feature)) / 2
        down_3 = (self.down_3(down_2) + self.tcu_3(trans_feature)) / 2
        down_4 = (self.down_4(down_3) + self.tcu_4(trans_feature)) / 2
        down_5 = (self.down_5(down_4) + self.tcu_5(trans_feature)) / 2
        down_6 = (self.down_6(down_5) + self.tcu_6(trans_feature)) / 2
        down_7 = (self.down_7(down_6) + self.tcu_7(trans_feature)) / 2
        down_8 = (self.down_8(down_7) + self.tcu_8(trans_feature)) / 2
        # down_4 = self.down_4(down_3)
        # down_5 = self.down_5(down_4)
        # down_6 = self.down_6(down_5)
        # down_7 = self.down_7(down_6)
        # down_8 = self.down_8(down_7)
        # up
        up_1 = self.up_1(down_8)
        up_2 = self.up_2(torch.cat([up_1, down_7], dim=1))
        up_3 = self.up_3(torch.cat([up_2, down_6], dim=1))
        up_4 = self.up_4(torch.cat([up_3, down_5], dim=1))
        up_5 = self.up_5(torch.cat([up_4, down_4], dim=1))
        up_6 = self.up_6(torch.cat([up_5, down_3], dim=1))
        up_7 = self.up_7(torch.cat([up_6, down_2], dim=1))
        out = self.last_Conv(torch.cat([up_7, down_1], dim=1))
        return out

# 相加后norm
    def forward_norm(self, x):
        
        trans_feature = self.trans_part(x) #  [N, 197,1024]

        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1 + self.tcu_1(trans_feature))
        down_3 = self.down_3(down_2 + self.tcu_2(trans_feature))
        down_4 = self.down_4(down_3 + self.tcu_3(trans_feature))
        down_5 = self.down_5(down_4 + self.tcu_4(trans_feature))
        down_6 = self.down_6(down_5 + self.tcu_5(trans_feature))
        down_7 = self.down_7(down_6 + self.tcu_6(trans_feature))
        down_8 = self.down_8(down_7 + self.tcu_7(trans_feature))
        # up
        up_1 = self.up_1(down_8)
        up_2 = self.up_2(torch.cat([up_1, down_7], dim=1))
        up_3 = self.up_3(torch.cat([up_2, down_6], dim=1))
        up_4 = self.up_4(torch.cat([up_3, down_5], dim=1))
        up_5 = self.up_5(torch.cat([up_4, down_4], dim=1))
        up_6 = self.up_6(torch.cat([up_5, down_3], dim=1))
        up_7 = self.up_7(torch.cat([up_6, down_2], dim=1))
        out = self.last_Conv(torch.cat([up_7, down_1], dim=1))
        return out
    
# concat
    def forward_concat (self, x):
        
        trans_feature = self.trans_part(x) #  [N, 197,1024]

        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1 + self.tcu_1(trans_feature))
        down_3 = self.down_3(down_2 + self.tcu_2(trans_feature))
        down_4 = self.down_4(down_3 + self.tcu_3(trans_feature))
        down_5 = self.down_5(down_4 + self.tcu_4(trans_feature))
        down_6 = self.down_6(down_5 + self.tcu_5(trans_feature))
        down_7 = self.down_7(down_6 + self.tcu_6(trans_feature))
        down_8 = self.down_8(down_7 + self.tcu_7(trans_feature))
        # up
        up_1 = self.up_1(down_8)
        up_2 = self.up_2(torch.cat([up_1, down_7], dim=1))
        up_3 = self.up_3(torch.cat([up_2, down_6], dim=1))
        up_4 = self.up_4(torch.cat([up_3, down_5], dim=1))
        up_5 = self.up_5(torch.cat([up_4, down_4], dim=1))
        up_6 = self.up_6(torch.cat([up_5, down_3], dim=1))
        up_7 = self.up_7(torch.cat([up_6, down_2], dim=1))
        out = self.last_Conv(torch.cat([up_7, down_1], dim=1))
        return out   

# Trans_to_CNN
class ToC(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, resize, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(ToC, self).__init__()

        self.resize = resize
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()
        self.drop = nn.Dropout(p=0.5)

    def forward_ori(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        x_r = self.drop(x_r)
        

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))

    # 用1024 reshape 32, 32
    def forward(self, x):
        B, C, embedding_num = x.shape
        # [N, 196, 1024]-->[N, C,32,32]
        x_r = x[:, 1:].reshape(B, C-1, 32, 32)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(self.resize, self.resize))
    
    # 用196 reshape
    def forward_196(self, x):
        B, _, C = x.shape
        # B, C, embedding_num = x.shape
        # [N, 196, 1024]-->[N, C,32,32]
        # x_r = x[:, 1:].reshape(B, C-1, H, W)

        # [N, 196, 1024]->[N, 1024, 196] ->[N,1024,14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, 14, 14)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(self.resize, self.resize))


class MAE_Feature(nn.Module):
    """ Get_MAE Features
    """

    def __init__(self):
        super(MAE_Feature, self).__init__()

        self.trans_part = transformer_part.__dict__['vit_large_patch16'](
        img_size=224,
        num_classes=5,
        drop_path_rate=0,
        global_pool=True,
        )

        # load model
        chkpt_dir = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-segmentation/model/retf/result/checkpoints/RETFound_cfp_weights.pth'
        # chkpt_dir = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/model/retf/result/checkpoints/RETFound_oct_weights.pth'
        checkpoint = torch.load(chkpt_dir, map_location='cpu')


        msg = self.trans_part.load_state_dict(checkpoint['model'], strict=False)

    def forward(self, x):
        torch_resize = Resize([224,224]) # 定义Resize类对象
        im1_resize = torch_resize(x)
        latent_features = self.trans_part.forward_features(im1_resize) #  [N, 197,1024]
        return latent_features



if __name__ == '__main__':
    # G = pix2pixG_256().to('cuda')
    G = lesion_G_256().to('cuda')
    
    summary(G, (3, 256, 256), batch_size=1)
