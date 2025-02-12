import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class clip_seg(nn.Module):
    def __init__(self,
                 image_encoder,
                #  freeze_image_encoder=False,
                 ):
        super().__init__()
        self.image_encoder = image_encoder
        self.decoder = ModifiedMedSAMDecoder()

    def forward(self, image):

        # do not compute gradients for pretrained prompt encoder

        vit_output = self.image_encoder.forward_features(image)  # (B, 512)
        B, C = vit_output.shape
        image_embedding = vit_output[:, 1:].transpose(1, 2).reshape(B, C, 14, 14)
        ori_res_masks = self.decoder(image_embedding)

        return ori_res_masks

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

    # 用1024 reshape 32, 32
    def forward(self, x):
        B, C, embedding_num = x.shape
        # [N, 196, 1024]-->[N, C,32,32]
        x_r = x[:, 1:].reshape(B, C-1, 32, 32)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(self.resize, self.resize))



class ModifiedMedSAMDecoder(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 out_channels=1,
                 inter_channels_list=[512, 256, 128, 64],
                 final_size=224):
        super(ModifiedMedSAMDecoder, self).__init__()

        # 假设通过多级上采样，逐渐从14x14恢复到更高分辨率
        # 我们需要上采样几次？如果都用2倍上采样:
        # 14 -> 28 -> 56 -> 112 -> 224
        # 224已经接近256，只差32个像素，可以最后用一次双线性插值直接到256

        # 构建多级上采样模块，每级包括 转置卷积/上采样 + 卷积
        self.up_blocks = nn.ModuleList()
        prev_channels = in_channels
        for ic in inter_channels_list:
            block = nn.Sequential(
                # 使用转置卷积进行2倍上采样
                nn.ConvTranspose2d(prev_channels, ic, kernel_size=2, stride=2),  # from HxW to 2H x 2W
                nn.BatchNorm2d(ic),
                nn.ReLU(inplace=True),
                nn.Conv2d(ic, ic, kernel_size=3, padding=1),
                nn.BatchNorm2d(ic),
                nn.ReLU(inplace=True)
            )
            self.up_blocks.append(block)
            prev_channels = ic

        # 最终输出层
        # 当我们完成多级上采样后，空间分辨率为224x224(经过4次×2上采样：14->28->56->112->224)
        # 最后使用F.interpolate将224x224插值到256x256
        self.final_conv = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

        self.final_size = final_size

    def forward(self, x):
        # x: [B, 1024, 14, 14]
        for block in self.up_blocks:
            x = block(x)
            # 每次block后分辨率 *2:
            # 第一次: 14->28
            # 第二次: 28->56
            # 第三次: 56->112
            # 第四次: 112->224

        # 此时x: [B, 64, 224, 224]
        # 插值到256x256
        x = F.interpolate(x, size=(self.final_size, self.final_size), mode='bilinear', align_corners=False)

        # 最终预测层
        x = self.final_conv(x)  # [B, out_channels, 256, 256]
        return x