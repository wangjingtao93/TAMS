import numpy as np  
import os 
import argparse
import torch 
from torch.autograd import Variable
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader
from torchvision import datasets 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist')
parser.add_argument('--dataset', type=str, default='mnist', help='cifar10 | lsun | mnist')
# parser.add_argument('--dataroot', required=True, help='path to data')
parser.add_argument('--dataroot', type=str, default='data', help='path to data')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='image size input')
parser.add_argument('--channels', type=int, default=1, help='number of channels')
parser.add_argument('--latentdim', type=int, default=100, help='size of latent vector')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes in data set')
parser.add_argument('--epoch', type=int, default=200, help='number of epoch')
parser.add_argument('--lrate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta', type=float, default=0.5, help='beta for adam optimizer')
parser.add_argument('--beta1', type=float, default=0.999, help='beta1 for adam optimizer')
parser.add_argument('--output', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--randomseed', type=int, help='seed')
 
opt = parser.parse_args()

img_shape = (opt.channels, opt.imageSize, opt.imageSize)

cuda = True if torch.cuda.is_available() else False 

os.makedirs(opt.output, exist_ok=True)

if opt.randomseed is None: 
	opt.randomseed = random.randint(1,10000)
random.seed(opt.randomseed)
torch.manual_seed(opt.randomseed)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
		
        # 将label编码成向量
        self.label_embedding = nn.Embedding(opt.n_classes, opt.label_dim) #10 , 50
        ## TODO: There are many ways to implement the model,  one alternative 
        ## architecture is (100+50)--->128--->256--->512--->1024--->(1,28,28)

        ### START CODE HERE
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.label_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
        ### END CODE HERE

    def forward(self, noise, labels):
       
        ### START CODE HERE
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_embedding(labels), noise), -1) #拼接两个向量
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img
        ### END CODE HERE
        
        return 

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.label_dim)#10,50
        ## TODO: There are many ways to implement the discriminator,  one alternative 
        ## architecture is (100+784)--->512--->512--->512--->1
        
        ### START CODE HERE
        self.model = nn.Sequential(
            nn.Linear(opt.label_dim + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )
        ### END CODE HERE
        
       
    def forward(self, img, labels):
        ### START CODE HERE
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        ### END CODE HERE
        
        return validity


## TODO: implement the training process

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        #创建一个大小为(batch_size,1),数值全为 1.0 的 tensor
        valid = FloatTensor(batch_size, 1).fill_(1.0)
        
        #创建一个大小为(batch_size,1),数值全为 0.0 的 tensor
        fake = FloatTensor(batch_size, 1).fill_(0.0)

        # Configure input
        real_imgs = imgs.type(FloatTensor)
        labels = labels.type(LongTensor)
  
        # -----------------
        #  Train Generator
        # -----------------

        ### START CODE HERE
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        #生成一批 noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        #生成一批 label
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # 输入 z 和 gen_labels ，通过生成器，生成一批图片
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        # 通过判别器，判断生成图像的真假，返回一批图像的判别结果
        validity = discriminator(gen_imgs, gen_labels)
        # 判别为假的产生loss，这里计算生成器的loss
        g_loss = adversarial_loss(validity, valid)
		# BP + 更新
        g_loss.backward()
        optimizer_G.step()
        ### END CODE HERE

        # ---------------------
        #  Train Discriminator
        # ---------------------

        ### START CODE HERE
        optimizer_D.zero_grad()

        # 计算真实图片的 loss
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # 计算生成图片的 loss
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total loss
        d_loss = (d_real_loss + d_fake_loss) / 2
		
        # BP + 更新
        d_loss.backward()
        optimizer_D.step()
        ### END CODE HERE

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
    if (epoch+1) % 20 ==0:
        torch.save(generator.state_dict(), "./cgan_generator %d.pth" % (epoch))
