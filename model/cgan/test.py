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

FT = torch.LongTensor
FT_a = torch.FloatTensor

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cuda = True if torch.cuda.is_available() else False 

if cuda: 
	generator.cuda()
	discriminator.cuda()
	a_loss.cuda()
	FT = torch.cuda.LongTensor
	FT_a = torch.cuda.FloatTensor
      
