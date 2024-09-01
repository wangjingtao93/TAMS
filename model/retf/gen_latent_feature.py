import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import models_vit
import torchvision.transforms as transform

# imagenet_mean = np.array([0.485, 0.456, 0.406])
# imagenet_std = np.array([0.229, 0.224, 0.225])
project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation'
transform = transform.Compose([
            transform.ToTensor(),
            transform.Resize((224, 224)),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])



def prepare_model(chkpt_dir, arch='vit_large_patch16'):
    # build model
    model = models_vit.__dict__[arch](
        img_size=224,
        num_classes=5,
        drop_path_rate=0,
        global_pool=True,
    )
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model

def run_one_image(x, model):
    x = x.to(device, non_blocking=True)
    latent = model.forward_features(x.float())
    latent = torch.squeeze(latent)
    
    return latent

# download pre-trained RETFound 

device = torch.device('cuda')

def gen_rips():
    # chkpt_dir = './RETFound_cfp.pth'
    chkpt_dir = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/model/retf/result/checkpoints/RETFound_cfp_weights.pth'
    model_ = prepare_model(chkpt_dir, 'vit_large_patch16')

    
    model_.to(device)
    print('Model loaded.')
    model_.eval()

    # get image list
    data_path = '/data1/wangjingtao/workplace/python/data/rp_data/rips/x'
    img_list = os.listdir(data_path)
    for i in img_list:
        img = Image.open(os.path.join(data_path, i))
        img = img.resize((224, 224))
        img = np.array(img) / 255.

        assert img.shape == (224, 224, 3)

        # normalize by mean and sd
        # can use customised mean and sd for your data
        imagenet_mean = np.array([0.5, 0.5, 0.5])
        imagenet_std = np.array([0.5, 0.5, 0.5])
        img = img - imagenet_mean
        img = img / imagenet_std
        
        x = torch.tensor(img)
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)
        latent_feature = run_one_image(x, model_)
        
        # name_list.append(i)
        # feature_list.append(latent_feature.detach().cpu().numpy())
        store_path = os.path.join(data_path.replace('/x', '/x_retf'), i.replace('.jpg', '') + '.npy')
        np.save(store_path, latent_feature.detach().cpu().numpy())


# latent_csv = pd.DataFrame({'Name':name_list, 'Latent_feature':feature_list})
# latent_csv.to_csv('Feature_latent.csv', index = False, encoding='utf8')


def gen_bv1000_cnv():
    # chkpt_dir = './RETFound_cfp.pth'
    chkpt_dir = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/model/retf/result/checkpoints/RETFound_oct_weights.pth'
    model_ = prepare_model(chkpt_dir, 'vit_large_patch16')

    
    model_.to(device)
    print('Model loaded.')
    model_.eval()

    data_path = '/data1/wangjingtao/workplace/python/data/meta-oct/seg/CNV/x_d'
    img_list = os.listdir(data_path)
    for i in img_list:
        img = Image.open(os.path.join(data_path, i)).convert('RGB')
        
        img = transform(img)
        img = img.unsqueeze(dim=0)
        latent_feature = run_one_image(img, model_)

        
        # name_list.append(i)
        # feature_list.append(latent_feature.detach().cpu().numpy())
        # store_path = os.path.join(data_path.replace('/x_d', '/x_retf'), i.replace('.jpg', '') + '.npy')
        # np.save(store_path, latent_feature.detach().cpu().numpy())

def gen_bv1000_cnv_sys():
    # chkpt_dir = './RETFound_cfp.pth'
    chkpt_dir = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/model/retf/result/checkpoints/RETFound_oct_weights.pth'
    model_ = prepare_model(chkpt_dir, 'vit_large_patch16')

    
    model_.to(device)
    print('Model loaded.')
    model_.eval()

    dframe = pd.read_csv(os.path.join(project_path, 'data/bv1000_ocv_cnv/meta_data/synthetic_data.csv'))
    
    img_list = dframe['Image_path']
    for i in img_list:
        i = i.replace('/x', '/x_d')
        img = Image.open(i).convert('RGB')
        
        img = transform(img)
        img = img.unsqueeze(dim=0)
        latent_feature = run_one_image(img, model_)      
        
        # name_list.append(i)
        # feature_list.append(latent_feature.detach().cpu().numpy())
        store_dir = os.path.join(i.replace('/x_d', '/x_retf').replace(i.split('/')[-1], ''))
        isExists = os.path.exists(store_dir)

        # 判断结果
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(store_dir)
        store_path = os.path.join(store_dir, i.split('/')[-1].replace('.jpg', '') + '.npy')
        np.save(store_path, latent_feature.detach().cpu().numpy())
        # os.remove(store_path)
        # os.removedirs(os.path.join(i.replace('/x_d', '/x_retf')))

#gen_rips()
# gen_bv1000_cnv()
gen_bv1000_cnv_sys()
        
