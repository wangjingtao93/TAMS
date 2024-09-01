from pix2Topix import pix2pixG_256
import torch
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
import os
import numpy as np
import torchvision
from mygen.mygenerator import lesion_G_256
import glob

def test_cnv(img_path, store_path=None):
    # if img_path.endswith('.png'):``
    #     img = cv2.imread(img_path)
    #     img = img[:, :, ::-1]
    # else:
    #     img = Image.open(img_path)

    img = Image.open(img_path)
    store_img = cv2.resize(np.array(img), (256,256))
    img = img.convert('RGB')

    transforms = transform.Compose([
        transform.ToTensor(),
        transform.Resize((256, 256)),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transforms(img.copy())
    img = img[None].to('cuda')  # [1,3,128,128]

    # latent_feature = torch.tensor(np.load(img_path.replace('x_d', 'x_retf').replace('.jpg', '.npy'))).unsqueeze(dim=0).to('cuda')
    out = G(img)
    # out = out.permute(1,2,0)
    # out = (0.5 * (out + 1)).cpu().detach().numpy()
    # out = out[:,:,0]
    out = (out + 1) * 0.5
    torchvision.utils.save_image(out, store_path)
    # if store_path == None:
    #     cv2.imwrite('out.jpg', np.concatenate(out*255, store_img))
    # else:
    #     cv2.imwrite(store_path, np.concatenate((out*255, store_img)))
    # plt.figure()
    # plt.imshow(out)
    # plt.show()

def improve_faker_cnv():
    systhsis_df = pd.read_csv(os.path.join(project_path, 'data/bv1000_ocv_cnv/meta_data/synthetic_data.csv'))
    img_ls = systhsis_df['Image_path']
    for img_path in img_ls:
        img_path = img_path.replace('/x', '/x_d')
        store_path = os.path.join(project_path,'model/gen_net/data/test_cnv/sys/improve',img_path.split('/')[-1])
        test_cnv(img_path, store_path=store_path)


def test_rips(img_path, store_path=None):
    # if img_path.endswith('.png'):
    #     img = cv2.imread(img_path)
    #     img = img[:, :, ::-1]
    # else:
    #     img = Image.open(img_path)

    img = Image.open(img_path) # RGB
    store_img = cv2.resize(np.array(img), (256,256))

    transforms = transform.Compose([
        transform.ToTensor(),
        transform.Resize((256, 256)),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transforms(img.copy())
    img = img[None].to('cuda')  # [1,3,128,128]


    out = G(img)[0]
    out = out.permute(1,2,0)
    out = (0.5 * (out + 1)).cpu().detach().numpy()#RGB

    store_out = np.concatenate((out*255,store_img))
    cv2.cvtColor(store_out, cv2.COLOR_RGB2BGR, store_out)

    if store_path == None:
        cv2.imwrite('out.jpg', store_out)
    else:
        cv2.imwrite(store_path, store_out)
    # plt.figure()
    # plt.imshow(out)
    # plt.show()
def improve_rips():
    # systhsis_df = pd.read_csv(os.path.join(project_path, 'data/rips_fundus_rp/sys_data/all_sys_data.csv'))
    # img_ls = systhsis_df['Image_path']
    img_ls = glob.glob('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/task_aug/result_sys/20240419/rips/2024-04-19-14-36-18/*/x/*')
    for img_path in img_ls[:100]:
        store_path = os.path.join(project_path,'model/gen_net/data/test_rips/sys/improve',img_path.split('/')[-1])
        test_rips(img_path, store_path=store_path)

if __name__ == '__main__':
    project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/'

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.cuda.set_device(0)
    # 实例化网络
    G = lesion_G_256().to('cuda')
    # G = pix2pixG_256().to('cuda')
    # 加载预训练权重
    # ckpt = torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/model/gen_net/results/result_20240416/rips/ori_gen/2024-04-23-10-12-02/pix2pix_256.pth')
    # cnv
    ckpt = torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-segmentation/model/gen_net/results/result_20240416/bv1000_cnv/my_gen/2024-04-18-09-45-31/pix2pix_256.pth')
    # rips
    # ckpt = torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-segmentation/model/gen_net/results/result_20240416/rips/my_gen/2024-04-24-15-53-51/gen_lesion_256.pth')
    G.load_state_dict(ckpt['G_model'], strict=False)

    G.eval()

    # test('data/test_cnv/sys/CNV_196_9_1_d.jpg')
    # improve_faker_cnv()
    # improve_rips()

    test_cnv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/model/gen_net/denoise_faker_cal.jpg', store_path='cnv_imp.jpg')


    # test_rips('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/model/gen_net/data/test_rips/lesion_trans_img.jpg', 'rips_imp.jpg')
