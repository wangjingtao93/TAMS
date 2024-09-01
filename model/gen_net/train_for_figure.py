import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/model/gen_net')

from torch.utils.tensorboard import SummaryWriter
from mygen.mygenerator import lesionD_256
from mygen.mygenerator import lesion_G_256
import argparse
from mydatasets import CreateDatasets, CreateDatasets_CNV, CreateDatasets_RIPS,CreateDatasets_CNV_Trans
from split_data import split_data
import os
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from utils import train_one_epoch, val, mkdir, set_gpu
import time
import shutil
import traceback


object_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation'

def train(args):
    batch = args.batch
    data_path = args.dataPath
    print_every = args.every
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = args.epoch
    img_size = args.imgsize

    # # 加载数据集
    # train_imglist, val_imglist = split_data(data_path)
    # train_datasets = CreateDatasets(train_imglist, img_size)
    # val_datasets = CreateDatasets(val_imglist, img_size)
    if args.datatype == 'bv1000_ocv_cnv':
        # # 加载CNV数据集
        train_path = os.path.join(object_path, 'data/bv1000_ocv_cnv/real_data/cnv/acc_eyes/linux/train.csv')
        val_path = os.path.join(object_path, 'data/bv1000_ocv_cnv/real_data/cnv/acc_eyes/linux/val.csv')  
        train_datasets = CreateDatasets_CNV(train_path,img_size)
        val_datasets = CreateDatasets_CNV(val_path, img_size)
        # train_datasets = CreateDatasets_CNV_Trans(train_path,img_size)
        # val_datasets = CreateDatasets_CNV_Trans(val_path, img_size)
    elif args.datatype == 'rips':
        # 加载RIPS数据集 
        all_data_path = os.path.join(object_path, 'data/rips_fundus_rp/real_data/for_paper_fig.csv')
        all_datasets = CreateDatasets_RIPS(all_data_path,img_size)
        train_size = int(len(all_datasets) * 0.5)
        val_size = len(all_datasets) - train_size
        train_datasets, val_datasets = torch.utils.data.random_split(all_datasets, [train_size, val_size])

    elif args.datatype == 'heshi':
        # 加载HeShi数据集 
        all_data_path = os.path.join(object_path, 'data/hsyk_fundus_rp/real_data/all_real_data.csv')
        all_datasets = CreateDatasets_RIPS(all_data_path,img_size)
        train_size = int(len(all_datasets) * 0.7)
        val_size = len(all_datasets) - train_size
        train_datasets, val_datasets = torch.utils.data.random_split(all_datasets, [train_size, val_size])
    else:
        raise NotImplementedError


    train_loader = DataLoader(dataset=train_datasets, batch_size=batch, shuffle=True, num_workers=args.numworker,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_datasets, batch_size=batch, shuffle=True, num_workers=args.numworker,
                            drop_last=True)

    # 实例化网络
    if args.g_net == 'ori_gen':
        # pix_G = pix2pixG_256().to(device)
        pass

    elif args.g_net == 'my_gen':
        pix_G = lesion_G_256().to(device)
        
    else:
        raise NotImplementedError

    pix_D = lesionD_256().to(device)

    # 定义优化器和损失函数
    # fundus 要使用lr=0.001
    optim_G = optim.Adam(pix_G.parameters(), lr=0.001, betas=(0.5, 0.999))
    optim_D = optim.Adam(pix_D.parameters(), lr=0.001, betas=(0.5, 0.999))
    loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    start_epoch = 0

    # 加载预训练权重
    args.weight = '/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-segmentation/model/gen_net/results/result_20240416/rips/my_gen/2024-04-24-15-53-51/gen_lesion_256.pth'
    if args.weight != '':
        ckpt = torch.load(args.weight)
        pix_G.load_state_dict(ckpt['G_model'], strict=False)
        pix_D.load_state_dict(ckpt['D_model'], strict=False)
        start_epoch = ckpt['epoch'] + 1

    writer = SummaryWriter(os.path.join(args.store_dir, 'train_logs'))
    # 开始训练
    for epoch in range(0, epochs):
        # 验证集
        val(args, G=pix_G, D=pix_D, val_loader=train_loader, loss=loss, l1_loss=l1_loss, device=device, epoch=epoch)
        # 验证集
        val(args, G=pix_G, D=pix_D, val_loader=val_loader, loss=loss, l1_loss=l1_loss, device=device, epoch=epoch+1)
        exit(0)

        loss_mG, loss_mD = train_one_epoch(G=pix_G, D=pix_D, train_loader=train_loader,
                                           optim_G=optim_G, optim_D=optim_D, writer=writer, loss=loss, device=device,
                                           plot_every=print_every, epoch=epoch, l1_loss=l1_loss)

        writer.add_scalars(main_tag='train_loss', tag_scalar_dict={
            'loss_G': loss_mG,
            'loss_D': loss_mD
        }, global_step=epoch)

        # 验证集
        val(args, G=pix_G, D=pix_D, val_loader=val_loader, loss=loss, l1_loss=l1_loss, device=device, epoch=epoch)

        # 保存模型
        torch.save({
            'G_model': pix_G.state_dict(),
            'D_model': pix_D.state_dict(),
            'epoch': epoch
        }, os.path.join(args.store_dir, 'pix2pix_256.pth'))


def create_store_dir(args):
    args.project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation/model/gen_net'

    time_name=time.strftime('%Y-%m-%d-%H-%M-%S')

    store_dir = os.path.join(args.project_path, args.save_path, args.datatype)
    args.store_dir =os.path.join(store_dir, args.g_net, time_name)
    mkdir(args.store_dir)

    # 创建一个说明文件
    description_file = os.path.join(args.store_dir,  args.description_name)
    with open(str(description_file), 'w') as f:
        f.write(args.description)


def cfg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch', type=int, default=4)
    parse.add_argument('--epoch', type=int, default=200)
    parse.add_argument('--imgsize', type=int, default=256)
    parse.add_argument('--dataPath', type=str, default='./data', help='data root path')
    parse.add_argument('--weight', type=str, default='', help='load pre train weight')
    parse.add_argument('--numworker', type=int, default=4)
    parse.add_argument('--every', type=int, default=2, help='plot train result every * iters')

    parse.add_argument('--save_path', type=str, default='results/result_20240416', help='data root path')
    parse.add_argument('--datatype', type=str, choices=['bv1000_oct_cnv', 'rips', 'heshi'],default='rips', help='')
    parse.add_argument('--description_name', type=str, default='drop_lr0.002', help='data root path')
    parse.add_argument('--description', type=str, default='ma', help='')
    parse.add_argument('--g_net', type=str, choices=['my_gen', 'ori_gen'], default='my_gen', help='generator net')
    parse.add_argument('--gpu', type=int, nargs='+', default=[0], help='0 = GPU.')

    # 删除错误文件
    parse.add_argument('--is_remove_exres', type=lambda x: (str(x).lower() == 'true'), default=True)

    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = cfg()
    # set_gpu(args.gpu)
    torch.cuda.set_device(0)
    create_store_dir(args)

    try:
        train(args)

    except Exception:
        print(traceback.print_exc())
        if args.is_remove_exres:
            shutil.rmtree(args.store_dir)
