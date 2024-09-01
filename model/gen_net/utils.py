import torchvision
from tqdm import tqdm
import torch
import os


def train_one_epoch(G, D, train_loader, optim_G, optim_D, writer, loss, device, plot_every, epoch, l1_loss):
    pd = tqdm(train_loader)
    loss_D, loss_G = 0, 0
    step = 0
    G.train()
    D.train()
    for idx, data in enumerate(pd):
        in_img = data[0].to(device)
        real_img = data[1].to(device)
        # lantent_feature = data[2].to(device)
        # 先训练D
        fake_img = G(in_img)
        # fake_img = G(in_img, lantent_feature)
        D_fake_out = D(fake_img.detach(), in_img).squeeze()
        D_real_out = D(real_img, in_img).squeeze()
        ls_D1 = loss(D_fake_out, torch.zeros(D_fake_out.size()).cuda())
        ls_D2 = loss(D_real_out, torch.ones(D_real_out.size()).cuda())
        ls_D = (ls_D1 + ls_D2) * 0.5

        optim_D.zero_grad()
        ls_D.backward()
        optim_D.step()

        # 再训练G
        fake_img = G(in_img)
        # fake_img = G(in_img, lantent_feature)
        D_fake_out = D(fake_img, in_img).squeeze()
        ls_G1 = loss(D_fake_out, torch.ones(D_fake_out.size()).cuda())
        ls_G2 = l1_loss(fake_img, real_img)
        ls_G = ls_G1 + ls_G2 * 100

        optim_G.zero_grad()
        ls_G.backward()
        optim_G.step()

        loss_D += ls_D
        loss_G += ls_G

        pd.desc = 'train_{} G_loss: {} D_loss: {}'.format(epoch, ls_G.item(), ls_D.item())
        # 绘制训练结果
        if idx % plot_every == 0:
            writer.add_images(tag='train_epoch_{}'.format(epoch), img_tensor=0.5 * (fake_img + 1), global_step=step)
            step += 1
    mean_lsG = loss_G / len(train_loader)
    mean_lsD = loss_D / len(train_loader)
    return mean_lsG, mean_lsD


@torch.no_grad()
def val(args, G, D, val_loader, loss, device, l1_loss, epoch):
    pd = tqdm(val_loader)
    loss_D, loss_G = 0, 0
    G.eval()
    D.eval()
    all_loss = 10000
    for idx, item in enumerate(pd):
        in_img = item[0].to(device)
        real_img = item[1].to(device)
        # latent_fearure = item[2].to(device)


        fake_img = G(in_img)
        # fake_img = G(in_img, latent_fearure)
        D_fake_out = D(fake_img, in_img).squeeze()
        ls_D1 = loss(D_fake_out, torch.zeros(D_fake_out.size()).cuda())
        ls_D = ls_D1 * 0.5
        ls_G1 = loss(D_fake_out, torch.ones(D_fake_out.size()).cuda())
        ls_G2 = l1_loss(fake_img, real_img)
        ls_G = ls_G1 + ls_G2 * 100
        loss_G += ls_G
        loss_D += ls_D
        pd.desc = 'val_{}: G_loss:{} D_Loss:{}'.format(epoch, ls_G.item(), ls_D.item())

        # 保存最好的结果
        all_ls = ls_G + ls_D
        if all_ls < all_loss:
            all_loss = all_ls
            best_image = fake_img
    result_img = (best_image + 1) * 0.5
    if not os.path.exists(os.path.join(args.store_dir, 'val_image')):
        os.mkdir(os.path.join(args.store_dir, 'val_image'))

    torchvision.utils.save_image(result_img, os.path.join(args.store_dir, 'val_image', 'val_epoch{}.jpg'.format(epoch)))


# 传入要创建的文件夹路径即可
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False
    
def set_gpu(x):
    x = [str(e) for e in x]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(x)
    print('using gpu:', ','.join(x))