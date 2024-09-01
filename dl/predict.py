import os
import csv
import torch
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import  DataLoader

from common.dl_comm import dl_comm
import common.utils as utils
import cv2
import numpy as np

def predict_enter(args,test_data_ls):

    predict(args,test_data_ls)


def predict(args, test_data_ls):
    #0是训练集 1 是验证集，2 是测试集
    final_test_task = test_data_ls[2]

    # 多构建几个任务，结果会更准确, build more tasks will get more precise result
    task_num = len(final_test_task)
    dl_ob = dl_comm(args)
    for task_idx in range(task_num):
        if args.is_save_fig:
            
            test_loader = final_test_task[task_idx]

            save_fig_val(args, dl_ob.net, test_loader)

        else:
            test_loader = final_test_task[task_idx]

            res_test= dl_ob.val(0, test_loader)

            print(f'task_id {task_idx}, res=' , res_test)

    


def save_fig_val(args, net, val_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.eval()
        n_val_batches = len(val_loader)
        count = 0
        store_path = os.path.join(args.store_dir,'figure')
        os.mkdir(store_path)
        for batch in tqdm(val_loader, total=n_val_batches, desc=f'val for save fig', unit='batch', leave=False):
            
            image, mask_true = batch['image'], batch['mask']
            
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net(image)

                for batch_idx in range(image.shape[0]):
                    count += 1
                    store_true = np.asarray(mask_true[batch_idx][0].cpu() * 255)
                    if image[batch_idx].shape[0] > 1:
                        store_img =  np.asarray(image[batch_idx].permute(1,2,0).cpu()*255.0)                        
                    else:
                        store_img =  np.asarray(image[batch_idx][0].cpu()*255.0)
                        store_img = cv2.cvtColor(store_img, cv2.COLOR_GRAY2BGR)
                   
                
                    store_pred = torch.sigmoid(mask_pred) > 0.5  
                    store_pred = np.asarray(store_pred[batch_idx][0].cpu()*255)

                    # store_out = np.concatenate([store_true, store_pred],axis=1)
                    # cv2.imwrite(os.path.join(store_path,str(count)+'.png'),  store_out)
                    # continue

                    color_res = color(store_true, store_pred)

                    # 均变成三通道
                    
                    store_true = cv2.cvtColor(store_true, cv2.COLOR_GRAY2BGR)

                    store_out = np.concatenate([store_img,store_true, color_res],axis=1)
                    cv2.imwrite(os.path.join(store_path,str(count)+'.png'),  store_out)


        return

def color(gt, res):
    

    # print(gt.shape)
    # print(res_name)

    # res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # res = res.reshape(1, res.shape[0], res.shape[1])
    # gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    # res = cv2.resize(res, (3100, 2848), interpolation=cv2.INTER_AREA)
    # gt = gt.reshape(1, gt.shape[0], gt.shape[1])
    # print(res.shape[0])
    # cv2.imwrite(save_see_path, res)  # 保存修改像素点后的图片


    w, h = res.shape[0], res.shape[1]
    result = np.zeros((w, h, 3))
    res = (res >= 100)
    gt = (gt >= 100)

    TP = res * gt
    FP = res * (1 - gt)
    FN = (1 - res) * gt
    TN = (1 - res) * (1 - gt)

    # FN
    result[:, :, 0] = np.where(FN == 1, 0, result[:, :, 0])
    result[:, :, 1] = np.where(FN == 1, 0, result[:, :, 1])
    result[:, :, 2] = np.where(FN == 1, 255, result[:, :, 2])

    # FP
    result[:, :, 0] = np.where(FP == 1, 0, result[:, :, 0])
    result[:, :, 1] = np.where(FP == 1, 255, result[:, :, 1])
    result[:, :, 2] = np.where(FP == 1, 0, result[:, :, 2])

    # TP
    result[:, :, 0] = np.where(TP == 1, 255, result[:, :, 0])
    result[:, :, 1] = np.where(TP == 1, 255, result[:, :, 1])
    result[:, :, 2] = np.where(TP == 1, 255, result[:, :, 2])

    # TN
    result[:, :, 0] = np.where(TN == 1, 0, result[:, :, 0])
    result[:, :, 1] = np.where(TN == 1, 0, result[:, :, 1])
    result[:, :, 2] = np.where(TN == 1, 0, result[:, :, 2])

    return result