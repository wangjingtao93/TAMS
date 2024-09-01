import os
import csv

import common.utils as utils

class RecordMetaResult:
    def __init__(self, args): #必须要有一个self参数，

        self.args = args
        self._init_csv()
    
    def _init_csv(self):
        store_dir_meta_test = self.args.store_dir
        store_meta_pth = str(os.path.join(self.args.store_dir, 'save_meta_pth'))
        utils.mkdir(store_meta_pth)

        # 创建一个记录meta_train, 和meta val用于观察
        # record meta_train and meta_val access for check
        self.metric_dir_meta_train = os.path.join(self.args.store_dir, 'metric_meta_train' + '.csv')
        with open(self.metric_dir_meta_train, 'w') as f:
            fields = ['Meta_Epoch', 'meta_train_Loss', 'meta_val_Loss', 'meta_train_dice', 'meta_val_dice', 'Grad', 'Lr', 'Best_Meta_Train_Epoch', 'Best_Meta_Train_Dice', 'Best_Meta_Val_Epoch', 'Best_Meta_Val_Dice']
            datawrite = csv.writer(f, delimiter=',')
            datawrite.writerow(fields)

        # 创建一个记录meta_test的query set
        # record meta_test querry set prediction
        self.metric_dir_meta_test = os.path.join(store_dir_meta_test, 'metric_meta_test' + '.csv')
        with open(self.metric_dir_meta_test, 'w') as f:
            # best_val_dice: per meta_epoch, 
            # best_Q_dice: all meta_epoch
            # best_Q_dice_1: all meta_epoch, another criterion
            fields = ['Meta_Epoch'] + list(range(0, self.args.n_epoch, 1)) + ['best_val_dice', 'best_val_epoch', 'best_Q_dice', 'best_meta_epoch', 'best_Q_dice_1', 'best_meta_epoch_1', 'final_test_dice',]#记录内循环每次的pred, record per inner_loop(meta_test) pred
            datawrite = csv.writer(f, delimiter=',')
            datawrite.writerow(fields)



        self.best_meta_epoch_for_meta_val = 0
        self.best_meta_val_pred = 0.0
        self.best_meta_epoch_for_meta_train = 0
        self.best_meta_train_pred = 0.0

        self.bpfttq = 0.0 # best_pred for test task querry set, 在meta_val上表现最好的epoch, 测试任务上querry set上的perd
        self.bmfttq = 0 # best_meta_peoch for test task query set

        # 另外一种计算方式
        # another criterion
        self.best_val = 0.0
        self.best_epoch = 0

        if self.args.is_mid_val:
            # use mid_dl_epoch pred, one dl epoch, exe multi-validation
            # 一个dl_epoch,进行多次validation
            
            # record meta_test querry set prediction
            self.metric_dir_meta_test_mid = os.path.join(store_dir_meta_test, 'metric_meta_test_mid' + '.csv')
            with open(self.metric_dir_meta_test_mid, 'w') as f:
                # best_val_dice: per meta_epoch, 
                # best_Q_dice: all meta_epoch
                # best_Q_dice_1: all meta_epoch, another criterion
                fields = ['Meta_Epoch'] + list(range(0, self.args.n_epoch, 1)) + ['best_val_dice', 'best_val_epoch', 'best_Q_dice', 'best_meta_epoch', 'best_Q_dice_1', 'best_meta_epoch_1', 'final_test_dice',]#记录内循环每次的pred, record per inner_loop(meta_test) pred
                datawrite = csv.writer(f, delimiter=',')
                datawrite.writerow(fields)
                
            self.bpfttq_mid = 0.0
            self.bmfttq_mid = 0
            self.best_val_mid = 0
            self.best_epoch_mid = 0

    def write_res(self, res):
        # meta_train
        is_best_meta_train = res['meta_train_dice'] > self.best_meta_train_pred
        self.best_meta_train_pred = max(res['meta_train_dice'], self.best_meta_train_pred)
        if is_best_meta_train:
            self.best_meta_epoch_for_meta_train = res['meta_epoch']

        # meta_val
        is_best_meta_val = res['meta_val_dice'] > self.best_meta_val_pred
        self.best_meta_val_pred = max(res['meta_val_dice'], self.best_meta_val_pred)
        if is_best_meta_val:
            self.best_meta_epoch_for_meta_val = res['meta_epoch']
        
        with open(self.metric_dir_meta_train, 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            data_row = [res['meta_epoch'], res['meta_train_loss'], res['meta_val_loss'], res['meta_train_dice'], res['meta_val_dice'], res['meta_train_grad'],  res['current_outer_lr'], self.best_meta_epoch_for_meta_train, self.best_meta_train_pred, self.best_meta_epoch_for_meta_val, self.best_meta_val_pred]
            csv_write.writerow(data_row)


        # in theory, meta test must exe after all meta epoch
        # meta test, every epoch, exe meta_test for check
        # use the same epoch's ave pred of all test  tasks
        # 使用所有任务，相同epoch的均值
        best_meta_test_Q_pred = max(res['meta_test_Q_values'])
        is_best_for_meta_test_Q = best_meta_test_Q_pred > self.bpfttq
        self.bpfttq = max(best_meta_test_Q_pred, self.bpfttq)
        if is_best_for_meta_test_Q:
            self.bmfttq = res['meta_epoch']

        # use the best pred of every test task
        # 使用所有任务，最好的val_acc
        is_best_2 = res['meta_final_ave_tasks'][-2] > self.best_val
        self.best_val = max(res['meta_final_ave_tasks'][-2], self.best_val)
        if is_best_2:
            self.best_epoch = res['meta_epoch']

        with open(self.metric_dir_meta_test, 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            best_val_pred = max(res['meta_test_Q_values'])
            best_index = res['meta_test_Q_values'].index(best_val_pred)
            data_row = [res['meta_epoch']] + res['meta_test_Q_values'] + [best_val_pred, best_index, self.bpfttq, self.bmfttq, self.best_val, self.best_epoch, res['meta_final_ave_tasks'][0]]

            csv_write.writerow(data_row)

        if self.args.is_mid_val:
            self.write_res_mid(res)

    def write_res_mid(self, res):
        # for mid dl_epoch,one dl epoch, exe multi-validation
        best_meta_test_Q_pred_mid = max(res['meta_test_Q_values_mid'])
        is_best = best_meta_test_Q_pred_mid > self.bpfttq_mid
        self.bpfttq_mid = max(best_meta_test_Q_pred_mid, self.bpfttq_mid)
        if is_best:
            self.bmfttq_mid = res['meta_epoch']

        # use the best pred of every test task
        # 使用所有任务，最好的val_acc
        is_best = res['meta_final_ave_tasks_mid'][-2] > self.best_val_mid
        self.best_val_mid = max(res['meta_final_ave_tasks_mid'][-2], self.best_val_mid)
        if is_best:
            self.best_epoch_mid = res['meta_epoch']

        with open(self.metric_dir_meta_test_mid, 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            best_val_pred = max(res['meta_test_Q_values_mid'])
            best_index = res['meta_test_Q_values_mid'].index(best_val_pred)
            data_row = [res['meta_epoch']] + res['meta_test_Q_values_mid'] + [best_val_pred, best_index, self.bpfttq_mid, self.bmfttq_mid, self.best_val_mid, self.best_epoch_mid, res['meta_final_ave_tasks_mid'][0]]

            csv_write.writerow(data_row)


        




