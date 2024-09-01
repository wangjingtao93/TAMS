import numpy as np
import random
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

class Task(object):
    def __init__(self, all_classes, ways, shot, query, dataframe):
        self.all_classes = all_classes
        self.ways = ways
        self.shot = shot
        self.dataframe = dataframe
        self.query = query
        self.query_roots = [] # test_task_query
        self.support_roots = [] # test_task_support

        # samples_per_class = 10 # 每类10张
        sampled_classes = random.sample(all_classes, ways)# 从所有类中随机选取几类
        
        for c in sampled_classes:
            samples_per_class = len(list(dataframe[dataframe["ID"] == c]['ID'])) # 一劳永逸
            cframe = dataframe[dataframe["ID"] == c].sample(samples_per_class) # 因为一个类只有10个样本，就全选了吧
            
            # 索引重新排序, sample, 顺序会乱，索引是sample之前的索引
            # 关键这样做没有屁用啊，因为iloc[idx]取得是第idx行，而不是索引为idx的行
            # 并且cframe=cframe.reset_index(inplace=True, drop=True)这样才能重排
            # cframe.reset_index(inplace=True, drop=True)
                        
            paths = cframe[["Image_path", "Label_path"]]
            sample_idxs = np.random.choice(samples_per_class, samples_per_class, replace=False)
            support_idxs = sample_idxs[:shot]
            query_idxs = sample_idxs[shot:(shot + query)]
            for idx in query_idxs:
                self.query_roots.append((paths.iloc[idx]['Image_path'], paths.iloc[idx]['Label_path']))
            for idx in support_idxs:
                self.support_roots.append((paths.iloc[idx]['Image_path'], paths.iloc[idx]['Label_path']))


'''
just build one tasks according to DL, use all real data, not include synthetic data
df_ls:include train_df, val_df, test_df
'''
class DL_Task():
    def __init__(self, args, df_ls):
        self.args = args
        self.df_train_data = df_ls[0]
        self.df_val_data = df_ls[1]
        self.df_test_data = df_ls[2]

        self.query_roots = [] # 相当于train, similar train_data  in DL
        self.support_roots = [] # 相当于val, similar to val data in DL       
        self.test_roots = [] # 用于最终测试, similar to test data in DL

        self.get_df()

    def get_df(self):
        # support set+++++
        paths = self.df_train_data[["Image_path", "Label_path"]]
        # iloc的是行号，不是索引号，行号和索引号可能不对应
        index_ls = self.df_train_data.index.to_list()
        support_idxs = index_ls
        for idx in support_idxs:
            self.support_roots.append((paths.iloc[index_ls.index(idx)]['Image_path'], paths.iloc[index_ls.index(idx)]['Label_path']))   

        # querry set++++++
        paths = self.df_val_data[["Image_path", "Label_path"]]
        index_ls = self.df_val_data.index.to_list()
        query_idxs = index_ls
        for idx in query_idxs:
            self.query_roots.append((paths.iloc[index_ls.index(idx)]['Image_path'], paths.iloc[index_ls.index(idx)]['Label_path']))   
        
        # test set++++++
        paths = self.df_test_data[["Image_path", "Label_path"]]
        index_ls = self.df_test_data.index.to_list()
        test_idxs = index_ls
        for idx in test_idxs:
            self.test_roots.append((paths.iloc[index_ls.index(idx)]['Image_path'], paths.iloc[index_ls.index(idx)]['Label_path']))

