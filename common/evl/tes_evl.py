import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-segmentation')
import common.evl.metrics as smp
import torch

output = torch.rand([10, 1, 256, 256])
target = torch.rand([10, 1, 256, 256]).round().long()
# first compute statistics for true positives, false positives, false negative and
# true negative "pixels"
tp, fp, fn, tn = smp.get_stats(output, target, mode='multilabel', threshold=0.5)