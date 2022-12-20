from collections import Iterable

import torch

import numpy as np


# noinspection PyAttributeOutsideInit
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    #更新值 更具val的类型
    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.cpu().numpy()
            #torch.tensor转换为ndarray
        if isinstance(val, Iterable):
            val = np.array(val)
            self.update(np.mean(np.array(val)), n=val.size)
        else:
            self.val = self.multiplier * val
            self.sum += self.multiplier * val * n
            self.count += n
            self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self):
        return "%.6f (%.6f)" % (self.val, self.avg)


#计算平均值 print可以返回当前平均值？和历史平均值
#用法AverageMeter.update(val,n=?)