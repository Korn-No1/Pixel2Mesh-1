"""
Helper functions that have not yet been implemented in pytorch
"""

import torch

#用detach()函数来切断一些分支的反向传播
#递归的使用detach（）
def recursive_detach(t):
    if isinstance(t, torch.Tensor):
        return t.detach()
    elif isinstance(t, list):
        return [recursive_detach(x) for x in t]
    elif isinstance(t, dict):
        return {k: recursive_detach(v) for k, v in t.items()}
    else:
        return t


#稀疏矩阵的乘法运算在pytorch中会有问题，这里自定义了一种方法
#小问题是这里的batch是什么
def batch_mm(matrix, batch):
    """
    https://github.com/pytorch/pytorch/issues/14489
    """
    # TODO: accelerate this with batch operations
    return torch.stack([matrix.mm(b) for b in batch], dim=0)


def dot(x, y, sparse=False):
    """Wrapper for torch.matmul (sparse vs dense)."""
    if sparse:
        return batch_mm(x, y)
    else:
        return torch.matmul(x, y)
