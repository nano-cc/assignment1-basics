import random

import numpy.typing as npt
import numpy as np
import torch


def get_batch(dataset:npt.NDArray,batch_size:int,context_length:int,device:str):
    """
    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.
    """
#     之前的理解是错误的，之前以为数据不能重叠
#     理解还是有误，根据测试代码貌似需要随机采样而不是固定平均取开始下标
    # 1. 得到batch_size个划分数据的起始点，这里应该向下取整
    # 这里可能每次只要取100个token或者其整数倍个
    total_start_count = dataset.shape[0] - context_length
    # 每个序列的起始点
    seq_start_pos = np.random.default_rng().choice(np.arange(0, total_start_count), size=batch_size, replace=False)
    label_start_pos = [i+1 for i in seq_start_pos]
    seqs = None
    labels = None
    for i in range(len(seq_start_pos)):
        seq = torch.tensor(dataset[seq_start_pos[i]:seq_start_pos[i]+context_length] )
        label = torch.tensor(dataset[label_start_pos[i]:label_start_pos[i]+context_length])
        if seqs is not None and labels is not None:
            seqs = torch.cat([seqs.view(-1,context_length),seq.unsqueeze(0)],dim=0)
            labels = torch.cat([labels.view(-1,context_length),label.unsqueeze(0)],dim=0)
        else:
            seqs = seq
            labels = label
    seqs = seqs.to(device)
    labels = labels.to(device)
    return seqs, labels



if __name__ == '__main__':
    dataset = np.arange(100)
    print(dataset)
    batch_size = 32
    context_length = 7
    device = 'cuda'
    print(get_batch(dataset,batch_size,context_length,device)[0].shape)
