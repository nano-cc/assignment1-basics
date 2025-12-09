import json
import argparse
import os
import random

import wandb
import torch
import yaml
import numpy as np
from cs336_basics.dataloader import get_batch
from cs336_basics.modules import cross_entropy, gradient_clipping, cosine_lr_scheduler, save_checkpoint, load_checkpoint
from cs336_basics.optimizer import AdamWOptimizer
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.transformer import Transformer, count_module_parameters
from tqdm import tqdm

def get_device():
    # 返回当前可用的计算设备
    # 如果检测到 GPU（CUDA），则返回 'cuda:0'
    # 否则返回 'cpu'
    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return  'cpu'


def init_wandb(config):
    import wandb

    # 初始化并启动一个新的 wandb 运行（experiment）
    # project：wandb 项目名称
    # name：当前 run 的名称（可选，由 config 指定）
    # config：记录超参数配置
    wandb.init(
        project=config['wandb_project_name'],
        name=config['wandb_run_name'],  # 可选的 run 名称
        config=config
    )


def get_weight_norms(tgt_model):
    # 计算模型所有参数（需梯度的参数）的 L2 范数
    # 用于监控模型权重是否出现异常增大（爆炸）
    total_norm = 0.0
    for p in tgt_model.parameters():
        if p.requires_grad:
            param_norm = p.data.norm(2)  # 单个参数张量的 L2 范数
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5  # 最终取平方根，得到总体 L2 范数
    return total_norm


def get_grad_norms(tgt_model):
    # 计算模型所有梯度的 L2 范数
    # 用于监控梯度是否爆炸或消失
    total_norm = 0.0
    import torch
    for p in tgt_model.parameters():
        if p.grad is not None:
            grad_norm = p.grad.data.norm(2)  # 单个梯度张量的 L2 范数
            total_norm += grad_norm.item() ** 2
    total_norm = total_norm ** 0.5  # 得到整体梯度范数
    return total_norm


def set_manual_seed(seed: int):
    # 设置随机种子，使实验具有可复现性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
