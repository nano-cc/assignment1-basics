import json
import argparse
import os
import random
from tabnanny import process_tokens

import wandb
import torch
import yaml
import numpy as np
from cs336_basics.dataloader import get_batch
from cs336_basics.generate_text import generate_text
from cs336_basics.modules import cross_entropy, gradient_clipping, cosine_lr_scheduler, save_checkpoint, \
    load_checkpoint, batch_perplexity
from cs336_basics.optimizer import AdamWOptimizer
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.utils import init_wandb, get_device, set_manual_seed, find_latest_checkpoint, get_weight_norms, \
    get_grad_norms
from cs336_basics.transformer import Transformer, count_module_parameters
from tqdm import tqdm

def cal_load_step():
    """
    计算每步迭代加载多少数据
    """
    return config['context_length'] * config['batch_size'] - config['context_length']



def get_ckpt_name(cur_step):
    return f"checkpoint_{cur_step}.ckpt"

def get_file_length_in_elements(filename, dtype):
    """
    获取文件按照指定的数据类型一共有多大，用于计算一共需要多少步才能遍历完数据
    """
    dtype = np.dtype(dtype)
    file_size = os.path.getsize(filename)      # 总字节数
    data_bytes = file_size            # 真正数据部分字节数
    if data_bytes % dtype.itemsize != 0:
        raise ValueError("文件大小不能被 dtype.itemsize 整除，可能有残缺或头信息错误")
    return data_bytes // dtype.itemsize

def load_batch(split:str='train', offset:int=0, tar_device:str= 'cuda:0'):
    if split == 'train':
        path = config['train_path']
    else:
        path = config['eval_path']
    # 这里有个问题，每次读的shape大小应该为多少
    data = np.memmap(path, dtype=np.uint16, mode='r', offset=offset * np.dtype(np.uint16).itemsize,
                     shape=cal_load_step())
    if data.shape[0] < cal_load_step():
        return None
    return get_batch(data, config['batch_size'], config['context_length'], tar_device)


def evaluate():
    print("开始eval")
    total_perplexity = 0.0
    cur_step = 1
    total_len = get_file_length_in_elements(config['eval_path'], np.uint16)
    load_step = cal_load_step()
    with torch.no_grad():
        offset = 0
        while offset + load_step + 1 < config['eval_ratio'] * total_len:
            print(f"eval进度：{(offset+load_step)/(config['eval_ratio']*total_len)*100}%")
            eval_data = load_batch("eval",offset,device)
            x = eval_data[0].to(dtype=torch.long,device=device)
            label = eval_data[1].to(dtype=torch.long,device=device)
            output = model(x)
            loss = cross_entropy(output, label)
            perplexity = batch_perplexity(loss)
            total_perplexity += perplexity.mean().item()
            offset += load_step
            cur_step += 1
    print(f"eval进度：100%")
    return total_perplexity / cur_step

def main():
    """
    训练循环：加载数据并进行训练
    """
    # 1. 首先计算一个epoch需要多少步
    load_step = cal_load_step()
    total_len = get_file_length_in_elements(config['train_path'], np.uint16) * config['train_ratio']
    one_epoch_steps = total_len // load_step
    total_num_steps = one_epoch_steps * config['num_epoch']
    print(f"load_step:{load_step}, total_len:{total_len}, num_steps:{total_num_steps}")
    # 2. 开始训练循环，可以从头开始也可以从上次的点继续
    epoch = 0
    # offset为读取数据文件的偏移量
    offset = 0
    if args.resume:
        epoch = config['cur_step'] // one_epoch_steps
        offset = config['cur_step'] % one_epoch_steps
    # 3. 设置进度条
    pbar = tqdm(total=total_num_steps)
    while epoch < config['num_epoch']:
        while offset+load_step < total_len:
            data = load_batch("train",offset,device)
            if data is None:
                break
            cur_step = offset // load_step + one_epoch_steps * epoch
            # 1. 调度学习率
            lr = cosine_lr_scheduler(cur_step,config['max_lr'],config['min_lr'],warmup_iters=total_num_steps*config['warmup_ratio'],cosine_cycle_iters=config['cosine_cycle_ratio']*total_num_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            #  2. 加载数据
            x = data[0].to(dtype=torch.long)
            label = data[1].to(dtype=torch.long)
            # 3. 前向传播
            output = model(x)
            loss = cross_entropy(output, label)
            # 4. 反向传播 梯度裁剪
            loss = loss.mean()
            loss.backward()
            gradient_clipping(model.parameters(), max_l2_norm=config['max_l2_norm'])
            # 5. 优化器优化
            optimizer.step()
            optimizer.zero_grad()
            # 6. 获取到当前步数，并且进行日志输出以及检查点保存
            pbar.update(1)
            pbar.set_description(f"step:{cur_step}")
            pbar.set_postfix({
                'loss':loss.item(),
                'total_step':total_num_steps,
            })
            if cur_step != 0 and cur_step % config['log_iter'] == 0 and config['enable_wandb']:
                weight_norms = get_weight_norms(model)
                grad_norms = get_grad_norms(model)
                log_data = {
                    "train/loss": loss,
                    "train/lr": lr,
                    "train/weight_norms":weight_norms,
                    "train/grad_norms":grad_norms,
                    "step":cur_step,
                }
                if config['enable_eval'] and cur_step % config['eval_iter'] == 0:
                    eval_loss = evaluate()
                    log_data["eval/perplexity"] = eval_loss
                # 日志记录
                wandb.log(log_data)
            if cur_step!=0 and cur_step % config['save_iter'] == 0:
                save_path = os.path.join(config['save_dir'],get_ckpt_name(cur_step))
                print(save_path)
                save_checkpoint(model,optimizer,cur_step,save_path)
                print("检查点保存成功")
            offset += load_step
        offset = 0
        epoch = epoch + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练脚本")
    parser.add_argument("--resume", action="store_true", help="是否继续上次的结果训练")
    parser.add_argument("--config",type=str,default='config.yaml')
    args,_ = parser.parse_known_args()
    # 1. 加载配置以及上次训练保存的各种参数
    config = yaml.safe_load(open('config.yaml'))
    print("\n")
    print(json.dumps(config,indent=2))
    if config['enable_wandb']:
        init_wandb(config)
    # 2. 配置设备
    device = get_device()
    # 3. 设置随机种子
    set_manual_seed(config['seed'])
    # 4. 加载模型
    model = Transformer(**config)
    model.to(device)
    optimizer = AdamWOptimizer(model.parameters(),betas=(config['betas'][0],config['betas'][1]),weight_decay=config['weight_decay'], eps=config['eps'])
    if args.resume:
        ckpt_path = os.path.join(config['save_dir'],find_latest_checkpoint(config['save_dir']))
        print("加载检查点{}".format(ckpt_path))
        config['cur_step'] = load_checkpoint(ckpt_path,model,optimizer)
        print(config['cur_step'])
    print(count_module_parameters(model))
    # 5. 加载分词器
    tokenizer = BPETokenizer.from_files(config['vocab_file_path'],config['merges_file_path'])
    main()