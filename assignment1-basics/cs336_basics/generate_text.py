import yaml
import os
import json
import torch
import torch.nn.functional as F
from typing import Optional
from utils import get_device,set_manual_seed,find_latest_checkpoint
from transformer import Transformer,count_module_parameters
from modules import load_checkpoint
from tokenizer import BPETokenizer

@torch.no_grad()
def generate_text(
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token: Optional[str] = "<endoftext>",
) -> str:
    """
    使用给定的语言模型从 prompt 生成文本。

    Args:
        model: 你的 TransformerLM 模型
        tokenizer: BPETokenizer，带 encode / decode
        prompt: 初始提示文本
        max_new_tokens: 最多生成多少个新 token
        temperature: 温度系数; =0 表示贪心解码
        top_p: nucleus / top-p 采样阈值，(0, 1]，=1.0 表示不用截断
        eos_token: 终止符号，对应 tokenizer 里某个 token（可以为 None 表示不使用）
        device: "cuda" 或 "cpu"，默认自动检测

    Returns:
        生成的完整文本（包含 prompt）
    """

    model.eval()

    device = next(model.parameters()).device

    # 1. 把 prompt 编成 token 序列
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)

    # 终止 token id（如果提供了 eos_token）
    eos_id = None
    if eos_token is not None:
        try:
            eos_id = tokenizer.encode(eos_token)
            # encode 可能返回多个 id，这里只取第一个，如果你确信是单个可以这样用
            eos_id = eos_id[0]
        except Exception:
            eos_id = None  # tokenizer 不认识就忽略

    # 2. 循环生成
    for _ in range(max_new_tokens):
        # 模型前向：拿到所有时间步的 logits
        # 假设 model(input_ids) -> (seq_len, vocab_size)
        logits = model(input_ids)

        # 如果你的模型返回的是 (1, seq_len, vocab_size)，改成：
        # logits = model(input_ids.unsqueeze(0))[0]

        # 只取最后一个时间步的 logits: v \in R^{vocab_size}
        next_token_logits = logits[-1]  # shape: (vocab_size,)

        # 3. 温度缩放
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        else:
            # temperature = 0 -> 直接贪心
            next_token_id = int(torch.argmax(next_token_logits))
            input_ids = torch.cat(
                [input_ids, torch.tensor([next_token_id], device=device)]
            )
            if eos_id is not None and next_token_id == eos_id:
                break
            continue

        # 4. 计算 softmax 概率
        probs = F.softmax(next_token_logits, dim=-1)  # (vocab_size,)

        # 5. top-p / nucleus 采样
        if 0 < top_p < 1.0:
            # 按概率从大到小排序
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)

            # 选出累计概率 >= top_p 的最小集合 V(p)
            mask = cumprobs <= top_p
            # 至少保留一个 token（防止 top_p 非常小导致全 False）
            mask[0] = True

            filtered_probs = sorted_probs * mask
            filtered_probs = filtered_probs / filtered_probs.sum()

            # 在截断后的分布中采样
            next_token_idx_in_sorted = torch.multinomial(filtered_probs, num_samples=1)
            next_token_id = sorted_indices[next_token_idx_in_sorted]
            next_token_id = int(next_token_id)
        else:
            # 不用 top-p，直接在 full softmax 里采样
            next_token_id = int(torch.multinomial(probs, num_samples=1))

        # 6. 把采样得到的 token 拼到序列后面
        input_ids = torch.cat(
            [input_ids, torch.tensor([next_token_id], device=device)]
        )

        # 7. 遇到结束符就停
        if eos_id is not None and next_token_id == eos_id:
            break

    # 8. 解码成字符串
    output_tokens = input_ids.tolist()
    return tokenizer.decode(output_tokens)

def load_ckpt_and_generate_text(save_dir:str, ckpt_name:str):
    # 2. 配置设备
    device = get_device()
    # 3. 设置随机种子
    set_manual_seed(config['seed'])
    # 4. 加载模型
    model = Transformer(**config)
    model.to(device)
    ckpt_path = os.path.join(save_dir, ckpt_name)
    print("加载检查点{}".format(ckpt_path))
    config['cur_step'] = load_checkpoint(ckpt_path, model, None)
    print(config['cur_step'])
    print(count_module_parameters(model))
    # 5. 加载分词器
    tokenizer = BPETokenizer.from_files(config['vocab_file_path'], config['merges_file_path'])
    print(generate_text(model, tokenizer, max_new_tokens=100, prompt="Once", ))


if __name__ == '__main__':
    # 1. 加载配置以及上次训练保存的各种参数
    config = yaml.safe_load(open('config.yaml'))
    load_ckpt_and_generate_text(config['save_dir'], find_latest_checkpoint(config['save_dir']))
