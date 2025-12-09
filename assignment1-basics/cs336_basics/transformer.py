import torch
import torch.nn as nn
from cs336_basics.modules import Embedding, RotaryPositionalEmbedding, TransformerBlock, RMSNorm, Linear, softmax
from collections import defaultdict


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float = 10000.0,**kwargs):
        super().__init__()
        # 定义需要的模块
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model)
        rope = RotaryPositionalEmbedding(
            theta=rope_theta, d_k=d_model//num_heads, max_seq_len=context_length)
        self.layers = nn.ModuleList([TransformerBlock(
            d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=rope) for i in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(
            in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        # 前向传播这里只需要返回logits即可，不需要进行softmax
        return x


def count_module_parameters(model: nn.Module) -> dict:
    """
    使用 PyTorch API 统计模型中每个命名子模块的参数总量。

    Args:
        model (nn.Module): 待统计的 PyTorch 模型。

    Returns:
        dict: 包含每个模块参数量的字典。
    """
    # 存储每个命名模块的参数总量
    module_params = defaultdict(int)
    # 存储所有参数的总量
    total_params = 0

    # 遍历所有命名参数 (name, parameter)
    for name, param in model.named_parameters():
        # 如果参数是可训练的（默认是 True）
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params

            # 提取模块的名称 (例如: 'layers.0.attn.q_proj' -> 'layers.0.attn')
            parts = name.split('.')

            # 对于具有多个子层的 Block，我们只统计到 Block 内部组件级别 (例如: attn, ffn)
            if len(parts) >= 3 and parts[0] == 'layers':
                # 提取 'layers.i.sub_module'
                module_name = "Block_Layer " + parts[1] + " - " + parts[2]
            else:
                # 对于顶层模块 (embed, final_norm, lm_head)
                module_name = parts[0]

            module_params[module_name] += num_params

    # 格式化输出
    formatted_results = {}
    for name, count in module_params.items():
        if count >= 1_000_000:
            formatted_results[name] = f"{count:,} ({round(count / 1_000_000, 2)} M)"
        else:
            formatted_results[name] = f"{count:,}"

    formatted_results['Total_Model_Parameters'] = f"{total_params:,} ({round(total_params / 1_000_000_000, 3)} B)"

    return formatted_results

if __name__ == '__main__':
    pass
