# Transformer用到的组件
import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce
from jaxtyping import Float, Int
import math
from typing import Iterable, Union
import os
import typing


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 使用softmax的最大值归一化，需要softmax的张量减去最大值其余的值都是0或者负数，而且最后的值不变
    # 1. 获取到dim上对应的最大值
    # 2. 该dim减掉对应的最大值
    x = x-x.max(dim=dim, keepdim=True).values
    # 3. 计算该dim上的softmax
    exp = torch.exp(x)
    x = exp / exp.sum(dim=dim, keepdim=True)
    return x


def scaled_dot_product_attention(Q: Float[torch.Tensor, "batch_size ... seq_len d_k"], K: Float[torch.Tensor, "batch_size ... seq_len d_k"], V: Float[torch.Tensor, "batch_size ... seq_len d_v"], mask: torch.Tensor):
    d_k = Q.shape[-1]
    attn_score = einsum(
        Q, K, "... q_seq_len d_k,... k_seq_len d_k -> ... q_seq_len k_seq_len")
    attn_score /= d_k ** 0.5
    attn_score = attn_score.masked_fill(mask == False, float('-inf'))
    attn_score = softmax(attn_score, dim=-1)
    attn_output = einsum(
        attn_score, V, "... q_seq_len k_seq_len , ... k_seq_len d_v -> ... q_seq_len d_v")
    return attn_output


def cross_entropy(inputs: Float[torch.Tensor, "... batch_size vocab_size"], targets: Int[torch.Tensor, "... batch_size"]) -> torch.Tensor:
    """_summary_
    这个交叉熵函数是计算的batch中一个token对应的交叉熵
    """
    # 这里虽然是-logsoftmax，但是没有直接使用softmax
    # 首先 exp 和 log 应该尽可能抵消
    target_logit: Float[torch.Tensor, "... batch_size 1"] = torch.gather(
        inputs, dim=-1, index=targets.unsqueeze(-1))
    # 然后这里同样是使用了把最大值减掉的技巧以确保数值稳定性
    max_logits: Float[torch.Tensor, "... batch_size"] = reduce(
        inputs, "... batch_size vocab_size -> ... batch_size", reduction="max").unsqueeze(-1)
    scaled_inputs = inputs - max_logits
    exp_sum = torch.exp(scaled_inputs).sum(dim=-1, keepdim=True)
    batch_cross_entropy = max_logits + torch.log(exp_sum) - target_logit
    # return reduce(batch_cross_entropy.squeeze(-1), "... batch_size -> ...", reduction="mean")
    return batch_cross_entropy.squeeze(-1)


def batch_perplexity(entropy_loss: Float[torch.Tensor, "... batch_size seq_len"]) -> torch.Tensor:
    """
    计算每个序列的困惑度（每个序列平均每个词面临多少个等概率的选项）
    """
    mean_cross_entropy = reduce(
        entropy_loss, "... batch_size seq_len -> ... batch_size", reduction="mean")
    return torch.exp(mean_cross_entropy)


def cosine_lr_scheduler(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + (max_learning_rate - min_learning_rate)*(1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) / 2
    return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    # 这里开始实现错了，针对每个梯度单独计算l2范数并缩放，应该考虑所有的梯度，计算l2范数并缩放
    l2_norm = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        l2_norm += p.grad.data.pow(2).sum().item()
    l2_norm = math.sqrt(l2_norm)
    if l2_norm <= max_l2_norm:
        return
    clip_coeff = max_l2_norm / (l2_norm+eps)
    # 确保缩放因子不大于 1
    for p in parameters:
        if p.grad is None:
            continue
        p.grad.data.mul_(clip_coeff)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, typing.BinaryIO]
) -> None:
    """
    保存模型的、优化器的状态字典以及当前的迭代次数。

    Args:
        model: PyTorch 模型 (nn.Module)。
        optimizer: PyTorch 优化器 (optim.Optimizer)。
        iteration: 当前的训练迭代次数 (int)。
        out: 输出路径 (str, PathLike) 或文件对象 (BinaryIO)。
    """
    # 1. 创建一个字典，包含所有需要保存的状态
    checkpoint_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }

    # 2. 使用 torch.save 将字典保存到文件
    # 注意：out 可以是路径字符串，也可以是文件对象
    torch.save(checkpoint_dict, out)


def load_checkpoint(
    src: Union[str, os.PathLike, typing.BinaryIO],
    model: nn.Module,
    optimizer: typing.Optional[torch.optim.Optimizer]
) -> int:
    """
    从源加载检查点，恢复模型和优化器的状态，并返回保存的迭代次数。

    Args:
        src: 检查点源路径 (str, PathLike) 或文件对象 (BinaryIO)。
        model: PyTorch 模型 (nn.Module)，将被原地修改。
        optimizer: PyTorch 优化器 (optim.Optimizer)，将被原地修改。

    Returns:
        保存的迭代次数 (int)。
    """

    # 1. 使用 torch.load 加载整个检查点字典
    # 注意：建议使用 map_location 参数来确保模型加载到正确的设备
    # 例如：map_location='cuda:0' 或 map_location='cpu'
    checkpoint_dict = torch.load(src)

    # 2. 恢复模型状态
    # strict=True (默认) 确保加载的键与模型中的键完全匹配
    model.load_state_dict(checkpoint_dict['model_state_dict'])

    # 3. 恢复优化器状态
    # 优化器状态必须在模型参数加载后才能恢复
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

    # 4. 返回迭代次数
    iteration = checkpoint_dict['iteration']

    return iteration


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters"""
        # 1. 调用父类初始化方法
        super().__init__()
        # 2. 创建权重张量
        factory_kwargs = {
            "device": device,
            "dtype": dtype
        }
        # 创建张量不应该使用torch.Tensor
        weight_tensor: Float[torch.Tensor, "out in"] = torch.empty(
            out_features, in_features, **factory_kwargs)
        # 3. 创建之后需要初始化，现在的LLM基本不使用bias
        # TODO 为什么这里使用这个值初始化
        std_dev = torch.sqrt(torch.tensor(
            2.0, **factory_kwargs)/(in_features+out_features))
        weight_tensor = nn.init.trunc_normal_(
            weight_tensor, mean=0, std=std_dev.item())
        # 4. 注册为Parameter
        self.weight = nn.Parameter(weight_tensor, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        return einsum(self.weight, x, "out in_dim,... seq_len in_dim -> ... seq_len out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """Construct an embedding module.
        This function should accept the following parameters:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters"""
        # 1. 调用父类初始化方法
        super().__init__()
        # 2. 创建并初始化自己的参数
        kwargs = {
            "device": device,
            "dtype": dtype
        }
        embedding_tensor: Float[torch.Tensor, "vocab_size d_model"] = torch.empty(
            num_embeddings, embedding_dim, **kwargs)
        embedding_tensor = nn.init.trunc_normal_(
            embedding_tensor, mean=0, std=1)
        # 3. 注册为Parameter
        self.weight = nn.Parameter(embedding_tensor, requires_grad=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 这里需要完成一个查表操作，相当于把每个id对应的tensor取出来
        # 查阅资料需要先转换成one hot向量，再做矩阵乘法
        # 不需要这么做，可以直接索引,相当于取每一个整数作为embedding第一维的索引，然后需要索引的张量前几维保留
        # 这里需要注意，索引张量尽可能应为long类型
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters"""
        # 1. 调用父类初始化方法
        super().__init__()
        # 2. 定义并初始化自己的参数
        kwargs = {
            "device": device,
            "dtype": dtype
        }
        g_tensor = torch.ones(d_model, **kwargs)
        # 3. 注册为Parameter
        self.weight = nn.Parameter(g_tensor, requires_grad=True)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape
            (batch_size, sequence_length, d_model) and return a tensor of the same shape."""
        # 需要将dtype临时转换成float32避免在平方的时候overflow
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # 计算均方
        mean_square = reduce(
            x.pow(2.0), "... d_model ->...", reduction="mean")
        x = x * torch.rsqrt(mean_square+self.eps).unsqueeze(-1) * self.weight
        return x.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # 直接使用Linear层
        kwargs = {
            "device": device,
            "dtype": dtype
        }
        # 我这里错误地使用了Parameter包装了Linear，实际上只有张量才能用Parameter包装
        self.w1: Float[torch.Tensor, "d_model d_ff"] = Linear(
            d_model, d_ff, **kwargs)
        self.w2: Float[torch.Tensor, "d_ff d_model"] = Linear(
            d_ff, d_model, **kwargs)
        self.w3: Float[torch.Tensor, "d_model d_ff"] = Linear(
            d_model, d_ff, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 按元素乘法直接相乘即可，不需要用einsum
        swiglu_res = self.w1(x)*torch.sigmoid(self.w1(x)) * self.w3(x)
        return self.w2(swiglu_res)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on"""
        super().__init__()
        # 需要计算出旋转位置编码矩阵，shape:s,d/2,2,2
        # 1. 首先计算出矩阵的行和列，然后矩阵乘法
        assert d_k % 2 == 0
        half_dim = d_k // 2
        col = torch.arange(0, half_dim, device=device).float().view(1, -1)
        row = torch.arange(0, max_seq_len, device=device).float().view(-1, 1)
        # 这里d_k不能像seq_len一样先定义一个大的然后截断处理更小的
        theta_mat = 1.0 / theta ** ((2*col)/d_k)
        pos_cis = row @ theta_mat
        assert pos_cis.shape == (max_seq_len, half_dim)
        sin_mat = torch.sin(pos_cis).float()
        cos_mat = torch.cos(pos_cis).float()
        # 这里根据gemini修改了堆叠顺序
        # 构建 [cos, -sin] 行
        # shape: (max_seq_len, half_dim, 2)
        row1 = torch.stack((cos_mat, -sin_mat), dim=-1)

        # 构建 [sin, cos] 行
        # shape: (max_seq_len, half_dim, 2)
        row2 = torch.stack((sin_mat, cos_mat), dim=-1)

        # 堆叠成 2x2 矩阵 R_k^i
        # shape: (max_seq_len, half_dim, 2, 2)
        pos_cis = torch.stack((row1, row2), dim=-2)
        self.register_buffer("pos_cis", pos_cis, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension."""
        d_k = x.shape[-1]
        assert d_k % 2 == 0
        half_dim = d_k // 2
        x = rearrange(x, "... (half_d_k two) -> ... half_d_k two",
                      two=2).unsqueeze(-1)
        x = self.pos_cis[token_positions, :half_dim] @ x
        x = rearrange(x.squeeze(-1), "... half_d_k two -> ... (half_d_k two)")
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding = None):
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_model // num_heads
        self.d_k = d_k
        # 这里 q k v的头数相同
        self.q_proj = Linear(d_model, d_k*num_heads)
        self.k_proj = Linear(d_model, d_k*num_heads)
        self.v_proj = Linear(d_model, d_k*num_heads)
        self.output_proj = Linear(d_k*num_heads, d_model)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 4,12,64
        seq_len = x.shape[-2]
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 这里踩了大坑，rearrange如果要分解或者合并一个或者两个维度，写的顺序很重要,下面三个顺序都写反了
        Q = rearrange(
            Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", d_k=self.d_k)
        K = rearrange(
            K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", d_k=self.d_k)
        V = rearrange(
            V, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", d_k=self.d_k)
        if self.rope is not None:
            pos = torch.arange(0, seq_len).view(1, -1)
            Q = self.rope(Q, pos)
            K = self.rope(K, pos)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)) == 1
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = rearrange(
            attn_output, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        return self.output_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding):
        super().__init__()
        # 定义需要的模块
        self.ln1 = RMSNorm(d_model=d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, rope=rope)
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre norm -> attn -> resid
        x = x + self.attn(self.ln1(x))
        # pre norm -> swiglu -> resid
        x = x + self.ffn(self.ln2(x))
        return x


if __name__ == '__main__':
    a = torch.arange(0, 16).view(2, 8)
    b = torch.tensor([1, 2]).view(2)
    print(a)
    b = b.unsqueeze(-1)
    print(b)
    print(torch.gather(a, -1, b))
