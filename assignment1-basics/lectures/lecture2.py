import torch
import torch.nn.functional as F
import timeit
import torch
from typing import Iterable
from torch import nn
import numpy as np
from jaxtyping import Float
from einops import rearrange, einsum, reduce


def get_memory_usage(x: torch.Tensor):
    return x.numel() * x.element_size()


def tensor_create():
    # 1. 不同方式创建张量
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # @inspect x
    x = torch.zeros(4, 8)  # 4x8 matrix of all zeros @inspect x
    x = torch.ones(4, 8)  # 4x8 matrix of all ones @inspect x
    # 4x8 matrix of iid Normal(0, 1) samples @inspect xx = torch.tensor()
    x = torch.randn(4, 8)


def tensor_mem():
    # 2. 张量内存占用
    # float32
    x = torch.ones(4, 8)
    assert x.dtype == torch.float32
    assert x.numel() == 4*8
    assert x.element_size() == 4
    assert get_memory_usage(x) == 4*8*4

    # float16会出现下溢
    x = torch.tensor([1e-8], dtype=torch.float16)
    assert x == 0

    # bfloat16对于相同的精度不会出现下溢
    x = torch.tensor([1e-8], dtype=torch.bfloat16)
    # assert x == 0

    float32_info = torch.finfo(torch.float32)
    float16_info = torch.finfo(torch.float16)
    bfloat16_info = torch.finfo(torch.bfloat16)
    print(float32_info)
    print(float16_info)
    print(bfloat16_info)


def tensor_on_gpu():
    x = torch.zeros(32, 32)
    if not torch.cuda.is_available():
        return
    num_gpus = torch.cuda.device_count()  # @inspect num_gpus
    for i in range(num_gpus):
        properties = torch.cuda.get_device_properties(i)  # @inspect properties
    print(properties)
    memory_allocated = torch.cuda.memory_allocated()  # @inspect memory_allocated
    y = x.to("cuda:0")
    assert y.device == torch.device("cuda", 0)
    z = torch.zeros(32, 32, device="cuda:0")
    # @inspect new_memory_allocated
    new_memory_allocated = torch.cuda.memory_allocated()
    memory_used = new_memory_allocated - memory_allocated  # @inspect memory_used
    # 2 32x32 matrices of 4-byte floats
    assert memory_used == 2 * (32 * 32 * 4)


def tensor_stride():
    x = torch.range(1, 16, 1).view(2, 2, 4)
    assert x.shape == (2, 2, 4)
    print(x.shape)
    print(x)
    print(x.transpose(0, 1))
    print(f"x.stride(0):{x.stride(0)}")
    print(f"x.stride(1):{x.stride(1)}")
    print(f"x.stride(2):{x.stride(2)}")


def tensor_slice():
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # @inspect x
    y = x[0]
    assert torch.equal(y, torch.tensor([1., 2, 3]))
    x[0][0] = 100
    # 切片操作并没有复制张量
    print(y[0])
    # 转置操作并没有复制张量
    y = x.transpose(0, 1)
    x[0][1] = 200
    print(y[1][0])
    # view操作
    y = x.view(3, 2)
    x[1][0] = 300
    print(x)
    print(y)
    # 有些操作会使张量变得不contiguous，这时不能继续进行view操作
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # @inspect x
    y = x.transpose(1, 0)  # @inspect y
    assert not y.is_contiguous()
    try:
        y.view(2, 3)
        assert False
    except RuntimeError as e:
        assert "view size is not compatible with input tensor's size and stride" in str(
            e)


def ein_sum():
    x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(
        2, 3, 4)  # @inspect x
    y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(
        2, 3, 4)  # @inspect y
    # @inspect z
    z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
    print(x)
    print(y)
    print(z)


def ein_reduce():
    x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)
    print(x)
    y = x.mean(dim=-2)
    print(y)
    z = reduce(x, "... seq hidden -> ... hidden", reduction="mean")
    print(z)


def ein_rearrange():
    x: Float[torch.Tensor, "batch seq hidden"] = torch.randn(2, 3, 4)
    print(x)
    # 多个维度合并成一个
    y = rearrange(x, "... seq hidden -> ... (seq hidden)")
    print(y)
    # 一个维度拆分成多个
    z = rearrange(x, "... (heads dim) -> ... heads dim", heads=2)
    print(z)
    w = rearrange(x, "... seq hidden -> ... hidden seq")
    print(w)


def time_matmul(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the number of seconds required to perform `a @ b`."""
    # Wait until previous CUDA threads are done
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def run():
        # Perform the operation
        a @ b
        # Wait until CUDA threads are done
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    # Time the operation `num_trials` times
    num_trials = 5
    total_time = timeit.timeit(run, number=num_trials)
    return total_time / num_trials


def forward_flops():
    if torch.cuda.is_available():
        B = 16384  # Number of points
        D = 32768  # Dimension
        K = 8192   # Number of outputs
    else:
        B = 1024
        D = 256
        K = 64
    device = "cuda:0"
    x = torch.ones(B, D, device=device)
    w = torch.randn(D, K, device=device)
    y = x @ w
    # 前向传播的FLOPs：2*数据点数量*参数量
    actual_num_flops = 2 * B * (D * K)
    actual_time = time_matmul(x, w)
    print(actual_time)
    actual_flop_per_sec = actual_num_flops / actual_time
    # 8.8e12
    print(actual_flop_per_sec)
    promised_flop_per_sec = 8.2e13
    # 查阅相关资料可得4090理想FLOPS为8.2e13
    mfu = actual_flop_per_sec / promised_flop_per_sec
    print(mfu)


def gradient():
    if torch.cuda.is_available():
        B = 16384  # Number of points
        D = 32768  # Dimension
        K = 8192   # Number of outputs
    else:
        B = 1024
        D = 256
        K = 64
    device = "cuda:0"
    x = torch.ones(B, D, device=device)
    w1 = torch.randn(D, D, device=device, requires_grad=True)
    w2 = torch.randn(D, K, device=device, requires_grad=True)
    h1 = x @ w1
    h2 = h1 @ w2
    loss = h2.pow(2).mean()
    assert w2.grad.size() == torch.Size([D, K])
    assert h1.size() == torch.Size([B, D])
    assert h2.grad.size() == torch.Size([B, K])


class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(SGD, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                grad = p.grad.data
                p.data -= lr * grad


class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                # Optimizer state
                state = self.state[p]
                grad = p.grad.data
                # Get squared gradients g2 = sum_{i<t} g_i^2
                g2 = state.get("g2", torch.zeros_like(grad))
                # Update optimizer state
                g2 += torch.square(grad)
                state["g2"] = g2
                # Update parameters
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)


def get_batch(data: np.array, batch_size: int, sequence_length: int, device: str) -> torch.Tensor:
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,))
    assert start_indices.size() == torch.Size([batch_size])
    x = torch.tensor([data[start:start + sequence_length]
                     for start in start_indices])
    assert x.size() == torch.Size([batch_size, sequence_length])
    if torch.cuda.is_available():
        x = x.pin_memory()
    x = x.to(device, non_blocking=True)
    return x


def checkpointing():
    # Training language models take a long time and certainly will certainly crash.
    # You don't want to lose all your progress.
    # During training, it is useful to periodically save your model and optimizer state to disk.
    model = Cruncher(dim=64, num_layers=3).to(get_device())
    optimizer = AdaGrad(model.parameters(), lr=0.01)
    # Save the checkpoint:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, "model_checkpoint.pt")
    # Load the checkpoint:
    loaded_checkpoint = torch.load("model_checkpoint.pt")


# tensor_stride()
# tensor_slice()
# ein_sum()
# ein_reduce()
# ein_rearrange()
# forward_flops()
gradient()
