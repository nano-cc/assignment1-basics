import torch
import torch.nn as nn
from typing import Optional, Callable
import math


class SGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = {
            "lr": lr
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        # 这里可以有多个参数组，每个参数组应用的超参数不同
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                t = group.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                group['t'] = t+1
        # 总结一下优化器实现的流程
        # 1. 遍历参数组，把每个参数组的超参数取出来
        # 2. 对参数组的参数按照优化器公式进行原地更新，直接取 p.data 或者 p.grad.data
        # 3. 可以修改参数组里面超参数的取值
        return loss


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, betas: tuple, eps=1e-8, weight_decay=0.01, lr=1e-5):
        # 注意这里params是一个列表或者字典列表，defaults是不同参数组公共参数的字典
        defaults = {
            'beta_1': betas[0],
            'beta_2': betas[1],
            'lr': lr,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta_1']
            beta2 = group['beta_2']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                # 接下来就是需要具体执行AdamW的更新过程
                # 1. 首先，应该获取这个参数相关的状态
                state = self.state[p]
                # 2. 如果状态为空，需要初始化
                if len(state) == 0:
                    state['m'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                # 3. 按照步骤进行更新操作
                t = group.get('t', 1)
                m = state['m']
                v = state['v']
                grad = p.grad.data
                # 使用原地操作，更加高效
                m.mul_(beta1).add_(grad, alpha=1.0-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                lr_t = lr * math.sqrt(1.0-beta2 ** t) / (1.0-beta1 ** t)
                p.data.addcdiv_(m, torch.sqrt(v)+eps, value=-lr_t)
                p.data.add_(p.data, alpha=-lr*weight_decay)
                # 4. 更新状态
                group['t'] = t + 1
        return loss


if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGDOptimizer([weights], lr=1e3)
    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
