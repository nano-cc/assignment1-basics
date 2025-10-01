import torch
import torch.nn as nn
from .modules import Embedding, RotaryPositionalEmbedding, TransformerBlock, RMSNorm, Linear, softmax


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
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
