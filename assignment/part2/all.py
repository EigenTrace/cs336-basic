from typing import Any

import torch

from torch import nn
from jaxtyping import Bool, Float, Int
from einops import einsum, rearrange
from .softmax import Softmax
from torch import Tensor
from .rope import Rope
from .rmsnorm import Rmsnorm
from .positionwise_feedforward import Pffn
from .embedding import Embedding
from .linear import Linear


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    qk = einsum(Q, K, "... query d_k, ... keys d_k -> ... query keys")
    qk = qk / (Q.shape[-1]) ** 0.5
    if mask is not None:
        qk = qk.masked_fill(~mask, float("-inf"))
    softmax = Softmax()
    qk = softmax(qk, dim=-1)
    qkv = einsum(qk, V, "... queries keys, ... keys d_v->... queries d_v")
    return qkv


class multihead_self_attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        self.d = d_model
        self.num = num_heads
        self.Wq = nn.Parameter(torch.empty([d_model, d_model]))
        self.Wk = nn.Parameter(torch.empty([d_model, d_model]))
        self.Wv = nn.Parameter(torch.empty([d_model, d_model]))
        self.Wo = nn.Parameter(torch.empty([d_model, d_model]))
        std = (1 / d_model) ** 0.5
        torch.nn.init.trunc_normal_(self.Wq, mean=0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.Wk, mean=0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.Wv, mean=0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.Wo, mean=0, std=std, a=-3 * std, b=3 * std)
        self.dk = int(d_model / num_heads)

        if theta and max_seq_len:
            self.rope = Rope(theta, self.dk, max_seq_len)
        else:
            self.rope = None

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        Q = einsum(self.Wq, in_features, "d_out d_in, ... seq d_in -> ... seq d_out")
        K = einsum(self.Wk, in_features, "d_out d_in, ... seq d_in -> ... seq d_out")
        V = einsum(self.Wv, in_features, "d_out d_in, ... seq d_in -> ... seq d_out")

        Q = rearrange(Q, "... seq (h dk)-> ... seq h dk", dk=self.dk)
        K = rearrange(K, "... seq (h dk)-> ... seq h dk", dk=self.dk)
        V = rearrange(V, "... seq (h dk)-> ... seq h dk", dk=self.dk)

        mask = torch.tril(torch.ones(Q.shape[-3], Q.shape[-3], dtype=torch.bool))
        outs = []
        for i in range(self.num):
            print("Q type")
            print(Q.shape)
            print("qi")

            qi, ki, vi = Q[:, :, i], K[:, :, i], V[:, :, i]
            print(qi.shape)
            if self.rope:
                qi = self.rope.forward(qi, token_positions)
                ki = self.rope.forward(ki, token_positions)
            outs.append(scaled_dot_product_attention(qi, ki, vi, mask))
        result = torch.stack(outs, dim=-2)
        result = rearrange(result, "... seq h dk -> ... seq (h dk)")
        return einsum(result, self.Wo, "... seq d, d_out d->... seq d_out")


class transformer_block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = Rmsnorm(d_model)
        self.mulatt = multihead_self_attention(d_model, num_heads, theta, max_seq_len)
        self.norm2 = Rmsnorm(d_model)
        self.pffn = Pffn(d_model, d_ff)

    def forward(
        self, in_features: Float[Tensor, " batch sequence_length d_model"]
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        x = self.norm1(in_features)
        position = torch.arange(in_features.shape[-2])
        x1 = self.mulatt.forward(x, position)
        x2 = x1 + in_features
        x3 = self.norm2(x2)
        x4 = self.pffn.forward(x3)
        return x4 + x2


class transformer_lm(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.attns = nn.ModuleList(
            [
                transformer_block(d_model, num_heads, d_ff, rope_theta, context_length)
                for _ in range(num_layers)
            ]
        )
        self.norm = Rmsnorm(d_model)
        self.linear = Linear(d_model, vocab_size)
        self.softmax = Softmax()

    def forward(
        self, x: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.embedding(x)
        for attn in self.attns:
            x=attn(x)
        x=self.norm(x)
        x=self.linear(x)
        return x
