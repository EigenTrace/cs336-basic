import torch
from torch import nn


class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        position = torch.arange(0, max_seq_len, device=device)

        k = torch.arange(0, d_k, 2, device=device) / d_k
        k = (theta) ** (-k)
        angle = torch.outer(position, k)

        self.register_buffer("cos", angle.cos(), persistent=False)
        self.register_buffer("sin", angle.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        r_even = self.cos[token_positions] * x_even - self.sin[token_positions] * x_odd
        r_odd = self.sin[token_positions] * x_even + self.cos[token_positions] * x_odd

        r = torch.stack((r_even, r_odd), dim=-1)
        r = r.flatten(-2)
        return r
