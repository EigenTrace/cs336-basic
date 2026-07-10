import torch
from torch import nn
from einops import einsum
from math import sqrt
class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        p=torch.empty( out_features, in_features, device=device,dtype=dtype)
        std=sqrt(2/(in_features+out_features))
        np=torch.nn.init.trunc_normal_(p,mean=0,std=std,a=-3*std,b=3*std)
        self.W = nn.Parameter(np)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x,self.W,"... in,out in -> ... out")
