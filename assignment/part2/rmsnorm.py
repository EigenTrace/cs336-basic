import torch
from torch import nn
from einops import einsum

class Rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.dim=d_model
        self.e=eps
        self.g=nn.Parameter(torch.ones(1,d_model,device=device,dtype=dtype))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type=x.dtype
        x=x.to(torch.float32)
        rms=einsum(x,x,"... d, ... d -> ...")
        rms=torch.sqrt(rms/self.dim+self.e).unsqueeze(-1)
        result=x/rms*self.g

        return result.to(in_type)

