
from typing import Any

import torch
from torch import nn

class Pffn(nn.Module):
    def __init__(self,d_model:int,dff:int,device=None,dtype=None) -> None:
        super().__init__()
        self.w1=nn.Parameter(torch.empty(dff,d_model,device=device,dtype=dtype))
        self.w3=nn.Parameter(torch.empty(dff,d_model,device=device,dtype=dtype))
        self.w2=nn.Parameter(torch.empty(d_model,dff,device=device,dtype=dtype))
        var=2/(d_model+dff)
        std=var**0.5
        torch.nn.init.trunc_normal_(self.w1,0,std,-3*std,3*std)
        torch.nn.init.trunc_normal_(self.w2,0,std,-3*std,3*std)
        torch.nn.init.trunc_normal_(self.w3,0,std,-3*std,3*std)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        mid=x@self.w1.T
        silu=mid*torch.sigmoid(mid)
        mid3=x@self.w3.T
        mlt=silu*mid3
        result=mlt@self.w2.T
        return result