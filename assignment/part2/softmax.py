from typing import Any

import torch
from torch import nn

class Softmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,tensor:torch.Tensor,dim:int)->torch.Tensor:

        maxnum=tensor.max(dim=dim,keepdim=True).values
        sub=tensor-maxnum
        exp=sub.exp()
        s=exp.sum(dim=dim,keepdim=True)
        return exp/s
        
