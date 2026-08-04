from collections.abc import Iterable
from typing import IO, Any, BinaryIO
import os
import typing
import numpy.typing as npt
import numpy as np
import torch
from assignment.part2.all import *
from torch import Tensor
from assignment.part1 import bpe
from assignment.part1.tokenizer import Tokenizer
from assignment.part2.linear import Linear
from torch import nn
from einops import einsum
from jaxtyping import Bool, Float, Int
import matplotlib.pyplot as plt
from math import cos
import time, json

def cross_entropy(
    inputs: torch.Tensor, targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:

    m = torch.max(inputs, dim=-1, keepdim=True).values
    shifted = inputs - m
    logsumex = m.squeeze(-1) + shifted.exp().sum(dim=-1).log()
    target_logit = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    loss = logsumex - target_logit  # (...,)  == -log p(target)
    return loss.mean()


from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: (
            Iterable[Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, Tensor]]
        ),
        lr,
        weight_decay,
        betas,
        eps,
    ) -> None:
        defaults = {"lr": lr, "decay": weight_decay, "beta": betas, "epi": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["beta"]
            decay = group["decay"]
            epi = group["epi"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                t = state["t"] + 1
                m = state["m"]
                v = state["v"]
                g = p.grad
                lrt = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                p.data = p.data - lr * decay * p.data
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g**2)
                p.data = p.data - lrt * m / (torch.sqrt(v) + epi)
                state["t"] = t
                state["m"] = m
                state["v"] = v
        return loss


def lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:

    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:

    ps = [p for p in parameters if p.grad is not None]
    l2 = torch.sqrt(sum([(p.grad**2).sum() for p in ps]))
    if l2 <= max_l2_norm:
        return
    scale = max_l2_norm / (l2 + 1e-6)
    for p in ps:
        p.grad.mul_(scale)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length
    starts = [np.random.randint(0, max_start) for _ in range(batch_size)]

    begin = np.stack([dataset[start : start + context_length] for start in starts])
    next = np.stack(
        [dataset[start + 1 : start + context_length + 1] for start in starts]
    )
    return torch.tensor(begin, dtype=torch.long, device=device), torch.tensor(
        next, dtype=torch.long, device=device
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    dic = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "ite": iteration,
    }

    torch.save(dic, out)


def load_checkpoint(src, model: torch.nn.Module, optimizer):
    dic = torch.load(src)
    model.load_state_dict(dic["model_state"])
    optimizer.load_state_dict(dic["optimizer_state"])
    return dic["ite"]


def training_loop(
    # model
    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    rope_theta,
    # data
    train_path,
    valid_path,
    batch_size,
    # optimization
    total_steps,
    lr_max,
    lr_min,
    warmup_iters,
    cosine_cycle_iters,
    weight_decay,
    betas,
    eps,
    max_grad_norm,
    # infra
    device="cuda",
    checkpoint_path="ckpt.pt",
    checkpoint_interval=1000,
    eval_interval=500,
    eval_iters=100,
    log_interval=50,
    seed=0,
    run_name:str="v1"
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    module = transformer_lm(
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta
    ).to(device)

    module = torch.compile(module, backend="aot_eager")   # mps
    adamw = AdamW(module.parameters(), lr_max, weight_decay, betas, eps)
    train_data = np.load(train_path, mmap_mode="r")
    valid_data = np.load(valid_path, mmap_mode="r")

    checkpoint_path = f"ckpt_{run_name}.pt"      # right after the device line

    i=0
    if os.path.exists(checkpoint_path):
        i=load_checkpoint(checkpoint_path,module,adamw)
    
    
    start = time.time()
    history = []
    while i<=total_steps:
        lr=lr_schedule(i,lr_max,lr_min,warmup_iters,cosine_cycle_iters)
        
        batch, result = get_batch(train_data, batch_size, context_length, device)

        adamw.zero_grad()

        predict = module.forward(batch)
        loss = cross_entropy(predict, result)

        loss.backward()
        gradient_clipping(module.parameters(),max_grad_norm)
        for group in adamw.param_groups:
            group["lr"]=lr
        adamw.step()


        if i%checkpoint_interval==0:
            save_checkpoint(module,adamw,i,checkpoint_path)

        if i%eval_interval==0:
            module.eval()
            with torch.no_grad():
                losses=[]
                for j in range(eval_iters):
                    sample,target=get_batch(valid_data,batch_size,context_length,device)
                    losses.append(cross_entropy(module(sample),target).item())
                avg_eval_loss=sum(losses)/len(losses)
                print(f"eval step:{i} loss:{avg_eval_loss:.4f}")

                history.append({"step": i, "wall_clock": time.time() - start,
                "train_loss": loss.item(), "val_loss": avg_eval_loss, "lr": lr})
                os.makedirs("logs", exist_ok=True)
                with open(f"logs/{run_name}.json", "w") as f:
                    json.dump(history, f)
            module.train()


        if i%log_interval==0:
            print(f"train step: {i} loss: {loss.item():.4f}  lr: {lr:.2e}")

        i+=1
    save_checkpoint(module, adamw, i, checkpoint_path)

@torch.no_grad()
def Decoding(module:nn.Module,prompt:torch.Tensor,max_len:int,tem:float,p:float, stop_id:int|None=None,context_length=256):
    module.eval()
    for i in range(max_len):
        logits=module(prompt[:, -context_length:])[:,-1,:]
        logits=logits/tem
        softmax=Softmax()
        probs=softmax(logits,dim=-1)
        value, ids=torch.sort(probs,descending=True)
        cumsum=torch.cumsum(value,dim=-1)
        keep=cumsum-value<p
        sample=value[keep]
        ids=ids[keep]
        
        next=ids[torch.multinomial(sample/sample.sum(),1)]


        if stop_id is not None and next.item()==stop_id:
            break
        else:
            prompt=torch.cat([prompt,next.unsqueeze(0)],dim=-1)

    return prompt


if __name__ == "__main__":

    def showloss(lr):
        torch.manual_seed(0)
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        losses = []
        for t in range(10):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean()  # Compute a scalar loss value.
            losses.append(loss.item())
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Run optimizer step.
        plt.figure()
        plt.plot(range(len(losses)), losses)  # x = iteration index, y = loss
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title(f"Training loss lr={lr}")
        plt.grid(True)
        # plt.savefig(save_path)                    # writes a PNG
        print(losses)
        plt.show()  # use instead of savefig in a notebook

    for lr in [1e1, 1e2, 1e3]:
        showloss(lr)
