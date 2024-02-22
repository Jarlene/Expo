import copy
import torch
import torch.nn as nn
from typing import Any
import torch
from ..layers import RMSNorm

class SoftMoE(nn.Module):
    def __init__(self, dim,
                 expers_num,
                 slots_num,
                 experts: nn.Module,
                 add_noise=True,
                 noise_scala=1.0) -> Any:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.slot_embeds = nn.Parameter(
            torch.rand((expers_num, slots_num, dim)))
        self.experts = nn.ModuleList([copy.deepcopy(experts)
                                     for _ in range(expers_num)])
        self.add_noise = add_noise
        self.noise_scala = noise_scala

    def log(self, t: torch.Tensor, eps=1e-20):
        return torch.log(t.clamp(min=eps))

    def gumbel_noise(self, t: torch.Tensor):
        noise = torch.zeros_like(t).uniform_(0, 1)
        return -self.log(-self.log(noise))

    def forward(self, hidden_states):
        slot_embeds = self.norm(self.slot_embeds)
        logits = torch.einsum(
            'bnd,esd->bnes', hidden_states, slot_embeds)

        if self.add_noise:
            noise = self.gumbel_noise(logits) * self.noise_scala
            logits = logits + noise

        dispatch_weights = logits.softmax(dim=1)
        combine_weights = logits.softmax(dim=-1)
        slots = torch.einsum('bnd,bnes->besd', hidden_states, dispatch_weights)
        outs = []
        for idx, expert in enumerate(self.experts):
            out = expert(slots[:, idx, ...])
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        outs = torch.einsum('besd, bnes->bnd', outs, combine_weights)
        return outs

