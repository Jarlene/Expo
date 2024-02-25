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
                 expert: nn.Module,
                 add_noise=True,
                 noise_scala=1.0) -> Any:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.slot_embeds = nn.Parameter(
            torch.rand((expers_num, slots_num, dim)))
        self.experts = nn.ModuleList([copy.deepcopy(expert).requires_grad_(True)
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

        if self.add_noise and self.training:
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


class DroplessMoE(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_experts: int,
            moe_num_experts_per_token: int,
            router_jitter_noise: float,
            expert: nn.Module,):
        super().__init__()
        num_experts = num_experts
        self.experts = nn.ModuleList([])
        for i in range(num_experts):
            self.experts.append(copy.deepcopy(expert).requires_grad_(True))
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts_per_token = moe_num_experts_per_token
        self.router_jitter_noise = router_jitter_noise

    def load_balancing_loss(self, router_probs: torch.Tensor, expert_mask: torch.Tensor = None):
        num_experts = router_probs.shape[-1]
        router_prob_per_expert = torch.mean(
            router_probs, dtype=torch.float32, dim=-2)

        if expert_mask is not None:
            expert_mask = torch.nn.functional.one_hot(
                expert_mask, num_experts).to(torch.int32)
            expert_mask, expert_index = torch.max(expert_mask, dim=-2)
            tokens_per_expert = torch.mean(
                expert_mask, dim=-2, dtype=torch.float32)
            return torch.mean(
                tokens_per_expert * router_prob_per_expert,
                dtype=torch.float32) * num_experts**2
        else:
            return torch.mean(
                router_prob_per_expert,
                dtype=torch.float32) * num_experts**2

    def add_noise(self, hidden_states, training: bool = False):
        if self.router_jitter_noise > 0 and training:

            distrib_lower_bound = 1.0 - self.router_jitter_noise
            distrib_upper_bound = 1.0 + self.router_jitter_noise

            uniform_distrib = torch.rand(
                hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=False)
            uniform_distrib = uniform_distrib * \
                (distrib_lower_bound - distrib_upper_bound)

            uniform_distrib = uniform_distrib + distrib_upper_bound
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states = hidden_states * uniform_distrib
        return hidden_states

    def router_z_loss(self, router_logits: torch.Tensor):
        token_num, _ = router_logits.shape
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = log_z**2
        return torch.sum(z_loss, dtype=torch.float32) / token_num

    def computer_loss(self, router_logits: torch.Tensor, router_probs: torch.Tensor, expert_mask: torch.Tensor = None):
        self.z_loss = self.router_z_loss(router_logits)
        self.auxiliary_loss = self.load_balancing_loss(
            router_probs, expert_mask)

    def forward(self, x: torch.Tensor):
        device = x.device
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])
        x = self.add_noise(x)
        router_logits = self.gate(x)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(
            routing_weights, self.num_experts_per_token, dim=-1)
        self.computer_loss(router_logits, expert_weights, expert_indices)
        flat_expert_indices = expert_indices.view(-1)
        x = x.repeat_interleave(self.num_experts_per_token, dim=0)
        y = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert = expert.to(device)
            y[flat_expert_indices == i] += expert(x[flat_expert_indices == i])
        y = (y.view(*expert_weights.shape, -1) *
             expert_weights.unsqueeze(-1)).sum(dim=1)
        return y.view(*orig_shape), self.z_loss, self.auxiliary_loss
