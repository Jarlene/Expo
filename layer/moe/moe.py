import copy
import torch
import torch.nn as nn
from typing import Any
import torch
import torch.nn.functional as F
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
        return -self.log(noise)

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
            num_experts_per_token: int,
            router_jitter_noise: float,
            expert: nn.Module,):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([])
        for i in range(num_experts):
            self.experts.append(copy.deepcopy(expert).requires_grad_(True))
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts_per_token = num_experts_per_token
        self.router_jitter_noise = router_jitter_noise

    def load_balancing_loss(self, router_probs: torch.Tensor, expert_mask: torch.Tensor = None):

        router_prob_per_expert = torch.mean(
            router_probs, dtype=torch.float32, dim=-2)

        if expert_mask is not None:
            expert_mask = torch.nn.functional.one_hot(
                expert_mask, self.num_experts).to(torch.int32)
            expert_mask, expert_index = torch.max(expert_mask, dim=-2)
            tokens_per_expert = torch.mean(
                expert_mask, dim=-2, dtype=torch.float32)
            return torch.mean(
                tokens_per_expert * router_prob_per_expert,
                dtype=torch.float32) * self.num_experts**2
        else:
            return torch.mean(
                router_prob_per_expert,
                dtype=torch.float32) * self.num_experts**2

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

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        hidden_states = self.add_noise(hidden_states, self.training)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_token, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        self.computer_loss(router_logits, routing_weights, selected_experts)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            top_x, idx = torch.where(expert_mask[..., expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None,
                                          top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(
                current_state)*routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)
        return final_hidden_states, self.auxiliary_loss, self.z_loss
