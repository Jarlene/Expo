from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Union
import torch

import torch.nn as nn


@dataclass
class RouterIndices:
    """Dispatch indices and combine weights for scatter/gather-based routing.

    Attributes:
      dispatch_indices: <int32>[num_groups, tokens_per_group,
        num_selected_experts, 2] dispatch indices indicating, for each token, its
        preferred expert and its priority in that expert's buffer.
      combine_weights: <float>[num_groups, tokens_per_group, num_selected_experts]
        combine weights used for scaling expert outputs with the router's dispatch
        probability/confidence.
      auxiliary_loss: Load balancing loss for router.
      router_z_loss: Router z-loss. Encourages router logits to remain small in an
        effort to improve stability.
    """
    dispatch_indices: torch.Tensor
    combine_weights: torch.Tensor
    auxiliary_loss: float
    router_z_loss: float = 0.


@dataclass
class RouterMask:
    """Dispatch and combine arrays for expert routing with masked matmuls.

    Attributes:
      dispatch_mask: <bool>[num_groups, tokens_per_group, num_experts,
        expert_capacity] dispatch array that is 1 if the token gets routed to the
        corresponding expert, and 0 otherwise.
      combine_array: <float>[num_groups, tokens_per_group, num_experts,
        expert_capacity] combine array used for combining expert outputs and
        scaling with router probability.
      auxiliary_loss: Load balancing loss for router.
      router_z_loss: Router z-loss. Encourages router logits to remain small in an
        effort to improve stability.
    """
    dispatch_mask: torch.Tensor
    combine_array: torch.Tensor
    auxiliary_loss: float
    router_z_loss: float = 0.


class Router(nn.Module):

    def __init__(self, input_dim: int, num_experts: int, top_k: int, expert_capacity: int, ignore_padding_tokens: bool = True, jitter_noise: float = 0.2, **kwargs) -> None:
        super().__init__()
        self.jitter_noise = jitter_noise
        self.ignore_padding_tokens = ignore_padding_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity
        self.weight = nn.Parameter(torch.randn(input_dim, num_experts))

    def _top_k_mask(array: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        top_k_indices = torch.topk(array, k)[-1]
        mask = torch.nn.functional.one_hot(
            top_k_indices, array.shape[-1], dtype=torch.float32)
        mask = torch.sum(mask, dim=-2)
        return mask, top_k_indices

    def compute_routing_instructions(self, router_probs, padding_mask, expert_capacity):
        """Computes instructions for routing inputs to experts."""
        raise NotImplementedError(
            'Router is an abstract class that should be subclassed.')

    def compute_router_probabilities(self, x: torch.Tensor):
        if self.jitter_noise > 0 and self.training:

            distrib_lower_bound = 1.0 - self.jitter_noise
            distrib_upper_bound = 1.0 + self.jitter_noise

            uniform_distrib = torch.rand(
                x.shape, device=x.device, dtype=x.dtype, requires_grad=False)
            uniform_distrib = uniform_distrib * \
                (distrib_lower_bound - distrib_upper_bound)

            uniform_distrib = uniform_distrib + distrib_upper_bound
            x *= uniform_distrib

        router_logits = x @ self.weight
        router_probabilities = nn.functional.softmax(router_logits, dim=-1)
        return router_probabilities, router_logits

    def load_balancing_loss(self, router_probs: torch.Tensor, expert_mask: torch.Tensor = None):
        num_experts = router_probs.shape[-1]
        router_prob_per_expert = torch.mean(
            router_probs, dtype=torch.float32, dim=-2)

        if expert_mask is not None:
            tokens_per_expert = torch.mean(
                expert_mask, dtype=torch.float32, dim=-2)
            return torch.mean(
                tokens_per_expert * router_prob_per_expert,
                dtype=torch.float32) * num_experts**2
        else:
            return torch.mean(
                router_prob_per_expert,
                dtype=torch.float32) * num_experts**2

    def router_z_loss(self, router_logits: torch.Tensor):
        num_groups, tokens_per_group, _ = router_logits.shape
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = log_z**2
        return torch.sum(z_loss, dtype=torch.float32) / (num_groups * tokens_per_group)

    def forward(self, x: torch.Tensor, **kwargs):
        batch_size, seq_len, dim = x.shape
        x = x.view(-1, dim)
        router_probs, router_logits = self.compute_router_probabilities(x)
        if self.ignore_padding_tokens:
            padding_mask = torch.tensor(
                (torch.sum(torch.abs(x), dim=-1) > 0), dtype=x.dtype)
            router_logits *= padding_mask.unsqueeze(dim=-1)
        else:
            padding_mask = None
        instructions = self.compute_routing_instructions(router_probs,
                                                         padding_mask,
                                                         self.expert_capacity)

        return instructions.replace(router_z_loss=self.router_z_loss(router_logits))


class ScatterRouter(Router):
    """Abstract base router class for scatter dispatch routers.

    ScatterRouter(s) return RouterIndices containing dispatch indices and combine
    weights for sending token inputs (via scatter) and receiving outputs (via
    gather) to and from experts.

    Scatter-based routing is generally faster than masked matmul routing on CPUs
    and GPUs.
    """

    def _compute_routing_instructions(self, router_probs: torch.Tensor,
                                      padding_mask: Optional[torch.Tensor],
                                      expert_capacity: int) -> RouterIndices:
        """Computes instructions for routing inputs to experts.

        Args:
          router_probs: <float32>[num_groups, tokens_per_group, num_experts]
            probabilities used to determine the routing of tokens to the experts.
          padding_mask: <float32>[num_groups, tokens_per_group] padding logit mask
            used to identify padding tokens that should be ignored by the router.
          expert_capacity: Each group will send this many tokens to each expert.

        Returns:
          Router indices containing dispatch indices and combine weights.
        """
        raise NotImplementedError(
            'ScatterRouter is an abstract class that should be subclassed.')


class MaskedRouter(Router):
    """Abstract base router class for masked matmul dispatch routers.

    MaskedRouter(s) return RouterMask(s) containing a dispatch mask and combine
    array for sending and receiving (via masked matmuls) inputs and outputs to and
    from experts.

    Routing using masked matmuls is generally faster than scatter-based routing on
    TPUs.
    """

    def _compute_routing_instructions(self, router_probs: torch.Tensor,
                                      padding_mask: Optional[torch.Tensor],
                                      expert_capacity: int) -> RouterMask:
        """Computes masks for the top-k experts per token.

        Args:
          router_probs: <float32>[num_groups, tokens_per_group, num_experts]
            probabilities used to determine the routing of tokens to the experts.
          padding_mask: <float32>[num_groups, tokens_per_group] padding logit mask
            used to identify padding tokens that should be ignored by the router.
          expert_capacity: Each group will send this many tokens to each expert.

        Returns:
          Router mask arrays.
        """
        raise NotImplementedError(
            'MaskedRouter is an abstract class that should be subclassed.')


class TokensChooseScatterRouter(ScatterRouter):
    def __init__(self, input_dim: int,
                 num_experts: int,
                 batch_prioritized_routing: bool,
                 top_k: int,
                 ignore_padding_tokens: bool = True,
                 jitter_noise: float = 0.2, **kwargs) -> None:
        super().__init__(input_dim, num_experts, top_k,
                         ignore_padding_tokens, jitter_noise, **kwargs)

        self.batch_prioritized_routing = batch_prioritized_routing

    def compute_routing_instructions(self, router_probs, padding_mask, expert_capacity):
        num_groups, tokens_per_group, num_experts = router_probs.shape
        if padding_mask is not None:
            # Because `expert_indices` are directly used for scatter-based routing, we
            # mask probabilities corresponding to tokens before the top-k operation.
            # Note that, unlike for mask-based tokens-choose routing, the
            # (down-weighted) padding tokens may still be selected.
            router_probs *= torch.unsqueeze(padding_mask, dim=-1)

        combine_weights, expert_indices = torch.topk(
            router_probs, k=self.top_k)
        auxiliary_loss = self.load_balancing_loss(router_probs, expert_indices)
        if self.batch_prioritized_routing:
            # Sort tokens according to their routing probability per token group, so
            # that the highest probability tokens are routed first.
            token_ordering = torch.argsort(-combine_weights[..., 0], dim=-1)
            expert_indices = torch.take_along_dim(
                expert_indices, torch.unsqueeze(token_ordering, dim=-1), dim=-2)
        preferred_experts = torch.swapaxes(expert_indices, 1, 2)
        preferred_experts = preferred_experts.reshape(num_groups, -1)
        expert_mask = torch.nn.functional.one_hot(
            preferred_experts, num_experts)

        token_priority = torch.cumsum(expert_mask, dim=1) * expert_mask - 1.0
        # Shape: [num_groups, num_selected_experts, tokens_per_group, num_experts].
        token_priority = token_priority.reshape(
            (num_groups, self.top_k, -1, num_experts))
        # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
        token_priority = torch.swapaxes(token_priority, 1, 2)
        # For each token, across all experts, select the only non-negative
        # (unmasked) priority. Shape: [num_groups, tokens_per_group,
        # num_selected_experts].
        token_priority = torch.max(token_priority, dim=-1)

        # Return to original index shape.
        preferred_experts = preferred_experts.reshape(num_groups,
                                                      self.top_k,
                                                      tokens_per_group)
        # Shape: [num_groups, tokens_per_group, num_selected_experts]
        preferred_experts = torch.swapaxes(preferred_experts, 1, 2)

        if self.batch_prioritized_routing:
            # Place tokens in their original ordering.
            inverse_token_ordering = torch.argsort(token_ordering, dim=-1)
            preferred_experts = torch.take_along_dim(
                preferred_experts,
                torch.unsqueeze(inverse_token_ordering, dim=-1),
                dim=-2)
            token_priority = torch.take_along_dim(
                token_priority,
                torch.unsqueeze(inverse_token_ordering, dim=-1),
                dim=-2)

        # Mask out tokens that overflow the maximum expert capacities.
        # Shape: [num_groups, tokens_per_group, num_selected_experts].
        combine_weights *= token_priority < expert_capacity

        # Expert index and priority within the expert capacity buffer.
        # Shape: [num_groups, tokens_per_group, num_selected_experts, 2].
        dispatch_indices = torch.stack(
            [preferred_experts, token_priority], dim=-1)

        return RouterIndices(dispatch_indices, combine_weights, auxiliary_loss)


class TokensChooseMaskedRouter(MaskedRouter):
    def __init__(self, input_dim: int,
                 num_experts: int,
                 batch_prioritized_routing: bool,
                 top_k: int,
                 ignore_padding_tokens: bool = True,
                 jitter_noise: float = 0.2, **kwargs) -> None:
        super().__init__(input_dim, num_experts, top_k,
                         ignore_padding_tokens, jitter_noise, **kwargs)

        self.batch_prioritized_routing = batch_prioritized_routing

    def compute_routing_instructions(self, router_probs: torch.Tensor, padding_mask: Optional[torch.Tensor], expert_capacity):
        num_groups, tokens_per_group, num_experts = router_probs.shape

        expert_gate, expert_index = torch.topk(
            router_probs, k=self.top_k)
        if padding_mask is not None:
            # Mask applied to gate. Exclude choices corresponding to padding tokens.
            gate_mask = torch.unsqueeze(padding_mask, dim=-1)
            expert_gate *= gate_mask

            # Set `expert_index` elements corresponding to padding to negative
            # numbers. Negative `expert_index` elements will ultimately be dropped in
            # the one_hot conversion to the `expert_mask`.
            # First convert nonzero padding elements to negative values.
            expert_index *= 2 * gate_mask - 1.
            # Handle zero padding elements by negatively shifting all padding.
            expert_index += torch.repeat(
                gate_mask - 1., self.num_selected_experts, dim=-1)

            # To correctly compute load balancing loss, we also mask out probs.
            router_probs *= gate_mask
        auxiliary_loss = self.load_balancing_loss(router_probs, expert_index)
        if self.batch_prioritized_routing:
            # Sort tokens according to their routing probability per group, so that
            # the highest probability tokens are routed first.
            permutation = torch.argsort(-expert_gate[..., 0], dim=-1)
            # Shape: [num_groups, tokens_per_group, num_selected_experts]
            expert_index = torch.take_along_dim(
                expert_index, torch.unsqueeze(permutation, dim=-1), dim=-2)

        # Make num_selected_experts the leading dim to ensure that top-1 choices
        # have priority over top-2 choices, which have priority over top-3 choices,
        # etc.
        expert_index = torch.swapaxes(expert_index, 1, 2)
        # Shape: [num_groups, num_selected_experts * tokens_per_group]
        expert_index = expert_index.reshape(num_groups, -1)

        # Create mask out of indices.
        # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
        expert_mask = torch.nn.functional.one_hot(expert_index, num_experts)

        # Experts have a fixed capacity that we cannot exceed. A token's priority
        # within the expert's buffer is given by the masked, cumulative capacity of
        # its target expert.
        # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
        token_priority = torch.cumsum(expert_mask, dim=1) * expert_mask - 1.0
        # Shape: [num_groups, num_selected_experts, tokens_per_group, num_experts].
        token_priority = token_priority.reshape(
            (num_groups, self.top_k, -1, num_experts))
        # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
        token_priority = torch.swapaxes(token_priority, 1, 2)
        # For each token, across all selected experts, select the only non-negative
        # (unmasked) priority. Now, for group G routing to expert E, token T has
        # non-negative priority (i.e. token_priority[G,T,E] >= 0) if and only if E
        # is its targeted expert.
        # Shape: [num_groups, tokens_per_group, num_experts].
        token_priority = torch.max(token_priority, dim=2)

        if self.batch_prioritized_routing:
            # Place token priorities in original ordering of tokens.
            inv_permutation = torch.argsort(permutation, dim=-1)
            token_priority = torch.take_along_dim(
                token_priority, torch.unsqueeze(inv_permutation, dim=-1), dim=-2)

        # Token T can only be routed to expert E if its priority is positive and
        # less than the expert capacity. One-hot matrix will ignore indices outside
        # the range [0, expert_capacity).
        # Shape: [num_groups, tokens_per_group, num_experts, expert_capacity].
        dispatch_mask = torch.nn.functional.one_hot(
            token_priority, expert_capacity)

        # The combine array will be used for combining expert outputs, scaled by the
        # router probabilities. Shape: [num_groups, tokens_per_group, num_experts,
        # expert_capacity].
        combine_array = torch.einsum(
            '...te,...tec->...tec',
            router_probs,
            dispatch_mask)
        return RouterMask(dispatch_mask, combine_array, auxiliary_loss)


class ExpertsChooseMaskedRouter(MaskedRouter):
    """Masked matmul router using experts choose tokens assignment.

    This router uses the same mechanism as in Mixture-of-Experts with Expert
    Choice (https://arxiv.org/abs/2202.09368): each expert selects its top
    expert_capacity tokens. An individual token may be processed by multiple
    experts or none at all.

    Note: "experts choose routing" should not be used in decoder blocks because it
    breaks the autoregressive behavior -- the model will learn to cheat by using
    future token information to improve current token predictions.
    """

    def _compute_routing_instructions(self, router_probs: torch.Tensor,
                                      padding_mask: Optional[torch.Tensor],
                                      expert_capacity: int) -> RouterMask:
        """Computes masks for the highest probability token per expert.

        Args:
          router_probs: <float32>[num_groups, tokens_per_group, num_experts]
            probabilities used to determine the routing of tokens to the experts.
          padding_mask: <float32>[num_groups, tokens_per_group] padding logit mask
            used to identify padding tokens that should be down-weighted by the
            router.
          expert_capacity: Each group will send this many tokens to each expert.

        Returns:
            Dispatch and combine arrays for routing with masked matmuls.
        """
        tokens_per_group = router_probs.shape[1]

        if padding_mask is not None:
            # Because experts choose tokens, we mask probabilities corresponding to
            # tokens before the top-k operation. Note that, unlike for masked-based
            # tokens-choose routing, the experts here may still choose to select the
            # (down-weighted) padding tokens.
            router_probs *= torch.unsqueeze(padding_mask, dim=-1)

        router_probs_t = router_probs.transpose(2, 1)

        # Top expert_capacity router probability and corresponding token indices for
        # each expert. Shapes: [num_groups, num_experts, expert_capacity].
        expert_gate, expert_index = torch.topk(
            router_probs_t, k=expert_capacity)

        # Convert to one-hot mask of expert indices for each token in each group.
        # Shape: [num_groups, num_experts, expert_capacity, tokens_per_group].
        dispatch_mask = torch.nn.functional.one_hot(
            expert_index, tokens_per_group)

        # Move axes to conform with shape expected by MoeLayer API.
        # Shape: [num_groups, tokens_per_group, num_experts, expert_capacity]
        dispatch_mask = torch.moveaxis(dispatch_mask, 3, 1)

        # The combine array will be used for combining expert outputs, scaled by the
        # router probabilities. Shape: [num_groups, num_experts, tokens_per_group,
        # expert_capacity].
        combine_array = torch.einsum(
            '...ec,...tec->...tec',
            expert_gate,
            dispatch_mask)

        # Each expert is choosing tokens until it reaches full capacity, so we don't
        # need an auxiliary loading balancing loss for expert choice routing.
        auxiliary_loss = 0.0

        return RouterMask(dispatch_mask, combine_array, auxiliary_loss)
