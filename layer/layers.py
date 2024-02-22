import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange
import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=- 1) * self.scale * self.gamma


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device,
                           dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


class RelPosBias2d(nn.Module):
    def __init__(self, size, heads):
        super().__init__()
        self.pos_bias = nn.Embedding((2 * size - 1) ** 2, heads)

        arange = torch.arange(size)

        pos = torch.stack(torch.meshgrid(
            arange, arange, indexing='ij'), dim=-1)
        pos = rearrange(pos, '... c -> (...) c')
        rel_pos = rearrange(pos, 'i c -> i 1 c') - \
            rearrange(pos, 'j c -> 1 j c')

        rel_pos = rel_pos + size - 1
        h_rel, w_rel = rel_pos.unbind(dim=-1)
        pos_indices = h_rel * (2 * size - 1) + w_rel
        self.register_buffer('pos_indices', pos_indices)

    def forward(self, qk):
        i, j = qk.shape[-2:]

        bias = self.pos_bias(self.pos_indices[:i, :(j - 1)])
        bias = rearrange(bias, 'i j h -> h i j')

        # account for null key / value for classifier free guidance
        bias = F.pad(bias, (j - bias.shape[-1], 0), value=0.)
        return bias


class SwiGLU(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_in * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return F.silu(gate) * x


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim_in
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.silu(gate)


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        glu=False,
        swish=False,
        relu_squared=False,
        post_act_ln=False,
        dropout=0.5,
        no_bias=False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=not no_bias),
            activation
        ) if not glu else GLU(dim, inner_dim)

        self.ff = nn.Sequential(
            project_in,
            LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=not no_bias)
        )

    def forward(self, x):
        return self.ff(x)
