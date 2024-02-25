from math import sqrt
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, einsum, repeat
from einops.layers.torch import Rearrange
import torch


class Transpose(nn.Module):
    def __init__(self, *size) -> None:
        super().__init__()
        self.size = size

    def forward(self, x):
        return torch.transpose(x, *self.size)


class SelfAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, dropout=0.5, causal=True, split_head=True) -> None:
        super(SelfAttention, self).__init__()
        self.split_head = split_head
        self.hidden_size = hidden_size
        if self.split_head:
            self.num_heads = num_heads
            self.head_dim = self.hidden_size//self.num_heads

        self.fused_proj = nn.Linear(
            self.hidden_size, self.hidden_size * 3, bias=False)

        self.causal = causal

        self.dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.register_buffer('mask', None, False)

    def get_mask(self, i, j, device):
        if self.mask is not None:
            return self.mask

        mask = torch.ones((i, j), device=device,
                          dtype=torch.bool).triu(j - i + 1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def forward(self, x: torch.Tensor, mask=None, **kwargs):
        n, device = x.shape[1], x.device

        query, key, value = self.fused_proj(
            x).split(self.hidden_size, dim=-1)
        if self.split_head:
            query = rearrange(
                query, 'b n ... (h d) ->  b h ... n d', h=self.num_heads)
            key = rearrange(
                key, 'b n ... (h d) ->  b h ... n d', h=self.num_heads)
            value = rearrange(
                value, 'b n ... (h d) ->  b h ... n d', h=self.num_heads)
        else:
            query = rearrange(query, 'b n ... h d -> b h ... n d')
            key = rearrange(key, 'b n ... h d -> b h ... n d')
            value = rearrange(value, 'b n ... h d -> b h ... n d')

        scale = torch.full([], value.size(-1) ** -0.5,
                           dtype=value.dtype, device=value.device)
        query = query * scale

        sim = query @ key.transpose(-1, -2)

        mask_value = -torch.finfo(sim.dtype).max
        if mask is not None:
            sim = sim.masked_fill(mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = self.get_mask(i, j, device)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum(attn, value, "... n d, ... d k-> ... n k")
        if self.split_head:
            out = rearrange(out, 'b h ... n d -> b n ... (h d)')
        else:
            out = rearrange(out, 'b h ... n d -> b n ... h d')
        out = self.proj(out)
        out = self.resid_dropout(out)
        return out, attn


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.5, split_head=True) -> None:
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.split_head = split_head
        if self.split_head:
            self.num_heads = num_heads
            self.head_dim = self.hidden_size//self.num_heads

        self.fused_proj = nn.Linear(
            self.hidden_size, self.hidden_size * 2, bias=False)

        self.register_buffer('mask', None, False)

        self.dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask=None, context_mask=None, **kwargs):

        n, device = query.shape[1], query.device
        m = key_value.shape[1]
        key, value = self.fused_proj(
            key_value).split(self.hidden_size, dim=-1)
        if self.split_head:
            query = rearrange(
                query, 'b n ... (h d) ->  b h ... n d', h=self.num_heads)
            key = rearrange(
                key, 'b n ... (h d) ->  b h ... n d', h=self.num_heads)
            value = rearrange(
                value, 'b n ... (h d) ->  b h ... n d', h=self.num_heads)
        else:
            query = rearrange(query, 'b n ... h d -> b h ... n d')
            key = rearrange(key, 'b n ... h d -> b h ... n d')
            value = rearrange(value, 'b n ... h d -> b h ... n d')

        scale = torch.full([], value.size(-1) ** -0.5,
                           dtype=value.dtype, device=value.device)
        query = query * scale

        sim = query @ key.transpose(-1, -2)

        mask_value = -torch.finfo(sim.dtype).max
        if mask is not None:
            sim = sim.masked_fill(mask, mask_value)

        if context_mask is not None:
            sim = sim.masked_fill(~context_mask[:, None, :], mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum(attn, value, "... n d, ... d k-> ... n k")
        if self.split_head:
            out = rearrange(out, 'b h ... n d -> b n ... (h d)')
        else:
            out = rearrange(out, 'b h ... n d -> b n ... h d')

        out = self.proj(out)
        out = self.resid_dropout(out)
        return out, attn


class AgentAttention(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_agent_tokens,
            heads_num=8,
            dropout=0.02,
            dim_inner=None,
            talking_heads=True,
            gate=True,):
        super().__init__()
        dim_head = hidden_size//heads_num
        self.scale = dim_head ** -0.5
        dim_inner = dim_inner if dim_inner is not None else hidden_size

        self.to_qkv = nn.Sequential(
            nn.Linear(hidden_size, dim_inner * 3, bias=False),
            Rearrange('b n (qkv h d) -> qkv b h n d', h=heads_num, qkv=3)
        )

        self.to_gates = nn.Sequential(
            nn.Linear(hidden_size, heads_num),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if gate else None

        self.agent_tokens = nn.Parameter(
            torch.zeros(heads_num, num_agent_tokens, dim_head))
        nn.init.normal_(self.agent_tokens, std=0.02)

        self.qa_talking_heads = nn.Conv2d(
            heads_num, heads_num, 1, bias=False) if talking_heads else nn.Identity()
        self.ak_talking_heads = nn.Conv2d(
            heads_num, heads_num, 1, bias=False) if talking_heads else nn.Identity()

        self.qa_dropout = nn.Dropout(dropout)
        self.ak_dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, hidden_size, bias=False)
        )

    def forward(
            self,
            x,
            mask=None,
            agent_tokens=None):
        batch = x.shape[0]
        q, k, v = self.to_qkv(x)
        a = agent_tokens if agent_tokens is not None else repeat(
            self.agent_tokens, 'h m d -> b h m d', b=batch)
        a = a * self.scale
        qa_sim = einsum('b h i d, b h j d -> b h i j', q, a)
        ak_sim = einsum('b h i d, b h j d -> b h i j', a, k)
        if mask is not None:
            max_neg_value = -torch.finfo(qa_sim.dtype).max
            ak_sim = ak_sim.masked_fill(~rearrange(
                mask, 'b j -> b 1 1 j'), max_neg_value)
        qa_attn = qa_sim.softmax(dim=-1)
        ak_attn = ak_sim.softmax(dim=-1)
        qa_attn = self.qa_dropout(qa_attn)
        ak_attn = self.ak_dropout(ak_attn)
        qa_attn = self.qa_talking_heads(qa_attn)
        ak_attn = self.ak_talking_heads(ak_attn)
        agent_gathered_tokens = einsum(
            'b h i j, b h j d -> b h i d', ak_attn, v)
        out = einsum('b h i j, b h j d -> b h i d',
                     qa_attn, agent_gathered_tokens)
        if mask is not None:
            out = out.masked_fill(~rearrange(mask, 'b n -> b 1 n 1'), 0.)
        if self.to_gates is not None:
            out = out * self.to_gates(x)
        out = self.to_out(out)
        return out, agent_gathered_tokens


class ProbAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_mask(self, B, H, L, index, scores):
        _mask = torch.ones(
            L, scores.shape[-1], dtype=torch.bool).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :]
        _mask = indicator.view(scores.shape)
        return _mask

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if attn_mask is None:
            attn_mask = self._get_mask(B, H, L_Q, index, scores).to(V.device)
        scores.masked_fill_(attn_mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
            np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class DualAttenion(nn.Module):
    def __init__(self, hidden_size, enc_in, num_heads, dropout, momentum, d_ff, dp_rank, total_token_number, alpha, over_channel=False):
        super(DualAttenion, self).__init__()
        self.over_channel = over_channel
        self.num_heads = num_heads
        self.c_in = enc_in
        # attention related
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.head_dim = hidden_size // num_heads
        self.dropout_mlp = nn.Dropout(dropout)
        self.mlp = nn.Linear(hidden_size, hidden_size)
        self.norm_post1 = nn.Sequential(Transpose(1, 2),
                                        nn.BatchNorm1d(hidden_size,
                                                       momentum=momentum),
                                        Transpose(1, 2))
        self.norm_post2 = nn.Sequential(Transpose(1, 2),
                                        nn.BatchNorm1d(hidden_size,
                                                       momentum=momentum),
                                        Transpose(1, 2))
        self.norm_attn = nn.Sequential(Transpose(1, 2),
                                       nn.BatchNorm1d(hidden_size,
                                                      momentum=momentum),
                                       Transpose(1, 2))
        self.ff_1 = nn.Sequential(nn.Linear(hidden_size, d_ff, bias=True),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_ff, hidden_size, bias=True))
        self.ff_2 = nn.Sequential(nn.Linear(hidden_size, d_ff, bias=True),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_ff, hidden_size, bias=True))

        # dynamic projection related
        self.dp_rank = dp_rank
        self.dp_k = nn.Linear(self.head_dim, self.dp_rank)
        self.dp_v = nn.Linear(self.head_dim, self.dp_rank)
        # EMA related
        ema_size = max(enc_in, total_token_number, dp_rank)
        ema_matrix = torch.zeros((ema_size, ema_size))
        alpha = alpha
        ema_matrix[0][0] = 1
        for i in range(1, total_token_number):
            for j in range(i):
                ema_matrix[i][j] = ema_matrix[i-1][j]*(1-alpha)
            ema_matrix[i][i] = alpha
        self.register_buffer('ema_matrix', ema_matrix)

    def ema(self, src):
        return torch.einsum('bnhad,ga -> bnhgd', src, self.ema_matrix[:src.shape[-2], :src.shape[-2]])

    def dynamic_projection(self, src, mlp):
        src_dp = mlp(src)
        src_dp = F.softmax(src_dp, dim=-1)
        src_dp = torch.einsum('bnhef,bnhec -> bnhcf', src, src_dp)
        return src_dp

    def forward(self, src):
        # construct Q,K,V
        B, nvars, H, C, = src.shape
        qkv = self.qkv(src).reshape(B, nvars, H, 3, self.num_heads,
                                    C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if not self.over_channel:
            attn_score_along_token = torch.einsum(
                'bnhed,bnhfd->bnhef', self.ema(q), self.ema(k)) / self.head_dim ** -0.5
            attn_along_token = self.attn_dropout(
                F.softmax(attn_score_along_token, dim=-1))
            output_along_token = torch.einsum(
                'bnhef,bnhfd->bnhed', attn_along_token, v)
        else:
            # dynamic project V and K
            v_dp, k_dp = self.dynamic_projection(
                v, self.dp_v), self.dynamic_projection(k, self.dp_k)
            attn_score_along_token = torch.einsum(
                'bnhed,bnhfd->bnhef', self.ema(q), self.ema(k_dp)) / self.head_dim ** -0.5
            attn_along_token = self.attn_dropout(
                F.softmax(attn_score_along_token, dim=-1))
            output_along_token = torch.einsum(
                'bnhef,bnhfd->bnhed', attn_along_token, v_dp)
        # attention over hidden dimensions
        attn_score_along_hidden = torch.einsum(
            'bnhae,bnhaf->bnhef', q, k) / q.shape[-2] ** -0.5
        attn_along_hidden = self.attn_dropout(
            F.softmax(attn_score_along_hidden, dim=-1))
        output_along_hidden = torch.einsum(
            'bnhef,bnhaf->bnhae', attn_along_hidden, v)
        # post_norm
        output1 = output_along_token.reshape(
            B*nvars, -1, self.num_heads * self.head_dim)
        output1 = self.norm_post1(output1)
        output1 = output1.reshape(B, nvars, -1, self.num_heads * self.head_dim)
        output2 = output_along_hidden.reshape(
            B*nvars, -1, self.num_heads * self.head_dim)
        output2 = self.norm_post2(output2)
        output2 = output2.reshape(B, nvars, -1, self.num_heads * self.head_dim)
        # add & norm
        src2 = self.ff_1(output1)+self.ff_2(output2)
        src = src + src2
        src = src.reshape(B*nvars, -1, self.num_heads * self.head_dim)
        src = self.norm_attn(src)
        src = src.reshape(B, nvars, -1, self.num_heads * self.head_dim)

        return src


class RingAttention(nn.Module):
    pass
