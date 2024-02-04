import copy
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd, custom_bwd
import numpy as np
from typing import Any, List, Tuple, Optional, Dict
from einops import rearrange, einsum, repeat
import torch
from math import sqrt, gcd
from utils.mask import ProbMask
import trees_layer


class TreeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_features, node_weights, leaf_weights,
                output_logits_dim, depth, smooth_step_param, training_mode):
        ctx.save_for_backward(input_features, node_weights, leaf_weights)
        ctx.output_logits_dim = output_logits_dim
        ctx.depth = depth
        ctx.smooth_step_param = smooth_step_param
        result = trees_layer.forward(input_features, node_weights, leaf_weights, output_logits_dim, depth, smooth_step_param,
                                     training_mode)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        input_features, node_weights, leaf_weights = ctx.saved_tensors()
        output_logits_dim, depth, smooth_step_param = ctx.output_logits_dim, ctx.depth, ctx.smooth_step_param
        result = trees_layer.backward(grad_outputs[0], input_features, node_weights, leaf_weights,
                                      output_logits_dim, depth, smooth_step_param)
        return result[0], result[1], result[2], None, None, None, None


class ParallelLinear(torch.autograd.Function):
    """
    A custom autograd function for Parallel Linear operation.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, expert_size_list, weight, bias=None):
        """
        Forward pass of the ParallelLinear operation.

        Args:
            ctx: Context object.
            input (Tensor): Input tensor.
            expert_size_list (List[int]): List of expert sizes.
            weight (Tensor): Weight tensor.
            bias (Optional[Tensor]): Bias tensor.

        Returns:
            Tensor: Output tensor.
        """
        # expert_size_list: List[int] = expert_size.tolist()
        output = ParallelLinear.forward_scriptable(
            input, expert_size_list, weight, bias)
        # assert torch.allclose(ParallelLinear._forward_scriptable(input, expert_size, weight, bias),  output)
        ctx.save_for_backward(input, weight, bias)
        ctx.expert_size_list = expert_size_list
        return output

    @staticmethod
    @torch.jit.script
    def forward_scriptable(input: torch.Tensor, expert_size_list: List[int],
                           weight: torch.Tensor, bias: Optional[torch.Tensor]):
        """
        Scriptable forward pass of the ParallelLinear operation.

        Args:
            input (Tensor): Input tensor.
            expert_size_list (List[int]): List of expert sizes.
            weight (Tensor): Weight tensor.
            bias (Optional[Tensor]): Bias tensor.

        Returns:
            Tensor: Output tensor.
        """
        output_buf = torch.empty((input.size(0), weight.size(2)),
                                 device=input.device, dtype=input.dtype)
        num_linears = weight.size(0)

        input_list = input.split(expert_size_list, dim=0)
        output_buf_list = output_buf.split(expert_size_list)

        for i in range(num_linears):
            torch.mm(input_list[i], weight[i], out=output_buf_list[i])

        if bias is not None:
            for i in range(num_linears):
                output_buf_list[i].add_(bias[i])

        output = output_buf
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        """
        Backward pass of the ParallelLinear operation.

        Args:
            ctx: Context object.
            grad_out (Tensor): Gradient of the output.

        Returns:
            Tuple of Tensors: Gradients with respect to input, weight, and bias.
        """
        input, weight, bias = ctx.saved_tensors
        expert_size_list = ctx.expert_size_list
        return ParallelLinear.backward_scriptable(
            grad_out, input, expert_size_list,
            weight, bias
        )

    @staticmethod
    @torch.jit.script
    def backward_scriptable(grad_out: torch.Tensor,
                            input: torch.Tensor, expert_size_list: List[int],
                            weight: torch.Tensor, bias: Optional[torch.Tensor]):
        """
        Scriptable backward pass of the ParallelLinear operation.

        Args:
            grad_out (Tensor): Gradient of the output.
            input (Tensor): Input tensor.
            expert_size_list (List[int]): List of expert sizes.
            weight (Tensor): Weight tensor.
            bias (Optional[Tensor]): Bias tensor.

        Returns:
            Tuple of Tensors: Gradients with respect to input, weight, and bias.
        """
        num_linears = weight.size(0)
        input_list = input.t().split(expert_size_list, dim=1)
        grad_list = grad_out.split(expert_size_list, dim=0)

        d_input_buf = torch.empty_like(input)
        d_input_buf_list = d_input_buf.split(expert_size_list, dim=0)
        d_weight_buf = torch.empty_like(weight)

        weight_t = weight.permute(0, 2, 1)

        for i in range(num_linears):
            torch.mm(grad_list[i], weight_t[i], out=d_input_buf_list[i])
            torch.mm(input_list[i], grad_list[i], out=d_weight_buf[i])

        d_input = d_input_buf
        d_weight = d_weight_buf

        if bias is not None:
            d_bias_buf = torch.empty_like(bias)
            for i in range(num_linears):
                torch.sum(grad_list[i], dim=0,
                          keepdim=False, out=d_bias_buf[i])
            d_bias = d_bias_buf
        else:
            d_bias = None

        return d_input, None, d_weight, d_bias


class TreesLayer(torch.nn.Module):
    def __init__(self, input_dim, output_logits_dim, smooth_step_param=1.0, trees_num=1, depth=3, sum_outputs=True):
        super().__init__()
        self.output_logits_dim = output_logits_dim
        self.depth = depth
        self.sum_outputs = sum_outputs
        self.smooth_step_param = smooth_step_param
        num_leaves = 2**self.depth
        num_internal_nodes = num_leaves - 1
        self.trees_num = trees_num
        self.node_weights = torch.nn.Parameter(torch.ones(
            self.trees_num, input_dim, num_internal_nodes))
        self.leaf_weights = torch.nn.Parameter(torch.ones(
            self.trees_num, output_logits_dim, num_leaves))

    def forward(self, input_feature: torch.Tensor):
        tree_logits = []
        for tree_index in range(self.trees_num):
            tree_out, _ = TreeFunction.apply(input_feature, self.node_weights[tree_index, ...],
                                             self.leaf_weights[tree_index, ...], self.output_logits_dim, self.depth,  self.smooth_step_param,  self.training)
            tree_logits.append(tree_out)
        if self.trees_num == 1:
            return tree_logits[0]
        if self.sum_outputs:
            return sum(tree_logits)
        else:
            return torch.stack(tree_logits, axis=1)


class LinearWithGroupNorm(torch.nn.Module):
    """Linear layer with group normalization activation used in LaneGCN."""

    def __init__(self, n_in: int, n_out: int, num_groups: int = 32, activation: bool = True) -> None:
        """
        Initialize layer.
        :param n_in: Number of input channels.
        :param n_out: Number of output channels.
        :param num_groups: Number of groups for GroupNorm.
        :param activation: Boolean indicating whether to apply ReLU activation.
        """
        super().__init__()

        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.norm = nn.GroupNorm(gcd(num_groups, n_out), n_out)
        self.relu = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear layer to input tensor.
        :param x: Input tensor.
        :return: Output of linear layer.
        """
        out = self.linear(x)
        out = self.norm(out)

        if self.activation:
            out = self.relu(out)

        return out


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


class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


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


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array(
            (0, *np.cumsum(field_dims)[:-1]))
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embed = FeaturesEmbedding(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        xs = self.embed(x)
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j, i] * xs[i, j])
        ix = torch.stack(ix, dim=1)
        return ix


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, hiden_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for dim in hiden_dims:
            layers.append(torch.nn.Linear(input_dim, dim))
            # layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim)``
        """
        return self.mlp(x)


class InnerProductNetwork(torch.nn.Module):

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)


class OuterProductNetwork(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        if self.kernel_type == 'mat':
            kp = torch.sum(p.unsqueeze(1) * self.kernel,
                           dim=-1).permute(0, 2, 1)
            return torch.sum(kp * q, -1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)


class CrossNetwork(torch.nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class AttentionalFactorizationMachine(torch.nn.Module):

    def __init__(self, embed_dim, attn_size, dropouts, output_layer=True):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.score_drop = torch.nn.Dropout(p=dropouts)
        self.output_drop = torch.nn.Dropout(p=dropouts)
        self.output_layer = output_layer
        if self.output_layer:
            self.fc = torch.nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        inner_product = p * q
        attn_scores = F.relu(self.attention(inner_product))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        attn_scores = self.score_drop(attn_scores)
        attn_output = torch.sum(attn_scores * inner_product, dim=1)
        attn_output = self.output_drop(attn_output)
        if self.output_layer:
            attn_output = self.fc(attn_output)
        return attn_output


class CompressedInteractionNetwork(torch.nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class AnovaKernel(torch.nn.Module):

    def __init__(self, order, reduce_sum=True):
        super().__init__()
        self.order = order
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        batch_size, num_fields, embed_dim = x.shape
        a_prev = torch.ones((batch_size, num_fields + 1,
                            embed_dim), dtype=torch.float).to(x.device)
        for t in range(self.order):
            a = torch.zeros((batch_size, num_fields + 1, embed_dim),
                            dtype=torch.float).to(x.device)
            a[:, t+1:, :] += x[:, t:, :] * a_prev[:, t:-1, :]
            a = torch.cumsum(a, dim=1)
            a_prev = a
        if self.reduce_sum:
            return torch.sum(a[:, -1, :], dim=-1, keepdim=True)
        else:
            return a[:, -1, :]


class CrossLayer(torch.nn.Module):
    def __init__(self,
                 feature_nums,  # 需要交叉的两个tensor的特征数
                 emb_size=8,
                 w_channels=1,
                 use_mask=True,  # 第一层交叉是特征自己与自己交叉，需要mask重复向量，后续不需要mask
                 use_bn=True,
                 **kwargs):
        super(CrossLayer, self).__init__(**kwargs)
        self.w_channels = w_channels
        self.use_bn = use_bn
        self.feature_num0 = feature_nums[0]
        self.feature_num1 = feature_nums[1]
        self.emb_size = emb_size
        self.use_mask = use_mask

        self.W = torch.nn.Parameter(torch.zeros(
            1, 1, self.w_channels, self.emb_size, self.emb_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.W)
        self.register_parameter('W', self.W)

        ones = torch.ones(self.feature_num1, self.feature_num0)
        ones = torch.tril(ones, diagonal=-1)
        if self.use_mask:
            self.mask = ones
            self.mask = torch.unsqueeze(self.mask, dim=0)
            self.mask = torch.unsqueeze(self.mask, dim=-1)
            self.mask = torch.unsqueeze(self.mask, dim=-1)
            self.mask = self.mask == 1
            self.mask = torch.nn.Parameter(self.mask, requires_grad=False)

        self.interaction_num = torch.sum(ones).numpy().astype(np.int).tolist()
        if self.use_bn:
            self.bn = torch.nn.BatchNorm2d(self.interaction_num)

    def forward(self, xi, xj):
        v_x_1 = torch.unsqueeze(xi, dim=1)  # [batch, 1, feature_num0, emb]
        v_x_2 = torch.unsqueeze(xj, dim=2)  # [batch, feature_num1, 1, emb]
        # [batch, 1, feature_num0, 1, emb]
        v_x_1 = torch.unsqueeze(v_x_1, dim=-2)
        # [batch, feature_num1, 1, emb, 1]
        v_x_2 = torch.unsqueeze(v_x_2, dim=-1)
        # [batch, feature_num1, feature_num0, emb, emb]
        raw_cross = v_x_1 * v_x_2
        if self.use_mask:
            self.mask = self.mask.to(xi.device)
            mask_cross = torch.masked_select(raw_cross, self.mask)
            mask_cross = torch.reshape(
                mask_cross, (-1, self.interaction_num, self.emb_size, self.emb_size))
            # shape mask be explicitly set for eager mode.
            # [batcsh, n*(n-1)/2, emb, emb]
        else:
            mask_cross = torch.reshape(
                raw_cross, [-1, self.interaction_num, self.emb_size, self.emb_size])

        if self.use_bn:
            mask_cross = self.bn(mask_cross)

        # broadcast feature map to w_channel
        # [batch, interaction_num, w_channel,  emb, emb)
        mask_cross = torch.unsqueeze(mask_cross, dim=2)
        mask_cross = torch.repeat_interleave(
            mask_cross, self.w_channels, dim=2)

        # step 3. optional structures
        # [batch, interaction_num, w_channel, emb, emb]
        mask_cross = mask_cross * self.W
        # [batch, w_channel, interaction_num, emb, emb]
        return torch.transpose(mask_cross, 1, 2)


class FuseLayer(torch.nn.Module):

    def __init__(self,
                 feature_nums,  # 需要交叉的两个tensor的特征数
                 w_channels=1,
                 use_bn=True,
                 **kwargs):
        super(FuseLayer, self).__init__(**kwargs)
        self.use_bn = use_bn
        self.w_channels = w_channels
        self.use_bn = use_bn
        self.feature_num0 = feature_nums[0]
        self.feature_num1 = feature_nums[1]
        ones = torch.ones(self.feature_num1, self.feature_num0)
        ones = torch.tril(ones, diagonal=-1)

        self.interaction_num = torch.sum(ones).numpy().astype(np.int).tolist()
        if use_bn:
            self.bn = torch.nn.BatchNorm3d(self.w_channels)

        self.W = torch.nn.Parameter(torch.zeros(
            1, self.w_channels, self.interaction_num,  1, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, inputs):
        if self.use_bn:
            inputs_bn = self.bn(inputs)
        else:
            inputs_bn = inputs
        # step 2. add weight
        z = inputs_bn * self.W
        z = torch.sum(z, dim=-1)
        z = torch.sum(z, dim=2)
        return z


class InteractionLayer(torch.nn.Module):
    def __init__(self,
                 field_nums,
                 emb_size,
                 use_bn=True,
                 use_atten=False,
                 attn_size: Optional[int] = None) -> None:
        super().__init__()
        self.feature_num0 = field_nums[0]
        self.feature_num1 = field_nums[1]
        self.emb_size = emb_size
        self.use_bn = use_bn
        self.use_atten = use_atten
        self.row, self.col = list(), list()
        for i in range(self.feature_num0):
            for j in range(i + 1, self.feature_num1):
                self.row.append(i), self.col.append(j)
        self.interaction_num = len(self.row)
        if self.use_bn:
            self.bn = torch.nn.BatchNorm1d(self.interaction_num)
        self.W = torch.nn.Parameter(torch.zeros(
            1, self.interaction_num, self.emb_size))
        torch.nn.init.xavier_uniform_(self.W)

        if self.use_atten:
            if attn_size is None:
                attn_size = self.emb_size
            self.attention = torch.nn.Linear(emb_size, attn_size)
            self.projection = torch.nn.Linear(attn_size, 1)

    def forward(self, xi, xj):
        p = xi[:, self.row, :]
        q = xj[:, self.col, :]
        out = p * q
        if self.use_bn:
            out = self.bn(out)
        out = out * self.W
        if self.use_atten:
            attn_scores = F.relu(self.attention(out))
            attn_scores = F.softmax(self.projection(attn_scores), dim=1)
            out = attn_scores * out
        return out


class FusionLayer(torch.nn.Module):

    def __init__(self,
                 field_nums,  # 需要交叉的两个tensor的特征数
                 use_bn=False,
                 **kwargs):
        super(FusionLayer, self).__init__(**kwargs)
        self.use_bn = use_bn
        self.use_bn = use_bn
        self.feature_num0 = field_nums[0]
        self.feature_num1 = field_nums[1]
        self.use_bn = use_bn
        ones = torch.ones(self.feature_num1, self.feature_num0)
        ones = torch.tril(ones, diagonal=-1)

        self.interaction_num = torch.sum(ones).numpy().astype(np.int).tolist()
        self.W = torch.nn.Parameter(torch.zeros(
            1, self.interaction_num, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.W)
        if use_bn:
            self.bn = torch.nn.BatchNorm1d(self.interaction_num)

    def forward(self, x):
        if self.use_bn:
            inputs_bn = self.bn(x)
        else:
            inputs_bn = x

        z = inputs_bn * self.W
        z = torch.sum(z, dim=1)
        return z


class FeatureEmbedding(nn.Module):
    def __init__(self,
                 features: List[Dict],
                 embedding_dim):
        super(FeatureEmbedding, self).__init__()
        self.embeds = {}
        self.feat_nums = 0
        for f in features:
            name = f['name']
            type = f['type']

            if type == 'categorical':
                max_idx = f['max_idx']
                padding_idx = f.get("padding_idx")
                pretrain = f.get('pretrain')
                if pretrain:
                    embedding_matrix = torch.load(pretrain)
                else:
                    embedding_matrix = nn.Embedding(max_idx,
                                                    embedding_dim,
                                                    padding_idx=padding_idx)
                self.feat_nums += embedding_dim
            elif type == 'numeric':
                need_embed = f.get('emb')
                numeric_size = f.get('size')
                if need_embed:
                    embedding_matrix = nn.Linear(
                        numeric_size if numeric_size is not None else 1, embedding_dim, bias=False)
                    self.feat_nums += embedding_dim
                else:
                    def embedding_matrix(x): return x
                    self.feat_nums += 1
            elif type == 'sequence':
                max_idx = f['max_idx']
                padding_idx = f.get("padding_idx")
                pretrain = f.get('pretrain')
                if pretrain:
                    embedding_matrix = torch.load(pretrain)
                else:
                    embedding_matrix = nn.Embedding(max_idx,
                                                    embedding_dim,
                                                    padding_idx=padding_idx)
                self.feat_nums += embedding_dim
            else:
                assert (False)
            self.embeds[name] = embedding_matrix

    def forward(self, x: Dict):
        res = {}
        for k, y in x.items():
            res[k] = self.embeds[k](y)
        return self.dictToTensor(res)

    def dictToTensor(self, x: Dict):
        data_list = []
        for k, v in x.items():
            data_list.append(v)

        return torch.cat(data_list, dim=-1)


# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(
            0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self,
                 input_size,
                 num_experts,
                 expert: Optional[nn.Module] = None,
                 noisy_gating=True,
                 k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.k = k
        self.expert = expert
        if expert is None:
            self.expert = MultiLayerPerceptron(
                input_size, hiden_dims=[512, 256, 128, num_experts], dropout=0.5)
        # instantiate experts
        self.experts = nn.ModuleList(
            [copy.deepcopy(self.expert) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(
            input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(
            input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(-1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(
            batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(
            top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(
            top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf(
            (clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, ... ,input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size,... ,num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + \
                (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(
            min(self.k + 1, self.num_experts), dim=-1)
        top_k_logits = top_logits[..., :self.k]
        top_k_indices = top_indices[..., :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits,
                    noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size,...., input_size]
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](
            expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss


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

    def get_mask(self, n, m, device):
        if self.mask is not None:
            return self.mask

        mask = torch.ones((n, m), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=True)
        return mask

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
            mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            attn_mask = mask.mask
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


class AgentSelfAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 num_agent_tokens,
                 dropout=0.5,
                 talking_heads=True,
                 gate=True):
        self.num_heads = num_heads
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.dim_head = hidden_size//num_heads
        self.scale = self.dim_head ** -0.5
        self.agent_tokens = nn.Parameter(
            torch.zeros(num_heads, num_agent_tokens, self.dim_head))
        self.fused_proj = nn.Linear(
            self.hidden_size, self.hidden_size * 3, bias=False)
        self.qa_talking_heads = nn.Conv2d(
            num_heads, num_heads, 1, bias=False) if talking_heads else nn.Identity()
        self.ak_talking_heads = nn.Conv2d(
            num_heads, num_heads, 1, bias=False) if talking_heads else nn.Identity()
        self.qa_dropout = nn.Dropout(dropout)
        self.ak_dropout = nn.Dropout(dropout)
        self.to_gates = nn.Sequential(
            nn.Linear(hidden_size, num_heads),
            rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if gate else None
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x: torch.Tensor, mask=None, **kwargs):
        n, device = x.shape[1], x.device
        query, key, value = self.fused_proj(x).split(self.hidden_size, dim=-1)
        query = rearrange(
            query, 'b n ... (h d) ->  b h ... n d', h=self.num_heads)
        key = rearrange(
            key, 'b n ... (h d) ->  b h ... n d', h=self.num_heads)
        value = rearrange(
            value, 'b n ... (h d) ->  b h ... n d', h=self.num_heads)

        a = repeat(self.agent_tokens, 'h m d -> b h m d', b=n)
        a = a * self.scale
        qa_sim = einsum('b h i d, b h j d -> b h i j', query, a)
        ak_sim = einsum('b h i d, b h j d -> b h i j', a, key)
        qa_attn = qa_sim.softmax(dim=-1)
        ak_attn = ak_sim.softmax(dim=-1)
        qa_attn = self.qa_dropout(qa_attn)
        ak_attn = self.ak_dropout(ak_attn)
        qa_attn = self.qa_talking_heads(qa_attn)
        ak_attn = self.ak_talking_heads(ak_attn)
        agent_gathered_tokens = einsum(
            'b h i j, b h j d -> b h i d', ak_attn, value)
        out = einsum('b h i j, b h j d -> b h i d',
                     qa_attn, agent_gathered_tokens)


class Transpose(nn.Module):
    def __init__(self, *size) -> None:
        super().__init__()
        self.size = size

    def forward(self, x):
        return torch.transpose(x, *self.size)


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


class Router(nn.Module):

    def __init__(self, num_experts, hidden_size, router_jitter_noise, num_experts_per_token):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(hidden_size, self.num_experts)
        self.jitter_noise = router_jitter_noise
        self.num_experts_per_token = num_experts_per_token

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes router probabilities from input hidden states.

        Args:
            hidden_states (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            router_probabilities (`torch.Tensor`):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor`):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        # float32 is used to ensure stability. See the discussion of "selective precision" in
        # https://arxiv.org/abs/2101.03961.
        # We also store the previous dtype to cast back the output to the previous dtype

        if self.jitter_noise > 0 and self.training:
            # Get the lower and upper bound of the uniform distribution
            # Adapted from: https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
            distrib_lower_bound = 1.0 - self.jitter_noise
            distrib_upper_bound = 1.0 + self.jitter_noise

            uniform_distrib = torch.rand(
                hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=False)
            uniform_distrib = uniform_distrib * \
                (distrib_lower_bound - distrib_upper_bound)

            uniform_distrib = uniform_distrib + distrib_upper_bound
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= uniform_distrib

        # Shape: [batch , seq_len, num_experts]
        router_logits = self.classifier(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1)
        return router_probabilities, router_logits

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        r"""
        Generic forward function for every Router class. Each Router expects to have the same input hidden states
        (`hidden_states`) corresponding to the hidden states for each token, the `expert_capacity` corresponding to the
        number of tokens the Router will send to each expert, some Routers can send up to few tokens to each expert.

        Each Router works as the following: it expects the hidden states for each token, gets the `router_probs` and
        `router_logits` from the `router_weights`. This will assign for each token, the raw probability to be assigned
        to an expert. Then each Router class will have to define its own `_compute_routing_instructions`.

        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`] Tuple containing the expert index, the router probs
            and the router logits. The router probabilities and logits are required to compute the loss.
        """
        router_probs, router_logits = self._compute_router_probabilities(
            hidden_states)
        expert_weights, expert_index = torch.topk(
            router_probs, self.num_experts_per_token, dim=-1)

        flat_expert_index = expert_index.view(-1)

        return flat_expert_index, expert_weights, router_logits


class MoERouterLayer(nn.Module):

    def __init__(self, num_experts, hidden_size, router_jitter_noise, num_experts_per_token, expert: nn.Module = None):
        super().__init__()
        # Step 1: Get the correct router according to its class
        self.router = Router(num_experts=num_experts, hidden_size=hidden_size,
                             router_jitter_noise=router_jitter_noise,
                             num_experts_per_token=num_experts_per_token)
        self.num_experts_per_token = num_experts_per_token
        # Step 2: Get the experts
        self.experts = nn.ModuleDict()
        if expert is None:
            expert = FeedForward(hidden_size, glu=True)
        for idx in range(num_experts):
            self.experts[f"expert_{idx}"] = copy.deepcopy(expert)

    def forward(self, hidden_states: torch.Tensor):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, orig_shape[-1])
        # Step 1: Get the router_mask from the router as wel as the probabilities
        router_mask, router_probs, router_logits = self.router(hidden_states)

        # The routers introduced might not always map all the tokens, to a router, which means that some hidden states
        # can be unchanged from one layer to another. That is why the hidden states are cloned before updating only the seleced ones.
        next_states = hidden_states.repeat_interleave(
            self.num_experts_per_token, dim=0)
        y = torch.empty_like(next_states)
        for idx, expert in enumerate(self.experts.values()):
            y[router_mask == idx] += expert(next_states[router_mask == idx])
        y = (y.view(*router_probs.shape, -1) *
             router_probs.unsqueeze(-1)).sum(dim=1)
        return y.view(*orig_shape), router_logits.reshape([orig_shape[0], -1, router_logits.shape[-1]])


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
