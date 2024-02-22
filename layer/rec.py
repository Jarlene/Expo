
import torch.nn.functional as F
import torch
from torch.cuda.amp import custom_fwd, custom_bwd
import numpy as np
from typing import List, Optional
import torch
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













