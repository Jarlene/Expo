import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange, einsum, repeat
import torch
from math import ceil

class Mamba(nn.Module):
    """
    Initialize a single Mamba block.

    Args:
        dim (int): The input dimension.
        d_state (int): The state dimension. Default is 16.
        expand (int): The expansion factor. Default is 2.
        d_conv (int): The dimension of the convolutional kernel. Default is 4.
        conv_bias (bool): Whether to include bias in the convolutional layer. Default is True.
        bias (bool): Whether to include bias in the linear layers. Default is False.
    """

    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=4,
        expand=2,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias

        # If dt_rank is not provided, set it to ceil(dim / d_state)
        dt_rank = ceil(self.dim / 16)
        self.dt_rank = dt_rank

        # If dim_inner is not provided, set it to dim * expand
        dim_inner = self.dim * self.expand
        self.dim_inner = dim_inner

        # If dim_inner is not provided, set it to dim * expand
        self.in_proj = nn.Linear(self.dim, dim_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=dim_inner,
            out_channels=dim_inner,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=dim_inner,
            padding=self.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(
            dim_inner, dt_rank + self.d_state * 2, bias=False
        )

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(dt_rank, dim_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=dim_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(dim_inner, dtype=torch.float32))
        self.out_proj = nn.Linear(dim_inner, self.dim, bias=bias)

    def forward(self, x: torch.Tensor):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)


        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        x_and_res = rearrange(x_and_res, "b l x -> b x l")
        (x, res) = x_and_res.split(
            split_size=[self.dim_inner, self.dim_inner], dim=1
        )

        x = self.conv1d(x)[:, :, :l]
        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(rearrange(y, "b dim l -> b l dim"))

        return output

    def ssm(self, x: torch.Tensor):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, d_in, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, d_in, l)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4)

        A = -torch.exp(self.A_log)  # shape (d_in, n)
        D = self.D
        x_dbl = rearrange(x, "b d l -> b l d")
        x_dbl = self.x_proj(x_dbl)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(
            split_size=[self.dt_rank, n, n], dim=-1
        )  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(
            x, delta, A, B, C, D
        )  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, d_in, l)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, d_in, l)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        (b, d_in, l) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (Δ, A, B)  (see Section 2 Equation 4 in the Mamba paper [1])
        # Note that B is parameterized directly
        deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b d_in l n"))
        deltaB_u = einsum(
            delta, B, u, "b l d_in, b l n, b d_in l -> b d_in l n"
        )

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        x = torch.zeros((b, d_in, n), dtype=deltaA.dtype).to(deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            y = einsum(x, C[:, i, :], "b d_in n , b n -> b d_in")
            ys.append(y)
        y = torch.stack(ys, dim=2)  # (b d_in l)

        if D is not None:
            y = y + u * rearrange(D, "d_in -> d_in 1")

        return y




class S4Block(nn.Module):
    """General block design wrapping an inner layer. Currently only layer=FFTConv is supported, but easy to incorporate others.

    Arguments:
    - bottleneck: Reduce dimension of inner layer (e.g. used in GSS).
    - gate: Add multiplicative gating (e.g. used in GSS), which is essentially a multiplicative instead of additive residual branch.
    - gate_act: Activation function to apply on the gate residual branch.
    - mult_act: Activation function to apply after gate multiplication (e.g. GELU in GSS).
    - final_act: Activation function to apply after final linear layer. 'id' for no activation, None for no linear layer at all.

    - initializer: Initializer on final linear layer.
    - weight_norm: Weight normalization on final linear layer.
    - dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

    - transposed: Choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=model dimension]

    Other options are all experimental and should not need to be configured.
    """

    def __init__(
        self,
        d_model,
        bottleneck=None,
        activation='gelu',
        gate=None,
        gate_act=None,
        mult_act=None,
        final_act='glu',
        postact=None,
        initializer=None,
        weight_norm=False,
        dropout=0.0,
        tie_dropout=False,
        transposed=True,
        layer='fftconv',
        **layer_args,  # Arguments into inner layer (e.g. FFTConv)
    ):
        super().__init__()

        self.d_model = d_model
        self.transposed = transposed

        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            self.d_model = self.d_model // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=None,
                activate=False,
                weight_norm=weight_norm,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=False,
                initializer=initializer,
                activation=gate_act,
                activate=True,
                weight_norm=weight_norm,
            )
            if self.layer.d_output != self.d_model * gate:
                self.output_gate = LinearActivation(
                    self.d_model*self.channels,
                    self.d_model * gate,
                    transposed=False,
                    initializer=initializer,
                    activation=None,
                    activate=False,
                    weight_norm=weight_norm,
                )

        # Currently this module only uses FFTConv for its inner module
        # But the options here are all agnostic to the inner block
        # If other types of inner layers are desired, it is easy
        # to add an option to swap a different module in
        # self.layer = FFTConv(d_model, transposed=False, dropout=dropout, tie_dropout=tie_dropout, **layer_args)
        layer_cfg = layer_args.copy()
        layer_cfg['_name_'] = layer
        layer_cfg['transposed'] = False
        layer_cfg['dropout'] = dropout
        self.layer = utils.instantiate(registry.layer, layer_cfg, d_model)

        # Pointwise operations
        # Activation after layer
        self.activation = Activation(activation)

        # Activation after (optional) multiplication by gate branch
        self.mult_activation = Activation(mult_act)
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact is not None:
            assert final_act is None
            log.warning("Warning: 'postact' option changed to 'final_act' and will be removed in a future version.")
            final_act, postact = postact, final_act
        if final_act is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                self.d_model*gate if gate is not None else self.layer.d_output,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=final_act,
                activate=True,
                weight_norm=weight_norm,
            )



    def forward(self, x, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as x
        """
        if self.transposed: x = rearrange(x, 'b d ... -> b ... d')
        L = x.size(1)

        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=x.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, x.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device)[:, None] < lengths[:, None, None], 1., 0.)
            x = x * mask

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)

        y, state = self.layer(x, **kwargs)

        y = self.activation(y)

        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)

        if self.transposed: y = rearrange(y, 'b d ... -> b ... d')

        return y, state

    def setup_step(self, **kwargs):
        self.layer.setup_step(**kwargs)

    def step(self, x, state):
        """Step one time step as a recurrent model. Intended to be used during validation.

        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)
        y, next_state = self.layer.step(x, state) # (B C H)
        y = self.activation(y)
        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.layer.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor
