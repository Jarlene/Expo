from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class SSMFormerConfig(PretrainedConfig):
    model_type = "ssmformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=64798,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        d_state=16,
        d_conv=4,
        expand=2,
        rms_norm_eps=1e-6,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_scaling=None,
        rope_theta=1000000.0,
        sliding_window=4096,
        use_moe=False,
        moe_soft=True,
        moe_num_experts=32,
        moe_num_slots=64,
        moe_add_noise=True,
        moe_noise_mult=1.0,
        moe_num_experts_per_token=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand =expand
        self.depth= depth
        
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.rope_theta = rope_theta

        self.use_moe = use_moe
        self.moe_soft = moe_soft
        self.moe_num_experts = moe_num_experts
        self.moe_num_experts_per_token = moe_num_experts_per_token
        self.moe_num_slots = moe_num_slots
        self.moe_add_noise = moe_add_noise
        self.moe_noise_mult = moe_noise_mult

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict):
            raise ValueError(
                "`rope_scaling` must be a dictionary, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic", "yarn", "dynamic-yarn"]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic', 'yarn', 'dynamic-yarn'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(
                f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")
        if rope_scaling_type == "yarn" or rope_scaling_type == "dynamic-yarn":
            original_max_position_embeddings = self.rope_scaling.get(
                "original_max_position_embeddings", None)
            if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
                raise ValueError(
                    f"`rope_scaling.original_max_position_embeddings` must be set to an int when using yarn, and dynamic-yarn")
