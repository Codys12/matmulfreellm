from typing import Optional

from transformers.configuration_utils import PretrainedConfig

class DSHybridConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DSHybridModel`]. It is used to instantiate an
    DS-Hybrid model according to the specified arguments, defining the model architecture. This configuration
    combines elements from both Transformer and HGRN architectures.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DSHybridModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the model.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads to use in attention. See Grouped Query Attention.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        hidden_act (`str`, *optional*, defaults to "silu"):
            The non-linear activation function in the model.
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to True):
            Whether or not the model should return the last key/values attentions.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to False):
            Whether the model's input and output word embeddings should be tied.
        attn_mode (`str`, *optional*, defaults to "fused_recurrent"):
            Attention mode for the HGRN layers.
        expand_ratio (`int`, *optional*, defaults to 1):
            Expansion ratio for the HGRN layers.
        use_short_conv (`bool`, *optional*, defaults to False):
            Whether to use short convolutions in the HGRN layers.
        conv_size (`int`, *optional*, defaults to 4):
            Size of the convolution kernel in HGRN layers.
        share_conv_kernel (`bool`, *optional*, defaults to True):
            Whether to share convolution kernels across layers.
        use_lower_bound (`bool`, *optional*, defaults to True):
            Whether to use lower bound in HGRN computations.
        num_prediction_heads (`int`, *optional*, defaults to 1):
            Number of prediction heads for predicting multiple future tokens.
        output_attention_logit (`bool`, *optional*, defaults to True):
            Whether to output the scaling logit for attention layers.
        attention_logit_aux_loss_coef (`float`, *optional*, defaults to 0.01):
            The auxiliary loss coefficient for the attention scaling logit.
        output_router_logits (`bool`, *optional*, defaults to False):
            Whether or not the router logits should be returned by the model.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The auxiliary loss factor for the total loss in routing.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            The number of experts to route per-token.
        num_experts (`int`, *optional*, defaults to 16):
            Number of experts per Sparse MLP layer.
        expert_layer_period (`int`, *optional*, defaults to 2):
            Once in this many layers, we will have an expert layer.
        expert_layer_offset (`int`, *optional*, defaults to 1):
            The first layer index that contains an expert MLP layer.
        attn_layer_period (`int`, *optional*, defaults to 8):
            Once in this many layers, we will have a vanilla attention layer.
        attn_layer_offset (`int`, *optional*, defaults to 4):
            The first layer index that contains a vanilla attention MLP layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        sliding_window (`int`, *optional*, defaults to None):
            Size of the sliding window for attention, if applicable.
    """

    model_type = "ds_hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        attn_mode="fused_recurrent",
        fuse_cross_entropy=False,
        num_heads: Optional[int] = 1,
        expand_ratio=1,
        hidden_ratio=4,
        use_short_conv=False,
        conv_size=4,
        share_conv_kernel=True,
        use_lower_bound=True,
        num_prediction_heads=1,
        output_attention_logit=True,
        attention_logit_aux_loss_coef=0.01,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        num_experts_per_tok=2,
        num_experts=16,
        expert_layer_period=2,
        expert_layer_offset=1,
        attn_layer_period=8,
        attn_layer_offset=4,
        attention_dropout=0.0,
        sliding_window=None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attn_mode = attn_mode
        self.fuse_cross_entropy = fuse_cross_entropy
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.hidden_ratio = hidden_ratio
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.share_conv_kernel = share_conv_kernel
        self.use_lower_bound = use_lower_bound
        self.num_prediction_heads = num_prediction_heads
        self.output_attention_logit = output_attention_logit
        self.attention_logit_aux_loss_coef = attention_logit_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.expert_layer_period = expert_layer_period
        self.expert_layer_offset = expert_layer_offset
        self.attn_layer_period = attn_layer_period
        self.attn_layer_offset = attn_layer_offset
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        return [
            "attention" if i % self.attn_layer_period == self.attn_layer_offset else "hgrn"
            for i in range(self.num_hidden_layers)
        ]

    @property
    def layers_num_experts(self):
        return [
            self.num_experts if i % self.expert_layer_period == self.expert_layer_offset else 1
            for i in range(self.num_hidden_layers)
        ]