# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from mmfreelm.models.ds_hybrid.modeling_outputs import (
    DSHybridModelOutputWithPast, DSHybridCausalLMOutputWithPast)
from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.models.ds_hybrid.configuration_ds_hybrid import DSHybridConfig
from mmfreelm.models.utils import RecurrentCache
from mmfreelm.modules import FusedCrossEntropyLoss, FusedSoftCrossEntropyLoss, RMSNorm
from mmfreelm.modules.layernorm import RMSNormLinear
from mmfreelm.modules.activations import swiglu_linear, swiglu
#from mmfreelm.ops.bitnet import BitLinear_Fuse as BitLinear
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear

logger = logging.get_logger(__name__)



def load_balancing_loss_func(
    router_logits: Tuple[torch.Tensor, ...],
    num_experts: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the Mutual Information (MI) loss for the Mixture of Experts (MoE) model.
    
    Args:
        router_logits (Tuple[torch.Tensor, ...]): Tuple of logits from the router, 
                                                  each with shape [batch_size * sequence_length, num_experts].
        num_experts (int): Number of experts in the model.
        attention_mask (Optional[torch.Tensor]): Attention mask, shape [batch_size, sequence_length].
    
    Returns:
        torch.Tensor: The MI loss.
    """
    # Concatenate all router logits
    router_logits = torch.cat(router_logits, dim=0)
    
    if attention_mask is not None:
        batch_size, sequence_length = attention_mask.shape
    else:
        batch_size = router_logits.shape[0] // len(router_logits)
        sequence_length = len(router_logits)
    
    # Compute expert probabilities
    expert_probs = F.softmax(router_logits, dim=-1)  # [total_tokens, num_experts]
    
    # Compute entropy of expert distribution H(e)
    avg_expert_probs = torch.mean(expert_probs, dim=0)  # [num_experts]
    entropy = -torch.sum(avg_expert_probs * torch.log(avg_expert_probs + 1e-10))
    
    # Compute conditional entropy H(e|X)
    conditional_entropy = -torch.mean(torch.sum(expert_probs * torch.log(expert_probs + 1e-10), dim=-1))
    
    # Compute MI loss
    mi_loss = -entropy + conditional_entropy
    
    if attention_mask is not None:
        # Repeat the attention mask to match the total number of tokens
        mask = attention_mask.repeat(len(router_logits) // batch_size, 1).view(-1, 1).float()
        mi_loss = mi_loss * mask
        mi_loss = torch.sum(mi_loss) / torch.sum(mask)
    
    return mi_loss

def distillation_loss(self, logits, soft_targets):
    """
    Compute the distillation loss.
    
    Args:
    - logits: The model's output logits (batch_size, seq_len, vocab_size)
    - soft_targets: Tensor of shape (batch_size, seq_len, top_k, 2) where
      [:, :, :, 0] contains the top-k indices and
      [:, :, :, 1] contains the corresponding probabilities
    
    Returns:
    - loss: The computed distillation loss
    """
    batch_size, seq_len, vocab_size = logits.shape
    _, _, top_k, _ = soft_targets.shape
    
    # Create a mask for the top-k positions
    mask = torch.zeros(batch_size, seq_len, vocab_size, device=logits.device)
    mask.scatter_(2, soft_targets[:, :, :, 0].long(), 1)
    
    # Apply log_softmax only to the top-k positions
    log_probs = torch.where(mask.bool(), F.log_softmax(logits, dim=-1), torch.zeros_like(logits))
    
    # Compute the loss
    loss = -torch.sum(soft_targets[:, :, :, 1] * log_probs.gather(2, soft_targets[:, :, :, 0].long()))
    
    # Normalize the loss
    loss = loss / (batch_size * seq_len)
    
    return loss

class DSHybridBitMLP(nn.Module):

    def __init__(
        self,
        config: DSHybridConfig
    ):
        super().__init__()

        hidden_size = config.hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        hidden_ratio = config.hidden_ratio if config.hidden_ratio is not None else 4
        
        self.intermediate_size = config.intermediate_size
        if config.intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            self.intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        
        self.hidden_size = hidden_size
        self.hidden_ratio = hidden_ratio

        self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size * 2, bias=False)
        #self.gate_proj_bit = RMSNormLinear(self.hidden_size)
        self.down_proj = BitLinear(self.intermediate_size, self.hidden_size, bias=False)
        #self.gate_proj_bit = RMSNormLinear(self.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        z = self.down_proj(swiglu(gate, y))
        return z

class ModifiedFFF(nn.Module):
    """
    A modified implementation of fast feedforward networks with 3 linear layers and binary tree gating.
    """
    def __init__(self, config: DSHybridConfig):
        """
        Initializes a modified fast feedforward network (FFF).

        Parameters
        ----------
        input_width : int
            The width of the input.
        hidden_width : int
            The width of the hidden layer.
        output_width : int
            The width of the output.
        depth : int
            The depth of the FFF tree. Will result in 2**depth leaves.
        activation : torch.nn.Module, optional
            The activation function to use. Defaults to `torch.nn.ReLU()`.
        """
        super().__init__()
        self.input_width = config.hidden_size
        self.hidden_width = config.intermediate_size
        self.output_width = config.hidden_size
        self.activation = ACT2FN[config.hidden_act]
        self.depth = config.depth

        if config.depth < 1 or config.hidden_size <= 0 or config.intermediate_size <= 0:
            raise ValueError("input/hidden/output widths must be positive integers and depth must be at least 1")
        if config.intermediate_size % (2**config.depth) != 0:
            raise ValueError("hidden_width must be divisible by 2**depth")

        self.n_leaves = 2 ** config.depth

        # First linear layer: input to hidden
        self.layer1 = BitLinear(config.hidden_size, config.intermediate_size, bias=False)

        # Second linear layer: hidden to output
        self.layer2 = BitLinear(config.intermediate_size, config.hidden_size, bias=False)

        # Third linear layer: input to gating
        self.layer3 = BitLinear(config.hidden_size, self.n_leaves - 1, bias=False)

    def create_gating_vector(self, gate_outputs: torch.Tensor) -> torch.Tensor:
        """
        Creates a gating vector using the binary tree structure.

        Parameters
        ----------
        gate_outputs : torch.Tensor
            The output of the gating layer, shape (batch_size, n_nodes)
            where n_nodes = 2^depth - 1

        Returns
        -------
        torch.Tensor
            The gating vector, shape (batch_size, n_leaves)
            where n_leaves = 2^depth
        """
        batch_size = gate_outputs.shape[0]
        device = gate_outputs.device

        # Initialize the mixture with ones
        current_mixture = torch.ones((batch_size, self.n_leaves), device=device)

        for current_depth in range(self.depth):
            platform = 2**current_depth - 1
            next_platform = 2**(current_depth + 1) - 1
            n_nodes = 2**current_depth

            # Get the boundary effect for the current level
            boundary_effect = torch.sigmoid(gate_outputs[:, platform:next_platform])  # (batch_size, n_nodes)
            not_boundary_effect = 1 - boundary_effect  # (batch_size, n_nodes)

            # Prepare mixture modifier
            mixture_modifier = torch.cat(
                (not_boundary_effect.unsqueeze(-1), boundary_effect.unsqueeze(-1)),
                dim=-1
            ).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)  # (batch_size, n_nodes*2, 1)

            # Apply mixture modifier
            current_mixture = current_mixture.view(batch_size, 2 * n_nodes, self.n_leaves // (2 * n_nodes))  # (batch_size, 2*n_nodes, n_leaves // (2*n_nodes))
            current_mixture.mul_(mixture_modifier)  # (batch_size, 2*n_nodes, n_leaves // (2*n_nodes))
            current_mixture = current_mixture.flatten(start_dim=1, end_dim=2)

        return current_mixture

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of this modified FFF.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Must have shape (..., input_width).

        Returns
        -------
        torch.Tensor
            The output tensor. Will have shape (..., output_width).
        """
        original_shape = x.shape
        x = x.view(-1, x.shape[-1])
        batch_size = x.shape[0]

        # First layer: input to hidden
        hidden = self.layer1(x)
        hidden = self.activation(hidden)

        # Third layer: input to gating
        gate_outputs = self.layer3(x)
        gating_vector = self.create_gating_vector(gate_outputs)

        # Apply gating
        hidden_per_leaf = self.hidden_width // self.n_leaves
        gating_vector = gating_vector.unsqueeze(2).expand(-1, -1, hidden_per_leaf).reshape(batch_size, self.hidden_width)
        gated_hidden = hidden * gating_vector

        # Second layer: gated hidden to output
        output = self.layer2(gated_hidden)

        return output.view(*original_shape[:-1], self.output_width)

class DSHybridMoEBlock(nn.Module):
    def __init__(self, config:DSHybridConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_experts

        self.router = BitLinear(self.hidden_dim, self.num_experts, bias=False)
        if config.fast_feed_forward:
            self.experts = nn.ModuleList([ModifiedFFF(config) for _ in range(self.num_experts)])
        else:
            self.experts = nn.ModuleList([DSHybridBitMLP(config) for _ in range(self.num_experts)])
        

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        
        # Convert routing_weights to input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Compute all expert outputs at once
        expert_outputs = torch.stack([expert(hidden_states) for expert in self.experts], dim=1)
        
        # Apply routing weights to expert outputs
        weighted_outputs = expert_outputs * routing_weights.unsqueeze(-1)
        
        # Sum the weighted outputs across all experts
        final_hidden_states = weighted_outputs.sum(dim=1)

        # Reshape the output back to the original shape
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits

class DSHybridAttentionDecoderLayer(nn.Module):
    def __init__(self, config: DSHybridConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attention_gate = BitLinear(self.hidden_size, 1, bias=False)
        self.scalar_activation = nn.Sigmoid()

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        # Attention implementation commented out as requested
        # self.attn = HGRNBitAttention(
        #     mode=config.attn_mode,
        #     hidden_size=config.hidden_size,
        #     num_heads=config.num_heads,
        #     expand_ratio=config.expand_ratio,
        #     use_short_conv=config.use_short_conv,
        #     conv_size=config.conv_size,
        #     share_conv_kernel=config.share_conv_kernel,
        #     layernorm_eps=config.rms_norm_eps,
        #     layer_idx=layer_idx
        # )
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        
        num_experts = config.layers_num_experts[layer_idx]
        if num_experts > 1:
            ffn_layer_class = DSHybridMoEBlock
        elif config.fast_feed_forward:
            ffn_layer_class = ModifiedFFF
        else:
            ffn_layer_class = DSHybridBitMLP
        self.mlp = ffn_layer_class(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        output_router_logits: Optional[bool] = False,
        output_attention_logits: Optional[bool] = False
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        
        # Attention step commented out
        # hidden_states, attentions, past_key_values = self.attn(
        #     hidden_states=hidden_states,
        #     attention_mask=attention_mask,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     lower_bound=lower_bound
        # )
        
        gate = self.attention_gate(hidden_states)
        # batch * seq_len * 1
        gate_weights = self.scalar_activation(gate)

        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        if isinstance(self.mlp, DSHybridMoEBlock):
            hidden_states, router_logits_output = self.mlp(hidden_states)
            router_logits = router_logits_output if output_router_logits else None
        else:
            hidden_states = self.mlp(hidden_states)
            router_logits = None
        hidden_states = hidden_states * gate_weights
        hidden_states = residual + hidden_states

        attention_logits = gate if output_attention_logits else None

        outputs = (hidden_states, attentions, past_key_values, router_logits, attention_logits)

        return outputs

class DSHybridBitDecoderLayer(nn.Module):
    def __init__(self, config: DSHybridConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.attn = HGRNBitAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_ratio=config.expand_ratio,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            share_conv_kernel=config.share_conv_kernel,
            layernorm_eps=config.rms_norm_eps,
            layer_idx=layer_idx
        )
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        
        num_experts = config.layers_num_experts[layer_idx]
        if num_experts > 1:
            ffn_layer_class = DSHybridMoEBlock
        elif config.fast_feed_forward:
            ffn_layer_class = ModifiedFFF
        else:
            ffn_layer_class = DSHybridBitMLP
        self.mlp = ffn_layer_class(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        output_router_logits: Optional[bool] = False,
        output_attention_logits: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            lower_bound=lower_bound
        )
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        router_logits = None
        attention_logits = None
        if isinstance(self.mlp, DSHybridMoEBlock):
            hidden_states, router_logits_output = self.mlp(hidden_states)
            router_logits = router_logits_output if output_router_logits else None
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values, router_logits, attention_logits)

        return outputs

class DSHybridBitPreTrainedModel(PreTrainedModel):

    config_class = DSHybridConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['DSHybridBitBlock']

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d, BitLinear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)

ALL_DECODER_LAYER_TYPES = {"attention": DSHybridAttentionDecoderLayer, "hgrn": DSHybridBitDecoderLayer}

class DSHybridModel(DSHybridBitPreTrainedModel):

    def __init__(self, config: DSHybridConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if config.use_lower_bound:
            self.lower_bounds = nn.Parameter(torch.zeros(config.num_hidden_layers, config.hidden_size))
        decoder_layers = []
        for i in range(config.num_hidden_layers):
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[i]]
            decoder_layers.append(layer_class(config, layer_idx=i))        
        self.layers = nn.ModuleList(decoder_layers)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_attention_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, DSHybridModelOutputWithPast]:
        if output_attentions:
            warnings.warn("`DSHybridBitModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        #PARITY HERE
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        output_attention_logits = output_attention_logits if output_attention_logits is not None else self.config.output_attention_logits

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if use_cache:
            if past_key_values is None:
                pass
                past_key_values = [layer.attn.init_state(batch_size) for layer in self.layers]
            if not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        all_attention_logits = () if output_attention_logits else None

        if self.config.use_lower_bound:
            lower_bounds = self.lower_bounds.softmax(0)
            lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            lower_bound = lower_bounds[i] if self.config.use_lower_bound else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    lower_bound,
                    output_router_logits,
                    output_attention_logits
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    lower_bound=lower_bound,
                    output_router_logits=output_router_logits,
                    output_attention_logits=output_attention_logits
                )

            if output_attentions:
                attns = layer_outputs[1]
                all_attns += (attns,)

        hidden_states = self.norm(layer_outputs[0])

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if output_router_logits:
            router_logits = layer_outputs[3]
            all_router_logits += (router_logits,)

        if output_attention_logits:
            attention_logits = layer_outputs[4]
            all_attention_logits += (attention_logits,)

        next_cache = None
        if use_cache:
            next_cache = past_key_values.to_legacy_cache()
        if not return_dict:
            return (hidden_states, next_cache, all_hidden_states, all_attns, all_router_logits, all_attention_logits)
        return DSHybridModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns,
            router_logits=all_router_logits,
            attention_logits=all_attention_logits
        )


class DSHybridForCausalLM(DSHybridBitPreTrainedModel):
    _tied_weights_keys = ["lm_heads.0.weight"]  # Only tie the first head

    def __init__(self, config):
        super().__init__(config)
        self.model = DSHybridModel(config)
        self.vocab_size = config.vocab_size
        self.num_heads = config.num_heads
        
        # Create multiple heads
        self.lm_heads = nn.ModuleList([
            BitLinear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(self.num_heads)
        ])

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_heads[0]  # Return the first head for compatibility

    def set_output_embeddings(self, new_embeddings):
        self.lm_heads[0] = new_embeddings  # Set only the first head

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # only last token for `inputs_ids` if the `past_key_values` is passed along.
        if past_key_values is not None:
            if not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(past_key_values, input_ids.shape[1] - 1)
            input_ids, attention_mask = input_ids[:, -1:], attention_mask[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        })
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        soft_targets: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_attention_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DSHybridCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        output_attention_logits = output_attention_logits if output_attention_logits is not None else self.config.output_attention_logits
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_attention_logits=output_attention_logits,
            return_dict=return_dict
        )

        hidden_states = outputs[0]

        # Prepare loss functions
        if labels is not None:
            if self.config.fuse_cross_entropy:
                hard_loss_fct = FusedCrossEntropyLoss(inplace_backward=True)
            else:
                hard_loss_fct = nn.CrossEntropyLoss()

        soft_loss_fct = FusedSoftCrossEntropyLoss(inplace_backward=True)

        total_loss = None
        all_logits = []

        for i, head in enumerate(self.lm_heads):
            # Forward pass for this head
            logits = head(hidden_states)
            
            head_loss = None
            if labels is not None:
                labels = torch.cat((labels[..., 1:], torch.full_like(labels[..., :1], hard_loss_fct.ignore_index)), 1)
                head_loss = hard_loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

            if soft_targets is not None:
                soft_targets = torch.cat((soft_targets[..., 1:, :], torch.full_like(soft_targets[..., :1, :], soft_loss_fct.ignore_index)), 1)
                soft_loss = soft_loss_func(logits, soft_targets)
                head_loss = soft_loss if head_loss is None else head_loss + soft_loss

            total_loss = head_loss if total_loss is None else total_loss + head_loss

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-2],
                self.config.num_experts,
                attention_mask,
            )
            if labels is not None or soft_targets is not None:
                total_loss += self.config.router_aux_loss_coef * aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output

        return DSHybridCausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            attention_logits=outputs.attention_logits
        )