# Huggingface compatible module
import copy
import math
import random
import warnings
from typing import Any, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import logging

# huggingface transformers imports
from transformers import PretrainedConfig
from transformers.models.bart.modeling_bart import (
    BartEncoder,
    BartAttention,
    BartLearnedPositionalEmbedding,
    ACT2FN,
)

from .modeling_outputs import MemBartEncoderOutput
from .membart_attention import MemBartEncoderAttention

# pylint:disable=no-member

logger = logging.getLogger(__name__)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

class UpdateGate(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self, input_dim: int, inner_dim: int, pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = pooler_dropout
        # self.zero_weight = nn.Parameter(torch.zeros(inner_dim, 1).T)
        # self.zero_bias = nn.Parameter(torch.zeros(1))
        self.dense2 = nn.Linear(input_dim, 1)
        self.act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        gate = self.dense2(hidden_states)
        gate = torch.sigmoid(gate)
        return gate


class MemBartEncoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # Fuse Attention Layer
        self.self_attn = MemBartEncoderAttention(
            embed_dim=self.embed_dim, num_heads=config.encoder_attention_heads, dropout=config.attention_dropout,
        )
        # Bart Layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        # Memory Module Layer
        self.mem_self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.mem_fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.mem_fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.mem_final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape *(seq_len, batch, embed_dim)*
            attention_mask (`torch.FloatTensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                *(encoder_attention_heads,)*.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        memory_residual = memory_states

        # Fusion Layer
        hidden_states, memory_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            memory_states=memory_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        # Bart Layer
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # Memory Module Layer
        memory_states = F.dropout(memory_states, p=self.dropout, training=self.training)
        memory_states = memory_residual + memory_states
        memory_states = self.mem_self_attn_layer_norm(memory_states)
        memory_residual = memory_states
        memory_states = self.activation_fn(self.mem_fc1(memory_states))
        memory_states = F.dropout(memory_states, p=self.activation_dropout, training=self.training)
        memory_states = self.mem_fc2(memory_states)
        memory_states = F.dropout(memory_states, p=self.dropout, training=self.training)
        memory_states = memory_residual + memory_states
        memory_states = self.mem_final_layer_norm(memory_states)

        # if hidden_states.dtype == torch.float16 and (
        #     torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        # ):
        #     clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        #     hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += attn_weights

        return outputs, memory_states


class MemBartEncoder(BartEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`MemBartEncoderLayer`.

    Args:
        config: MemBartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: PretrainedConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.head_dim = config.d_model // config.encoder_attention_heads

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(config.max_position_embeddings, embed_dim,)
        self.layers = nn.ModuleList([MemBartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing

        # memory states
        self.memory_layer_norm = nn.LayerNorm(embed_dim)
        self.memory_bias = nn.Parameter(torch.randn(1, config.memory_len, config.d_model))
        self.memory_gate_net = UpdateGate(embed_dim, embed_dim, pooler_dropout=self.dropout)

        self.post_init()
        nn.init.normal_(self.memory_bias.data, std=config.init_std)

    def forward(
        self,
        input_ids=None,
        memory_states=None,
        memory_resets=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        r"""
        Args:
            see BartEncoder
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # construct memory
        batch_size = hidden_states.shape[0]
        if memory_states is None:
            memory_states = self.construct_memory(batch_size)

        # expand attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).bool()
        else:
            attention_mask = attention_mask.bool()

        attention_mask = F.pad(attention_mask, (self.config.memory_len, 0), "constant", True)

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        encoder_attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
        encoder_attention_mask[:, :, :self.config.memory_len, :self.config.memory_len].fill_(torch.finfo(inputs_embeds.dtype).min)
        for i in range(self.config.memory_len):
            encoder_attention_mask[:, :, i, i].fill_(0)

        # Normalize memory states
        if memory_resets is None:
            # default to not reset
            memory_resets = torch.zeros_like(memory_states)[:, 0, 0]

        memory_resets = memory_resets.float()
        memory_states = self.memory_layer_norm(
            (1 - memory_resets).view(batch_size, 1, 1) * memory_states + self.memory_bias
        )
        old_memory_states = memory_states

        # record states
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs, memory_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        memory_states,
                        encoder_attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs, memory_states = encoder_layer(
                        hidden_states,
                        memory_states,
                        encoder_attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                # all_cached_hidden_states.append(cache_hidden_states)
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Update memory
        memory_alpha = self.memory_gate_net(memory_states)
        memory_states = memory_alpha * memory_states + (1 - memory_alpha) * old_memory_states

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        hidden_states = torch.cat([memory_states, hidden_states], dim=1)

        return MemBartEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            memory_states=memory_states,
            encoder_attention_mask=attention_mask,
        )

    def construct_memory(self, batch_size):
        # zeros
        device = self.memory_bias.device
        memory_states = torch.zeros(batch_size, self.config.memory_len, self.config.d_model).to(device)
        return memory_states

