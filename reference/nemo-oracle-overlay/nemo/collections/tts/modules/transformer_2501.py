# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Escenda for MagpieTTS-RT, 2026.
import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from nemo.collections.tts.modules.ffn_modules import PositionwiseConvFF, PositionwiseConvFFIncrementalState
from nemo.collections.tts.modules.moe_modules import PositionwiseConvFFMoE
from nemo.utils import logging


@dataclass(frozen=True)
class TensorSignature:
    """Identity of an immutable conditioning tensor within one inference session."""

    data_ptr: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorSignature":
        return cls(
            data_ptr=tensor.data_ptr(),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=tensor.device,
        )


@dataclass
class SelfAttentionIncrementalState:
    """Preallocated K/V and mask storage owned by one inference session."""

    key: torch.Tensor
    value: torch.Tensor
    key_mask: torch.Tensor
    max_length: int


@dataclass
class CrossAttentionIncrementalState:
    """Cross-attention K/V for one immutable session context."""

    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    context_signature: Optional[TensorSignature] = None
    mask_signature: Optional[TensorSignature] = None


@dataclass(frozen=True)
class MoEFeedForwardIncrementalState:
    """Stateless incremental contract for a pointwise MoE feed-forward layer."""

    max_length: int


@dataclass
class TransformerLayerIncrementalState:
    """All caller-owned incremental state for one Transformer layer."""

    self_attention: SelfAttentionIncrementalState
    cross_attention: Optional[CrossAttentionIncrementalState]
    feed_forward: Union[PositionwiseConvFFIncrementalState, MoEFeedForwardIncrementalState]


@dataclass
class TransformerIncrementalState:
    """Independent, reusable buffers for one autoregressive session."""

    layers: List[TransformerLayerIncrementalState]
    batch_size: int
    max_length: int
    device: torch.device
    dtype: torch.dtype


class MoEInferenceRouting(NamedTuple):
    """Typed routing tensors emitted by one MoE layer during inference."""

    router_logits: torch.Tensor
    router_probs: torch.Tensor
    expert_indices: torch.Tensor


class TransformerLayerInferenceOutput(NamedTuple):
    """Incremental output for one layer without retained attention matrices."""

    output: torch.Tensor
    alignment_score: Optional[torch.Tensor]
    moe_routing: Optional[MoEInferenceRouting]


class TransformerInferenceOutput(NamedTuple):
    """Incremental stack output and the exact auxiliary tensors requested by its caller."""

    output: torch.Tensor
    alignment_scores: Optional[torch.Tensor]
    moe_routing: Tuple[MoEInferenceRouting, ...]


class Attention(torch.nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        p_dropout: float,
        is_causal: bool = True,
        d_head: Optional[int] = None,
    ):
        """
        Base Attention parent class. Users should not be instantiating this class, but rather use SelfAttention or
        CrossAttention classes as appropriate.
        Does DotProductionAttention and additionally dropout inside the module. The class does not currently support
        RoPE nor ALiBi.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Dimension of the model.
            p_dropout (float): Dropout probability.
            is_causal (bool): Whether to use causal attention. Only supported when used in SelfAttention.
            d_head (int): Head dimension. Defaults to d_model // n_heads.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model % n_head != 0"
        self.d_head = d_head if d_head is not None else d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model
        self.scale = self.d_head**-0.5
        self.is_causal = is_causal
        self.o_net = torch.nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.dropout = torch.nn.Dropout(p_dropout)
        self.use_cache = False
        self.cache = self._init_cache()

    @abstractmethod
    def compute_qkv_and_mask(
        self,
        query: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ):
        pass

    @staticmethod
    def _init_cache() -> Dict[str, Optional[Union[bool, torch.Tensor]]]:
        return {
            'is_initialized': False,
            'self_k': None,
            'self_v': None,
            'cross_kv': None,
            'cross_k': None,
            'cross_v': None,
        }

    def reset_cache(self, use_cache: bool = False):
        self.use_cache = use_cache
        self.cache = self._init_cache()

    def attn_naive(
        self,
        query: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        attn_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.use_cache:
            if self.cache['is_initialized']:
                query = query[:, -1:, :]
                query_mask = query_mask[:, -1:] if query_mask is not None else None
            else:
                self.cache['is_initialized'] = True

        # Calls into children classes to compute qkv tensors and mask tensor
        q, k, v, mask = self.compute_qkv_and_mask(
            query=query, query_mask=query_mask, memory=memory, memory_mask=memory_mask
        )

        # (B, T, nh, dh) -> (B, nh, T, dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        B, T, _ = query.shape
        attn_score = torch.matmul(q, k.transpose(2, 3)) * self.scale
        if mask is not None:
            # assumes there's at least one mask
            attn_score.masked_fill_(mask == 0, float('-inf'))
        if self.is_causal:
            attn_score.masked_fill_(self.causal_mask[..., :T, :T] == 0, float('-inf'))

        # attn_prior or square mask or vanilla attention
        if attn_prior is not None:
            eps = torch.finfo(attn_prior.dtype).tiny
            attn_prior = attn_prior[:, :T]  # trim for inference
            attn_prior = attn_prior[:, None] + eps
            # Use PyTorch's built-in training flag to branch behavior
            if self.training:
                attn_prior_log = torch.log(attn_prior)
                attn_score_log = F.log_softmax(attn_score, dim=-1) + attn_prior_log
                if self.make_prior_window_strict:
                    # Make sure attention scores are lowest (eps) where prior is zero.
                    min_score = torch.log(torch.tensor(eps)).to(attn_score_log.device)
                    attn_score_log = attn_score_log.masked_fill(
                        attn_prior == 0, min_score
                    )  # Wherever prior is zero, set scores to eps.
                    attn_score_log = torch.clamp(
                        attn_score_log, min=min_score
                    )  # Make sure scores are not less than eps.
                attn_prob = F.softmax(attn_score_log, dim=-1)
            else:
                attn_prob = F.softmax(attn_score, dim=-1)
                attn_prob = attn_prob * attn_prior
                attn_prob = attn_prob / (attn_prob.sum(dim=-1, keepdim=True))  # normalize
        else:
            attn_prob = F.softmax(attn_score, dim=-1)

        if mask is not None:
            attn_prob = attn_prob.masked_fill(mask == 0, 0.0)
        attn_prob = self.dropout(attn_prob)

        y = torch.matmul(attn_prob, v)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        return y, [attn_prob, attn_score]

    def forward(
        self,
        query: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        attn_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the Attention module.

        Args:
            query (torch.Tensor): Input tensor of shape (B, T1, C).
            query_mask (Optional[torch.Tensor]): Mask for query tensor of shape (B, T1).
            memory (Optional[torch.Tensor]): Memory tensor for cross-attention of shape (B, T2, C).
            memory_mask (Optional[torch.Tensor]): Mask for memory tensor of shape (B, T2).
            attn_prior (Optional[torch.Tensor]): Prior attention weights of shape (B, T1, T2).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - y: Attention module tensor output of shape (B, T1, C).
                - attn_prob: List containing attention probabilities and scores. returned only in attn_naive.
                    [0]: Attention probabilities used for logging during validation.
                    [1]: Attention scores used for CTC loss (only in naive attention).
        """

        y, attn_prob = self.attn_naive(query, query_mask, memory, memory_mask, attn_prior)
        y = self.dropout(self.o_net(y))

        return y, attn_prob


class SelfAttention(Attention):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        p_dropout: float,
        is_causal: bool = True,
        max_length_causal_mask: int = 4096,
    ):
        """
        Implements SelfAttention. See parent class for forward implementation.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Dimension of the model.
            p_dropout (float): Dropout probability.
            is_causal (bool): Whether to use causal attention. Only supported when used in SelfAttention.
            max_length_causal_mask (int): Maximum sequence length for Attention module.
        """
        super().__init__(
            n_heads=n_heads,
            d_model=d_model,
            p_dropout=p_dropout,
            is_causal=is_causal,
        )
        if is_causal:
            if max_length_causal_mask is None or max_length_causal_mask < 0:
                raise ValueError(
                    "Self Attention was called with is_causal True, but received an inappropriate value"
                    f"of {max_length_causal_mask} for max_length_causal_mask"
                )
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(max_length_causal_mask, max_length_causal_mask)).view(
                    1, 1, max_length_causal_mask, max_length_causal_mask
                ),
            )
        self.qkv_net = torch.nn.Linear(d_model, 3 * n_heads * self.d_head, bias=False)

    def compute_qkv_and_mask(
        self,
        query: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ):
        B, T, _ = query.shape
        qkv = self.qkv_net(query).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.chunk(3, dim=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
        if self.use_cache:
            if self.cache['self_k'] is not None:
                k = torch.cat([self.cache['self_k'], k], dim=1)
                v = torch.cat([self.cache['self_v'], v], dim=1)
            self.cache['self_k'] = k
            self.cache['self_v'] = v

        mask = None
        if query_mask is not None:
            # query_mask is a boolean mask of shape (B, T)
            # mask should be of shape (B, 1, T, T) where mask[:,0,i,:] == mask[:,0,:,i] == query_mask
            mask = query_mask.unsqueeze(1) * query_mask.unsqueeze(2)
            mask = mask.unsqueeze(1)

        return q, k, v, mask

    def create_incremental_state(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SelfAttentionIncrementalState:
        if not self.is_causal:
            raise ValueError("Incremental self-attention requires causal attention")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        shape = (batch_size, max_length, self.n_heads, self.d_head)
        return SelfAttentionIncrementalState(
            key=torch.empty(shape, device=device, dtype=dtype),
            value=torch.empty(shape, device=device, dtype=dtype),
            key_mask=torch.empty((batch_size, max_length), device=device, dtype=torch.bool),
            max_length=max_length,
        )

    def prefill_incremental_state(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        *,
        state: SelfAttentionIncrementalState,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Run a complete prefix once and populate the session K/V buffers."""
        if position_offset != 0:
            raise ValueError("Vectorized self-attention prefill currently requires position_offset=0")
        batch_size, prefix_length, _ = query.shape
        if prefix_length <= 0 or prefix_length > state.max_length:
            raise ValueError(f"Prefix length {prefix_length} is outside the allocated length {state.max_length}")

        qkv = self.qkv_net(query).reshape(batch_size, prefix_length, 3, self.n_heads, self.d_head)
        q, key, value = qkv.unbind(dim=2)
        state.key[:, :prefix_length].copy_(key)
        state.value[:, :prefix_length].copy_(value)
        state.key_mask[:, :prefix_length].copy_(query_mask)

        q = q.transpose(1, 2)
        key_heads = key.transpose(1, 2)
        value_heads = value.transpose(1, 2)
        attn_score = torch.matmul(q, key_heads.transpose(2, 3)) * self.scale
        mask = query_mask[:, None, :, None] & query_mask[:, None, None, :]
        attn_score.masked_fill_(~mask, float('-inf'))
        attn_score.masked_fill_(
            self.causal_mask[..., :prefix_length, :prefix_length] == 0,
            float('-inf'),
        )
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = attn_prob.masked_fill(~mask, 0.0)
        attn_prob = self.dropout(attn_prob)

        output = torch.matmul(attn_prob, value_heads)
        output = output.transpose(1, 2).contiguous().view(batch_size, prefix_length, -1)
        output = self.dropout(self.o_net(output))
        return output

    def forward_incremental(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        *,
        state: SelfAttentionIncrementalState,
        position_offset: int,
    ) -> torch.Tensor:
        """Attend one query to its cached causal prefix."""
        if query.size(1) != 1 or query_mask.size(1) != 1:
            raise ValueError(
                f"Incremental self-attention expects one timestep, got query={tuple(query.shape)}, "
                f"mask={tuple(query_mask.shape)}"
            )
        if position_offset < 0 or position_offset >= state.max_length:
            raise ValueError(f"position_offset {position_offset} is outside the allocated length {state.max_length}")

        batch_size = query.size(0)
        qkv = self.qkv_net(query).reshape(batch_size, 1, 3, self.n_heads, self.d_head)
        q, key_step, value_step = qkv.unbind(dim=2)
        state.key[:, position_offset : position_offset + 1].copy_(key_step)
        state.value[:, position_offset : position_offset + 1].copy_(value_step)
        state.key_mask[:, position_offset : position_offset + 1].copy_(query_mask)

        q = q.transpose(1, 2)
        key = state.key[:, : position_offset + 1].transpose(1, 2)
        value = state.value[:, : position_offset + 1].transpose(1, 2)
        attn_score = torch.matmul(q, key.transpose(2, 3)) * self.scale
        mask = query_mask[:, None, :, None] & state.key_mask[:, None, None, : position_offset + 1]
        attn_score.masked_fill_(~mask, float('-inf'))
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = attn_prob.masked_fill(~mask, 0.0)
        attn_prob = self.dropout(attn_prob)

        output = torch.matmul(attn_prob, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        output = self.dropout(self.o_net(output))
        return output


class CrossAttention(Attention):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_memory: int,
        p_dropout: float,
        make_prior_window_strict: bool = False,
        d_head: Optional[int] = None,
    ):
        """
        Implements CrossAttention. See parent class for forward implementation. Must be non-causal.

        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Dimension of the model.
            d_memory (int): Dimension of the conditioning / cross-attention input.
            p_dropout (float): Dropout probability.
            make_prior_window_strict (bool): Make attention scores lowest where prior is zero.
            d_head (int): Head dimension. if None, defaults to d_model // n_heads in parent class.
        """
        super().__init__(
            n_heads=n_heads,
            d_model=d_model,
            p_dropout=p_dropout,
            is_causal=False,
            d_head=d_head,
        )
        self.q_net = torch.nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.kv_net = torch.nn.Linear(d_memory, 2 * n_heads * self.d_head, bias=False)
        self.make_prior_window_strict = make_prior_window_strict

    def compute_qkv_and_mask(
        self,
        query: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ):
        Bq, Tq, _ = query.shape
        Bkv, Tkv, _ = memory.shape
        q = self.q_net(query).reshape(Bq, Tq, self.n_heads, self.d_head)
        if self.use_cache and self.cache['cross_kv'] is not None:
            kv = self.cache['cross_kv']
        else:
            kv = self.kv_net(memory).reshape(Bkv, Tkv, 2, self.n_heads, self.d_head)

        if self.use_cache and self.cache['cross_k'] is not None:
            k = self.cache['cross_k']
            v = self.cache['cross_v']
        else:
            k, v = kv.chunk(2, dim=2)
            k, v = k.squeeze(2), v.squeeze(2)
            if self.use_cache:
                self.cache['cross_kv'] = kv
                self.cache['cross_k'] = k
                self.cache['cross_v'] = v

        mask = memory_mask[:, None, None] if memory_mask is not None else None
        return q, k, v, mask

    def forward_incremental(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        *,
        state: CrossAttentionIncrementalState,
        memory: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
        attn_prior: Optional[torch.Tensor],
        return_alignment_score: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Cross-attend one query using session-owned, context-stable K/V."""
        if query.size(1) != 1 or query_mask.size(1) != 1:
            raise ValueError(
                f"Incremental cross-attention expects one timestep, got query={tuple(query.shape)}, "
                f"mask={tuple(query_mask.shape)}"
            )
        batch_size = query.size(0)
        q = self.q_net(query).reshape(batch_size, 1, self.n_heads, self.d_head).transpose(1, 2)

        if state.key is None or state.value is None:
            if memory is None:
                raise RuntimeError("Cross-attention memory is required when initializing incremental state")
            memory_batch, memory_length, _ = memory.shape
            kv = self.kv_net(memory).reshape(memory_batch, memory_length, 2, self.n_heads, self.d_head)
            key, value = kv.unbind(dim=2)
            state.key = key
            state.value = value

        key = state.key.transpose(1, 2)
        value = state.value.transpose(1, 2)
        attn_score = torch.matmul(q, key.transpose(2, 3)) * self.scale
        mask = memory_mask[:, None, None] if memory_mask is not None else None
        if mask is not None:
            attn_score.masked_fill_(~mask, float('-inf'))

        if attn_prior is not None:
            if attn_prior.size(1) != 1:
                raise ValueError(
                    f"Incremental attention prior must contain one query timestep, got {tuple(attn_prior.shape)}"
                )
            epsilon = torch.finfo(attn_prior.dtype).tiny
            prior = attn_prior[:, None] + epsilon
            attn_prob = F.softmax(attn_score, dim=-1)
            attn_prob = attn_prob * prior
            attn_prob = attn_prob / attn_prob.sum(dim=-1, keepdim=True)
        else:
            attn_prob = F.softmax(attn_score, dim=-1)

        if mask is not None:
            attn_prob = attn_prob.masked_fill(~mask, 0.0)
        attn_prob = self.dropout(attn_prob)
        output = torch.matmul(attn_prob, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        output = self.dropout(self.o_net(output))
        alignment_score = attn_prob[:, :, -1, :].mean(dim=1) if return_alignment_score else None
        return output, alignment_score

    def prefill_incremental_state(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        *,
        state: CrossAttentionIncrementalState,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        attn_prior: Optional[torch.Tensor],
        return_alignment_score: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run vectorized prefix cross-attention and retain only context K/V."""
        batch_size, query_length, _ = query.shape
        memory_batch, memory_length, _ = memory.shape
        q = self.q_net(query).reshape(batch_size, query_length, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_net(memory).reshape(memory_batch, memory_length, 2, self.n_heads, self.d_head)
        key, value = kv.unbind(dim=2)
        state.key = key
        state.value = value

        key_heads = key.transpose(1, 2)
        value_heads = value.transpose(1, 2)
        attn_score = torch.matmul(q, key_heads.transpose(2, 3)) * self.scale
        mask = memory_mask[:, None, None] if memory_mask is not None else None
        if mask is not None:
            attn_score.masked_fill_(~mask, float('-inf'))

        if attn_prior is not None:
            epsilon = torch.finfo(attn_prior.dtype).tiny
            prior = attn_prior[:, None] + epsilon
            attn_prob = F.softmax(attn_score, dim=-1)
            attn_prob = attn_prob * prior
            attn_prob = attn_prob / attn_prob.sum(dim=-1, keepdim=True)
        else:
            attn_prob = F.softmax(attn_score, dim=-1)

        if mask is not None:
            attn_prob = attn_prob.masked_fill(~mask, 0.0)
        attn_prob = self.dropout(attn_prob)
        output = torch.matmul(attn_prob, value_heads)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_length, -1)
        output = self.dropout(self.o_net(output))
        alignment_score = attn_prob[:, :, -1, :].mean(dim=1) if return_alignment_score else None
        return output, alignment_score


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        sa_n_heads: int,
        kernel_size: int,
        p_dropout: float,
        has_xattn: bool,
        xa_d_memory: Optional[int] = None,
        xa_n_heads: Optional[int] = None,
        xa_d_head: Optional[int] = None,
        is_causal: bool = True,
        apply_norm_to_cond: bool = True,
        max_length_causal_mask: int = 4096,
        conv_non_linearity: Callable = torch.nn.GELU(approximate="tanh"),
        make_prior_window_strict: bool = False,
        # MoE parameters
        use_moe: bool = False,
        num_experts: int = 8,
        top_k_experts: int = 2,
        router_jitter_noise: float = 0.0,
        routing_strategy: str = "top_k",
    ):
        """
        One layer of the Transformer.
        Args:
            d_model <int>: Model dimension
            d_ffn <int>: Feed forward dimension (usually 4*d_model)
            sa_n_heads <int>: Number of attention heads used in self-attention
            kernel_size <int>: Convolution kernel size for FFN
            p_dropout <float>: Dropout probability
            has_xattn <bool>: Whether to use cross attention
            xa_d_memory <int>: Hidden dimension for cross attention
            xa_n_heads <int>: Number of attention heads used in cross attention
            xa_d_head <int>: Head dimension for cross attention. if None, defaults to d_model // xa_n_heads in Attention class.
            is_causal <bool>: Whether to use causal attention
            apply_norm_to_cond <bool>: Whether to apply normalization to conditioning tensor
            max_length_causal_mask <int>: Maximum length of causal mask
            conv_non_linearity <Callable>: Convolution non-linearity
            make_prior_window_strict <bool>: Make attention scores lowest where prior is zero.
            use_moe <bool>: Whether to use Mixture of Experts for FFN
            num_experts <int>: Number of experts in MoE
            top_k_experts <int>: Number of experts to use per token
            router_jitter_noise <float>: Noise for router exploration
            routing_strategy <str>: Routing strategy ("top_k" or "sinkhorn")
        """
        super().__init__()
        self.has_xattn = has_xattn
        self.use_moe = use_moe

        # TODO @xueyang: maybe we can replace LayerNorm with RMSNorm here for training efficiency?
        self.norm_self = torch.nn.LayerNorm(d_model, bias=False)
        self.self_attention = SelfAttention(
            n_heads=sa_n_heads,
            d_model=d_model,
            p_dropout=p_dropout,
            max_length_causal_mask=max_length_causal_mask,
            is_causal=is_causal,
        )

        if self.has_xattn:
            self.norm_xattn_query = torch.nn.LayerNorm(d_model, bias=False)
            self.cross_attention = CrossAttention(
                n_heads=xa_n_heads,
                d_model=d_model,
                d_memory=xa_d_memory,
                p_dropout=p_dropout,
                make_prior_window_strict=make_prior_window_strict,
                d_head=xa_d_head,
            )

            self.norm_xattn_memory = torch.nn.Identity()
            if apply_norm_to_cond:
                self.norm_xattn_memory = torch.nn.LayerNorm(xa_d_memory, bias=False)

        self.norm_pos_ff = torch.nn.LayerNorm(d_model, bias=False)

        # Use MoE or standard FFN based on configuration
        if use_moe:
            self.pos_ff = PositionwiseConvFFMoE(
                d_model=d_model,
                d_ffn=d_ffn,
                p_dropout=p_dropout,
                num_experts=num_experts,
                top_k_experts=top_k_experts,
                kernel_size=kernel_size,
                is_causal=is_causal,
                non_linearity=conv_non_linearity,
                router_jitter_noise=router_jitter_noise,
                routing_strategy=routing_strategy,
            )
        else:
            self.pos_ff = PositionwiseConvFF(
                d_model,
                d_ffn,
                p_dropout,
                kernel_size=kernel_size,
                is_causal=is_causal,
                non_linearity=conv_non_linearity,
            )

        self.use_cache = False
        self.cache = self._init_cache()

    @staticmethod
    def _init_cache() -> Dict:
        return {
            'self_attn_output': None,
            'cross_attn_output': None,
            'memory': None,
        }

    def reset_cache(self, use_cache=False):
        self.use_cache = use_cache
        self.cache = self._init_cache()
        self.self_attention.reset_cache(use_cache)
        if self.has_xattn:
            self.cross_attention.reset_cache(use_cache)

    def create_incremental_state(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TransformerLayerIncrementalState:
        if self.use_moe:
            if not isinstance(self.pos_ff, PositionwiseConvFFMoE):
                raise TypeError(f"Expected a MoE feed-forward module, got {type(self.pos_ff).__name__}")
            feed_forward_state = MoEFeedForwardIncrementalState(max_length=max_length)
        else:
            if not isinstance(self.pos_ff, PositionwiseConvFF):
                raise TypeError(f"Expected a dense feed-forward module, got {type(self.pos_ff).__name__}")
            feed_forward_state = self.pos_ff.create_incremental_state(
                batch_size=batch_size,
                max_length=max_length,
                device=device,
                dtype=dtype,
            )
        return TransformerLayerIncrementalState(
            self_attention=self.self_attention.create_incremental_state(
                batch_size=batch_size,
                max_length=max_length,
                device=device,
                dtype=dtype,
            ),
            cross_attention=CrossAttentionIncrementalState() if self.has_xattn else None,
            feed_forward=feed_forward_state,
        )

    def forward_incremental(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        *,
        state: TransformerLayerIncrementalState,
        position_offset: int,
        cond: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        attn_prior: Optional[torch.Tensor] = None,
        return_alignment_score: bool,
    ) -> TransformerLayerInferenceOutput:
        """Run exactly one timestep with state supplied by the inference session."""
        if x.size(1) != 1 or x_mask.size(1) != 1:
            raise ValueError(
                f"Incremental Transformer layer expects one timestep, got x={tuple(x.shape)}, "
                f"mask={tuple(x_mask.shape)}"
            )
        if return_alignment_score and (not self.has_xattn or cond is None):
            raise ValueError("Alignment was requested from a layer without active cross-attention")
        bool_mask = x_mask.bool()
        x = x * bool_mask.unsqueeze(-1)
        self_output = self.self_attention.forward_incremental(
            query=self.norm_self(x),
            query_mask=bool_mask,
            state=state.self_attention,
            position_offset=position_offset,
        )
        x = x + self_output

        alignment_score = None
        if self.has_xattn and cond is not None:
            if state.cross_attention is None:
                raise RuntimeError("Cross-attention incremental state was not allocated")
            context_signature = TensorSignature.from_tensor(cond)
            mask_signature = TensorSignature.from_tensor(cond_mask) if cond_mask is not None else None
            if state.cross_attention.context_signature is None:
                state.cross_attention.context_signature = context_signature
                state.cross_attention.mask_signature = mask_signature
            elif (
                state.cross_attention.context_signature != context_signature
                or state.cross_attention.mask_signature != mask_signature
            ):
                raise RuntimeError(
                    "Conditioning changed inside an incremental session. Create a new TransformerIncrementalState "
                    "instead of reusing stale cross-attention K/V."
                )

            normalized_memory = self.norm_xattn_memory(cond) if state.cross_attention.key is None else None
            cross_output, alignment_score = self.cross_attention.forward_incremental(
                query=self.norm_xattn_query(x),
                query_mask=bool_mask,
                state=state.cross_attention,
                memory=normalized_memory,
                memory_mask=cond_mask.bool() if cond_mask is not None else None,
                attn_prior=attn_prior,
                return_alignment_score=return_alignment_score,
            )
            x = x + cross_output

        moe_routing = None
        normalized_x = self.norm_pos_ff(x)
        if self.use_moe:
            if not isinstance(self.pos_ff, PositionwiseConvFFMoE):
                raise TypeError(f"Expected a MoE feed-forward module, got {type(self.pos_ff).__name__}")
            if not isinstance(state.feed_forward, MoEFeedForwardIncrementalState):
                raise TypeError("MoE inference requires MoEFeedForwardIncrementalState")
            ffn_output, router_logits, router_probs, expert_indices = self.pos_ff(normalized_x, bool_mask)
            moe_routing = MoEInferenceRouting(
                router_logits=router_logits,
                router_probs=router_probs,
                expert_indices=expert_indices,
            )
        else:
            if not isinstance(self.pos_ff, PositionwiseConvFF):
                raise TypeError(f"Expected a dense feed-forward module, got {type(self.pos_ff).__name__}")
            if not isinstance(state.feed_forward, PositionwiseConvFFIncrementalState):
                raise TypeError("Dense inference requires PositionwiseConvFFIncrementalState")
            ffn_output = self.pos_ff.forward_incremental(
                normalized_x,
                bool_mask,
                state=state.feed_forward,
                position_offset=position_offset,
            )
        x = x + ffn_output
        x = x * bool_mask.unsqueeze(-1)
        return TransformerLayerInferenceOutput(
            output=x,
            alignment_score=alignment_score,
            moe_routing=moe_routing,
        )

    def prefill_incremental_state(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        *,
        state: TransformerLayerIncrementalState,
        cond: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        attn_prior: Optional[torch.Tensor] = None,
        return_alignment_score: bool,
    ) -> TransformerLayerInferenceOutput:
        """Run a vectorized prefix and initialize all layer session state."""
        if return_alignment_score and (not self.has_xattn or cond is None):
            raise ValueError("Alignment was requested from a layer without active cross-attention")
        bool_mask = x_mask.bool()
        x = x * bool_mask.unsqueeze(-1)
        self_output = self.self_attention.prefill_incremental_state(
            query=self.norm_self(x),
            query_mask=bool_mask,
            state=state.self_attention,
        )
        x = x + self_output

        alignment_score = None
        if self.has_xattn and cond is not None:
            if state.cross_attention is None:
                raise RuntimeError("Cross-attention incremental state was not allocated")
            state.cross_attention.context_signature = TensorSignature.from_tensor(cond)
            state.cross_attention.mask_signature = (
                TensorSignature.from_tensor(cond_mask) if cond_mask is not None else None
            )
            cross_output, alignment_score = self.cross_attention.prefill_incremental_state(
                query=self.norm_xattn_query(x),
                query_mask=bool_mask,
                state=state.cross_attention,
                memory=self.norm_xattn_memory(cond),
                memory_mask=cond_mask.bool() if cond_mask is not None else None,
                attn_prior=attn_prior,
                return_alignment_score=return_alignment_score,
            )
            x = x + cross_output

        moe_routing = None
        normalized_x = self.norm_pos_ff(x)
        if self.use_moe:
            if not isinstance(self.pos_ff, PositionwiseConvFFMoE):
                raise TypeError(f"Expected a MoE feed-forward module, got {type(self.pos_ff).__name__}")
            if not isinstance(state.feed_forward, MoEFeedForwardIncrementalState):
                raise TypeError("MoE inference requires MoEFeedForwardIncrementalState")
            ffn_output, router_logits, router_probs, expert_indices = self.pos_ff(normalized_x, bool_mask)
            moe_routing = MoEInferenceRouting(
                router_logits=router_logits,
                router_probs=router_probs,
                expert_indices=expert_indices,
            )
        else:
            if not isinstance(self.pos_ff, PositionwiseConvFF):
                raise TypeError(f"Expected a dense feed-forward module, got {type(self.pos_ff).__name__}")
            if not isinstance(state.feed_forward, PositionwiseConvFFIncrementalState):
                raise TypeError("Dense inference requires PositionwiseConvFFIncrementalState")
            ffn_output = self.pos_ff.prefill_incremental_state(
                normalized_x,
                bool_mask,
                state=state.feed_forward,
            )
        x = x + ffn_output
        x = x * bool_mask.unsqueeze(-1)
        return TransformerLayerInferenceOutput(
            output=x,
            alignment_score=alignment_score,
            moe_routing=moe_routing,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        attn_prior: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Args:
            x <torch tensor> (B, T1, C): Input tensor
            x_mask <bool mask> (B, T1): Multiplicative mask where True means we keep the input, False we zero it out.
                Mask for self attention input.
            cond <torch tensor> (B, T2, C): Conditioning tensor
            cond_mask <bool mask> (B, T2): Multiplicative mask where True means we keep the input, False we zero
                it out. Mask for cross attention input if it exists.

        Returns dict with keys
            output <torch tensor> (B, T1, C): Output tensor
            attn_probabilities <dict>: Attention probabilities with keys
                'self_attn_probabilities': Self-attention probabilities
                'cross_attn_probabilities': Cross-attention probabilities (None if no cross-attention)
            moe_routing_info <dict or None>: MoE routing information (None if MoE is disabled).
                If MoE is enabled, contains:
                    'router_logits' <torch tensor> (B, T, num_experts): Raw router logits for z-loss
                    'router_probs' <torch tensor> (B, T, num_experts): Router probabilities for load balancing loss
                    'expert_indices' <torch tensor> (B, T, top_k): Selected expert indices for usage statistics
        """
        x = x * x_mask.unsqueeze(-1)
        x_, s_attn_prob = self.self_attention(query=self.norm_self(x), query_mask=x_mask)
        if self.use_cache:
            if self.cache['self_attn_output'] is not None:
                x_ = torch.cat([self.cache['self_attn_output'], x_], dim=1)
            self.cache['self_attn_output'] = x_
        x = x + x_

        x_attn_prob = None
        if self.has_xattn and cond is not None:
            x_normed = self.norm_xattn_query(x)
            if self.use_cache and self.cache['memory'] is not None:
                memory = self.cache['memory']
            else:
                memory = self.norm_xattn_memory(cond)
                if self.use_cache:
                    self.cache['memory'] = memory

            x_res, x_attn_prob = self.cross_attention(
                query=x_normed, query_mask=x_mask, memory=memory, memory_mask=cond_mask, attn_prior=attn_prior
            )
            if self.use_cache:
                if self.cache['cross_attn_output'] is not None:
                    x_res = torch.cat([self.cache['cross_attn_output'], x_res], dim=1)
                self.cache['cross_attn_output'] = x_res
            x = x + x_res

        # mlp final projection
        moe_routing_info = None
        if self.use_moe:
            ffn_out, router_logits, router_probs, expert_indices = self.pos_ff(self.norm_pos_ff(x), x_mask)
            x = x + ffn_out
            # Store routing information for loss computation and statistics in the model
            moe_routing_info = {
                'router_logits': router_logits,
                'router_probs': router_probs,
                'expert_indices': expert_indices,
            }
        else:
            x = x + self.pos_ff(self.norm_pos_ff(x), x_mask)
        x = x * x_mask.unsqueeze(-1)

        return {
            'output': x,
            'attn_probabilities': {'self_attn_probabilities': s_attn_prob, 'cross_attn_probabilities': x_attn_prob},
            'moe_routing_info': moe_routing_info,
        }


class Transformer(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        sa_n_heads: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        p_dropout_out: float = 0.0,
        has_xattn: bool = False,
        xa_d_memory: Optional[int] = None,
        xa_n_heads: Optional[int] = None,
        xa_d_head: Optional[int] = None,
        is_causal: bool = True,
        apply_norm_to_cond: bool = True,
        apply_norm_out: bool = False,
        max_length_causal_mask: int = 4096,
        use_learnable_pos_emb: bool = False,
        conv_non_linearity: Callable = torch.nn.GELU(approximate="tanh"),
        make_prior_window_strict: bool = False,
        # MoE parameters
        use_moe: bool = False,
        num_experts: int = 8,
        top_k_experts: int = 2,
        router_jitter_noise: float = 0.0,
        routing_strategy: str = "top_k",
    ):
        """
        Initializes a stack of transformer layers. Can be used for both encoder and decoder.
        Set is_causal is True for autoregressive models. Equivalent to TransformerBlock from Megatron-LM
        Args:
            n_layers <int>: Number of transformer layers
            d_model <int>: Model dimension
            d_ffn <int>: Feed forward dimension (usually 4*d_model)
            sa_n_heads <int>: Number of attention heads used in self-attention
            kernel_size <int>: Convolution kernel size for FFN
            p_dropout <float>: Dropout probability
            p_dropout_out <float>: Dropout probability for output
            has_xattn <bool>: Whether to use cross attention
            xa_d_memory <int>: Hidden dimension for cross attention; required if has_xattn is True
            xa_n_heads <int>: Number of attention heads used in cross attention; required if has_xattn is True
            xa_d_head <int>: Head dimension for cross attention. if None, defaults to d_model // xa_n_heads in Attention class.
            is_causal <bool>: Whether to make attention and the convolution feedforward networks causal.
            apply_norm_to_cond <bool>: Whether to apply normalization to conditioning tensor; conditioning tensor being
                the input to the memory part of cross-attention.
            apply_norm_out <bool>: Whether to apply normalization to output
            max_length_causal_mask <int>: Maximum length of causal mask
            use_learnable_pos_emb <bool>: Whether to add a learnable positionable embedding inside the class
            conv_non_linearity <Callable>: Convolution non-linearity
            make_prior_window_strict <bool>: Make attention scores lowest where prior is zero
            use_moe <bool>: Whether to use Mixture of Experts for FFN layers
            num_experts <int>: Number of experts in MoE
            top_k_experts <int>: Number of experts to use per token
            router_jitter_noise <float>: Noise for router exploration during training
            routing_strategy <str>: Routing strategy ("top_k" or "sinkhorn")
        """
        if has_xattn and (xa_d_memory is None or xa_n_heads is None):
            raise ValueError("It requires that `xa_d_memory` and `xa_n_heads` are specified when `has_xattn` is True!")

        super().__init__()
        self.n_layers = n_layers
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.dropout = torch.nn.Dropout(p_dropout)
        self.p_dropout_out = p_dropout_out

        self.dropout_out = torch.nn.Identity()
        if self.p_dropout_out > 0.0:
            self.dropout_out = torch.nn.Dropout(self.p_dropout_out)

        self.norm_out = torch.nn.Identity()
        if apply_norm_out:
            self.norm_out = torch.nn.LayerNorm(d_model, bias=False)

        self.layers = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerLayer(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    sa_n_heads=sa_n_heads,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                    has_xattn=has_xattn,
                    xa_d_memory=xa_d_memory,
                    xa_n_heads=xa_n_heads,
                    xa_d_head=xa_d_head,
                    is_causal=is_causal,
                    apply_norm_to_cond=apply_norm_to_cond,
                    max_length_causal_mask=max_length_causal_mask,
                    conv_non_linearity=conv_non_linearity,
                    make_prior_window_strict=make_prior_window_strict,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k_experts=top_k_experts,
                    router_jitter_noise=router_jitter_noise,
                    routing_strategy=routing_strategy,
                )
            )

        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.position_embeddings = None
        if self.use_learnable_pos_emb:
            self.position_embeddings = torch.nn.Embedding(max_length_causal_mask, d_model)
        # Apply random uniform init for all layers, except for output layers: The second of the two layers in the MLP
        # and the last linear projection in dot product attention. The output layers are scaled depending on the
        # number of layers
        self.apply(self._init_weights_gpt2)
        for name, param in self.named_parameters():
            if 'o_net' in name and name.endswith('weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

    def reset_cache(self, use_cache=False):
        for layer in self.layers:
            layer.reset_cache(use_cache)

    def create_incremental_state(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TransformerIncrementalState:
        """Preallocate an independent inference session.

        No state is stored on the module, so sessions can be interleaved and
        discarded when text conditioning changes.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        return TransformerIncrementalState(
            layers=[
                layer.create_incremental_state(
                    batch_size=batch_size,
                    max_length=max_length,
                    device=device,
                    dtype=dtype,
                )
                for layer in self.layers
            ],
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            dtype=dtype,
        )

    def forward_incremental(
        self,
        x_step: torch.Tensor,
        x_mask: torch.Tensor,
        *,
        state: TransformerIncrementalState,
        position_offset: int,
        cond: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        cond_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        attn_prior: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        multi_encoder_mapping: Optional[List[Optional[int]]] = None,
        alignment_layer_indices: Optional[Tuple[int, ...]] = None,
    ) -> TransformerInferenceOutput:
        """Run one causal Transformer timestep without prefix recomputation."""
        if self.training:
            raise RuntimeError("Incremental Transformer inference requires eval() mode")
        if x_step.size(0) != state.batch_size:
            raise ValueError(f"State batch size is {state.batch_size}, but input batch is {x_step.size(0)}")
        if x_step.device != state.device or x_step.dtype != state.dtype:
            raise ValueError(
                f"Input device/dtype ({x_step.device}, {x_step.dtype}) does not match state "
                f"({state.device}, {state.dtype})"
            )
        if x_step.size(1) != 1 or x_mask.size(1) != 1:
            raise ValueError(
                f"Incremental Transformer expects one timestep, got x={tuple(x_step.shape)}, "
                f"mask={tuple(x_mask.shape)}"
            )
        if position_offset < 0 or position_offset >= state.max_length:
            raise ValueError(f"position_offset {position_offset} is outside the allocated length {state.max_length}")
        if isinstance(cond, list) and len(self.layers) < len(cond):
            raise ValueError(
                f"Insufficient Transformer layers for {len(cond)} conditionals: {len(self.layers)} layers"
            )
        self._validate_alignment_layer_indices(alignment_layer_indices)

        x = x_step
        if self.use_learnable_pos_emb:
            x = x + self.position_embeddings.weight[position_offset : position_offset + 1].unsqueeze(0)

        x = self.dropout(x)
        selected_alignment_scores: List[torch.Tensor] = []
        moe_routing: List[MoEInferenceRouting] = []
        for layer_index, (layer, layer_state) in enumerate(zip(self.layers, state.layers)):
            layer_cond, layer_cond_mask, layer_attn_prior = self._get_layer_inputs(
                layer_index,
                cond,
                cond_mask,
                attn_prior,
                multi_encoder_mapping,
            )
            output = layer.forward_incremental(
                x,
                x_mask,
                state=layer_state,
                position_offset=position_offset,
                cond=layer_cond,
                cond_mask=layer_cond_mask,
                attn_prior=layer_attn_prior,
                return_alignment_score=(
                    alignment_layer_indices is not None and layer_index in alignment_layer_indices
                ),
            )
            x = output.output
            if output.alignment_score is not None:
                selected_alignment_scores.append(output.alignment_score)
            if output.moe_routing is not None:
                moe_routing.append(output.moe_routing)

        x = self.norm_out(x)
        x = self.dropout_out(x)
        alignment_scores = self._mean_selected_alignment_scores(
            selected_alignment_scores,
            alignment_layer_indices,
        )
        return TransformerInferenceOutput(
            output=x,
            alignment_scores=alignment_scores,
            moe_routing=tuple(moe_routing),
        )

    def prefill_incremental_state(
        self,
        x_prefix: torch.Tensor,
        x_mask: torch.Tensor,
        *,
        state: TransformerIncrementalState,
        cond: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        cond_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        attn_prior: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        multi_encoder_mapping: Optional[List[Optional[int]]] = None,
        alignment_layer_indices: Optional[Tuple[int, ...]] = None,
    ) -> TransformerInferenceOutput:
        """Vectorize the fixed prefix and populate external K/V in one pass."""
        if self.training:
            raise RuntimeError("Incremental Transformer prefill requires eval() mode")
        if x_prefix.size(0) != state.batch_size:
            raise ValueError(f"State batch size is {state.batch_size}, but input batch is {x_prefix.size(0)}")
        if x_prefix.device != state.device or x_prefix.dtype != state.dtype:
            raise ValueError(
                f"Input device/dtype ({x_prefix.device}, {x_prefix.dtype}) does not match state "
                f"({state.device}, {state.dtype})"
            )
        prefix_length = x_prefix.size(1)
        if prefix_length <= 0 or prefix_length > state.max_length:
            raise ValueError(f"Prefix length {prefix_length} is outside the allocated length {state.max_length}")
        if isinstance(cond, list) and len(self.layers) < len(cond):
            raise ValueError(
                f"Insufficient Transformer layers for {len(cond)} conditionals: {len(self.layers)} layers"
            )
        self._validate_alignment_layer_indices(alignment_layer_indices)

        x = x_prefix
        if self.use_learnable_pos_emb:
            x = x + self.position_embeddings.weight[:prefix_length].unsqueeze(0)
        x = self.dropout(x)

        selected_alignment_scores: List[torch.Tensor] = []
        moe_routing: List[MoEInferenceRouting] = []
        for layer_index, (layer, layer_state) in enumerate(zip(self.layers, state.layers)):
            layer_cond, layer_cond_mask, layer_attn_prior = self._get_layer_inputs(
                layer_index,
                cond,
                cond_mask,
                attn_prior,
                multi_encoder_mapping,
            )
            output = layer.prefill_incremental_state(
                x,
                x_mask,
                state=layer_state,
                cond=layer_cond,
                cond_mask=layer_cond_mask,
                attn_prior=layer_attn_prior,
                return_alignment_score=(
                    alignment_layer_indices is not None and layer_index in alignment_layer_indices
                ),
            )
            x = output.output
            if output.alignment_score is not None:
                selected_alignment_scores.append(output.alignment_score)
            if output.moe_routing is not None:
                moe_routing.append(output.moe_routing)

        x = self.norm_out(x)
        x = self.dropout_out(x)
        alignment_scores = self._mean_selected_alignment_scores(
            selected_alignment_scores,
            alignment_layer_indices,
        )
        return TransformerInferenceOutput(
            output=x,
            alignment_scores=alignment_scores,
            moe_routing=tuple(moe_routing),
        )

    def _validate_alignment_layer_indices(self, alignment_layer_indices: Optional[Tuple[int, ...]]) -> None:
        if alignment_layer_indices is None:
            return
        if not isinstance(alignment_layer_indices, tuple):
            raise TypeError("alignment_layer_indices must be a tuple of layer indices")
        if not alignment_layer_indices:
            raise ValueError("alignment_layer_indices must not be empty")
        if len(set(alignment_layer_indices)) != len(alignment_layer_indices):
            raise ValueError(f"alignment_layer_indices contains duplicates: {alignment_layer_indices}")
        for layer_index in alignment_layer_indices:
            if type(layer_index) is not int:
                raise TypeError(f"Alignment layer index must be an int, got {type(layer_index).__name__}")
            if layer_index < 0 or layer_index >= self.n_layers:
                raise ValueError(
                    f"Alignment layer index {layer_index} is outside the decoder layer range [0, {self.n_layers})"
                )
            if not self.layers[layer_index].has_xattn:
                raise ValueError(f"Alignment layer {layer_index} does not have cross-attention")

    @staticmethod
    def _mean_selected_alignment_scores(
        selected_alignment_scores: List[torch.Tensor],
        alignment_layer_indices: Optional[Tuple[int, ...]],
    ) -> Optional[torch.Tensor]:
        if alignment_layer_indices is None:
            if selected_alignment_scores:
                raise RuntimeError("Alignment scores were produced without an explicit layer request")
            return None
        if len(selected_alignment_scores) != len(alignment_layer_indices):
            raise RuntimeError(
                f"Requested {len(alignment_layer_indices)} alignment layers, "
                f"but received {len(selected_alignment_scores)} scores"
            )
        return torch.stack(selected_alignment_scores, dim=1).mean(dim=1)

    @staticmethod
    def _init_weights_gpt2(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    @staticmethod
    def _get_layer_inputs(
        idx: int,
        cond: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        cond_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        attn_prior: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        multi_encoder_mapping: Optional[List[Optional[int]]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if multi_encoder_mapping is not None:
            if multi_encoder_mapping[idx] is None:
                return None, None, None
            else:
                _attn_prior = attn_prior[multi_encoder_mapping[idx]] if attn_prior is not None else None
                if isinstance(_attn_prior, list):
                    # @pneekhara: This means, we are passing layerwise attn_prior
                    _attn_prior = _attn_prior[idx]
                return (
                    cond[multi_encoder_mapping[idx]],
                    cond_mask[multi_encoder_mapping[idx]] if cond_mask is not None else None,
                    _attn_prior,
                )
        else:
            if isinstance(attn_prior, list):
                # @pneekhara: This means, we are passing layerwise attn_prior
                _attn_prior = attn_prior[idx]
            else:
                _attn_prior = attn_prior
            return cond, cond_mask, _attn_prior

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cond: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        cond_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        attn_prior: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        multi_encoder_mapping: Optional[List[Optional[int]]] = None,
        max_layer_idx: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Args:
            x <torch tensor> (B, T1, C):
            x_mask <bool mask> (B, T1): Multiplicative mask where True means we keep the input, False we zero it out.
                Mostly used in non-causal self-attention to zero out padding values. In causal self-attention, the
                causal mask will be used in place of this.
            cond <torch tensor> (B, T2, C) or list of such tensors (from different encoders)
            cond_mask <bool mask> (B, T2): Multiplicative mask where True means we keep the input, False we zero it
                out or list of such tensors (from different encoders) output <torch tensor> (B, T1, C)
            multi_encoder_mapping <list> <int>: None or Same size as n_layers, value indicates which cond input to use
                for this layer

        Returns dict with keys:
            output <torch tensor> (B, T1, C): Output tensor
            attn_probabilities <list>: Attention probabilities of each layer
            moe_routing_info <list or None>: List of MoE routing info dicts from each layer (None if MoE disabled).
                Each dict contains 'router_logits', 'router_probs', and 'expert_indices' for
                loss computation and usage statistics in the model.
        """
        if isinstance(cond, list) and len(self.layers) < len(cond):
            raise ValueError(
                f"Insufficient Transformer layers for multiple conditionals. Each layer must cross-attend one conditional."
                f"Found {len(self.layers)} layers for {len(cond)} conditionals."
            )

        if self.use_learnable_pos_emb:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            x = x + self.position_embeddings(positions)

        attn_probabilities = []
        # Collect MoE routing information from all layers
        moe_routing_info_all_layers = []
        x = self.dropout(x)
        for idx, layer in enumerate(self.layers):
            _cond, _cond_mask, _attn_prior = self._get_layer_inputs(
                idx, cond, cond_mask, attn_prior, multi_encoder_mapping
            )
            out_dict = layer(x, x_mask, _cond, _cond_mask, attn_prior=_attn_prior)
            x = out_dict['output']
            attn_probabilities.append(out_dict['attn_probabilities'])

            # Collect MoE routing info for loss computation in the model
            if self.use_moe and out_dict['moe_routing_info'] is not None:
                moe_routing_info_all_layers.append(out_dict['moe_routing_info'])

            if max_layer_idx is not None and idx == max_layer_idx:
                break

        x = self.norm_out(x)
        x = self.dropout_out(x)
        return {
            'output': x,
            'attn_probabilities': attn_probabilities,
            'moe_routing_info': moe_routing_info_all_layers if len(moe_routing_info_all_layers) > 0 else None,
        }
