# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Dedicated Triton sampling kernels for MagpieTTS local autoregression.

The local-AR output projection remains a GEMM.  This module combines every
operation after that GEMM -- CFG, status/special-token constraints, top-k
sampling, and the next-token embedding lookup -- into one CUDA kernel.

The API is deliberately in-place.  Callers provide stable output buffers so
the operation can be captured in the fixed-shape local-AR CUDA Graph without
allocations or host synchronization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor


_NUM_MAGPIE_SPECIAL_TOKENS = 8


@triton.jit
def _load_constrained_logits(
    logits_ptr,
    unfinished_ptr,
    finished_ptr,
    forbid_eos_ptr,
    row,
    offsets,
    cfg_scale,
    ACTUAL_BATCH_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    CODEBOOK_SIZE: tl.constexpr,
    AUDIO_EOS_ID: tl.constexpr,
    USE_CFG: tl.constexpr,
):
    in_bounds = offsets < VOCAB_SIZE
    conditional = tl.load(logits_ptr + row * VOCAB_SIZE + offsets, mask=in_bounds, other=-float("inf"))

    if USE_CFG:
        unconditional = tl.load(
            logits_ptr + (row + ACTUAL_BATCH_SIZE) * VOCAB_SIZE + offsets,
            mask=in_bounds,
            other=-float("inf"),
        )
        # Match the two in-place BF16/FP16 operations used by the PyTorch
        # reference: mul_ rounds once, then add_(alpha=...) rounds again.
        guided = (conditional * cfg_scale).to(logits_ptr.dtype.element_ty)
        guided = (guided + unconditional * (1.0 - cfg_scale)).to(logits_ptr.dtype.element_ty)
        values = guided.to(tl.float32)
    else:
        values = conditional.to(tl.float32)

    invalid_input = in_bounds & ((values != values) | (values == float("inf")) | (values == -float("inf")))
    # Keep the kernel numerically defined until the asynchronous assertion
    # fails the request. Invalid model output must never be converted into a
    # plausible token.
    values = tl.where(invalid_input, -float("inf"), values)

    unfinished = tl.load(unfinished_ptr + row) != 0
    finished = tl.load(finished_ptr + row) != 0
    forbid_eos = tl.load(forbid_eos_ptr + row) != 0

    values = tl.where(finished, tl.where(offsets == AUDIO_EOS_ID, 0.0, -float("inf")), values)
    forbidden_special = (offsets >= CODEBOOK_SIZE) & (offsets != AUDIO_EOS_ID)
    values = tl.where(forbidden_special, -float("inf"), values)
    values = tl.where((unfinished | forbid_eos) & (offsets == AUDIO_EOS_ID), -float("inf"), values)
    values = tl.where(in_bounds, values, -float("inf"))
    return values, invalid_input


@triton.jit
def _store_token_and_embedding(
    sampled_id,
    token_output_ptr,
    embedding_weight_ptr,
    embedding_output_ptr,
    row,
    token_output_stride,
    embedding_weight_stride_0,
    embedding_weight_stride_1,
    embedding_output_stride_0,
    embedding_output_stride_1,
    ACTUAL_BATCH_SIZE: tl.constexpr,
    EMBEDDING_DIM: tl.constexpr,
    BLOCK_EMBEDDING: tl.constexpr,
    USE_CFG: tl.constexpr,
    WRITE_EMBEDDING: tl.constexpr,
):
    tl.store(token_output_ptr + row * token_output_stride, sampled_id)

    if WRITE_EMBEDDING:
        embedding_offsets = tl.arange(0, BLOCK_EMBEDDING)
        embedding_mask = embedding_offsets < EMBEDDING_DIM
        embedding = tl.load(
            embedding_weight_ptr
            + sampled_id * embedding_weight_stride_0
            + embedding_offsets * embedding_weight_stride_1,
            mask=embedding_mask,
        )
        tl.store(
            embedding_output_ptr + row * embedding_output_stride_0 + embedding_offsets * embedding_output_stride_1,
            embedding,
            mask=embedding_mask,
        )
        if USE_CFG:
            tl.store(
                embedding_output_ptr
                + (row + ACTUAL_BATCH_SIZE) * embedding_output_stride_0
                + embedding_offsets * embedding_output_stride_1,
                embedding,
                mask=embedding_mask,
            )


@triton.jit
def _fused_argmax_embedding_kernel(
    logits_ptr,
    unfinished_ptr,
    finished_ptr,
    forbid_eos_ptr,
    invalid_rows_ptr,
    token_output_ptr,
    embedding_weight_ptr,
    embedding_output_ptr,
    token_output_stride,
    embedding_weight_stride_0,
    embedding_weight_stride_1,
    embedding_output_stride_0,
    embedding_output_stride_1,
    cfg_scale,
    ACTUAL_BATCH_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    CODEBOOK_SIZE: tl.constexpr,
    AUDIO_EOS_ID: tl.constexpr,
    EMBEDDING_DIM: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
    BLOCK_EMBEDDING: tl.constexpr,
    USE_CFG: tl.constexpr,
    WRITE_EMBEDDING: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_VOCAB)
    values, invalid_input = _load_constrained_logits(
        logits_ptr,
        unfinished_ptr,
        finished_ptr,
        forbid_eos_ptr,
        row,
        offsets,
        cfg_scale,
        ACTUAL_BATCH_SIZE,
        VOCAB_SIZE,
        CODEBOOK_SIZE,
        AUDIO_EOS_ID,
        USE_CFG,
    )
    row_invalid = (tl.sum(invalid_input.to(tl.int32), axis=0) != 0) | (tl.max(values, axis=0) == -float("inf"))
    tl.atomic_or(invalid_rows_ptr + row, row_invalid.to(tl.int32))
    sampled_id = tl.argmax(values, axis=0, tie_break_left=True)
    _store_token_and_embedding(
        sampled_id,
        token_output_ptr,
        embedding_weight_ptr,
        embedding_output_ptr,
        row,
        token_output_stride,
        embedding_weight_stride_0,
        embedding_weight_stride_1,
        embedding_output_stride_0,
        embedding_output_stride_1,
        ACTUAL_BATCH_SIZE,
        EMBEDDING_DIM,
        BLOCK_EMBEDDING,
        USE_CFG,
        WRITE_EMBEDDING,
    )


@triton.jit
def _fused_topk_embedding_kernel(
    logits_ptr,
    unfinished_ptr,
    finished_ptr,
    forbid_eos_ptr,
    invalid_rows_ptr,
    rng_counters_ptr,
    rng_seed_ptr,
    token_output_ptr,
    embedding_weight_ptr,
    embedding_output_ptr,
    token_output_stride,
    embedding_weight_stride_0,
    embedding_weight_stride_1,
    embedding_output_stride_0,
    embedding_output_stride_1,
    cfg_scale,
    temperature,
    ACTUAL_BATCH_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    CODEBOOK_SIZE: tl.constexpr,
    AUDIO_EOS_ID: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_K_ROUNDED: tl.constexpr,
    EMBEDDING_DIM: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
    BLOCK_EMBEDDING: tl.constexpr,
    USE_CFG: tl.constexpr,
    WRITE_EMBEDDING: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_VOCAB)
    values, invalid_input = _load_constrained_logits(
        logits_ptr,
        unfinished_ptr,
        finished_ptr,
        forbid_eos_ptr,
        row,
        offsets,
        cfg_scale,
        ACTUAL_BATCH_SIZE,
        VOCAB_SIZE,
        CODEBOOK_SIZE,
        AUDIO_EOS_ID,
        USE_CFG,
    )
    row_invalid = (tl.sum(invalid_input.to(tl.int32), axis=0) != 0) | (tl.max(values, axis=0) == -float("inf"))
    tl.atomic_or(invalid_rows_ptr + row, row_invalid.to(tl.int32))

    # Triton's bitonic top-k requires a power-of-two K.  Selecting the next
    # power of two and reading element TOP_K - 1 gives the exact same threshold
    # as torch.topk(..., TOP_K).values[..., -1].
    top_values = tl.topk(values, TOP_K_ROUNDED)
    top_offsets = tl.arange(0, TOP_K_ROUNDED)
    threshold = tl.sum(tl.where(top_offsets == TOP_K - 1, top_values, 0.0), axis=0)
    eligible = values >= threshold

    # One independent device counter per batch item is copied from and back to
    # the caller's generation-session state around each graph replay.
    counter = tl.atomic_add(rng_counters_ptr + row, 1)
    # Keep the Philox offset at 64 bits.  Casting to uint32 would repeat the
    # stream after 2**32 / BLOCK_VOCAB local-AR positions.
    random_offsets = counter * BLOCK_VOCAB + offsets
    # Signed representation of the 32-bit golden-ratio constant 0x9E3779B9.
    rng_seed = tl.load(rng_seed_ptr)
    row_seed = (rng_seed + row * -1640531527).to(tl.uint32)
    uniform = tl.rand(row_seed, random_offsets)
    uniform = tl.maximum(tl.minimum(uniform, 0.99999994), 0.00000006)
    gumbel = -tl.log(-tl.log(uniform))
    sample_scores = tl.where(eligible, values / temperature + gumbel, -float("inf"))
    sampled_id = tl.argmax(sample_scores, axis=0, tie_break_left=True)

    _store_token_and_embedding(
        sampled_id,
        token_output_ptr,
        embedding_weight_ptr,
        embedding_output_ptr,
        row,
        token_output_stride,
        embedding_weight_stride_0,
        embedding_weight_stride_1,
        embedding_output_stride_0,
        embedding_output_stride_1,
        ACTUAL_BATCH_SIZE,
        EMBEDDING_DIM,
        BLOCK_EMBEDDING,
        USE_CFG,
        WRITE_EMBEDDING,
    )


@dataclass(frozen=True)
class FusedLocalARSamplingConfig:
    """Fixed-shape contract used to compile a local-AR sampling kernel."""

    actual_batch_size: int
    vocab_size: int
    codebook_size: int
    audio_eos_id: int
    top_k: int
    temperature: float
    cfg_scale: float
    use_cfg: bool

    def __post_init__(self) -> None:
        if self.actual_batch_size < 1:
            raise ValueError(f"actual_batch_size must be positive, got {self.actual_batch_size}")
        if self.codebook_size < 1:
            raise ValueError(f"codebook_size must be positive, got {self.codebook_size}")
        expected_vocab_size = self.codebook_size + _NUM_MAGPIE_SPECIAL_TOKENS
        if self.vocab_size != expected_vocab_size:
            raise ValueError(
                "The fused kernel requires the contiguous Magpie special-token layout: "
                f"vocab_size must equal codebook_size + {_NUM_MAGPIE_SPECIAL_TOKENS}, "
                f"got {self.vocab_size} and {self.codebook_size}"
            )
        if self.audio_eos_id < self.codebook_size or self.audio_eos_id >= self.vocab_size:
            raise ValueError(
                f"audio_eos_id must refer to a special token in [{self.codebook_size}, {self.vocab_size}), "
                f"got {self.audio_eos_id}"
            )
        if self.top_k < 1 or self.top_k > self.vocab_size:
            raise ValueError(f"top_k must be in [1, {self.vocab_size}], got {self.top_k}")
        if not math.isfinite(self.temperature):
            raise ValueError(f"temperature must be finite, got {self.temperature}")
        if not math.isfinite(self.cfg_scale):
            raise ValueError(f"cfg_scale must be finite, got {self.cfg_scale}")


@dataclass(frozen=True)
class LocalARRandomState:
    """Device-resident random state owned by one generation session."""

    seed: Tensor
    counters: Tensor

    @classmethod
    def create(
        cls,
        *,
        actual_batch_size: int,
        device: torch.device,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> LocalARRandomState:
        if actual_batch_size < 1:
            raise ValueError(f"actual_batch_size must be positive, got {actual_batch_size}")
        if device.type != "cuda":
            raise ValueError(f"Local-AR random state requires CUDA, got {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if seed is not None and (seed < 0 or seed >= 2**32):
            raise ValueError(f"seed must be in [0, 2**32), got {seed}")
        if seed is None:
            seed_tensor = torch.randint(
                0,
                2**31,
                (1,),
                dtype=torch.int64,
                device=device,
                generator=generator,
            )
        else:
            seed_tensor = torch.full((1,), seed, dtype=torch.int64, device=device)
        return cls(
            seed=seed_tensor,
            counters=torch.zeros(actual_batch_size, dtype=torch.int64, device=device),
        )

    def validate(self, *, actual_batch_size: int, device: torch.device) -> None:
        if self.seed.shape != (1,) or self.seed.dtype != torch.int64 or self.seed.device != device:
            raise ValueError(f"random-state seed must be a one-element int64 tensor on {device}")
        if (
            self.counters.shape != (actual_batch_size,)
            or self.counters.dtype != torch.int64
            or self.counters.device != device
            or not self.counters.is_contiguous()
        ):
            raise ValueError(
                f"random-state counters must be a contiguous int64 tensor of shape "
                f"{(actual_batch_size,)} on {device}"
            )


class FusedLocalARSampler:
    """Allocation-free, CUDA-Graph-safe fused sampler.

    ``sample_into`` launches exactly one dedicated Triton kernel.  When
    embedding buffers are supplied, the kernel also gathers and duplicates the
    next embedding for the conditional/unconditional CFG rows.

    Status masks must already be validated by the request boundary.  In
    particular, a row cannot be both unfinished and finished, and a finished
    row cannot globally forbid EOS.  Reading device masks on the host here
    would reintroduce the synchronization this implementation removes.
    """

    def __init__(self, config: FusedLocalARSamplingConfig, device: torch.device):
        if device.type != "cuda":
            raise ValueError(f"FusedLocalARSampler requires a CUDA device, got {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.config = config
        self.device = device
        self._invalid_rows = torch.zeros(config.actual_batch_size, dtype=torch.int32, device=device)

    def sample_into(
        self,
        logits: Tensor,
        unfinished_mask: Tensor,
        finished_mask: Tensor,
        forbid_eos_mask: Tensor,
        token_output: Tensor,
        embedding_weight: Tensor | None = None,
        embedding_output: Tensor | None = None,
        random_state: LocalARRandomState | None = None,
        invalid_rows: Tensor | None = None,
    ) -> None:
        """Sample tokens and optionally gather their next-step embeddings.

        Args:
            logits: ``(B, V)`` or CFG ``(2B, V)`` CUDA logits.
            unfinished_mask: ``(B,)`` CUDA bool mask; EOS is forbidden.
            finished_mask: ``(B,)`` CUDA bool mask; EOS is forced.
            forbid_eos_mask: ``(B,)`` CUDA bool mask for the global EOS gate.
            token_output: Caller-owned strided ``(B,)`` CUDA int64 output.
            embedding_weight: Optional ``(V, D)`` embedding table.
            embedding_output: Optional ``(B, D)`` or CFG ``(2B, D)`` output.
            random_state: Required session-owned seed/counters for stochastic sampling.
            invalid_rows: Optional graph-owned accumulation buffer. When omitted,
                this call checks its private buffer and raises on invalid logits.
        """

        self._validate_inputs(
            logits=logits,
            unfinished_mask=unfinished_mask,
            finished_mask=finished_mask,
            forbid_eos_mask=forbid_eos_mask,
            token_output=token_output,
            embedding_weight=embedding_weight,
            embedding_output=embedding_output,
        )
        config = self.config
        stochastic = config.temperature > 0.0 and config.top_k > 1
        if stochastic:
            if random_state is None:
                raise ValueError("Stochastic fused sampling requires a session-owned LocalARRandomState")
            random_state.validate(actual_batch_size=config.actual_batch_size, device=self.device)
        elif random_state is not None:
            random_state.validate(actual_batch_size=config.actual_batch_size, device=self.device)
        owns_invalid_rows = invalid_rows is None
        if owns_invalid_rows:
            invalid_rows = self._invalid_rows
            invalid_rows.zero_()
        elif (
            invalid_rows.shape != (config.actual_batch_size,)
            or invalid_rows.dtype != torch.int32
            or invalid_rows.device != self.device
            or not invalid_rows.is_contiguous()
        ):
            raise ValueError(
                f"invalid_rows must be a contiguous int32 tensor of shape "
                f"{(config.actual_batch_size,)} on {self.device}"
            )
        block_vocab = triton.next_power_of_2(config.vocab_size)
        write_embedding = embedding_weight is not None
        if write_embedding:
            embedding_dim = embedding_weight.size(1)
            block_embedding = triton.next_power_of_2(embedding_dim)
            weight = embedding_weight
            output = embedding_output
        else:
            embedding_dim = 1
            block_embedding = 1
            weight = logits
            output = logits

        common_arguments = (
            logits,
            unfinished_mask,
            finished_mask,
            forbid_eos_mask,
            invalid_rows,
            token_output,
            weight,
            output,
            token_output.stride(0),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            config.cfg_scale,
        )
        common_meta = {
            "ACTUAL_BATCH_SIZE": config.actual_batch_size,
            "VOCAB_SIZE": config.vocab_size,
            "CODEBOOK_SIZE": config.codebook_size,
            "AUDIO_EOS_ID": config.audio_eos_id,
            "EMBEDDING_DIM": embedding_dim,
            "BLOCK_VOCAB": block_vocab,
            "BLOCK_EMBEDDING": block_embedding,
            "USE_CFG": config.use_cfg,
            "WRITE_EMBEDDING": write_embedding,
            "num_warps": 8 if block_vocab >= 1024 else 4,
        }
        grid = (config.actual_batch_size,)
        if not stochastic:
            _fused_argmax_embedding_kernel[grid](*common_arguments, **common_meta)
        else:
            top_k_rounded = triton.next_power_of_2(config.top_k)
            _fused_topk_embedding_kernel[grid](
                logits,
                unfinished_mask,
                finished_mask,
                forbid_eos_mask,
                invalid_rows,
                random_state.counters,
                random_state.seed,
                token_output,
                weight,
                output,
                token_output.stride(0),
                weight.stride(0),
                weight.stride(1),
                output.stride(0),
                output.stride(1),
                config.cfg_scale,
                config.temperature,
                TOP_K=config.top_k,
                TOP_K_ROUNDED=top_k_rounded,
                **common_meta,
            )
        if owns_invalid_rows:
            torch.ops.aten._assert_async.msg(
                (invalid_rows == 0).all(),
                "Fused local-AR sampling received NaN, infinity, or no valid candidate",
            )

    def _validate_inputs(
        self,
        logits: Tensor,
        unfinished_mask: Tensor,
        finished_mask: Tensor,
        forbid_eos_mask: Tensor,
        token_output: Tensor,
        embedding_weight: Tensor | None,
        embedding_output: Tensor | None,
    ) -> None:
        config = self.config
        expected_input_batch = config.actual_batch_size * (2 if config.use_cfg else 1)
        if logits.ndim != 2 or tuple(logits.shape) != (expected_input_batch, config.vocab_size):
            raise ValueError(
                f"logits must have shape {(expected_input_batch, config.vocab_size)}, got {tuple(logits.shape)}"
            )
        if logits.device != self.device:
            raise ValueError(f"logits must be on {self.device}, got {logits.device}")
        if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"logits dtype must be float16, bfloat16, or float32, got {logits.dtype}")
        if not logits.is_contiguous():
            raise ValueError("logits must be contiguous")

        for name, mask in (
            ("unfinished_mask", unfinished_mask),
            ("finished_mask", finished_mask),
            ("forbid_eos_mask", forbid_eos_mask),
        ):
            if mask.shape != (config.actual_batch_size,):
                raise ValueError(f"{name} must have shape {(config.actual_batch_size,)}, got {tuple(mask.shape)}")
            if mask.device != self.device or mask.dtype != torch.bool or not mask.is_contiguous():
                raise ValueError(f"{name} must be a contiguous bool tensor on {self.device}")

        if token_output.shape != (config.actual_batch_size,):
            raise ValueError(
                f"token_output must have shape {(config.actual_batch_size,)}, got {tuple(token_output.shape)}"
            )
        if token_output.device != self.device or token_output.dtype != torch.int64:
            raise ValueError(f"token_output must be an int64 tensor on {self.device}")

        if (embedding_weight is None) != (embedding_output is None):
            raise ValueError("embedding_weight and embedding_output must either both be supplied or both be omitted")
        if embedding_weight is None:
            return
        if embedding_weight.ndim != 2 or embedding_weight.size(0) != config.vocab_size:
            raise ValueError(
                f"embedding_weight must have shape ({config.vocab_size}, D), got {tuple(embedding_weight.shape)}"
            )
        expected_embedding_batch = expected_input_batch
        if embedding_output.shape != (expected_embedding_batch, embedding_weight.size(1)):
            raise ValueError(
                f"embedding_output must have shape {(expected_embedding_batch, embedding_weight.size(1))}, "
                f"got {tuple(embedding_output.shape)}"
            )
        for name, tensor in (("embedding_weight", embedding_weight), ("embedding_output", embedding_output)):
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on {self.device}, got {tensor.device}")
            if tensor.dtype != embedding_weight.dtype:
                raise ValueError(
                    f"{name} dtype must match embedding_weight dtype {embedding_weight.dtype}, got {tensor.dtype}"
                )
            if tensor.stride(1) != 1:
                raise ValueError(f"{name} must be contiguous in its embedding dimension")
