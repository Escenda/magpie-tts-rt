"""Export-only tensor wrappers for the accepted Sofia Main Decoder.

The reference NeMo implementation owns K/V state through Python dataclasses
and mutates it with ``copy_``.  TensorRT needs that state to be an explicit
engine contract.  These wrappers express the same accepted operations with
tensor inputs and outputs only.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from nemo.collections.tts.models import MagpieTTSModel
from nemo.collections.tts.modules.ffn_modules import PositionwiseConvFF

from text_encoder_wrapper import (
    PLUGIN_NAMESPACE,
    PLUGIN_VERSION,
    _OracleGeluTanh,
    _OracleLayerNorm,
    _OracleSoftmax,
)


CFG_BATCH = 2
MODEL_WIDTH = 768
DECODER_LAYERS = 12
SELF_HEADS = 12
SELF_HEAD_WIDTH = 64
CROSS_HEADS = 1
CROSS_HEAD_WIDTH = 128
SOFIA_INDEX = 4
SOFIA_PREFIX_LENGTH = 217
PREFILL_LENGTH = 218
SELF_CACHE_CAPACITY = 467
FRAME_STACKING = 2
AUDIO_CODEBOOKS = 8
ALIGNMENT_LAYERS = (4, 5, 8, 9)
PRIOR_LAYERS = frozenset(range(2, 11))


def _int64_constant(graph, values: tuple[int, ...]):
    return graph.op(
        "Constant",
        value_t=torch.tensor(values, dtype=torch.int64),
    )


class _OraclePointwiseConvolution(torch.autograd.Function):
    """Execute the oracle Conv1d and lower kernel-one inference to MatMul."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        value: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        return functional.conv1d(
            value,
            weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )

    @staticmethod
    def symbolic(graph, value, weight):
        transposed_value = graph.op(
            "Transpose",
            value,
            perm_i=(0, 2, 1),
        )
        matrix_weight = graph.op(
            "Squeeze",
            weight,
            _int64_constant(graph, (2,)),
        )
        transposed_weight = graph.op(
            "Transpose",
            matrix_weight,
            perm_i=(1, 0),
        )
        projected = graph.op(
            "MatMul",
            transposed_value,
            transposed_weight,
        )
        return graph.op(
            "Transpose",
            projected,
            perm_i=(0, 2, 1),
        )


class _OracleMainCrossAttentionSoftmax(torch.autograd.Function):
    """Keep oracle QK math and lower it with Softmax mode 1."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query_heads: torch.Tensor,
        key: torch.Tensor,
        memory_mask: torch.Tensor,
        shape_reference: torch.Tensor,
    ) -> torch.Tensor:
        del ctx, shape_reference
        key_heads = key.transpose(1, 2)
        scores = (
            torch.matmul(query_heads, key_heads.transpose(2, 3))
            * (CROSS_HEAD_WIDTH**-0.5)
        )
        valid = memory_mask[:, None, None, :]
        scores = scores.masked_fill(~valid, float("-inf"))
        probabilities = functional.softmax(scores, dim=-1)
        return probabilities.masked_fill(~valid, 0.0)

    @staticmethod
    def symbolic(
        graph,
        query_heads,
        key,
        memory_mask,
        shape_reference,
    ):
        output = graph.op(
            "magpie_tts_rt::MagpieSoftmax",
            query_heads,
            key,
            memory_mask,
            shape_reference,
            mode_i=1,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
        )
        output.setType(shape_reference.type())
        return output


class _OracleMainSelfAttentionContext(torch.autograd.Function):
    """Lower the accepted prefill probabilities-by-value GEMM as mode 2."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        probabilities: torch.Tensor,
        value_heads: torch.Tensor,
        shape_reference: torch.Tensor,
    ) -> torch.Tensor:
        del ctx, shape_reference
        return torch.matmul(probabilities, value_heads)

    @staticmethod
    def symbolic(
        graph,
        probabilities,
        value_heads,
        shape_reference,
    ):
        output = graph.op(
            "magpie_tts_rt::MagpieSoftmax",
            probabilities,
            value_heads,
            shape_reference,
            mode_i=2,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
        )
        output.setType(shape_reference.type())
        return output


class _OracleMainSelfAttentionStepContext(torch.autograd.Function):
    """Slice to the active cache and lower the one-step context as mode 6."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        probabilities: torch.Tensor,
        value_heads: torch.Tensor,
        position: torch.Tensor,
        shape_reference: torch.Tensor,
    ) -> torch.Tensor:
        del ctx, shape_reference
        active_length = int(position.item()) + 1
        if active_length < PREFILL_LENGTH + 1:
            raise ValueError(
                "Main Decoder step position precedes the accepted prefix: "
                f"{active_length - 1}"
            )
        if active_length > SELF_CACHE_CAPACITY:
            raise ValueError(
                "Main Decoder step position exceeds the cache capacity: "
                f"{active_length - 1}"
            )
        return torch.matmul(
            probabilities[..., :active_length],
            value_heads[..., :active_length, :],
        )

    @staticmethod
    def symbolic(
        graph,
        probabilities,
        value_heads,
        position,
        shape_reference,
    ):
        active_length = graph.op(
            "Add",
            position,
            _int64_constant(graph, (1,)),
        )
        active_probabilities = graph.op(
            "Slice",
            probabilities,
            _int64_constant(graph, (0,)),
            active_length,
            _int64_constant(graph, (3,)),
            _int64_constant(graph, (1,)),
        )
        output = graph.op(
            "magpie_tts_rt::MagpieSoftmax",
            active_probabilities,
            value_heads,
            shape_reference,
            mode_i=6,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
        )
        output.setType(shape_reference.type())
        return output


class _OracleMainSelfAttentionStepScores(torch.autograd.Function):
    """Lower the accepted one-step self-attention QK GEMM as mode 5."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query_heads: torch.Tensor,
        key_transposed: torch.Tensor,
        shape_reference: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        active_length = shape_reference.size(-1)
        return torch.matmul(
            query_heads,
            key_transposed[..., :active_length],
        )

    @staticmethod
    def symbolic(
        graph,
        query_heads,
        key_transposed,
        shape_reference,
    ):
        output = graph.op(
            "magpie_tts_rt::MagpieSoftmax",
            query_heads,
            key_transposed,
            shape_reference,
            mode_i=5,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
        )
        output.setType(shape_reference.type())
        return output


class _OracleMainActiveCacheSlice(torch.autograd.Function):
    """Slice a fixed-capacity cache to the oracle's active prefix.

    The NeMo incremental self-attention implementation slices K/V state before
    the QK GEMM and softmax.  Masking a full-capacity score tensor afterwards
    is not numerically equivalent: the CUDA softmax reduction changes with the
    reduction width.  ``position`` remains a TensorRT shape input, so the
    active prefix length is selected at runtime instead of being frozen at
    export time.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        value: torch.Tensor,
        position: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        del ctx
        active_length = int(position.item()) + 1
        if active_length < PREFILL_LENGTH + 1:
            raise ValueError(
                "Main Decoder step position precedes the accepted prefix: "
                f"{active_length - 1}"
            )
        if active_length > SELF_CACHE_CAPACITY:
            raise ValueError(
                "Main Decoder step position exceeds the cache capacity: "
                f"{active_length - 1}"
            )
        return value.narrow(axis, 0, active_length)

    @staticmethod
    def symbolic(graph, value, position, axis):
        active_length = graph.op(
            "Add",
            position,
            _int64_constant(graph, (1,)),
        )
        return graph.op(
            "Slice",
            value,
            _int64_constant(graph, (0,)),
            active_length,
            _int64_constant(graph, (axis,)),
            _int64_constant(graph, (1,)),
        )


class _OracleMainCrossAttentionContext(torch.autograd.Function):
    """Lower the accepted prefill cross-attention context GEMM as mode 3."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        probabilities: torch.Tensor,
        value_heads: torch.Tensor,
        shape_reference: torch.Tensor,
    ) -> torch.Tensor:
        del ctx, shape_reference
        return torch.matmul(probabilities, value_heads)

    @staticmethod
    def symbolic(
        graph,
        probabilities,
        value_heads,
        shape_reference,
    ):
        output = graph.op(
            "magpie_tts_rt::MagpieSoftmax",
            probabilities,
            value_heads,
            shape_reference,
            mode_i=3,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
        )
        output.setType(shape_reference.type())
        return output


class _OracleMainAttentionPriorNormalization(torch.autograd.Function):
    """Preserve the complete oracle BF16 prior renormalization."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        probabilities: torch.Tensor,
        attention_prior: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        prior = (
            attention_prior[:, None]
            + torch.finfo(attention_prior.dtype).tiny
        )
        numerator = probabilities * prior
        denominator = numerator.sum(dim=-1, keepdim=True)
        return numerator / denominator

    @staticmethod
    def symbolic(graph, probabilities, attention_prior):
        output = graph.op(
            "magpie_tts_rt::MagpieSoftmax",
            probabilities,
            attention_prior,
            mode_i=4,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
        )
        output.setType(probabilities.type())
        return output


class _OracleMainAlignmentMean(torch.autograd.Function):
    """Lower the fixed four-layer alignment mean without reassociation."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        alignment_0: torch.Tensor,
        alignment_1: torch.Tensor,
        alignment_2: torch.Tensor,
        alignment_3: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        return torch.stack(
            (
                alignment_0,
                alignment_1,
                alignment_2,
                alignment_3,
            ),
            dim=1,
        ).mean(dim=1)

    @staticmethod
    def symbolic(
        graph,
        alignment_0,
        alignment_1,
        alignment_2,
        alignment_3,
    ):
        output = graph.op(
            "magpie_tts_rt::MagpieSoftmax",
            alignment_0,
            alignment_1,
            alignment_2,
            alignment_3,
            mode_i=7,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
        )
        output.setType(alignment_0.type())
        return output


def prefill_output_names() -> list[str]:
    names = ["last_hidden", "alignment"]
    for layer_index in range(DECODER_LAYERS):
        names.extend(
            [
                f"prefill_self_key_{layer_index}",
                f"prefill_self_value_{layer_index}",
                f"prefill_self_mask_{layer_index}",
                f"prefill_cross_key_{layer_index}",
                f"prefill_cross_value_{layer_index}",
            ]
        )
    return names


def step_input_names() -> list[str]:
    names = [
        "previous_codec_tokens",
        "position",
        "alignment_prior",
        "condition_mask",
    ]
    for layer_index in range(DECODER_LAYERS):
        names.extend(
            [
                f"step_self_key_in_{layer_index}",
                f"step_self_value_in_{layer_index}",
                f"step_self_mask_in_{layer_index}",
                f"step_cross_key_in_{layer_index}",
                f"step_cross_value_in_{layer_index}",
            ]
        )
    return names


def step_output_names() -> list[str]:
    names = ["decoder_hidden", "alignment"]
    for layer_index in range(DECODER_LAYERS):
        names.extend(
            [
                f"step_self_key_out_{layer_index}",
                f"step_self_value_out_{layer_index}",
                f"step_self_mask_out_{layer_index}",
            ]
        )
    return names


def _require_accepted_model(model: MagpieTTSModel) -> None:
    decoder = model.decoder
    failures: list[str] = []
    if model.model_type != "decoder_ce":
        failures.append(f"model_type={model.model_type!r}")
    if not model.has_baked_context_embedding:
        failures.append("baked_context_embedding is absent")
    if decoder.n_layers != DECODER_LAYERS:
        failures.append(f"decoder_layers={decoder.n_layers}")
    decoder_width = int(model.cfg.decoder.d_model)
    if decoder_width != MODEL_WIDTH:
        failures.append(f"decoder_width={decoder_width}")
    if model.frame_stacking_factor != FRAME_STACKING:
        failures.append(f"frame_stacking={model.frame_stacking_factor}")
    if model.num_audio_codebooks != AUDIO_CODEBOOKS:
        failures.append(f"audio_codebooks={model.num_audio_codebooks}")
    if tuple(model.inference_parameters.estimate_alignment_from_layers) != ALIGNMENT_LAYERS:
        failures.append(
            "alignment_layers="
            f"{tuple(model.inference_parameters.estimate_alignment_from_layers)}"
        )
    if frozenset(model.inference_parameters.apply_prior_to_layers) != PRIOR_LAYERS:
        failures.append(
            f"prior_layers={tuple(model.inference_parameters.apply_prior_to_layers)}"
        )
    for layer_index, layer in enumerate(decoder.layers):
        if not layer.has_xattn:
            failures.append(f"layer_{layer_index}_has_no_cross_attention")
        if layer.use_moe:
            failures.append(f"layer_{layer_index}_uses_moe")
        if not isinstance(layer.pos_ff, PositionwiseConvFF):
            failures.append(
                f"layer_{layer_index}_ffn={type(layer.pos_ff).__name__}"
            )
        elif layer.pos_ff.proj.kernel_size != 1 or layer.pos_ff.o_net.kernel_size != 1:
            failures.append(f"layer_{layer_index}_ffn_is_not_pointwise")
        elif (
            layer.pos_ff.proj.conv.bias is not None
            or layer.pos_ff.o_net.conv.bias is not None
            or layer.pos_ff.proj.conv.stride != (1,)
            or layer.pos_ff.o_net.conv.stride != (1,)
            or layer.pos_ff.proj.conv.padding != (0,)
            or layer.pos_ff.o_net.conv.padding != (0,)
            or layer.pos_ff.proj.conv.dilation != (1,)
            or layer.pos_ff.o_net.conv.dilation != (1,)
            or layer.pos_ff.proj.conv.groups != 1
            or layer.pos_ff.o_net.conv.groups != 1
        ):
            failures.append(
                f"layer_{layer_index}_ffn_pointwise_execution_mismatch"
            )
        if layer.self_attention.n_heads != SELF_HEADS:
            failures.append(
                f"layer_{layer_index}_self_heads={layer.self_attention.n_heads}"
            )
        if layer.self_attention.d_head != SELF_HEAD_WIDTH:
            failures.append(
                f"layer_{layer_index}_self_head_width={layer.self_attention.d_head}"
            )
        if layer.cross_attention.n_heads != CROSS_HEADS:
            failures.append(
                f"layer_{layer_index}_cross_heads={layer.cross_attention.n_heads}"
            )
        if layer.cross_attention.d_head != CROSS_HEAD_WIDTH:
            failures.append(
                f"layer_{layer_index}_cross_head_width={layer.cross_attention.d_head}"
            )
        if float(layer.cross_attention.scale) != CROSS_HEAD_WIDTH**-0.5:
            failures.append(
                f"layer_{layer_index}_cross_scale={layer.cross_attention.scale}"
            )
        for name, normalization in (
            ("self", layer.norm_self),
            ("cross_memory", layer.norm_xattn_memory),
            ("cross_query", layer.norm_xattn_query),
            ("ffn", layer.norm_pos_ff),
        ):
            if (
                tuple(normalization.normalized_shape) != (MODEL_WIDTH,)
                or normalization.bias is not None
            ):
                failures.append(
                    f"layer_{layer_index}_{name}_normalization_mismatch"
                )
    if (
        tuple(decoder.norm_out.normalized_shape) != (MODEL_WIDTH,)
        or decoder.norm_out.bias is not None
    ):
        failures.append("decoder_output_normalization_mismatch")
    if failures:
        raise RuntimeError(
            "model does not match the accepted Sofia Main Decoder contract: "
            + ", ".join(failures)
        )


def _self_attention_prefill(
    layer: torch.nn.Module,
    normalized_query: torch.Tensor,
    query_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, prefix_length, _ = normalized_query.shape
    attention = layer.self_attention
    qkv = attention.qkv_net(normalized_query).reshape(
        batch_size,
        prefix_length,
        3,
        SELF_HEADS,
        SELF_HEAD_WIDTH,
    )
    query, key, value = qkv.unbind(dim=2)
    query_heads = query.transpose(1, 2)
    key_heads = key.transpose(1, 2)
    value_heads = value.transpose(1, 2)
    scores = torch.matmul(query_heads, key_heads.transpose(2, 3)) * attention.scale
    valid = query_mask[:, None, :, None] & query_mask[:, None, None, :]
    causal = torch.tril(
        torch.ones(
            (prefix_length, prefix_length),
            dtype=torch.bool,
            device=normalized_query.device,
        )
    )[None, None]
    scores = scores.masked_fill(~valid, float("-inf"))
    scores = scores.masked_fill(~causal, float("-inf"))
    probabilities = _OracleSoftmax.apply(scores)
    probabilities = probabilities.masked_fill(~valid, 0.0)
    shape_reference = (
        probabilities[..., :1] + value_heads[:, :, :1, :]
    )
    output = _OracleMainSelfAttentionContext.apply(
        probabilities,
        value_heads,
        shape_reference,
    )
    output = output.transpose(1, 2).contiguous().view(
        batch_size, prefix_length, MODEL_WIDTH
    )
    return attention.o_net(output), key, value


def _cross_attention(
    layer: torch.nn.Module,
    normalized_query: torch.Tensor,
    memory_mask: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_prior: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, query_length, _ = normalized_query.shape
    attention = layer.cross_attention
    query = attention.q_net(normalized_query).reshape(
        batch_size,
        query_length,
        CROSS_HEADS,
        CROSS_HEAD_WIDTH,
    )
    query_heads = query.transpose(1, 2)
    value_heads = value.transpose(1, 2)
    query_shape_reference = query_heads[..., :1]
    key_shape_reference = key.transpose(1, 2).transpose(2, 3)[:, :, :1, :]
    shape_reference = query_shape_reference + key_shape_reference
    probabilities = _OracleMainCrossAttentionSoftmax.apply(
        query_heads,
        key,
        memory_mask,
        shape_reference,
    )
    if attention_prior is not None:
        probabilities = _OracleMainAttentionPriorNormalization.apply(
            probabilities,
            attention_prior,
        )
    context_shape_reference = (
        probabilities[..., :1] + value_heads[:, :, :1, :]
    )
    output = _OracleMainCrossAttentionContext.apply(
        probabilities,
        value_heads,
        context_shape_reference,
    )
    output = output.transpose(1, 2).contiguous().view(
        batch_size, query_length, CROSS_HEAD_WIDTH
    )
    alignment = probabilities[:, :, -1, :].mean(dim=1)
    return attention.o_net(output), alignment


def _pointwise_feed_forward(
    layer: torch.nn.Module,
    normalized_x: torch.Tensor,
) -> torch.Tensor:
    """Apply the accepted kernel-size-one FFN without BOOL arithmetic casts."""
    feed_forward = layer.pos_ff
    projected = _OraclePointwiseConvolution.apply(
        normalized_x.transpose(1, 2),
        feed_forward.proj.conv.weight,
    )
    projected = _OracleGeluTanh.apply(projected)
    return _OraclePointwiseConvolution.apply(
        projected,
        feed_forward.o_net.conv.weight,
    ).transpose(1, 2)


def _oracle_layer_norm(
    normalization: torch.nn.LayerNorm,
    value: torch.Tensor,
) -> torch.Tensor:
    return _OracleLayerNorm.apply(value, normalization.weight)


def _apply_query_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))


def _pad_prefill_cache(tensor: torch.Tensor, fill_value: float | bool) -> torch.Tensor:
    remaining = SELF_CACHE_CAPACITY - tensor.size(1)
    if tensor.ndim == 4:
        padding = (0, 0, 0, 0, 0, remaining)
    elif tensor.ndim == 2:
        padding = (0, remaining)
    else:
        raise ValueError(f"unsupported prefill cache rank: {tensor.ndim}")
    return functional.pad(tensor, padding, value=fill_value)


class MainDecoderPrefillWrapper(torch.nn.Module):
    """Sofia prefill with fixed 217-token voice prefix and AUDIO_BOS."""

    def __init__(self, model: MagpieTTSModel) -> None:
        super().__init__()
        _require_accepted_model(model)
        self.decoder = model.decoder

        prefix, lengths = model.get_baked_context_embeddings_batch(
            batch_size=1,
            speaker_indices=SOFIA_INDEX,
        )
        prefix_length = int(lengths[0].item())
        if prefix_length != SOFIA_PREFIX_LENGTH or prefix.size(1) != SOFIA_PREFIX_LENGTH:
            raise RuntimeError(
                "Sofia prefix must be exactly "
                f"{SOFIA_PREFIX_LENGTH} tokens, got length={prefix_length}, "
                f"shape={tuple(prefix.shape)}"
            )
        conditional_prefix = prefix
        unconditional_prefix = torch.zeros_like(prefix)
        cfg_prefix = torch.cat([conditional_prefix, unconditional_prefix], dim=0)
        cfg_prefix_mask = torch.ones(
            (CFG_BATCH, SOFIA_PREFIX_LENGTH),
            dtype=torch.bool,
            device=prefix.device,
        )

        audio_bos_codes = torch.full(
            (1, AUDIO_CODEBOOKS, FRAME_STACKING),
            model.audio_bos_id,
            dtype=torch.long,
            device=prefix.device,
        )
        audio_bos_embedding: torch.Tensor | None = None
        for frame_index in range(FRAME_STACKING):
            for codebook_index in range(AUDIO_CODEBOOKS):
                tokens = audio_bos_codes[
                    :, codebook_index, frame_index : frame_index + 1
                ]
                embedding = model.audio_embeddings[
                    codebook_index + frame_index * AUDIO_CODEBOOKS
                ](tokens)
                audio_bos_embedding = (
                    embedding
                    if audio_bos_embedding is None
                    else audio_bos_embedding + embedding
                )
        if audio_bos_embedding is None:
            raise AssertionError("AUDIO_BOS embedding was not constructed")
        audio_bos_embedding = audio_bos_embedding / (
            AUDIO_CODEBOOKS * FRAME_STACKING
        )
        cfg_audio_bos = audio_bos_embedding.expand(CFG_BATCH, -1, -1).clone()
        cfg_audio_mask = torch.ones(
            (CFG_BATCH, 1), dtype=torch.bool, device=prefix.device
        )
        self.register_buffer(
            "cfg_prefix_and_bos",
            torch.cat([cfg_prefix, cfg_audio_bos], dim=1),
            persistent=True,
        )
        self.register_buffer(
            "cfg_prefix_and_bos_mask",
            torch.cat([cfg_prefix_mask, cfg_audio_mask], dim=1),
            persistent=True,
        )

    def forward(
        self,
        condition: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        x = self.cfg_prefix_and_bos
        x_mask = self.cfg_prefix_and_bos_mask
        x = x + self.decoder.position_embeddings.weight[:PREFILL_LENGTH].unsqueeze(0)
        alignment_scores: list[torch.Tensor] = []
        state_outputs: list[torch.Tensor] = []

        for layer_index, layer in enumerate(self.decoder.layers):
            x = _apply_query_mask(x, x_mask)
            self_output, self_key, self_value = _self_attention_prefill(
                layer,
                _oracle_layer_norm(layer.norm_self, x),
                x_mask,
            )
            x = x + self_output

            normalized_memory = _oracle_layer_norm(
                layer.norm_xattn_memory,
                condition,
            )
            memory_batch, memory_length, _ = normalized_memory.shape
            cross_kv = layer.cross_attention.kv_net(normalized_memory).reshape(
                memory_batch,
                memory_length,
                2,
                CROSS_HEADS,
                CROSS_HEAD_WIDTH,
            )
            cross_key, cross_value = cross_kv.unbind(dim=2)
            cross_output, alignment = _cross_attention(
                layer,
                _oracle_layer_norm(layer.norm_xattn_query, x),
                condition_mask,
                cross_key,
                cross_value,
                None,
            )
            x = x + cross_output
            if layer_index in ALIGNMENT_LAYERS:
                alignment_scores.append(alignment)

            x = x + _pointwise_feed_forward(
                layer,
                _oracle_layer_norm(layer.norm_pos_ff, x),
            )
            x = _apply_query_mask(x, x_mask)
            state_outputs.extend(
                [
                    _pad_prefill_cache(self_key, 0.0),
                    _pad_prefill_cache(self_value, 0.0),
                    _pad_prefill_cache(x_mask, False),
                    cross_key,
                    cross_value,
                ]
            )

        x = _oracle_layer_norm(self.decoder.norm_out, x)
        alignment = _OracleMainAlignmentMean.apply(*alignment_scores)
        return (x[:, -1:, :], alignment, *state_outputs)


class MainDecoderStepWrapper(torch.nn.Module):
    """One exact decoder position with explicit external K/V state."""

    def __init__(self, model: MagpieTTSModel) -> None:
        super().__init__()
        _require_accepted_model(model)
        self.decoder = model.decoder
        self.audio_embeddings = model.audio_embeddings
        self.register_buffer(
            "cache_positions",
            torch.arange(
                SELF_CACHE_CAPACITY,
                dtype=torch.long,
                device=model.decoder.position_embeddings.weight.device,
            ),
            persistent=False,
        )

    def _embed_codec_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        audio_embedding: torch.Tensor | None = None
        for frame_index in range(FRAME_STACKING):
            for codebook_index in range(AUDIO_CODEBOOKS):
                embedding = self.audio_embeddings[
                    codebook_index + frame_index * AUDIO_CODEBOOKS
                ](tokens[:, codebook_index, frame_index : frame_index + 1])
                audio_embedding = (
                    embedding
                    if audio_embedding is None
                    else audio_embedding + embedding
                )
        if audio_embedding is None:
            raise AssertionError("codec embedding was not constructed")
        return audio_embedding / (AUDIO_CODEBOOKS * FRAME_STACKING)

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        expected_inputs = 4 + DECODER_LAYERS * 5
        if len(inputs) != expected_inputs:
            raise ValueError(
                f"Main Decoder step expects {expected_inputs} tensors, got {len(inputs)}"
            )
        previous_codec_tokens, position, attention_prior, condition_mask = inputs[:4]
        layer_inputs = inputs[4:]

        embedded = self._embed_codec_tokens(previous_codec_tokens)
        x = embedded.expand(CFG_BATCH, -1, -1)
        position_index = position.reshape(1)
        x = x + self.decoder.position_embeddings(position_index).unsqueeze(0)
        x_mask = torch.ones(
            (CFG_BATCH, 1),
            dtype=torch.bool,
            device=previous_codec_tokens.device,
        )
        write_position = self.cache_positions == position
        key_value_selector = write_position[None, :, None, None]
        mask_selector = write_position[None, :]

        alignment_scores: list[torch.Tensor] = []
        state_outputs: list[torch.Tensor] = []
        for layer_index, layer in enumerate(self.decoder.layers):
            base = layer_index * 5
            self_key = layer_inputs[base]
            self_value = layer_inputs[base + 1]
            self_mask = layer_inputs[base + 2]
            cross_key = layer_inputs[base + 3]
            cross_value = layer_inputs[base + 4]

            x = _apply_query_mask(x, x_mask)
            normalized_query = _oracle_layer_norm(layer.norm_self, x)
            qkv = layer.self_attention.qkv_net(normalized_query).reshape(
                CFG_BATCH,
                1,
                3,
                SELF_HEADS,
                SELF_HEAD_WIDTH,
            )
            query, key_step, value_step = qkv.unbind(dim=2)
            updated_key = torch.where(key_value_selector, key_step, self_key)
            updated_value = torch.where(key_value_selector, value_step, self_value)
            updated_mask = torch.where(mask_selector, x_mask, self_mask)

            query_heads = query.transpose(1, 2)
            key_heads = updated_key.transpose(1, 2)
            value_heads = updated_value.transpose(1, 2)
            key_transposed = key_heads.transpose(2, 3)
            active_key_transposed = _OracleMainActiveCacheSlice.apply(
                key_transposed,
                position,
                3,
            )
            active_mask = _OracleMainActiveCacheSlice.apply(
                updated_mask,
                position,
                1,
            )
            self_scores_shape_reference = (
                query_heads[..., :1]
                + active_key_transposed[:, :, :1, :]
            )
            scores = (
                _OracleMainSelfAttentionStepScores.apply(
                    query_heads,
                    active_key_transposed,
                    self_scores_shape_reference,
                )
                * layer.self_attention.scale
            )
            valid = x_mask[:, None, :, None] & active_mask[:, None, None, :]
            scores = scores.masked_fill(~valid, float("-inf"))
            probabilities = _OracleSoftmax.apply(scores)
            probabilities = probabilities.masked_fill(~valid, 0.0)
            self_context_shape_reference = (
                probabilities[..., :1] + value_heads[:, :, :1, :]
            )
            self_output = _OracleMainSelfAttentionStepContext.apply(
                probabilities,
                value_heads,
                position,
                self_context_shape_reference,
            )
            self_output = self_output.transpose(1, 2).contiguous().view(
                CFG_BATCH, 1, MODEL_WIDTH
            )
            x = x + layer.self_attention.o_net(self_output)

            layer_prior = attention_prior if layer_index in PRIOR_LAYERS else None
            cross_output, alignment = _cross_attention(
                layer,
                _oracle_layer_norm(layer.norm_xattn_query, x),
                condition_mask,
                cross_key,
                cross_value,
                layer_prior,
            )
            x = x + cross_output
            if layer_index in ALIGNMENT_LAYERS:
                alignment_scores.append(alignment)

            x = x + _pointwise_feed_forward(
                layer,
                _oracle_layer_norm(layer.norm_pos_ff, x),
            )
            x = _apply_query_mask(x, x_mask)
            state_outputs.extend([updated_key, updated_value, updated_mask])

        x = _oracle_layer_norm(self.decoder.norm_out, x)
        alignment = _OracleMainAlignmentMean.apply(*alignment_scores)
        return (x[:, -1, :], alignment, *state_outputs)


def prefill_dynamic_axes() -> dict[str, dict[int, str]]:
    axes: dict[str, dict[int, str]] = {
        "condition": {1: "text_tokens"},
        "condition_mask": {1: "text_tokens"},
        "alignment": {1: "text_tokens"},
    }
    for layer_index in range(DECODER_LAYERS):
        axes[f"prefill_cross_key_{layer_index}"] = {1: "text_tokens"}
        axes[f"prefill_cross_value_{layer_index}"] = {1: "text_tokens"}
    return axes


def step_dynamic_axes() -> dict[str, dict[int, str]]:
    axes: dict[str, dict[int, str]] = {
        "alignment_prior": {2: "text_tokens"},
        "condition_mask": {1: "text_tokens"},
        "alignment": {1: "text_tokens"},
    }
    for layer_index in range(DECODER_LAYERS):
        axes[f"step_cross_key_in_{layer_index}"] = {1: "text_tokens"}
        axes[f"step_cross_value_in_{layer_index}"] = {1: "text_tokens"}
    return axes


def make_prefill_example(
    *,
    text_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if text_tokens < 1:
        raise ValueError(f"text_tokens must be positive, got {text_tokens}")
    condition = torch.randn(
        (CFG_BATCH, text_tokens, MODEL_WIDTH),
        dtype=dtype,
        device=device,
    )
    condition_mask = torch.ones(
        (CFG_BATCH, text_tokens),
        dtype=torch.bool,
        device=device,
    )
    condition_mask[1, 1:] = False
    return condition, condition_mask


def make_step_example(
    *,
    text_tokens: int,
    position: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    if text_tokens < 1:
        raise ValueError(f"text_tokens must be positive, got {text_tokens}")
    if position < PREFILL_LENGTH or position >= SELF_CACHE_CAPACITY:
        raise ValueError(
            f"position must be in [{PREFILL_LENGTH}, {SELF_CACHE_CAPACITY}), "
            f"got {position}"
        )
    tensors: list[torch.Tensor] = [
        torch.zeros(
            (1, AUDIO_CODEBOOKS, FRAME_STACKING),
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(position, dtype=torch.long, device=device),
        torch.ones(
            (CFG_BATCH, 1, text_tokens),
            dtype=dtype,
            device=device,
        ),
    ]
    condition_mask = torch.ones(
        (CFG_BATCH, text_tokens), dtype=torch.bool, device=device
    )
    condition_mask[1, 1:] = False
    tensors.append(condition_mask)
    for _ in range(DECODER_LAYERS):
        key = torch.zeros(
            (CFG_BATCH, SELF_CACHE_CAPACITY, SELF_HEADS, SELF_HEAD_WIDTH),
            dtype=dtype,
            device=device,
        )
        value = torch.zeros_like(key)
        mask = torch.zeros(
            (CFG_BATCH, SELF_CACHE_CAPACITY),
            dtype=torch.bool,
            device=device,
        )
        mask[:, :position] = True
        cross_key = torch.randn(
            (CFG_BATCH, text_tokens, CROSS_HEADS, CROSS_HEAD_WIDTH),
            dtype=dtype,
            device=device,
        )
        cross_value = torch.randn_like(cross_key)
        tensors.extend([key, value, mask, cross_key, cross_value])
    return tuple(tensors)


def require_unique_names(names: Sequence[str], label: str) -> None:
    if len(names) != len(set(names)):
        raise RuntimeError(f"{label} contains duplicate tensor names")
