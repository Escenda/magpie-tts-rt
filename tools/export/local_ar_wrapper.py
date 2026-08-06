"""Tensor-only export wrapper for the accepted 16-position Local AR path.

The wrapper deliberately keeps the model's autoregressive dependency visible:
each sampled token is embedded and becomes the next Local Transformer input.
Sampling and EOS are represented by custom ONNX operations. TensorRT resolves
those operations to the CUDA IPluginV3 implementations in
``libmagpie_tts_rt_plugins.so``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from nemo.collections.tts.models import MagpieTTSModel
from nemo.collections.tts.modules.ffn_modules import PositionwiseConvFF
from nemo.collections.tts.modules.magpietts_fused_sampling import (
    FusedLocalARSampler,
    FusedLocalARSamplingConfig,
    LocalARRandomState,
)


CFG_BATCH = 2
ACTUAL_BATCH = 1
MODEL_WIDTH = 768
LOCAL_LAYERS = 2
LOCAL_HEADS = 12
LOCAL_HEAD_WIDTH = 64
LOCAL_POSITIONS = 16
POSITION_TABLE_ROWS = 18
CODEBOOKS = 8
FRAMES_PER_STEP = 2
CODEBOOK_SIZE = 2016
VOCABULARY_SIZE = 2024
AUDIO_EOS_ID = 2017
TOP_K = 80
TEMPERATURE = 0.6
CFG_SCALE = 2.5
PHILOX_STRIDE = 2048
UINT32_MODULUS = 1 << 32
UINT32_MASK = UINT32_MODULUS - 1
INT64_COUNTER_MAX = ((1 << 63) - 1 - (PHILOX_STRIDE - 1)) // PHILOX_STRIDE
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "magpie_tts_rt"
LAYER_NORM_EPSILON = 1.0e-5


def _require_accepted_model(model: MagpieTTSModel) -> None:
    helper = model._lt_helper
    transformer = helper.local_transformer
    failures: list[str] = []
    if model.local_transformer_type.value != "autoregressive":
        failures.append(f"local_transformer_type={model.local_transformer_type!s}")
    if model.num_audio_codebooks != CODEBOOKS:
        failures.append(f"codebooks={model.num_audio_codebooks}")
    if model.frame_stacking_factor != FRAMES_PER_STEP:
        failures.append(f"frame_stacking={model.frame_stacking_factor}")
    if model.codebook_size != CODEBOOK_SIZE:
        failures.append(f"codebook_size={model.codebook_size}")
    if model.audio_eos_id != AUDIO_EOS_ID:
        failures.append(f"audio_eos_id={model.audio_eos_id}")
    if len(helper.audio_embeddings) != LOCAL_POSITIONS:
        failures.append(f"audio_embeddings={len(helper.audio_embeddings)}")
    if len(helper.local_transformer_out_projections) != LOCAL_POSITIONS:
        failures.append(
            "local_output_projections="
            f"{len(helper.local_transformer_out_projections)}"
        )
    if transformer.n_layers != LOCAL_LAYERS:
        failures.append(f"local_layers={transformer.n_layers}")
    if not transformer.use_learnable_pos_emb:
        failures.append("learnable_position_embedding=false")
    if transformer.position_embeddings is None:
        failures.append("learnable_position_embedding_is_absent")
    elif (
        transformer.position_embeddings.num_embeddings != POSITION_TABLE_ROWS
        or transformer.position_embeddings.embedding_dim != MODEL_WIDTH
    ):
        failures.append(
            "learnable_position_embedding_shape="
            f"{tuple(transformer.position_embeddings.weight.shape)}"
        )
    if not isinstance(transformer.norm_out, torch.nn.Identity):
        failures.append(f"local_norm_out={type(transformer.norm_out).__name__}")
    if not isinstance(transformer.dropout, torch.nn.Dropout):
        failures.append(f"local_dropout={type(transformer.dropout).__name__}")
    elif transformer.dropout.p != 0.0:
        failures.append(f"local_dropout_p={transformer.dropout.p}")
    for layer_index, layer in enumerate(transformer.layers):
        if layer.has_xattn:
            failures.append(f"local_layer_{layer_index}_has_cross_attention")
        if layer.use_moe:
            failures.append(f"local_layer_{layer_index}_uses_moe")
        if layer.self_attention.n_heads != LOCAL_HEADS:
            failures.append(
                f"local_layer_{layer_index}_heads={layer.self_attention.n_heads}"
            )
        if layer.self_attention.d_head != LOCAL_HEAD_WIDTH:
            failures.append(
                "local_layer_"
                f"{layer_index}_head_width={layer.self_attention.d_head}"
            )
        if not isinstance(layer.pos_ff, PositionwiseConvFF):
            failures.append(
                f"local_layer_{layer_index}_ffn={type(layer.pos_ff).__name__}"
            )
        elif (
            layer.pos_ff.proj.kernel_size != 1
            or layer.pos_ff.o_net.kernel_size != 1
        ):
            failures.append(f"local_layer_{layer_index}_ffn_kernel_is_not_one")
        if (
            not isinstance(layer.pos_ff.non_linearity, torch.nn.GELU)
            or layer.pos_ff.non_linearity.approximate != "tanh"
        ):
            failures.append(
                f"local_layer_{layer_index}_gelu_is_not_tanh"
            )
        for norm_name, normalization in (
            ("norm_self", layer.norm_self),
            ("norm_pos_ff", layer.norm_pos_ff),
        ):
            if (
                not isinstance(normalization, torch.nn.LayerNorm)
                or tuple(normalization.normalized_shape) != (MODEL_WIDTH,)
                or normalization.eps != LAYER_NORM_EPSILON
                or normalization.weight is None
                or normalization.bias is not None
            ):
                failures.append(
                    f"local_layer_{layer_index}_{norm_name}_contract_mismatch"
                )
    if model.inference_parameters.topk != TOP_K:
        failures.append(f"top_k={model.inference_parameters.topk}")
    if model.inference_parameters.temperature != TEMPERATURE:
        failures.append(f"temperature={model.inference_parameters.temperature}")
    if model.inference_parameters.cfg_scale != CFG_SCALE:
        failures.append(f"cfg_scale={model.inference_parameters.cfg_scale}")
    if model.inference_parameters.eos_detection_method != "argmax_or_multinomial_any":
        failures.append(
            "eos_detection_method="
            f"{model.inference_parameters.eos_detection_method!r}"
        )
    if failures:
        raise RuntimeError(
            "model does not match the accepted Local AR contract: "
            + ", ".join(failures)
        )


def _umulhi_u32(left: torch.Tensor, right: int) -> torch.Tensor:
    """Return the high word of unsigned 32-bit multiplication.

    PyTorch INT64 multiplication wraps in two's-complement. Masking the
    arithmetic right shift recovers the upper unsigned word exactly.
    """

    product = left * right
    return (product >> 32) & UINT32_MASK


def philox_uniform(seed: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Match Triton 3.7 ``tl.rand`` for a scalar 32-bit seed and INT64 offsets."""

    c0 = offsets & UINT32_MASK
    c1 = (offsets >> 32) & UINT32_MASK
    c2 = torch.zeros_like(c0)
    c3 = torch.zeros_like(c0)
    k0 = seed.reshape(1).expand_as(c0) & UINT32_MASK
    k1 = torch.zeros_like(k0)
    for _ in range(10):
        old_c0 = c0
        old_c2 = c2
        c0 = _umulhi_u32(old_c2, 0xCD9E8D57) ^ c1 ^ k0
        c2 = _umulhi_u32(old_c0, 0xD2511F53) ^ c3 ^ k1
        c1 = (old_c2 * 0xCD9E8D57) & UINT32_MASK
        c3 = (old_c0 * 0xD2511F53) & UINT32_MASK
        k0 = (k0 + 0x9E3779B9) & UINT32_MASK
        k1 = (k1 + 0xBB67AE85) & UINT32_MASK
    signed = torch.where(c0 >= (1 << 31), c0 - UINT32_MODULUS, c0)
    magnitude = torch.where(signed < 0, -signed - 1, signed)
    return magnitude.to(torch.float32) * 4.6566127342e-10


def _reference_sample(
    logits: torch.Tensor,
    unfinished: torch.Tensor,
    finished: torch.Tensor,
    forbid_eos: torch.Tensor,
    rng_seed: torch.Tensor,
    rng_counter: torch.Tensor,
    embedding_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure tensor statement of the locked fused-sampling contract."""

    conditional = logits[:ACTUAL_BATCH]
    unconditional = logits[ACTUAL_BATCH:]
    guided_bf16 = (conditional * CFG_SCALE).to(logits.dtype)
    guided_bf16 = (
        guided_bf16 + unconditional * (1.0 - CFG_SCALE)
    ).to(logits.dtype)
    guided = guided_bf16.to(torch.float32)
    finite = torch.isfinite(guided).all(dim=1)

    token_ids = torch.arange(
        VOCABULARY_SIZE,
        dtype=torch.int64,
        device=logits.device,
    ).unsqueeze(0)
    special_forbidden = (token_ids >= CODEBOOK_SIZE) & (
        token_ids != AUDIO_EOS_ID
    )
    constrained = guided.masked_fill(special_forbidden, float("-inf"))
    eos_forbidden = unfinished | forbid_eos
    constrained = torch.where(
        eos_forbidden.unsqueeze(1) & (token_ids == AUDIO_EOS_ID),
        torch.full_like(constrained, float("-inf")),
        constrained,
    )
    forced_eos = torch.where(
        token_ids == AUDIO_EOS_ID,
        torch.zeros_like(constrained),
        torch.full_like(constrained, float("-inf")),
    )
    constrained = torch.where(finished.unsqueeze(1), forced_eos, constrained)
    has_candidate = torch.isfinite(constrained).any(dim=1)

    threshold = torch.topk(
        constrained,
        k=TOP_K,
        dim=1,
        largest=True,
        sorted=True,
    ).values[:, -1:]
    eligible = constrained >= threshold

    counter_valid = (rng_counter >= 0) & (rng_counter <= INT64_COUNTER_MAX)
    safe_counter = torch.where(
        counter_valid,
        rng_counter,
        torch.zeros_like(rng_counter),
    )
    offsets = (
        safe_counter.unsqueeze(1) * PHILOX_STRIDE
        + torch.arange(
            VOCABULARY_SIZE,
            dtype=torch.int64,
            device=logits.device,
        ).unsqueeze(0)
    )
    row_ids = torch.arange(
        ACTUAL_BATCH,
        dtype=torch.int64,
        device=logits.device,
    )
    row_seeds = (
        rng_seed.reshape(1) + row_ids * 0x9E3779B9
    ) & UINT32_MASK
    uniform_rows: list[torch.Tensor] = []
    for row in range(ACTUAL_BATCH):
        uniform_rows.append(philox_uniform(row_seeds[row : row + 1], offsets[row]))
    uniform = torch.stack(uniform_rows, dim=0)
    uniform = uniform.clamp(0.00000006, 0.99999994)
    gumbel = -torch.log(-torch.log(uniform))
    scores = torch.where(
        eligible,
        constrained / TEMPERATURE + gumbel,
        torch.full_like(constrained, float("-inf")),
    )
    sampled = torch.argmax(scores, dim=1).to(torch.int64)
    next_embedding = functional.embedding(sampled, embedding_weight)
    next_embedding = next_embedding.repeat(CFG_BATCH, 1)

    valid = finite & has_candidate & counter_valid
    status_overlap = unfinished & finished
    invalid_forced_gate = finished & forbid_eos
    valid = valid & ~status_overlap & ~invalid_forced_gate
    invalid_rows = (~valid).to(torch.int32)
    updated_counter = torch.where(
        counter_valid,
        rng_counter + 1,
        rng_counter,
    )
    return sampled, next_embedding, updated_counter, invalid_rows


def _accepted_triton_sample(
    logits: torch.Tensor,
    unfinished: torch.Tensor,
    finished: torch.Tensor,
    forbid_eos: torch.Tensor,
    rng_seed: torch.Tensor,
    rng_counter: torch.Tensor,
    embedding_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invoke the authenticated Triton oracle used by the accepted fixture."""

    sampler = FusedLocalARSampler(
        FusedLocalARSamplingConfig(
            actual_batch_size=ACTUAL_BATCH,
            vocab_size=VOCABULARY_SIZE,
            codebook_size=CODEBOOK_SIZE,
            audio_eos_id=AUDIO_EOS_ID,
            top_k=TOP_K,
            temperature=TEMPERATURE,
            cfg_scale=CFG_SCALE,
            use_cfg=True,
        ),
        logits.device,
    )
    sampled = torch.empty(
        (ACTUAL_BATCH,),
        dtype=torch.int64,
        device=logits.device,
    )
    next_embedding = torch.empty(
        (CFG_BATCH, MODEL_WIDTH),
        dtype=embedding_weight.dtype,
        device=embedding_weight.device,
    )
    invalid_rows = torch.zeros(
        (ACTUAL_BATCH,),
        dtype=torch.int32,
        device=logits.device,
    )
    random_state = LocalARRandomState(
        seed=rng_seed,
        counters=rng_counter.clone(),
    )
    sampler.sample_into(
        logits,
        unfinished,
        finished,
        forbid_eos,
        sampled,
        embedding_weight,
        next_embedding,
        random_state=random_state,
        invalid_rows=invalid_rows,
    )
    return sampled, next_embedding, random_state.counters, invalid_rows


class _FusedSampling(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        logits: torch.Tensor,
        unfinished: torch.Tensor,
        finished: torch.Tensor,
        forbid_eos: torch.Tensor,
        rng_seed: torch.Tensor,
        rng_counter: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del ctx
        if torch.onnx.is_in_onnx_export():
            # The legacy exporter executes forward before replacing this
            # autograd function with its symbolic custom node. Running Triton
            # under TorchScript tracing changes Python shape values into trace
            # values and is unsupported. These tensors define only the output
            # dtype/shape; the symbolic node below retains every real input.
            sampled = torch.zeros_like(rng_counter, dtype=torch.int64)
            next_embedding = embedding_weight[:1].repeat(CFG_BATCH, 1)
            updated_counter = rng_counter + 1
            invalid_rows = torch.zeros_like(rng_counter, dtype=torch.int32)
            return (
                sampled,
                next_embedding,
                updated_counter,
                invalid_rows,
            )
        return _accepted_triton_sample(
            logits,
            unfinished,
            finished,
            forbid_eos,
            rng_seed,
            rng_counter,
            embedding_weight,
        )

    @staticmethod
    def symbolic(
        graph,
        logits,
        unfinished,
        finished,
        forbid_eos,
        rng_seed,
        rng_counter,
        embedding_weight,
    ):
        sampled, next_embedding, updated_counter, invalid_rows = graph.op(
            "magpie_tts_rt::MagpieLocalARSampling",
            logits,
            unfinished,
            finished,
            forbid_eos,
            rng_seed,
            rng_counter,
            embedding_weight,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
            outputs=4,
        )
        sampled.setType(
            rng_counter.type().with_dtype(torch.int64).with_sizes([1])
        )
        next_embedding.setType(
            embedding_weight.type().with_sizes([CFG_BATCH, MODEL_WIDTH])
        )
        updated_counter.setType(
            rng_counter.type().with_dtype(torch.int64).with_sizes([1])
        )
        invalid_rows.setType(
            rng_counter.type().with_dtype(torch.int32).with_sizes([1])
        )
        return sampled, next_embedding, updated_counter, invalid_rows


def _reference_eos(
    decoder_hidden: torch.Tensor,
    codec_tokens: torch.Tensor,
    unfinished: torch.Tensor,
    finished: torch.Tensor,
    forbid_eos: torch.Tensor,
    final_weight: torch.Tensor,
    final_bias: torch.Tensor,
) -> torch.Tensor:
    combined = functional.linear(decoder_hidden, final_weight, final_bias)
    guided = (
        combined[ACTUAL_BATCH:] * (1.0 - CFG_SCALE)
        + combined[:ACTUAL_BATCH] * CFG_SCALE
    )
    logits = guided.reshape(
        ACTUAL_BATCH,
        FRAMES_PER_STEP,
        CODEBOOKS,
        VOCABULARY_SIZE,
    )
    eos_wins = logits[..., AUDIO_EOS_ID] > logits[..., :CODEBOOK_SIZE].amax(
        dim=-1
    )
    sampled_eos = codec_tokens.transpose(1, 2) == AUDIO_EOS_ID
    eos_frames = eos_wins | sampled_eos
    eos_frames = torch.where(
        (unfinished | forbid_eos).reshape(ACTUAL_BATCH, 1, 1),
        torch.zeros_like(eos_frames),
        eos_frames,
    )
    eos_frames = torch.where(
        finished.reshape(ACTUAL_BATCH, 1, 1),
        torch.ones_like(eos_frames),
        eos_frames,
    )
    eos_by_frame = eos_frames.any(dim=2)
    positions = torch.arange(
        FRAMES_PER_STEP,
        dtype=torch.int32,
        device=decoder_hidden.device,
    ).reshape(1, FRAMES_PER_STEP)
    candidates = torch.where(
        eos_by_frame,
        positions,
        torch.full_like(positions, FRAMES_PER_STEP),
    )
    first = candidates.amin(dim=1)
    return torch.where(
        first == FRAMES_PER_STEP,
        torch.full_like(first, -1),
        first,
    )


class _FusedEos(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        decoder_hidden: torch.Tensor,
        codec_tokens: torch.Tensor,
        unfinished: torch.Tensor,
        finished: torch.Tensor,
        forbid_eos: torch.Tensor,
        final_weight: torch.Tensor,
        final_bias: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        if torch.onnx.is_in_onnx_export():
            return torch.full_like(codec_tokens[:, 0, 0], -1, dtype=torch.int32)
        return _reference_eos(
            decoder_hidden,
            codec_tokens,
            unfinished,
            finished,
            forbid_eos,
            final_weight,
            final_bias,
        )

    @staticmethod
    def symbolic(
        graph,
        decoder_hidden,
        codec_tokens,
        unfinished,
        finished,
        forbid_eos,
        final_weight,
        final_bias,
    ):
        output = graph.op(
            "magpie_tts_rt::MagpieLocalAREos",
            decoder_hidden,
            codec_tokens,
            unfinished,
            finished,
            forbid_eos,
            final_weight,
            final_bias,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
        )
        output.setType(
            codec_tokens.type().with_dtype(torch.int32).with_sizes([1])
        )
        return output


class _OracleLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        value: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        if torch.onnx.is_in_onnx_export():
            return torch.zeros_like(value)
        return functional.layer_norm(
            value,
            (MODEL_WIDTH,),
            weight,
            None,
            LAYER_NORM_EPSILON,
        )

    @staticmethod
    def symbolic(graph, value, weight):
        output = graph.op(
            "magpie_tts_rt::MagpieLayerNorm",
            value,
            weight,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
        )
        output.setType(value.type())
        return output


class _OracleGeluTanh(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        value: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        if torch.onnx.is_in_onnx_export():
            return torch.zeros_like(value)
        return functional.gelu(value, approximate="tanh")

    @staticmethod
    def symbolic(graph, value):
        output = graph.op(
            "magpie_tts_rt::MagpieGeluTanh",
            value,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
        )
        output.setType(value.type())
        return output


class _OracleSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        value: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        if torch.onnx.is_in_onnx_export():
            return torch.zeros_like(value)
        return functional.softmax(value, dim=-1)

    @staticmethod
    def symbolic(graph, value):
        output = graph.op(
            "magpie_tts_rt::MagpieSoftmax",
            value,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
        )
        output.setType(value.type())
        return output


def _local_layer_step(
    layer: torch.nn.Module,
    x: torch.Tensor,
    prior_keys: Sequence[torch.Tensor],
    prior_values: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention = layer.self_attention
    normalized = _OracleLayerNorm.apply(x, layer.norm_self.weight)
    qkv = attention.qkv_net(normalized).reshape(
        CFG_BATCH,
        1,
        3,
        LOCAL_HEADS,
        LOCAL_HEAD_WIDTH,
    )
    query, key_step, value_step = qkv.unbind(dim=2)
    keys = torch.cat((*prior_keys, key_step), dim=1)
    values = torch.cat((*prior_values, value_step), dim=1)
    scores = torch.matmul(
        query.transpose(1, 2),
        keys.transpose(1, 2).transpose(2, 3),
    ) * attention.scale
    probabilities = _OracleSoftmax.apply(scores)
    attended = torch.matmul(probabilities, values.transpose(1, 2))
    attended = attended.transpose(1, 2).contiguous().reshape(
        CFG_BATCH,
        1,
        MODEL_WIDTH,
    )
    x = x + attention.o_net(attended)

    feed_forward = layer.pos_ff
    normalized_ff = _OracleLayerNorm.apply(x, layer.norm_pos_ff.weight)
    projected = feed_forward.proj.conv(normalized_ff.transpose(1, 2))
    projected = _OracleGeluTanh.apply(projected)
    feed_forward_output = feed_forward.o_net.conv(projected).transpose(1, 2)
    return x + feed_forward_output, key_step, value_step


class LocalARWrapper(torch.nn.Module):
    """Fixed 16-position Local Transformer, sampling, embedding, and EOS."""

    def __init__(self, model: MagpieTTSModel) -> None:
        super().__init__()
        _require_accepted_model(model)
        helper = model._lt_helper
        self.local_transformer = helper.local_transformer
        self.audio_embeddings = helper.audio_embeddings
        self.audio_in_projection = helper.audio_in_projection
        self.local_transformer_in_projection = (
            helper.local_transformer_in_projection
        )
        self.local_transformer_audio_out_projection = (
            helper.local_transformer_audio_out_projection
        )
        self.local_transformer_out_projections = (
            helper.local_transformer_out_projections
        )
        self.final_weight = model.final_proj.weight
        if model.final_proj.bias is None:
            raise RuntimeError("accepted final projection must have a bias")
        self.final_bias = model.final_proj.bias

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        unfinished: torch.Tensor,
        finished: torch.Tensor,
        forbid_eos: torch.Tensor,
        rng_seed: torch.Tensor,
        rng_counter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_step = self.local_transformer_in_projection(
            decoder_hidden.unsqueeze(1)
        )
        keys: list[list[torch.Tensor]] = [[] for _ in range(LOCAL_LAYERS)]
        values: list[list[torch.Tensor]] = [[] for _ in range(LOCAL_LAYERS)]
        sampled_positions: list[torch.Tensor] = []
        invalid_rows = torch.zeros(
            (ACTUAL_BATCH,),
            dtype=torch.int32,
            device=decoder_hidden.device,
        )
        counter = rng_counter

        for position in range(LOCAL_POSITIONS):
            if self.local_transformer.position_embeddings is None:
                raise RuntimeError("accepted Local AR position embedding is absent")
            x = x_step + self.local_transformer.position_embeddings.weight[
                position : position + 1
            ].unsqueeze(0)
            for layer_index, layer in enumerate(self.local_transformer.layers):
                x, key_step, value_step = _local_layer_step(
                    layer,
                    x,
                    keys[layer_index],
                    values[layer_index],
                )
                keys[layer_index].append(key_step)
                values[layer_index].append(value_step)
            transformer_output = self.local_transformer.norm_out(x)
            projection_input = self.local_transformer_audio_out_projection(
                transformer_output[:, 0, :]
            )
            logits = self.local_transformer_out_projections[position](
                projection_input
            )
            (
                sampled,
                next_embedding,
                counter,
                position_invalid,
            ) = _FusedSampling.apply(
                logits,
                unfinished,
                finished,
                forbid_eos,
                rng_seed,
                counter,
                self.audio_embeddings[position].weight,
            )
            sampled_positions.append(sampled)
            invalid_rows = torch.maximum(invalid_rows, position_invalid)
            if position + 1 < LOCAL_POSITIONS:
                embedded = self.audio_in_projection(next_embedding.unsqueeze(1))
                x_step = self.local_transformer_in_projection(embedded)

        position_major = torch.stack(sampled_positions, dim=1)
        codec_tokens = position_major.reshape(
            ACTUAL_BATCH,
            FRAMES_PER_STEP,
            CODEBOOKS,
        ).permute(0, 2, 1)
        end_frame_index = _FusedEos.apply(
            decoder_hidden,
            codec_tokens,
            unfinished,
            finished,
            forbid_eos,
            self.final_weight,
            self.final_bias,
        )
        return codec_tokens, counter, invalid_rows, end_frame_index


def make_example(
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros((CFG_BATCH, MODEL_WIDTH), device=device, dtype=dtype),
        torch.ones((ACTUAL_BATCH,), device=device, dtype=torch.bool),
        torch.zeros((ACTUAL_BATCH,), device=device, dtype=torch.bool),
        torch.ones((ACTUAL_BATCH,), device=device, dtype=torch.bool),
        torch.full((1,), seed, device=device, dtype=torch.int64),
        torch.zeros((ACTUAL_BATCH,), device=device, dtype=torch.int64),
    )


INPUT_NAMES = [
    "decoder_hidden",
    "unfinished",
    "finished",
    "forbid_eos",
    "rng_seed",
    "rng_counter",
]

OUTPUT_NAMES = [
    "codec_tokens",
    "updated_rng_counter",
    "invalid_rows",
    "end_frame_index",
]
