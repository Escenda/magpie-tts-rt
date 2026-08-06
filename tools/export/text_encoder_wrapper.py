"""Oracle-math Text Encoder graph for the locked v2607 model.

TensorRT's native BF16 LayerNorm, GELU, and Softmax implementations are not
bit-stable enough to preserve the sampled codec sequence.  This wrapper keeps
the accepted NeMo weights and graph order, but lowers those three operations
to the MagpieTTS-RT plugin contract.  The plugin accepts only the predeclared
Text Encoder and Local AR shapes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional


MODEL_WIDTH = 768
TEXT_ENCODER_LAYERS = 6
TEXT_ENCODER_HEADS = 12
TEXT_ENCODER_HEAD_WIDTH = 64
TEXT_ENCODER_FFN_WIDTH = 3072
TEXT_ENCODER_CONV_KERNEL = 3
TEXT_ENCODER_OUTPUT_PROJECTION_WIDTH = (
    TEXT_ENCODER_FFN_WIDTH * TEXT_ENCODER_CONV_KERNEL
)
LAYER_NORM_EPSILON = 1.0e-5
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "magpie_tts_rt"


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
            mode_i=0,
            plugin_version_s=PLUGIN_VERSION,
            plugin_namespace_s=PLUGIN_NAMESPACE,
        )
        output.setType(value.type())
        return output


def _int64_constant(graph, values: tuple[int, ...]):
    return graph.op(
        "Constant",
        value_t=torch.tensor(values, dtype=torch.int64),
    )


class _OracleCausalOutputProjection(torch.autograd.Function):
    """Keep PyTorch Conv1d math while lowering TensorRT to exact im2col GEMM."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        value: torch.Tensor,
        weight: torch.Tensor,
        numeric_text_mask: torch.Tensor,
    ) -> torch.Tensor:
        del ctx
        expanded_mask = numeric_text_mask.unsqueeze(1)
        masked = value * expanded_mask
        padded = functional.pad(
            masked,
            (TEXT_ENCODER_CONV_KERNEL - 1, 0),
        )
        projected = functional.conv1d(
            padded,
            weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        return projected * expanded_mask

    @staticmethod
    def symbolic(graph, value, weight, numeric_text_mask):
        expanded_mask = graph.op(
            "Unsqueeze",
            numeric_text_mask,
            _int64_constant(graph, (1,)),
        )
        masked = graph.op("Mul", value, expanded_mask)
        padded = graph.op(
            "Pad",
            masked,
            _int64_constant(graph, (0, 0, 2, 0, 0, 0)),
            mode_s="constant",
        )
        windows = []
        slice_ends = (-2, -1, 2**63 - 1)
        for offset, end in enumerate(slice_ends):
            window = graph.op(
                "Slice",
                padded,
                _int64_constant(graph, (offset,)),
                _int64_constant(graph, (end,)),
                _int64_constant(graph, (2,)),
                _int64_constant(graph, (1,)),
            )
            windows.append(
                graph.op(
                    "Transpose",
                    window,
                    perm_i=(0, 2, 1),
                )
            )
        im2col = graph.op("Concat", *windows, axis_i=2)
        flattened_weight = graph.op(
            "Reshape",
            graph.op(
                "Transpose",
                weight,
                perm_i=(2, 1, 0),
            ),
            _int64_constant(
                graph,
                (
                    TEXT_ENCODER_OUTPUT_PROJECTION_WIDTH,
                    MODEL_WIDTH,
                ),
            ),
        )
        projected = graph.op("MatMul", im2col, flattened_weight)
        channel_first = graph.op(
            "Transpose",
            projected,
            perm_i=(0, 2, 1),
        )
        return graph.op("Mul", channel_first, expanded_mask)


def _require_locked_encoder(encoder: torch.nn.Module) -> None:
    if len(encoder.layers) != TEXT_ENCODER_LAYERS:
        raise RuntimeError(
            f"expected {TEXT_ENCODER_LAYERS} Text Encoder layers, "
            f"got {len(encoder.layers)}"
        )
    if encoder.use_moe:
        raise RuntimeError("the locked Text Encoder must not use MoE")
    if not encoder.use_learnable_pos_emb:
        raise RuntimeError("the locked Text Encoder position embedding is disabled")
    if not isinstance(encoder.position_embeddings, torch.nn.Embedding):
        raise RuntimeError("the locked Text Encoder position embedding is absent")
    if not isinstance(encoder.norm_out, torch.nn.LayerNorm):
        raise RuntimeError("the locked Text Encoder final LayerNorm is absent")
    if tuple(encoder.norm_out.normalized_shape) != (MODEL_WIDTH,):
        raise RuntimeError(
            "Text Encoder final LayerNorm width mismatch: "
            f"{tuple(encoder.norm_out.normalized_shape)}"
        )
    if encoder.norm_out.bias is not None:
        raise RuntimeError("Text Encoder final LayerNorm bias must be absent")

    for layer_index, layer in enumerate(encoder.layers):
        attention = layer.self_attention
        feed_forward = layer.pos_ff
        failures: list[str] = []
        if layer.has_xattn:
            failures.append("cross-attention is enabled")
        if layer.use_moe:
            failures.append("MoE is enabled")
        if attention.n_heads != TEXT_ENCODER_HEADS:
            failures.append(f"heads={attention.n_heads}")
        if attention.d_head != TEXT_ENCODER_HEAD_WIDTH:
            failures.append(f"head_width={attention.d_head}")
        if not attention.is_causal:
            failures.append("self-attention is non-causal")
        if attention.use_cache:
            failures.append("module-owned attention cache is enabled")
        if tuple(layer.norm_self.normalized_shape) != (MODEL_WIDTH,):
            failures.append(
                f"self_norm_shape={tuple(layer.norm_self.normalized_shape)}"
            )
        if tuple(layer.norm_pos_ff.normalized_shape) != (MODEL_WIDTH,):
            failures.append(
                f"ffn_norm_shape={tuple(layer.norm_pos_ff.normalized_shape)}"
            )
        if layer.norm_self.bias is not None or layer.norm_pos_ff.bias is not None:
            failures.append("LayerNorm bias is present")
        if feed_forward.proj.conv.in_channels != MODEL_WIDTH:
            failures.append(
                f"ffn_input_width={feed_forward.proj.conv.in_channels}"
            )
        if feed_forward.proj.conv.out_channels != TEXT_ENCODER_FFN_WIDTH:
            failures.append(
                f"ffn_width={feed_forward.proj.conv.out_channels}"
            )
        if (
            feed_forward.proj.kernel_size != TEXT_ENCODER_CONV_KERNEL
            or not feed_forward.proj.is_causal
        ):
            failures.append(
                "input FFN convolution is not locked causal kernel-3"
            )
        if (
            feed_forward.o_net.kernel_size != TEXT_ENCODER_CONV_KERNEL
            or not feed_forward.o_net.is_causal
        ):
            failures.append(
                "output FFN convolution is not locked causal kernel-3"
            )
        if (
            tuple(feed_forward.proj.conv.weight.shape)
            != (
                TEXT_ENCODER_FFN_WIDTH,
                MODEL_WIDTH,
                TEXT_ENCODER_CONV_KERNEL,
            )
            or feed_forward.proj.conv.bias is not None
        ):
            failures.append("input FFN convolution weight contract mismatch")
        if (
            tuple(feed_forward.o_net.conv.weight.shape)
            != (
                MODEL_WIDTH,
                TEXT_ENCODER_FFN_WIDTH,
                TEXT_ENCODER_CONV_KERNEL,
            )
            or feed_forward.o_net.conv.bias is not None
        ):
            failures.append("output FFN convolution weight contract mismatch")
        if (
            feed_forward.o_net.conv.stride != (1,)
            or feed_forward.o_net.conv.padding != (0,)
            or feed_forward.o_net.conv.dilation != (1,)
            or feed_forward.o_net.conv.groups != 1
            or feed_forward.o_net.causal_padding
            != (TEXT_ENCODER_CONV_KERNEL - 1, 0)
        ):
            failures.append("output FFN convolution execution contract mismatch")
        if failures:
            raise RuntimeError(
                f"Text Encoder layer {layer_index} contract mismatch: "
                + "; ".join(failures)
            )


def _encoder_layer(
    layer: torch.nn.Module,
    value: torch.Tensor,
    text_mask: torch.Tensor,
    numeric_text_mask: torch.Tensor,
) -> torch.Tensor:
    numeric_mask = numeric_text_mask.unsqueeze(-1)
    value = value * numeric_mask

    attention = layer.self_attention
    normalized = _OracleLayerNorm.apply(value, layer.norm_self.weight)
    batch, text_tokens, _ = normalized.shape
    qkv = attention.qkv_net(normalized).reshape(
        batch,
        text_tokens,
        3,
        TEXT_ENCODER_HEADS,
        TEXT_ENCODER_HEAD_WIDTH,
    )
    query, key, attended_value = qkv.unbind(dim=2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    attended_value = attended_value.transpose(1, 2)
    scores = torch.matmul(query, key.transpose(2, 3)) * attention.scale
    valid = (
        text_mask[:, None, :, None]
        & text_mask[:, None, None, :]
        & attention.causal_mask[..., :text_tokens, :text_tokens].bool()
    )
    scores = torch.where(
        valid,
        scores,
        torch.full_like(scores, float("-inf")),
    )
    probabilities = _OracleSoftmax.apply(scores)
    probabilities = torch.where(
        valid,
        probabilities,
        torch.zeros_like(probabilities),
    )
    probabilities = attention.dropout(probabilities)
    attended = torch.matmul(probabilities, attended_value)
    attended = attended.transpose(1, 2).contiguous().reshape(
        batch,
        text_tokens,
        MODEL_WIDTH,
    )
    value = value + attention.dropout(attention.o_net(attended))

    feed_forward = layer.pos_ff
    normalized_ff = _OracleLayerNorm.apply(value, layer.norm_pos_ff.weight)
    projected = feed_forward.proj(
        normalized_ff.transpose(1, 2),
        numeric_text_mask,
    )
    projected = _OracleGeluTanh.apply(projected)
    feed_forward_output = _OracleCausalOutputProjection.apply(
        projected,
        feed_forward.o_net.conv.weight,
        numeric_text_mask,
    ).transpose(
        1,
        2,
    )
    value = value + feed_forward.dropout(feed_forward_output)
    return value * numeric_mask


class TextEncoderWrapper(torch.nn.Module):
    """Locked six-layer Text Encoder with explicit oracle-math plugin nodes."""

    def __init__(
        self,
        text_embedding: torch.nn.Embedding,
        encoder: torch.nn.Module,
    ) -> None:
        super().__init__()
        _require_locked_encoder(encoder)
        self.text_embedding = text_embedding
        self.layers = encoder.layers
        self.position_embeddings = encoder.position_embeddings
        self.norm_out_weight = encoder.norm_out.weight
        self.dropout = encoder.dropout
        self.dropout_out = encoder.dropout_out
        self.register_buffer(
            "mask_true",
            torch.ones((), dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "mask_false",
            torch.zeros((), dtype=torch.bfloat16),
            persistent=False,
        )

    def forward(
        self,
        text_token_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        value = self.text_embedding(text_token_ids)
        positions = torch.arange(
            text_token_ids.shape[1],
            device=text_token_ids.device,
        ).unsqueeze(0)
        value = value + self.position_embeddings(positions)
        value = self.dropout(value)
        numeric_text_mask = torch.where(
            text_mask,
            self.mask_true,
            self.mask_false,
        )
        for layer in self.layers:
            value = _encoder_layer(
                layer,
                value,
                text_mask,
                numeric_text_mask,
            )
        value = _OracleLayerNorm.apply(value, self.norm_out_weight)
        value = self.dropout_out(value)
        return value.reshape(
            1,
            text_token_ids.shape[1],
            MODEL_WIDTH,
        )
