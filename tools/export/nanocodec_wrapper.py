"""Functional, explicit-state NanoCodec decoder used by the TensorRT exporter.

The accepted NeMo implementation mutates Python dataclass state and reuses
preallocated concatenation buffers.  Neither detail belongs in a serialized
engine.  This module expresses the same computation as a pure tensor
function: every causal history and transposed-convolution overlap is an input
and an output, while concatenation storage remains engine workspace.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as torch_functional


EXPECTED_CODEBOOKS = 8
EXPECTED_STATE_TENSORS = 97
SAMPLES_PER_FRAME = 1024


@dataclass(frozen=True)
class NanoCodecStateSpec:
    logical_name: str
    shape: tuple[int, int, int]

    @property
    def input_binding(self) -> str:
        return f"state_in.{self.logical_name}"

    @property
    def output_binding(self) -> str:
        return f"state_out.{self.logical_name}"

    @property
    def fixture_suffix(self) -> str:
        return self.logical_name.replace("_", "-")


def _history_samples(causal_conv) -> int:
    raw_conv = causal_conv.conv
    if raw_conv.stride != (1,):
        raise RuntimeError(
            f"NanoCodec causal convolution must have stride 1, got {raw_conv.stride}"
        )
    if raw_conv.padding != (0,):
        raise RuntimeError(
            f"NanoCodec causal convolution must have zero raw padding, got {raw_conv.padding}"
        )
    kernel_size = raw_conv.kernel_size[0]
    dilation = raw_conv.dilation[0]
    history_samples = (kernel_size - 1) * dilation
    if int(causal_conv.padding_total) != history_samples:
        raise RuntimeError(
            "NanoCodec causal padding no longer matches its finite history: "
            f"{int(causal_conv.padding_total)} != {history_samples}"
        )
    if causal_conv.extra_pad_mode != "constant":
        raise RuntimeError(
            "NanoCodec causal convolution requires zero constant history, got "
            f"{causal_conv.extra_pad_mode!r}"
        )
    return history_samples


def _overlap_samples(causal_transposed_conv) -> int:
    raw_conv = causal_transposed_conv.conv
    if raw_conv.dilation != (1,):
        raise RuntimeError(
            "NanoCodec transposed convolution must have dilation 1, got "
            f"{raw_conv.dilation}"
        )
    if raw_conv.padding != (0,) or raw_conv.output_padding != (0,):
        raise RuntimeError(
            "NanoCodec transposed convolution must have zero padding and "
            f"output padding, got {raw_conv.padding}/{raw_conv.output_padding}"
        )
    stride = raw_conv.stride[0]
    overlap = raw_conv.kernel_size[0] - stride
    if causal_transposed_conv.padding_left != 0:
        raise RuntimeError(
            "NanoCodec transposed convolution has a future-frame dependency"
        )
    if causal_transposed_conv.padding_right != overlap:
        raise RuntimeError(
            "NanoCodec transposed-convolution trimming changed: "
            f"{causal_transposed_conv.padding_right} != {overlap}"
        )
    if overlap < 0 or overlap > stride:
        raise RuntimeError(
            f"NanoCodec transposed-convolution overlap is unsupported: {overlap}"
        )
    return overlap


def enumerate_persistent_state(decoder) -> tuple[NanoCodecStateSpec, ...]:
    """Enumerate the exact 92 histories and five pending overlaps."""

    if not (
        len(decoder.up_sample_conv_layers)
        == len(decoder.res_layers)
        == len(decoder.up_sample_rates)
        == 5
    ):
        raise RuntimeError("NanoCodec must contain exactly five decoder stages")

    histories: list[NanoCodecStateSpec] = []
    overlaps: list[NanoCodecStateSpec] = []

    pre_channels = decoder.pre_conv.conv.in_channels
    histories.append(
        NanoCodecStateSpec(
            logical_name="pre_conv.input_history",
            shape=(1, pre_channels, _history_samples(decoder.pre_conv)),
        )
    )

    current_channels = decoder.pre_conv.conv.out_channels
    residual_histories: list[NanoCodecStateSpec] = []
    for stage_index, (upsample_conv, residual_layer, configured_rate) in enumerate(
        zip(
            decoder.up_sample_conv_layers,
            decoder.res_layers,
            decoder.up_sample_rates,
        )
    ):
        raw_upsample = upsample_conv.conv
        if raw_upsample.in_channels != current_channels:
            raise RuntimeError(
                f"NanoCodec stage {stage_index} input channels changed"
            )
        if raw_upsample.stride[0] != configured_rate:
            raise RuntimeError(
                f"NanoCodec stage {stage_index} stride/configuration mismatch"
            )
        current_channels = raw_upsample.out_channels
        overlaps.append(
            NanoCodecStateSpec(
                logical_name=f"upsample_convs.{stage_index}.pending_overlap",
                shape=(
                    1,
                    current_channels,
                    _overlap_samples(upsample_conv),
                ),
            )
        )
        if len(residual_layer.res_blocks) != 3:
            raise RuntimeError(
                f"NanoCodec stage {stage_index} must contain three residual branches"
            )
        for branch_index, branch in enumerate(residual_layer.res_blocks):
            if len(branch.res_blocks) != 3:
                raise RuntimeError(
                    "NanoCodec residual branch must contain three blocks: "
                    f"stage={stage_index}, branch={branch_index}"
                )
            for block_index, residual_block in enumerate(branch.res_blocks):
                prefix = (
                    f"residual_layers.{stage_index}.branches.{branch_index}."
                    f"residual_blocks.{block_index}"
                )
                for convolution_name, convolution in (
                    ("input_conv", residual_block.input_conv),
                    ("skip_conv", residual_block.skip_conv),
                ):
                    if convolution.conv.in_channels != current_channels:
                        raise RuntimeError(
                            f"NanoCodec state channel count changed at {prefix}"
                        )
                    residual_histories.append(
                        NanoCodecStateSpec(
                            logical_name=(
                                f"{prefix}.{convolution_name}.input_history"
                            ),
                            shape=(
                                1,
                                current_channels,
                                _history_samples(convolution),
                            ),
                        )
                    )

    histories.extend(overlaps)
    histories.extend(residual_histories)
    histories.append(
        NanoCodecStateSpec(
            logical_name="post_conv.input_history",
            shape=(
                1,
                decoder.post_conv.conv.in_channels,
                _history_samples(decoder.post_conv),
            ),
        )
    )
    result = tuple(histories)
    if len(result) != EXPECTED_STATE_TENSORS:
        raise RuntimeError(
            "NanoCodec persistent state count changed: "
            f"{len(result)} != {EXPECTED_STATE_TENSORS}"
        )
    logical_names = tuple(spec.logical_name for spec in result)
    if len(set(logical_names)) != len(logical_names):
        raise RuntimeError("NanoCodec persistent state names are not unique")
    if any(
        dimension < 1
        for spec in result
        for dimension in spec.shape
    ):
        raise RuntimeError("NanoCodec persistent state contains an empty tensor")
    return result


def build_codebook_table(vector_quantizer) -> torch.Tensor:
    """Materialize the locked FSQ codebooks as one exact FP32 gather table."""

    fsqs = getattr(vector_quantizer, "fsqs", None)
    if fsqs is None or len(fsqs) != EXPECTED_CODEBOOKS:
        raise RuntimeError(
            "NanoCodec must expose eight finite-scalar codebooks"
        )
    tables: list[torch.Tensor] = []
    codebook_size: int | None = None
    codebook_dimension: int | None = None
    with torch.inference_mode():
        for index, fsq in enumerate(fsqs):
            table = fsq.codes.detach()
            if table.dtype != torch.float32 or table.ndim != 2:
                raise RuntimeError(
                    f"NanoCodec FSQ table {index} must be rank-2 FP32, "
                    f"got {table.dtype}/{tuple(table.shape)}"
                )
            if codebook_size is None:
                codebook_size, codebook_dimension = table.shape
            elif table.shape != (codebook_size, codebook_dimension):
                raise RuntimeError(
                    "NanoCodec FSQ codebook shapes are not homogeneous"
                )
            tables.append(table)
    result = torch.stack(tables, dim=0).contiguous()
    if result.device.type != "cuda":
        raise RuntimeError(
            f"NanoCodec codebook table must be on CUDA, got {result.device}"
        )
    return result


class ExplicitStateNanoCodec(torch.nn.Module):
    """Decode only new frames and return the complete replacement state."""

    def __init__(
        self,
        decoder,
        codebook_table: torch.Tensor,
        state_specs: tuple[NanoCodecStateSpec, ...],
        *,
        initial: bool,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.register_buffer("codebook_table", codebook_table)
        self.state_specs = state_specs
        self.initial = initial
        self._state_index = {
            spec.logical_name: index for index, spec in enumerate(state_specs)
        }
        if len(self._state_index) != EXPECTED_STATE_TENSORS:
            raise RuntimeError("NanoCodec state index is incomplete")

    def _dequantize(self, codec_tokens: torch.Tensor) -> torch.Tensor:
        groups: list[torch.Tensor] = []
        for codebook_index in range(EXPECTED_CODEBOOKS):
            group = torch_functional.embedding(
                codec_tokens[:, codebook_index, :],
                self.codebook_table[codebook_index],
            )
            groups.append(group.permute(0, 2, 1))
        return torch.cat(groups, dim=1)

    @staticmethod
    def _causal_conv(
        convolution,
        inputs: torch.Tensor,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_samples = _history_samples(convolution)
        if history_samples == 0:
            convolution_inputs = inputs
            updated_history = history
        else:
            convolution_inputs = torch.cat((history, inputs), dim=-1)
            updated_history = convolution_inputs[
                :, :, -history_samples:
            ].reshape(
                1,
                convolution.conv.in_channels,
                history_samples,
            )
        outputs = convolution.conv(convolution_inputs)
        return convolution.activation(outputs), updated_history

    @staticmethod
    def _causal_transposed_conv(
        convolution,
        inputs: torch.Tensor,
        pending_overlap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stride = convolution.conv.stride[0]
        overlap_samples = _overlap_samples(convolution)
        raw_outputs = convolution.conv(inputs)
        finalized_samples = inputs.shape[-1] * stride
        if overlap_samples > 0:
            finalized = torch.cat(
                (
                    raw_outputs[:, :, :overlap_samples] + pending_overlap,
                    raw_outputs[:, :, overlap_samples:finalized_samples],
                ),
                dim=-1,
            )
            updated_overlap = raw_outputs[
                :, :, finalized_samples:
            ].reshape(
                1,
                convolution.conv.out_channels,
                overlap_samples,
            )
            if convolution.conv.bias is not None:
                updated_overlap = updated_overlap - convolution.conv.bias.view(
                    1, -1, 1
                )
        else:
            finalized = raw_outputs[:, :, :finalized_samples]
            updated_overlap = pending_overlap
        return convolution.activation(finalized), updated_overlap

    def _state(
        self,
        states: tuple[torch.Tensor, ...],
        logical_name: str,
    ) -> torch.Tensor:
        return states[self._state_index[logical_name]]

    def forward(
        self,
        codec_tokens: torch.Tensor,
        *state_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        dequantized = self._dequantize(codec_tokens)
        if self.initial:
            if state_inputs:
                raise RuntimeError("initial NanoCodec route does not accept state")
            states = tuple(
                dequantized.new_zeros(spec.shape) for spec in self.state_specs
            )
        else:
            if len(state_inputs) != len(self.state_specs):
                raise RuntimeError(
                    "stateful NanoCodec route received "
                    f"{len(state_inputs)} states, expected {len(self.state_specs)}"
                )
            states = state_inputs

        updated: dict[str, torch.Tensor] = {}
        outputs, new_history = self._causal_conv(
            self.decoder.pre_conv,
            dequantized,
            self._state(states, "pre_conv.input_history"),
        )
        updated["pre_conv.input_history"] = new_history

        for stage_index, (
            activation,
            upsample_conv,
            residual_layer,
            configured_rate,
        ) in enumerate(
            zip(
                self.decoder.activations,
                self.decoder.up_sample_conv_layers,
                self.decoder.res_layers,
                self.decoder.up_sample_rates,
            )
        ):
            outputs = activation(outputs)
            overlap_name = (
                f"upsample_convs.{stage_index}.pending_overlap"
            )
            outputs, new_overlap = self._causal_transposed_conv(
                upsample_conv,
                outputs,
                self._state(states, overlap_name),
            )
            updated[overlap_name] = new_overlap
            if outputs.shape[-1] % configured_rate != 0:
                raise RuntimeError(
                    f"NanoCodec stage {stage_index} emitted an invalid length"
                )

            branch_outputs: list[torch.Tensor] = []
            for branch_index, branch in enumerate(residual_layer.res_blocks):
                branch_output = outputs
                for block_index, residual_block in enumerate(
                    branch.res_blocks
                ):
                    prefix = (
                        f"residual_layers.{stage_index}.branches.{branch_index}."
                        f"residual_blocks.{block_index}"
                    )
                    residual = residual_block.input_activation(branch_output)
                    input_name = f"{prefix}.input_conv.input_history"
                    residual, new_input_history = self._causal_conv(
                        residual_block.input_conv,
                        residual,
                        self._state(states, input_name),
                    )
                    updated[input_name] = new_input_history
                    residual = residual_block.skip_activation(residual)
                    skip_name = f"{prefix}.skip_conv.input_history"
                    residual, new_skip_history = self._causal_conv(
                        residual_block.skip_conv,
                        residual,
                        self._state(states, skip_name),
                    )
                    updated[skip_name] = new_skip_history
                    residual = residual_block.dropout(residual)
                    branch_output = branch_output + residual
                branch_outputs.append(branch_output)
            outputs = sum(branch_outputs) / len(branch_outputs)

        outputs = self.decoder.post_activation(outputs)
        outputs, new_post_history = self._causal_conv(
            self.decoder.post_conv,
            outputs,
            self._state(states, "post_conv.input_history"),
        )
        updated["post_conv.input_history"] = new_post_history
        pcm = self.decoder.out_activation(outputs)[:, 0, :]
        valid_sample_length = (
            codec_tokens.new_ones((1,))
            * codec_tokens.shape[-1]
            * SAMPLES_PER_FRAME
        )
        return (
            pcm,
            valid_sample_length,
            *(updated[spec.logical_name] for spec in self.state_specs),
        )


def initial_input_names() -> list[str]:
    return ["codec_tokens"]


def stateful_input_names(
    state_specs: tuple[NanoCodecStateSpec, ...],
) -> list[str]:
    return ["codec_tokens", *(spec.input_binding for spec in state_specs)]


def output_names(
    state_specs: tuple[NanoCodecStateSpec, ...],
) -> list[str]:
    return [
        "pcm",
        "valid_sample_length",
        *(spec.output_binding for spec in state_specs),
    ]
