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

"""Exact stateful decode for causal HiFi-GAN audio codecs.

The decoder is advanced with new codec frames only. Each causal convolution
keeps its own input history and each transposed convolution keeps the
not-yet-final overlap contributed by the previous chunk. No encoded frame or
intermediate layer prefix is recomputed.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator, Optional

import torch

from nemo.collections.tts.modules.audio_codec_modules import (
    CausalConv1dNorm,
    CausalConvTranspose1dNorm,
    CausalHiFiGANDecoder,
    HiFiGANResLayer,
    ResidualBlock,
)


@dataclass
class _CausalConvState:
    input_history: Optional[torch.Tensor] = None
    work_buffer: Optional[torch.Tensor] = None


@dataclass
class _CausalConvTransposeState:
    pending_overlap: Optional[torch.Tensor] = None


@dataclass
class _ResidualBlockState:
    input_conv: _CausalConvState = field(default_factory=_CausalConvState)
    skip_conv: _CausalConvState = field(default_factory=_CausalConvState)


@dataclass
class _HiFiGANBranchState:
    residual_blocks: list[_ResidualBlockState] = field(default_factory=list)


@dataclass
class _HiFiGANResidualLayerState:
    branches: list[_HiFiGANBranchState] = field(default_factory=list)


@dataclass
class _CausalHiFiGANState:
    pre_conv: _CausalConvState = field(default_factory=_CausalConvState)
    upsample_convs: list[_CausalConvTransposeState] = field(default_factory=list)
    residual_layers: list[_HiFiGANResidualLayerState] = field(default_factory=list)
    post_conv: _CausalConvState = field(default_factory=_CausalConvState)


@dataclass
class CausalCodecStreamingState:
    """State owned by one synthesis session.

    A state belongs to exactly one stream shape, device, and dtype. Create a
    fresh state for each synthesis session instead of sharing it across calls.
    """

    decoder: Optional[_CausalHiFiGANState] = None
    batch_size: Optional[int] = None
    codebook_count: Optional[int] = None
    codes_device: Optional[torch.device] = None
    codes_dtype: Optional[torch.dtype] = None


@dataclass(frozen=True)
class _CausalCodecLengthTableToken:
    """Private proof that length tensors came from the exact preallocator."""


_CAUSAL_CODEC_LENGTH_TABLE_TOKEN = _CausalCodecLengthTableToken()


@dataclass(frozen=True)
class CausalCodecPreallocatedLengths:
    """Immutable metadata and persistent tensors for one codec frame count."""

    frame_count: int
    sample_count: int
    converter_frame_lens: torch.Tensor
    decoder_frame_lens: torch.Tensor
    finalized_sample_lens: torch.Tensor
    _token: _CausalCodecLengthTableToken
    _data_ptrs: tuple[int, int, int]
    _versions: tuple[int, int, int]

    def validate(
        self,
        *,
        codes: torch.Tensor,
        samples_per_frame: int,
    ) -> None:
        """Fail closed if semantic metadata or persistent storage changed."""

        if self._token is not _CAUSAL_CODEC_LENGTH_TABLE_TOKEN:
            raise RuntimeError("Codec length tensors were not created by preallocate_causal_codec_lengths")
        if self.frame_count != codes.shape[-1]:
            raise ValueError(
                f"Preallocated codec frame count does not match input: {self.frame_count} != {codes.shape[-1]}"
            )
        expected_sample_count = self.frame_count * samples_per_frame
        if self.sample_count != expected_sample_count:
            raise ValueError(
                "Preallocated codec sample count does not match the decoder ratio: "
                f"{self.sample_count} != {expected_sample_count}"
            )

        tensors = (
            ("converter_frame_lens", self.converter_frame_lens),
            ("decoder_frame_lens", self.decoder_frame_lens),
            ("finalized_sample_lens", self.finalized_sample_lens),
        )
        if tuple(tensor.data_ptr() for _, tensor in tensors) != self._data_ptrs:
            raise RuntimeError("Preallocated codec length tensor storage was replaced")
        if tuple(tensor._version for _, tensor in tensors) != self._versions:
            raise RuntimeError("Preallocated codec length tensor values were modified")
        for name, tensor in tensors:
            if tensor.shape != (codes.shape[0],):
                raise ValueError(
                    f"{name} must have one value per codec batch item: "
                    f"{tuple(tensor.shape)} != {(codes.shape[0],)}"
                )
            if tensor.dtype != torch.long:
                raise TypeError(f"{name} must use torch.int64, got {tensor.dtype}")
            if tensor.device != codes.device:
                raise ValueError(f"{name} must remain on {codes.device}, got {tensor.device}")
            if tensor.requires_grad:
                raise ValueError(f"{name} must not require gradients")


def preallocate_causal_codec_lengths(
    *,
    batch_size: int,
    max_codec_frames: int,
    samples_per_frame: int,
    device: torch.device | str,
) -> tuple[CausalCodecPreallocatedLengths, ...]:
    """Create exact persistent converter, decoder, and PCM lengths."""

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_codec_frames < 1:
        raise ValueError(f"max_codec_frames must be positive, got {max_codec_frames}")
    if samples_per_frame < 1:
        raise ValueError(f"samples_per_frame must be positive, got {samples_per_frame}")
    resolved_device = torch.device(device)
    frame_values = torch.arange(
        1,
        max_codec_frames + 1,
        dtype=torch.long,
        device=resolved_device,
    ).unsqueeze(1)
    converter_values = frame_values.expand(max_codec_frames, batch_size).clone()
    decoder_values = frame_values.expand(max_codec_frames, batch_size).clone()
    finalized_values = decoder_values.clone().mul_(samples_per_frame)

    result = []
    for frame_index in range(max_codec_frames):
        converter_frame_lens = converter_values[frame_index]
        decoder_frame_lens = decoder_values[frame_index]
        finalized_sample_lens = finalized_values[frame_index]
        tensors = (converter_frame_lens, decoder_frame_lens, finalized_sample_lens)
        result.append(
            CausalCodecPreallocatedLengths(
                frame_count=frame_index + 1,
                sample_count=(frame_index + 1) * samples_per_frame,
                converter_frame_lens=converter_frame_lens,
                decoder_frame_lens=decoder_frame_lens,
                finalized_sample_lens=finalized_sample_lens,
                _token=_CAUSAL_CODEC_LENGTH_TABLE_TOKEN,
                _data_ptrs=tuple(tensor.data_ptr() for tensor in tensors),
                _versions=tuple(tensor._version for tensor in tensors),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class WeightNormMaterializationReceipt:
    """Proof that every causal HiFi-GAN convolution was fixed for inference."""

    target_count: int
    target_names: tuple[str, ...]
    weight_shapes: tuple[tuple[int, ...], ...]
    weight_strides: tuple[tuple[int, ...], ...]
    devices: tuple[str, ...]
    dtypes: tuple[str, ...]


type _WeightNormConv = torch.nn.Conv1d | torch.nn.ConvTranspose1d


def _named_causal_hifigan_convolutions(
    decoder: CausalHiFiGANDecoder,
) -> tuple[tuple[str, _WeightNormConv], ...]:
    """Enumerate the exact causal decoder convolutions in execution order."""

    if not isinstance(decoder, CausalHiFiGANDecoder):
        raise TypeError(
            "Weight-norm materialization is only defined for CausalHiFiGANDecoder, " f"got {type(decoder).__name__}"
        )
    if not isinstance(decoder.pre_conv, CausalConv1dNorm) or not isinstance(decoder.post_conv, CausalConv1dNorm):
        raise TypeError("Causal HiFi-GAN pre/post convolutions have an unsupported implementation")
    if not (len(decoder.up_sample_conv_layers) == len(decoder.res_layers) == len(decoder.up_sample_rates)):
        raise ValueError("Causal HiFi-GAN upsampling and residual layer counts do not match")

    targets: list[tuple[str, _WeightNormConv]] = [
        ("pre_conv.conv", decoder.pre_conv.conv),
    ]
    for stage_index, (upsample_conv, residual_layer) in enumerate(
        zip(decoder.up_sample_conv_layers, decoder.res_layers)
    ):
        if not isinstance(upsample_conv, CausalConvTranspose1dNorm):
            raise TypeError(
                "Weight-norm materialization requires causal transposed convolutions, "
                f"got {type(upsample_conv).__name__} at stage {stage_index}"
            )
        if not isinstance(residual_layer, HiFiGANResLayer):
            raise TypeError(
                "Weight-norm materialization requires HiFi-GAN residual layers, "
                f"got {type(residual_layer).__name__} at stage {stage_index}"
            )
        targets.append((f"up_sample_conv_layers.{stage_index}.conv", upsample_conv.conv))
        for branch_index, branch in enumerate(residual_layer.res_blocks):
            for block_index, residual_block in enumerate(branch.res_blocks):
                if not isinstance(residual_block, ResidualBlock):
                    raise TypeError(
                        "Weight-norm materialization requires the causal ResidualBlock implementation, "
                        f"got {type(residual_block).__name__} at stage {stage_index}, "
                        f"branch {branch_index}, block {block_index}"
                    )
                if not isinstance(residual_block.input_conv, CausalConv1dNorm) or not isinstance(
                    residual_block.skip_conv, CausalConv1dNorm
                ):
                    raise TypeError("Weight-norm materialization requires causal residual convolutions")
                block_prefix = f"res_layers.{stage_index}.res_blocks.{branch_index}." f"res_blocks.{block_index}"
                targets.extend(
                    (
                        (f"{block_prefix}.input_conv.conv", residual_block.input_conv.conv),
                        (f"{block_prefix}.skip_conv.conv", residual_block.skip_conv.conv),
                    )
                )
    targets.append(("post_conv.conv", decoder.post_conv.conv))
    return tuple(targets)


def materialize_causal_hifigan_weight_norm_for_inference(
    decoder: CausalHiFiGANDecoder,
    *,
    expected_target_count: int = 97,
) -> WeightNormMaterializationReceipt:
    """Permanently fix every decoder weight-norm value for inference.

    This is a destructive, runtime-only preparation step. Restore the
    checkpoint and move it to its final device/dtype first, then freeze and
    evaluate the decoder. The prepared instance must not be trained or saved
    using the original parametrized checkpoint schema. CUDA Graph capture must
    happen after this function returns.

    All targets are checked before the first mutation. A second call or a
    partially materialized decoder fails closed rather than silently changing
    the remaining layers.
    """

    if expected_target_count < 1:
        raise ValueError(f"expected_target_count must be positive, got {expected_target_count}")
    training_modules = tuple(name for name, module in decoder.named_modules() if module.training)
    if training_modules:
        raise RuntimeError(
            "decoder.eval() is required before inference weight-norm materialization; "
            f"training mode is enabled for {list(training_modules[:8])}"
        )
    trainable_parameters = tuple(name for name, parameter in decoder.named_parameters() if parameter.requires_grad)
    if trainable_parameters:
        raise RuntimeError(
            "The decoder must be frozen before inference weight-norm materialization; "
            f"trainable parameters include {list(trainable_parameters[:8])}"
        )

    targets = _named_causal_hifigan_convolutions(decoder)
    if len(targets) != expected_target_count:
        raise RuntimeError(
            "Causal HiFi-GAN convolution count does not match the explicit inference contract: "
            f"{len(targets)} != {expected_target_count}"
        )
    target_names = tuple(name for name, _ in targets)
    if "post_conv.conv" not in target_names:
        raise RuntimeError("Causal HiFi-GAN post_conv is missing from weight-norm materialization")
    target_identities = tuple(id(conv) for _, conv in targets)
    if len(set(target_identities)) != len(target_identities):
        raise RuntimeError("Causal HiFi-GAN weight-norm targets contain duplicate modules")

    parametrized_targets = tuple(
        name for name, conv in targets if torch.nn.utils.parametrize.is_parametrized(conv, "weight")
    )
    if len(parametrized_targets) != expected_target_count:
        raise RuntimeError(
            "Expected every causal HiFi-GAN convolution to have weight norm before materialization: "
            f"expected {expected_target_count}, found {len(parametrized_targets)}"
        )
    all_parametrized_weights = tuple(
        name
        for name, module in decoder.named_modules()
        if torch.nn.utils.parametrize.is_parametrized(module, "weight")
    )
    if set(all_parametrized_weights) != set(target_names):
        unexpected = sorted(set(all_parametrized_weights) - set(target_names))
        missing = sorted(set(target_names) - set(all_parametrized_weights))
        raise RuntimeError(
            "Decoder weight parametrizations do not match the explicit causal convolution targets: "
            f"unexpected={unexpected}, missing={missing}"
        )

    weight_snapshots = tuple(conv.weight.detach().clone() for _, conv in targets)
    weight_shapes = tuple(tuple(weight.shape) for weight in weight_snapshots)
    weight_strides = tuple(tuple(weight.stride()) for weight in weight_snapshots)
    weight_devices = tuple(weight.device for weight in weight_snapshots)
    weight_dtypes = tuple(weight.dtype for weight in weight_snapshots)

    for _, conv in targets:
        torch.nn.utils.parametrize.remove_parametrizations(
            conv,
            "weight",
            leave_parametrized=True,
        )

    remaining_parametrizations = tuple(
        name
        for name, module in decoder.named_modules()
        if torch.nn.utils.parametrize.is_parametrized(module, "weight")
    )
    if remaining_parametrizations:
        raise RuntimeError(
            "Weight-norm materialization left parametrized decoder weights: " f"{list(remaining_parametrizations)}"
        )
    for (name, conv), expected_weight, expected_shape, expected_stride, expected_device, expected_dtype in zip(
        targets,
        weight_snapshots,
        weight_shapes,
        weight_strides,
        weight_devices,
        weight_dtypes,
    ):
        actual_weight = conv.weight
        if (
            tuple(actual_weight.shape) != expected_shape
            or tuple(actual_weight.stride()) != expected_stride
            or actual_weight.device != expected_device
            or actual_weight.dtype != expected_dtype
        ):
            raise RuntimeError(
                f"Materialized weight metadata changed for {name}: "
                f"shape={tuple(actual_weight.shape)}, stride={tuple(actual_weight.stride())}, "
                f"device={actual_weight.device}, "
                f"dtype={actual_weight.dtype}"
            )
        if not torch.equal(actual_weight, expected_weight):
            raise RuntimeError(f"Materialized weight value changed for {name}")

    return WeightNormMaterializationReceipt(
        target_count=len(targets),
        target_names=target_names,
        weight_shapes=weight_shapes,
        weight_strides=weight_strides,
        devices=tuple(sorted({str(device) for device in weight_devices})),
        dtypes=tuple(sorted({str(dtype) for dtype in weight_dtypes})),
    )


def _causal_conv_history_samples(conv: CausalConv1dNorm) -> int:
    stride = conv.conv.stride[0]
    if stride != 1:
        raise ValueError(f"Streaming decode requires stride-1 causal convolutions, got stride={stride}")
    kernel_size = conv.conv.kernel_size[0]
    dilation = conv.conv.dilation[0]
    expected_padding = (kernel_size - 1) * dilation
    if int(conv.padding_total) != expected_padding:
        raise ValueError(
            "Causal convolution padding does not match its finite past receptive field: "
            f"padding_total={int(conv.padding_total)}, expected={expected_padding}"
        )
    return expected_padding


def _residual_layer_history_samples(layer: HiFiGANResLayer) -> int:
    branch_histories = []
    for branch in layer.res_blocks:
        branch_history = 0
        for residual_block in branch.res_blocks:
            if not isinstance(residual_block.input_conv, CausalConv1dNorm) or not isinstance(
                residual_block.skip_conv, CausalConv1dNorm
            ):
                raise TypeError("Streaming decode requires causal convolutions in every HiFi-GAN residual block")
            branch_history += _causal_conv_history_samples(residual_block.input_conv)
            branch_history += _causal_conv_history_samples(residual_block.skip_conv)
        branch_histories.append(branch_history)
    if not branch_histories:
        raise ValueError("HiFi-GAN residual layer has no branches")
    return max(branch_histories)


def _earliest_transposed_conv_input(output_index: int, conv: CausalConvTranspose1dNorm) -> int:
    stride = conv.conv.stride[0]
    kernel_size = conv.conv.kernel_size[0]
    raw_output_index = output_index + conv.padding_left
    return math.ceil((raw_output_index - (kernel_size - 1)) / stride)


def causal_hifigan_history_frames(decoder: CausalHiFiGANDecoder) -> int:
    """Return the exact number of preceding codec frames needed by ``decoder``.

    Dependency indices are propagated backwards through the actual loaded
    convolution modules for every output phase in one complete upsampling
    period. A phase one period later has the same dependency shifted by one
    codec frame, so this covers every possible PCM sample position.
    """

    if not isinstance(decoder, CausalHiFiGANDecoder):
        raise TypeError(
            "Exact rolling decode is only defined for CausalHiFiGANDecoder, " f"got {type(decoder).__name__}"
        )
    if not isinstance(decoder.pre_conv, CausalConv1dNorm) or not isinstance(decoder.post_conv, CausalConv1dNorm):
        raise TypeError("Streaming decode requires causal pre/post convolutions")
    if not (len(decoder.up_sample_conv_layers) == len(decoder.res_layers) == len(decoder.up_sample_rates)):
        raise ValueError("HiFi-GAN upsampling and residual layer counts do not match")

    samples_per_frame = math.prod(decoder.up_sample_rates)
    earliest_input_by_phase = []
    for output_phase in range(samples_per_frame):
        earliest_index = output_phase - _causal_conv_history_samples(decoder.post_conv)
        for upsample_conv, residual_layer, configured_rate in reversed(
            list(zip(decoder.up_sample_conv_layers, decoder.res_layers, decoder.up_sample_rates))
        ):
            if not isinstance(upsample_conv, CausalConvTranspose1dNorm):
                raise TypeError("Streaming decode requires causal transposed convolutions")
            if not isinstance(residual_layer, HiFiGANResLayer):
                raise TypeError("Streaming decode requires HiFi-GAN residual layers")
            if upsample_conv.padding_left != 0:
                raise ValueError(
                    "Rolling decode requires transposed convolutions with no future-frame dependency; "
                    f"received padding_left={upsample_conv.padding_left}"
                )
            stride = upsample_conv.conv.stride[0]
            if stride != configured_rate:
                raise ValueError(
                    f"Configured upsample rate {configured_rate} does not match convolution stride {stride}"
                )
            expected_padding = upsample_conv.conv.kernel_size[0] - stride
            if upsample_conv.padding_left + upsample_conv.padding_right != expected_padding:
                raise ValueError(
                    "Causal transposed-convolution trimming does not preserve the configured output alignment"
                )

            earliest_index -= _residual_layer_history_samples(residual_layer)
            earliest_index = _earliest_transposed_conv_input(earliest_index, upsample_conv)

        earliest_index -= _causal_conv_history_samples(decoder.pre_conv)
        earliest_input_by_phase.append(earliest_index)

    return max(0, -min(earliest_input_by_phase))


def _new_hifigan_state(decoder: CausalHiFiGANDecoder) -> _CausalHiFiGANState:
    residual_layers = []
    for residual_layer in decoder.res_layers:
        branches = []
        for branch in residual_layer.res_blocks:
            branches.append(
                _HiFiGANBranchState(
                    residual_blocks=[_ResidualBlockState() for _ in branch.res_blocks],
                )
            )
        residual_layers.append(_HiFiGANResidualLayerState(branches=branches))
    return _CausalHiFiGANState(
        upsample_convs=[_CausalConvTransposeState() for _ in decoder.up_sample_conv_layers],
        residual_layers=residual_layers,
    )


def _validate_causal_conv(conv: CausalConv1dNorm) -> int:
    history_samples = _causal_conv_history_samples(conv)
    if conv.extra_pad_mode != "constant":
        raise ValueError(
            "Exact stateful decode requires zero-valued causal padding; "
            f"received extra_pad_mode={conv.extra_pad_mode!r}"
        )
    if conv.conv.padding[0] != 0:
        raise ValueError(f"Exact stateful decode requires convolution padding=0, got {conv.conv.padding[0]}")
    return history_samples


def _stateful_causal_conv(
    conv: CausalConv1dNorm,
    inputs: torch.Tensor,
    state: _CausalConvState,
    *,
    structure_prevalidated: bool = False,
) -> torch.Tensor:
    """Run a stride-one causal convolution on new samples only."""

    if structure_prevalidated:
        kernel_size = conv.conv.kernel_size[0]
        dilation = conv.conv.dilation[0]
        history_samples = (kernel_size - 1) * dilation
    else:
        history_samples = _validate_causal_conv(conv)
    history = state.input_history
    if history is None:
        history = inputs.new_zeros(inputs.shape[0], inputs.shape[1], history_samples)
        state.input_history = history
    elif (
        history.shape != (inputs.shape[0], inputs.shape[1], history_samples)
        or history.device != inputs.device
        or history.dtype != inputs.dtype
    ):
        raise ValueError("Causal convolution state shape, device, or dtype changed within a synthesis session")

    if history_samples == 0:
        conv_inputs = inputs
    else:
        required_shape = (inputs.shape[0], inputs.shape[1], history_samples + inputs.shape[-1])
        work_buffer = state.work_buffer
        if (
            work_buffer is None
            or work_buffer.shape[:2] != required_shape[:2]
            or work_buffer.shape[-1] < required_shape[-1]
            or work_buffer.device != inputs.device
            or work_buffer.dtype != inputs.dtype
        ):
            work_buffer = inputs.new_empty(required_shape)
            state.work_buffer = work_buffer
        # Native CUDA Graph routes share work buffers sized for eight frames.
        # Each route captures its exact prefix view without replacing storage.
        conv_inputs = work_buffer[:, :, : required_shape[-1]]
        conv_inputs[:, :, :history_samples].copy_(history)
        conv_inputs[:, :, history_samples:].copy_(inputs)
        history.copy_(conv_inputs[:, :, -history_samples:])

    outputs = conv.conv(conv_inputs)
    if outputs.shape[-1] != inputs.shape[-1]:
        raise RuntimeError(
            "Stateful causal convolution returned an unexpected number of samples: "
            f"{outputs.shape[-1]} != {inputs.shape[-1]}"
        )
    return conv.activation(outputs)


def _validate_causal_transposed_conv(conv: CausalConvTranspose1dNorm) -> tuple[int, int]:
    raw_conv = conv.conv
    stride = raw_conv.stride[0]
    kernel_size = raw_conv.kernel_size[0]
    overlap_samples = kernel_size - stride
    if raw_conv.dilation[0] != 1:
        raise ValueError(f"Exact stateful transposed convolution requires dilation=1, got {raw_conv.dilation[0]}")
    if raw_conv.padding[0] != 0 or raw_conv.output_padding[0] != 0:
        raise ValueError(
            "Exact stateful transposed convolution requires padding=0 and output_padding=0, "
            f"got padding={raw_conv.padding[0]}, output_padding={raw_conv.output_padding[0]}"
        )
    if conv.padding_left != 0 or conv.padding_right != overlap_samples:
        raise ValueError(
            "Exact stateful transposed convolution requires all overlap to be trimmed from the right: "
            f"padding_left={conv.padding_left}, padding_right={conv.padding_right}, overlap={overlap_samples}"
        )
    if overlap_samples < 0 or overlap_samples > stride:
        raise ValueError(
            "Exact stateful transposed convolution currently requires 0 <= kernel_size - stride <= stride, "
            f"got kernel_size={kernel_size}, stride={stride}"
        )
    return stride, overlap_samples


def _stateful_causal_transposed_conv(
    conv: CausalConvTranspose1dNorm,
    inputs: torch.Tensor,
    state: _CausalConvTransposeState,
    *,
    structure_prevalidated: bool = False,
) -> torch.Tensor:
    """Finalize the current upsampled chunk while retaining its right overlap."""

    if structure_prevalidated:
        stride = conv.conv.stride[0]
        overlap_samples = conv.conv.kernel_size[0] - stride
    else:
        stride, overlap_samples = _validate_causal_transposed_conv(conv)
    raw_outputs = conv.conv(inputs)
    finalized_samples = inputs.shape[-1] * stride
    expected_raw_samples = finalized_samples + overlap_samples
    if raw_outputs.shape[-1] != expected_raw_samples:
        raise RuntimeError(
            "Stateful transposed convolution returned an unexpected number of samples: "
            f"{raw_outputs.shape[-1]} != {expected_raw_samples}"
        )

    expected_overlap_shape = (raw_outputs.shape[0], raw_outputs.shape[1], overlap_samples)
    pending_overlap = state.pending_overlap
    if pending_overlap is None:
        pending_overlap = raw_outputs.new_zeros(expected_overlap_shape)
        state.pending_overlap = pending_overlap
    elif (
        pending_overlap.shape != expected_overlap_shape
        or pending_overlap.device != raw_outputs.device
        or pending_overlap.dtype != raw_outputs.dtype
    ):
        raise ValueError("Transposed convolution state shape, device, or dtype changed within a synthesis session")

    if overlap_samples > 0:
        raw_outputs[:, :, :overlap_samples].add_(pending_overlap)
        # ConvTranspose1d applies its bias to both the emitted region and the
        # future overlap. Store only the signal contribution so the next
        # chunk's bias is counted exactly once when the two regions are added.
        pending_overlap.copy_(raw_outputs[:, :, finalized_samples:])
        if conv.conv.bias is not None:
            pending_overlap.sub_(conv.conv.bias.view(1, -1, 1))

    return conv.activation(raw_outputs[:, :, :finalized_samples])


def _stateful_residual_layer(
    layer: HiFiGANResLayer,
    inputs: torch.Tensor,
    state: _HiFiGANResidualLayerState,
    *,
    structure_prevalidated: bool = False,
) -> torch.Tensor:
    if len(layer.res_blocks) != len(state.branches):
        raise RuntimeError("HiFi-GAN residual branch count changed after streaming state initialization")

    branch_outputs = []
    for branch, branch_state in zip(layer.res_blocks, state.branches):
        if len(branch.res_blocks) != len(branch_state.residual_blocks):
            raise RuntimeError("HiFi-GAN residual block count changed after streaming state initialization")
        branch_output = inputs
        for residual_block, residual_state in zip(branch.res_blocks, branch_state.residual_blocks):
            if not isinstance(residual_block, ResidualBlock):
                raise TypeError(
                    "Exact stateful decode requires the causal HiFi-GAN ResidualBlock implementation, "
                    f"got {type(residual_block).__name__}"
                )
            if not isinstance(residual_block.input_conv, CausalConv1dNorm) or not isinstance(
                residual_block.skip_conv, CausalConv1dNorm
            ):
                raise TypeError("Exact stateful decode requires causal convolutions in every residual block")
            residual = residual_block.input_activation(branch_output)
            residual = _stateful_causal_conv(
                conv=residual_block.input_conv,
                inputs=residual,
                state=residual_state.input_conv,
                structure_prevalidated=structure_prevalidated,
            )
            residual = residual_block.skip_activation(residual)
            residual = _stateful_causal_conv(
                conv=residual_block.skip_conv,
                inputs=residual,
                state=residual_state.skip_conv,
                structure_prevalidated=structure_prevalidated,
            )
            residual = residual_block.dropout(residual)
            branch_output = branch_output + residual
        branch_outputs.append(branch_output)

    if not branch_outputs:
        raise ValueError("HiFi-GAN residual layer has no branches")
    return sum(branch_outputs) / len(branch_outputs)


def _stateful_hifigan_decode(
    decoder: CausalHiFiGANDecoder,
    inputs: torch.Tensor,
    state: _CausalHiFiGANState,
    *,
    structure_prevalidated: bool = False,
) -> torch.Tensor:
    if not (
        len(decoder.activations)
        == len(decoder.up_sample_conv_layers)
        == len(decoder.res_layers)
        == len(decoder.up_sample_rates)
        == len(state.upsample_convs)
        == len(state.residual_layers)
    ):
        raise RuntimeError("HiFi-GAN decoder structure changed after streaming state initialization")

    outputs = _stateful_causal_conv(
        decoder.pre_conv,
        inputs,
        state.pre_conv,
        structure_prevalidated=structure_prevalidated,
    )
    for activation, upsample_conv, residual_layer, configured_rate, upsample_state, residual_state in zip(
        decoder.activations,
        decoder.up_sample_conv_layers,
        decoder.res_layers,
        decoder.up_sample_rates,
        state.upsample_convs,
        state.residual_layers,
    ):
        outputs = activation(outputs)
        outputs = _stateful_causal_transposed_conv(
            upsample_conv,
            outputs,
            upsample_state,
            structure_prevalidated=structure_prevalidated,
        )
        if outputs.shape[-1] % configured_rate != 0:
            raise RuntimeError("Stateful HiFi-GAN upsampling returned a non-integral configured frame count")
        outputs = _stateful_residual_layer(
            residual_layer,
            outputs,
            residual_state,
            structure_prevalidated=structure_prevalidated,
        )

    outputs = decoder.post_activation(outputs)
    outputs = _stateful_causal_conv(
        decoder.post_conv,
        outputs,
        state.post_conv,
        structure_prevalidated=structure_prevalidated,
    )
    outputs = decoder.out_activation(outputs)
    if outputs.shape[1] != 1:
        raise RuntimeError(f"HiFi-GAN decoder returned {outputs.shape[1]} audio channels instead of one")
    return outputs[:, 0, :]


def _materialize_causal_conv_state(
    conv: CausalConv1dNorm,
    state: _CausalConvState,
    *,
    batch_size: int,
    channels: int,
    samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Allocate every persistent tensor used by one fixed-shape causal convolution."""

    history_samples = _validate_causal_conv(conv)
    if conv.conv.in_channels != channels:
        raise ValueError(
            f"Causal convolution expects {conv.conv.in_channels} channels, "
            f"but fixed-shape materialization received {channels}"
        )
    if state.input_history is not None or state.work_buffer is not None:
        raise RuntimeError("Fixed-shape causal convolution state was already materialized")

    state.input_history = torch.zeros(
        (batch_size, channels, history_samples),
        device=device,
        dtype=dtype,
    )
    if history_samples > 0:
        state.work_buffer = torch.zeros(
            (batch_size, channels, history_samples + samples),
            device=device,
            dtype=dtype,
        )


def _materialize_causal_transposed_conv_state(
    conv: CausalConvTranspose1dNorm,
    state: _CausalConvTransposeState,
    *,
    batch_size: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, int]:
    """Allocate overlap storage and return the fixed output channel/stride pair."""

    stride, overlap_samples = _validate_causal_transposed_conv(conv)
    if conv.conv.in_channels != channels:
        raise ValueError(
            f"Causal transposed convolution expects {conv.conv.in_channels} channels, "
            f"but fixed-shape materialization received {channels}"
        )
    if state.pending_overlap is not None:
        raise RuntimeError("Fixed-shape transposed convolution state was already materialized")

    output_channels = conv.conv.out_channels
    state.pending_overlap = torch.zeros(
        (batch_size, output_channels, overlap_samples),
        device=device,
        dtype=dtype,
    )
    return output_channels, stride


def _materialize_hifigan_state(
    decoder: CausalHiFiGANDecoder,
    state: _CausalHiFiGANState,
    *,
    batch_size: int,
    input_channels: int,
    input_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Materialize all fixed-shape history, work, and overlap buffers before capture."""

    if not (
        len(decoder.up_sample_conv_layers)
        == len(decoder.res_layers)
        == len(state.upsample_convs)
        == len(state.residual_layers)
    ):
        raise RuntimeError("HiFi-GAN decoder structure changed before fixed-shape state materialization")

    channels = input_channels
    samples = input_frames
    _materialize_causal_conv_state(
        decoder.pre_conv,
        state.pre_conv,
        batch_size=batch_size,
        channels=channels,
        samples=samples,
        device=device,
        dtype=dtype,
    )
    channels = decoder.pre_conv.conv.out_channels

    for upsample_conv, residual_layer, upsample_state, residual_state in zip(
        decoder.up_sample_conv_layers,
        decoder.res_layers,
        state.upsample_convs,
        state.residual_layers,
    ):
        channels, stride = _materialize_causal_transposed_conv_state(
            upsample_conv,
            upsample_state,
            batch_size=batch_size,
            channels=channels,
            device=device,
            dtype=dtype,
        )
        samples *= stride
        if len(residual_layer.res_blocks) != len(residual_state.branches):
            raise RuntimeError("HiFi-GAN residual branch count changed before state materialization")

        for branch, branch_state in zip(residual_layer.res_blocks, residual_state.branches):
            if len(branch.res_blocks) != len(branch_state.residual_blocks):
                raise RuntimeError("HiFi-GAN residual block count changed before state materialization")
            for residual_block, block_state in zip(branch.res_blocks, branch_state.residual_blocks):
                if not isinstance(residual_block, ResidualBlock):
                    raise TypeError(
                        "Fixed-shape state materialization requires the causal HiFi-GAN ResidualBlock "
                        f"implementation, got {type(residual_block).__name__}"
                    )
                if not isinstance(residual_block.input_conv, CausalConv1dNorm) or not isinstance(
                    residual_block.skip_conv, CausalConv1dNorm
                ):
                    raise TypeError("Fixed-shape state materialization requires causal residual convolutions")
                _materialize_causal_conv_state(
                    residual_block.input_conv,
                    block_state.input_conv,
                    batch_size=batch_size,
                    channels=channels,
                    samples=samples,
                    device=device,
                    dtype=dtype,
                )
                if residual_block.input_conv.conv.out_channels != channels:
                    raise ValueError("Fixed-shape residual input convolution changed the channel count")
                _materialize_causal_conv_state(
                    residual_block.skip_conv,
                    block_state.skip_conv,
                    batch_size=batch_size,
                    channels=channels,
                    samples=samples,
                    device=device,
                    dtype=dtype,
                )
                if residual_block.skip_conv.conv.out_channels != channels:
                    raise ValueError("Fixed-shape residual skip convolution changed the channel count")

    _materialize_causal_conv_state(
        decoder.post_conv,
        state.post_conv,
        batch_size=batch_size,
        channels=channels,
        samples=samples,
        device=device,
        dtype=dtype,
    )


def _iter_named_causal_conv_state_tensors(
    prefix: str,
    state: _CausalConvState,
) -> Iterator[tuple[str, torch.Tensor]]:
    if state.input_history is not None:
        yield f"{prefix}.input_history", state.input_history
    if state.work_buffer is not None:
        yield f"{prefix}.work_buffer", state.work_buffer


def _iter_named_hifigan_state_tensors(state: _CausalHiFiGANState) -> Iterator[tuple[str, torch.Tensor]]:
    yield from _iter_named_causal_conv_state_tensors("pre_conv", state.pre_conv)
    for stage_index, upsample_state in enumerate(state.upsample_convs):
        if upsample_state.pending_overlap is not None:
            yield f"upsample_convs.{stage_index}.pending_overlap", upsample_state.pending_overlap
    for stage_index, residual_layer in enumerate(state.residual_layers):
        for branch_index, branch in enumerate(residual_layer.branches):
            for block_index, residual_block in enumerate(branch.residual_blocks):
                block_prefix = f"residual_layers.{stage_index}.branches.{branch_index}.residual_blocks.{block_index}"
                yield from _iter_named_causal_conv_state_tensors(
                    f"{block_prefix}.input_conv",
                    residual_block.input_conv,
                )
                yield from _iter_named_causal_conv_state_tensors(
                    f"{block_prefix}.skip_conv",
                    residual_block.skip_conv,
                )
    yield from _iter_named_causal_conv_state_tensors("post_conv", state.post_conv)


def _iter_hifigan_state_tensors(state: _CausalHiFiGANState) -> Iterator[torch.Tensor]:
    for _, tensor in _iter_named_hifigan_state_tensors(state):
        yield tensor


def _reset_hifigan_state(state: _CausalHiFiGANState) -> None:
    for tensor in _iter_hifigan_state_tensors(state):
        tensor.zero_()


class CausalCodecStreamingDecoder:
    """Decode new codec frames with exact per-layer causal state."""

    def __init__(self, codec_model, codec_converter=None):
        if not hasattr(codec_model, "audio_decoder"):
            raise TypeError("Codec model does not expose audio_decoder")
        if not hasattr(codec_model, "samples_per_frame"):
            raise TypeError("Codec model does not expose samples_per_frame")
        if not hasattr(codec_model, "dequantize"):
            raise TypeError("Stateful codec decode requires a frame-local dequantize(tokens, tokens_len) method")

        self.codec_model = codec_model
        self.codec_converter = codec_converter
        self.history_frames = causal_hifigan_history_frames(codec_model.audio_decoder)
        self.samples_per_frame = math.prod(codec_model.audio_decoder.up_sample_rates)
        if self.samples_per_frame != codec_model.samples_per_frame:
            raise ValueError(
                "Codec samples_per_frame does not match the causal decoder upsampling product: "
                f"{codec_model.samples_per_frame} != {self.samples_per_frame}"
            )

    @property
    def input_codebook_count(self) -> int:
        """Return the exact codebook count accepted before optional conversion."""

        if self.codec_converter is not None:
            if not hasattr(self.codec_converter, "vector_quantizer_new"):
                raise TypeError(
                    "Streaming codec converter must expose vector_quantizer_new so input slots can be preallocated"
                )
            codebook_count = self.codec_converter.vector_quantizer_new.num_codebooks
        else:
            if not hasattr(self.codec_model, "num_codebooks"):
                raise TypeError("Streaming codec model must expose num_codebooks so input slots can be preallocated")
            codebook_count = self.codec_model.num_codebooks
        if not isinstance(codebook_count, int) or isinstance(codebook_count, bool) or codebook_count < 1:
            raise TypeError(f"Streaming codec input codebook count must be a positive integer, got {codebook_count!r}")
        return codebook_count

    def _convert_codes(
        self,
        codes: torch.Tensor,
        converter_frame_lens: torch.Tensor,
    ) -> torch.Tensor:
        if self.codec_converter is None:
            return codes
        return self.codec_converter.convert_new_to_original(
            audio_tokens=codes,
            audio_lens=converter_frame_lens,
        )

    @staticmethod
    def _bind_state(codes: torch.Tensor, state: CausalCodecStreamingState) -> None:
        if state.batch_size is None:
            state.batch_size = codes.shape[0]
            state.codebook_count = codes.shape[1]
            state.codes_device = codes.device
            state.codes_dtype = codes.dtype
            return
        if (state.batch_size, state.codebook_count) != codes.shape[:2]:
            raise ValueError("Streaming state batch or codebook shape changed; create a fresh state for a new stream")
        if state.codes_device != codes.device:
            raise ValueError("Streaming state device changed; create a fresh state for a new stream")
        if state.codes_dtype != codes.dtype:
            raise ValueError("Streaming state dtype changed; create a fresh state for a new stream")

    def _decode_converted_codes(
        self,
        converted_codes: torch.Tensor,
        frame_lens: torch.Tensor,
        state: CausalCodecStreamingState,
        *,
        structure_prevalidated: bool = False,
    ) -> torch.Tensor:
        self._bind_state(converted_codes, state)
        if state.decoder is None:
            state.decoder = _new_hifigan_state(self.codec_model.audio_decoder)

        dequantized = self.codec_model.dequantize(tokens=converted_codes, tokens_len=frame_lens)
        if dequantized.ndim != 3 or dequantized.shape[0] != converted_codes.shape[0]:
            raise RuntimeError(
                "Codec dequantizer returned an invalid shape: "
                f"{tuple(dequantized.shape)} for codes {tuple(converted_codes.shape)}"
            )
        if dequantized.shape[-1] != converted_codes.shape[-1]:
            raise RuntimeError(
                "Codec dequantizer changed the number of frames: "
                f"{dequantized.shape[-1]} != {converted_codes.shape[-1]}"
            )
        return _stateful_hifigan_decode(
            decoder=self.codec_model.audio_decoder,
            inputs=dequantized,
            state=state.decoder,
            structure_prevalidated=structure_prevalidated,
        )

    def decode_new(
        self,
        codes: torch.Tensor,
        state: CausalCodecStreamingState,
        *,
        lengths: CausalCodecPreallocatedLengths,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode finalized PCM for new codec frames and advance ``state``.

        Args:
            codes: New frames only, shaped ``[batch, codebooks, frames]``.
            state: Explicit state owned by this synthesis session.
            lengths: Persistent exact lengths for this frame count.

        Returns:
            The finalized PCM suffix and its per-item sample length.
        """

        if codes.ndim != 3:
            raise ValueError(f"Expected [batch, codebooks, frames], got shape={tuple(codes.shape)}")
        if codes.shape[-1] == 0:
            raise ValueError("At least one new codec frame is required")
        lengths.validate(codes=codes, samples_per_frame=self.samples_per_frame)

        converted_codes = self._convert_codes(
            codes,
            converter_frame_lens=lengths.converter_frame_lens,
        )
        lengths.validate(codes=codes, samples_per_frame=self.samples_per_frame)
        if converted_codes.ndim != 3 or converted_codes.shape[0] != codes.shape[0]:
            raise RuntimeError(
                "Codec token converter returned an invalid batch shape: "
                f"{tuple(converted_codes.shape)} for {tuple(codes.shape)}"
            )
        if converted_codes.shape[-1] != lengths.frame_count:
            raise RuntimeError(
                "Codec token converter changed the number of frames: "
                f"{converted_codes.shape[-1]} != {lengths.frame_count}"
            )
        if converted_codes.device != codes.device:
            raise RuntimeError(f"Codec token converter moved tokens from {codes.device} to {converted_codes.device}")
        self.codec_model.eval()
        # NanoCodec is evaluated in its checkpoint dtype. Explicitly disable an
        # outer autocast region so TTS decoder precision cannot silently change it.
        with torch.no_grad(), torch.autocast(device_type=converted_codes.device.type, enabled=False):
            audio = self._decode_converted_codes(
                converted_codes=converted_codes,
                frame_lens=lengths.decoder_frame_lens,
                state=state,
                structure_prevalidated=False,
            )

        if audio.shape[-1] != lengths.sample_count:
            raise RuntimeError(
                "Stateful codec decoder returned an unexpected number of samples: "
                f"{audio.shape[-1]} != {lengths.sample_count}"
            )
        return audio, lengths.finalized_sample_lens

    def create_cuda_graph_runtime(self) -> "CausalCodecStreamingCudaGraphRuntime":
        """Explicitly initialize a reusable native-shape CUDA Graph runtime."""

        return CausalCodecStreamingCudaGraphRuntime(self)


@dataclass(frozen=True)
class _CodecGraphTensorSignature:
    name: str
    tensor_id: int
    data_ptr: int
    storage_data_ptr: int
    storage_offset_bytes: int
    storage_nbytes: int
    version: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _CodecGraphMutableTensorSignature:
    name: str
    tensor_id: int
    data_ptr: int
    storage_data_ptr: int
    storage_offset_bytes: int
    storage_nbytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _CodecCudaGraphCapturePolicyReceipt:
    generic_fp32_precision: str
    cuda_matmul_fp32_precision: str
    cudnn_fp32_precision: str
    cudnn_conv_fp32_precision: str
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    deterministic_algorithms: bool
    deterministic_warn_only: bool


@dataclass(frozen=True)
class _CausalCodecCudaGraphRoute:
    frame_count: int
    static_codes: torch.Tensor
    lengths: CausalCodecPreallocatedLengths
    static_audio: torch.Tensor
    graph: torch.cuda.CUDAGraph


_CODEC_CUDA_GRAPH_CAPTURE_LOCK = Lock()


@contextmanager
def _strict_fp32_codec_cuda_graph_capture_policy() -> Iterator[_CodecCudaGraphCapturePolicyReceipt]:
    """Apply the validated strict-FP32 policy only while graphs are captured.

    These PyTorch controls are process-global. The lock serializes capture
    performed by this module, while the public runtime contract additionally
    requires capture to finish before GPU serving workers are started.
    """

    with _CODEC_CUDA_GRAPH_CAPTURE_LOCK:
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        previous_benchmark = torch.backends.cudnn.benchmark
        previous_cudnn_deterministic = torch.backends.cudnn.deterministic
        previous_generic_precision = torch.backends.fp32_precision
        previous_matmul_precision = torch.backends.cuda.matmul.fp32_precision
        previous_cudnn_precision = torch.backends.cudnn.fp32_precision
        previous_cudnn_conv_precision = torch.backends.cudnn.conv.fp32_precision
        previous = (
            previous_generic_precision,
            previous_matmul_precision,
            previous_cudnn_precision,
            previous_cudnn_conv_precision,
            previous_benchmark,
            previous_cudnn_deterministic,
            previous_deterministic,
            previous_warn_only,
        )
        try:
            torch.use_deterministic_algorithms(False, warn_only=False)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            torch.backends.fp32_precision = "ieee"
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            receipt = _CodecCudaGraphCapturePolicyReceipt(
                generic_fp32_precision=torch.backends.fp32_precision,
                cuda_matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision,
                cudnn_fp32_precision=torch.backends.cudnn.fp32_precision,
                cudnn_conv_fp32_precision=torch.backends.cudnn.conv.fp32_precision,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            )
            expected = _CodecCudaGraphCapturePolicyReceipt(
                generic_fp32_precision="ieee",
                cuda_matmul_fp32_precision="ieee",
                cudnn_fp32_precision="ieee",
                cudnn_conv_fp32_precision="ieee",
                cudnn_benchmark=False,
                cudnn_deterministic=False,
                deterministic_algorithms=False,
                deterministic_warn_only=False,
            )
            if receipt != expected:
                raise RuntimeError(f"Strict FP32 codec CUDA Graph capture policy was not applied: {receipt}")
            yield receipt
        finally:
            torch.backends.cudnn.conv.fp32_precision = previous_cudnn_conv_precision
            torch.backends.cudnn.fp32_precision = previous_cudnn_precision
            torch.backends.cuda.matmul.fp32_precision = previous_matmul_precision
            torch.backends.fp32_precision = previous_generic_precision
            torch.backends.cudnn.deterministic = previous_cudnn_deterministic
            torch.backends.cudnn.benchmark = previous_benchmark
            torch.use_deterministic_algorithms(
                previous_deterministic,
                warn_only=previous_warn_only,
            )
            restored = (
                torch.backends.fp32_precision,
                torch.backends.cuda.matmul.fp32_precision,
                torch.backends.cudnn.fp32_precision,
                torch.backends.cudnn.conv.fp32_precision,
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
            )
            if restored != previous:
                raise RuntimeError(
                    "Process-global CUDA math policy was not restored after codec CUDA Graph capture: "
                    f"{restored} != {previous}"
                )


class CausalCodecStreamingCudaGraphRuntime:
    """Exclusively leased native-shape CUDA Graph routes for one codec stream.

    Routes one through eight own exact-shape input, output, length, and graph
    allocations. They share one maximum-capacity set of causal histories, work
    buffers, and pending overlaps. Every accepted chunk is replayed through its
    native route; this runtime has no eager decode path.

    Each route deliberately uses its own CUDA Graph memory pool. Native route
    order is arbitrary, so the capture/replay ordering requirement of a shared
    graph pool cannot be satisfied safely.
    """

    _BATCH_SIZE = 1
    _CODEBOOK_COUNT = 8
    _MAX_CHUNK_FRAMES = 8
    _ROUTE_FRAME_COUNTS = tuple(range(1, _MAX_CHUNK_FRAMES + 1))
    _SAMPLES_PER_FRAME = 1024
    _CODE_DTYPE = torch.int64
    _CAPTURE_WARMUP_RUNS = 3

    def __init__(self, decoder: CausalCodecStreamingDecoder):
        if decoder.samples_per_frame != self._SAMPLES_PER_FRAME:
            raise ValueError(
                "Native-shape codec CUDA Graph runtime requires samples_per_frame=1024, "
                f"got {decoder.samples_per_frame}"
            )
        self.decoder = decoder
        self.codec_model = decoder.codec_model
        self.audio_decoder = self.codec_model.audio_decoder
        self.codec_converter = decoder.codec_converter
        self.vector_quantizer = getattr(self.codec_model, "vector_quantizer", None)
        self._require_eval_mode()

        if not torch.cuda.is_available():
            raise RuntimeError("Stateful codec CUDA Graph runtime requires CUDA")
        first_parameter = next(self.audio_decoder.parameters(), None)
        if first_parameter is None:
            raise RuntimeError("Stateful codec CUDA Graph runtime requires a parameterized audio decoder")
        self.device = first_parameter.device
        self.decoder_dtype = first_parameter.dtype
        if self.device.type != "cuda":
            raise RuntimeError(
                "Stateful codec CUDA Graph runtime requires the codec on a CUDA device, " f"got {self.device}"
            )
        if self.decoder_dtype != torch.float32:
            raise TypeError(f"Codec CUDA Graph decoder parameters must remain FP32, got {self.decoder_dtype}")
        if decoder.input_codebook_count != self._CODEBOOK_COUNT:
            raise ValueError(
                "Codec CUDA Graph input codebook count does not match the native route contract: "
                f"{decoder.input_codebook_count} != {self._CODEBOOK_COUNT}"
            )
        self._require_fp32_runtime_tensors()

        self._lock = Lock()
        self._ready = False
        self._failed = False
        self._active_lease_id: Optional[int] = None
        self._next_lease_id = 0
        self._capture_policy_receipt: Optional[_CodecCudaGraphCapturePolicyReceipt] = None
        with torch.cuda.device(self.device):
            self.execution_stream = torch.cuda.Stream(device=self.device)
            self._completion_event = torch.cuda.Event(blocking=False, interprocess=False)
            self._routes = self._create_routes()
            self._state = self._create_materialized_state()
            self.parameter_signature = self._read_parameter_signature()
            self._state_signature = self._read_state_signature()
            self._route_mutable_signature = self._read_route_mutable_signature()
            self._route_length_signature = self._read_route_length_signature()
            self._validate_captured_storage_aliases()
            try:
                self._warmup_and_capture_routes()
                self._validate_model_invariants()
            except Exception as error:
                self._failed = True
                raise RuntimeError(
                    "Stateful codec CUDA Graph initialization failed; graph mode has no eager fallback"
                ) from error
        self._ready = True

    @property
    def max_codec_frames(self) -> int:
        return self._MAX_CHUNK_FRAMES

    @property
    def supported_frame_counts(self) -> tuple[int, ...]:
        return self._ROUTE_FRAME_COUNTS

    def acquire(self) -> "CausalCodecStreamingCudaGraphLease":
        """Exclusively lease this pre-captured runtime to one synthesis session.

        Acquisition never waits for another conversation. A pool must provide
        a different pre-captured runtime when concurrent synthesis is needed.
        The causal state is reset before ownership is published.
        """

        with self._lock:
            if self._active_lease_id is not None:
                raise RuntimeError("Stateful codec CUDA Graph runtime is already leased by another synthesis session")
            if not self._ready or self._failed:
                raise RuntimeError("Stateful codec CUDA Graph runtime is not usable")
            try:
                self._validate_model_invariants()
                self._reset_runtime_locked()
            except Exception:
                self._failed = True
                raise
            lease_id = self._next_lease_id
            self._next_lease_id += 1
            self._active_lease_id = lease_id
            return CausalCodecStreamingCudaGraphLease(runtime=self, lease_id=lease_id)

    def _runtime_module_roots(self) -> Iterator[tuple[str, torch.nn.Module]]:
        yield "codec_model", self.codec_model
        if isinstance(self.codec_converter, torch.nn.Module):
            yield "codec_converter", self.codec_converter

    def _runtime_modules(self) -> Iterator[tuple[str, torch.nn.Module]]:
        seen_modules = set()
        for root_name, root in self._runtime_module_roots():
            for child_name, module in root.named_modules():
                if id(module) in seen_modules:
                    continue
                seen_modules.add(id(module))
                qualified_name = root_name if not child_name else f"{root_name}.{child_name}"
                yield qualified_name, module

    def _runtime_named_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        seen_tensors = set()
        for root_name, root in self._runtime_module_roots():
            for tensor_name, tensor in root.named_parameters(recurse=True):
                if id(tensor) in seen_tensors:
                    continue
                seen_tensors.add(id(tensor))
                yield f"{root_name}.{tensor_name}", tensor
            for tensor_name, tensor in root.named_buffers(recurse=True):
                if id(tensor) in seen_tensors:
                    continue
                seen_tensors.add(id(tensor))
                yield f"{root_name}.{tensor_name}", tensor

    def _require_eval_mode(self) -> None:
        training_modules = tuple(name for name, module in self._runtime_modules() if module.training)
        if training_modules:
            raise RuntimeError(
                "codec_model.eval() is required before stateful codec CUDA Graph initialization; "
                f"training mode is enabled for {list(training_modules[:8])}"
            )

    def _require_fp32_runtime_tensors(self) -> None:
        non_fp32_tensors = tuple(
            (name, tensor.dtype)
            for name, tensor in self._runtime_named_tensors()
            if tensor.is_floating_point() and tensor.dtype != torch.float32
        )
        if non_fp32_tensors:
            raise RuntimeError(
                "Codec CUDA Graph runtime requires every floating parameter and buffer to remain FP32: "
                f"{list(non_fp32_tensors[:8])}"
            )

    @staticmethod
    def _immutable_tensor_signature(name: str, tensor: torch.Tensor) -> _CodecGraphTensorSignature:
        return _CodecGraphTensorSignature(
            name=name,
            tensor_id=id(tensor),
            data_ptr=tensor.data_ptr(),
            storage_data_ptr=tensor.untyped_storage().data_ptr(),
            storage_offset_bytes=tensor.storage_offset() * tensor.element_size(),
            storage_nbytes=tensor.untyped_storage().nbytes(),
            version=tensor._version,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=tensor.device,
        )

    @staticmethod
    def _mutable_tensor_signature(name: str, tensor: torch.Tensor) -> _CodecGraphMutableTensorSignature:
        return _CodecGraphMutableTensorSignature(
            name=name,
            tensor_id=id(tensor),
            data_ptr=tensor.data_ptr(),
            storage_data_ptr=tensor.untyped_storage().data_ptr(),
            storage_offset_bytes=tensor.storage_offset() * tensor.element_size(),
            storage_nbytes=tensor.untyped_storage().nbytes(),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=tensor.device,
        )

    def _read_parameter_signature(self) -> tuple[_CodecGraphTensorSignature, ...]:
        return tuple(self._immutable_tensor_signature(name, tensor) for name, tensor in self._runtime_named_tensors())

    def _read_state_signature(self) -> tuple[_CodecGraphMutableTensorSignature, ...]:
        if self._state.decoder is None:
            raise RuntimeError("CUDA Graph runtime lost its materialized codec state")
        return tuple(
            self._mutable_tensor_signature(f"state.{name}", tensor)
            for name, tensor in _iter_named_hifigan_state_tensors(self._state.decoder)
        )

    def _read_route_mutable_signature(self) -> tuple[_CodecGraphMutableTensorSignature, ...]:
        result = []
        for route in self._routes:
            result.extend(
                (
                    self._mutable_tensor_signature(
                        f"routes.{route.frame_count}.static_codes",
                        route.static_codes,
                    ),
                    self._mutable_tensor_signature(
                        f"routes.{route.frame_count}.static_audio",
                        route.static_audio,
                    ),
                )
            )
        return tuple(result)

    def _read_route_length_signature(self) -> tuple[_CodecGraphTensorSignature, ...]:
        result = []
        for route in self._routes:
            result.extend(
                (
                    self._immutable_tensor_signature(
                        f"routes.{route.frame_count}.converter_frame_lens",
                        route.lengths.converter_frame_lens,
                    ),
                    self._immutable_tensor_signature(
                        f"routes.{route.frame_count}.decoder_frame_lens",
                        route.lengths.decoder_frame_lens,
                    ),
                    self._immutable_tensor_signature(
                        f"routes.{route.frame_count}.finalized_sample_lens",
                        route.lengths.finalized_sample_lens,
                    ),
                )
            )
        return tuple(result)

    def _iter_captured_storage_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        if self._state.decoder is None:
            raise RuntimeError("CUDA Graph runtime lost its materialized codec state")
        for name, tensor in _iter_named_hifigan_state_tensors(self._state.decoder):
            yield f"state.{name}", tensor
        for route in self._routes:
            route_prefix = f"routes.{route.frame_count}"
            yield f"{route_prefix}.static_codes", route.static_codes
            yield f"{route_prefix}.static_audio", route.static_audio
            yield f"{route_prefix}.converter_frame_lens", route.lengths.converter_frame_lens
            yield f"{route_prefix}.decoder_frame_lens", route.lengths.decoder_frame_lens
            yield f"{route_prefix}.finalized_sample_lens", route.lengths.finalized_sample_lens

    def _validate_captured_storage_aliases(self) -> None:
        storage_owners: dict[int, str] = {}
        for name, tensor in self._iter_captured_storage_tensors():
            if tensor.numel() == 0:
                continue
            storage_data_ptr = tensor.untyped_storage().data_ptr()
            previous_owner = storage_owners.get(storage_data_ptr)
            if previous_owner is not None:
                raise RuntimeError(
                    "Captured codec state, route I/O, and route lengths must use disjoint storage: "
                    f"{previous_owner} aliases {name}"
                )
            storage_owners[storage_data_ptr] = name

    def _validate_model_invariants(self) -> None:
        if self.decoder.codec_model is not self.codec_model:
            raise RuntimeError("Codec model changed after CUDA Graph capture")
        if self.codec_model.audio_decoder is not self.audio_decoder:
            raise RuntimeError("Codec audio decoder changed after CUDA Graph capture")
        if self.decoder.codec_converter is not self.codec_converter:
            raise RuntimeError("Codec token converter changed after CUDA Graph capture")
        if getattr(self.codec_model, "vector_quantizer", None) is not self.vector_quantizer:
            raise RuntimeError("Codec vector quantizer changed after CUDA Graph capture")
        if self.decoder.samples_per_frame != self._SAMPLES_PER_FRAME:
            raise RuntimeError("Codec samples_per_frame changed after CUDA Graph capture")
        if self.decoder.input_codebook_count != self._CODEBOOK_COUNT:
            raise RuntimeError("Codec input codebook count changed after CUDA Graph capture")
        if tuple(route.frame_count for route in self._routes) != self._ROUTE_FRAME_COUNTS:
            raise RuntimeError("Codec CUDA Graph native route table changed after capture")
        if self._read_parameter_signature() != self.parameter_signature:
            raise RuntimeError(
                "Codec parameters or buffers changed or moved after CUDA Graph capture; "
                "initialize a new graph runtime after loading the final inference model"
            )
        self._require_eval_mode()
        self._require_fp32_runtime_tensors()
        if self._read_state_signature() != self._state_signature:
            raise RuntimeError("Codec causal state storage changed or moved after CUDA Graph capture")
        if self._read_route_mutable_signature() != self._route_mutable_signature:
            raise RuntimeError("Codec native route input or output storage changed after CUDA Graph capture")
        if self._read_route_length_signature() != self._route_length_signature:
            raise RuntimeError("Codec native route length storage or value changed after CUDA Graph capture")
        self._validate_captured_storage_aliases()

    def _create_route_lengths(self, frame_count: int) -> CausalCodecPreallocatedLengths:
        converter_frame_lens = torch.full(
            (self._BATCH_SIZE,),
            frame_count,
            dtype=torch.long,
            device=self.device,
        )
        decoder_frame_lens = torch.full(
            (self._BATCH_SIZE,),
            frame_count,
            dtype=torch.long,
            device=self.device,
        )
        finalized_sample_lens = torch.full(
            (self._BATCH_SIZE,),
            frame_count * self._SAMPLES_PER_FRAME,
            dtype=torch.long,
            device=self.device,
        )
        tensors = (
            converter_frame_lens,
            decoder_frame_lens,
            finalized_sample_lens,
        )
        return CausalCodecPreallocatedLengths(
            frame_count=frame_count,
            sample_count=frame_count * self._SAMPLES_PER_FRAME,
            converter_frame_lens=converter_frame_lens,
            decoder_frame_lens=decoder_frame_lens,
            finalized_sample_lens=finalized_sample_lens,
            _token=_CAUSAL_CODEC_LENGTH_TABLE_TOKEN,
            _data_ptrs=tuple(tensor.data_ptr() for tensor in tensors),
            _versions=tuple(tensor._version for tensor in tensors),
        )

    def _create_routes(self) -> tuple[_CausalCodecCudaGraphRoute, ...]:
        return tuple(
            _CausalCodecCudaGraphRoute(
                frame_count=frame_count,
                static_codes=torch.zeros(
                    (self._BATCH_SIZE, self._CODEBOOK_COUNT, frame_count),
                    dtype=self._CODE_DTYPE,
                    device=self.device,
                ),
                lengths=self._create_route_lengths(frame_count),
                static_audio=torch.zeros(
                    (self._BATCH_SIZE, frame_count * self._SAMPLES_PER_FRAME),
                    dtype=self.decoder_dtype,
                    device=self.device,
                ),
                graph=torch.cuda.CUDAGraph(),
            )
            for frame_count in self._ROUTE_FRAME_COUNTS
        )

    def _create_materialized_state(self) -> CausalCodecStreamingState:
        converted_codebook_count: Optional[int] = None
        converted_dtype: Optional[torch.dtype] = None
        maximum_dequantized: Optional[torch.Tensor] = None
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            for route in self._routes:
                converted_codes = self.decoder._convert_codes(
                    route.static_codes,
                    converter_frame_lens=route.lengths.converter_frame_lens,
                )
                if (
                    converted_codes.ndim != 3
                    or converted_codes.shape[0] != self._BATCH_SIZE
                    or converted_codes.shape[-1] != route.frame_count
                ):
                    raise RuntimeError(
                        "Codec token converter is incompatible with native CUDA Graph input "
                        f"{route.frame_count}: {tuple(converted_codes.shape)}"
                    )
                if converted_codes.device != self.device:
                    raise RuntimeError(
                        f"Codec token converter moved graph input from {self.device} to {converted_codes.device}"
                    )
                if converted_codebook_count is None:
                    converted_codebook_count = converted_codes.shape[1]
                    converted_dtype = converted_codes.dtype
                elif converted_codes.shape[1] != converted_codebook_count or converted_codes.dtype != converted_dtype:
                    raise RuntimeError("Codec token converter changed codebook count or dtype between native routes")
                dequantized = self.codec_model.dequantize(
                    tokens=converted_codes,
                    tokens_len=route.lengths.decoder_frame_lens,
                )
                expected_shape = (
                    self._BATCH_SIZE,
                    self.audio_decoder.pre_conv.conv.in_channels,
                    route.frame_count,
                )
                if dequantized.shape != expected_shape:
                    raise RuntimeError(
                        "Codec dequantizer is incompatible with native CUDA Graph route "
                        f"{route.frame_count}: {tuple(dequantized.shape)} != {expected_shape}"
                    )
                if dequantized.device != self.device or dequantized.dtype != self.decoder_dtype:
                    raise RuntimeError(
                        f"Codec dequantizer must produce {self.device}/{self.decoder_dtype}, "
                        f"got {dequantized.device}/{dequantized.dtype}"
                    )
                if route.frame_count == self._MAX_CHUNK_FRAMES:
                    maximum_dequantized = dequantized
        if converted_codebook_count is None or converted_dtype is None or maximum_dequantized is None:
            raise RuntimeError("Codec native route preflight did not produce the maximum route")
        if maximum_dequantized.shape != (
            self._BATCH_SIZE,
            self.audio_decoder.pre_conv.conv.in_channels,
            self._MAX_CHUNK_FRAMES,
        ):
            raise RuntimeError(
                "Codec dequantizer is incompatible with maximum native CUDA Graph route: "
                f"{tuple(maximum_dequantized.shape)}"
            )

        state = CausalCodecStreamingState(
            decoder=_new_hifigan_state(self.audio_decoder),
            batch_size=self._BATCH_SIZE,
            codebook_count=converted_codebook_count,
            codes_device=self.device,
            codes_dtype=converted_dtype,
        )
        _materialize_hifigan_state(
            decoder=self.audio_decoder,
            state=state.decoder,
            batch_size=self._BATCH_SIZE,
            input_channels=maximum_dequantized.shape[1],
            input_frames=self._MAX_CHUNK_FRAMES,
            device=self.device,
            dtype=self.decoder_dtype,
        )
        return state

    def _reset_runtime_buffers(self) -> None:
        if self._state.decoder is None:
            raise RuntimeError("CUDA Graph runtime lost its materialized codec state")
        for route in self._routes:
            route.static_codes.zero_()
            route.static_audio.zero_()
        _reset_hifigan_state(self._state.decoder)

    def _reset_shared_state(self) -> None:
        if self._state.decoder is None:
            raise RuntimeError("CUDA Graph runtime lost its materialized codec state")
        _reset_hifigan_state(self._state.decoder)

    def _execute_route(self, route: _CausalCodecCudaGraphRoute) -> None:
        audio = self.decoder._decode_converted_codes(
            converted_codes=self.decoder._convert_codes(
                route.static_codes,
                converter_frame_lens=route.lengths.converter_frame_lens,
            ),
            frame_lens=route.lengths.decoder_frame_lens,
            state=self._state,
            structure_prevalidated=True,
        )
        if audio.shape != route.static_audio.shape:
            raise RuntimeError(
                f"Codec graph route {route.frame_count} must produce PCM shape "
                f"{tuple(route.static_audio.shape)}, "
                f"got {tuple(audio.shape)}"
            )
        route.static_audio.copy_(audio)

    def _warmup_and_capture_routes(self) -> None:
        caller_stream = torch.cuda.current_stream(self.device)
        self.execution_stream.wait_stream(caller_stream)
        with _strict_fp32_codec_cuda_graph_capture_policy() as receipt:
            self._capture_policy_receipt = receipt
            for route in self._routes:
                with (
                    torch.cuda.stream(self.execution_stream),
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", enabled=False),
                ):
                    for _ in range(self._CAPTURE_WARMUP_RUNS):
                        self._reset_shared_state()
                        route.static_audio.zero_()
                        self._execute_route(route)
                self.execution_stream.synchronize()

                with torch.cuda.stream(self.execution_stream), torch.inference_mode():
                    self._reset_shared_state()
                    route.static_audio.zero_()
                self.execution_stream.synchronize()
                with (
                    torch.cuda.graph(route.graph, stream=self.execution_stream),
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", enabled=False),
                ):
                    self._execute_route(route)

                # Capture executes once and advances the shared causal state.
                # Reset the same allocations in place before another route is
                # captured; replacing any tensor would invalidate raw pointers.
                with torch.cuda.stream(self.execution_stream), torch.inference_mode():
                    self._reset_shared_state()
                    route.static_audio.zero_()
                self.execution_stream.synchronize()
                self._validate_model_invariants()

            with torch.cuda.stream(self.execution_stream), torch.inference_mode():
                self._reset_runtime_buffers()
            self.execution_stream.synchronize()

        # Runtime replay happens under the caller's ambient math policy. Smoke
        # every route after restoring process-global flags before publishing it.
        with torch.cuda.stream(self.execution_stream), torch.inference_mode():
            for route in self._routes:
                self._reset_shared_state()
                route.graph.replay()
        self.execution_stream.synchronize()
        for route in self._routes:
            if not bool(torch.isfinite(route.static_audio).all().item()):
                raise RuntimeError(f"Codec CUDA Graph route {route.frame_count} produced non-finite PCM")
        with torch.cuda.stream(self.execution_stream), torch.inference_mode():
            self._reset_runtime_buffers()
        self.execution_stream.synchronize()
        caller_stream.wait_stream(self.execution_stream)

    def _validate_input(self, codes: torch.Tensor) -> int:
        if codes.ndim != 3 or codes.shape[:2] != (self._BATCH_SIZE, self._CODEBOOK_COUNT):
            raise ValueError(
                "CUDA Graph codec input shape must be one native route "
                f"(1, 8, 1..{self._MAX_CHUNK_FRAMES}); got {tuple(codes.shape)}"
            )
        frame_count = codes.shape[-1]
        if frame_count not in self._ROUTE_FRAME_COUNTS:
            raise ValueError(
                "CUDA Graph codec input shape must be one native route "
                f"(1, 8, 1..{self._MAX_CHUNK_FRAMES}); got {tuple(codes.shape)}"
            )
        if codes.dtype != self._CODE_DTYPE:
            raise ValueError(f"CUDA Graph codec input must use int64 tokens, got {codes.dtype}")
        if codes.device != self.device:
            raise ValueError(f"CUDA Graph codec input must remain on CUDA device {self.device}, got {codes.device}")
        return frame_count

    def _replay_route(
        self,
        route: _CausalCodecCudaGraphRoute,
        codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay into graph-owned PCM storage.

        The returned audio tensor is borrowed from this runtime and remains
        valid only until the next decode or reset. The asynchronous synthesis
        bridge immediately copies it to a leased pinned host slot before
        another replay can begin.
        """

        caller_stream = torch.cuda.current_stream(self.device)
        self.execution_stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.execution_stream), torch.inference_mode():
            route.static_codes.copy_(codes)
            codes.record_stream(self.execution_stream)
            route.graph.replay()
            self._completion_event.record(self.execution_stream)

        caller_stream.wait_event(self._completion_event)
        route.static_audio.record_stream(caller_stream)
        route.lengths.finalized_sample_lens.record_stream(caller_stream)
        return route.static_audio, route.lengths.finalized_sample_lens

    def decode_new(
        self,
        codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode without a lease for isolated single-session diagnostics.

        Production asynchronous synthesis must use :meth:`acquire`. Direct
        decode is rejected while a lease exists, so it cannot corrupt an
        actively owned conversation. Returned PCM is graph-owned borrowed
        storage and must be consumed or cloned before the next decode/reset.
        """

        frame_count = self._validate_input(codes)
        with self._lock:
            if self._active_lease_id is not None:
                raise RuntimeError("Direct CUDA Graph decode is forbidden while the runtime is leased")
            return self._decode_new_locked(
                codes=codes,
                frame_count=frame_count,
            )

    def _decode_from_lease(
        self,
        *,
        lease_id: int,
        codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_count = self._validate_input(codes)
        with self._lock:
            if self._active_lease_id != lease_id:
                raise RuntimeError("CUDA Graph codec lease is no longer active")
            return self._decode_new_locked(
                codes=codes,
                frame_count=frame_count,
            )

    def _decode_new_locked(
        self,
        *,
        codes: torch.Tensor,
        frame_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._ready or self._failed:
            raise RuntimeError("Stateful codec CUDA Graph runtime is not usable")
        try:
            self._validate_model_invariants()
            route = self._routes[frame_count - 1]
            if route.frame_count != frame_count:
                raise RuntimeError(
                    f"Codec CUDA Graph route table returned {route.frame_count} for input {frame_count}"
                )
            route.lengths.validate(codes=codes, samples_per_frame=self.decoder.samples_per_frame)
            return self._replay_route(route, codes)
        except Exception:
            self._failed = True
            raise

    def _reset_runtime_locked(self) -> None:
        self.execution_stream.synchronize()
        with torch.cuda.stream(self.execution_stream), torch.inference_mode():
            self._reset_runtime_buffers()
        self.execution_stream.synchronize()

    def _release_lease(self, *, lease_id: int) -> None:
        with self._lock:
            if self._active_lease_id != lease_id:
                raise RuntimeError("CUDA Graph codec lease is no longer active")
            try:
                if not self._failed:
                    self._validate_model_invariants()
                    self._reset_runtime_locked()
                else:
                    self.execution_stream.synchronize()
            except Exception:
                self._failed = True
                raise
            finally:
                self._active_lease_id = None

    def reset(self) -> None:
        """Synchronously clear one session's state without replacing graph-owned storage."""

        with self._lock:
            if self._active_lease_id is not None:
                raise RuntimeError("Cannot reset a CUDA Graph codec runtime while it is leased")
            if self._failed:
                raise RuntimeError("A failed CUDA Graph codec runtime cannot be reset or reused")
            try:
                self._validate_model_invariants()
                self._reset_runtime_locked()
            except Exception:
                self._failed = True
                raise

    def state_tensor_pointers(self) -> tuple[int, ...]:
        """Return state allocation pointers for lifecycle and isolation verification."""

        if self._state.decoder is None:
            raise RuntimeError("CUDA Graph runtime lost its materialized codec state")
        return tuple(tensor.data_ptr() for tensor in _iter_hifigan_state_tensors(self._state.decoder))

    def synchronize_before_release(self) -> None:
        """Wait until this runtime's private execution stream is idle."""

        with self._lock:
            try:
                self.execution_stream.synchronize()
            except Exception:
                self._failed = True
                raise


class CausalCodecStreamingCudaGraphLease:
    """Exclusive request-scoped ownership of one pre-captured codec runtime.

    PCM returned by :meth:`decode_new` is borrowed native-route graph output.
    The caller must finish its D2H copy before invoking the next decode.
    """

    def __init__(self, *, runtime: CausalCodecStreamingCudaGraphRuntime, lease_id: int):
        self._runtime = runtime
        self._lease_id = lease_id
        self._released = False

    @property
    def samples_per_frame(self) -> int:
        return self._runtime.decoder.samples_per_frame

    def decode_new(
        self,
        codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._released:
            raise RuntimeError("CUDA Graph codec lease was already released")
        return self._runtime._decode_from_lease(
            lease_id=self._lease_id,
            codes=codes,
        )

    def release(self) -> None:
        if self._released:
            raise RuntimeError("CUDA Graph codec lease was already released")
        try:
            self._runtime._release_lease(lease_id=self._lease_id)
        finally:
            self._released = True

    def __enter__(self) -> "CausalCodecStreamingCudaGraphLease":
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.release()
