# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""
Feed-forward network modules used by the MagpieTTS transformer stack.

This file exists to break a circular import between ``transformer_2501``
(which needs ``PositionwiseConvFFMoE``) and ``moe_modules`` (which needs
``ConvolutionLayer``).  Both can safely import from this leaf module.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from nemo.utils import logging


@dataclass
class ConvolutionIncrementalState:
    """Session-owned input history for one causal convolution."""

    input_history: Optional[torch.Tensor]
    max_length: int


@dataclass
class PositionwiseConvFFIncrementalState:
    """Session-owned histories for both convolutions in ``PositionwiseConvFF``."""

    projection: ConvolutionIncrementalState
    output: ConvolutionIncrementalState


class ConvolutionLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        is_causal: bool = False,
    ):
        """
        A convolutional layer that supports causal convolutions with padding. Replaces the standard MLP layer used in
        the original transformer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (Optional[int]): Padding added to both sides of the input. If None, it's calculated automatically.
            dilation (int): Spacing between kernel elements.
            bias (bool): If True, adds a learnable bias to the output.
            is_causal (bool): If True, uses causal convolution.
        """
        super().__init__()

        # Setup up padding; should be 0 if set to causal
        # If not causal and padding is None, set an appropriate value for padding
        self.causal_padding = None
        if is_causal:
            self.causal_padding = ((kernel_size - 1) * dilation, 0)
            if padding is not None:
                logging.warning(
                    f'{self} was initialized with is_causal set to True, and padding set to {padding}. '
                    f'The provided padding value will be ignored and set to {self.causal_padding}.'
                )
            padding = 0
        elif padding is None:
            if kernel_size % 2 == 0:
                raise ValueError("`kernel_size` must be odd when `padding` is None.")
            else:
                padding = int(dilation * (kernel_size - 1) / 2)

        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal, signal_mask=None):
        # signal: (B, C, T)
        # signal_mask: (B, T) or None (if None, assumes all positions are valid)
        if signal_mask is not None:
            signal = signal * signal_mask.unsqueeze(1)
        if self.is_causal:  # TODO: maybe replace with identify rather than keep conditional if in forward
            signal = F.pad(signal, self.causal_padding)

        conv_signal = self.conv(signal)
        if signal_mask is not None:
            conv_signal = conv_signal * signal_mask.unsqueeze(1)

        return conv_signal

    def create_incremental_state(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ConvolutionIncrementalState:
        """Allocate state for incremental causal inference.

        Kernel-size-one convolutions need no history and therefore allocate no
        tensor. Wider kernels preallocate the complete session buffer so CUDA
        Graph replay does not grow or concatenate tensors.
        """
        if not self.is_causal:
            raise ValueError("Incremental convolution requires a causal convolution")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        input_history = None
        if self.kernel_size > 1:
            input_history = torch.empty(
                batch_size,
                self.conv.in_channels,
                max_length,
                device=device,
                dtype=dtype,
            )
        return ConvolutionIncrementalState(input_history=input_history, max_length=max_length)

    def forward_incremental(
        self,
        signal: torch.Tensor,
        signal_mask: Optional[torch.Tensor],
        *,
        state: ConvolutionIncrementalState,
        position_offset: int,
    ) -> torch.Tensor:
        """Run one causal convolution timestep using caller-owned state."""
        if not self.is_causal:
            raise ValueError("Incremental convolution requires a causal convolution")
        if signal.size(-1) != 1:
            raise ValueError(f"Incremental convolution expects one timestep, got shape {tuple(signal.shape)}")
        if position_offset < 0 or position_offset >= state.max_length:
            raise ValueError(f"position_offset {position_offset} is outside the allocated length {state.max_length}")

        if signal_mask is not None:
            signal = signal * signal_mask.unsqueeze(1)

        if self.kernel_size == 1:
            output = self.conv(signal)
        else:
            if state.input_history is None:
                raise RuntimeError("Incremental convolution history was not allocated")
            state.input_history[:, :, position_offset : position_offset + 1].copy_(signal)
            receptive_field = (self.kernel_size - 1) * self.dilation
            history_start = max(0, position_offset - receptive_field)
            window = state.input_history[:, :, history_start : position_offset + 1]
            missing_history = receptive_field - position_offset
            if missing_history > 0:
                window = F.pad(window, (missing_history, 0))
            output = self.conv(window)

        if signal_mask is not None:
            output = output * signal_mask.unsqueeze(1)
        return output


class PositionwiseConvFF(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        p_dropout: float,
        kernel_size: int = 1,
        bias: bool = False,
        is_causal: bool = True,
        non_linearity: Callable = torch.nn.GELU(approximate="tanh"),
    ):
        """
        Positionwise Convolutional Feed-Forward layer to replace the MLP layer in transformers.

        Module will take the input with d_model hidden state, project it to d_ffn hidden dimension, perform nonlinear
        transformation, and project the state back into d_model hidden dimension. Finally, it applied dropout.

        Args:
            d_model (int): Input and output dimension of the model.
            d_ffn (int): Hidden dimension of the feed-forward network (usually 4 * d_model).
            p_dropout (float): Dropout probability.
            kernel_size (int): Size of the convolving kernel.
            bias (bool): If True, adds a learnable bias to the convolution layers.
            is_causal (bool): If True, uses causal convolution.
            non_linearity (Callable): Activation function to use (default: GELU).
        """
        super().__init__()
        # d_ffn is usually 4*d_model
        self.d_model = d_model
        self.non_linearity = non_linearity

        self.proj = ConvolutionLayer(d_model, d_ffn, bias=bias, kernel_size=kernel_size, is_causal=is_causal)
        self.o_net = ConvolutionLayer(d_ffn, d_model, bias=bias, kernel_size=kernel_size, is_causal=is_causal)
        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        """
        x (B, T, C)
        x_mask (B, T)
        """
        x = self.non_linearity(self.proj(x.transpose(1, 2), x_mask))
        x = self.dropout(self.o_net(x, x_mask).transpose(1, 2))
        return x

    def create_incremental_state(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PositionwiseConvFFIncrementalState:
        return PositionwiseConvFFIncrementalState(
            projection=self.proj.create_incremental_state(batch_size, max_length, device, dtype),
            output=self.o_net.create_incremental_state(batch_size, max_length, device, dtype),
        )

    def prefill_incremental_state(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        *,
        state: PositionwiseConvFFIncrementalState,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Run a vectorized prefix and populate causal convolution histories."""
        prefix_length = x.size(1)
        if position_offset < 0 or position_offset + prefix_length > state.projection.max_length:
            raise ValueError(
                f"Prefix [{position_offset}, {position_offset + prefix_length}) exceeds allocated length "
                f"{state.projection.max_length}"
            )
        projection_input = x.transpose(1, 2) * x_mask.unsqueeze(1)
        if state.projection.input_history is not None:
            state.projection.input_history[:, :, position_offset : position_offset + prefix_length].copy_(
                projection_input
            )
        projected = self.non_linearity(self.proj(projection_input, x_mask))
        if state.output.input_history is not None:
            state.output.input_history[:, :, position_offset : position_offset + prefix_length].copy_(projected)
        return self.dropout(self.o_net(projected, x_mask).transpose(1, 2))

    def forward_incremental(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        *,
        state: PositionwiseConvFFIncrementalState,
        position_offset: int,
    ) -> torch.Tensor:
        """Run one feed-forward timestep without recomputing its prefix."""
        projected = self.proj.forward_incremental(
            x.transpose(1, 2),
            x_mask,
            state=state.projection,
            position_offset=position_offset,
        )
        projected = self.non_linearity(projected)
        output = self.o_net.forward_incremental(
            projected,
            x_mask,
            state=state.output,
            position_offset=position_offset,
        )
        return self.dropout(output.transpose(1, 2))
