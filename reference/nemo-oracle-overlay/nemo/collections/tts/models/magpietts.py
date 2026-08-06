# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import copy
import json
import os
import random
import re
import time
from dataclasses import dataclass, field, fields
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import wandb
from lhotse.serialization import load_yaml
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch import nn

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import (
    MagpieTTSLhotseDataset,
    check_text_embedding_matches_tokenizer,
    setup_tokenizers,
)
from nemo.collections.tts.losses.aligner_loss import ForwardSumLoss
from nemo.collections.tts.losses.moe_loss import MoEAuxiliaryLoss, compute_expert_usage
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules import transformer_2501
from nemo.collections.tts.modules.aligner import AlignmentEncoder
from nemo.collections.tts.modules.audio_codec_modules import (
    VectorQuantizerIndexConverter,
)
from nemo.collections.tts.modules.magpietts_fused_sampling import LocalARRandomState
from nemo.collections.tts.modules.magpietts_modules import (
    CharAwareSubwordEncoder,
    CodecHelper,
    EOSDetectionMethod,
    LocalARDirectCaptureRuntime,
    LocalTransformerHelper,
    LocalTransformerType,
    SpecialAudioToken,
    add_special_tokens,
    clear_forbidden_logits,
    pad_audio_codes,
    remove_bos_token,
    remove_embedded_bos_token,
    remove_embedded_eos_token,
    remove_eos_token,
    remove_special_tokens,
    worker_init_fn,
)
from nemo.collections.tts.modules.streaming_codec import (
    CausalCodecStreamingCudaGraphRuntime,
    CausalCodecStreamingDecoder,
    WeightNormMaterializationReceipt,
    materialize_causal_hifigan_weight_norm_for_inference,
)
from nemo.collections.tts.modules.streaming_synthesis import (
    AsyncCodecSynthesisSession,
    StreamingCodecChunk,
    StreamingPcmChunk,
)
from nemo.collections.tts.parts.utils.helpers import (
    binarize_attention_parallel,
    get_mask_from_lengths,
    plot_alignment_to_numpy,
    plot_expert_usage_heatmap_to_numpy,
)
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    chunk_text_for_inference,
    get_tokenizer_for_language,
    stack_tensors,
)
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, safe_instantiate
from nemo.utils import logging
from nemo.utils.exceptions import NeMoBaseException


@dataclass
class InferBatchOutput:
    """Output dataclass for MagpieTTS infer_batch method.

    This provides a consistent return type regardless of which optional outputs
    are requested.

    Attributes:
        predicted_audio: Generated audio waveforms. Shape: (B, T_audio).
        predicted_audio_lens: Length of each audio in samples. Shape: (B,).
        predicted_codes: Generated audio codec tokens. Shape: (B, num_codebooks, T_frames).
        predicted_codes_lens: Length of each code sequence in frames. Shape: (B,).
        rtf_metrics: Dictionary containing real-time factor and timing metrics.
        cross_attention_maps: Always ``None``. Incremental inference does not
            expose cross-attention visualizations.
        headwise_cross_attention_maps: Always ``None``. Requests for attention
            maps fail explicitly in ``infer_batch``.
    """

    predicted_audio: torch.Tensor
    predicted_audio_lens: torch.Tensor
    predicted_codes: torch.Tensor
    predicted_codes_lens: torch.Tensor
    rtf_metrics: Dict[str, Any]
    cross_attention_maps: Optional[List[Any]] = None
    headwise_cross_attention_maps: Optional[List[Any]] = None


@dataclass
class ContextTensorsOutput:
    """Output container for prepare_context_tensors method.

    This dataclass provides typed access to all tensors prepared for the decoder,
    replacing the previous untyped dictionary return.

    Attributes:
        text_encoder_out: Encoded text from the encoder. Shape: (B, T_text, E).
        text_embedded: Embedded text before encoding. Shape: (B, T_text, E).
        text_mask: Boolean mask for text. Shape: (B, T_text).
        text_lens: Length of each text sequence. Shape: (B,).
        text: Original text token IDs. Shape: (B, T_text).
        cond: Conditioning tensor(s) for decoder cross-attention.
            Either a single tensor or list of tensors for multi-encoder models.
        cond_mask: Mask(s) for conditioning tensors.
        attn_prior: Attention prior matrix for guided attention.
            Can be None, a tensor, or a list of tensors per layer.
        prior_used: Whether attention prior is being used.
        multi_encoder_mapping: Mapping for multi-encoder models (or None).
        additional_decoder_input: Context embeddings prepended to decoder input.
        additional_decoder_mask: Mask for additional decoder input.
        dec_context_size: Number of context frames prepended to decoder.
        context_audio_codes: Extracted context audio codes. Shape: (B, C, T_ctx).
        context_audio_codes_lens: Length of context audio codes. Shape: (B,).
        beta_binomial_attn_prior: Original beta-binomial prior from batch.
    """

    text_encoder_out: torch.Tensor
    text_embedded: torch.Tensor
    text_mask: torch.Tensor
    text_lens: torch.Tensor
    text: torch.Tensor
    cond: Union[torch.Tensor, List[torch.Tensor]]
    cond_mask: Union[torch.Tensor, List[torch.Tensor]]
    attn_prior: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None
    prior_used: bool = False
    multi_encoder_mapping: Optional[Dict[str, Any]] = None
    additional_decoder_input: Optional[torch.Tensor] = None
    additional_decoder_mask: Optional[torch.Tensor] = None
    dec_context_size: int = 0
    context_audio_codes: Optional[torch.Tensor] = None
    context_audio_codes_lens: Optional[torch.Tensor] = None
    beta_binomial_attn_prior: Optional[torch.Tensor] = None


@dataclass
class ChunkedDecoderState:
    """Tracks state during chunked speech generation (single- or multi-chunk).

    This dataclass encapsulates all the mutable state variables used in the
    autoregressive decoding loop of generate_speech, reducing parameter
    passing and improving code organization.

    Attributes:
        audio_codes_step: Codes for the next one-token decoder input. Shape: (B, num_codebooks, frame_stack).
        audio_codes_mask: One-token decoder input mask. Shape: (B, 1).
        attended_timestep_counter: List of dicts tracking attention counts per timestep.
        prediction_buffer: Preallocated generated codec frames.
        num_prediction_steps: Number of decoder steps written to ``prediction_buffer``.
        chunk_end_dict: Maps batch indices to their chunk end timesteps.
        unfinished_texts: Maps batch indices to whether text is still being processed.
        finished_texts_counter: Maps batch indices to counts of timesteps near text end.
        attn_prior: Current attention prior tensor. Shape: (B, 1, T_text).
        packed_alignment_eos: Whether alignment and EOS share one host transfer.
        packed_streaming_submission_profile: Selected packed frame schedule, or
            None when the general per-boundary path is active.
        alignment_scratch: GPU state for the packed first-chunk alignment algorithm.
        packed_status_pending_steps: Number of ordered GPU status rows awaiting
            one host transfer.
        text_lens_host: Text lengths copied to the host once before decoding.
    """

    audio_codes_step: torch.Tensor
    audio_codes_mask: torch.Tensor
    attended_timestep_counter: List[Dict[int, int]]
    prediction_buffer: torch.Tensor
    num_prediction_steps: int
    chunk_end_dict: Dict[int, int]
    unfinished_texts: Dict[int, bool]
    finished_texts_counter: Dict[int, int]
    attn_prior: Optional[torch.Tensor] = None
    packed_alignment_eos: bool = False
    packed_streaming_submission_profile: Optional["PackedStreamingSubmissionProfile"] = None
    alignment_scratch: Optional["FirstChunkAlignmentScratch"] = None
    packed_status_pending_steps: int = 0
    text_lens_host: List[int] = field(default_factory=list)


@dataclass
class IncrementalDecoderSession:
    """Session-scoped main-decoder state for one ``generate_speech`` call."""

    transformer_state: transformer_2501.TransformerIncrementalState
    cond: Union[torch.Tensor, List[torch.Tensor]]
    cond_mask: Union[torch.Tensor, List[torch.Tensor]]
    additional_decoder_input: Optional[torch.Tensor]
    additional_decoder_mask: Optional[torch.Tensor]
    multi_encoder_mapping: Optional[List[Optional[int]]]
    conditional_batch_size: int
    cfg_scale: float
    use_cfg: bool
    cfg_decoder_step: Optional[torch.Tensor]
    cfg_decoder_mask_step: Optional[torch.Tensor]
    eos_scratch: "EOSDetectionScratch"
    alignment_layer_indices: Optional[Tuple[int, ...]]
    next_position: int = 0


@dataclass
class EOSDetectionScratch:
    """Fixed-shape scratch tensors reused by every decoder step."""

    base_max: torch.Tensor
    eos_wins_argmax: torch.Tensor
    sampled_eos: torch.Tensor
    eos_frames: torch.Tensor
    sampled_eos_frames: torch.Tensor
    unfinished_mask: torch.Tensor
    finished_mask: torch.Tensor
    frame_positions: torch.Tensor
    no_eos_positions: torch.Tensor
    end_frame_candidates: torch.Tensor
    end_frame_indices: torch.Tensor


@dataclass
class FirstChunkAlignmentScratch:
    """GPU tracking for the packed alignment/EOS host-boundary algorithm.

    ``host_status`` stores ordered ``(attended, end_frame_index)`` rows. The
    protected initial decoder steps can therefore remain device-resident and
    cross the host boundary together once EOS becomes legal or codec frames
    must be submitted.
    """

    text_lens: torch.Tensor
    last_attended: torch.Tensor
    counters: torch.Tensor
    positions: torch.Tensor
    valid_window: torch.Tensor
    auxiliary_mask: torch.Tensor
    masked_scores: torch.Tensor
    search_start: torch.Tensor
    window_end: torch.Tensor
    attended: torch.Tensor
    ended_attended: torch.Tensor
    has_valid_window: torch.Tensor
    counter_increment: torch.Tensor
    prior: torch.Tensor
    prior_indices: torch.Tensor
    sink_candidates: torch.Tensor
    max_sink_position: torch.Tensor
    host_status: torch.Tensor


@dataclass(frozen=True)
class PackedStreamingSubmissionProfile:
    """Explicit frame schedule supported by the packed Sofia streaming path."""

    first_chunk_frames: int
    steady_chunk_frames: int

    def __post_init__(self) -> None:
        supported_profiles = ((4, 8), (8, 8))
        profile = (self.first_chunk_frames, self.steady_chunk_frames)
        if profile not in supported_profiles:
            raise ValueError(
                f"Packed streaming submission profile {profile} is unsupported; "
                f"supported profiles are {supported_profiles}"
            )

    @staticmethod
    def _decoder_steps(frame_count: int, frame_stacking_factor: int) -> int:
        if frame_stacking_factor < 1:
            raise ValueError(f"frame_stacking_factor must be positive, got {frame_stacking_factor}")
        if frame_count % frame_stacking_factor != 0:
            raise ValueError(
                f"Packed frame count must be divisible by frame_stacking_factor: "
                f"{frame_count} % {frame_stacking_factor} != 0"
            )
        return frame_count // frame_stacking_factor

    def first_decoder_steps(self, frame_stacking_factor: int) -> int:
        return self._decoder_steps(self.first_chunk_frames, frame_stacking_factor)

    def steady_decoder_steps(self, frame_stacking_factor: int) -> int:
        return self._decoder_steps(self.steady_chunk_frames, frame_stacking_factor)

    def status_capacity(self, frame_stacking_factor: int, min_generated_frames: int) -> int:
        if frame_stacking_factor < 1:
            raise ValueError(f"frame_stacking_factor must be positive, got {frame_stacking_factor}")
        if min_generated_frames < 0:
            raise ValueError(f"min_generated_frames must be non-negative, got {min_generated_frames}")
        eos_forbidden_steps = (min_generated_frames + frame_stacking_factor - 1) // frame_stacking_factor
        return max(
            self.first_decoder_steps(frame_stacking_factor),
            self.steady_decoder_steps(frame_stacking_factor),
            eos_forbidden_steps + 1,
        )


FIRST_4_STEADY_8 = PackedStreamingSubmissionProfile(first_chunk_frames=4, steady_chunk_frames=8)
FIRST_8_STEADY_8 = PackedStreamingSubmissionProfile(first_chunk_frames=8, steady_chunk_frames=8)


@dataclass(frozen=True)
class FirstSubmissionCudaGraphKey:
    """Static contract for one explicitly warmed first-submission graph."""

    device_index: int
    dtype: torch.dtype
    actual_text_length: int
    conditional_shape: Tuple[int, ...]
    conditional_mask_shape: Tuple[int, ...]
    additional_input_shape: Optional[Tuple[int, ...]]
    additional_mask_shape: Optional[Tuple[int, ...]]
    max_decoder_steps: int
    temperature: float
    topk: int
    cfg_scale: float
    eos_detection_method: EOSDetectionMethod
    attention_prior_epsilon: float
    attention_prior_lookahead_window: int
    attention_sink_threshold: int
    estimate_alignment_from_layers: Optional[Tuple[int, ...]]
    apply_prior_to_layers: Optional[Tuple[int, ...]]
    frame_stacking_factor: int
    num_audio_codebooks: int
    codebook_size: int
    submission_profile: PackedStreamingSubmissionProfile


@dataclass(frozen=True)
class FirstSubmissionCapturedTensorSignature:
    """Identity and mutation version of one tensor read by the outer graph."""

    name: str
    data_ptr: int
    version: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class FirstSubmissionCudaGraphReceipt:
    """Capture and memory evidence for one exact-shape warmup bucket."""

    key: FirstSubmissionCudaGraphKey
    capture_seconds: float
    allocated_bytes_before: int
    allocated_bytes_after: int
    reserved_bytes_before: int
    reserved_bytes_after: int


@dataclass(frozen=True)
class FirstSubmissionCudaGraphReplay:
    """Graph-owned state handed to the established decoder continuation."""

    decoder_session: IncrementalDecoderSession
    alignment_scratch: FirstChunkAlignmentScratch
    predicted_codes: torch.Tensor
    packed_status: List[List[List[int]]]
    decoder_step_count: int


class FirstSubmissionCudaGraphLease:
    """Exclusive ownership of graph state through decoder continuation."""

    def __init__(self, runtime: "FirstSubmissionCudaGraphRuntime"):
        self._runtime = runtime
        self._released = False

    def replay(
        self,
        *,
        context_tensors: ContextTensorsOutput,
        dummy_cond: torch.Tensor,
        dummy_cond_mask: torch.Tensor,
        dummy_additional_decoder_input: Optional[torch.Tensor],
        dummy_addition_dec_mask: Optional[torch.Tensor],
        text_lens: torch.Tensor,
        last_attended: List[int],
        random_state: Optional[LocalARRandomState],
    ) -> FirstSubmissionCudaGraphReplay:
        if self._released:
            raise RuntimeError("First-submission CUDA graph lease was already released")
        return self._runtime._replay(
            context_tensors=context_tensors,
            dummy_cond=dummy_cond,
            dummy_cond_mask=dummy_cond_mask,
            dummy_additional_decoder_input=dummy_additional_decoder_input,
            dummy_addition_dec_mask=dummy_addition_dec_mask,
            text_lens=text_lens,
            last_attended=last_attended,
            random_state=random_state,
        )

    def release(self) -> None:
        if self._released:
            raise RuntimeError("First-submission CUDA graph lease was already released")
        self._released = True
        self._runtime._release()


class FirstSubmissionCudaGraphRuntime:
    """The selected first submission captured as one fixed-shape CUDA graph.

    This runtime owns main-decoder K/V and FFN history, attention tracking,
    local-AR state, RNG state, and output buffers. The owner must retain the
    exclusive lease while the established incremental decoder continues from
    the captured step count, because that continuation writes the same
    graph-owned state.
    """

    def __init__(
        self,
        *,
        model: "MagpieTTSModel",
        key: FirstSubmissionCudaGraphKey,
        context_tensors: ContextTensorsOutput,
        dummy_cond: torch.Tensor,
        dummy_cond_mask: torch.Tensor,
        dummy_additional_decoder_input: Optional[torch.Tensor],
        dummy_addition_dec_mask: Optional[torch.Tensor],
        last_attended: List[int],
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("First-submission CUDA graph requires CUDA")
        if model.training:
            raise RuntimeError("First-submission CUDA graph capture requires model.eval()")
        if model.device.type != "cuda":
            raise RuntimeError(f"First-submission CUDA graph requires a CUDA model, got {model.device}")
        if key.actual_text_length < 1:
            raise ValueError(f"actual_text_length must be positive, got {key.actual_text_length}")
        if key.frame_stacking_factor != model.frame_stacking_factor:
            raise ValueError(
                f"First-submission key frame stacking factor {key.frame_stacking_factor} "
                f"does not match model value {model.frame_stacking_factor}"
            )
        self.decoder_step_count = key.submission_profile.first_decoder_steps(model.frame_stacking_factor)
        if key.max_decoder_steps < self.decoder_step_count:
            raise ValueError(
                f"First-submission CUDA graph needs {self.decoder_step_count} decoder steps, "
                f"got {key.max_decoder_steps}"
            )

        self.model = model
        self.key = key
        self.device = torch.device("cuda", key.device_index)
        self._lease_lock = Lock()
        self._lease_active = False
        self.execution_stream = torch.cuda.Stream(device=self.device)
        self._completion_event = torch.cuda.Event(blocking=False, interprocess=False)
        self.graph = torch.cuda.CUDAGraph()

        self._allocate_owned_state(
            context_tensors=context_tensors,
            dummy_cond=dummy_cond,
            dummy_cond_mask=dummy_cond_mask,
            dummy_additional_decoder_input=dummy_additional_decoder_input,
            dummy_addition_dec_mask=dummy_addition_dec_mask,
            last_attended=last_attended,
        )
        self.execution_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.execution_stream), torch.inference_mode():
            self._execute_unrolled_body()
        torch.cuda.current_stream(self.device).wait_stream(self.execution_stream)
        torch.cuda.synchronize(self.device)

        # Warmup mutates K/V, cross-attention references, and Python position
        # bookkeeping. Capture therefore receives entirely fresh owned state.
        self._allocate_owned_state(
            context_tensors=context_tensors,
            dummy_cond=dummy_cond,
            dummy_cond_mask=dummy_cond_mask,
            dummy_additional_decoder_input=dummy_additional_decoder_input,
            dummy_addition_dec_mask=dummy_addition_dec_mask,
            last_attended=last_attended,
        )
        with torch.cuda.graph(self.graph, stream=self.execution_stream), torch.inference_mode():
            self._execute_unrolled_body()
        self.continuation_position = self.decoder_session.next_position
        self.parameter_signature = self._record_parameter_invariants()

    def _allocate_owned_state(
        self,
        *,
        context_tensors: ContextTensorsOutput,
        dummy_cond: torch.Tensor,
        dummy_cond_mask: torch.Tensor,
        dummy_additional_decoder_input: Optional[torch.Tensor],
        dummy_addition_dec_mask: Optional[torch.Tensor],
        last_attended: List[int],
    ) -> None:
        if not isinstance(context_tensors.cond, torch.Tensor) or not isinstance(
            context_tensors.cond_mask, torch.Tensor
        ):
            raise TypeError("First-submission graph requires single-tensor conditioning")
        if context_tensors.multi_encoder_mapping is not None:
            raise ValueError("First-submission graph does not support multi-encoder mapping")
        if len(last_attended) != 1:
            raise ValueError(f"First-submission graph requires one attended position, got {len(last_attended)}")

        self.decoder_session = self.model._create_incremental_decoder_session(
            context_tensors=context_tensors,
            use_cfg=True,
            cfg_scale=self.key.cfg_scale,
            dummy_cond=dummy_cond,
            dummy_cond_mask=dummy_cond_mask,
            dummy_additional_decoder_input=dummy_additional_decoder_input,
            dummy_addition_dec_mask=dummy_addition_dec_mask,
            batch_size=1,
            device=self.device,
            dtype=self.key.dtype,
        )
        if not isinstance(self.decoder_session.cond, torch.Tensor) or not isinstance(
            self.decoder_session.cond_mask, torch.Tensor
        ):
            raise TypeError("First-submission graph requires combined tensor conditioning")

        self.audio_codes_step = torch.full(
            (1, self.model.num_audio_codebooks, self.model.frame_stacking_factor),
            self.model.audio_bos_id,
            device=self.device,
            dtype=torch.long,
        )
        self.audio_codes_mask = torch.ones((1, 1), device=self.device, dtype=torch.bool)
        self.predicted_codes = torch.empty(
            (
                1,
                self.model.num_audio_codebooks,
                self.decoder_step_count * self.model.frame_stacking_factor,
            ),
            device=self.device,
            dtype=torch.long,
        )
        self.invalid_rows = torch.empty(
            (self.decoder_step_count, 1),
            device=self.device,
            dtype=torch.int32,
        )
        self.alignment_scratch = self.model._create_first_chunk_alignment_scratch(
            text_lens=context_tensors.text_lens,
            last_attended_timesteps=[last_attended],
            effective_batch_size=2,
            text_length=context_tensors.text_encoder_out.size(1),
            dtype=self.key.dtype,
            status_capacity=self.key.submission_profile.status_capacity(
                self.model.frame_stacking_factor,
                self.model.inference_parameters.min_generated_frames,
            ),
        )
        self.local_ar = self.model._lt_helper.create_autoregressive_direct_capture_runtime(
            actual_batch_size=1,
            input_dim=int(self.model.cfg.decoder.d_model),
            device=self.device,
            dtype=self.key.dtype,
            temperature=self.key.temperature,
            topk=self.key.topk,
            use_cfg=True,
            cfg_scale=self.key.cfg_scale,
        )

    def _prepare_layer_attention_prior(
        self, prior: Optional[torch.Tensor]
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        if self.model.inference_parameters.apply_prior_to_layers is None:
            if prior is None:
                raise RuntimeError("First-submission graph requires a dynamic attention prior after step zero")
            return prior
        layer_prior = [None for _ in range(self.model.cfg.decoder.n_layers)]
        for layer_index in self.model.inference_parameters.apply_prior_to_layers:
            layer_prior[layer_index] = prior
        return layer_prior

    def _execute_unrolled_body(self) -> None:
        next_prior = None
        for decoder_step in range(self.decoder_step_count):
            embedded = self.model._embed_audio_step(self.audio_codes_step)
            if decoder_step == 0:
                attention_prior = None
            else:
                attention_prior = self._prepare_layer_attention_prior(next_prior)
            forbid_audio_eos = (
                decoder_step * self.model.frame_stacking_factor < self.model.inference_parameters.min_generated_frames
            )
            all_code_logits, alignment_scores, decoder_output = self.model._run_chunked_forward_with_cfg(
                session=self.decoder_session,
                audio_codes_embedded=embedded,
                audio_codes_mask=self.audio_codes_mask,
                attn_prior=attention_prior,
                project_code_logits=not forbid_audio_eos,
            )
            if alignment_scores is None:
                raise RuntimeError("First-submission graph requires decoder alignment scores")
            next_prior = self.model._compute_first_chunk_alignment_prior(
                alignment_scores,
                self.alignment_scratch,
            )

            frame_start = decoder_step * self.model.frame_stacking_factor
            frame_end = frame_start + self.model.frame_stacking_factor
            codes_step = self.predicted_codes[:, :, frame_start:frame_end]
            self.local_ar.execute_capture_step(
                dec_output=decoder_output[:, -1, :],
                forbid_audio_eos=forbid_audio_eos,
                output_codes=codes_step,
                invalid_rows=self.invalid_rows[decoder_step],
            )
            if forbid_audio_eos:
                if all_code_logits is not None:
                    raise AssertionError("Forbidden-EOS first-submission step unexpectedly projected main logits")
                end_frame_indices = self.model.detect_forbidden_eos_batch(
                    codes_step,
                    self.decoder_session.eos_scratch,
                )
            else:
                if all_code_logits is None:
                    raise AssertionError("EOS-enabled first-submission step requires main logits")
                end_frame_indices = self.model.detect_eos_batch(
                    codes_step,
                    all_code_logits[:, -1, :],
                    self.key.eos_detection_method,
                    {},
                    {},
                    forbid_audio_eos,
                    self.decoder_session.eos_scratch,
                )
            self.model._record_packed_alignment_eos_status(
                self.alignment_scratch,
                pending_index=decoder_step,
                end_frame_indices=end_frame_indices,
            )
            self.audio_codes_step.copy_(codes_step)

    def _runtime_named_tensors(self) -> Iterator[Tuple[str, torch.Tensor]]:
        module_groups = (
            ("decoder", self.model.decoder),
            ("final_proj", self.model.final_proj),
            ("audio_embeddings", self.model.audio_embeddings),
            ("local_transformer", self.model.local_transformer),
            ("local_transformer_in_projection", self.model.local_transformer_in_projection),
            ("local_transformer_audio_out_projection", self.model.local_transformer_audio_out_projection),
            ("local_transformer_out_projections", self.model.local_transformer_out_projections),
        )
        for group_name, module in module_groups:
            for tensor_name, parameter in module.named_parameters():
                yield f"{group_name}.parameter.{tensor_name}", parameter
            for tensor_name, buffer in module.named_buffers():
                yield f"{group_name}.buffer.{tensor_name}", buffer

    def _read_parameter_signature(self) -> Tuple[FirstSubmissionCapturedTensorSignature, ...]:
        return tuple(
            FirstSubmissionCapturedTensorSignature(
                name=name,
                data_ptr=tensor.data_ptr(),
                version=tensor._version,
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            for name, tensor in self._runtime_named_tensors()
        )

    def _record_parameter_invariants(self) -> Tuple[FirstSubmissionCapturedTensorSignature, ...]:
        signature = self._read_parameter_signature()
        wrong_device = tuple(item.name for item in signature if item.device != self.device)
        if wrong_device:
            raise RuntimeError(
                f"First-submission graph tensors must all be on {self.device}; "
                f"found different devices for {list(wrong_device)}"
            )
        return signature

    def validate_parameter_invariants(self) -> None:
        """Reject mode, storage, or in-place mutation after graph capture."""

        if self.model.training:
            raise RuntimeError("First-submission CUDA graph requires model.eval() after capture")
        training_modules = tuple(
            name
            for name, module in (
                ("decoder", self.model.decoder),
                ("local_transformer", self.model.local_transformer),
            )
            if module.training
        )
        if training_modules:
            raise RuntimeError(f"First-submission CUDA graph modules entered training mode: {list(training_modules)}")
        current_signature = self._record_parameter_invariants()
        if current_signature != self.parameter_signature:
            raise RuntimeError(
                "First-submission CUDA graph parameters or buffers changed after capture; "
                "invalidate and explicitly warm a new runtime"
            )
        self.local_ar.validate_parameter_invariants()

    def acquire(self) -> FirstSubmissionCudaGraphLease:
        self._lease_lock.acquire()
        if self._lease_active:
            self._lease_lock.release()
            raise RuntimeError("First-submission CUDA graph lease state is inconsistent")
        try:
            self.validate_parameter_invariants()
            self._lease_active = True
            return FirstSubmissionCudaGraphLease(self)
        except BaseException:
            self._lease_lock.release()
            raise

    def _copy_tensor(self, target: torch.Tensor, source: torch.Tensor, *, name: str) -> None:
        if source.shape != target.shape or source.device != target.device or source.dtype != target.dtype:
            raise ValueError(
                f"{name} does not match warmed graph bucket: "
                f"{tuple(source.shape)}/{source.device}/{source.dtype} != "
                f"{tuple(target.shape)}/{target.device}/{target.dtype}"
            )
        target.copy_(source)

    def _copy_request_inputs(
        self,
        *,
        context_tensors: ContextTensorsOutput,
        dummy_cond: torch.Tensor,
        dummy_cond_mask: torch.Tensor,
        dummy_additional_decoder_input: Optional[torch.Tensor],
        dummy_addition_dec_mask: Optional[torch.Tensor],
        text_lens: torch.Tensor,
        last_attended: List[int],
    ) -> None:
        if not isinstance(context_tensors.cond, torch.Tensor) or not isinstance(
            context_tensors.cond_mask, torch.Tensor
        ):
            raise TypeError("First-submission graph requires tensor conditioning")
        if not isinstance(self.decoder_session.cond, torch.Tensor) or not isinstance(
            self.decoder_session.cond_mask, torch.Tensor
        ):
            raise AssertionError("Captured first-submission conditioning is not tensor-valued")

        self._copy_tensor(
            self.decoder_session.cond[:1],
            context_tensors.cond,
            name="conditional context",
        )
        self._copy_tensor(
            self.decoder_session.cond[1:],
            dummy_cond,
            name="unconditional context",
        )
        self._copy_tensor(
            self.decoder_session.cond_mask[:1],
            context_tensors.cond_mask,
            name="conditional context mask",
        )
        self._copy_tensor(
            self.decoder_session.cond_mask[1:],
            dummy_cond_mask,
            name="unconditional context mask",
        )

        if self.decoder_session.additional_decoder_input is None:
            if (
                context_tensors.additional_decoder_input is not None
                or dummy_additional_decoder_input is not None
                or context_tensors.additional_decoder_mask is not None
                or dummy_addition_dec_mask is not None
            ):
                raise ValueError("Request adds a decoder prefix absent from the warmed graph bucket")
        else:
            if (
                context_tensors.additional_decoder_input is None
                or dummy_additional_decoder_input is None
                or context_tensors.additional_decoder_mask is None
                or dummy_addition_dec_mask is None
                or self.decoder_session.additional_decoder_mask is None
            ):
                raise ValueError("Request is missing the decoder prefix required by the warmed graph bucket")
            self._copy_tensor(
                self.decoder_session.additional_decoder_input[:1],
                context_tensors.additional_decoder_input,
                name="conditional decoder prefix",
            )
            self._copy_tensor(
                self.decoder_session.additional_decoder_input[1:],
                dummy_additional_decoder_input,
                name="unconditional decoder prefix",
            )
            self._copy_tensor(
                self.decoder_session.additional_decoder_mask[:1],
                context_tensors.additional_decoder_mask,
                name="conditional decoder prefix mask",
            )
            self._copy_tensor(
                self.decoder_session.additional_decoder_mask[1:],
                dummy_addition_dec_mask,
                name="unconditional decoder prefix mask",
            )

        self._copy_tensor(self.alignment_scratch.text_lens, text_lens, name="text lengths")
        if len(last_attended) != 1:
            raise ValueError(f"First-submission graph requires one attended position, got {len(last_attended)}")
        self.alignment_scratch.last_attended.fill_(last_attended[0])
        self.alignment_scratch.counters.zero_()
        self.audio_codes_step.fill_(self.model.audio_bos_id)
        self.audio_codes_mask.fill_(True)
        self.invalid_rows.zero_()

    def _restore_continuation_position(self) -> None:
        """Reset the host cursor that CUDA graph replay cannot mutate."""

        self.decoder_session.next_position = self.continuation_position

    def _replay(
        self,
        *,
        context_tensors: ContextTensorsOutput,
        dummy_cond: torch.Tensor,
        dummy_cond_mask: torch.Tensor,
        dummy_additional_decoder_input: Optional[torch.Tensor],
        dummy_addition_dec_mask: Optional[torch.Tensor],
        text_lens: torch.Tensor,
        last_attended: List[int],
        random_state: Optional[LocalARRandomState],
    ) -> FirstSubmissionCudaGraphReplay:
        if not self._lease_active:
            raise RuntimeError("First-submission CUDA graph replay requires an active lease")
        caller_stream = torch.cuda.current_stream(self.device)
        self.execution_stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.execution_stream):
            self._copy_request_inputs(
                context_tensors=context_tensors,
                dummy_cond=dummy_cond,
                dummy_cond_mask=dummy_cond_mask,
                dummy_additional_decoder_input=dummy_additional_decoder_input,
                dummy_addition_dec_mask=dummy_addition_dec_mask,
                text_lens=text_lens,
                last_attended=last_attended,
            )
            self.local_ar.copy_random_state_from(random_state)
            self.graph.replay()
            # Continuation advances this Python-side cursor beyond step four.
            # CUDA replay only restores graph-owned tensors, so explicitly
            # restore the matching host cursor for every new request.
            self._restore_continuation_position()
            self.local_ar.copy_random_state_to(random_state)
            torch.ops.aten._assert_async.msg(
                (self.invalid_rows == 0).all(),
                "First-submission graph local-AR received NaN, infinity, or no valid candidate",
            )
            self._completion_event.record(self.execution_stream)

        caller_stream.wait_event(self._completion_event)
        self.predicted_codes.record_stream(caller_stream)
        packed_status = self.model._transfer_packed_alignment_eos_status(
            self.alignment_scratch,
            pending_steps=self.decoder_step_count,
        )
        return FirstSubmissionCudaGraphReplay(
            decoder_session=self.decoder_session,
            alignment_scratch=self.alignment_scratch,
            predicted_codes=self.predicted_codes,
            packed_status=packed_status,
            decoder_step_count=self.decoder_step_count,
        )

    def _release(self) -> None:
        if not self._lease_active:
            raise RuntimeError("First-submission CUDA graph lease is not active")
        caller_stream = torch.cuda.current_stream(self.device)
        self.execution_stream.wait_stream(caller_stream)
        self._lease_active = False
        self._lease_lock.release()

    def synchronize_before_release(self) -> None:
        """Wait for an active continuation before dropping graph-owned state."""

        self._lease_lock.acquire()
        try:
            self.execution_stream.synchronize()
        finally:
            self._lease_lock.release()


@dataclass
class StreamingCodecEmissionState:
    """Producer-side emission state shared across text chunks.

    The state advances only after the asynchronous codec session accepts a
    submission. Codec tensors are never retained here; the model keeps them in
    its generation buffer until the selected emission boundary is reached.
    """

    accepted_chunk_count: int = 0

    def __post_init__(self) -> None:
        if self.accepted_chunk_count < 0:
            raise ValueError("accepted_chunk_count must be non-negative, " f"got {self.accepted_chunk_count}")

    def record_accepted_chunk(self) -> None:
        self.accepted_chunk_count += 1


@dataclass(frozen=True)
class StreamingCodecEmissionSchedule:
    """Frame-count and playback-priority schedule for codec submissions.

    The first PCM callback is intentionally completed before generation
    continues. On a shared GPU, merely enqueueing the codec on another stream
    lets subsequent decoder work delay audible output by hundreds of
    milliseconds. The barrier applies only to the first accepted chunk; later
    codec chunks remain asynchronous.
    """

    first_chunk_frames: int = 4
    steady_chunk_frames: int = 8
    prioritize_first_pcm: bool = True

    def __post_init__(self) -> None:
        if self.first_chunk_frames < 1:
            raise ValueError(f"first_chunk_frames must be positive, got {self.first_chunk_frames}")
        if self.steady_chunk_frames < 1:
            raise ValueError(f"steady_chunk_frames must be positive, got {self.steady_chunk_frames}")

    def validate_frame_stacking_factor(self, frame_stacking_factor: int) -> None:
        if frame_stacking_factor < 1:
            raise ValueError("frame_stacking_factor must be positive, " f"got {frame_stacking_factor}")
        if self.first_chunk_frames % frame_stacking_factor != 0:
            raise ValueError(
                "first_chunk_frames must be divisible by frame_stacking_factor: "
                f"{self.first_chunk_frames} % {frame_stacking_factor} != 0"
            )
        if self.steady_chunk_frames % frame_stacking_factor != 0:
            raise ValueError(
                "steady_chunk_frames must be divisible by frame_stacking_factor: "
                f"{self.steady_chunk_frames} % {frame_stacking_factor} != 0"
            )

    def target_frame_count(self, state: StreamingCodecEmissionState) -> int:
        if state.accepted_chunk_count == 0:
            return self.first_chunk_frames
        return self.steady_chunk_frames

    def should_wait_for_completion(self, state: StreamingCodecEmissionState) -> bool:
        return self.prioritize_first_pcm and state.accepted_chunk_count == 0

    def should_emit(
        self,
        state: StreamingCodecEmissionState,
        *,
        available_frame_count: int,
        model_chunk_ended: bool,
    ) -> bool:
        if available_frame_count < 0:
            raise ValueError("available_frame_count must be non-negative, " f"got {available_frame_count}")
        return model_chunk_ended or available_frame_count >= self.target_frame_count(state)


@dataclass
class ChunkState:
    """Mutable state persisting across chunks during chunked generation.

    Created by the inference runner via model.create_chunk_state(),
    passed to generate_speech(), and updated in-place across chunk iterations.

    Attributes:
        batch_size: Number of items in the batch.
        history_text: Text tokens from previous chunks. Shape: (B, T).
        history_text_lens: Lengths of history text per batch item. Shape: (B,).
        history_context_tensor: Encoder output from previous chunks. Shape: (B, T, E).
        end_indices: Maps batch indices to overall timestep where they ended.
        overall_idx: Global timestep counter across all chunks.
        left_offset: Sliding window offset per batch item for attention tracking.
        previous_attn_len: Attention lengths from previous chunk per batch item.
        last_attended_timesteps: Tracking of attended positions across decoding.
        streaming_codec_emission_state: Producer-side streaming emission state
            shared by every text chunk in this synthesis session.
        first_submission_graph_lease: Internal lease retained until the current
            final-chunk generation call completes.
    """

    batch_size: int
    history_text: Optional[torch.Tensor] = None
    history_text_lens: Optional[torch.Tensor] = None
    history_context_tensor: Optional[torch.Tensor] = None
    end_indices: Dict[int, int] = field(default_factory=dict)
    overall_idx: int = 0
    left_offset: List[int] = field(default_factory=list)
    previous_attn_len: List[int] = field(default_factory=list)
    last_attended_timesteps: List[List[int]] = field(default_factory=list)
    streaming_codec_emission_state: StreamingCodecEmissionState = field(default_factory=StreamingCodecEmissionState)
    first_submission_graph_lease: Optional[FirstSubmissionCudaGraphLease] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize batch-sized lists if not provided."""
        if not self.left_offset:
            self.left_offset = [0] * self.batch_size
        if not self.last_attended_timesteps:
            self.last_attended_timesteps = [[1] * self.batch_size]


@dataclass
class ModelInferenceParameters:
    """Model specific parameters that are sent to inference functions.

    This dataclass should contain all parameters that are model specific and should not change on a per run basis.

    Attributes:
        max_decoder_steps (int): Maximum number of decoder steps. Autoregressive for loop will terminate here.
        temperature (float): Sampling temperature.
        topk (int): Number of top-probability tokens to consider in sampling.
        cfg_scale (float): Scale factor for classifier-free guidance. Only used if use_cfg=True.
        apply_attention_prior (bool): Whether to apply attention prior.
        attention_prior_epsilon (float): Base probability for non-targeted positions.
        attention_prior_lookahead_window (int): Size of the forward-looking window to search for the next attended
            timestep. Determines how far ahead from the last attended timestep to look.
        estimate_alignment_from_layers (Optional[List[int]]): Layers to use for alignment estimation.
        apply_prior_to_layers (Optional[List[int]]): Layers to apply prior to.
        start_prior_after_n_audio_steps (int): Which step to start enabling the attention prior.
        ignore_finished_sentence_tracking (bool): Whether to ignore finished sentence tracking.
        eos_detection_method (str): EOS detection method. See the EOSDetectionMethod class.
        min_generated_frames (int): Setting this greater than 0 prevents rare cases of first-frame termination. Any
            number greater between 1 and 4 should work, but 4 lines up with the codec's minimum frame requirement.
        attention_sink_threshold (int): Times a position may be attended before standard inference advances past it.
        history_len_heuristic (int): Maximum history tokens retained across text chunks.
        prior_weights_init (Tuple[float, ...]): Attention prior weights used when initializing a new chunk.
        prior_weights (Tuple[float, ...]): Attention prior weights used during chunked generation.
        finished_limit_with_eot (int): Near-end steps before allowing EOS in the final chunk.
        finished_limit_without_eot (int): Near-end steps before allowing EOS in a non-final chunk.
        finished_limit_first_chunk (int): Near-end steps before allowing EOS in the first chunk.
        forceful_chunk_end_threshold (int): Near-end steps before forcibly ending a non-final chunk.
        argmax_temperature (float): Temperature used for the argmax EOS-detection sample.
        short_sentence_threshold (int): Texts at or below this length use a uniform chunked attention prior.
        chunked_attention_sink_threshold (int): Times a position may be attended before the chunked prior penalizes it.
        near_end_threshold (int): Positions from the text end that are treated as near the end.
    """

    max_decoder_steps: int = 500
    temperature: float = 0.7
    topk: int = 80
    cfg_scale: float = 2.5
    apply_attention_prior: bool = True
    attention_prior_epsilon: float = 0.1
    attention_prior_lookahead_window: int = 5
    estimate_alignment_from_layers: Optional[List[int]] = None
    apply_prior_to_layers: Optional[List[int]] = None
    start_prior_after_n_audio_steps: int = 0
    ignore_finished_sentence_tracking: bool = True
    eos_detection_method: str = "argmax_or_multinomial_any"
    min_generated_frames: int = 4
    attention_sink_threshold: int = 8
    history_len_heuristic: int = 20
    prior_weights_init: Tuple[float, ...] = (0.5, 1.0, 0.8, 0.2, 0.2)
    prior_weights: Tuple[float, ...] = (0.2, 1.0, 0.6, 0.4, 0.2, 0.2)
    finished_limit_with_eot: int = 5
    finished_limit_without_eot: int = 1
    finished_limit_first_chunk: int = 20
    forceful_chunk_end_threshold: int = 3
    argmax_temperature: float = 0.01
    short_sentence_threshold: int = 35
    chunked_attention_sink_threshold: int = 10
    near_end_threshold: int = 3

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelInferenceParameters':
        # Get the names of fields defined in the dataclass
        field_names = {field.name for field in fields(cls)}
        # Filter the input dictionary to include only valid fields
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        # Instantiate the dataclass with the filtered data

        # Double check for renamed fields: prior_epsilon and lookahead_window_size
        # These fields are currently used in nvidia/magpie_tts_multilingual_357m with commit hash: 291da79
        if 'prior_epsilon' in data:
            filtered_data['attention_prior_epsilon'] = data['prior_epsilon']
        if 'lookahead_window_size' in data:
            filtered_data['attention_prior_lookahead_window'] = data['lookahead_window_size']
        for field_name in ('prior_weights_init', 'prior_weights'):
            if field_name in filtered_data:
                filtered_data[field_name] = tuple(filtered_data[field_name])
        return cls(**filtered_data)


class MagpieTTSModel(ModelPT):
    """
    Magpie-TTS Model Base Class used for training a TTS model that can generate audio codes from transcript and a context
    audio/text

    Supports multiple model types:

    - multi_encoder_context_tts: Transcript and context audio go to different encoders. Transcript encoding feeds to
      layers given by cfg.model.transcript_decoder_layers and the context encoding feeds into the layers given by
      context_decoder_layers .Also supports text context which gets encoded by the same encoder as context audio.
      Only one of context audio or contex text is supported.

    - decoder_context_tts: Text goes into the encoder; context & target audio go to the decoder. Also supports text
      context. Supports fixed sized context so we set context_duration_min and context_duration_max to the same
      value (5 seconds). Text context, which is usually shorter than number of codec frames of 5 second of audio, is
      padded to the max context duration in this model.

    - decoder_ce: Same as decoder_context_tts except there is a small neural network between the context tensors and
      the decoder input.
    """

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        # Register tokenizer artifacts (phoneme_dict, heteronyms, etc.) for .nemo packaging
        self._register_tokenizer_artifacts(cfg)
        self._setup_inference_parameters(cfg)

        # load codec, disable loading of loss modules not needed during inference
        codec_model_path = cfg.get('codecmodel_path')
        if codec_model_path.startswith('nvidia/'):
            codec_model = AudioCodecModel.from_pretrained(codec_model_path)
        else:
            codec_model_cfg = AudioCodecModel.restore_from(codec_model_path, return_config=True)
            if "use_scl_loss" in codec_model_cfg:
                codec_model_cfg.use_scl_loss = False
            codec_model = AudioCodecModel.restore_from(
                codec_model_path, strict=False, override_config_path=codec_model_cfg
            )
        self.sample_rate = codec_model.sample_rate
        self.output_sample_rate = codec_model.output_sample_rate
        self.codec_model_samples_per_frame = codec_model.samples_per_frame
        # del codec discriminator to free memory
        del codec_model.discriminator

        # When using FSQ tokens, the codebook structure can be changed at any time.
        # An FSQ definition can be provided in `vector_quantizer` config to train with a codebook structure
        # that is different than in the audio codec checkpoint.
        vector_quantizer = cfg.get('vector_quantizer')
        if vector_quantizer is not None:
            vector_quantizer = safe_instantiate(vector_quantizer)
            num_audio_codebooks = vector_quantizer.num_codebooks
            codebook_size = vector_quantizer.codebook_size
            codec_converter = VectorQuantizerIndexConverter(
                vector_quantizer_original=codec_model.vector_quantizer,
                vector_quantizer_new=vector_quantizer,
            )
            data_num_audio_codebooks = codec_model.vector_quantizer.num_codebooks
        else:
            num_audio_codebooks = codec_model.num_codebooks
            data_num_audio_codebooks = num_audio_codebooks
            codebook_size = codec_model.codebook_size
            codec_converter = None
        # The dataloader needs to know the number of codebooks that the context codes were stored in
        # In the case where there are no context codes saved, and there is no context audio (in the text context path),
        # We create a dummy context code tensor that is only [context_BOS, context_EOS] that is repeated for
        # data_num_audio_codebooks
        self.data_num_audio_codebooks = data_num_audio_codebooks
        self.num_audio_codebooks = num_audio_codebooks
        self.codebook_size = codebook_size

        # Our codebooks start with actual audio codec tokens, followed by special tokens.
        # The `forced_*` options are for backward compatibility for models trained with older code.
        get_token_index = partial(SpecialAudioToken.get_index, base_codebook_size=self.codebook_size)
        self.audio_bos_id = cfg.get('forced_audio_bos_id', get_token_index(SpecialAudioToken.AUDIO_BOS))
        self.audio_eos_id = cfg.get('forced_audio_eos_id', get_token_index(SpecialAudioToken.AUDIO_EOS))
        self.context_audio_bos_id = cfg.get(
            'forced_context_audio_bos_id', get_token_index(SpecialAudioToken.AUDIO_CONTEXT_BOS)
        )
        self.context_audio_eos_id = cfg.get(
            'forced_context_audio_eos_id', get_token_index(SpecialAudioToken.AUDIO_CONTEXT_EOS)
        )
        self.mask_token_id = cfg.get('forced_mask_token_id', get_token_index(SpecialAudioToken.MASK_TOKEN))
        self.num_all_tokens_per_codebook = cfg.get(
            'forced_num_all_tokens_per_codebook', self.codebook_size + len(SpecialAudioToken)
        )
        self.use_bpe_char_tokenizer = cfg.get('use_bpe_char_tokenizer', False)

        # The frame stacking factor controls how many consecutive frames are processed together by the base decoder
        # (and then refined into individual frames by the local transformer). A frame stacking factor of 1 means no
        # frame stacking. We have a separate embedding table for each of the stacked frames, e.g. for frame stacking
        # factor of 3, the entries of codebook 0 appear 3 times in the embedding table.
        self.frame_stacking_factor = cfg.get('frame_stacking_factor', 1)
        assert 'downsample_factor' not in cfg, '`downsample_factor` is deprecated, use `frame_stacking_factor` instead'
        # Setup tokenizer
        if hasattr(cfg, 'text_tokenizer'):
            # For backward compatibility for English-only models
            with open_dict(cfg):
                cfg.text_tokenizers = {"english_phoneme": cfg.text_tokenizer}
                del cfg['text_tokenizer']

        self.use_text_conditioning_encoder = cfg.get('use_text_conditioning_encoder', False)
        # Using google-t5/t5-small as default text conditioning tokenizer for backward compatibility.
        self.text_conditioning_tokenizer_name = cfg.get('text_conditioning_tokenizer_name', None)
        self.legacy_text_conditioning = cfg.get('legacy_text_conditioning', False)

        if self.legacy_text_conditioning:
            if self.text_conditioning_tokenizer_name is None:
                self.text_conditioning_tokenizer_name = "google-t5/t5-small"

            tokenizer_target = "AutoTokenizer"
            if self.text_conditioning_tokenizer_name == "google-t5/t5-small":
                tokenizer_target = "T5Tokenizer"

            with open_dict(cfg):
                cfg.text_tokenizers[self.text_conditioning_tokenizer_name] = {
                    '_target_': tokenizer_target,
                    'pretrained_model': self.text_conditioning_tokenizer_name,
                }
        elif self.text_conditioning_tokenizer_name is None:
            # If no text_conditioning_tokenizer_name is specified, use the first one as default
            # For text context tokenization
            self.text_conditioning_tokenizer_name = list(cfg.text_tokenizers.keys())[0]

        # TODO @xueyang: both tokenizers are only used to get some token ids. We
        # should kill them to save a small amount of mem resources since dataloader will initialize them
        # again after the worker processes are spawned.
        self.tokenizer = setup_tokenizers(
            all_tokenizers_config=cfg.text_tokenizers,
            mode='train',
            # Read before super().__init__, which stamps the *current* version into configs that lack one.
            cfg_nemo_version=cfg.get('nemo_version', None),
        )

        num_tokens_tokenizer = len(self.tokenizer.tokens)
        if self.legacy_text_conditioning:
            # Text context tokens are not a part of the the regular transcript embedding table in legacy models
            num_tokens_tokenizer -= self.tokenizer.num_tokens_per_tokenizer[self.text_conditioning_tokenizer_name]

        num_tokens = num_tokens_tokenizer + 2  # +2 for BOS and EOS
        self.bos_id = num_tokens - 2
        self.eos_id = num_tokens - 1

        self.model_type = cfg.get("model_type", None)
        self.pad_context_text_to_max_duration = self.model_type in [
            "decoder_context_tts",
            "decoder_ce",
        ]

        # Below args (text_context_remapping_json, text_context_remapping_prob) are
        # for combining multiple context_texts into a single one during training.
        # Eg. if we want to treat Emma_neutral and Emma_conversational as one speaker,
        # we can create an override dict {'Emma_neutral' : 'Emma', 'Emma_conversational' : 'Emma'}
        # This dict is saved in a json file given by cfg.model.text_context_remapping_json
        # If we want to preserve both behaviours i.e (Emma_neutral, Emma_conversational) and just (Emma)
        # we can do this mapping with a probability during training, as specified by text_context_remapping_prob
        self.text_context_remapping = None
        text_context_remapping_json = cfg.get('text_context_remapping_json', None)
        self.text_context_remapping_prob = cfg.get('text_context_remapping_prob', 0.0)
        if text_context_remapping_json is not None:
            with open(text_context_remapping_json, 'r') as f:
                self.text_context_remapping = json.load(f)

        super().__init__(cfg=cfg, trainer=trainer)

        if self.legacy_text_conditioning:
            tc_tokenizer = self.tokenizer.tokenizers[self.text_conditioning_tokenizer_name]
            tc_vocab_size = tc_tokenizer.vocab_size
            # In transformers v5+, T5Tokenizer is a fast tokenizer whose vocab_size includes
            # extra_id sentinel tokens (e.g. 32100 = 32000 + 100). Subtract them to match
            # the vocab size used when training legacy checkpoints.
            if hasattr(tc_tokenizer, '_extra_ids'):
                tc_vocab_size -= tc_tokenizer._extra_ids
            self.context_text_embedding = nn.Embedding(tc_vocab_size, cfg.embedding_dim)

        # This needs to happen after super().__init__()
        self._codec_model = codec_model
        self._codec_model.freeze()  # Lightning does requires_grad = False and self.eval()
        self._codec_converter = codec_converter
        self._codec_helper = CodecHelper(self._codec_model, self._codec_converter)
        self._streaming_codec_weight_norm_receipt: Optional[WeightNormMaterializationReceipt] = None
        self._first_submission_cuda_graph_runtimes: Dict[
            FirstSubmissionCudaGraphKey, FirstSubmissionCudaGraphRuntime
        ] = {}
        self._first_submission_cuda_graph_receipts: Dict[
            FirstSubmissionCudaGraphKey, FirstSubmissionCudaGraphReceipt
        ] = {}
        self._first_submission_cuda_graph_enabled = False

        audio_embeddings = []
        for _ in range(self.num_audio_codebooks * self.frame_stacking_factor):
            audio_embeddings.append(nn.Embedding(self.num_all_tokens_per_codebook, cfg.embedding_dim))
        self.audio_embeddings = nn.ModuleList(audio_embeddings)

        # Identity projections required by LocalTransformerHelper methods.
        # MagpieTTSModel embeds directly in embedding_dim, so no projection is needed.
        self.audio_in_projection = nn.Identity()
        self.local_transformer_audio_out_projection = nn.Identity()

        if self.use_bpe_char_tokenizer:
            # BPE char tokenizer
            assert len(self.tokenizer.tokenizers) == 1, "BPE char tokenizer should only be used with one tokenizer"
            tokenizer_name = self.tokenizer.tokenizer_names[0]
            tokenizer = self.tokenizer.tokenizers[tokenizer_name]
            subword_vocab = tokenizer.get_vocab()
            # special tokens will be stored as it is in the char_vocab
            # Each special token will only be mapped to one char id
            special_vocab = {
                '<BOS>': self.bos_id,
                '<EOS>': self.eos_id,
            }
            self.cas_encoder = CharAwareSubwordEncoder(
                d_embed=cfg.embedding_dim,
                llm_tokenizer_vocab=subword_vocab,
                subword_padding_idx=self.tokenizer.pad,
                special_vocab=special_vocab,
            )
        else:
            # Regular text embedding
            self.text_embedding = nn.Embedding(num_tokens, cfg.embedding_dim)

        self.encoder = transformer_2501.Transformer(**dict(cfg.encoder))
        self.decoder = transformer_2501.Transformer(**dict(cfg.decoder))

        self.final_proj = nn.Linear(
            cfg.decoder.d_model,
            self.num_audio_codebooks * self.num_all_tokens_per_codebook * self.frame_stacking_factor,
        )

        self.local_transformer_type = LocalTransformerType(cfg.get('local_transformer_type', 'none').lower())
        logging.info(f"Local transformer type: {self.local_transformer_type}")
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            local_transformer_hidden_dim = cfg.get('local_transformer_hidden_dim', 256)
            if local_transformer_hidden_dim != cfg.decoder.d_model:
                self.local_transformer_in_projection = nn.Linear(cfg.decoder.d_model, local_transformer_hidden_dim)
            else:
                self.local_transformer_in_projection = nn.Identity()
            self.local_transformer = transformer_2501.Transformer(
                n_layers=self.cfg.get('local_transformer_n_layers', 2),
                d_model=local_transformer_hidden_dim,
                d_ffn=local_transformer_hidden_dim * 4,
                sa_n_heads=self.cfg.get('local_transformer_n_heads', 1),
                kernel_size=1,
                is_causal=self.local_transformer_type == LocalTransformerType.AR,
                max_length_causal_mask=self.frame_stacking_factor * self.num_audio_codebooks + 2,
                use_learnable_pos_emb=True,
            )
            local_transformer_out_projections = []
            for _ in range(self.num_audio_codebooks * self.frame_stacking_factor):
                # Have a separate projection layer for each codebook, to distinguish between them
                local_transformer_out_projections.append(
                    nn.Linear(local_transformer_hidden_dim, self.num_all_tokens_per_codebook)
                )
            self.local_transformer_out_projections = nn.ModuleList(local_transformer_out_projections)

            self._lt_helper = LocalTransformerHelper(
                local_transformer=self.local_transformer,
                audio_embeddings=self.audio_embeddings,
                audio_in_projection=self.audio_in_projection,
                local_transformer_in_projection=self.local_transformer_in_projection,
                local_transformer_audio_out_projection=self.local_transformer_audio_out_projection,
                local_transformer_out_projections=self.local_transformer_out_projections,
                num_audio_codebooks=self.num_audio_codebooks,
                frame_stacking_factor=self.frame_stacking_factor,
                audio_eos_id=self.audio_eos_id,
                mask_token_id=self.mask_token_id,
                codebook_size=self.codebook_size,
            )

        if cfg.get('use_alignment_encoder', False):
            self.alignment_encoder = AlignmentEncoder(
                n_mel_channels=cfg.embedding_dim,
                n_text_channels=cfg.embedding_dim,
                dist_type="cosine",
                temperature=15.0,
            )

        if self.model_type == 'multi_encoder_context_tts':
            logging.warning(f"The multi_encoder_context_tts model type for {self} is deprecated.")

            # Transcript and context audio/text go to different encoders.
            # Output of the encoders goes to the decoder through the cross-attention layers
            self.transcript_decoder_layers = cfg.get('transcript_decoder_layers', [3, 4, 5, 6, 7, 8])
            self.context_decoder_layers = cfg.get(
                'context_decoder_layers', [0, 1, 2, 9, 10, 11]
            )  # For backward compatibility
            multi_encoder_mapping = [None for _ in range(self.decoder.n_layers)]
            for layer in self.transcript_decoder_layers:
                multi_encoder_mapping[layer] = 0  # 0 means text goes to this layer, 1 means context goes to this layer
            for layer in self.context_decoder_layers:
                multi_encoder_mapping[layer] = 1
            self.multi_encoder_mapping = multi_encoder_mapping
            # Create context encoder.
            # Note: router_* loss coefficients are model-level config, not consumed by the Transformer module.
            context_encoder_cfg = dict(cfg.context_encoder)
            if context_encoder_cfg.get('use_moe', False):
                raise NeMoBaseException(
                    "MoE is not recommended for the context encoder. Please set context_encoder.use_moe to False."
                )
            if 'router_load_balancing_loss_coeff' in context_encoder_cfg:
                logging.warning(
                    "Detected `router_load_balancing_loss_coeff` in context encoder config. "
                    "MoE is not recommended for the context encoder."
                )
            if 'router_z_loss_coeff' in context_encoder_cfg:
                logging.warning(
                    "Detected `router_z_loss_coeff` in context encoder config. "
                    "MoE is not recommended for the context encoder."
                )
            self.context_encoder = transformer_2501.Transformer(**context_encoder_cfg)
        elif self.model_type == 'decoder_context_tts':
            # Context audio/text goes directly to the decoder (before the target audio codes)
            self.transcript_decoder_layers = [
                idx for idx in range(self.decoder.n_layers)
            ]  # All layers are used for text
        elif self.model_type == 'decoder_ce':
            # Similar to decoder_context_tts, but we use context encoder
            # Decoder gets output from context encoder instead of raw context tokens embeddings
            # Note: router_* loss coefficients are model-level config, not consumed by the Transformer module.
            context_encoder_cfg = dict(cfg.context_encoder)
            if context_encoder_cfg.get('use_moe', False):
                raise NeMoBaseException(
                    "MoE is not recommended for the context encoder. Please set context_encoder.use_moe to False."
                )
            if 'router_load_balancing_loss_coeff' in context_encoder_cfg:
                logging.warning(
                    "Detected `router_load_balancing_loss_coeff` in context encoder config. "
                    "MoE is not recommended for the context encoder."
                )
            if 'router_z_loss_coeff' in context_encoder_cfg:
                logging.warning(
                    "Detected `router_z_loss_coeff` in context encoder config. "
                    "MoE is not recommended for the context encoder."
                )
            self.context_encoder = transformer_2501.Transformer(**context_encoder_cfg)
            self.transcript_decoder_layers = [
                idx for idx in range(cfg.decoder.n_layers)
            ]  # All layers are used for text
            # Baked context embedding: nn.Embedding with flattened (N, T*D), reshaped to (N, T, D) at retrieval
            # register_buffer does not work with nn.Embedding, so we use a regular variable.
            self.baked_context_embedding: Optional[nn.Embedding] = None
            self.register_buffer('_baked_embedding_T', None)  # Time dimension
            self.register_buffer('_baked_embedding_D', None)  # Embedding dimension
            self.register_buffer('baked_context_embedding_len', None)  # Per-speaker lengths (N,)
            # Probability of bypassing the context encoder during training and instead feeding
            # batch-shuffled raw context embeddings, so the model learns not to clone voices
            # from untransformed (i.e. not encoded by the context encoder) input.
            self.train_shuffle_context_embedding_prob = cfg.get('train_shuffle_context_embedding_prob', 0.0)
        else:
            raise ValueError(f"Unsupported model type {self.model_type}")

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.alignment_loss_scale = cfg.get('alignment_loss_scale', 0.0)
        self.alignment_encoder_loss_scale = cfg.get('alignment_encoder_loss_scale', 0.0)
        if self.alignment_loss_scale > 0.0:
            self.alignment_loss = ForwardSumLoss(loss_scale=self.alignment_loss_scale)
        if self.alignment_encoder_loss_scale > 0.0:
            self.alignment_encoder_loss = ForwardSumLoss(loss_scale=self.alignment_encoder_loss_scale)

        # Initialize MoE losses if MoE is enabled in decoder
        self.use_moe = cfg.get('use_moe', False)
        if self.use_moe:
            num_experts = cfg.decoder.get('num_experts', 8)
            routing_strategy = cfg.decoder.get('routing_strategy', 'top_k')

            router_load_balancing_loss_coeff = cfg.get('router_load_balancing_loss_coeff', 0.01)
            router_z_loss_coeff = cfg.get('router_z_loss_coeff', 0.001)

            # Sinkhorn routing already ensures balanced expert assignment through its doubly stochastic property
            # Load balancing loss is redundant and incompatible with Sinkhorn
            if routing_strategy == 'sinkhorn' and router_load_balancing_loss_coeff > 0:
                raise ValueError(
                    f"Invalid configuration: routing_strategy='sinkhorn' with router_load_balancing_loss_coeff={router_load_balancing_loss_coeff} > 0. "
                    f"Sinkhorn routing already ensures balanced expert load through doubly stochastic constraints. "
                    f"Set router_load_balancing_loss_coeff=0.0 when using Sinkhorn routing to avoid redundant penalization."
                )

            self.moe_auxiliary_loss = MoEAuxiliaryLoss(
                num_experts=num_experts,
                load_balancing_loss_scale=router_load_balancing_loss_coeff,
                router_z_loss_scale=router_z_loss_coeff,
            )
            logging.info(
                f"MoE enabled in decoder with {num_experts} experts, routing_strategy={routing_strategy}. "
                f"Each expert has d_ffn={cfg.decoder.d_ffn}. "
                f"Loss scales: router_load_balancing={router_load_balancing_loss_coeff}, router_z={router_z_loss_coeff}"
            )
            # Training-side accumulator for layer-wise expert usage heatmap.
            # Accumulated every training_step, rendered + reset at each validation interval.
            self._moe_num_experts = num_experts
            self._moe_train_layer_usage_accum: Optional[torch.Tensor] = None  # (n_layers, num_experts)
            self._moe_train_accum_steps: int = 0

        # Define cfg parameters into self parameters
        self.prior_end_step = self.cfg.prior_end_step
        self.prior_scaledown_start_step = self.cfg.prior_scaledown_start_step
        self.indefinite_prior_prob = self.cfg.get('indefinite_prior_prob', 0.0)
        self.ctc_prior_layer_ids = self.cfg.get('ctc_prior_layer_ids', self.transcript_decoder_layers)
        self.cfg_unconditional_prob = self.cfg.get('cfg_unconditional_prob', 0.0)
        self.decoder_input_dropout_prob = self.cfg.get('decoder_input_dropout_prob', 0.0)
        self.binarize_attn_method = self.cfg.get('binarize_attn_method', 'argmax')
        self.binarize_repeat_audio_factor = self.cfg.get('binarize_repeat_audio_factor', 2)
        self.prior_future_decay = self.cfg.get('prior_future_decay', 1.0)
        self.prior_past_decay = self.cfg.get('prior_past_decay', 1.0)
        self.binarized_prior_epsilon = self.cfg.get('binarized_prior_epsilon', 0.0)
        self.prior_future_context = self.cfg.get('prior_future_context', 1)
        self.prior_past_context = self.cfg.get('prior_past_context', 1)
        self.binarize_prior_after_step = self.cfg.get('binarize_prior_after_step', 0)
        self.codebook_loss_scale = self.cfg.get('codebook_loss_scale', 1.0)
        self.local_transformer_loss_scale = self.cfg.get('local_transformer_loss_scale', 1.0)
        self.use_alignment_encoder = self.cfg.get('use_alignment_encoder', False)
        self.use_prior_for_aligner = self.cfg.get('use_prior_for_aligner', False)
        self.aligner_encoder_train_steps = self.cfg.get('aligner_encoder_train_steps', float('inf'))
        self.dec_random_input_max = self.cfg.get('dec_random_input_max', self.num_all_tokens_per_codebook)

        # Configuration validity checks
        self.check_frame_stacking_config_validity()

        # Class-level cache for text normalizers. Used during inference.
        self._text_normalizers: Dict[str, Any] = {}

    def _register_tokenizer_artifacts(self, cfg: DictConfig) -> None:
        """
        Register tokenizer file artifacts (phoneme_dict, heteronyms, etc.) for .nemo packaging.

        This method iterates through all tokenizer configs and registers any local file paths
        as artifacts. When the model is saved to a .nemo file, these files will be packaged
        inside the archive and automatically restored when loading from .nemo.

        Supported artifact types:
        - g2p.phoneme_dict: Phoneme dictionary file for G2P conversion
        - g2p.heteronyms: Heteronyms file for G2P conversion

        Args:
            cfg: Model configuration containing text_tokenizers config
        """
        if 'text_tokenizers' not in cfg:
            return

        for tokenizer_name in cfg.text_tokenizers:
            tokenizer_cfg = cfg.text_tokenizers[tokenizer_name]

            # Skip HuggingFace tokenizers (AutoTokenizer, T5Tokenizer) - they don't need local files
            if hasattr(tokenizer_cfg, '_target_') and tokenizer_cfg._target_ in ['AutoTokenizer', 'T5Tokenizer']:
                continue

            # Register G2P artifacts if present
            if hasattr(tokenizer_cfg, 'g2p') and tokenizer_cfg.g2p is not None:
                g2p_cfg = tokenizer_cfg.g2p

                # Register phoneme_dict (or resolve nemo: path if restoring from .nemo)
                phoneme_dict_path = (
                    g2p_cfg.get('phoneme_dict', None)
                    if hasattr(g2p_cfg, 'get')
                    else getattr(g2p_cfg, 'phoneme_dict', None)
                )
                if phoneme_dict_path and isinstance(phoneme_dict_path, (list, ListConfig)):
                    # Handle list of phoneme dicts (e.g. Hindi code-switching: hi_prondict + ipa_cmudict)
                    registered = []
                    for i, path_item in enumerate(phoneme_dict_path):
                        if isinstance(path_item, str) and path_item.strip():
                            try:
                                # Use a list-index path (phoneme_dict.{i}, dot) so the connector's
                                # OmegaConf.update writes the element back into the list. With an
                                # underscore (phoneme_dict_{i}) the saved config gets sibling keys
                                # phoneme_dict_0/_1 (and phoneme_dict null), which IpaG2p rejects on
                                # restore ("unexpected keyword argument 'phoneme_dict_0'").
                                artifact_path = self.register_artifact(
                                    f'text_tokenizers.{tokenizer_name}.g2p.phoneme_dict.{i}',
                                    path_item,
                                    verify_src_exists=True,
                                )
                                registered.append(artifact_path if artifact_path else path_item)
                            except FileNotFoundError:
                                logging.warning(
                                    f"phoneme_dict[{i}] file not found for tokenizer '{tokenizer_name}': "
                                    f"{path_item}. Artifact will not be packaged in .nemo file."
                                )
                                registered.append(path_item)
                        else:
                            registered.append(path_item)
                    with open_dict(cfg):
                        cfg.text_tokenizers[tokenizer_name].g2p.phoneme_dict = registered
                elif phoneme_dict_path and isinstance(phoneme_dict_path, str) and phoneme_dict_path.strip():
                    try:
                        # register_artifact handles both:
                        # - Local paths: registers for .nemo packaging, returns absolute path
                        # - nemo: paths: resolves to extracted file location
                        artifact_path = self.register_artifact(
                            f'text_tokenizers.{tokenizer_name}.g2p.phoneme_dict',
                            phoneme_dict_path,
                            verify_src_exists=True,
                        )
                        if artifact_path:
                            with open_dict(cfg):
                                cfg.text_tokenizers[tokenizer_name].g2p.phoneme_dict = artifact_path
                    except FileNotFoundError:
                        logging.warning(
                            f"phoneme_dict file not found for tokenizer '{tokenizer_name}': "
                            f"{phoneme_dict_path}. Artifact will not be packaged in .nemo file."
                        )

                # Register heteronyms (or resolve nemo: path if restoring from .nemo)
                heteronyms_path = (
                    g2p_cfg.get('heteronyms', None)
                    if hasattr(g2p_cfg, 'get')
                    else getattr(g2p_cfg, 'heteronyms', None)
                )
                if heteronyms_path and isinstance(heteronyms_path, str) and heteronyms_path.strip():
                    try:
                        artifact_path = self.register_artifact(
                            f'text_tokenizers.{tokenizer_name}.g2p.heteronyms',
                            heteronyms_path,
                            verify_src_exists=True,
                        )
                        if artifact_path:
                            with open_dict(cfg):
                                cfg.text_tokenizers[tokenizer_name].g2p.heteronyms = artifact_path
                    except FileNotFoundError:
                        logging.warning(
                            f"heteronyms file not found for tokenizer '{tokenizer_name}': "
                            f"{heteronyms_path}. Artifact will not be packaged in .nemo file."
                        )

    def _setup_inference_parameters(self, cfg: DictConfig) -> None:
        """
        Create the self.inference_parameters which instantiates the InferenceParameters dataclass
        """
        self.inference_parameters = ModelInferenceParameters.from_dict(cfg.get("inference_parameters", {}))

    def _get_state_dict_keys_to_exclude(self):
        """
        We remove _speaker_verification_model and _codec_model
        from the checkpoint and optimizer param groups. The codec model is saved in a separate checkpoint.
        _speaker_verification_model is only included in older checkpoints with the older single_encoder_sv_tts
        model_type that is no longer supported and can likely be removed in a future version.
        If the model has a baked context embedding, the context_encoder weights are also excluded
        since they are no longer needed for inference.
        """
        keys = ['_speaker_verification_model', '_codec_model']
        if self.has_baked_context_embedding:
            keys.append('context_encoder')
        return keys

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Only used for saving checkpoints.
        We exclude the keys in the state_dict that are in the list returned by _get_state_dict_keys_to_exclude.
        """
        if hasattr(self, '_no_state_dict') and self._no_state_dict:
            return {}
        state_dict = super().state_dict(destination, prefix, keep_vars)
        keys_substrings_to_exclude = self._get_state_dict_keys_to_exclude()
        for key in list(state_dict.keys()):
            if any(substring in key for substring in keys_substrings_to_exclude):
                del state_dict[key]
        return state_dict

    def setup_optimizer_param_groups(self):
        """Exclude frozen eval/inference-only models from the optimizer.
        Saves memory by excluding the keys in the state_dict that are in the list returned by _get_state_dict_keys_to_exclude.
        """
        modules_to_exclude = set(self._get_state_dict_keys_to_exclude())

        excluded_param_ids = set()
        for name, module in self.named_children():
            if name in modules_to_exclude:
                for param in module.parameters():
                    excluded_param_ids.add(id(param))

        trainable_params = [p for p in self.parameters() if id(p) not in excluded_param_ids]

        logging.info(
            f"setup_optimizer_param_groups: {len(trainable_params)} params in optimizer, "
            f"{len(excluded_param_ids)} params excluded (eval models)"
        )

        self._optimizer_param_groups = [{"params": trainable_params}]

    def check_frame_stacking_config_validity(self):
        """
        Check if the configuration is compatible with frame stacking.
        """
        if self.frame_stacking_factor > 1:
            # Reject configurations that are not supported with frame stacking.
            # Some of them may work - but they have not been tested.

            # disallow alignment encoder
            if self.use_alignment_encoder:
                raise ValueError("Alignment encoder is not supported for frame stacking")
            # disallow alignment loss
            if self.alignment_loss_scale > 0.0:
                raise ValueError("Alignment loss is not supported for frame stacking")
            # disallow training prior
            if self.cfg.prior_scaling_factor is not None and self.cfg.prior_scaling_factor > 0:
                raise ValueError("Training-time attention prior is not supported for frame stacking")
            # With frame stacking, the audio context sequence length is divided by the
            # frame stacking factor (e.g., 108 tokens at 21fps --> 54 positions with 2x stacking).
            # The text context is NOT stacked but must fit within the same sequence length
            # as the audio context. If needed, this constraint could be likey be removed by also
            # stacking the text context, but that would require some experimentation.
            if self.use_text_conditioning_encoder:
                # Use 5 seconds as the baseline context length since it is known to fit
                # existing text contexts.
                min_required_context_sec = 5.0 * self.frame_stacking_factor
                actual_context_length_sec = self.cfg.get('context_duration_max')
                if actual_context_length_sec < min_required_context_sec:
                    raise ValueError(
                        f"With text context and a frame stacking factor of {self.frame_stacking_factor}, "
                        f"context_duration_max must be >= {min_required_context_sec} seconds "
                        f"(5 seconds x frame_stacking_factor); got context_duration_max={actual_context_length_sec}"
                    )

    @property
    def has_baked_context_embedding(self) -> bool:
        """Check if the model has a baked context embedding.

        Returns:
            True if baked_context_embedding is set with valid dimensions.
        """
        return (
            self.model_type == 'decoder_ce'
            and self.baked_context_embedding is not None
            and self._baked_embedding_T is not None
            and self._baked_embedding_D is not None
        )

    @property
    def num_baked_speakers(self) -> int:
        """Return number of baked speakers.

        Returns:
            0 if no baked embedding, N for embedding with N speakers.
        """
        if not self.has_baked_context_embedding:
            return 0
        return self.baked_context_embedding.num_embeddings

    @property
    def validation_step_outputs(self):
        """Always use list-of-lists structure for uniform single/multi-dataloader handling.

        Overrides ModelPT which uses a flat list for single dataloader and list-of-lists
        for multiple dataloaders. This override always returns list-of-lists so that
        validation_step, on_validation_epoch_end, etc. don't need conditional branching.
        """
        if self._validation_step_outputs is not None:
            return self._validation_step_outputs
        num_dl = len(self._validation_dl) if self._validation_dl is not None else 1
        self._validation_step_outputs = [[] for _ in range(num_dl)]
        return self._validation_step_outputs

    @validation_step_outputs.setter
    def validation_step_outputs(self, value):
        self._validation_step_outputs = value

    def _normalize_speaker_indices(
        self,
        speaker_indices: Optional[Union[int, List[int], torch.Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize speaker_indices to a tensor of shape (batch_size,).

        Args:
            speaker_indices: Speaker selection. Can be:
                - None: Use first speaker (index 0) for all batch elements
                - int: Same speaker for all batch elements
                - List[int] or Tensor: One speaker index per batch element
            batch_size: Number of elements in the batch.
            device: Device to create tensor on.

        Returns:
            Tensor of shape (batch_size,) with speaker indices.

        Raises:
            ValueError: If speaker_indices length doesn't match batch_size or indices are out of range.
        """
        # Default to first speaker (index 0) if none specified
        if speaker_indices is None:
            speaker_indices = 0

        # Normalize to tensor
        if isinstance(speaker_indices, int):
            indices = torch.full((batch_size,), speaker_indices, dtype=torch.long, device=device)
        elif isinstance(speaker_indices, list):
            if len(speaker_indices) != batch_size:
                raise ValueError(
                    f"speaker_indices length ({len(speaker_indices)}) must match batch_size ({batch_size})"
                )
            indices = torch.tensor(speaker_indices, dtype=torch.long, device=device)
        elif isinstance(speaker_indices, torch.Tensor):
            if speaker_indices.numel() != batch_size:
                raise ValueError(
                    f"speaker_indices length ({speaker_indices.numel()}) must match batch_size ({batch_size})"
                )
            indices = speaker_indices.to(device=device, dtype=torch.long)
        else:
            raise ValueError(f"speaker_indices must be int, list, or tensor, got {type(speaker_indices)}")

        # Validate indices
        if (indices < 0).any() or (indices >= self.num_baked_speakers).any():
            raise ValueError(
                f"speaker_indices values must be in range [0, {self.num_baked_speakers - 1}], "
                f"got min={indices.min().item()}, max={indices.max().item()}"
            )

        return indices

    def get_baked_context_embeddings_batch(
        self,
        batch_size: int,
        speaker_indices: Optional[Union[int, List[int], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get baked context embeddings for a batch, with per-element speaker selection.

        Args:
            batch_size: Number of elements in the batch.
            speaker_indices: Speaker selection. Can be:
                - None: Use first speaker (index 0) for all batch elements
                - int: Same speaker for all batch elements
                - List[int] or Tensor: One speaker index per batch element (length must match batch_size)

        Returns:
            Tuple of (embeddings, lengths) where:
                - embeddings: (B, T, D) tensor
                - lengths: (B,) tensor with embedding lengths per batch element

        Raises:
            ValueError: If speaker_indices length doesn't match batch_size or indices are out of range.
        """
        if not self.has_baked_context_embedding:
            raise ValueError("No baked context embedding available")

        device = self.baked_context_embedding.weight.device
        indices = self._normalize_speaker_indices(speaker_indices, batch_size, device)

        # Lookup flattened embeddings via nn.Embedding: (B,) -> (B, T*D)
        flat_embeddings = self.baked_context_embedding(indices)

        # Reshape to 3D: (B, T*D) -> (B, T, D)
        T = self._baked_embedding_T.item()
        D = self._baked_embedding_D.item()
        embeddings = flat_embeddings.view(batch_size, T, D)

        lengths = self.baked_context_embedding_len[indices]  # (B,)
        return embeddings, lengths

    def update_ckpt(self, state_dict):
        """
        Backward compatibility for checkpoints saved with old model names.
        """
        new_state_dict = {}
        for key in state_dict.keys():
            if 't5_encoder' in key:
                new_key = key.replace('t5_encoder', 'encoder')
                new_state_dict[new_key] = state_dict[key]
            elif 't5_decoder' in key:
                new_key = key.replace('t5_decoder', 'decoder')
                new_state_dict[new_key] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        return new_state_dict

    def _apply(self, fn, recurse=True):
        """Invalidate captured inference graphs before moving model storage."""

        if hasattr(self, "_first_submission_cuda_graph_runtimes"):
            self.invalidate_first_submission_cuda_graphs()
        if hasattr(self, "_lt_helper"):
            self._lt_helper.invalidate_autoregressive_graphs()
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, state_dict, strict=True):
        """
        Modify load_state_dict so that we don't restore weights to _speaker_verification_model and _codec_model when
        strict is True.
        When strict is False, we can call pytorch's load_state_dict.
        When strict is True, we loop through all parameters and rename them to enable loading.

        _speaker_verification_model is only included in older checkpoints with the older single_encoder_sv_tts
        model_type that is no longer supported and can likely be removed in a future version.

        Also handles loading baked context embeddings. If the checkpoint contains baked_speaker_embedding.weight,
        context_encoder weights are not expected to be present. The embedding is stored in flattened format
        (N, T*D) and reconstructed to (N, T, D) at inference time using stored T and D dimensions.
        """
        if hasattr(self, "_first_submission_cuda_graph_runtimes"):
            self.invalidate_first_submission_cuda_graphs()
        if hasattr(self, "_lt_helper"):
            self._lt_helper.invalidate_autoregressive_graphs()
        state_dict = self.update_ckpt(state_dict)
        # `text_embedding` is absent on the CAS-encoder variant, which has no such table to compare.
        check_text_embedding_matches_tokenizer(
            state_dict,
            text_embedding=getattr(self, 'text_embedding', None),
            tokenizer=self.tokenizer,
            model_cfg=self.cfg,
        )

        # Check if checkpoint has baked context embedding (nn.Embedding format)
        has_baked_embedding_in_ckpt = 'baked_context_embedding.weight' in state_dict

        # Load baked embedding if present
        if has_baked_embedding_in_ckpt:
            weight = state_dict['baked_context_embedding.weight']  # (N, T*D)
            self._baked_embedding_T = state_dict['_baked_embedding_T']
            self._baked_embedding_D = state_dict['_baked_embedding_D']
            self.baked_context_embedding_len = state_dict['baked_context_embedding_len']

            num_speakers = weight.size(0)
            embedding_dim = weight.size(1)
            T = self._baked_embedding_T.item()
            D = self._baked_embedding_D.item()

            # Create nn.Embedding and load weights (no gradients for inference)
            self.baked_context_embedding = nn.Embedding(num_speakers, embedding_dim)
            self.baked_context_embedding.weight.data = weight
            self.baked_context_embedding.weight.requires_grad_(False)

            logging.info(
                f"Loaded baked context embedding: num_speakers={num_speakers}, T={T}, D={D}, "
                f"shape=({num_speakers}, {embedding_dim}), lengths={self.baked_context_embedding_len.tolist()}"
            )

        if not strict:
            super().load_state_dict(state_dict, strict=False)

        # Build list of modules to skip
        modules_to_skip = [
            '_speaker_verification_model',
            '_codec_model',
            '_reference_model',
            'eval_asr_model',
            'eval_speaker_verification_model',
            'whisper_model',
            'squim_objective_model',
            '_teacher_model',
        ]
        # Skip context_encoder if checkpoint has baked embedding (weights won't be in checkpoint)
        if has_baked_embedding_in_ckpt:
            modules_to_skip.append('context_encoder')

        for name, child in self.named_children():
            if name in modules_to_skip:
                continue
            if any(param.numel() > 0 for param in child.parameters()):
                # If the module has parameters, we want to change the default mapping so that the state_dict gets
                # loaded.
                # Ex: state_dict[encoder.position_embeddings.weight] -> new_state_dict[position_embeddings.weight]
                new_state_dict = {}
                for key in state_dict.keys():
                    name_with_dot = f"{name}."
                    if key.startswith(name_with_dot):
                        new_state_dict[key[len(name_with_dot) :]] = state_dict[key]
                child.load_state_dict(new_state_dict)

    def embed_audio_tokens(self, audio_tokens, audio_tokens_lens):
        B, C, T = audio_tokens.shape
        audio_tokens = pad_audio_codes(audio_tokens, self.frame_stacking_factor).long()
        audio_embedding = None
        for i in range(self.frame_stacking_factor):
            for c in range(C):
                tokens = audio_tokens[:, c, i :: self.frame_stacking_factor]
                embedding = self.audio_embeddings[c + i * C](tokens)
                if audio_embedding is None:
                    audio_embedding = embedding
                else:
                    audio_embedding += embedding
        audio_embedding = audio_embedding / (C * self.frame_stacking_factor)  # [B, T, E]

        audio_embedding_lens = torch.ceil(audio_tokens_lens / self.frame_stacking_factor).long()
        mask = get_mask_from_lengths(audio_embedding_lens)
        audio_embedding = audio_embedding * mask.unsqueeze(2)

        return audio_embedding, audio_embedding_lens

    def compute_loss(
        self,
        logits,
        audio_codes,
        audio_codes_lens,
        mask_tokens_mask=None,
        frame_stacking_factor=1,
    ):
        """
        Computes the audio codebook loss. Used by:

        (1) The main Magpie-TTS transformer
        (2) The local transformer, for both autoregressive and MaskGit methods

        Args:
            logits: (B, T', num_codebooks * num_tokens_per_codebook)
            audio_codes: (B, C, T')
            audio_codes_lens: (B,)
            mask_tokens_mask: (B, C, T') True for tokens that were replaced with the MASK_TOKEN and should
                therefore be the only ones included in the loss computation (for MaskGit).
            frame_stacking_factor: int, the stacking factor used in the model
        """
        loss_mask = get_mask_from_lengths(audio_codes_lens, pad_to_factor=frame_stacking_factor)
        if mask_tokens_mask is not None:
            # For MaskGit we only compute loss for the masked tokens.
            # *Both* conditions must be true:
            # 1. the token is masked
            # 2. the token is not padding
            loss_mask = loss_mask.unsqueeze(1) * mask_tokens_mask
            if not loss_mask.any():
                # Without this we were very rarely getting NaNs in the loss
                logging.warning("No tokens valid were found in compute_loss()!")
                return torch.tensor(0.0, device=loss_mask.device), loss_mask
        else:
            # repeat loss mask for each codebook to simplify code below
            loss_mask = loss_mask.unsqueeze(1).repeat(1, audio_codes.size(1), 1)
        total_codebook_loss = None
        audio_codes = pad_audio_codes(audio_codes, self.frame_stacking_factor).long()
        for fs_index in range(frame_stacking_factor):
            for codebook in range(audio_codes.size(1)):
                si = (codebook + self.num_audio_codebooks * fs_index) * self.num_all_tokens_per_codebook
                ei = si + self.num_all_tokens_per_codebook
                codebook_logits = logits[:, :, si:ei]  # (B, T', num_tokens_per_codebook)
                codebook_targets = audio_codes[:, codebook, fs_index::frame_stacking_factor]  # (B, T')
                codebook_loss = self.cross_entropy_loss(
                    codebook_logits.permute(0, 2, 1),
                    codebook_targets,  # (B, num_tokens_per_codebook, T')
                )  # (B, T')
                codebook_loss_mask = loss_mask[:, codebook, fs_index::frame_stacking_factor]
                codebook_loss = codebook_loss * codebook_loss_mask
                if codebook_loss_mask.sum() == 0:
                    logging.warning(f"Loss mask for codebook {codebook} is all zeros, global_step: {self.global_step}")
                    continue
                codebook_loss = codebook_loss.sum() / codebook_loss_mask.sum()
                if total_codebook_loss is None:
                    total_codebook_loss = codebook_loss
                else:
                    total_codebook_loss = total_codebook_loss + codebook_loss

        total_codebook_loss = total_codebook_loss / (audio_codes.size(1) * frame_stacking_factor)
        return total_codebook_loss, loss_mask

    def forward(
        self,
        dec_input_embedded,
        dec_input_mask,
        cond,
        cond_mask,
        attn_prior,
        multi_encoder_mapping,
    ):
        """
        Forward pass through the decoder transformer, followed by a linear projection to audio codebook logits.

        Args:
            dec_input_embedded (torch.Tensor): Embedded decoder input of shape (B, T, C).
            dec_input_mask (torch.Tensor): Boolean mask for decoder input of shape (B, T).
            cond (torch.Tensor or List[torch.Tensor]): Conditioning tensor(s) for cross-attention.
            cond_mask (torch.Tensor or List[torch.Tensor]): Mask(s) for conditioning tensor(s).
            attn_prior (torch.Tensor or None): Prior attention weights for cross-attention.
            multi_encoder_mapping (List[Optional[int]] or None): Per-layer mapping to conditioning inputs.

        Returns:
            Tuple of:

            - all_code_logits (torch.Tensor): Logits of shape (B, T', num_codebooks * num_tokens_per_codebook).
            - attn_probabilities (list): Attention probabilities from each decoder layer.
            - dec_output (torch.Tensor): Raw decoder output of shape (B, T', d_model).
            - moe_routing_info (list or None): None if MoE is disabled. If MoE is enabled,
              a list of dicts (one per layer) each containing:

              - 'router_logits' (torch.Tensor): Raw router logits (B, T, num_experts).
              - 'router_probs' (torch.Tensor): Router probabilities (B, T, num_experts).
              - 'expert_indices' (torch.Tensor): Selected expert indices (B, T, top_k).
        """
        decoder_out = self.decoder(
            dec_input_embedded,
            dec_input_mask,
            cond=cond,
            cond_mask=cond_mask,
            attn_prior=attn_prior,
            multi_encoder_mapping=multi_encoder_mapping,
        )
        attn_probabilities = decoder_out['attn_probabilities']
        moe_routing_info = decoder_out.get('moe_routing_info', None)  # Extract MoE routing info for loss computation
        all_code_logits = self.final_proj(decoder_out['output'])  # (B, T', num_codebooks * num_tokens_per_codebook)
        return all_code_logits, attn_probabilities, decoder_out['output'], moe_routing_info

    def logits_to_audio_codes(self, all_code_logits, audio_codes_lens):
        # all_code_logits: (B, T', num_codebooks * num_tokens_per_codebook)
        # audio_codes_lens: (B,)
        all_preds = [[] for _ in range(self.frame_stacking_factor)]
        for fs_index in range(self.frame_stacking_factor):
            for idx in range(self.num_audio_codebooks):
                si = (idx + self.num_audio_codebooks * fs_index) * self.num_all_tokens_per_codebook
                ei = si + self.num_all_tokens_per_codebook
                codebook_logits = all_code_logits[:, :, si:ei]
                codebook_probs = torch.softmax(codebook_logits, dim=-1)  # (B, T', num_tokens_per_codebook)
                # argmax to get the tokens
                codebook_preds = torch.argmax(codebook_probs, dim=-1)  # (B, T')
                all_preds[fs_index].append(codebook_preds)
        all_preds = [
            torch.stack(p, dim=1) for p in all_preds
        ]  # list of `frame_stacking_factor`` elements of shape (B,C,T) each
        all_preds = torch.stack(all_preds, dim=-1)  # B, C, T, frame_stacking_factor
        # undo the frame stacking
        all_preds = all_preds.reshape(all_preds.size(0), all_preds.size(1), -1)  # B, C, T*frame_stacking_factor
        pred_max_len = all_preds.size(2)
        real_max_len = audio_codes_lens.max()
        assert (pred_max_len - real_max_len) < self.frame_stacking_factor
        # trim padding introduced for frame stacking
        all_preds = all_preds[:, :, :real_max_len]
        audio_mask = get_mask_from_lengths(audio_codes_lens)
        all_preds = all_preds * audio_mask.unsqueeze(1)

        return all_preds

    def visualize_codes(self, codes, mask_id=2020, frame_stacking_rate=2):
        """
        Visualize codes for analysis purposes
        codes: (B, C)
        """

        def code_to_str(code):
            if code == mask_id:
                return "M    "
            else:
                return f"{code:04d} "

        B, C = codes.shape
        if B > 1:
            logging.debug("Warning: visualizing only first batch element")
        codes = codes.clone().detach().cpu().numpy()[0]
        codes = [code_to_str(c) for c in codes]
        output_str = ""
        for i, c in enumerate(codes):
            if (i) % (C / frame_stacking_rate) == 0:
                output_str += "|timestep| "
            output_str += c
        logging.debug(output_str)

    def sample_codes_from_logits(
        self,
        all_code_logits_t: torch.Tensor,
        temperature: float = 0.7,
        topk: int = 80,
        unfinished_items: Dict[int, bool] = {},
        finished_items: Dict[int, bool] = {},
        forbid_audio_eos: bool = False,
    ) -> torch.Tensor:
        """
        Sample codes for all codebooks at a given timestep. Uses multinomial sampling
        with temperature and top-k. If frame stacking is on (i.e. `frame_stacking_factor
        > 1`), this function will sample across the entire frame stack.

        Special handling:
        * forbids special tokens (like AUDIO_BOS, AUDIO_CONTEXT_EOS, etc.) from being sampled
        * forces / forbids EOS for finished / unfinished items respectively
        * optionally, globally forbids audio EOS (useful early in the generation process)

        Args:
            all_code_logits_t (torch.Tensor): Logits at a given timestep with shape
                (B, num_tokens_per_codebook * num_codebooks * frame_stacking_factor)
            temperature (float, optional): Sampling temperature
            topk (int, optional): Number of top-probability tokens to consider in sampling.
            unfinished_items (dict, optional): Dictionary containing indices of batch
            items that we are confident have not completed generation. For these items, audio EOS
                sampling is forbidden.
            finished_items (dict, optional): Dictionary containing indices of batch
                items that we are confident are completed. For these items, audio EOS sampling
                is forced.
            forbid_audio_eos (bool, optional): Whether to globally forbid audio EOS for the entire
                batch.

        Returns:
            torch.Tensor: Sampled audio codes with shape (B, num_codebooks, frame_stacking_factor).
        """
        all_preds = [[] for _ in range(self.frame_stacking_factor)]
        for fs_index in range(self.frame_stacking_factor):
            for idx in range(self.num_audio_codebooks):
                si = (idx + self.num_audio_codebooks * fs_index) * self.num_all_tokens_per_codebook
                ei = si + self.num_all_tokens_per_codebook
                codebook_logits = all_code_logits_t[:, si:ei]  # (B, num_tokens_per_codebook)

                for item_idx in unfinished_items:
                    codebook_logits[item_idx, self.audio_eos_id] = float('-inf')
                for item_idx in finished_items:
                    codebook_logits[item_idx, :] = float('-inf')
                    codebook_logits[item_idx, self.audio_eos_id] = 0.0

                # Disallow generation of special tokens
                codebook_logits = clear_forbidden_logits(
                    codebook_logits.unsqueeze(1),
                    self.codebook_size,
                    forbid_audio_eos=forbid_audio_eos,
                ).squeeze(1)

                codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]  # (B, topk)
                indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(
                    -1
                )  # (B, num_tokens_per_codebook)
                codebook_logits_rescored = codebook_logits.clone()
                codebook_logits_rescored[indices_to_remove] = float('-inf')

                codebook_probs = torch.softmax(
                    codebook_logits_rescored / temperature, dim=-1
                )  # (B, num_tokens_per_codebook)
                codebook_preds = torch.multinomial(codebook_probs, 1)  # (B, 1)
                all_preds[fs_index].append(codebook_preds)

        all_preds = [
            torch.cat(ds_preds, dim=1) for ds_preds in all_preds
        ]  # list of `frame_stacking_factor` elements, each of shape (B, num_codebooks)
        all_preds = torch.stack(all_preds, dim=2)  # (B, num_codebooks, frame_stacking_factor)
        return all_preds

    def _prepare_attention_images(
        self,
        attention_prob_matrix: List[torch.Tensor],
        audio_codes_lens: torch.Tensor,
        text_lens: torch.Tensor,
        dec_context_size: int = 0,
        max_examples: int = 3,
    ) -> List[np.ndarray]:
        """
        Convert attention probability matrices to numpy images for logging.

        Args:
            attention_prob_matrix: List of attention tensors, each (B, H, audio_timesteps, text_timesteps).
            audio_codes_lens: Audio sequence lengths per example.
            text_lens: Text sequence lengths per example.
            dec_context_size: Number of context audio frames to skip in attention visualization.
            max_examples: Maximum number of examples to generate images for.

        Returns:
            List of numpy arrays in HWC format, one per example.
        """
        with torch.no_grad():
            # Concatenate attention heads and average
            attention_prob_matrix = torch.cat(attention_prob_matrix, dim=1)  # (B, C, audio_timesteps, text_timesteps)
            attention_prob_matrix_mean = attention_prob_matrix.mean(dim=1)  # (B, audio_timesteps, text_timesteps)

            images = []
            num_examples = min(max_examples, attention_prob_matrix_mean.size(0))
            for idx in range(num_examples):
                # Slice attention matrix to valid region (excluding context frames)
                audio_len = int(audio_codes_lens[idx])
                text_len = int(text_lens[idx])
                item_attn_matrix = attention_prob_matrix_mean[idx][
                    dec_context_size : dec_context_size + audio_len, :text_len
                ]
                item_attn_matrix = item_attn_matrix.detach().cpu().numpy()
                img_np = plot_alignment_to_numpy(item_attn_matrix.T)
                images.append(img_np)

            return images

    def _prepare_audio_examples(
        self,
        logits: torch.Tensor,
        target_audio_codes: torch.Tensor,
        audio_codes_lens: torch.Tensor,
        context_audio_codes: Optional[torch.Tensor] = None,
        context_audio_codes_lens: Optional[torch.Tensor] = None,
        max_examples: int = 3,
    ) -> Dict[str, List[Optional[np.ndarray]]]:
        """
        Decode audio codes to waveforms and convert to numpy arrays for logging.

        Args:
            logits: Model output logits to convert to predicted audio.
            target_audio_codes: Ground truth audio codes.
            audio_codes_lens: Lengths of target audio codes.
            context_audio_codes: Optional context audio codes for voice cloning.
            context_audio_codes_lens: Lengths of context audio codes.
            max_examples: Maximum number of examples to process.

        Returns:
            Dict with keys 'pred_audios', 'target_audios', 'context_audios',
            each containing a list of numpy arrays (or None for context if unavailable).
        """
        with torch.no_grad():
            # Decode predictions: convert logits to codes, remove EOS token, then decode to audio
            pred_audio_codes = self.logits_to_audio_codes(logits, audio_codes_lens)
            pred_audio_codes, pred_audio_codes_lens = remove_eos_token(
                codes=pred_audio_codes, codes_len=audio_codes_lens
            )
            pred_audio, pred_audio_lens, _ = self._codec_helper.codes_to_audio(pred_audio_codes, pred_audio_codes_lens)

            # Decode targets: remove EOS token, then decode to audio
            target_audio_codes, target_audio_codes_lens = remove_eos_token(
                codes=target_audio_codes, codes_len=audio_codes_lens
            )
            target_audio, target_audio_lens, _ = self._codec_helper.codes_to_audio(
                target_audio_codes, target_audio_codes_lens
            )

            # Decode context audio if available (shape check ensures it's not a dummy tensor used in text context)
            # This does not handle the case in which a batch has a mixture of text and audio context examples
            context_audio, context_audio_lens = None, None
            if context_audio_codes is not None and context_audio_codes.shape[2] > 3:
                context_audio_codes, context_audio_codes_lens = remove_special_tokens(
                    codes=context_audio_codes, codes_len=context_audio_codes_lens
                )
                context_audio, context_audio_lens, _ = self._codec_helper.codes_to_audio(
                    context_audio_codes, context_audio_codes_lens
                )

            pred_audios = []
            target_audios = []
            context_audios = []

            num_examples = min(max_examples, pred_audio.size(0))
            for idx in range(num_examples):
                # Convert to numpy and trim to actual length
                pred_audio_np = pred_audio[idx, : pred_audio_lens[idx]].float().cpu().numpy()
                target_audio_np = target_audio[idx, : target_audio_lens[idx]].float().cpu().numpy()

                pred_audios.append(pred_audio_np)
                target_audios.append(target_audio_np)

                if context_audio is not None:
                    context_audio_np = context_audio[idx, : context_audio_lens[idx]].float().cpu().numpy()
                    context_audios.append(context_audio_np)
                else:
                    context_audios.append(None)

            return {
                'pred_audios': pred_audios,
                'target_audios': target_audios,
                'context_audios': context_audios,
            }

    def _collect_wandb_media_and_log_tb(
        self,
        *,
        dataset_prefix: str,
        pred_audios: List[np.ndarray],
        target_audios: List[np.ndarray],
        context_audios: List[Optional[np.ndarray]],
        attention_data: Dict[str, List[np.ndarray]],
        global_step: int,
    ) -> Dict[str, Any]:
        """
        Collect WandB media entries and log audio/attention to TensorBoard.

        TensorBoard logging happens directly within this method.
        WandB media is returned as a dict to be merged with other WandB media
        (e.g., MoE heatmaps) into a single wandb.log() call by the caller,
        ensuring all media shares the same WandB step index.

        Args:
            dataset_prefix: Prefix for log keys (e.g., 'val', 'val_set_0').
            pred_audios: List of predicted audio waveforms as numpy arrays.
            target_audios: List of target audio waveforms as numpy arrays.
            context_audios: List of context audio waveforms (or None per entry if unavailable).
            attention_data: Dict mapping attention names to lists of numpy images.
            global_step: Current training step for logging.

        Returns:
            Dict of WandB-ready media entries (audio + attention images).
            Empty dict if no WandB logger is configured.
        """
        wandb_media: Dict[str, Any] = {}

        for logger in self.loggers:
            is_wandb = isinstance(logger, WandbLogger)
            is_tb = isinstance(logger, TensorBoardLogger)
            if not is_wandb and not is_tb:
                raise ValueError(
                    f"Unsupported logger type: {type(logger)}. "
                    f"Only WandbLogger and TensorBoardLogger are supported for media logging."
                )

            for idx, (pred_audio_np, target_audio_np, context_audio_np) in enumerate(
                zip(pred_audios, target_audios, context_audios)
            ):
                if is_wandb:
                    audio_list = []
                    if context_audio_np is not None and context_audio_np.shape[0] > 0:
                        audio_list.append(
                            wandb.Audio(
                                context_audio_np,
                                sample_rate=self.output_sample_rate,
                                caption="context",
                            )
                        )
                    audio_list.append(
                        wandb.Audio(
                            pred_audio_np,
                            sample_rate=self.output_sample_rate,
                            caption="prediction",
                        )
                    )
                    audio_list.append(
                        wandb.Audio(
                            target_audio_np,
                            sample_rate=self.output_sample_rate,
                            caption="target",
                        )
                    )
                    wandb_media[f"Audio:{dataset_prefix}/Example_{idx:02d}"] = audio_list

                if is_tb:
                    if context_audio_np is not None and context_audio_np.shape[0] > 0:
                        logger.experiment.add_audio(
                            f'{dataset_prefix}/Example_{idx}/context',
                            context_audio_np,
                            global_step=global_step,
                            sample_rate=self.output_sample_rate,
                        )
                    logger.experiment.add_audio(
                        f'{dataset_prefix}/Example_{idx}/prediction',
                        pred_audio_np,
                        global_step=global_step,
                        sample_rate=self.output_sample_rate,
                    )
                    logger.experiment.add_audio(
                        f'{dataset_prefix}/Example_{idx}/target',
                        target_audio_np,
                        global_step=global_step,
                        sample_rate=self.output_sample_rate,
                    )

            # Log attention images
            for attn_key, images in attention_data.items():
                # Determine log prefix: 'overall' uses dataset_prefix directly, others are nested
                if attn_key == 'overall':
                    prefix = dataset_prefix
                else:
                    prefix = f"{dataset_prefix}/{attn_key}"

                if is_wandb:
                    wandb_media[f"Image:{prefix}/attention_matrix"] = [
                        wandb.Image(img_np, caption=f"Example_{idx:02d}") for idx, img_np in enumerate(images)
                    ]

                if is_tb:
                    for idx, img_np in enumerate(images):
                        logger.experiment.add_image(
                            f'{prefix}/attention_matrix/Example_{idx:02d}',
                            img_np,
                            global_step=global_step,
                            dataformats="HWC",
                        )

        return wandb_media

    def scale_prior(self, prior, global_step):
        if prior is None:
            return None
        if global_step < self.prior_scaledown_start_step:
            return prior
        elif global_step >= self.prior_end_step:
            if random.random() < self.indefinite_prior_prob:
                print("Using Prior")
                return prior
            else:
                print("Not using Prior")
                return None
        else:
            with torch.no_grad():
                # Interpolate between all ones and the prior
                residual = 1.0 - prior
                new_prior = prior + (
                    residual
                    * (global_step - self.prior_scaledown_start_step)
                    / (self.prior_end_step - self.prior_scaledown_start_step)
                )
                return new_prior

    def embed_text(self, text, text_mask):
        if self.use_bpe_char_tokenizer:
            text_embedded = self.cas_encoder(text, subword_mask=text_mask)
        else:
            text_embedded = self.text_embedding(text)

        return text_embedded

    def compute_alignment_loss(self, attention_scores, text_lens, audio_lens, dec_context_size=0):
        # attention scores: List of (B, C, audio_timesteps, text_timesteps)
        attention_scores_combined = torch.cat(attention_scores, dim=1)  # (B, C, audio_timesteps, text_timesteps)
        attention_scores_mean = attention_scores_combined.mean(
            dim=1, keepdim=True
        )  # (B, 1, audio_timesteps, text_timesteps)
        attention_scores_mean = attention_scores_mean[
            :, :, dec_context_size:, :
        ]  # Remove the context audio embeddings from the attention scores
        alignment_loss = self.alignment_loss(
            attn_logprob=attention_scores_mean, in_lens=text_lens, out_lens=audio_lens
        )
        return alignment_loss

    def embed_context_text(self, context_text_tokens):
        if self.legacy_text_conditioning:
            context_text_tokens = (
                context_text_tokens - self.tokenizer.tokenizer_offsets[self.text_conditioning_tokenizer_name]
            )
            context_text_embedded = self.context_text_embedding(context_text_tokens)  # (B, L, E)
        else:
            context_text_embedded = self.text_embedding(context_text_tokens)  # (B, L, E)

        return context_text_embedded

    def _encode_text_input(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text from batch.

        Args:
            batch: Dictionary containing 'text' and 'text_lens'.

        Returns:
            Tuple of (text, text_lens, text_mask, text_embedded, text_encoder_out).
        """
        text = batch['text']
        text_lens = batch['text_lens']
        text_mask = get_mask_from_lengths(text_lens)  # (B, T)
        text_embedded = self.embed_text(text, text_mask)  # (B, T, E)
        text_encoder_out = self.encoder(text_embedded, text_mask, cond=None, cond_mask=None)['output']  # (B, T, E)
        return text, text_lens, text_mask, text_embedded, text_encoder_out

    def _get_context_audio_codes(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract or compute context audio codes from batch.

        Args:
            batch: Dictionary containing either 'context_audio_codes' or 'context_audio'.

        Returns:
            Tuple of (context_audio_codes, context_audio_codes_lens) where codes are
            padded according to frame_stacking_factor.
        """
        if 'context_audio_codes' in batch:
            codes = batch['context_audio_codes']
            lens = batch['context_audio_codes_lens']
        else:
            codes, lens = self._codec_helper.audio_to_codes(
                batch['context_audio'],
                batch['context_audio_lens'],
                sample_rate=batch.get('context_sample_rate'),
            )

        if self._codec_converter is not None:
            codes = self._codec_converter.convert_original_to_new(audio_tokens=codes, audio_lens=lens)

        codes, lens = add_special_tokens(
            codes=codes,
            codes_len=lens,
            bos_id=self.context_audio_bos_id,
            eos_id=self.context_audio_eos_id,
        )

        return codes, lens

    def _pad_tensors_to_match(
        self, tensor_a: torch.Tensor, tensor_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad two 3D tensors along dim=1 so they have the same sequence length.

        Args:
            tensor_a: First tensor of shape (B, T_a, E).
            tensor_b: Second tensor of shape (B, T_b, E).

        Returns:
            Tuple of (tensor_a, tensor_b) both with shape (B, max(T_a, T_b), E).
        """
        len_a, len_b = tensor_a.size(1), tensor_b.size(1)
        if len_a < len_b:
            padding = torch.zeros(
                tensor_a.size(0),
                len_b - len_a,
                tensor_a.size(2),
                device=tensor_a.device,
                dtype=tensor_a.dtype,
            )
            tensor_a = torch.cat([tensor_a, padding], dim=1)
        elif len_a > len_b:
            padding = torch.zeros(
                tensor_b.size(0),
                len_a - len_b,
                tensor_b.size(2),
                device=tensor_b.device,
                dtype=tensor_b.dtype,
            )
            tensor_b = torch.cat([tensor_b, padding], dim=1)
        return tensor_a, tensor_b

    def _get_context_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
        context_audio_codes: torch.Tensor,
        context_audio_codes_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get context embeddings, handling text conditioning if enabled.

        Args:
            batch: Batch dictionary containing context tokens if text conditioning is used.
            context_audio_codes: Context audio codes. Shape: (B, C, T_ctx).
            context_audio_codes_lens: Length of context audio codes. Shape: (B,).

        Returns:
            Tuple of (context_embedded, context_lens) where:
                context_embedded: Combined context embedding. Shape: (B, T, E).
                context_lens: Length of context sequences. Shape: (B,).
        """
        context_audio_embedded, context_lens = self.embed_audio_tokens(
            audio_tokens=context_audio_codes, audio_tokens_lens=context_audio_codes_lens
        )  # (B, T/frame_stacking, E)

        if not self.use_text_conditioning_encoder:
            return context_audio_embedded, context_lens

        # Text conditioning path
        context_text_tokens = batch['context_text_tokens']
        context_text_lens = batch['context_text_tokens_lens']
        context_text_embedded = self.embed_context_text(context_text_tokens)  # (B, L, E)

        # Pad tensors to match sequence lengths
        context_audio_embedded, context_text_embedded = self._pad_tensors_to_match(
            context_audio_embedded, context_text_embedded
        )

        # For 3D tensor - need to broadcast the boolean mask
        has_text_context = batch['has_text_context'].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1), bool
        context_embedded = torch.where(has_text_context, context_text_embedded, context_audio_embedded)

        # For 1D tensor - direct use
        context_lens = torch.where(batch['has_text_context'], context_text_lens, context_audio_codes_lens)
        context_embedded = context_embedded[:, : context_lens.max(), :]

        return context_embedded, context_lens

    def _prepare_multi_encoder_context(
        self,
        context_input_embedded: torch.Tensor,
        context_mask: torch.Tensor,
        text_encoder_out: torch.Tensor,
        text_mask: torch.Tensor,
        attn_prior: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[Dict], Optional[List], int]:
        """Prepare context tensors for multi_encoder_context_tts model type.

        Args:
            context_input_embedded: Context embeddings. Shape: (B, T_ctx, E).
            context_mask: Mask for context. Shape: (B, T_ctx).
            text_encoder_out: Text encoder output. Shape: (B, T_text, E).
            text_mask: Mask for text. Shape: (B, T_text).
            attn_prior: Attention prior matrix.

        Returns:
            Tuple of (cond, cond_mask, multi_encoder_mapping, attn_prior_list, dec_context_size).
        """
        context_embeddings = self.context_encoder(context_input_embedded, context_mask, cond=None, cond_mask=None)[
            'output'
        ]
        cond = [text_encoder_out, context_embeddings]
        cond_mask = [text_mask, context_mask]
        multi_encoder_mapping = self.multi_encoder_mapping
        attn_prior_list = [attn_prior, None]
        return cond, cond_mask, multi_encoder_mapping, attn_prior_list, 0

    def _prepare_decoder_context(
        self,
        context_input_embedded: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        text_encoder_out: torch.Tensor,
        text_mask: torch.Tensor,
        attn_prior: Optional[torch.Tensor],
        speaker_indices: Optional[torch.Tensor],
        text: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        None,
        Optional[torch.Tensor],
        int,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Prepare context tensors for decoder_context_tts and decoder_ce model types.

        Args:
            batch: Full batch dictionary.
            context_input_embedded: Context embeddings. Shape: (B, T_ctx, E).
                Can be None if model_type is 'decoder_ce' with baked context embedding.
            context_mask: Mask for context. Shape: (B, T_ctx).
                Can be None if model_type is 'decoder_ce' with baked context embedding.
            text_encoder_out: Text encoder output. Shape: (B, T_text, E).
            text_mask: Mask for text. Shape: (B, T_text).
            attn_prior: Attention prior matrix.
            speaker_indices: Speaker indices for multi-speaker baked embeddings.
            text: Text tensor (used for batch_size).

        Returns:
            Tuple of (cond, cond_mask, multi_encoder_mapping, attn_prior,
                     dec_context_size, additional_decoder_input, additional_decoder_mask).
        """
        if self.model_type == 'decoder_context_tts':
            context_embeddings = context_input_embedded
        elif self.model_type == 'decoder_ce':
            if self.has_baked_context_embedding:
                # Baked context embedding replaces the context encoder
                batch_size = text.size(0)
                context_embeddings, context_input_lens = self.get_baked_context_embeddings_batch(
                    batch_size=batch_size, speaker_indices=speaker_indices
                )
                context_input_lens = context_input_lens.to(text.device)
                context_mask = get_mask_from_lengths(context_input_lens)
            else:
                # Zero-shot disable: with some probability, bypass the context encoder and feed
                # batch-shuffled raw embeddings so the model learns to not clone from untransformed input.
                # Skip when batch_size == 1: rolling a single sample maps it back to itself,
                # so the context would remain matched to the correct speaker.
                batch_size = context_input_embedded.size(0)
                if (
                    self.training
                    and batch_size > 1
                    and self.train_shuffle_context_embedding_prob > 0
                    and random.random() < self.train_shuffle_context_embedding_prob
                ):
                    shift = random.randint(1, batch_size - 1)
                    context_embeddings = context_input_embedded.roll(shift, dims=0)
                    context_mask = context_mask.roll(shift, dims=0)
                else:
                    context_embeddings = self.context_encoder(
                        context_input_embedded, context_mask, cond=None, cond_mask=None
                    )['output']
        else:
            raise ValueError(f"Unsupported model type for decoder context: {self.model_type}")

        dec_context_size = context_mask.size(1)

        # Pad attention prior if present
        if attn_prior is not None:
            padding_zeros = torch.zeros(
                attn_prior.size(0),
                dec_context_size,
                attn_prior.size(2),
                device=attn_prior.device,
            )
            attn_prior = torch.cat([padding_zeros, attn_prior], dim=1)

        return (
            text_encoder_out,  # cond
            text_mask,  # cond_mask
            None,  # multi_encoder_mapping
            attn_prior,
            dec_context_size,
            context_embeddings,  # additional_decoder_input
            context_mask,  # additional_decoder_mask
        )

    def _apply_ctc_prior_layers(
        self, attn_prior: Optional[Union[torch.Tensor, List]]
    ) -> Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]]:
        """Apply CTC prior layer filtering to attention prior.

        Args:
            attn_prior: Attention prior tensor or list of tensors.

        Returns:
            Filtered attention prior with None for layers not in ctc_prior_layer_ids.
        """
        if attn_prior is None or self.ctc_prior_layer_ids is None:
            return attn_prior

        if self.model_type == 'multi_encoder_context_tts':
            text_attn_prior = [
                attn_prior[0] if layer_idx in self.ctc_prior_layer_ids else None
                for layer_idx in range(self.decoder.n_layers)
            ]
            return [text_attn_prior, attn_prior[1]]
        else:
            return [
                attn_prior if layer_idx in self.ctc_prior_layer_ids else None
                for layer_idx in range(self.decoder.n_layers)
            ]

    def prepare_context_tensors(self, batch: Dict[str, torch.Tensor]) -> ContextTensorsOutput:
        """Prepare all context tensors for the decoder.

        This method orchestrates text encoding, context extraction, and model-type-specific
        processing to prepare tensors for decoder inference or training.

        Args:
            batch: Dictionary containing:
                - 'text': Text token IDs. Shape: (B, T_text).
                - 'text_lens': Text lengths. Shape: (B,).
                - 'context_audio_codes' or 'context_audio': Context audio.
                - 'align_prior_matrix' (optional): Beta-binomial attention prior.
                - 'speaker_indices' (optional): Speaker IDs for multi-speaker models.
                - Text conditioning fields if use_text_conditioning_encoder is True.

        Returns:
            ContextTensorsOutput dataclass containing all prepared tensors.

        Raises:
            ValueError: If model_type is not supported.
        """
        # Step 1: Encode text input (always needed)
        text, text_lens, text_mask, text_embedded, text_encoder_out = self._encode_text_input(batch)

        # Step 2: Get and scale attention prior
        _attn_prior = batch.get('align_prior_matrix', None)
        _attn_prior = self.scale_prior(_attn_prior, self.global_step)
        speaker_indices = batch.get('speaker_indices', None)

        # Step 3: Process context based on model type
        if self.model_type not in ['multi_encoder_context_tts', 'decoder_context_tts', 'decoder_ce']:
            raise ValueError(f"Unsupported model type {self.model_type}")

        # For decoder_ce with baked context embedding, skip context audio/text processing entirely
        # The baked embedding replaces the context encoder, so we don't need context inputs
        skip_context_processing = self.model_type == 'decoder_ce' and self.has_baked_context_embedding

        if skip_context_processing:
            # Use baked context embedding directly - no need for context audio/text
            context_audio_codes = None
            context_audio_codes_lens = None
            context_input_embedded = None
            context_mask = None
        else:
            # Extract context audio codes and compute embeddings
            context_audio_codes, context_audio_codes_lens = self._get_context_audio_codes(batch)
            context_input_embedded, context_input_lens = self._get_context_embeddings(
                batch, context_audio_codes, context_audio_codes_lens
            )
            context_mask = get_mask_from_lengths(context_input_lens)

        # Step 4: Dispatch to model-type-specific handler
        if self.model_type == 'multi_encoder_context_tts':
            cond, cond_mask, multi_encoder_mapping, attn_prior, dec_context_size = self._prepare_multi_encoder_context(
                context_input_embedded,
                context_mask,
                text_encoder_out,
                text_mask,
                _attn_prior,
            )
            additional_decoder_input = None
            additional_decoder_mask = None
        else:  # decoder_context_tts or decoder_ce
            (
                cond,
                cond_mask,
                multi_encoder_mapping,
                attn_prior,
                dec_context_size,
                additional_decoder_input,
                additional_decoder_mask,
            ) = self._prepare_decoder_context(
                context_input_embedded,
                context_mask,
                text_encoder_out,
                text_mask,
                _attn_prior,
                speaker_indices,
                text,
            )

        # Step 5: Apply CTC prior layer filtering
        attn_prior = self._apply_ctc_prior_layers(attn_prior)

        # Step 6: Return typed output
        return ContextTensorsOutput(
            text_encoder_out=text_encoder_out,
            text_embedded=text_embedded,
            text_mask=text_mask,
            text_lens=text_lens,
            text=text,
            cond=cond,
            cond_mask=cond_mask,
            attn_prior=attn_prior,
            prior_used=_attn_prior is not None,
            multi_encoder_mapping=multi_encoder_mapping,
            additional_decoder_input=additional_decoder_input,
            additional_decoder_mask=additional_decoder_mask,
            dec_context_size=dec_context_size,
            context_audio_codes=context_audio_codes,
            context_audio_codes_lens=context_audio_codes_lens,
            beta_binomial_attn_prior=batch.get('align_prior_matrix', None),
        )

    def replace_beta_binomial_prior_with_binarized(self, attn_prior, aligner_attn_hard):
        # aligner_attn_hard B, audio_timesteps, text_timesteps
        if self.model_type == 'multi_encoder_context_tts':
            text_attn_prior = attn_prior[0]
        else:
            text_attn_prior = attn_prior

        assert text_attn_prior is not None, "Prior is None"

        if isinstance(text_attn_prior, list):
            # Layer wise prior
            prior_updated = False
            for idx, prior in enumerate(text_attn_prior):
                if prior is not None:
                    text_attn_prior[idx][:, -aligner_attn_hard.size(1) :, :] = aligner_attn_hard
                    prior_updated = True
            assert prior_updated, "Did not find any prior to update"
        else:
            # Same prior for all layers
            text_attn_prior[:, -aligner_attn_hard.size(1) :, :] = aligner_attn_hard

        if self.model_type == 'multi_encoder_context_tts':
            attn_prior[0] = text_attn_prior
        else:
            attn_prior = text_attn_prior

        return attn_prior

    def get_binarized_prior_matrix(self, aligner_attn_soft, audio_lens, text_lens):
        # aligner_attn_soft B, 1, audio_timesteps, text_timesteps
        if self.binarize_attn_method == 'nemo_binarize':
            logging.debug("Binarizing attention using nemo_binarize")
            binarize_repeat_audio_factor = self.binarize_repeat_audio_factor
            aligner_attn_soft_repeated = aligner_attn_soft.repeat_interleave(
                binarize_repeat_audio_factor, dim=2
            )  # B, 1, 2*audio_timesteps, text_timesteps
            aligner_attn_hard = binarize_attention_parallel(
                aligner_attn_soft_repeated,
                text_lens,
                audio_lens * binarize_repeat_audio_factor,
            ).squeeze(
                1
            )  # B, 2*audio_timesteps, text_timesteps
            aligner_attn_hard = aligner_attn_hard[:, ::2, :]  # B, audio_timesteps, text_timesteps
        elif self.binarize_attn_method == 'argmax':
            logging.debug("Binarizing attention using argmax")
            aligner_attn_hard = torch.argmax(aligner_attn_soft.squeeze(1), dim=-1)
            aligner_attn_hard = torch.nn.functional.one_hot(
                aligner_attn_hard, num_classes=aligner_attn_soft.size(-1)
            ).float()
        else:
            raise ValueError(
                f"self.binarize_attn_method '{self.binarize_attn_method}' must be one of 'nemo_binarize' or 'argmax'."
            )

        aligner_attn_hard_wider = aligner_attn_hard + self.binarized_prior_epsilon

        for future_timestep in range(self.prior_future_context):
            decay_factor = self.prior_future_decay ** (future_timestep + 1)
            aligner_attn_hard_wider[:, :, future_timestep + 1 :] += (
                decay_factor * aligner_attn_hard[:, :, : -(future_timestep + 1)]
            )

        for past_timestep in range(self.prior_past_context):
            decay_factor = self.prior_past_decay ** (past_timestep + 1)
            aligner_attn_hard_wider[:, :, : -past_timestep - 1] += (
                decay_factor * aligner_attn_hard[:, :, past_timestep + 1 :]
            )

        aligner_attn_hard_wider = torch.clamp(aligner_attn_hard_wider, 0.0, 1.0)
        return aligner_attn_hard_wider

    def prepare_dummy_cond_for_cfg(self, cond, cond_mask, additional_decoder_input, additional_dec_mask):
        dummy_additional_decoder_input = None
        dummy_additional_dec_mask = None
        if additional_decoder_input is not None:
            dummy_additional_decoder_input = torch.zeros_like(additional_decoder_input)
            # all ones mask means dont ignore any timesteps (so that it is consistent with usual decoder mask)
            dummy_additional_dec_mask = torch.ones_like(additional_dec_mask)

        if isinstance(cond, list):
            # multi encoder conditioning
            dummy_cond = [torch.zeros_like(cond_item) for cond_item in cond]
            attn_prior = [None for _ in cond]
            dummy_mask = []
            for mask_item in cond_mask:
                # ignore all timesteps except the first one
                mask = torch.zeros_like(mask_item)
                mask[:, 0] = 1  # Make first timestep all zeros
                dummy_mask.append(mask)

        elif isinstance(cond, torch.Tensor):
            # single encoder conditioning
            dummy_cond = torch.zeros_like(cond)
            dummy_mask = torch.zeros_like(cond_mask)
            dummy_mask[:, 0] = 1  # ignore all timesteps except the first one
            attn_prior = None
        else:
            raise ValueError(f"Unsupported type for cond {type(cond)}")

        return (
            dummy_cond,
            dummy_mask,
            dummy_additional_decoder_input,
            dummy_additional_dec_mask,
            attn_prior,
        )

    def process_batch(self, batch):
        context_tensors = self.prepare_context_tensors(batch)
        disable_alignment_loss = False

        if 'audio_codes' not in batch:
            audio_codes, audio_codes_lens = self._codec_helper.audio_to_codes(
                batch['audio'],
                batch['audio_lens'],
                sample_rate=batch.get('sample_rate'),
            )
        else:
            audio_codes = batch['audio_codes']
            audio_codes_lens = batch['audio_codes_lens']

        if self._codec_converter:
            audio_codes = self._codec_converter.convert_original_to_new(
                audio_tokens=audio_codes, audio_lens=audio_codes_lens
            )

        audio_codes, audio_codes_lens = add_special_tokens(
            codes=audio_codes,
            codes_len=audio_codes_lens,
            bos_id=self.audio_bos_id,
            eos_id=self.audio_eos_id,
            num_bos_tokens=self.frame_stacking_factor,
        )  # (B, C, T)

        audio_codes_embedded_all, audio_codes_lens_all = self.embed_audio_tokens(
            audio_tokens=audio_codes, audio_tokens_lens=audio_codes_lens
        )  # (B, T/frame_stacking_factor, E)
        # Note: if a tensor lacks the `_unstacked` suffix, it can be assumed to be in the frame-stacked domain

        # Remove EOS token for decoder inputs
        audio_codes_embedded_input, audio_codes_lens_input = remove_embedded_eos_token(
            embedded=audio_codes_embedded_all, embedded_len=audio_codes_lens_all
        )
        use_cfg = self.training and (self.cfg_unconditional_prob > 0.0) and (context_tensors.cond is not None)
        if use_cfg and torch.rand(1).item() < self.cfg_unconditional_prob:
            (
                cond,
                cond_mask,
                additional_decoder_input,
                additional_decoder_mask,
                attn_prior,
            ) = self.prepare_dummy_cond_for_cfg(
                context_tensors.cond,
                context_tensors.cond_mask,
                context_tensors.additional_decoder_input,
                context_tensors.additional_decoder_mask,
            )
            disable_alignment_loss = True
        else:
            cond = context_tensors.cond
            cond_mask = context_tensors.cond_mask
            additional_decoder_input = context_tensors.additional_decoder_input
            additional_decoder_mask = context_tensors.additional_decoder_mask
            attn_prior = context_tensors.attn_prior

            if self.training and self.decoder_input_dropout_prob > 0.0 and torch.rand(1).item() < 0.5:
                # For some batches (half of them), replace decoder_input_dropout_prob of the timesteps with random tokens
                max_codebook_val = self.dec_random_input_max
                # @pneekhara: Keeping dec_random_input_max configurable since num_all_tokens_per_codebook usually has padding tokens
                # which can cause errors when doing codes_to_audio for audio_codes_input. We are not currently calling codes_to_audio on
                # audio_codes_input so should not matter if we don't supply dec_random_input_max.
                random_audio_tokens = torch.randint(
                    low=0,
                    high=max_codebook_val,
                    size=audio_codes.size(),
                    device=audio_codes_embedded_input.device,
                )  # (B, C, T)
                random_embedded, random_embedded_lens = self.embed_audio_tokens(
                    audio_tokens=random_audio_tokens, audio_tokens_lens=audio_codes_lens
                )  # (B T E)
                random_embedded, random_embedded_lens = remove_embedded_eos_token(
                    embedded=random_embedded, embedded_len=random_embedded_lens
                )
                dec_dropout_mask = (
                    torch.rand(
                        (1, 1, audio_codes_embedded_input.size(2)),
                        device=audio_codes_embedded_input.device,
                    )
                    > self.decoder_input_dropout_prob
                )  # (1, 1, T)
                audio_codes_embedded_input = torch.where(
                    dec_dropout_mask,
                    audio_codes_embedded_input,
                    random_embedded,
                )

        audio_codes_mask = get_mask_from_lengths(audio_codes_lens_input)
        if additional_decoder_input is not None:
            audio_codes_embedded_input = torch.cat([additional_decoder_input, audio_codes_embedded_input], dim=1)
            audio_codes_mask = torch.cat([additional_decoder_mask, audio_codes_mask], dim=1)

        # Remove BOS token for aligner targets
        audio_codes_embedded_target, audio_codes_lens_target = remove_embedded_bos_token(
            embedded=audio_codes_embedded_all, embedded_len=audio_codes_lens_all
        )
        aligner_encoder_loss = None
        aligner_attn_soft = None
        aligner_attn_hard = None
        if self.use_alignment_encoder and not disable_alignment_loss:
            aligner_prior = None
            if self.use_prior_for_aligner:
                aligner_prior = context_tensors.beta_binomial_attn_prior

            train_aligner = self.global_step < self.aligner_encoder_train_steps

            with torch.set_grad_enabled(train_aligner):
                # Passing target audio embeddings to the alignment encoder
                aligner_queries = audio_codes_embedded_target.permute(0, 2, 1)  # (B, E, T')
                aligner_keys = context_tensors.text_encoder_out.permute(0, 2, 1)  # (B, E, T)
                # Aligner uses inverted mask
                aligner_mask = ~context_tensors.text_mask.unsqueeze(-1)  # (B, T, 1)
                aligner_attn_soft, aligner_attn_logprobs = self.alignment_encoder(
                    queries=aligner_queries,
                    keys=aligner_keys,
                    mask=aligner_mask,
                    attn_prior=aligner_prior,
                )

            if train_aligner:
                aligner_encoder_loss = self.alignment_encoder_loss(
                    attn_logprob=aligner_attn_logprobs,
                    in_lens=context_tensors.text_lens,
                    out_lens=audio_codes_lens_target,
                )

            with torch.no_grad():
                aligner_attn_hard = self.get_binarized_prior_matrix(
                    aligner_attn_soft, audio_codes_lens_input, context_tensors.text_lens
                )
                if (self.global_step > self.binarize_prior_after_step) and context_tensors.prior_used:
                    attn_prior = self.replace_beta_binomial_prior_with_binarized(attn_prior, aligner_attn_hard)

        logits, attn_info, dec_out, moe_routing_info = self.forward(
            dec_input_embedded=audio_codes_embedded_input,
            dec_input_mask=audio_codes_mask,
            cond=cond,
            cond_mask=cond_mask,
            attn_prior=attn_prior,
            multi_encoder_mapping=context_tensors.multi_encoder_mapping,
        )
        # logits: (B, T', num_codebooks * num_tokens_per_codebook)
        # dec_out: (B, T', E)
        # moe_routing_info: List of routing info dicts from each layer (if MoE enabled)
        dec_context_size = context_tensors.dec_context_size
        logits = logits[:, dec_context_size:, :]  # Remove the context audio embeddings from the logits

        # Remove BOS tokens from decoder targets
        audio_codes_target_unstacked, audio_codes_lens_target_unstacked = remove_bos_token(
            codes=audio_codes,
            codes_len=audio_codes_lens,
            num_tokens=self.frame_stacking_factor,
        )
        # Codebook loss (parallel)
        codebook_loss, loss_mask = self.compute_loss(
            logits,
            audio_codes_target_unstacked,
            audio_codes_lens_target_unstacked,
            frame_stacking_factor=self.frame_stacking_factor,
        )
        # Alignment loss
        alignment_loss = None
        if self.alignment_loss_scale > 0.0 and not disable_alignment_loss:
            text_lens = context_tensors.text_lens
            cross_attention_scores = [
                attn['cross_attn_probabilities'][1]
                for layer_idx, attn in enumerate(attn_info)
                if layer_idx in self.ctc_prior_layer_ids
            ]
            alignment_loss = self.compute_alignment_loss(
                cross_attention_scores,
                text_lens,
                audio_codes_lens_target,
                dec_context_size,
            )
            loss = self.codebook_loss_scale * codebook_loss + alignment_loss
        else:
            loss = self.codebook_loss_scale * codebook_loss

        # Local Transformer loss
        local_transformer_loss = None
        local_transformer_logits = None
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            if self.local_transformer_type == LocalTransformerType.MASKGIT:
                # Maskgit
                # randomly replace some positions with MASK_TOKEN
                audio_codes_masked, mask_tokens_mask = self._lt_helper.apply_random_mask(audio_codes_target_unstacked)
                # TODO @rfejgin: the very last position might be padding but the local transformer might look at it as part of
                #                of a pair where the first position is valid. Is this an issue?
                local_transformer_logits = self._lt_helper.compute_logits(
                    dec_out[:, dec_context_size:, :],
                    audio_codes_masked,
                    targets_offset_by_one=True,
                )
                local_transformer_loss, _ = self.compute_loss(
                    local_transformer_logits,
                    audio_codes_target_unstacked,
                    audio_codes_lens_target_unstacked,
                    mask_tokens_mask,
                    frame_stacking_factor=self.frame_stacking_factor,
                )
            else:
                # Autoregressive
                assert self.local_transformer_type == LocalTransformerType.AR, "Unexpected local transformer type"
                local_transformer_logits = self._lt_helper.compute_logits(
                    dec_out[:, dec_context_size:, :],
                    audio_codes_target_unstacked,
                    targets_offset_by_one=False,
                )
                local_transformer_loss, _ = self.compute_loss(
                    local_transformer_logits,
                    audio_codes_target_unstacked,
                    audio_codes_lens_target_unstacked,
                    None,
                    frame_stacking_factor=self.frame_stacking_factor,
                )
            loss = loss + self.local_transformer_loss_scale * local_transformer_loss

        if aligner_encoder_loss is not None:
            loss = loss + aligner_encoder_loss

        # Compute MoE auxiliary losses and expert usage statistics if MoE is enabled
        moe_load_balancing_loss = None
        moe_router_z_loss = None
        moe_expert_usage_stats = None

        if self.use_moe and moe_routing_info is not None:
            # The decoder input is: [context_audio | target_audio | padding]. MoE routing runs on this full concatenated
            # sequence, so router_logits, router_probs, and expert_indices contain context audio dimensions. We include
            # context audio in the MoE loss computation (not stripped like the main CE loss) because:
            #   1. Load balancing loss needs to see all tokens the router dispatches, including context. Excluding
            #      context would make experts that specialize in processing context audio look underused, producing
            #      misleading gradients.
            #   2. At inference, context audio is always present and routed through experts. Training the router to
            #      balance load only on target tokens would create a train/inference mismatch in routing behavior.
            # Padding is excluded via x_mask. Router already masks padded positions, router_logits/router_probs=0,
            # expert_indices=-1, and we pass x_mask to loss functions to ensure averages are computed only over valid (non-padding) tokens.
            all_router_logits = []
            all_router_probs = []
            all_expert_indices = []
            for layer_routing_info in moe_routing_info:
                all_router_logits.append(layer_routing_info['router_logits'])
                all_router_probs.append(layer_routing_info['router_probs'])
                all_expert_indices.append(layer_routing_info['expert_indices'])

            # Concatenate across layers (batch dimension)
            stacked_logits = torch.stack(all_router_logits, dim=0)  # (n_layers, B, T, num_experts)
            stacked_probs = torch.stack(all_router_probs, dim=0)  # (n_layers, B, T, num_experts)
            stacked_indices = torch.stack(all_expert_indices, dim=0)  # (n_layers, B, T, top_k)

            # Reshape for loss computation
            # merged_logits and merged_probs are (n_layers*B, T, num_experts)
            merged_logits = stacked_logits.view(-1, stacked_logits.size(2), stacked_logits.size(3))
            merged_probs = stacked_probs.view(-1, stacked_probs.size(2), stacked_probs.size(3))
            # merged_indices is (n_layers*B, T, top_k)
            merged_indices = stacked_indices.view(-1, stacked_indices.size(2), stacked_indices.size(3))

            # Repeat mask for each layer: (B, T) -> (n_layers*B, T)
            # Include ALL decoder input positions (context audio + target audio) in loss computation
            # Context audio routing is important for inference quality. We want Expert specialization where some experts
            # may specialize in processing context, or some may specialize in generating target, or both.
            merged_mask = (
                audio_codes_mask.unsqueeze(0).repeat(len(moe_routing_info), 1, 1).view(-1, audio_codes_mask.size(1))
            )

            # Compute MoE losses using the loss module (both train and val)
            # Pass mask to ensure losses are computed only over valid tokens (excluding padding)
            moe_load_balancing_loss, moe_router_z_loss, moe_total_loss = self.moe_auxiliary_loss(
                router_logits=merged_logits,
                router_probs=merged_probs,
                x_mask=merged_mask,
            )

            # Compute expert usage statistics
            with torch.no_grad():
                num_experts = stacked_probs.size(-1)
                n_moe_layers = stacked_probs.size(0)

                # Per-layer expert usage: (n_layers, num_experts)
                layer_expert_usage = torch.stack(
                    [compute_expert_usage(stacked_probs[i], audio_codes_mask) for i in range(n_moe_layers)]
                )

                # Global expert usage: mean across layers (for scalar logging)
                expert_usage = layer_expert_usage.mean(dim=0)  # (num_experts,)

                # Compute how often each expert is selected in top-k
                # For padded positions, expert_indices=-1, so they don't match any valid expert (0 to num_experts-1)
                expert_selection_counts = torch.zeros(num_experts, device=merged_probs.device)
                for expert_idx in range(num_experts):
                    expert_selection_counts[expert_idx] = (merged_indices == expert_idx).float().sum()

                # Normalize to get selection frequency over valid (non-padded) selections only
                # Padded positions have expert_indices=-1, which don't match any valid expert
                valid_selections = (merged_indices != -1).sum().float().clamp_min(1.0)
                expert_selection_freq = expert_selection_counts / valid_selections

                moe_expert_usage_stats = {
                    'expert_usage': expert_usage.detach(),  # (num_experts,)
                    'layer_expert_usage': layer_expert_usage.detach(),  # (n_layers, num_experts)
                    'expert_selection_freq': expert_selection_freq.detach(),  # (num_experts,)
                    'batch_expert_usage_variance': expert_usage.var().detach(),
                    'ideal_usage': 1.0 / num_experts,
                }

            # Add MoE loss to total loss (only in training mode)
            if self.training:
                loss = loss + moe_total_loss

        return {
            'logits': logits,
            'attn_info': attn_info,
            'loss': loss,
            'codebook_loss': codebook_loss,
            'local_transformer_loss': local_transformer_loss,
            'local_transformer_logits': local_transformer_logits,
            'loss_mask': loss_mask,
            'alignment_loss': alignment_loss,
            'aligner_encoder_loss': aligner_encoder_loss,
            'moe_load_balancing_loss': moe_load_balancing_loss,
            'moe_router_z_loss': moe_router_z_loss,
            'moe_expert_usage_stats': moe_expert_usage_stats,
            'audio_codes_target': audio_codes_target_unstacked,
            'audio_codes_lens_target': audio_codes_lens_target_unstacked,
            'text': context_tensors.text,
            'text_lens': context_tensors.text_lens,
            'context_audio_codes': context_tensors.context_audio_codes,
            'context_audio_codes_lens': context_tensors.context_audio_codes_lens,
            'dec_context_size': dec_context_size,
            'aligner_attn_soft': aligner_attn_soft,
            'aligner_attn_hard': aligner_attn_hard,
        }

    def training_step(self, batch, batch_idx):
        batch_output = self.process_batch(batch)
        loss = batch_output['loss']
        codebook_loss = batch_output['codebook_loss']
        self.log('Loss:train/codebook_loss', codebook_loss, prog_bar=True, sync_dist=True)
        if self.cfg_unconditional_prob == 0.0:
            # Only log alignment loss when not using cfg to avoid sync issues when
            # alignment loss is None on some ranks
            alignment_loss = batch_output['alignment_loss']
            if alignment_loss is not None:
                self.log('Loss:train/alignment_loss', alignment_loss, prog_bar=True, sync_dist=True)
        self.log('Loss:train/loss', loss, prog_bar=True, sync_dist=True)
        local_transformer_loss = batch_output['local_transformer_loss']
        if local_transformer_loss is not None:
            self.log('Loss:train/local_transformer_loss', local_transformer_loss, prog_bar=True, sync_dist=True)

        # Log MoE losses and expert usage if MoE is enabled
        moe_load_balancing_loss = batch_output.get('moe_load_balancing_loss', None)
        moe_router_z_loss = batch_output.get('moe_router_z_loss', None)
        moe_expert_usage_stats = batch_output.get('moe_expert_usage_stats', None)
        if moe_load_balancing_loss is not None and self.moe_auxiliary_loss.load_balancing_loss.loss_scale > 0:
            self.log('Loss:train/moe_load_balancing_loss', moe_load_balancing_loss, prog_bar=True, sync_dist=True)
        if moe_router_z_loss is not None and self.moe_auxiliary_loss.router_z_loss.loss_scale > 0:
            self.log('Loss:train/moe_router_z_loss', moe_router_z_loss, prog_bar=True, sync_dist=True)
        if moe_expert_usage_stats is not None:
            expert_usage = moe_expert_usage_stats['expert_usage']
            layer_expert_usage = moe_expert_usage_stats['layer_expert_usage']

            self.log(
                'Loss:train/moe_expert_usage_variance',
                moe_expert_usage_stats['batch_expert_usage_variance'],
                sync_dist=True,
            )

            # Per-expert usage scalars
            for eidx in range(len(expert_usage)):
                self.log(f'MoE:train/Expert_{eidx:02d}_usage', expert_usage[eidx], sync_dist=True)

            # Accumulate layer-wise usage for training heatmap
            if self._moe_train_layer_usage_accum is None:
                self._moe_train_layer_usage_accum = torch.zeros_like(layer_expert_usage)
            self._moe_train_layer_usage_accum += layer_expert_usage.detach()
            self._moe_train_accum_steps += 1

        # Log batch info
        batch_size, text_token_max_len = batch["text"].shape
        text_token_total_num = batch["text_lens"].sum()
        batch_info_dict = {
            "BatchInfo:train/batch_size": batch_size,
            "BatchInfo:train/text_token_max_len": text_token_max_len,
            "BatchInfo:train/text_token_total_num_in_batch": text_token_total_num.item(),
            "BatchInfo:train/text_token_pad_ratio_percent_in_batch": 100
            * (1 - text_token_total_num / (batch_size * text_token_max_len)),
        }

        if "audio_codes" in batch:
            audio_codes_max_len = batch["audio_codes"].shape[-1]
            audio_codes_total_num = batch["audio_codes_lens"].sum()
            batch_info_dict.update(
                {
                    "BatchInfo:train/audio_codes_max_len": audio_codes_max_len,
                    "BatchInfo:train/audio_codes_total_num_in_batch": audio_codes_total_num.item(),
                    "BatchInfo:train/audio_codes_pad_ratio_percent_in_batch": 100
                    * (1 - audio_codes_total_num / (batch_size * audio_codes_max_len)),
                }
            )
        else:
            audio_samples_max_len = batch["audio"].shape[-1]
            audio_samples_total_num = batch["audio_lens"].sum()
            batch_info_dict.update(
                {
                    "BatchInfo:train/audio_samples_max_len": audio_samples_max_len,
                    "BatchInfo:train/audio_samples_total_num_in_batch": audio_samples_total_num.item(),
                    "BatchInfo:train/audio_samples_pad_ratio_percent_in_batch": 100
                    * (1 - audio_samples_total_num / (batch_size * audio_samples_max_len)),
                }
            )

        self.log_dict(batch_info_dict, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Validation step with support for multiple dataloaders.

        Args:
            batch: Input batch
            batch_idx: Batch index
            dataloader_idx: Index of the dataloader (0 for single dataloader)
        """
        batch_output = self.process_batch(batch)
        # self.process_batch returns a dict. We currently only log "logits" which come from the parallel prediction
        # head. If we use local_transformer, then the local_transformer returns "local_transformer_logits"

        loss = batch_output['loss']
        codebook_loss = batch_output['codebook_loss']
        alignment_loss = batch_output['alignment_loss']
        aligner_encoder_loss = batch_output['aligner_encoder_loss']
        local_transformer_loss = batch_output['local_transformer_loss']

        # Extract MoE losses and expert usage statistics if MoE is enabled
        moe_load_balancing_loss = batch_output.get('moe_load_balancing_loss', None)
        moe_router_z_loss = batch_output.get('moe_router_z_loss', None)
        moe_expert_usage_stats = batch_output.get('moe_expert_usage_stats', None)

        logits = batch_output['logits']
        audio_codes_target = batch_output['audio_codes_target']
        audio_codes_lens_target = batch_output['audio_codes_lens_target']
        context_audio_codes = batch_output['context_audio_codes']
        context_audio_codes_lens = batch_output['context_audio_codes_lens']
        attn_info = batch_output['attn_info']
        text_lens = batch_output['text_lens']
        dec_context_size = batch_output['dec_context_size']

        val_output = {
            'val_loss': loss,
            'val_codebook_loss': codebook_loss,
        }

        # Only add optional losses if they were computed (not None)
        if alignment_loss is not None:
            val_output['val_alignment_loss'] = alignment_loss
        if local_transformer_loss is not None:
            val_output['val_local_transformer_loss'] = local_transformer_loss
        if aligner_encoder_loss is not None:
            val_output['val_aligner_encoder_loss'] = aligner_encoder_loss
        if moe_load_balancing_loss is not None:
            val_output['val_moe_load_balancing_loss'] = moe_load_balancing_loss
        if moe_router_z_loss is not None:
            val_output['val_moe_router_z_loss'] = moe_router_z_loss
        if moe_expert_usage_stats is not None:
            val_output['val_moe_expert_usage_stats'] = moe_expert_usage_stats

        # Prepare media data for logging (only first batch of each dataloader, rank 0 only).
        if batch_idx == 0 and self.global_rank == 0:
            dataset_prefix = self.get_validation_dataloader_prefix(dataloader_idx)

            # Prepare audio examples (decode via vocoder, convert to numpy)
            audio_data = self._prepare_audio_examples(
                logits=logits,
                target_audio_codes=audio_codes_target,
                audio_codes_lens=audio_codes_lens_target,
                context_audio_codes=context_audio_codes,
                context_audio_codes_lens=context_audio_codes_lens,
                max_examples=3,
            )

            # Prepare attention images (only when cross-attention is available)
            attention_data = {}
            has_cross_attn = (
                self.model_type != 'decoder_pretrain_synthesizer'
                and len(attn_info[self.transcript_decoder_layers[0]].get('cross_attn_probabilities', [])) > 1
            )

            if has_cross_attn:
                # Overall attention: average across CTC prior layers
                cross_attention_probs = [
                    attn['cross_attn_probabilities'][0]
                    for layer_idx, attn in enumerate(attn_info)
                    if layer_idx in self.ctc_prior_layer_ids
                ]
                attention_data['overall'] = self._prepare_attention_images(
                    cross_attention_probs,
                    audio_codes_lens_target,
                    text_lens,
                    dec_context_size=dec_context_size,
                    max_examples=3,
                )

                # Per-layer attention visualization
                for layer_idx in self.transcript_decoder_layers:
                    layer_cross_attention_probs = [attn_info[layer_idx]['cross_attn_probabilities'][0]]
                    attention_data[f'layer_{layer_idx:02d}'] = self._prepare_attention_images(
                        layer_cross_attention_probs,
                        audio_codes_lens_target,
                        text_lens,
                        dec_context_size=dec_context_size,
                        max_examples=3,
                    )

                # Aligner encoder attention (if available)
                if batch_output['aligner_attn_soft'] is not None:
                    attention_data['aligner_encoder_attn'] = self._prepare_attention_images(
                        [batch_output['aligner_attn_soft']],
                        audio_codes_lens_target,
                        text_lens,
                        dec_context_size=0,
                        max_examples=3,
                    )

                if batch_output['aligner_attn_hard'] is not None:
                    attention_data['aligner_encoder_attn_hard'] = self._prepare_attention_images(
                        [batch_output['aligner_attn_hard'].unsqueeze(1)],
                        audio_codes_lens_target,
                        text_lens,
                        dec_context_size=0,
                        max_examples=3,
                    )

            val_output['media_data'] = {
                'dataset_prefix': dataset_prefix,
                'pred_audios': audio_data['pred_audios'],
                'target_audios': audio_data['target_audios'],
                'context_audios': audio_data['context_audios'],
                'attention_data': attention_data,
            }

        self.validation_step_outputs[dataloader_idx].append(val_output)

        return val_output

    def get_most_attended_text_timestep(
        self,
        alignment_attention_scores,
        last_attended_timesteps,
        text_lens,
        lookahead_window_size,
        attended_timestep_counter,
        batch_size,
        left_offset=None,
    ):
        """
        Returns the most attended timestep for each batch item

        This method identifies which text token is most attended to within a lookahead window, starting from
        the last attended timestep. It includes logic to detect attention sinks (tokens attended to excessively)
        and move past them. The method also tracks how many times each timestep has been attended.

        Args:
            alignment_attention_scores (torch.Tensor): Attention scores between audio and text tokens.
                Shape: (batch_size, text_length).
            last_attended_timesteps (list): List containing the last attended timestep for each batch item.
                The last element [-1] should be a list/tensor of length batch_size.
            text_lens (torch.Tensor): Length of text sequence for each batch item. Shape: (batch_size,).
            lookahead_window_size (int): Size of the forward-looking window to search for the next attended
                timestep. Determines how far ahead from the last attended timestep to look.
            attended_timestep_counter (Optional[list]): List of dictionaries (one per batch item) tracking how many
                times each timestep has been attended. Used to detect attention sinks.
            batch_size (int): Number of items in the batch.
            left_offset (list, optional): List of offsets to adjust timestep indices for each batch item,
                used in chunked inference when text is provided in chunks. Relevant only in multi-chunk
                generation.

        Returns:
            tuple: A tuple containing:
                - text_time_step_attended (list): List of integers, one per batch item, indicating the most
                  attended text timestep for that item.
                - attended_timestep_counter (list): Updated counter tracking attendance frequency for each
                  timestep across all batch items.
        """
        if left_offset is None:
            left_offset = [0] * batch_size
        text_time_step_attended = []
        for bidx in range(batch_size):
            last_attended_timestep = last_attended_timesteps[-1][bidx]
            if (
                attended_timestep_counter[bidx].get(last_attended_timestep, 0)
                >= self.inference_parameters.attention_sink_threshold
            ):
                # This is probably an attention sink! Move to the next timestep
                last_attended_timestep += 1
            last_attended_timestep = max(last_attended_timestep, left_offset[bidx])
            last_attended_timestep_in_this_window = last_attended_timestep - left_offset[bidx]
            window_size = lookahead_window_size
            window_end = min(
                last_attended_timestep_in_this_window + window_size, text_lens[bidx] - 3
            )  # Ignore the last 3 timesteps
            item_attention_scores = alignment_attention_scores[bidx, last_attended_timestep_in_this_window:window_end]
            if item_attention_scores.size(0) == 0:
                # This means the sentence has ended
                attended_timestep = text_lens[bidx].item() - 1 + left_offset[bidx]
            else:
                attended_timestep = item_attention_scores.argmax().item() + last_attended_timestep
            text_time_step_attended.append(attended_timestep)
            attended_timestep_counter[bidx][attended_timestep] = (
                attended_timestep_counter[bidx].get(attended_timestep, 0) + 1
            )
        return text_time_step_attended, attended_timestep_counter

    def construct_inference_prior(
        self,
        prior_epsilon,
        cross_attention_scores,
        text_lens,
        text_time_step_attended,
        attended_timestep_counter,
        unfinished_texts,
        finished_texts_counter,
        end_indices,
        lookahead_window_size,
        batch_size,
    ):
        # Attn prior for the next timestep
        _attn_prior = cross_attention_scores.new_full(
            (cross_attention_scores.shape[0], 1, cross_attention_scores.shape[1]),
            prior_epsilon,
        )
        for bidx in range(cross_attention_scores.shape[0]):
            if bidx < batch_size:
                _text_len = text_lens[bidx]
                if text_lens[bidx] <= 5:
                    # Very short sentences, No Prior
                    _attn_prior[bidx, 0, :] = 1.0
                else:
                    _attn_prior[bidx, 0, max(1, text_time_step_attended[bidx] - 1)] = (
                        1.0  # Slight exposure to history for better pronounciation. Not very important.
                    )
                    _attn_prior[bidx, 0, text_time_step_attended[bidx]] = (
                        1.0  # Slightly bias to continue moving forward. Not very important.
                    )
                    for ind in range(1, lookahead_window_size + 1):
                        _attn_prior[
                            bidx,
                            0,
                            min(text_time_step_attended[bidx] + ind, _text_len - 1),
                        ] = 1.0

                # Penalize positions that have become attention sinks.
                for _timestep in attended_timestep_counter[bidx]:
                    if (
                        attended_timestep_counter[bidx][_timestep]
                        >= self.inference_parameters.attention_sink_threshold
                    ):
                        _attn_prior[bidx, 0, : _timestep + 1] = prior_epsilon

                unfinished_texts[bidx] = False
                if text_time_step_attended[bidx] < text_lens[bidx] - 3:
                    # This means the sentence has not ended
                    if bidx not in end_indices:
                        unfinished_texts[bidx] = True

                if text_time_step_attended[bidx] >= text_lens[bidx] - 2 or bidx in end_indices:
                    if bidx not in finished_texts_counter:
                        finished_texts_counter[bidx] = 0

        for bidx in finished_texts_counter:
            finished_texts_counter[bidx] += 1
            if finished_texts_counter[bidx] > 5:
                # This means we have been within the text EOS window for at least 5 timesteps
                # We should allow EOS to be predicted now.
                unfinished_texts[bidx] = False

        return _attn_prior, unfinished_texts, finished_texts_counter

    def find_eos_frame_index(self, codes, eos_detection_method) -> Union[int, float]:
        """
        Checks for EOS in the predicted codes. Returns the index of the first frame within the frame stack
        that contains an EOS token across any codebook, or `None` if no EOS is found.
        Args:
            codes: (num_codebooks, frame_stacking_factor)
        Returns:
            index (within the frame stack) of the first frame with EOS, or `float('inf')` if no EOS is found
        """
        eos_mask = codes == self.audio_eos_id  # (codebooks, frame_stacking_factor)
        detection_type = EOSDetectionMethod.detection_type(eos_detection_method)
        if detection_type == "any":
            eos_per_frame = eos_mask.any(
                dim=0
            )  # (frame_stacking_factor,) - True if any codebook has EOS in this frame
        elif detection_type == "all":
            eos_per_frame = eos_mask.all(
                dim=0
            )  # (frame_stacking_factor,) - True if all codebooks have EOS in this frame
        elif detection_type == "zero_cb":
            eos_per_frame = eos_mask[:1, :].any(
                dim=0
            )  # (frame_stacking_factor,) - True if zeroth codebook has EOS in this frame
        else:
            raise ValueError(f"Invalid EOS detection method: {eos_detection_method}")
        # find first frame with EOS
        if eos_per_frame.any():
            # return index of the first frame with EOS
            return eos_per_frame.nonzero()[0].item()
        return float('inf')

    def detect_eos(self, audio_codes_multinomial, audio_codes_argmax, eos_detection_method) -> Union[int, float]:
        """
        Detects EOS in the predicted codes. Returns the index of the first frame within the frame stack
        that triggers EOS detection, or `float('inf')` if no EOS is found.
        Args:
            audio_codes_multinomial: (num_codebooks, frame_stacking_factor) - Multinomial samples
            audio_codes_argmax: (num_codebooks, frame_stacking_factor) - Argmax samples
            eos_detection_method: EOS detection method
        Returns:
            index (within the frame stack) of the first frame with EOS, or `float('inf')` if no EOS is found
        """
        sampling_type = EOSDetectionMethod.sampling_type(eos_detection_method)
        if sampling_type == "argmax":
            return self.find_eos_frame_index(audio_codes_argmax, eos_detection_method)
        elif sampling_type == "argmax_or_multinomial":
            argmax_eos_frame = self.find_eos_frame_index(audio_codes_argmax, eos_detection_method)
            multinomial_eos_frame = self.find_eos_frame_index(audio_codes_multinomial, eos_detection_method)
            return min(argmax_eos_frame, multinomial_eos_frame)
        else:
            raise ValueError(f"Invalid EOS detection method: {eos_detection_method}")

    def _create_eos_detection_scratch(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> EOSDetectionScratch:
        """Allocate the fixed EOS workspace once for one generation session."""
        codebook_shape = (
            batch_size,
            self.frame_stacking_factor,
            self.num_audio_codebooks,
        )
        frame_shape = (batch_size, self.frame_stacking_factor)
        frame_positions = (
            torch.arange(self.frame_stacking_factor, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
        )
        return EOSDetectionScratch(
            base_max=torch.empty(codebook_shape, device=device, dtype=dtype),
            eos_wins_argmax=torch.empty(codebook_shape, device=device, dtype=torch.bool),
            sampled_eos=torch.empty(codebook_shape, device=device, dtype=torch.bool),
            eos_frames=torch.empty(frame_shape, device=device, dtype=torch.bool),
            sampled_eos_frames=torch.empty(frame_shape, device=device, dtype=torch.bool),
            unfinished_mask=torch.empty(batch_size, device=device, dtype=torch.bool),
            finished_mask=torch.empty(batch_size, device=device, dtype=torch.bool),
            frame_positions=frame_positions,
            no_eos_positions=torch.full(
                frame_shape,
                self.frame_stacking_factor,
                device=device,
                dtype=torch.long,
            ),
            end_frame_candidates=torch.empty(frame_shape, device=device, dtype=torch.long),
            end_frame_indices=torch.empty(batch_size, device=device, dtype=torch.long),
        )

    def detect_eos_batch(
        self,
        audio_codes_multinomial: torch.Tensor,
        all_code_logits_t: torch.Tensor,
        eos_detection_method: EOSDetectionMethod,
        unfinished_items: Dict[int, bool],
        finished_items: Dict[int, bool],
        forbid_audio_eos: bool,
        scratch: EOSDetectionScratch,
    ) -> torch.Tensor:
        """Detect EOS directly from sampled codes and the EOS-vs-base-logit margin."""

        def reduce_codebooks(eos_mask: torch.Tensor, output: torch.Tensor) -> None:
            detection_type = EOSDetectionMethod.detection_type(eos_detection_method)
            if detection_type == "any":
                torch.any(eos_mask, dim=-1, out=output)
            elif detection_type == "all":
                torch.all(eos_mask, dim=-1, out=output)
            elif detection_type == "zero_cb":
                output.copy_(eos_mask[:, :, 0])
            else:
                raise ValueError(f"Invalid EOS detection method: {eos_detection_method}")

        batch_size = all_code_logits_t.size(0)
        expected_codebook_shape = (
            batch_size,
            self.frame_stacking_factor,
            self.num_audio_codebooks,
        )
        if (
            scratch.base_max.shape != expected_codebook_shape
            or scratch.base_max.device != all_code_logits_t.device
            or scratch.base_max.dtype != all_code_logits_t.dtype
        ):
            raise ValueError(
                "EOS scratch does not match current logits: "
                f"expected {expected_codebook_shape}/{all_code_logits_t.device}/{all_code_logits_t.dtype}, "
                f"got {tuple(scratch.base_max.shape)}/{scratch.base_max.device}/{scratch.base_max.dtype}"
            )
        logits = all_code_logits_t.view(
            batch_size,
            self.frame_stacking_factor,
            self.num_audio_codebooks,
            self.num_all_tokens_per_codebook,
        )
        torch.amax(logits[:, :, :, : self.codebook_size], dim=-1, out=scratch.base_max)
        torch.gt(
            logits[:, :, :, self.audio_eos_id],
            scratch.base_max,
            out=scratch.eos_wins_argmax,
        )

        scratch.unfinished_mask.zero_()
        for item_index, active in unfinished_items.items():
            if active:
                scratch.unfinished_mask[item_index] = True
        if forbid_audio_eos:
            scratch.eos_wins_argmax.zero_()
        else:
            scratch.eos_wins_argmax.masked_fill_(scratch.unfinished_mask[:, None, None], False)

        scratch.finished_mask.zero_()
        for item_index, active in finished_items.items():
            if active:
                scratch.finished_mask[item_index] = True
        scratch.eos_wins_argmax.masked_fill_(scratch.finished_mask[:, None, None], True)

        reduce_codebooks(scratch.eos_wins_argmax, scratch.eos_frames)
        if EOSDetectionMethod.sampling_type(eos_detection_method) == "argmax_or_multinomial":
            torch.eq(
                audio_codes_multinomial.transpose(1, 2),
                self.audio_eos_id,
                out=scratch.sampled_eos,
            )
            reduce_codebooks(scratch.sampled_eos, scratch.sampled_eos_frames)
            torch.logical_or(
                scratch.eos_frames,
                scratch.sampled_eos_frames,
                out=scratch.eos_frames,
            )
        torch.where(
            scratch.eos_frames,
            scratch.frame_positions,
            scratch.no_eos_positions,
            out=scratch.end_frame_candidates,
        )
        torch.amin(
            scratch.end_frame_candidates,
            dim=1,
            out=scratch.end_frame_indices,
        )
        return scratch.end_frame_indices

    def detect_forbidden_eos_batch(
        self,
        audio_codes_multinomial: torch.Tensor,
        scratch: EOSDetectionScratch,
    ) -> torch.Tensor:
        """Validate the global EOS prohibition without projecting main-decoder logits."""
        expected_shape = (
            scratch.end_frame_indices.size(0),
            self.num_audio_codebooks,
            self.frame_stacking_factor,
        )
        if tuple(audio_codes_multinomial.shape) != expected_shape:
            raise ValueError(
                f"Forbidden-EOS codes must have shape {expected_shape}, got {tuple(audio_codes_multinomial.shape)}"
            )
        if audio_codes_multinomial.dtype != torch.long:
            raise ValueError(f"Forbidden-EOS codes must use torch.long, got {audio_codes_multinomial.dtype}")
        if audio_codes_multinomial.device != scratch.end_frame_indices.device:
            raise ValueError(
                "Forbidden-EOS codes and scratch must use the same device: "
                f"{audio_codes_multinomial.device} != {scratch.end_frame_indices.device}"
            )
        expected_mask_shape = (
            expected_shape[0],
            self.frame_stacking_factor,
            self.num_audio_codebooks,
        )
        if tuple(scratch.sampled_eos.shape) != expected_mask_shape:
            raise ValueError(
                f"Forbidden-EOS scratch must have shape {expected_mask_shape}, got {tuple(scratch.sampled_eos.shape)}"
            )

        torch.eq(
            audio_codes_multinomial.transpose(1, 2),
            self.audio_eos_id,
            out=scratch.sampled_eos,
        )
        torch.ops.aten._assert_async.msg(
            torch.logical_not(scratch.sampled_eos).all(),
            "Local-AR sampled audio EOS while audio EOS was globally forbidden",
        )
        scratch.end_frame_indices.fill_(self.frame_stacking_factor)
        return scratch.end_frame_indices

    def infer_batch(
        self,
        batch,
        use_cfg=False,
        return_cross_attn_probs=False,
        compute_all_heads_attn_maps=False,
        use_local_transformer_for_inference=False,
        maskgit_n_steps=3,
        maskgit_noise_scale=0.0,
        maskgit_fixed_schedule=None,
        maskgit_dynamic_cfg_scale=False,
        maskgit_sampling_type=None,
    ):
        """Generate and decode one complete text batch through ``generate_speech``.

        Cross-attention visualizations are not available from the incremental
        decoder path. Requesting them fails explicitly rather than selecting a
        second inference implementation.
        """

        if return_cross_attn_probs or compute_all_heads_attn_maps:
            raise NotImplementedError("Cross-attention map export is not supported by incremental Magpie inference")
        if "text" not in batch:
            raise KeyError("infer_batch requires batch['text']")
        text = batch["text"]
        if not isinstance(text, torch.Tensor):
            raise TypeError(f"batch['text'] must be a torch.Tensor, got {type(text).__name__}")
        if text.ndim != 2:
            raise ValueError(f"batch['text'] must have shape (B, T), got {tuple(text.shape)}")
        batch_size = text.size(0)
        if batch_size < 1:
            raise ValueError("infer_batch requires at least one batch item")

        with torch.no_grad():
            start_time = time.perf_counter()
            generation_output = self.generate_speech(
                batch,
                chunk_state=self.create_chunk_state(batch_size=batch_size),
                end_of_text=[True] * batch_size,
                beginning_of_text=True,
                use_cfg=use_cfg,
                use_local_transformer_for_inference=use_local_transformer_for_inference,
                maskgit_n_steps=maskgit_n_steps,
                maskgit_noise_scale=maskgit_noise_scale,
                maskgit_fixed_schedule=maskgit_fixed_schedule,
                maskgit_dynamic_cfg_scale=maskgit_dynamic_cfg_scale,
                maskgit_sampling_type=maskgit_sampling_type,
            )
            predicted_audio, predicted_audio_lens, decoded_codes = self._codec_helper.codes_to_audio(
                generation_output.predicted_codes,
                generation_output.predicted_codes_lens,
            )
            if predicted_audio.device.type == "cuda":
                torch.cuda.synchronize(predicted_audio.device)
            elapsed = time.perf_counter() - start_time
            if elapsed <= 0.0:
                raise RuntimeError(f"Inference elapsed time must be positive, got {elapsed}")

        required_generation_metrics = (
            "time_to_first_prediction",
            "tts_generation_time",
            "max_frames_generated",
            "tts_generation_time_per_frame",
            "batch_size",
        )
        missing_metrics = tuple(
            metric for metric in required_generation_metrics if metric not in generation_output.rtf_metrics
        )
        if missing_metrics:
            raise RuntimeError(f"generate_speech omitted required timing metrics: {list(missing_metrics)}")

        total_audio_samples = int(predicted_audio_lens.sum().item())
        if total_audio_samples < 1:
            raise RuntimeError(f"Codec decode produced invalid total audio length {total_audio_samples}")
        total_audio_duration_generated = total_audio_samples / self.output_sample_rate
        rtf_metrics = {
            "rtf": elapsed / total_audio_duration_generated,
            **generation_output.rtf_metrics,
        }
        return InferBatchOutput(
            predicted_audio=predicted_audio,
            predicted_audio_lens=predicted_audio_lens,
            predicted_codes=decoded_codes,
            predicted_codes_lens=generation_output.predicted_codes_lens,
            rtf_metrics=rtf_metrics,
            cross_attention_maps=None,
            headwise_cross_attention_maps=None,
        )

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            test_dl_batch_size = self._test_dl.batch_size
            use_cfg = self.cfg.get('inference_use_cfg', False)
            self.inference_parameters.max_decoder_steps = self.cfg.get('max_decoder_steps', 500)
            self.inference_parameters.temperature = self.cfg.get('inference_temperature', 0.7)
            self.inference_parameters.topk = self.cfg.get('inference_topk', 80)
            self.inference_parameters.cfg_scale = self.cfg.get('inference_cfg_scale', 1.0)

            output = self.infer_batch(
                batch,
                use_cfg=use_cfg,
            )
            predicted_audio = output.predicted_audio
            predicted_audio_lens = output.predicted_audio_lens

            for logger in self.loggers:
                is_wandb = isinstance(logger, WandbLogger)
                is_tb = isinstance(logger, TensorBoardLogger)
                if not is_wandb and not is_tb:
                    raise ValueError(
                        "Invalid logger type for audio logging: {type(logger)}. Only `WandbLogger` and `TensorBoardLogger` are supported."
                    )

                for idx in range(predicted_audio.size(0)):
                    predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                    predicted_audio_np = predicted_audio_np[: predicted_audio_lens[idx]]
                    item_idx = batch_idx * test_dl_batch_size + idx

                    if is_wandb:
                        log_dict = {
                            "test/predicted_audio": wandb.Audio(
                                predicted_audio_np,
                                sample_rate=self.output_sample_rate,
                                caption="Predicted Audio",
                            ),
                        }
                        logger.experiment.log(log_dict, step=item_idx)

                    if is_tb:
                        logger.experiment.add_audio(
                            'test/predicted_audio',
                            predicted_audio_np,
                            global_step=item_idx,
                            sample_rate=self.output_sample_rate,
                        )

                    # Save the predicted audio
                    log_dir = logger.log_dir
                    audio_dir = os.path.join(log_dir, 'audios')
                    if not os.path.exists(audio_dir):
                        os.makedirs(audio_dir)
                    audio_path = os.path.join(audio_dir, f'predicted_audioRank{self.global_rank}_{item_idx}.wav')
                    sf.write(audio_path, predicted_audio_np, self.output_sample_rate)

    def multi_validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Called for each validation dataloader at the end of validation epoch.
        Computes metrics for this specific dataloader.

        Args:
            outputs: List of outputs from validation_step for this specific dataloader
            dataloader_idx: Index of the current dataloader

        Returns:
            A tuple of (log_dict, moe_expert_data):
                - log_dict: scalar metrics suitable for self.log()
                - moe_expert_data: per-expert usage/selection_freq tensors of shape (num_experts,), or None
        """

        def collect_required_metric(outputs, key, dim=None):
            values = [x[key] for x in outputs if key in x and x[key] is not None]
            if len(values) == 0:
                raise ValueError(
                    f"No valid values found for required metric '{key}' in validation outputs "
                    f"for dataloader {dataloader_idx}. This indicates an issue with validation."
                )
            return torch.stack(values).mean(dim=dim)

        def collect_optional_metric(outputs, key, dim=None):
            """Collect optional metric - returns None if not found."""
            values = [x[key] for x in outputs if key in x and x[key] is not None]
            if len(values) == 0:
                return None
            return torch.stack(values).mean(dim=dim)

        if len(outputs) == 0:
            raise ValueError(
                f"No validation outputs for dataloader {dataloader_idx}. "
                f"This indicates an issue with the validation dataloader or validation step."
            )

        # Compute required metrics
        val_loss = collect_required_metric(outputs, 'val_loss')
        val_codebook_loss = collect_required_metric(outputs, 'val_codebook_loss')

        log_dict = {
            'loss': val_loss,
            'codebook_loss': val_codebook_loss,
        }

        # Compute optional metrics
        VAL_OPTIONAL_METRICS = [
            'val_alignment_loss',
            'val_aligner_encoder_loss',
            'val_local_transformer_loss',
            'val_moe_load_balancing_loss',
            'val_moe_router_z_loss',
        ]
        for metric_key in VAL_OPTIONAL_METRICS:
            metric_value = collect_optional_metric(outputs, metric_key)
            if metric_value is not None:
                log_dict[metric_key.removeprefix('val_')] = metric_value

        # Exclude MoE metrics whose loss scale is disabled
        if self.use_moe:
            if self.moe_auxiliary_loss.load_balancing_loss.loss_scale <= 0:
                log_dict.pop('moe_load_balancing_loss', None)
            if self.moe_auxiliary_loss.router_z_loss.loss_scale <= 0:
                log_dict.pop('moe_router_z_loss', None)

        # Collect per-expert usage vectors
        val_moe_expert_usage_stats = [
            x.get('val_moe_expert_usage_stats') for x in outputs if x.get('val_moe_expert_usage_stats') is not None
        ]
        moe_expert_data = None
        if len(val_moe_expert_usage_stats) > 0:
            val_moe_expert_usage = collect_required_metric(val_moe_expert_usage_stats, 'expert_usage', dim=0)
            val_moe_expert_selection_freq = collect_required_metric(
                val_moe_expert_usage_stats, 'expert_selection_freq', dim=0
            )
            val_layer_expert_usage = collect_required_metric(val_moe_expert_usage_stats, 'layer_expert_usage', dim=0)
            ideal_usage = val_moe_expert_usage_stats[0]['ideal_usage']
            moe_expert_data = {
                'moe_expert_usage': val_moe_expert_usage,
                'moe_expert_selection_freq': val_moe_expert_selection_freq,
                'layer_expert_usage': val_layer_expert_usage,
                'ideal_usage': ideal_usage,
            }

        return log_dict, moe_expert_data

    def on_validation_epoch_end(self):
        """
        Computes and logs metrics across all validation dataloaders.

        Three-phase structure:
        1. Compute — aggregates metrics and collect media/heatmap data from all dataloaders.
        2. WandB media — logs all non-scalar media (audio, attention images, MoE heatmaps).
        3. Scalars — logs loss metrics and per-expert usage scalars.
        """
        if len(self.validation_step_outputs) == 0:
            return {}

        num_dataloaders = len(self.validation_step_outputs)

        # --- Phase 1: Compute all metrics + collect media data ---
        all_moe_expert_data: List[Tuple[str, Dict[str, torch.Tensor]]] = []
        all_media_data: List[Dict[str, Any]] = []
        per_dl_logs: List[Tuple[str, Dict[str, torch.Tensor]]] = []
        aggregated_metrics: Dict[str, List[torch.Tensor]] = {}

        for dataloader_idx, val_outputs in enumerate(self.validation_step_outputs):
            if len(val_outputs) == 0:
                raise ValueError(
                    f"Validation dataloader {dataloader_idx} produced no outputs. "
                    f"Check that the dataset is not empty and validation_step is working correctly."
                )

            dataloader_logs, moe_expert_data = self.multi_validation_epoch_end(
                val_outputs, dataloader_idx=dataloader_idx
            )

            dataloader_prefix = self.get_validation_dataloader_prefix(dataloader_idx)
            per_dl_logs.append((dataloader_prefix, dataloader_logs))

            if moe_expert_data is not None:
                all_moe_expert_data.append((dataloader_prefix, moe_expert_data))

            if len(val_outputs) > 0 and 'media_data' in val_outputs[0]:
                all_media_data.append(val_outputs[0]['media_data'])

            for metric_name, metric_value in dataloader_logs.items():
                aggregated_metrics.setdefault(metric_name, []).append(metric_value)

        for idx in range(num_dataloaders):
            self.validation_step_outputs[idx].clear()

        # Validate required metrics were collected
        for required_metric in ['loss', 'codebook_loss']:
            if required_metric not in aggregated_metrics or len(aggregated_metrics[required_metric]) == 0:
                raise ValueError(f"No {required_metric} collected from any dataloader.")

        # --- Phase 2: Single WandB media log (rank 0 only) ---
        if self.global_rank == 0:
            global_step = int(self.global_step)
            wandb_media: Dict[str, Any] = {}

            for media_data in all_media_data:
                media_entries = self._collect_wandb_media_and_log_tb(**media_data, global_step=global_step)
                wandb_media.update(media_entries)

            # heatmaps show layer×expert routing structure
            if all_moe_expert_data:
                for dataset_name, moe_data in all_moe_expert_data:
                    heatmap_np = plot_expert_usage_heatmap_to_numpy(
                        layer_expert_usage=moe_data['layer_expert_usage'].float().cpu().numpy(),
                        ideal_usage=moe_data['ideal_usage'],
                        title=f"MoE Expert Usage — {dataset_name} (step {int(self.global_step)})",
                    )
                    wandb_media[f"MoE:{dataset_name}/expert_usage_heatmap"] = wandb.Image(heatmap_np)

                if self._moe_train_layer_usage_accum is not None and self._moe_train_accum_steps > 0:
                    avg_layer_usage = self._moe_train_layer_usage_accum / self._moe_train_accum_steps
                    heatmap_np = plot_expert_usage_heatmap_to_numpy(
                        layer_expert_usage=avg_layer_usage.float().cpu().numpy(),
                        ideal_usage=1.0 / self._moe_num_experts,
                        title=f"MoE Expert Usage — train ({self._moe_train_accum_steps} steps avg, step {int(self.global_step)})",
                    )
                    wandb_media["MoE:train/expert_usage_heatmap"] = wandb.Image(heatmap_np)

                    self._moe_train_layer_usage_accum.zero_()
                    self._moe_train_accum_steps = 0

            if wandb_media:
                for logger in self.loggers:
                    if isinstance(logger, WandbLogger):
                        logger.experiment.log(wandb_media, commit=False)

        # --- Phase 3: Scalar metrics ---
        for dataloader_prefix, dataloader_logs in per_dl_logs:
            for metric_name, metric_value in dataloader_logs.items():
                self.log(
                    f"Loss:{dataloader_prefix}/{metric_name}",
                    metric_value,
                    prog_bar=(num_dataloaders == 1),
                    sync_dist=True,
                )

        checkpoint_loss = aggregated_metrics['loss'][0]
        if num_dataloaders > 1:
            for metric_name, metric_values in aggregated_metrics.items():
                if "loss" in metric_name:
                    avg_value = torch.stack(metric_values).mean()
                    self.log(f"Loss:val_avg/{metric_name}", avg_value, prog_bar=True, sync_dist=True)
                    if metric_name == 'loss':
                        checkpoint_loss = avg_value

        self.log(
            "val_loss",
            checkpoint_loss,
            prog_bar=False,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            logger=False,
            enable_graph=False,
        )

        if all_moe_expert_data:
            for dataset_name, moe_data in all_moe_expert_data:
                expert_usage = moe_data['moe_expert_usage']
                expert_sel_freq = moe_data['moe_expert_selection_freq']

                for eidx in range(len(expert_usage)):
                    self.log(f'MoE:{dataset_name}/Expert_{eidx:02d}_usage', expert_usage[eidx], sync_dist=True)
                    self.log(
                        f'MoE:{dataset_name}/Expert_{eidx:02d}_selection_freq', expert_sel_freq[eidx], sync_dist=True
                    )

        return {}

    def get_dataset(self, dataset_cfg, dataset_type):
        if 'datasets' not in dataset_cfg or not isinstance(dataset_cfg.datasets, (dict, DictConfig)):
            raise ValueError(
                "Expected 'datasets' key (dict) in dataset config with _target_, dataset_meta, etc. "
                f"Got keys: {list(dataset_cfg.keys())}"
            )

        dataset = safe_instantiate(
            dataset_cfg.datasets,
            sample_rate=self.sample_rate,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            num_audio_codebooks=self.data_num_audio_codebooks,
            codec_model_samples_per_frame=self.codec_model_samples_per_frame,
            prior_scaling_factor=self.cfg.prior_scaling_factor,
            load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
            dataset_type=dataset_type,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
            use_text_conditioning_tokenizer=self.cfg.use_text_conditioning_encoder,
            text_conditioning_tokenizer_name=self.text_conditioning_tokenizer_name,
            pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
            context_duration_min=self.cfg.context_duration_min,
            context_duration_max=self.cfg.context_duration_max,
            text_context_remapping=self.text_context_remapping,
            text_context_remapping_prob=self.text_context_remapping_prob,
        )
        dataset.load_16khz_audio = False
        dataset.tokenizer_config = (
            self.cfg.text_tokenizers
        )  # This will be used in worker_init_fn for instantiating tokenizer
        return dataset

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """
        Setup validation data with support for multiple datasets.
        Overrides parent class to handle both non-lhotse and lhotse dataloaders.

        Non-lhotse config (datasets is a dict -- single dataloader, multiplicity via dataset_meta)::

            validation_ds:
                datasets:
                    _target_: nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDataset
                    dataset_meta: ...
                    min_duration: 0.2
                    max_duration: 20.0
                dataloader_params: ...

        Note: Non-lhotse creates a single dataloader even when dataset_meta contains
        multiple entries (e.g., ``{en: ..., es: ...}``). All datasets are mixed
        in one dataloader, so validation metrics are logged jointly (e.g.,
        prefix ``"en+es"``) rather than per-dataset. For per-dataset validation
        metrics, use the lhotse config with separate datasets list entries.

        Lhotse config (datasets is a list -- multiple dataloaders)::

            validation_ds:
                use_lhotse: true
                # ... shared settings ...
                datasets:
                    - name: "val_set_0"
                      input_cfg: [...] or path to an external YAML file
                    - name: "val_set_1"
                      input_cfg: [...] or path to an external YAML file
        """
        # Set placeholders that may be overridden
        self._val_dl_idx: int = 0
        self._validation_names: Optional[List[str]] = None
        self._validation_dl: Optional[torch.utils.data.DataLoader] = None

        # Preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        if 'datasets' not in val_data_config:
            raise ValueError(
                "validation_ds config must contain a 'datasets' key. "
                "For non-lhotse: a dict with _target_, dataset_meta, etc. "
                "For lhotse: a list of dataset configurations. "
                "See magpietts.yaml or magpietts_lhotse.yaml for examples."
            )

        datasets_value = val_data_config.datasets

        # Non-lhotse: datasets is a dict (single dataloader, multiplicity via dataset_meta)
        if isinstance(datasets_value, (dict, DictConfig)):
            dataset_meta = datasets_value.get('dataset_meta', {})
            if dataset_meta:
                val_name = '+'.join(dataset_meta.keys())
            else:
                val_name = 'val_set_0'
            logging.info(f"Setting up single non-lhotse validation dataloader: '{val_name}'")
            self._validation_names = [val_name]
            self._validation_dl = [self._setup_test_dataloader(val_data_config)]
            return

        # Lhotse: datasets is a path to an external YAML file (supports local paths and remote URLs like s3://) or a list
        if isinstance(datasets_value, (str, Path)):
            logging.info(f"Loading validation datasets from external file: {datasets_value}")
            datasets_list = OmegaConf.create(load_yaml(datasets_value))
        elif isinstance(datasets_value, (list, ListConfig)):
            datasets_list = datasets_value
        else:
            raise ValueError(
                f"Lhotse 'datasets' in `validation_ds` must be a non-empty list of dataset configurations. "
                f"Got: {type(datasets_value).__name__}"
            )

        if len(datasets_list) == 0:
            raise ValueError("Lhotse 'datasets' in `validation_ds` must be a non-empty list.")

        logging.info(f"Setting up {len(datasets_list)} validation dataset(s)")

        dataloaders = []
        dataset_names = []

        # Extract shared config (everything except 'datasets' key)
        shared_config = OmegaConf.create(val_data_config)
        shared_config.pop('datasets', None)

        for idx, dataset_config in enumerate(datasets_list):
            merged_config = OmegaConf.merge(shared_config, dataset_config)

            if isinstance(dataset_config, (dict, DictConfig)) and 'name' in dataset_config:
                dataset_name = dataset_config['name']
            else:
                dataset_name = f"val_set_{idx}"

            dataset_names.append(dataset_name)

            # Remove 'name' field from config as it's not needed for dataloader setup
            temp_config = OmegaConf.create(merged_config)
            temp_config.pop('name', None)

            dataloader = self._setup_test_dataloader(temp_config)
            dataloaders.append(dataloader)
            logging.info(f"  - Validation dataset {idx}: '{dataset_name}'")

        self._validation_names = dataset_names
        self._validation_dl = dataloaders
        logging.info(f"Successfully setup {len(dataloaders)} validation dataloader(s)")

    def get_lhotse_dataloader(self, dataset_cfg, mode='train') -> torch.utils.data.DataLoader:
        # TODO @xueyang: better to distinguish cfg. self.cfg is the model cfg, while cfg here is train_ds cfg. Also
        #   cfg is a classifier-free guidance.
        dataset = MagpieTTSLhotseDataset(
            sample_rate=self.sample_rate,
            volume_norm=dataset_cfg.volume_norm,
            codec_model_samples_per_frame=self.codec_model_samples_per_frame,
            num_audio_codebooks=self.data_num_audio_codebooks,
            prior_scaling_factor=self.cfg.prior_scaling_factor,
            load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
            dataset_type=mode,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
            load_16khz_audio=False,
            pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
            context_duration_min=self.cfg.context_duration_min,
            context_duration_max=self.cfg.context_duration_max,
            use_text_conditioning_tokenizer=self.cfg.use_text_conditioning_encoder,
            text_conditioning_tokenizer_name=self.text_conditioning_tokenizer_name,
            tokenizer_config=self.cfg.text_tokenizers,
            text_context_remapping=self.text_context_remapping,
            text_context_remapping_prob=self.text_context_remapping_prob,
        )
        data_loader = get_lhotse_dataloader_from_config(
            config=dataset_cfg,
            global_rank=self.global_rank,
            world_size=self.world_size,
            dataset=dataset,
        )
        return data_loader

    def setup_training_data(self, dataset_cfg):
        if dataset_cfg.get("use_lhotse", False):
            # TODO @xueyang: better to distinguish cfg. self.cfg is the model cfg, while cfg here is train_ds cfg. Also
            #   cfg is a classifier-free guidance.

            # specify target sampling rate the same as codec model's because lhotse config defaults 16_000.
            if not isinstance(dataset_cfg, DictConfig):
                dataset_cfg = OmegaConf.create(dataset_cfg)
            OmegaConf.set_struct(dataset_cfg, False)
            dataset_cfg.update({"sample_rate": self.sample_rate})
            OmegaConf.set_struct(dataset_cfg, True)

            self._train_dl = self.get_lhotse_dataloader(dataset_cfg, mode='train')
        else:
            dataset = self.get_dataset(dataset_cfg, dataset_type='train')
            sampler = dataset.get_sampler(dataset_cfg.dataloader_params.batch_size, world_size=self.trainer.world_size)
            persistent_workers = True
            if dataset_cfg.dataloader_params.num_workers == 0:
                persistent_workers = False
                # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
                dataset.text_tokenizer = setup_tokenizers(
                    all_tokenizers_config=self.cfg.text_tokenizers,
                    mode='train',
                )
            self._train_dl = torch.utils.data.DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                sampler=sampler,
                **dataset_cfg.dataloader_params,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
            )

    def _setup_test_dataloader(self, dataset_cfg) -> torch.utils.data.DataLoader:
        if dataset_cfg.get("use_lhotse", False):
            # specify target sampling rate the same as codec model's because lhotse config defaults 16_000.
            if not isinstance(dataset_cfg, DictConfig):
                dataset_cfg = OmegaConf.create(dataset_cfg)
            OmegaConf.set_struct(dataset_cfg, False)
            dataset_cfg.update({"sample_rate": self.sample_rate})
            OmegaConf.set_struct(dataset_cfg, True)
            data_loader = self.get_lhotse_dataloader(dataset_cfg, mode='test')
        else:
            dataset = self.get_dataset(dataset_cfg, dataset_type='test')
            persistent_workers = True
            if dataset_cfg.dataloader_params.num_workers == 0:
                persistent_workers = False
                # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
                dataset.text_tokenizer = setup_tokenizers(all_tokenizers_config=self.cfg.text_tokenizers, mode='test')

            data_loader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                **dataset_cfg.dataloader_params,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
            )
        return data_loader

    def setup_validation_data(self, dataset_cfg):
        """Required by ModelPT (abstract). Use setup_multiple_validation_data instead."""
        self._validation_names = ['val_set_0']
        self._validation_dl = [self._setup_test_dataloader(dataset_cfg)]

    def setup_test_data(self, dataset_cfg):
        self._test_dl = self._setup_test_dataloader(dataset_cfg)

    def _get_normalized_text(self, transcript: str, language: str) -> str:
        """Get normalized text using cached normalizer for the specified language.

        Args:
            transcript: Raw text to normalize.
            language: Language code (e.g., 'en', 'de', 'es').

        Returns:
            Normalized text, or original text if normalization fails/unavailable.
        """
        # Check if normalizer for this language is already cached
        if language not in self._text_normalizers:
            try:
                from nemo_text_processing.text_normalization.normalize import Normalizer

                normalizer = Normalizer(input_case='cased', lang=language)
                self._text_normalizers[language] = normalizer
                logging.info(f"Initialized text normalizer for language: {language}")
            except ImportError:
                self._text_normalizers[language] = None
                logging.warning(
                    "nemo_text_processing not installed. Skipping text normalization. "
                    "Install with: pip install nemo_text_processing"
                )
            except Exception as e:
                # Handle unsupported language or other initialization errors
                self._text_normalizers[language] = None
                logging.warning(
                    f"Failed to initialize text normalizer for language '{language}': {e}. "
                    f"Skipping text normalization. Text will be used as-is."
                )

        # Use cached normalizer if available
        normalizer = self._text_normalizers[language]
        if normalizer is not None:
            normalized_text = normalizer.normalize(transcript, verbose=False)
            return normalized_text

        return transcript

    def do_tts(
        self,
        transcript: str,
        language: str = "en",
        apply_TN: bool = False,
        use_cfg: bool = True,
        speaker_index: Optional[int] = None,
        local_ar_seed: Optional[int] = None,
    ) -> tuple:
        """
        Generate speech from raw text transcript.

        This is a convenience method for single-utterance text-to-speech synthesis.
        For batch processing, use `infer_batch` directly. Only supports baked context embedding
        context injection, NO audio conditioning and text conditioning.
        Custom voice generation is not supported by this method.

        Args:
            transcript: Raw text to synthesize.
            language: Language code for text normalization and tokenization.
                Supported values depend on model's tokenizer configuration.
                Common: "en" (English), "de" (German), "es" (Spanish), etc.
            apply_TN: Whether to apply text normalization to the transcript.
                If True, uses nemo_text_processing for normalization.
            use_cfg: Whether to use classifier-free guidance.
            speaker_index: Speaker index for multi-speaker baked embeddings.
                Valid range: [0, num_baked_speakers - 1]. If None, uses speaker 0.
                Only applicable for models with baked context embeddings.
            local_ar_seed: Explicit seed for stochastic Local AR generation.
                Required when Local AR sampling is stochastic.

        Returns:
            Tuple of (audio, audio_len) where:
                audio: Generated audio waveform. Shape: (1, T_audio).
                audio_len: Length of generated audio in samples. Shape: (1,).

        Raises:
            ValueError: If model does not have a baked context embedding.
            ValueError: If speaker_index is out of valid range.
            ImportError: If apply_TN=True but nemo_text_processing is not installed.

        Example:
            >>> # If text does not need to be normalized
            >>> audio, audio_len = model.do_tts("Hello, how are you today?")
            >>>
            >>> # If text needs to be normalized
            >>> audio, audio_len = model.do_tts(
            ...     "Hello, how are you today?",
            ...     apply_TN=True,
            ... )
            >>>
            >>> # Use a specific speaker (for multi-speaker models)
            >>> audio, audio_len = model.do_tts(
            ...     "Hello!", speaker_index=2
            ... )
        """
        if not self.has_baked_context_embedding:
            raise ValueError(
                "Model does not have a baked context embedding. Please use a checkpoint with a baked context embedding."
            )
        # Workaround for bug in Ja normalizer, Ja normalizer does not work well with spaces.
        if language == "ja":
            transcript = re.sub(r'\s+', '', transcript)
        # Apply text normalization if requested
        normalized_text = (
            self._get_normalized_text(transcript=transcript, language=language) if apply_TN else transcript
        )

        # Determine tokenizer name based on language using centralized mapping
        available_tokenizers = list(self.tokenizer.tokenizers.keys())
        available_mapping = self.cfg.get("language_to_tokenizer_mapping", None)
        tokenizer_name = get_tokenizer_for_language(
            language, available_tokenizers, language_tokenizer_map=available_mapping
        )
        logging.info(f"Using tokenizer '{tokenizer_name}' for language '{language}'")

        # Unified inference path: chunk_text_for_inference automatically decides
        # whether to split based on language-specific thresholds
        # - Short text (below threshold): returns single chunk
        # - Long text (above threshold): returns multiple sentence chunks
        chunked_tokens, chunked_tokens_len, _ = chunk_text_for_inference(
            text=normalized_text,
            language=language,
            tokenizer_name=tokenizer_name,
            text_tokenizer=self.tokenizer,
            eos_token_id=self.eos_id,
        )

        num_chunks = len(chunked_tokens)

        with torch.no_grad():
            chunk_state = self.create_chunk_state(batch_size=1)
            all_codes = []

            for chunk_idx, (tokens, tokens_len) in enumerate(zip(chunked_tokens, chunked_tokens_len)):
                batch = {
                    'text': tokens.unsqueeze(0).to(self.device),
                    'text_lens': torch.tensor([tokens_len], device=self.device, dtype=torch.long),
                    'speaker_indices': speaker_index,
                }
                end_of_text = [chunk_idx == num_chunks - 1]
                beginning_of_text = chunk_idx == 0

                output = self.generate_speech(
                    batch,
                    chunk_state=chunk_state,
                    end_of_text=end_of_text,
                    beginning_of_text=beginning_of_text,
                    local_ar_seed=local_ar_seed,
                    use_cfg=use_cfg,
                    use_local_transformer_for_inference=self.local_transformer_type != LocalTransformerType.NO_LT,
                )
                if output.predicted_codes_lens[0] > 0:
                    all_codes.append(output.predicted_codes[0, :, : output.predicted_codes_lens[0]])

            # Concatenate and convert to audio
            if len(all_codes) > 0:
                concatenated_codes = torch.cat(all_codes, dim=1).unsqueeze(0)
                codes_lens = torch.tensor([concatenated_codes.shape[2]], device=self.device, dtype=torch.long)
                predicted_audio, predicted_audio_lens, _ = self._codec_helper.codes_to_audio(
                    concatenated_codes, codes_lens
                )
                return predicted_audio, predicted_audio_lens
            else:
                raise RuntimeError("Magpie generation completed without producing any codec frames")

    def _build_first_submission_cuda_graph_key(
        self,
        *,
        context_tensors: ContextTensorsOutput,
        dummy_cond: torch.Tensor,
        dummy_cond_mask: torch.Tensor,
        dummy_additional_decoder_input: Optional[torch.Tensor],
        dummy_addition_dec_mask: Optional[torch.Tensor],
        actual_text_length: int,
        submission_profile: PackedStreamingSubmissionProfile,
    ) -> FirstSubmissionCudaGraphKey:
        """Describe one exact, explicitly warmed first-submission shape."""

        if actual_text_length < 1:
            raise ValueError(f"actual_text_length must be positive, got {actual_text_length}")
        if not isinstance(context_tensors.cond, torch.Tensor) or not isinstance(
            context_tensors.cond_mask, torch.Tensor
        ):
            raise TypeError("First-submission CUDA graph requires tensor conditioning")
        if context_tensors.cond.shape != dummy_cond.shape:
            raise ValueError(
                "CFG conditional and unconditional context shapes differ: "
                f"{tuple(context_tensors.cond.shape)} != {tuple(dummy_cond.shape)}"
            )
        if context_tensors.cond_mask.shape != dummy_cond_mask.shape:
            raise ValueError(
                "CFG conditional and unconditional mask shapes differ: "
                f"{tuple(context_tensors.cond_mask.shape)} != {tuple(dummy_cond_mask.shape)}"
            )
        if context_tensors.cond.device.type != "cuda":
            raise RuntimeError(f"First-submission CUDA graph requires CUDA context, got {context_tensors.cond.device}")
        device_index = context_tensors.cond.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        additional_input_shape = None
        additional_mask_shape = None
        if context_tensors.additional_decoder_input is None:
            if (
                context_tensors.additional_decoder_mask is not None
                or dummy_additional_decoder_input is not None
                or dummy_addition_dec_mask is not None
            ):
                raise ValueError("CFG decoder prefix is only partially specified")
        else:
            if (
                context_tensors.additional_decoder_mask is None
                or dummy_additional_decoder_input is None
                or dummy_addition_dec_mask is None
            ):
                raise ValueError("CFG decoder prefix is only partially specified")
            if context_tensors.additional_decoder_input.shape != dummy_additional_decoder_input.shape:
                raise ValueError(
                    "CFG conditional and unconditional decoder prefix shapes differ: "
                    f"{tuple(context_tensors.additional_decoder_input.shape)} != "
                    f"{tuple(dummy_additional_decoder_input.shape)}"
                )
            if context_tensors.additional_decoder_mask.shape != dummy_addition_dec_mask.shape:
                raise ValueError(
                    "CFG conditional and unconditional decoder prefix mask shapes differ: "
                    f"{tuple(context_tensors.additional_decoder_mask.shape)} != "
                    f"{tuple(dummy_addition_dec_mask.shape)}"
                )
            additional_input_shape = tuple(context_tensors.additional_decoder_input.shape)
            additional_mask_shape = tuple(context_tensors.additional_decoder_mask.shape)

        return FirstSubmissionCudaGraphKey(
            device_index=device_index,
            dtype=context_tensors.text_encoder_out.dtype,
            actual_text_length=actual_text_length,
            conditional_shape=tuple(context_tensors.cond.shape),
            conditional_mask_shape=tuple(context_tensors.cond_mask.shape),
            additional_input_shape=additional_input_shape,
            additional_mask_shape=additional_mask_shape,
            max_decoder_steps=self.inference_parameters.max_decoder_steps // self.frame_stacking_factor,
            temperature=self.inference_parameters.temperature,
            topk=self.inference_parameters.topk,
            cfg_scale=self.inference_parameters.cfg_scale,
            eos_detection_method=EOSDetectionMethod(self.inference_parameters.eos_detection_method),
            attention_prior_epsilon=self.inference_parameters.attention_prior_epsilon,
            attention_prior_lookahead_window=self.inference_parameters.attention_prior_lookahead_window,
            attention_sink_threshold=self.inference_parameters.attention_sink_threshold,
            estimate_alignment_from_layers=(
                tuple(self.inference_parameters.estimate_alignment_from_layers)
                if self.inference_parameters.estimate_alignment_from_layers is not None
                else None
            ),
            apply_prior_to_layers=(
                tuple(self.inference_parameters.apply_prior_to_layers)
                if self.inference_parameters.apply_prior_to_layers is not None
                else None
            ),
            frame_stacking_factor=self.frame_stacking_factor,
            num_audio_codebooks=self.num_audio_codebooks,
            codebook_size=self.codebook_size,
            submission_profile=submission_profile,
        )

    def warmup_first_submission_cuda_graph(
        self,
        batch,
        *,
        submission_profile: PackedStreamingSubmissionProfile,
    ) -> FirstSubmissionCudaGraphReceipt:
        """Explicitly capture one exact text-shape bucket before serving.

        This prototype intentionally does not capture unseen shapes on a
        request. Once enabled, a target request whose key was not warmed fails
        closed. Production deployment should either enumerate its accepted
        exact shapes or add a separately gated padded-bucket contract.
        """

        if not isinstance(submission_profile, PackedStreamingSubmissionProfile):
            raise TypeError(
                "warmup_first_submission_cuda_graph requires a PackedStreamingSubmissionProfile, "
                f"got {type(submission_profile).__name__}"
            )
        if self.training:
            raise RuntimeError("warmup_first_submission_cuda_graph requires model.eval()")
        if self.device.type != "cuda":
            raise RuntimeError(f"warmup_first_submission_cuda_graph requires a CUDA model, got {self.device}")
        if self.local_transformer_type != LocalTransformerType.AR:
            raise ValueError(
                "warmup_first_submission_cuda_graph requires an autoregressive local transformer, "
                f"got {self.local_transformer_type}"
            )
        if self.model_type == "multi_encoder_context_tts":
            raise ValueError("First-submission CUDA graph does not support multi-encoder context")
        if self.frame_stacking_factor != 2 or self.inference_parameters.min_generated_frames != 4:
            raise ValueError("First-submission CUDA graph requires frame_stacking_factor=2 and min_generated_frames=4")
        if (
            not self.inference_parameters.apply_attention_prior
            or self.inference_parameters.start_prior_after_n_audio_steps != 0
            or not self.inference_parameters.ignore_finished_sentence_tracking
        ):
            raise ValueError(
                "First-submission CUDA graph requires Sofia packed alignment from decoder step zero "
                "with finished-sentence tracking disabled"
            )
        if "text" not in batch or "text_lens" not in batch:
            raise KeyError("First-submission CUDA graph warmup requires text and text_lens")

        warmup_batch = copy.deepcopy(batch)
        if warmup_batch["text"].size(0) != 1 or warmup_batch["text_lens"].numel() != 1:
            raise ValueError("First-submission CUDA graph warmup requires batch size one")
        chunk_state = self.create_chunk_state(batch_size=1)
        current_chunk_len = copy.deepcopy(warmup_batch["text_lens"].detach())
        warmup_batch, max_text_len = self._prepare_chunked_text_tensors(
            chunk_state,
            warmup_batch,
            current_chunk_len,
            True,
            warmup_batch["text"].device,
        )
        context_tensors = self.prepare_context_tensors(warmup_batch)
        self._update_context_from_history(
            chunk_state,
            context_tensors,
            current_chunk_len,
            max_text_len,
            True,
            warmup_batch["text_lens"],
            1,
        )
        (
            dummy_cond,
            dummy_cond_mask,
            dummy_additional_decoder_input,
            dummy_addition_dec_mask,
            _,
        ) = self.prepare_dummy_cond_for_cfg(
            context_tensors.cond,
            context_tensors.cond_mask,
            context_tensors.additional_decoder_input,
            context_tensors.additional_decoder_mask,
        )
        if not isinstance(dummy_cond, torch.Tensor) or not isinstance(dummy_cond_mask, torch.Tensor):
            raise TypeError("First-submission CUDA graph requires tensor CFG conditioning")
        actual_text_length = int(warmup_batch["text_lens"][0].item())
        key = self._build_first_submission_cuda_graph_key(
            context_tensors=context_tensors,
            dummy_cond=dummy_cond,
            dummy_cond_mask=dummy_cond_mask,
            dummy_additional_decoder_input=dummy_additional_decoder_input,
            dummy_addition_dec_mask=dummy_addition_dec_mask,
            actual_text_length=actual_text_length,
            submission_profile=submission_profile,
        )
        if key in self._first_submission_cuda_graph_runtimes:
            raise RuntimeError(f"First-submission CUDA graph bucket was already captured: {key}")

        torch.cuda.synchronize(self.device)
        allocated_before = torch.cuda.memory_allocated(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        capture_started = time.perf_counter()
        runtime = FirstSubmissionCudaGraphRuntime(
            model=self,
            key=key,
            context_tensors=context_tensors,
            dummy_cond=dummy_cond,
            dummy_cond_mask=dummy_cond_mask,
            dummy_additional_decoder_input=dummy_additional_decoder_input,
            dummy_addition_dec_mask=dummy_addition_dec_mask,
            last_attended=chunk_state.last_attended_timesteps[-1],
        )
        torch.cuda.synchronize(self.device)
        receipt = FirstSubmissionCudaGraphReceipt(
            key=key,
            capture_seconds=time.perf_counter() - capture_started,
            allocated_bytes_before=allocated_before,
            allocated_bytes_after=torch.cuda.memory_allocated(self.device),
            reserved_bytes_before=reserved_before,
            reserved_bytes_after=torch.cuda.memory_reserved(self.device),
        )
        self._first_submission_cuda_graph_runtimes[key] = runtime
        self._first_submission_cuda_graph_receipts[key] = receipt
        self._first_submission_cuda_graph_enabled = True
        return receipt

    def first_submission_cuda_graph_receipts(self) -> Tuple[FirstSubmissionCudaGraphReceipt, ...]:
        """Return immutable evidence for every explicitly warmed bucket."""

        return tuple(self._first_submission_cuda_graph_receipts.values())

    def invalidate_first_submission_cuda_graphs(self) -> None:
        """Release every outer graph after active continuations finish."""

        runtimes = tuple(self._first_submission_cuda_graph_runtimes.values())
        self._first_submission_cuda_graph_runtimes.clear()
        self._first_submission_cuda_graph_receipts.clear()
        self._first_submission_cuda_graph_enabled = False
        for runtime in runtimes:
            runtime.synchronize_before_release()

    def _acquire_first_submission_cuda_graph(
        self,
        key: FirstSubmissionCudaGraphKey,
    ) -> FirstSubmissionCudaGraphLease:
        if not self._first_submission_cuda_graph_enabled:
            raise RuntimeError("First-submission CUDA graph was not enabled by explicit warmup")
        runtime = self._first_submission_cuda_graph_runtimes.get(key)
        if runtime is None:
            warmed_lengths = sorted(
                {warmed_key.actual_text_length for warmed_key in self._first_submission_cuda_graph_runtimes}
            )
            warmed_profiles = sorted(
                {
                    (
                        warmed_key.submission_profile.first_chunk_frames,
                        warmed_key.submission_profile.steady_chunk_frames,
                    )
                    for warmed_key in self._first_submission_cuda_graph_runtimes
                }
            )
            raise RuntimeError(
                "First-submission CUDA graph request shape was not explicitly warmed; "
                "request-time capture and fallback are disabled: "
                f"requested={key}, warmed_text_lengths={warmed_lengths}, warmed_profiles={warmed_profiles}"
            )
        return runtime.acquire()

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    def warmup_local_ar_graph(self, *, batch_size: int, use_cfg: bool = False) -> None:
        """Capture the local-AR CUDA Graph before serving inference requests.

        This method does not run the text encoder, main decoder, or codec. Call
        ``do_tts`` separately with a validated short utterance when full-pipeline
        warmup is required.
        """

        if self.training:
            raise RuntimeError("warmup_local_ar_graph requires model.eval() before CUDA Graph capture")
        if self.local_transformer_type != LocalTransformerType.AR:
            raise ValueError(
                "warmup_local_ar_graph requires an autoregressive local transformer, "
                f"got {self.local_transformer_type}"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        graph_input = self.audio_embeddings[0].weight
        self._lt_helper.warmup_autoregressive(
            actual_batch_size=batch_size,
            input_dim=int(self.cfg.decoder.d_model),
            device=graph_input.device,
            dtype=graph_input.dtype,
            temperature=self.inference_parameters.temperature,
            topk=self.inference_parameters.topk,
            use_cfg=use_cfg,
            cfg_scale=self.inference_parameters.cfg_scale,
        )

    def warmup_streaming_codec(self, *, chunk_frames: int = 8) -> None:
        """Warm the exact stateful CUDA codec and pinned D2H path.

        The disposable session owns fresh causal state, so dummy warmup audio
        cannot enter a later synthesis session. Failures are propagated; a
        requested CUDA warmup never falls back to an unwarmed path.
        """

        if self.training:
            raise RuntimeError("warmup_streaming_codec requires model.eval()")
        if chunk_frames < 1:
            raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")
        if self.device.type != "cuda":
            raise RuntimeError(f"warmup_streaming_codec requires a CUDA model, got {self.device}")

        decoder = CausalCodecStreamingDecoder(
            codec_model=self._codec_helper.codec_model,
            codec_converter=self._codec_helper.codec_converter,
        )
        callback_chunks: list[StreamingPcmChunk] = []
        session = AsyncCodecSynthesisSession.for_cuda(
            decoder=decoder,
            callback=callback_chunks.append,
            device=self.device,
            max_queued_chunks=1,
        )
        dummy_codes = torch.zeros(
            (1, self.num_audio_codebooks, chunk_frames),
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            session.submit_chunk(
                StreamingCodecChunk(
                    codes=dummy_codes,
                    first_codec_frame=0,
                    codec_frame_count=chunk_frames,
                    final=True,
                ),
                producer_stream=torch.cuda.current_stream(self.device),
            )
            session.close()

        if len(callback_chunks) != 1:
            raise RuntimeError(f"Streaming codec warmup delivered {len(callback_chunks)} callbacks, expected 1")
        warmup_chunk = callback_chunks[0]
        expected_samples = chunk_frames * decoder.samples_per_frame
        if warmup_chunk.samples.shape != (1, expected_samples):
            raise RuntimeError(
                "Streaming codec warmup produced invalid PCM shape: "
                f"{tuple(warmup_chunk.samples.shape)} != {(1, expected_samples)}"
            )

    def materialize_streaming_codec_for_inference(self) -> WeightNormMaterializationReceipt:
        """Fix all NanoCodec decoder weights after final CUDA placement.

        This permanently changes the runtime instance's state-dict schema and
        is therefore only valid after checkpoint restore and before graph
        capture or request serving.
        """

        if self.training:
            raise RuntimeError("materialize_streaming_codec_for_inference requires model.eval()")
        if self.device.type != "cuda":
            raise RuntimeError(
                "materialize_streaming_codec_for_inference requires final CUDA placement, " f"got {self.device}"
            )
        if self._streaming_codec_weight_norm_receipt is not None:
            raise RuntimeError("Streaming codec weight norm was already materialized")

        receipt = materialize_causal_hifigan_weight_norm_for_inference(
            self._codec_helper.codec_model.audio_decoder,
            expected_target_count=97,
        )
        if receipt.devices != (str(self.device),):
            raise RuntimeError(
                "Streaming codec weights are not on the model's final CUDA device: "
                f"{receipt.devices} != {(str(self.device),)}"
            )
        self._streaming_codec_weight_norm_receipt = receipt
        return receipt

    def create_streaming_codec_cuda_graph_runtime(self) -> CausalCodecStreamingCudaGraphRuntime:
        """Pre-capture one reusable, exclusively leased NanoCodec runtime."""

        receipt = self._streaming_codec_weight_norm_receipt
        if receipt is None:
            raise RuntimeError("Call materialize_streaming_codec_for_inference before CUDA Graph capture")
        audio_decoder = self._codec_helper.codec_model.audio_decoder
        current_devices = tuple(
            sorted({str(audio_decoder.get_submodule(name).weight.device) for name in receipt.target_names})
        )
        if current_devices != receipt.devices:
            raise RuntimeError(
                "Streaming codec moved after weight-norm materialization: " f"{current_devices} != {receipt.devices}"
            )
        decoder = CausalCodecStreamingDecoder(
            codec_model=self._codec_helper.codec_model,
            codec_converter=self._codec_helper.codec_converter,
        )
        return decoder.create_cuda_graph_runtime()

    def create_chunk_state(self, batch_size: int) -> ChunkState:
        """Create fresh state for chunked inference over a batch.

        This method creates a ChunkState dataclass instance that tracks
        mutable state across multiple calls to generate_speech() when
        processing text in one or more chunks.

        The returned state object should be:
        1. Created once per batch by the inference runner
        2. Passed to each call of generate_speech()
        3. Updated in-place during generation

        Args:
            batch_size: Number of items in the batch.

        Returns:
            ChunkState with initialized state for the batch.

        Example:
            >>> chunk_state = model.create_chunk_state(batch_size=4)
            >>> for chunk in text_chunks:
            ...     output = model.generate_speech(batch, chunk_state, ...)
        """
        return ChunkState(batch_size=batch_size)

    def _set_attention_prior_weights(
        self,
        attn_prior: torch.Tensor,
        batch_idx: int,
        attended_pos: int,
        text_len: int,
        eps_sq: float,
    ) -> None:
        """
        Set attention prior weights around the currently attended position.

        Creates a distribution that:
        - Strongly suppresses positions before (attended - 1)
        - Peaks at the current attended position
        - Gradually decays for lookahead positions
        - Suppresses far-future positions

        Args:
            attn_prior: Prior tensor to modify in-place. Shape: (B, 1, T_text).
            batch_idx: Index of current batch item.
            attended_pos: Currently attended text position (chunk-relative).
            text_len: Length of text for this batch item.
            eps_sq: Squared epsilon for strong suppression.
        """
        prior_weights = self.inference_parameters.prior_weights

        # Suppress history (before attended - 1)
        history_end = max(1, attended_pos - 1)
        attn_prior[batch_idx, 0, :history_end] = eps_sq

        # Set weights around attended position
        attn_prior[batch_idx, 0, history_end] = prior_weights[0]  # History exposure
        attn_prior[batch_idx, 0, attended_pos] = prior_weights[1]  # Current (peak)

        # Lookahead positions with bounds checking
        for offset, weight in enumerate(prior_weights[2:], start=1):
            pos = attended_pos + offset
            if pos < text_len:
                attn_prior[batch_idx, 0, pos] = weight

        # Suppress far future (position +5 onwards)
        future_start = attended_pos + len(prior_weights) - 1
        if future_start < text_len:
            attn_prior[batch_idx, 0, future_start:] = eps_sq

    def _penalize_attention_sinks(
        self,
        attn_prior: torch.Tensor,
        batch_idx: int,
        attended_timestep_counter: Dict[int, int],
        left_offset: int,
        eps_sq: float,
    ) -> None:
        """
        Penalize timesteps that have been over-attended (attention sinks).

        When a position is attended more than the threshold, suppress all
        positions up to and including it to force the model to move forward.

        Args:
            attn_prior: Prior tensor to modify in-place. Shape: (B, 1, T_text).
            batch_idx: Index of current batch item.
            attended_timestep_counter: Dict tracking attention counts per timestep.
            left_offset: Chunk offset for this batch item.
            eps_sq: Squared epsilon for strong suppression.
        """
        threshold = self.inference_parameters.chunked_attention_sink_threshold

        for timestep, count in attended_timestep_counter.items():
            if timestep > left_offset and count >= threshold:
                logging.debug(f"Attention sink at timestep {timestep} for batch {batch_idx}, count: {count}")
                relative_pos = timestep - left_offset
                attn_prior[batch_idx, 0, : relative_pos + 1] = eps_sq

    def _update_text_completion_state(
        self,
        batch_idx: int,
        attended_pos: int,
        text_len: int,
        is_finished: bool,
        unfinished_texts: Dict[int, bool],
        finished_texts_counter: Dict[int, int],
    ) -> None:
        """
        Update tracking state for text completion detection.

        A text is considered "near end" when the attended position is within
        ``near_end_threshold`` positions of the text end.

        Args:
            batch_idx: Index of current batch item.
            attended_pos: Currently attended text position (chunk-relative).
            text_len: Length of text for this batch item.
            is_finished: Whether this batch item has already finished.
            unfinished_texts: Dict to update in-place.
            finished_texts_counter: Dict to update in-place.
        """
        is_near_end = attended_pos >= text_len - self.inference_parameters.near_end_threshold

        # Text is unfinished if not near end AND not already marked finished
        unfinished_texts[batch_idx] = not is_near_end and not is_finished

        # Start counting when near end or already finished
        if is_near_end or is_finished:
            finished_texts_counter.setdefault(batch_idx, 0)

    def construct_multi_chunk_prior(
        self,
        prior_epsilon: float,
        cross_attention_scores: torch.Tensor,
        text_lens: torch.Tensor,
        text_time_step_attended: List[int],
        attended_timestep_counter: List[Dict[int, int]],
        unfinished_texts: Dict[int, bool],
        finished_texts_counter: Dict[int, int],
        end_indices: Dict[int, int],
        chunk_end_dict: Dict[int, int],
        batch_size: int,
        left_offset: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, bool], Dict[int, int]]:
        """
        Construct attention prior for multi-chunk inference with chunked text.

        Builds a soft attention prior that guides the decoder to attend to appropriate
        text positions, preventing attention drift and encouraging monotonic progression.

        Args:
            prior_epsilon: Base probability for non-targeted positions.
            cross_attention_scores: Attention scores for shape/device inference.
                Shape: (effective_batch, text_length).
            text_lens: Length of text for each batch item. Shape: (batch_size,).
            text_time_step_attended: Most attended text position (absolute) per batch item.
            attended_timestep_counter: Per-batch dicts tracking attention counts per timestep.
            unfinished_texts: Updated in-place. True if text still being processed.
            finished_texts_counter: Updated in-place. Counts consecutive near-end timesteps.
            end_indices: Batch indices that have reached end-of-sequence.
            chunk_end_dict: Batch indices that have reached chunk end.
            batch_size: Number of items in the batch.
            left_offset: Chunk offset for each batch item. Defaults to zeros.

        Returns:
            Tuple of (attention_prior, unfinished_texts, finished_texts_counter).
        """
        # Initialize with safe default (avoid mutable default argument)
        if left_offset is None:
            left_offset = [0] * batch_size

        # Extract shape info and create prior tensor
        device = cross_attention_scores.device
        effective_batch = cross_attention_scores.shape[0]  # 2 * batch_size if CFG else batch_size
        text_dim = cross_attention_scores.shape[1]
        eps_sq = prior_epsilon * prior_epsilon

        attn_prior = torch.full(
            (effective_batch, 1, text_dim),
            prior_epsilon,
            device=device,
            dtype=cross_attention_scores.dtype,
        )

        # Process each batch item
        for bidx in range(min(effective_batch, batch_size)):
            text_len = int(text_lens[bidx])
            attended_pos = text_time_step_attended[bidx] - left_offset[bidx]
            is_finished = bidx in end_indices or bidx in chunk_end_dict

            # Short sentences: uniform prior (no guidance needed)
            if text_len <= self.inference_parameters.short_sentence_threshold:
                attn_prior[bidx, 0, :] = 1.0
            else:
                # Set attention weights around attended position
                self._set_attention_prior_weights(attn_prior, bidx, attended_pos, text_len, eps_sq)

            # Penalize attention sinks (stuck positions)
            if not is_finished:
                self._penalize_attention_sinks(
                    attn_prior,
                    bidx,
                    attended_timestep_counter[bidx],
                    left_offset[bidx],
                    eps_sq,
                )

            # Update text completion tracking
            self._update_text_completion_state(
                bidx,
                attended_pos,
                text_len,
                is_finished,
                unfinished_texts,
                finished_texts_counter,
            )

        return attn_prior, unfinished_texts, finished_texts_counter

    def _uses_packed_alignment_eos_boundary(
        self,
        *,
        beginning_of_text: bool,
        chunk_state: ChunkState,
    ) -> bool:
        """Select the explicit one-boundary algorithm used by the Sofia configuration."""
        return (
            beginning_of_text
            and self._should_update_attention_prior(decoder_step=0)
            and self.inference_parameters.ignore_finished_sentence_tracking
            and self.model_type != "multi_encoder_context_tts"
            and not any(chunk_state.left_offset)
            and not chunk_state.previous_attn_len
        )

    def _select_packed_streaming_submission_profile(
        self,
        *,
        packed_alignment_eos: bool,
        batch_size: int,
        end_of_text: List[bool],
        use_cfg: bool,
        use_local_transformer_for_inference: bool,
        streaming_session: Optional[AsyncCodecSynthesisSession],
        streaming_emission_schedule: StreamingCodecEmissionSchedule,
        chunk_state: ChunkState,
    ) -> Optional[PackedStreamingSubmissionProfile]:
        """Select an explicitly supported Sofia packed submission profile.

        The grouped path is deliberately restricted to the batch-one, CFG,
        local-AR, single-final-chunk configuration for which all decoder, RNG,
        and alignment state is session-local and can be discarded after a
        terminal status. Unsupported schedules use the general path and never
        trigger request-time graph capture.
        """

        eligible = (
            packed_alignment_eos
            and batch_size == 1
            and end_of_text == [True]
            and use_cfg
            and use_local_transformer_for_inference
            and self.local_transformer_type == LocalTransformerType.AR
            and streaming_session is not None
            and chunk_state.streaming_codec_emission_state.accepted_chunk_count == 0
            and streaming_emission_schedule.prioritize_first_pcm
            and self.frame_stacking_factor == 2
            and self.inference_parameters.min_generated_frames == 4
        )
        if not eligible:
            return None

        frame_profile = (
            streaming_emission_schedule.first_chunk_frames,
            streaming_emission_schedule.steady_chunk_frames,
        )
        if frame_profile == (FIRST_4_STEADY_8.first_chunk_frames, FIRST_4_STEADY_8.steady_chunk_frames):
            return FIRST_4_STEADY_8
        if frame_profile == (FIRST_8_STEADY_8.first_chunk_frames, FIRST_8_STEADY_8.steady_chunk_frames):
            return FIRST_8_STEADY_8
        return None

    def _should_update_attention_prior(self, *, decoder_step: int) -> bool:
        """Return whether this step may advance alignment and construct the next prior.

        The prior constructed from a step's cross-attention is consumed by the
        following decoder step. Before the configured start step, alignment and
        sentence-tracking state must remain untouched.
        """
        if decoder_step < 0:
            raise ValueError(f"decoder_step must be non-negative, got {decoder_step}")
        return (
            self.inference_parameters.apply_attention_prior
            and decoder_step >= self.inference_parameters.start_prior_after_n_audio_steps
        )

    def _create_first_chunk_alignment_scratch(
        self,
        *,
        text_lens: torch.Tensor,
        last_attended_timesteps: List[List[int]],
        effective_batch_size: int,
        text_length: int,
        dtype: torch.dtype,
        status_capacity: int,
    ) -> FirstChunkAlignmentScratch:
        """Allocate fixed GPU state for first-chunk alignment tracking."""
        batch_size = text_lens.size(0)
        if not last_attended_timesteps or len(last_attended_timesteps[-1]) != batch_size:
            raise ValueError("Packed alignment requires one last-attended position per batch item")
        if effective_batch_size not in (batch_size, batch_size * 2):
            raise ValueError(
                f"Packed alignment effective batch must be B or 2B, got B={batch_size}, "
                f"effective={effective_batch_size}"
            )
        if status_capacity < 1:
            raise ValueError(f"Packed alignment status capacity must be positive, got {status_capacity}")
        device = text_lens.device
        positions = (
            torch.arange(text_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).clone()
        )
        prior_index_count = self.inference_parameters.attention_prior_lookahead_window + 2
        return FirstChunkAlignmentScratch(
            text_lens=text_lens,
            last_attended=torch.tensor(
                last_attended_timesteps[-1],
                device=device,
                dtype=torch.long,
            ),
            counters=torch.zeros((batch_size, text_length), device=device, dtype=torch.long),
            positions=positions,
            valid_window=torch.empty((batch_size, text_length), device=device, dtype=torch.bool),
            auxiliary_mask=torch.empty((batch_size, text_length), device=device, dtype=torch.bool),
            masked_scores=torch.empty((batch_size, text_length), device=device, dtype=dtype),
            search_start=torch.empty(batch_size, device=device, dtype=torch.long),
            window_end=torch.empty(batch_size, device=device, dtype=torch.long),
            attended=torch.empty(batch_size, device=device, dtype=torch.long),
            ended_attended=torch.empty(batch_size, device=device, dtype=torch.long),
            has_valid_window=torch.empty(batch_size, device=device, dtype=torch.bool),
            counter_increment=torch.ones((batch_size, 1), device=device, dtype=torch.long),
            prior=torch.empty(
                (effective_batch_size, 1, text_length),
                device=device,
                dtype=dtype,
            ),
            prior_indices=torch.empty(
                (batch_size, prior_index_count),
                device=device,
                dtype=torch.long,
            ),
            sink_candidates=torch.empty((batch_size, text_length), device=device, dtype=torch.long),
            max_sink_position=torch.empty(batch_size, device=device, dtype=torch.long),
            host_status=torch.empty(
                (status_capacity, batch_size, 2),
                device=device,
                dtype=torch.long,
            ),
        )

    def _compute_first_chunk_alignment_prior(
        self,
        alignment_attention_scores: torch.Tensor,
        scratch: FirstChunkAlignmentScratch,
    ) -> torch.Tensor:
        """Update attended positions and next-step prior without a host synchronization."""
        batch_size, text_length = scratch.counters.shape
        if (
            alignment_attention_scores.ndim != 2
            or alignment_attention_scores.size(0) < batch_size
            or alignment_attention_scores.size(1) != text_length
            or alignment_attention_scores.device != scratch.masked_scores.device
            or alignment_attention_scores.dtype != scratch.masked_scores.dtype
        ):
            raise ValueError(
                "Packed alignment scores do not match the session scratch: "
                f"scores={tuple(alignment_attention_scores.shape)}/"
                f"{alignment_attention_scores.device}/{alignment_attention_scores.dtype}, "
                f"scratch={(batch_size, text_length)}/"
                f"{scratch.masked_scores.device}/{scratch.masked_scores.dtype}"
            )

        last_positions = scratch.last_attended.clamp(min=0, max=text_length - 1)
        last_counts = scratch.counters.gather(1, last_positions.unsqueeze(1)).squeeze(1)
        scratch.search_start.copy_(scratch.last_attended)
        scratch.search_start.add_(
            last_counts.ge(self.inference_parameters.attention_sink_threshold).to(dtype=torch.long)
        )
        scratch.search_start.clamp_(min=0, max=text_length)
        torch.minimum(
            scratch.search_start + self.inference_parameters.attention_prior_lookahead_window,
            scratch.text_lens - 3,
            out=scratch.window_end,
        )

        torch.ge(
            scratch.positions,
            scratch.search_start.unsqueeze(1),
            out=scratch.valid_window,
        )
        torch.lt(
            scratch.positions,
            scratch.window_end.unsqueeze(1),
            out=scratch.auxiliary_mask,
        )
        torch.logical_and(
            scratch.valid_window,
            scratch.auxiliary_mask,
            out=scratch.valid_window,
        )
        scratch.masked_scores.copy_(alignment_attention_scores[:batch_size])
        torch.logical_not(scratch.valid_window, out=scratch.auxiliary_mask)
        scratch.masked_scores.masked_fill_(scratch.auxiliary_mask, float("-inf"))
        torch.argmax(scratch.masked_scores, dim=1, out=scratch.attended)

        torch.gt(
            scratch.window_end,
            scratch.search_start,
            out=scratch.has_valid_window,
        )
        scratch.ended_attended.copy_(scratch.text_lens).sub_(1)
        torch.where(
            scratch.has_valid_window,
            scratch.attended,
            scratch.ended_attended,
            out=scratch.search_start,
        )
        scratch.attended.copy_(scratch.search_start)
        scratch.last_attended.copy_(scratch.attended)
        scratch.counters.scatter_add_(
            1,
            scratch.attended.unsqueeze(1),
            scratch.counter_increment,
        )

        prior_epsilon = self.inference_parameters.attention_prior_epsilon
        scratch.prior.fill_(prior_epsilon)
        conditional_prior = scratch.prior[:batch_size, 0]
        history_index_floor = min(1, text_length - 1)
        scratch.prior_indices[:, 0].copy_(scratch.attended).sub_(1).clamp_(
            min=history_index_floor,
            max=text_length - 1,
        )
        scratch.prior_indices[:, 1].copy_(scratch.attended)
        for offset in range(1, self.inference_parameters.attention_prior_lookahead_window + 1):
            scratch.prior_indices[:, offset + 1].copy_(scratch.attended).add_(offset)
            torch.minimum(
                scratch.prior_indices[:, offset + 1],
                scratch.text_lens - 1,
                out=scratch.prior_indices[:, offset + 1],
            )
        conditional_prior.scatter_(1, scratch.prior_indices, 1.0)
        torch.le(
            scratch.text_lens,
            5,
            out=scratch.has_valid_window,
        )
        conditional_prior.masked_fill_(scratch.has_valid_window.unsqueeze(1), 1.0)

        torch.ge(
            scratch.counters,
            self.inference_parameters.attention_sink_threshold,
            out=scratch.valid_window,
        )
        scratch.sink_candidates.copy_(scratch.positions)
        torch.logical_not(scratch.valid_window, out=scratch.auxiliary_mask)
        scratch.sink_candidates.masked_fill_(scratch.auxiliary_mask, -1)
        torch.amax(
            scratch.sink_candidates,
            dim=1,
            out=scratch.max_sink_position,
        )
        torch.le(
            scratch.positions,
            scratch.max_sink_position.unsqueeze(1),
            out=scratch.valid_window,
        )
        conditional_prior.masked_fill_(scratch.valid_window, prior_epsilon)
        return scratch.prior

    def _update_first_chunk_alignment_host_state(
        self,
        *,
        attended_timesteps: List[int],
        state: ChunkedDecoderState,
        chunk_state: ChunkState,
    ) -> None:
        """Mirror the established Python tracking state after the packed host transfer."""
        if len(attended_timesteps) != len(state.text_lens_host):
            raise ValueError(
                f"Packed alignment returned {len(attended_timesteps)} items for " f"batch {len(state.text_lens_host)}"
            )
        for batch_index, attended_timestep in enumerate(attended_timesteps):
            counter = state.attended_timestep_counter[batch_index]
            counter[attended_timestep] = counter.get(attended_timestep, 0) + 1

            text_len = state.text_lens_host[batch_index]
            state.unfinished_texts[batch_index] = False
            if attended_timestep < text_len - 3 and batch_index not in chunk_state.end_indices:
                state.unfinished_texts[batch_index] = True
            if attended_timestep >= text_len - 2 or batch_index in chunk_state.end_indices:
                state.finished_texts_counter.setdefault(batch_index, 0)

        for batch_index in state.finished_texts_counter:
            state.finished_texts_counter[batch_index] += 1
            if state.finished_texts_counter[batch_index] > 5:
                state.unfinished_texts[batch_index] = False
        chunk_state.last_attended_timesteps.append(attended_timesteps)

    @staticmethod
    def _record_packed_alignment_eos_status(
        scratch: FirstChunkAlignmentScratch,
        *,
        pending_index: int,
        end_frame_indices: torch.Tensor,
    ) -> None:
        """Append one ordered device-resident alignment/EOS status row."""
        if pending_index < 0 or pending_index >= scratch.host_status.size(0):
            raise RuntimeError(
                "Packed alignment status buffer is full: "
                f"index={pending_index}, capacity={scratch.host_status.size(0)}"
            )
        if end_frame_indices.shape != scratch.attended.shape:
            raise ValueError(
                "Packed EOS status does not match alignment batch shape: "
                f"{tuple(end_frame_indices.shape)} != {tuple(scratch.attended.shape)}"
            )
        if end_frame_indices.device != scratch.host_status.device or end_frame_indices.dtype != torch.long:
            raise ValueError(
                "Packed EOS status must be an int64 tensor on "
                f"{scratch.host_status.device}, got {end_frame_indices.device}/{end_frame_indices.dtype}"
            )
        scratch.host_status[pending_index, :, 0].copy_(scratch.attended)
        scratch.host_status[pending_index, :, 1].copy_(end_frame_indices)

    @staticmethod
    def _packed_status_requires_transfer(
        *,
        forbid_audio_eos: bool,
        final_text_chunk: bool,
        codec_submission_due: bool,
        decoder_limit_reached: bool,
        defer_until_codec_submission: bool = False,
    ) -> bool:
        """Return whether pending packed status must cross to the host now.

        The fixed streaming macro may defer legal EOS status until the next
        codec submission because terminal state is replayed in order and
        speculative suffix state is discarded. Outside that narrow mode, only
        a final text chunk while EOS is globally forbidden may defer the
        transfer. Non-final chunks retain their per-step forceful-end behavior.
        Every codec submission is preceded by a transfer so asynchronous
        local-AR assertions fail closed before any audio leaves the producer.
        """
        if defer_until_codec_submission:
            if not final_text_chunk:
                raise ValueError("Packed streaming submission macro requires a final text chunk")
            return codec_submission_due or decoder_limit_reached
        return not forbid_audio_eos or not final_text_chunk or codec_submission_due or decoder_limit_reached

    @staticmethod
    def _transfer_packed_alignment_eos_status(
        scratch: FirstChunkAlignmentScratch,
        *,
        pending_steps: int,
    ) -> List[List[List[int]]]:
        """Perform one D2H boundary for all ordered pending Sofia statuses."""
        if pending_steps < 1 or pending_steps > scratch.host_status.size(0):
            raise RuntimeError(
                "Packed alignment pending status count is outside its buffer: "
                f"pending={pending_steps}, capacity={scratch.host_status.size(0)}"
            )
        return scratch.host_status[:pending_steps].tolist()

    def _apply_packed_alignment_eos_status(
        self,
        *,
        packed_status: List[List[List[int]]],
        first_pending_step: int,
        state: ChunkedDecoderState,
        chunk_state: ChunkState,
        chunk_end_frame_lens: Dict[int, int],
        end_of_text: List[bool],
        batch_size: int,
    ) -> Optional[int]:
        """Replay one ordered packed transfer and return its terminal step.

        ``chunk_state.overall_idx`` is moved to the historical value for each
        row. If a row terminates the model chunk, later rows came from
        speculative generation and are intentionally not committed.
        """
        if not packed_status:
            raise ValueError("Packed alignment status transfer must contain at least one decoder step")
        if first_pending_step < 0:
            raise ValueError(f"first_pending_step must be non-negative, got {first_pending_step}")

        transfer_overall_idx = chunk_state.overall_idx
        last_pending_step = first_pending_step + len(packed_status) - 1
        for pending_offset, step_status in enumerate(packed_status):
            if len(step_status) != batch_size:
                raise ValueError(
                    "Packed alignment status batch does not match generation: " f"{len(step_status)} != {batch_size}"
                )
            current_step = first_pending_step + pending_offset
            chunk_state.overall_idx = transfer_overall_idx - (last_pending_step - current_step)
            if chunk_state.overall_idx < 0:
                raise RuntimeError(
                    "Packed status history predates the generation session: "
                    f"overall={transfer_overall_idx}, pending_step={current_step}, "
                    f"last_pending_step={last_pending_step}"
                )
            attended_timesteps = [item[0] for item in step_status]
            end_frame_indices = [item[1] for item in step_status]
            self._update_first_chunk_alignment_host_state(
                attended_timesteps=attended_timesteps,
                state=state,
                chunk_state=chunk_state,
            )
            self._update_eos_state(
                chunk_state,
                end_frame_indices,
                state.chunk_end_dict,
                chunk_end_frame_lens,
                state.finished_texts_counter,
                end_of_text,
                current_step,
                batch_size,
            )
            if MagpieTTSModel._should_terminate_loop(
                self,
                chunk_state,
                state.chunk_end_dict,
                end_of_text,
                batch_size,
            ):
                return current_step
        chunk_state.overall_idx = transfer_overall_idx
        return None

    @staticmethod
    def _commit_decoder_prediction_step(
        state: ChunkedDecoderState,
        audio_codes_next: torch.Tensor,
        *,
        current_step: int,
        frame_stacking_factor: int,
        terminal_step: Optional[int],
    ) -> bool:
        """Commit the current prediction or trim an uncommitted speculative suffix."""
        if current_step < 0:
            raise ValueError(f"current_step must be non-negative, got {current_step}")
        if frame_stacking_factor < 1:
            raise ValueError(f"frame_stacking_factor must be positive, got {frame_stacking_factor}")
        if state.num_prediction_steps != current_step:
            raise RuntimeError(
                "Prediction step is not contiguous: " f"buffered={state.num_prediction_steps}, current={current_step}"
            )
        if terminal_step is not None:
            if terminal_step < 0 or terminal_step > current_step:
                raise RuntimeError(f"Terminal step {terminal_step} is outside generated range [0, {current_step}]")
            if terminal_step < current_step:
                state.num_prediction_steps = terminal_step + 1
                return False

        frame_start = state.num_prediction_steps * frame_stacking_factor
        frame_end = frame_start + frame_stacking_factor
        if frame_end > state.prediction_buffer.size(-1):
            raise RuntimeError(
                "Prediction buffer is full before the decoder limit: "
                f"required={frame_end}, capacity={state.prediction_buffer.size(-1)}"
            )
        expected_codes_shape = (
            state.prediction_buffer.size(0),
            state.prediction_buffer.size(1),
            frame_stacking_factor,
        )
        if tuple(audio_codes_next.shape) != expected_codes_shape:
            raise ValueError(
                f"Prediction codes must have shape {expected_codes_shape}, got {tuple(audio_codes_next.shape)}"
            )
        state.prediction_buffer[:, :, frame_start:frame_end].copy_(audio_codes_next)
        state.num_prediction_steps += 1
        return True

    @staticmethod
    def _transfer_eos_status(end_frame_indices: torch.Tensor) -> List[int]:
        """Transfer EOS results for tracking-enabled or multi-chunk algorithms."""
        return end_frame_indices.tolist()

    @staticmethod
    def _to_int(value: Union[int, torch.Tensor]) -> int:
        """Convert tensor scalar to Python int if needed."""
        return value.item() if not isinstance(value, int) else value

    def _update_eos_state(
        self,
        chunk_state: ChunkState,
        end_frame_indices: List[int],
        chunk_end_dict: Dict[int, int],
        chunk_end_frame_lens: Dict[int, int],
        finished_texts_counter: Dict[int, int],
        end_of_text: List[bool],
        current_step: int,
        batch_size: int,
    ) -> None:
        """Apply host-visible EOS positions to chunk/end tracking state.

        Args:
            chunk_state: Mutable state object tracking history across chunks.
            end_frame_indices: First EOS frame per item, or ``frame_stacking_factor`` when absent.
            chunk_end_dict: Maps batch indices to chunk end timesteps.
            chunk_end_frame_lens: Maps batch indices to frame-level length (for codes_to_audio); aligned with infer().
            finished_texts_counter: Counter for near-end timesteps.
            end_of_text: Whether text has ended for each batch item.
            current_step: Current decoding step index.
            batch_size: Number of items in the batch.
        """
        if len(end_frame_indices) != batch_size:
            raise ValueError(f"EOS result has {len(end_frame_indices)} items for batch size {batch_size}")
        for item_idx in range(batch_size):
            if item_idx in chunk_state.end_indices or item_idx in chunk_end_dict:
                continue

            end_frame_index = end_frame_indices[item_idx]

            # End of speech detected. Update the state.
            if end_frame_index < self.frame_stacking_factor:
                frame_len = current_step * self.frame_stacking_factor + end_frame_index
                chunk_end_frame_lens[item_idx] = frame_len
                if end_of_text[item_idx]:
                    # Speech for entire multi-chunk text has ended. Update the state.
                    chunk_state.end_indices[item_idx] = chunk_state.overall_idx
                    chunk_end_dict[item_idx] = current_step
                    logging.info(
                        f"End detected for item {item_idx} at local timestep {current_step} "
                        f"and overall timestep {chunk_state.overall_idx}"
                    )
                elif item_idx not in chunk_end_dict:
                    # Chunk end detected. Update the state.
                    chunk_end_dict[item_idx] = current_step
                    logging.info(f"Chunk end detected for item {item_idx} at local timestep {current_step}")
            elif (
                not end_of_text[item_idx]
                and finished_texts_counter.get(item_idx, -1) >= self.inference_parameters.forceful_chunk_end_threshold
            ):
                chunk_end_dict[item_idx] = current_step
                chunk_end_frame_lens[item_idx] = (current_step + 1) * self.frame_stacking_factor
                logging.info(f"Forceful chunk end detected for item {item_idx} at local timestep {current_step}")

    def _should_terminate_loop(
        self,
        chunk_state: ChunkState,
        chunk_end_dict: Dict[int, int],
        end_of_text: List[bool],
        batch_size: int,
    ) -> bool:
        """
        Check if all batch items have reached their end condition.

        Args:
            chunk_state: Mutable state object tracking history across chunks.
            chunk_end_dict: Maps batch indices to chunk end timesteps.
            end_of_text: Whether text has ended for each batch item.
            batch_size: Number of items in the batch.

        Returns:
            True if all items have reached end, False otherwise.
        """
        if len(chunk_state.end_indices) == batch_size:
            logging.info("All ends reached")
            return True

        completed_count = 0
        for bidx in range(batch_size):
            if not end_of_text[bidx] and bidx in chunk_end_dict:
                completed_count += 1
            elif end_of_text[bidx] and bidx in chunk_state.end_indices:
                completed_count += 1

        if completed_count == batch_size:
            logging.info("All ends reached via chunk end")
            return True

        return False

    def _resolve_incremental_alignment_layer_indices(self) -> Optional[Tuple[int, ...]]:
        """Resolve the exact cross-attention layers retained by incremental inference."""
        if not self.inference_parameters.apply_attention_prior:
            return None

        configured_layers = self.inference_parameters.estimate_alignment_from_layers
        alignment_layer_indices = (
            tuple(self.transcript_decoder_layers) if configured_layers is None else tuple(configured_layers)
        )
        if not alignment_layer_indices:
            raise ValueError("Attention-prior inference requires at least one alignment layer")
        if len(set(alignment_layer_indices)) != len(alignment_layer_indices):
            raise ValueError(f"Alignment layer configuration contains duplicates: {alignment_layer_indices}")
        for layer_index in alignment_layer_indices:
            if type(layer_index) is not int:
                raise TypeError(f"Alignment layer index must be an int, got {type(layer_index).__name__}")
            if layer_index < 0 or layer_index >= self.decoder.n_layers:
                raise ValueError(
                    f"Alignment layer index {layer_index} is outside the decoder layer range "
                    f"[0, {self.decoder.n_layers})"
                )
            if layer_index not in self.transcript_decoder_layers:
                raise ValueError(f"Alignment layer {layer_index} is not a transcript cross-attention layer")
        return alignment_layer_indices

    def _create_incremental_decoder_session(
        self,
        context_tensors: ContextTensorsOutput,
        use_cfg: bool,
        cfg_scale: float,
        dummy_cond: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        dummy_cond_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        dummy_additional_decoder_input: Optional[torch.Tensor],
        dummy_addition_dec_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> IncrementalDecoderSession:
        """Create an isolated main-decoder cache and stable CFG inputs."""
        alignment_layer_indices = self._resolve_incremental_alignment_layer_indices()
        if use_cfg:
            if isinstance(context_tensors.cond, list):
                if not isinstance(dummy_cond, list) or not isinstance(dummy_cond_mask, list):
                    raise TypeError("List conditioning requires list dummy conditioning for CFG")
                cond = [
                    torch.cat([conditional, unconditional], dim=0)
                    for conditional, unconditional in zip(context_tensors.cond, dummy_cond)
                ]
                cond_mask = [
                    torch.cat([conditional, unconditional], dim=0)
                    for conditional, unconditional in zip(context_tensors.cond_mask, dummy_cond_mask)
                ]
            else:
                if not isinstance(dummy_cond, torch.Tensor) or not isinstance(dummy_cond_mask, torch.Tensor):
                    raise TypeError("Tensor conditioning requires tensor dummy conditioning for CFG")
                cond = torch.cat([context_tensors.cond, dummy_cond], dim=0)
                cond_mask = torch.cat([context_tensors.cond_mask, dummy_cond_mask], dim=0)

            if context_tensors.additional_decoder_input is not None:
                if dummy_additional_decoder_input is None or dummy_addition_dec_mask is None:
                    raise RuntimeError("CFG additional decoder input requires an unconditional counterpart")
                additional_decoder_input = torch.cat(
                    [
                        context_tensors.additional_decoder_input,
                        dummy_additional_decoder_input,
                    ],
                    dim=0,
                )
                additional_decoder_mask = torch.cat(
                    [context_tensors.additional_decoder_mask, dummy_addition_dec_mask],
                    dim=0,
                )
            else:
                additional_decoder_input = None
                additional_decoder_mask = None
            effective_batch_size = batch_size * 2
        else:
            cond = context_tensors.cond
            cond_mask = context_tensors.cond_mask
            additional_decoder_input = context_tensors.additional_decoder_input
            additional_decoder_mask = context_tensors.additional_decoder_mask
            effective_batch_size = batch_size

        prefix_length = additional_decoder_input.size(1) if additional_decoder_input is not None else 0
        max_generated_steps = self.inference_parameters.max_decoder_steps // self.frame_stacking_factor
        max_length = prefix_length + max_generated_steps
        if (
            self.decoder.position_embeddings is not None
            and max_length > self.decoder.position_embeddings.num_embeddings
        ):
            raise ValueError(
                f"Incremental decoder requires {max_length} positions, but only "
                f"{self.decoder.position_embeddings.num_embeddings} are available"
            )
        transformer_state = self.decoder.create_incremental_state(
            batch_size=effective_batch_size,
            max_length=max_length,
            device=device,
            dtype=dtype,
        )
        cfg_decoder_step = None
        cfg_decoder_mask_step = None
        if use_cfg:
            cfg_decoder_step = torch.empty(
                (effective_batch_size, 1, self.cfg.decoder.d_model),
                device=device,
                dtype=dtype,
            )
            cfg_decoder_mask_step = torch.empty(
                (effective_batch_size, 1),
                device=device,
                dtype=torch.bool,
            )
        return IncrementalDecoderSession(
            transformer_state=transformer_state,
            cond=cond,
            cond_mask=cond_mask,
            additional_decoder_input=additional_decoder_input,
            additional_decoder_mask=additional_decoder_mask,
            multi_encoder_mapping=context_tensors.multi_encoder_mapping,
            conditional_batch_size=batch_size,
            cfg_scale=cfg_scale,
            use_cfg=use_cfg,
            cfg_decoder_step=cfg_decoder_step,
            cfg_decoder_mask_step=cfg_decoder_mask_step,
            eos_scratch=self._create_eos_detection_scratch(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            ),
            alignment_layer_indices=alignment_layer_indices,
        )

    def _embed_audio_step(self, audio_codes_step: torch.Tensor) -> torch.Tensor:
        """Embed exactly one fixed frame stack without length-derived masks."""
        expected_shape = (
            audio_codes_step.size(0),
            self.num_audio_codebooks,
            self.frame_stacking_factor,
        )
        if tuple(audio_codes_step.shape) != expected_shape:
            raise ValueError(
                f"Incremental audio step must have shape {expected_shape}, got {tuple(audio_codes_step.shape)}"
            )
        if audio_codes_step.dtype != torch.long:
            raise ValueError(f"Incremental audio step must use torch.long codes, got {audio_codes_step.dtype}")

        audio_embedding = None
        for frame_index in range(self.frame_stacking_factor):
            for codebook_index in range(self.num_audio_codebooks):
                tokens = audio_codes_step[:, codebook_index, frame_index : frame_index + 1]
                embedding = self.audio_embeddings[codebook_index + frame_index * self.num_audio_codebooks](tokens)
                if audio_embedding is None:
                    audio_embedding = embedding
                else:
                    audio_embedding.add_(embedding)
        if audio_embedding is None:
            raise AssertionError("At least one audio codebook and frame are required")
        return audio_embedding / (self.num_audio_codebooks * self.frame_stacking_factor)

    def _run_chunked_forward_with_cfg(
        self,
        session: IncrementalDecoderSession,
        audio_codes_embedded: torch.Tensor,
        audio_codes_mask: torch.Tensor,
        attn_prior: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        *,
        project_code_logits: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Advance the main decoder by one audio token using session-owned K/V.

        On the first call, fixed speaker/context prefix tokens are prefetched
        once. Later calls pass only the newly sampled audio token. A dynamic
        attention prior therefore applies to the current query only; it never
        rewrites cached past states. This is the causal incremental contract,
        and intentionally differs from recomputing the full prefix under every
        newly observed prior.
        """
        if audio_codes_embedded.size(1) != 1 or audio_codes_mask.size(1) != 1:
            raise ValueError(
                f"Main decoder incremental input must be one timestep, got "
                f"{tuple(audio_codes_embedded.shape)} and {tuple(audio_codes_mask.shape)}"
            )
        if session.use_cfg:
            if session.cfg_decoder_step is None or session.cfg_decoder_mask_step is None:
                raise RuntimeError("CFG decoder session is missing its fixed step buffers")
            if session.cfg_decoder_step.shape[2] != audio_codes_embedded.shape[2]:
                raise ValueError(
                    f"CFG decoder step width is {session.cfg_decoder_step.shape[2]}, "
                    f"received {audio_codes_embedded.shape[2]}"
                )
            conditional_batch_size = session.conditional_batch_size
            session.cfg_decoder_step[:conditional_batch_size].copy_(audio_codes_embedded)
            session.cfg_decoder_step[conditional_batch_size:].copy_(audio_codes_embedded)
            session.cfg_decoder_mask_step[:conditional_batch_size].copy_(audio_codes_mask)
            session.cfg_decoder_mask_step[conditional_batch_size:].copy_(audio_codes_mask)
            decoder_input = session.cfg_decoder_step
            decoder_mask = session.cfg_decoder_mask_step
        else:
            decoder_input = audio_codes_embedded
            decoder_mask = audio_codes_mask

        if session.next_position == 0 and session.additional_decoder_input is not None:
            decoder_input = torch.cat([session.additional_decoder_input, decoder_input], dim=1)
            decoder_mask = torch.cat([session.additional_decoder_mask, decoder_mask], dim=1)

        if session.next_position == 0:
            decoder_output = self.decoder.prefill_incremental_state(
                x_prefix=decoder_input,
                x_mask=decoder_mask,
                state=session.transformer_state,
                cond=session.cond,
                cond_mask=session.cond_mask,
                attn_prior=attn_prior,
                multi_encoder_mapping=session.multi_encoder_mapping,
                alignment_layer_indices=session.alignment_layer_indices,
            )
            session.next_position = decoder_input.size(1)
        else:
            if decoder_input.size(1) != 1:
                raise RuntimeError(
                    f"Only the initial decoder call may contain a prefix, got {decoder_input.size(1)} timesteps"
                )
            decoder_output = self.decoder.forward_incremental(
                x_step=decoder_input,
                x_mask=decoder_mask,
                state=session.transformer_state,
                position_offset=session.next_position,
                cond=session.cond,
                cond_mask=session.cond_mask,
                attn_prior=attn_prior,
                multi_encoder_mapping=session.multi_encoder_mapping,
                alignment_layer_indices=session.alignment_layer_indices,
            )
            session.next_position += 1

        dec_out = decoder_output.output[:, -1:, :]
        alignment_scores = decoder_output.alignment_scores
        if session.alignment_layer_indices is not None and alignment_scores is None:
            raise RuntimeError("Incremental decoder did not return the requested alignment scores")
        if not project_code_logits:
            return None, alignment_scores, dec_out

        combined_logits = self.final_proj(dec_out)
        if session.use_cfg:
            conditional_logits = combined_logits[: session.conditional_batch_size]
            unconditional_logits = combined_logits[session.conditional_batch_size :]
            all_code_logits = (1 - session.cfg_scale) * unconditional_logits + session.cfg_scale * conditional_logits
        else:
            all_code_logits = combined_logits

        return all_code_logits, alignment_scores, dec_out

    def _initialize_chunked_attn_prior(
        self,
        chunk_state: ChunkState,
        current_chunk_len: torch.Tensor,
        batch_text_lens: torch.Tensor,
        max_text_len: int,
        batch_size: int,
        use_cfg: bool,
        prior_epsilon: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Initialize attention prior for chunked generation with left offset tracking.

        This method constructs the initial attention prior when continuing from
        previous chunks, accounting for the sliding window over text history.

        Args:
            chunk_state: Mutable state object tracking history across chunks.
            current_chunk_len: Length of the current text chunk for each batch item.
            batch_text_lens: Text lengths for each batch item.
            max_text_len: Maximum text length in the batch.
            batch_size: Number of items in the batch.
            use_cfg: Whether classifier-free guidance is being used.
            prior_epsilon: Base epsilon value for attention prior.
            device: Target device for tensors.
            dtype: Floating-point dtype used by the decoder context.

        Returns:
            Attention prior tensor or None if no history exists.
        """
        if len(chunk_state.previous_attn_len) == 0:
            return None

        # Initialize prior tensor
        cfg_multiplier = 2 if use_cfg else 1
        _attn_prior = torch.full(
            (batch_size * cfg_multiplier, 1, max_text_len),
            prior_epsilon,
            device=device,
            dtype=dtype,
        )

        for _idx in range(batch_size):
            # Calculate left offset for sliding window
            delta_in_len = self._to_int(current_chunk_len[_idx])
            len_to_delete = self._to_int(chunk_state.previous_attn_len[_idx] + delta_in_len - batch_text_lens[_idx])
            chunk_state.left_offset[_idx] = self._to_int(chunk_state.left_offset[_idx] + len_to_delete)

            # Skip if text has ended
            if _idx in chunk_state.end_indices and chunk_state.end_indices[_idx] is not None:
                continue

            # Set prior weights for new chunk
            current_starting_point = batch_text_lens[_idx] - current_chunk_len[_idx]
            prior_weights = self.inference_parameters.prior_weights_init
            _attn_prior[_idx, :, :current_starting_point] = prior_epsilon * prior_epsilon
            for offset, weight in enumerate(prior_weights):
                current_offset_idx = current_starting_point + offset
                if current_offset_idx < max_text_len:
                    _attn_prior[_idx, :, current_offset_idx] = weight

        return _attn_prior

    def _update_context_from_history(
        self,
        chunk_state: ChunkState,
        context_tensors: Dict[str, Any],
        current_chunk_len: torch.Tensor,
        max_text_len: int,
        beginning_of_text: bool,
        batch_text_lens: torch.Tensor,
        batch_size: int,
    ) -> None:
        """
        Update context tensors with cached history for chunked generation.

        This method splices historical context embeddings into the current context
        tensors to maintain continuity across text chunks.

        Args:
            chunk_state: Mutable state object tracking history across chunks.
            context_tensors: ContextTensorsOutput containing 'cond' tensor to update.
            current_chunk_len: Length of the current text chunk for each batch item.
            max_text_len: Maximum text length in the batch.
            beginning_of_text: Whether this is the first chunk.
            batch_text_lens: Text lengths for each batch item.
            batch_size: Number of items in the batch.
        """
        for _idx in range(batch_size):
            # Skip if text has ended
            if _idx in chunk_state.end_indices and chunk_state.end_indices[_idx] is not None:
                continue
            if not beginning_of_text:
                pad_len_idx = max_text_len - batch_text_lens[_idx]
                history_context_len = self._to_int(
                    context_tensors.cond[_idx].shape[0] - current_chunk_len[_idx] - pad_len_idx
                )
                context_tensors.cond[_idx, :history_context_len] = chunk_state.history_context_tensor[
                    _idx, -history_context_len - 1 : -1
                ]
        chunk_state.history_context_tensor = context_tensors.cond

    def _prepare_chunked_text_tensors(
        self,
        chunk_state: ChunkState,
        batch: Dict[str, torch.Tensor],
        current_chunk_len: torch.Tensor,
        beginning_of_text: bool,
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Prepare text tensors with history for chunked inference.

        This method handles the sliding window logic for text tokens, combining
        historical text with new chunks and applying window size constraints.

        Args:
            chunk_state: Mutable state object tracking history across chunks.
            batch: Input batch containing 'text' and 'text_lens'.
            current_chunk_len: Length of the current text chunk for each batch item.
            beginning_of_text: Whether this is the first chunk.
            device: Target device for tensors.

        Returns:
            Tuple of (modified batch, max_text_len).
        """
        batch_size = batch["text"].size(0)
        text_tensors = []

        for _idx in range(batch_size):
            # If text has ended, use minimal placeholder
            if _idx in chunk_state.end_indices and chunk_state.end_indices[_idx] is not None:
                batch['text_lens'][_idx] = torch.tensor(1).to(device).long()
                text_tensors.append(batch['text'][_idx])
                continue

            # Combine history with current chunk
            if chunk_state.history_text is not None:
                history_text_len = self._to_int(chunk_state.history_text_lens[_idx]) - 1
                current_text = torch.cat(
                    [
                        chunk_state.history_text[_idx][:history_text_len],
                        batch["text"][_idx][: current_chunk_len[_idx]],
                    ]
                )
            else:
                current_text = batch["text"][_idx][: current_chunk_len[_idx]]

            # Apply sliding window
            history_len = min(current_chunk_len[_idx], self.inference_parameters.history_len_heuristic)
            true_window_size = current_chunk_len[_idx] + history_len
            if not beginning_of_text:
                current_text = current_text[max(0, current_text.shape[0] - true_window_size) :]

            current_text_lens = current_text.shape[0]
            text_tensors.append(current_text)
            batch['text_lens'][_idx] = torch.tensor(current_text_lens).to(device).long()

        # Pad and stack text tensors
        max_text_len = max(batch['text_lens']).item()
        batch['text'] = stack_tensors(text_tensors, max_lens=[max_text_len])

        # Update history
        chunk_state.history_text = batch['text']
        chunk_state.history_text_lens = batch['text_lens']

        return batch, max_text_len

    @staticmethod
    def _submit_streaming_codec_chunk(
        streaming_session: AsyncCodecSynthesisSession,
        prediction_buffer: torch.Tensor,
        *,
        emission_state: StreamingCodecEmissionState,
        chunk_first_codec_frame: int,
        emitted_frame_count: int,
        valid_frame_count: int,
        final: bool,
        wait_for_completion: bool,
    ) -> int:
        """Submit the newly valid suffix and return the new local frame offset."""

        if valid_frame_count < emitted_frame_count:
            raise RuntimeError(
                "EOS shortened codec frames that were already submitted: "
                f"{valid_frame_count} < {emitted_frame_count}"
            )
        new_frame_count = valid_frame_count - emitted_frame_count
        if new_frame_count == 0 and not final:
            return emitted_frame_count

        new_codes = prediction_buffer[:, :, emitted_frame_count:valid_frame_count]
        codec_chunk = StreamingCodecChunk(
            codes=new_codes,
            first_codec_frame=chunk_first_codec_frame + emitted_frame_count,
            codec_frame_count=new_frame_count,
            final=final,
        )
        producer_stream = (
            torch.cuda.current_stream(prediction_buffer.device) if prediction_buffer.device.type == "cuda" else None
        )
        sequence_index = streaming_session.submit_chunk(
            codec_chunk,
            producer_stream=producer_stream,
        )
        emission_state.record_accepted_chunk()
        if wait_for_completion:
            # This is the TTFA scheduling barrier selected by
            # StreamingCodecEmissionSchedule.prioritize_first_pcm. At the first
            # accepted submission this is sequence zero. Waiting for that exact
            # callback avoids broad queue-drain semantics and leaves all later
            # submissions asynchronous.
            streaming_session.wait_for_sequence(sequence_index)
        return valid_frame_count

    @staticmethod
    def _raise_decoder_limit_error(
        streaming_session: Optional[AsyncCodecSynthesisSession],
        *,
        generated_frame_count: int,
    ) -> None:
        """Fail closed when decoding exhausts its limit without a chunk end."""

        if streaming_session is not None:
            # No terminal chunk exists in this state. Cancel immediately so
            # queued codec work cannot continue producing callbacks and the
            # caller is not left with an unclosable live session.
            streaming_session.cancel(wait=True)
        raise RuntimeError(
            "Magpie generation reached the decoder limit without detecting "
            "the required EOS or text-chunk end: "
            f"generated_frame_count={generated_frame_count}"
        )

    def generate_speech(
        self,
        batch,
        chunk_state: ChunkState,
        end_of_text,
        beginning_of_text,
        local_ar_seed: Optional[int] = None,
        use_cfg=True,
        use_local_transformer_for_inference=False,
        maskgit_n_steps=3,
        maskgit_noise_scale=0.0,
        maskgit_fixed_schedule=None,
        maskgit_dynamic_cfg_scale=False,
        maskgit_sampling_type=None,
        streaming_session: Optional[AsyncCodecSynthesisSession] = None,
        streaming_first_codec_frame: Optional[int] = None,
        streaming_emission_schedule: StreamingCodecEmissionSchedule = StreamingCodecEmissionSchedule(),
    ):
        """Generate speech and invalidate an asynchronous codec stream on failure.

        A successful non-final call leaves the caller-owned codec session open
        for the next text chunk. Any exception makes that stream impossible to
        complete coherently, so accepted and queued codec work is cancelled
        before the original generation exception is propagated.

        Stochastic Local AR requires an explicit ``local_ar_seed`` in
        ``[0, 2**32)``. An absent seed fails closed; this path never infers one
        from process-global RNG state.
        """

        if local_ar_seed is not None and (
            local_ar_seed < 0 or local_ar_seed >= 2**32
        ):
            raise ValueError(
                f"local_ar_seed must be in [0, 2**32), got {local_ar_seed}"
            )
        stochastic_local_ar = (
            use_local_transformer_for_inference
            and self.local_transformer_type == LocalTransformerType.AR
            and self.inference_parameters.temperature > 0.0
            and self.inference_parameters.topk > 1
        )
        if stochastic_local_ar and local_ar_seed is None:
            raise ValueError(
                "stochastic Local AR requires an explicit local_ar_seed"
            )
        try:
            return self._generate_speech_impl(
                batch=batch,
                chunk_state=chunk_state,
                end_of_text=end_of_text,
                beginning_of_text=beginning_of_text,
                local_ar_seed=local_ar_seed,
                use_cfg=use_cfg,
                use_local_transformer_for_inference=use_local_transformer_for_inference,
                maskgit_n_steps=maskgit_n_steps,
                maskgit_noise_scale=maskgit_noise_scale,
                maskgit_fixed_schedule=maskgit_fixed_schedule,
                maskgit_dynamic_cfg_scale=maskgit_dynamic_cfg_scale,
                maskgit_sampling_type=maskgit_sampling_type,
                streaming_session=streaming_session,
                streaming_first_codec_frame=streaming_first_codec_frame,
                streaming_emission_schedule=streaming_emission_schedule,
            )
        except BaseException as generation_error:
            if streaming_session is not None:
                try:
                    streaming_session.cancel(wait=False, timeout=0.0)
                except BaseException as cancellation_error:
                    logging.error(
                        "Asynchronous codec cancellation also failed: "
                        f"{type(cancellation_error).__name__}: {cancellation_error}"
                    )
            raise
        finally:
            first_submission_lease = chunk_state.first_submission_graph_lease
            if first_submission_lease is not None:
                chunk_state.first_submission_graph_lease = None
                first_submission_lease.release()

    def _generate_speech_impl(
        self,
        batch,
        chunk_state: ChunkState,
        end_of_text,
        beginning_of_text,
        local_ar_seed: Optional[int] = None,
        use_cfg=True,
        use_local_transformer_for_inference=False,
        maskgit_n_steps=3,
        maskgit_noise_scale=0.0,
        maskgit_fixed_schedule=None,
        maskgit_dynamic_cfg_scale=False,
        maskgit_sampling_type=None,
        streaming_session: Optional[AsyncCodecSynthesisSession] = None,
        streaming_first_codec_frame: Optional[int] = None,
        streaming_emission_schedule: StreamingCodecEmissionSchedule = StreamingCodecEmissionSchedule(),
    ):
        """
        Unified speech generation supporting both single-chunk and multi-chunk modes.

        This method is the unified inference entry point. For short text (single chunk where
        beginning_of_text=True and end_of_text=[True]), it behaves similarly to standard inference.
        For long text (multiple chunks), it maintains a sliding window over text and audio histories,
        tracking how many audio tokens were generated for each text position.

        The behaviour is strongly dependent on self.inference_parameters.

        Args:
            batch (dict): Input batch containing 'text' and 'text_lens'.
            chunk_state (ChunkState): Mutable state object tracking history across chunks.
                Created via model.create_chunk_state() and updated in-place.
            end_of_text (List[bool]): Whether entire text has been provided for each batch item.
            beginning_of_text (bool): Whether this is the first chunk.
            use_cfg (bool): Whether to use classifier-free guidance.
            use_local_transformer_for_inference (bool): Whether to use local transformer for sampling.
            maskgit_n_steps (int): Number of MaskGit refinement steps.
            maskgit_noise_scale (float): Noise scale for MaskGit sampling.
            maskgit_fixed_schedule (Optional[List[int]]): Fixed schedule for MaskGit.
            maskgit_dynamic_cfg_scale (bool): Whether to use dynamic CFG scale in MaskGit.
            maskgit_sampling_type (Optional[str]): Type of MaskGit sampling.
            streaming_session: Optional asynchronous rolling-codec session. The
                model submits new codec frames directly; cumulative prefixes are
                not exposed.
            streaming_first_codec_frame: Absolute first frame for this text
                chunk. It must match ``streaming_session.next_codec_frame`` so
                frame numbering remains continuous across multiple text chunks.
            streaming_emission_schedule: Producer-side codec frame schedule.
                The default emits four frames for the first audible chunk and
                eight frames thereafter. A model chunk boundary always flushes
                its remaining frames because the local prediction buffer does
                not survive the call.

        Returns:
            InferBatchOutput: Contains predicted_codes, predicted_codes_lens, and empty audio fields.
        """
        generation_start_time = time.perf_counter()
        time_to_first_prediction = None
        cfg_scale = self.inference_parameters.cfg_scale
        if streaming_session is None and streaming_first_codec_frame is not None:
            raise ValueError("streaming_first_codec_frame requires streaming_session")
        if streaming_session is not None and streaming_first_codec_frame is None:
            raise ValueError("streaming_session requires an explicit streaming_first_codec_frame")
        if streaming_session is not None:
            streaming_emission_schedule.validate_frame_stacking_factor(self.frame_stacking_factor)

        eos_detection_method = EOSDetectionMethod(self.inference_parameters.eos_detection_method)
        device = batch["text"].device
        generation_stream = None
        generation_start_event = None
        first_prediction_event = None
        generation_end_event = None
        first_prediction_recorded = False
        if device.type == "cuda":
            generation_stream = torch.cuda.current_stream(device)
            generation_start_event = torch.cuda.Event(enable_timing=True)
            first_prediction_event = torch.cuda.Event(enable_timing=True)
            generation_end_event = torch.cuda.Event(enable_timing=True)
            generation_start_event.record(generation_stream)
        with torch.no_grad():
            current_chunk_len = copy.deepcopy(batch['text_lens'].detach())
            batch_size = batch["text"].size(0)
            streaming_chunk_first_frame = 0
            if streaming_session is not None:
                if batch_size != 1:
                    raise ValueError(
                        "Streaming Magpie generation currently has an explicit batch=1 contract, "
                        f"got batch={batch_size}"
                    )
                if len(end_of_text) != 1:
                    raise ValueError(
                        f"Streaming Magpie generation requires one end_of_text flag, got {len(end_of_text)}"
                    )
                if streaming_first_codec_frame != streaming_session.next_codec_frame:
                    raise ValueError(
                        "streaming_first_codec_frame does not match the codec session: "
                        f"{streaming_first_codec_frame} != {streaming_session.next_codec_frame}"
                    )
                streaming_chunk_first_frame = int(streaming_first_codec_frame)
            streaming_emitted_frames = 0

            # Prepare text tensors with history
            batch, max_text_len = self._prepare_chunked_text_tensors(
                chunk_state, batch, current_chunk_len, beginning_of_text, device
            )
            context_tensors = self.prepare_context_tensors(batch)

            # Update context with historical embeddings
            self._update_context_from_history(
                chunk_state,
                context_tensors,
                current_chunk_len,
                max_text_len,
                beginning_of_text,
                batch['text_lens'],
                batch_size,
            )

            audio_codes_step = (
                torch.full(
                    (batch_size, self.num_audio_codebooks, self.frame_stacking_factor),
                    self.audio_bos_id,
                )
                .long()
                .to(device)
            )
            audio_codes_mask = torch.ones((batch_size, 1), device=device, dtype=torch.bool)

            # Initialize dummy variables for CFG
            dummy_cond = None
            dummy_cond_mask = None
            dummy_additional_decoder_input = None
            dummy_addition_dec_mask = None
            if use_cfg:
                (
                    dummy_cond,
                    dummy_cond_mask,
                    dummy_additional_decoder_input,
                    dummy_addition_dec_mask,
                    _,
                ) = self.prepare_dummy_cond_for_cfg(
                    context_tensors.cond,
                    context_tensors.cond_mask,
                    context_tensors.additional_decoder_input,
                    context_tensors.additional_decoder_mask,
                )

            # Initialize attention prior for chunked generation
            initial_attn_prior = self._initialize_chunked_attn_prior(
                chunk_state,
                current_chunk_len,
                batch['text_lens'],
                max_text_len,
                batch_size,
                use_cfg,
                self.inference_parameters.attention_prior_epsilon,
                device,
                context_tensors.text_encoder_out.dtype,
            )
            text_lens_host = batch["text_lens"].detach().tolist()
            packed_alignment_eos = self._uses_packed_alignment_eos_boundary(
                beginning_of_text=beginning_of_text,
                chunk_state=chunk_state,
            )
            local_ar_writes_audio_step = (
                use_local_transformer_for_inference and self.local_transformer_type == LocalTransformerType.AR
            )
            packed_streaming_submission_profile = self._select_packed_streaming_submission_profile(
                packed_alignment_eos=packed_alignment_eos,
                batch_size=batch_size,
                end_of_text=end_of_text,
                use_cfg=use_cfg,
                use_local_transformer_for_inference=use_local_transformer_for_inference,
                streaming_session=streaming_session,
                streaming_emission_schedule=streaming_emission_schedule,
                chunk_state=chunk_state,
            )
            alignment_scratch = None
            if packed_alignment_eos:
                if initial_attn_prior is not None:
                    raise AssertionError("Packed first-chunk alignment requires an empty initial dynamic prior")
                eos_forbidden_steps = (
                    self.inference_parameters.min_generated_frames + self.frame_stacking_factor - 1
                ) // self.frame_stacking_factor
                status_capacity = eos_forbidden_steps + 1
                if packed_streaming_submission_profile is not None:
                    status_capacity = packed_streaming_submission_profile.status_capacity(
                        self.frame_stacking_factor,
                        self.inference_parameters.min_generated_frames,
                    )
                alignment_scratch = self._create_first_chunk_alignment_scratch(
                    text_lens=context_tensors.text_lens,
                    last_attended_timesteps=chunk_state.last_attended_timesteps,
                    effective_batch_size=batch_size * (2 if use_cfg else 1),
                    text_length=max_text_len,
                    dtype=context_tensors.text_encoder_out.dtype,
                    status_capacity=status_capacity,
                )
                logging.info("decoder_host_boundary_mode=packed_alignment_eos")
                if packed_streaming_submission_profile is not None:
                    logging.info(
                        "decoder_streaming_submission_macro=enabled,"
                        f"first_steps={packed_streaming_submission_profile.first_decoder_steps(self.frame_stacking_factor)},"
                        f"steady_steps={packed_streaming_submission_profile.steady_decoder_steps(self.frame_stacking_factor)}"
                    )
            else:
                logging.info("decoder_host_boundary_mode=tracking_or_multichunk")
            chunk_state.previous_attn_len = copy.deepcopy(text_lens_host)

            max_steps = self.inference_parameters.max_decoder_steps // self.frame_stacking_factor

            # Create decoder state object to track all local mutable state
            state = ChunkedDecoderState(
                audio_codes_step=audio_codes_step,
                audio_codes_mask=audio_codes_mask,
                attended_timestep_counter=[{} for _ in range(batch_size)],
                prediction_buffer=torch.empty(
                    (
                        batch_size,
                        self.num_audio_codebooks,
                        max_steps * self.frame_stacking_factor,
                    ),
                    device=device,
                    dtype=torch.long,
                ),
                num_prediction_steps=0,
                chunk_end_dict={},
                unfinished_texts={},
                finished_texts_counter={},
                attn_prior=initial_attn_prior,
                packed_alignment_eos=packed_alignment_eos,
                packed_streaming_submission_profile=packed_streaming_submission_profile,
                alignment_scratch=alignment_scratch,
                text_lens_host=text_lens_host,
            )
            local_ar_random_state = None
            if (
                local_ar_writes_audio_step
                and self.inference_parameters.temperature > 0.0
                and self.inference_parameters.topk > 1
            ):
                local_ar_random_state = self._lt_helper.create_random_state(
                    actual_batch_size=batch_size,
                    device=device,
                    seed=local_ar_seed,
                )
            # Frame-level lengths for this chunk only: batch_idx -> number of codec frames to keep
            # per item (used for predicted_codes_lens and trimming). Filled when EOS or chunk end
            # is detected.
            chunk_end_frame_lens: Dict[int, int] = {}

            decoder_session = None
            loop_start = 0
            terminate_before_loop = False
            if packed_streaming_submission_profile is not None and self._first_submission_cuda_graph_enabled:
                if not isinstance(dummy_cond, torch.Tensor) or not isinstance(dummy_cond_mask, torch.Tensor):
                    raise TypeError("First-submission CUDA graph requires tensor CFG conditioning")
                first_submission_key = self._build_first_submission_cuda_graph_key(
                    context_tensors=context_tensors,
                    dummy_cond=dummy_cond,
                    dummy_cond_mask=dummy_cond_mask,
                    dummy_additional_decoder_input=dummy_additional_decoder_input,
                    dummy_addition_dec_mask=dummy_addition_dec_mask,
                    actual_text_length=text_lens_host[0],
                    submission_profile=packed_streaming_submission_profile,
                )
                first_submission_lease = self._acquire_first_submission_cuda_graph(first_submission_key)
                if chunk_state.first_submission_graph_lease is not None:
                    first_submission_lease.release()
                    raise RuntimeError("Chunk state already owns a first-submission CUDA graph lease")
                chunk_state.first_submission_graph_lease = first_submission_lease
                first_submission = first_submission_lease.replay(
                    context_tensors=context_tensors,
                    dummy_cond=dummy_cond,
                    dummy_cond_mask=dummy_cond_mask,
                    dummy_additional_decoder_input=dummy_additional_decoder_input,
                    dummy_addition_dec_mask=dummy_addition_dec_mask,
                    text_lens=context_tensors.text_lens,
                    last_attended=chunk_state.last_attended_timesteps[-1],
                    random_state=local_ar_random_state,
                )
                decoder_session = first_submission.decoder_session
                state.alignment_scratch = first_submission.alignment_scratch
                state.attn_prior = first_submission.alignment_scratch.prior
                state.packed_status_pending_steps = 0

                first_submission_step_count = first_submission.decoder_step_count
                expected_first_submission_steps = packed_streaming_submission_profile.first_decoder_steps(
                    self.frame_stacking_factor
                )
                if first_submission_step_count != expected_first_submission_steps:
                    raise RuntimeError(
                        "First-submission graph replay step count does not match its selected profile: "
                        f"{first_submission_step_count} != {expected_first_submission_steps}"
                    )

                # The established host replay consumes graph rows in order and
                # discards a terminal speculative suffix.
                chunk_state.overall_idx += first_submission_step_count - 1
                terminal_status_step = self._apply_packed_alignment_eos_status(
                    packed_status=first_submission.packed_status,
                    first_pending_step=0,
                    state=state,
                    chunk_state=chunk_state,
                    chunk_end_frame_lens=chunk_end_frame_lens,
                    end_of_text=end_of_text,
                    batch_size=batch_size,
                )
                for graph_step in range(first_submission_step_count):
                    frame_start = graph_step * self.frame_stacking_factor
                    frame_end = frame_start + self.frame_stacking_factor
                    committed_graph_step = self._commit_decoder_prediction_step(
                        state,
                        first_submission.predicted_codes[:, :, frame_start:frame_end],
                        current_step=graph_step,
                        frame_stacking_factor=self.frame_stacking_factor,
                        terminal_step=(
                            terminal_status_step if graph_step == first_submission_step_count - 1 else None
                        ),
                    )
                    if committed_graph_step and state.num_prediction_steps == 1:
                        if first_prediction_event is None:
                            time_to_first_prediction = time.perf_counter() - generation_start_time
                        else:
                            first_prediction_event.record(generation_stream)
                        first_prediction_recorded = True

                should_terminate = self._should_terminate_loop(
                    chunk_state,
                    state.chunk_end_dict,
                    end_of_text,
                    batch_size,
                )
                generated_frame_count = state.num_prediction_steps * self.frame_stacking_factor
                valid_frame_count = chunk_end_frame_lens.get(0, generated_frame_count)
                should_emit = streaming_session is not None and streaming_emission_schedule.should_emit(
                    chunk_state.streaming_codec_emission_state,
                    available_frame_count=(valid_frame_count - streaming_emitted_frames),
                    model_chunk_ended=should_terminate,
                )
                if should_emit:
                    streaming_emitted_frames = self._submit_streaming_codec_chunk(
                        streaming_session,
                        state.prediction_buffer,
                        emission_state=chunk_state.streaming_codec_emission_state,
                        chunk_first_codec_frame=streaming_chunk_first_frame,
                        emitted_frame_count=streaming_emitted_frames,
                        valid_frame_count=valid_frame_count,
                        final=should_terminate and end_of_text[0],
                        wait_for_completion=streaming_emission_schedule.should_wait_for_completion(
                            chunk_state.streaming_codec_emission_state
                        ),
                    )

                if should_terminate:
                    terminate_before_loop = True
                else:
                    if terminal_status_step is not None:
                        raise RuntimeError(
                            f"First-submission graph reported terminal step {terminal_status_step} "
                            "without terminating a final text chunk"
                        )
                    state.audio_codes_step.copy_(
                        first_submission.predicted_codes[
                            :,
                            :,
                            -self.frame_stacking_factor :,
                        ]
                    )
                    chunk_state.overall_idx += 1
                    loop_start = first_submission_step_count

            for idx in range(loop_start, max_steps):
                if terminate_before_loop:
                    break
                if idx % 30 == 0:
                    logging.info(f"Decoding timestep {idx}")

                forbid_audio_eos = idx * self.frame_stacking_factor < self.inference_parameters.min_generated_frames

                # Embed only the newest frame stack. The fixed speaker/context
                # prefix is handled once by the decoder session prefill.
                audio_codes_embedded = self._embed_audio_step(state.audio_codes_step)
                if decoder_session is None:
                    decoder_session = self._create_incremental_decoder_session(
                        context_tensors=context_tensors,
                        use_cfg=use_cfg,
                        cfg_scale=cfg_scale,
                        dummy_cond=dummy_cond,
                        dummy_cond_mask=dummy_cond_mask,
                        dummy_additional_decoder_input=dummy_additional_decoder_input,
                        dummy_addition_dec_mask=dummy_addition_dec_mask,
                        batch_size=batch_size,
                        device=audio_codes_embedded.device,
                        dtype=audio_codes_embedded.dtype,
                    )

                # Prepare attention prior for layers
                if self.inference_parameters.apply_prior_to_layers is not None:
                    attn_prior = [None for _ in range(self.cfg.decoder.n_layers)]
                    for layer_idx in self.inference_parameters.apply_prior_to_layers:
                        attn_prior[layer_idx] = state.attn_prior
                else:
                    attn_prior = state.attn_prior

                if self.model_type == 'multi_encoder_context_tts':
                    attn_prior = [attn_prior, None]

                skip_main_projection = (
                    use_local_transformer_for_inference
                    and self.local_transformer_type == LocalTransformerType.AR
                    and forbid_audio_eos
                )

                # Run forward pass with optional CFG
                all_code_logits, alignment_attention_scores, dec_out = self._run_chunked_forward_with_cfg(
                    session=decoder_session,
                    audio_codes_embedded=audio_codes_embedded,
                    audio_codes_mask=state.audio_codes_mask,
                    attn_prior=attn_prior,
                    project_code_logits=not skip_main_projection,
                )  # (B, T, num_codebooks * num_tokens_per_codebook), (B, T, d_model)

                if self.inference_parameters.apply_attention_prior and alignment_attention_scores is None:
                    raise RuntimeError("Attention-prior inference requires decoder alignment scores")

                if self._should_update_attention_prior(decoder_step=idx):
                    if state.packed_alignment_eos:
                        if state.alignment_scratch is None:
                            raise AssertionError("Packed host-boundary mode requires alignment scratch")
                        if not self.inference_parameters.ignore_finished_sentence_tracking:
                            raise AssertionError("Packed host-boundary mode requires ignored sentence tracking")
                        state.attn_prior = self._compute_first_chunk_alignment_prior(
                            alignment_attention_scores,
                            state.alignment_scratch,
                        )
                    else:
                        text_time_step_attended, state.attended_timestep_counter = (
                            self.get_most_attended_text_timestep(
                                alignment_attention_scores=alignment_attention_scores,
                                last_attended_timesteps=chunk_state.last_attended_timesteps,
                                text_lens=context_tensors.text_lens,
                                lookahead_window_size=self.inference_parameters.attention_prior_lookahead_window,
                                attended_timestep_counter=state.attended_timestep_counter,
                                batch_size=batch_size,
                                left_offset=chunk_state.left_offset,
                            )
                        )
                        chunk_state.last_attended_timesteps.append(
                            text_time_step_attended.detach()
                            if isinstance(text_time_step_attended, torch.Tensor)
                            else text_time_step_attended
                        )

                        # Use different attention priors for first chunk vs subsequent chunks:
                        # - First chunk: use standard inference prior (more permissive, no history suppression)
                        # - Subsequent chunks: use multi-chunk prior (more restrictive, suppresses history/future)
                        if beginning_of_text:
                            # First chunk: use standard inference prior
                            (
                                state.attn_prior,
                                state.unfinished_texts,
                                state.finished_texts_counter,
                            ) = self.construct_inference_prior(
                                prior_epsilon=self.inference_parameters.attention_prior_epsilon,
                                cross_attention_scores=alignment_attention_scores,
                                text_lens=context_tensors.text_lens,
                                text_time_step_attended=text_time_step_attended,
                                attended_timestep_counter=state.attended_timestep_counter,
                                unfinished_texts=state.unfinished_texts,
                                finished_texts_counter=state.finished_texts_counter,
                                end_indices=chunk_state.end_indices,
                                lookahead_window_size=self.inference_parameters.attention_prior_lookahead_window,
                                batch_size=batch_size,
                            )
                        else:
                            # Subsequent chunks: use multi-chunk inference prior
                            (
                                state.attn_prior,
                                state.unfinished_texts,
                                state.finished_texts_counter,
                            ) = self.construct_multi_chunk_prior(
                                prior_epsilon=self.inference_parameters.attention_prior_epsilon,
                                cross_attention_scores=alignment_attention_scores,
                                text_lens=context_tensors.text_lens,
                                text_time_step_attended=text_time_step_attended,
                                attended_timestep_counter=state.attended_timestep_counter,
                                unfinished_texts=state.unfinished_texts,
                                finished_texts_counter=state.finished_texts_counter,
                                end_indices=chunk_state.end_indices,
                                chunk_end_dict=state.chunk_end_dict,
                                batch_size=batch_size,
                                left_offset=chunk_state.left_offset,
                            )

                if not beginning_of_text:
                    # Only increment here for multi-chunk path; construct_inference_prior
                    # (used when beginning_of_text=True) already increments internally.
                    for key in state.finished_texts_counter:
                        state.finished_texts_counter[key] += 1
                        limit = (
                            self.inference_parameters.finished_limit_with_eot
                            if end_of_text[key]
                            else self.inference_parameters.finished_limit_without_eot
                        )
                        if state.finished_texts_counter[key] > limit:
                            state.unfinished_texts[key] = False

                if self.inference_parameters.ignore_finished_sentence_tracking:
                    finished_items = {}
                    unfinished_items = {}
                else:
                    finished_threshold = (
                        self.inference_parameters.finished_limit_first_chunk
                        if beginning_of_text
                        else self.inference_parameters.finished_limit_with_eot
                    )
                    finished_items = {k: v for k, v in state.finished_texts_counter.items() if v >= finished_threshold}
                    unfinished_items = {k: v for k, v in state.unfinished_texts.items() if v}

                all_code_logits_t = (
                    None if all_code_logits is None else all_code_logits[:, -1, :]
                )  # (B, num_codebooks * num_tokens_per_codebook)

                if use_local_transformer_for_inference:
                    if self.local_transformer_type == LocalTransformerType.AR:
                        # Autoregressive sampling with local transformer
                        audio_codes_next = self._lt_helper.sample_autoregressive(
                            dec_output=dec_out[:, -1, :],
                            output_codes=state.audio_codes_step,
                            temperature=self.inference_parameters.temperature,
                            topk=self.inference_parameters.topk,
                            unfinished_items=unfinished_items,
                            finished_items=finished_items,
                            use_cfg=use_cfg,
                            cfg_scale=cfg_scale,
                            forbid_audio_eos=forbid_audio_eos,
                            random_state=local_ar_random_state,
                        )
                    elif self.local_transformer_type == LocalTransformerType.MASKGIT:
                        audio_codes_next = self._lt_helper.sample_maskgit(
                            dec_output=dec_out[:, -1, :],
                            temperature=self.inference_parameters.temperature,
                            topk=self.inference_parameters.topk,
                            unfinished_items=unfinished_items,
                            finished_items=finished_items,
                            use_cfg=use_cfg,
                            cfg_scale=cfg_scale,
                            n_steps=maskgit_n_steps,
                            noise_scale=maskgit_noise_scale,
                            fixed_schedule=maskgit_fixed_schedule,
                            dynamic_cfg_scale=maskgit_dynamic_cfg_scale,
                            sampling_type=maskgit_sampling_type,
                            forbid_audio_eos=forbid_audio_eos,
                        )
                    else:
                        raise ValueError(
                            f"Local transformer inference requested but local transformer type is {self.local_transformer_type}"
                        )
                else:
                    if all_code_logits_t is None:
                        raise AssertionError("Main-decoder sampling requires projected code logits")
                    audio_codes_next = self.sample_codes_from_logits(
                        all_code_logits_t,
                        temperature=self.inference_parameters.temperature,
                        topk=self.inference_parameters.topk,
                        unfinished_items=unfinished_items,
                        finished_items=finished_items,
                        forbid_audio_eos=forbid_audio_eos,
                    )  # (B, num_codebooks, frame_stacking_factor)
                if all_code_logits_t is None:
                    if not skip_main_projection:
                        raise AssertionError("Main logits are absent outside the forbidden local-AR path")
                    end_frame_indices_device = self.detect_forbidden_eos_batch(
                        audio_codes_next,
                        decoder_session.eos_scratch,
                    )
                else:
                    end_frame_indices_device = self.detect_eos_batch(
                        audio_codes_next,
                        all_code_logits_t,
                        eos_detection_method,
                        unfinished_items,
                        finished_items,
                        forbid_audio_eos,
                        decoder_session.eos_scratch,
                    )
                terminal_status_step = None
                if state.packed_alignment_eos:
                    if state.alignment_scratch is None:
                        raise AssertionError("Packed host-boundary mode requires alignment scratch")
                    self._record_packed_alignment_eos_status(
                        state.alignment_scratch,
                        pending_index=state.packed_status_pending_steps,
                        end_frame_indices=end_frame_indices_device,
                    )
                    state.packed_status_pending_steps += 1

                    generated_frame_count_after_step = (state.num_prediction_steps + 1) * self.frame_stacking_factor
                    codec_submission_due = streaming_session is not None and streaming_emission_schedule.should_emit(
                        chunk_state.streaming_codec_emission_state,
                        available_frame_count=(generated_frame_count_after_step - streaming_emitted_frames),
                        model_chunk_ended=False,
                    )
                    transfer_packed_status = self._packed_status_requires_transfer(
                        forbid_audio_eos=forbid_audio_eos,
                        final_text_chunk=all(end_of_text),
                        codec_submission_due=codec_submission_due,
                        decoder_limit_reached=(idx + 1 == max_steps),
                        defer_until_codec_submission=state.packed_streaming_submission_profile is not None,
                    )
                    if transfer_packed_status:
                        packed_status = self._transfer_packed_alignment_eos_status(
                            state.alignment_scratch,
                            pending_steps=state.packed_status_pending_steps,
                        )
                        first_pending_step = idx + 1 - state.packed_status_pending_steps
                        terminal_status_step = self._apply_packed_alignment_eos_status(
                            packed_status=packed_status,
                            first_pending_step=first_pending_step,
                            state=state,
                            chunk_state=chunk_state,
                            chunk_end_frame_lens=chunk_end_frame_lens,
                            end_of_text=end_of_text,
                            batch_size=batch_size,
                        )
                        state.packed_status_pending_steps = 0
                else:
                    end_frame_indices = self._transfer_eos_status(end_frame_indices_device)
                    self._update_eos_state(
                        chunk_state,
                        end_frame_indices,
                        state.chunk_end_dict,
                        chunk_end_frame_lens,
                        state.finished_texts_counter,
                        end_of_text,
                        idx,
                        batch_size,
                    )

                committed_current_step = self._commit_decoder_prediction_step(
                    state,
                    audio_codes_next,
                    current_step=idx,
                    frame_stacking_factor=self.frame_stacking_factor,
                    terminal_step=terminal_status_step,
                )
                if committed_current_step and state.num_prediction_steps == 1:
                    if first_prediction_event is None:
                        time_to_first_prediction = time.perf_counter() - generation_start_time
                    else:
                        first_prediction_event.record(generation_stream)
                    first_prediction_recorded = True
                if committed_current_step and not local_ar_writes_audio_step:
                    state.audio_codes_step.copy_(audio_codes_next)

                # Check termination condition
                should_terminate = self._should_terminate_loop(
                    chunk_state, state.chunk_end_dict, end_of_text, batch_size
                )
                generated_frame_count = state.num_prediction_steps * self.frame_stacking_factor
                valid_frame_count = chunk_end_frame_lens.get(0, generated_frame_count)
                if valid_frame_count < streaming_emitted_frames:
                    raise RuntimeError(
                        "EOS shortened codec frames that were already submitted: "
                        f"{valid_frame_count} < {streaming_emitted_frames}"
                    )
                should_emit = streaming_session is not None and streaming_emission_schedule.should_emit(
                    chunk_state.streaming_codec_emission_state,
                    available_frame_count=(valid_frame_count - streaming_emitted_frames),
                    model_chunk_ended=should_terminate,
                )
                if should_emit:
                    is_final_stream_chunk = should_terminate and end_of_text[0]
                    wait_for_completion = streaming_emission_schedule.should_wait_for_completion(
                        chunk_state.streaming_codec_emission_state
                    )
                    streaming_emitted_frames = self._submit_streaming_codec_chunk(
                        streaming_session,
                        state.prediction_buffer,
                        emission_state=chunk_state.streaming_codec_emission_state,
                        chunk_first_codec_frame=streaming_chunk_first_frame,
                        emitted_frame_count=streaming_emitted_frames,
                        valid_frame_count=valid_frame_count,
                        final=is_final_stream_chunk,
                        wait_for_completion=wait_for_completion,
                    )

                if should_terminate:
                    break

                chunk_state.overall_idx += 1
            else:
                self._raise_decoder_limit_error(
                    streaming_session,
                    generated_frame_count=(state.num_prediction_steps * self.frame_stacking_factor),
                )

            default_frame_len = state.num_prediction_steps * self.frame_stacking_factor
            predicted_codes_lens = torch.tensor(
                [chunk_end_frame_lens.get(item_idx, default_frame_len) for item_idx in range(batch_size)],
                device=device,
            )
            predicted_codes = state.prediction_buffer[:, :, :default_frame_len]
            if not first_prediction_recorded:
                raise RuntimeError("Incremental generation completed without producing a prediction")
            if default_frame_len < 1:
                raise RuntimeError(f"Incremental generation produced invalid frame length {default_frame_len}")
            if generation_end_event is None:
                tts_generation_time = time.perf_counter() - generation_start_time
            else:
                if generation_start_event is None or first_prediction_event is None or generation_stream is None:
                    raise AssertionError("CUDA generation timing events were only partially initialized")
                if not first_prediction_recorded:
                    raise RuntimeError("CUDA generation completed without recording its first prediction")
                generation_end_event.record(generation_stream)
                generation_end_event.synchronize()
                time_to_first_prediction = generation_start_event.elapsed_time(first_prediction_event) / 1000.0
                tts_generation_time = generation_start_event.elapsed_time(generation_end_event) / 1000.0
            rtf_metrics = {
                "time_to_first_prediction": time_to_first_prediction,
                "tts_generation_time": tts_generation_time,
                "max_frames_generated": state.num_prediction_steps,
                "tts_generation_time_per_frame": tts_generation_time / default_frame_len,
                "batch_size": batch_size,
            }

            return InferBatchOutput(
                predicted_audio=torch.empty(0, device=device),
                predicted_audio_lens=torch.empty(0, device=device, dtype=torch.long),
                predicted_codes=predicted_codes,
                predicted_codes_lens=predicted_codes_lens,
                rtf_metrics=rtf_metrics,
                cross_attention_maps=[],
                headwise_cross_attention_maps=[],
            )
