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

"""Asynchronous hand-off from streaming code generation to causal codec decode.

The generation thread only copies newly finalized codec frames on its CUDA
stream, records an event, and enqueues the resulting submission.  A session
owned worker waits for that event from a separate CUDA stream, performs exact
rolling decode, copies only the new PCM suffix to pinned host memory, and calls
the consumer in submission order.

Queue saturation is an explicit error.  Chunks are never dropped, reordered,
or silently decoded synchronously on the producer thread.
"""

from __future__ import annotations

import queue
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import torch

from nemo.collections.tts.modules.streaming_codec import (
    CausalCodecStreamingCudaGraphLease,
    CausalCodecStreamingCudaGraphRuntime,
    CausalCodecStreamingDecoder,
    CausalCodecStreamingState,
    preallocate_causal_codec_lengths,
)


class AsyncCodecSynthesisError(RuntimeError):
    """Base class for asynchronous codec session failures."""


class AsyncCodecBackpressureError(AsyncCodecSynthesisError):
    """The bounded work queue has no capacity for another chunk."""


class AsyncCodecWorkerError(AsyncCodecSynthesisError):
    """The codec worker or PCM callback failed."""


class AsyncCodecCancelledError(AsyncCodecSynthesisError):
    """An operation was requested after session cancellation."""


class AsyncCodecClosedError(AsyncCodecSynthesisError):
    """An operation was requested after the session was closed."""


@dataclass(frozen=True)
class _PcmSlotToken:
    slot_index: int
    generation: int


class _PcmSlotState(Enum):
    FREE = "free"
    QUEUED = "queued"
    IN_FLIGHT = "in_flight"
    DELIVERED_RETAINED = "delivered_retained"
    ABANDONED = "abandoned"


class _BoundedPcmSlotPool:
    """Thread-safe ownership state for preallocated CUDA/D2H slots.

    Queue capacity and retained PCM capacity are deliberately independent.
    A callback-owned PCM tensor keeps its slot in ``DELIVERED_RETAINED`` until
    that exact tensor is destroyed. Cancelled work moves to ``ABANDONED`` and
    is never reused because its producer-stream copy may still be in flight.
    """

    def __init__(self, slot_count: int):
        if slot_count < 1:
            raise ValueError(f"PCM slot_count must be at least 1, got {slot_count}")
        self.slot_count = slot_count
        self._lock = threading.Lock()
        self._states = [_PcmSlotState.FREE] * slot_count
        self._generations = [0] * slot_count
        self._free_slots = list(reversed(range(slot_count)))
        self._closed = False

    def acquire(self) -> _PcmSlotToken:
        with self._lock:
            if self._closed:
                raise AsyncCodecClosedError("PCM output slot pool is closed")
            if not self._free_slots:
                counts = self._state_counts_locked()
                raise AsyncCodecBackpressureError(
                    "PCM output slot pool is exhausted independently of the codec work queue: "
                    f"capacity={self.slot_count}, queued={counts[_PcmSlotState.QUEUED]}, "
                    f"in_flight={counts[_PcmSlotState.IN_FLIGHT]}, "
                    f"delivered_retained={counts[_PcmSlotState.DELIVERED_RETAINED]}, "
                    f"abandoned={counts[_PcmSlotState.ABANDONED]}"
                )
            slot_index = self._free_slots.pop()
            if self._states[slot_index] != _PcmSlotState.FREE:
                raise RuntimeError(f"PCM slot {slot_index} was listed free but is {self._states[slot_index].value}")
            self._generations[slot_index] += 1
            self._states[slot_index] = _PcmSlotState.QUEUED
            return _PcmSlotToken(slot_index=slot_index, generation=self._generations[slot_index])

    def mark_in_flight(self, token: _PcmSlotToken) -> None:
        self._transition(token, expected=_PcmSlotState.QUEUED, target=_PcmSlotState.IN_FLIGHT)

    def mark_delivered(self, token: _PcmSlotToken) -> None:
        self._transition(
            token,
            expected=_PcmSlotState.IN_FLIGHT,
            target=_PcmSlotState.DELIVERED_RETAINED,
        )

    def release_delivered(self, token: _PcmSlotToken) -> None:
        self._release(token, expected=_PcmSlotState.DELIVERED_RETAINED)

    def abandon_queued(self, token: _PcmSlotToken) -> None:
        self._transition(token, expected=_PcmSlotState.QUEUED, target=_PcmSlotState.ABANDONED)

    def abandon_in_flight(self, token: _PcmSlotToken) -> None:
        self._transition(token, expected=_PcmSlotState.IN_FLIGHT, target=_PcmSlotState.ABANDONED)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("PCM output slot pool was already closed")
            counts = self._state_counts_locked()
            if counts[_PcmSlotState.QUEUED] or counts[_PcmSlotState.IN_FLIGHT]:
                raise RuntimeError(
                    "Cannot close PCM output slot pool with unfinished codec ownership: "
                    f"queued={counts[_PcmSlotState.QUEUED]}, in_flight={counts[_PcmSlotState.IN_FLIGHT]}"
                )
            self._closed = True

    def state_counts(self) -> dict[_PcmSlotState, int]:
        with self._lock:
            return self._state_counts_locked()

    def slot_indices(self, state: _PcmSlotState) -> tuple[int, ...]:
        with self._lock:
            return tuple(index for index, actual in enumerate(self._states) if actual == state)

    def _transition(
        self,
        token: _PcmSlotToken,
        *,
        expected: _PcmSlotState,
        target: _PcmSlotState,
    ) -> None:
        with self._lock:
            self._validate_token_locked(token)
            actual = self._states[token.slot_index]
            if actual != expected:
                raise RuntimeError(
                    f"PCM slot {token.slot_index} transition requires {expected.value}, got {actual.value}"
                )
            self._states[token.slot_index] = target

    def _release(self, token: _PcmSlotToken, *, expected: _PcmSlotState) -> None:
        with self._lock:
            self._validate_token_locked(token)
            actual = self._states[token.slot_index]
            if actual != expected:
                raise RuntimeError(
                    f"PCM slot {token.slot_index} release requires {expected.value}, got {actual.value}"
                )
            self._states[token.slot_index] = _PcmSlotState.FREE
            if not self._closed:
                self._free_slots.append(token.slot_index)

    def _validate_token_locked(self, token: _PcmSlotToken) -> None:
        if token.slot_index < 0 or token.slot_index >= self.slot_count:
            raise RuntimeError(f"PCM slot token index is out of range: {token.slot_index}")
        actual_generation = self._generations[token.slot_index]
        if token.generation != actual_generation:
            raise RuntimeError(
                f"PCM slot {token.slot_index} token is stale: {token.generation} != {actual_generation}"
            )

    def _state_counts_locked(self) -> dict[_PcmSlotState, int]:
        return {state: self._states.count(state) for state in _PcmSlotState}


class _PcmSlotLifetime:
    """Release one output slot only after every callback-owned Tensor dies."""

    def __init__(
        self,
        *,
        pool: _BoundedPcmSlotPool,
        token: _PcmSlotToken,
        owner_count: int,
    ):
        if owner_count < 1:
            raise ValueError(f"PCM slot lifetime needs at least one owner, got {owner_count}")
        self._pool = pool
        self._token = token
        self._remaining_owners = owner_count
        self._lock = threading.Lock()

    def release_owner(self) -> None:
        release_slot = False
        with self._lock:
            if self._remaining_owners < 1:
                raise RuntimeError("PCM slot lifetime owner was released more than once")
            self._remaining_owners -= 1
            release_slot = self._remaining_owners == 0
        if release_slot:
            self._pool.release_delivered(self._token)


def _attach_pcm_slot_lifetimes(
    tensors: tuple[torch.Tensor, ...],
    *,
    pool: _BoundedPcmSlotPool,
    token: _PcmSlotToken,
) -> None:
    """Keep a delivered slot until every exact callback Tensor is destroyed."""

    lifetime = _PcmSlotLifetime(pool=pool, token=token, owner_count=len(tensors))
    finalizers: list[weakref.finalize] = []
    try:
        for tensor in tensors:
            finalizer = weakref.finalize(tensor, lifetime.release_owner)
            if not finalizer.alive:
                raise RuntimeError("Failed to attach PCM output slot lifetime to callback Tensor")
            finalizers.append(finalizer)
    except BaseException:
        for finalizer in finalizers:
            finalizer.detach()
        raise


@dataclass(frozen=True)
class StreamingPcmChunk:
    """A finalized, ordered PCM suffix delivered to the consumer.

    CUDA-backed ``samples`` owns a preallocated output slot for exactly as long
    as this Tensor object is retained. A consumer that needs a derived view
    after dropping ``samples`` must first make an owning copy.
    """

    sequence_index: int
    first_codec_frame: int
    codec_frame_count: int
    final: bool
    samples: torch.Tensor
    sample_lengths: torch.Tensor
    submit_duration_seconds: float
    submitted_at: float
    ready_at: float


@dataclass(frozen=True)
class StreamingCodecChunk:
    """Newly finalized codec frames produced by Magpie.

    ``first_codec_frame`` is absolute across every text chunk in one synthesis
    session. A terminal zero-frame chunk is permitted so an EOS in the first
    position of a frame stack can still close the stream without decoding EOS.
    """

    codes: torch.Tensor
    first_codec_frame: int
    codec_frame_count: int
    final: bool

    def __post_init__(self) -> None:
        if self.codes.ndim != 3:
            raise ValueError(f"Expected [batch, codebooks, frames], got shape={tuple(self.codes.shape)}")
        if self.codes.shape[0] != 1:
            raise ValueError(
                "Streaming codec synthesis currently has an explicit batch=1 contract, "
                f"got batch={self.codes.shape[0]}"
            )
        if self.codes.shape[1] < 1:
            raise ValueError("Streaming codec synthesis requires at least one codebook")
        if self.codes.dtype != torch.long:
            raise ValueError(f"Streaming codec frames must use torch.int64, got {self.codes.dtype}")
        if self.first_codec_frame < 0:
            raise ValueError(f"first_codec_frame must be non-negative, got {self.first_codec_frame}")
        if self.codec_frame_count != self.codes.shape[-1]:
            raise ValueError(
                "codec_frame_count must match the new-frame tensor: "
                f"{self.codec_frame_count} != {self.codes.shape[-1]}"
            )
        if self.codec_frame_count < 0:
            raise ValueError(f"codec_frame_count must be non-negative, got {self.codec_frame_count}")
        if self.codec_frame_count == 0 and not self.final:
            raise ValueError("A zero-frame codec chunk is only valid as the terminal marker")


class CodecSubmission(ABC):
    """Backend-owned immutable submission captured from the producer."""


class CodecChunkProcessor(ABC):
    """Session-owned bridge used by the generic ordered worker.

    Production uses :class:`CudaCodecChunkProcessor`. Tests can provide a
    deterministic processor without requiring CUDA while exercising the same
    queue, lifecycle, and exception behavior.
    """

    @abstractmethod
    def prepare(
        self,
        codes: torch.Tensor,
        producer_stream: Optional[torch.cuda.Stream],
    ) -> CodecSubmission:
        """Capture immutable new codec frames without waiting for decode."""

    @abstractmethod
    def decode_to_host(self, submission: CodecSubmission) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one submission and return finalized PCM in host memory."""

    @abstractmethod
    def discard(self, submission: CodecSubmission) -> None:
        """Invalidate an accepted preparation that will never be decoded."""

    @abstractmethod
    def close(self) -> None:
        """Release all request-scoped processing resources."""


@dataclass(frozen=True)
class _CudaCodecSubmission(CodecSubmission):
    slot_token: _PcmSlotToken
    codec_frame_count: int


@dataclass(frozen=True)
class _CudaCodecIoSlot:
    code_views: tuple[torch.Tensor, ...]
    host_audio_views: tuple[torch.Tensor, ...]
    host_length_views: tuple[torch.Tensor, ...]
    ready_event: torch.cuda.Event
    copy_complete_event: torch.cuda.Event


class _CudaCodecChunkProcessorBase(CodecChunkProcessor):
    """Common CUDA stream and preallocated immutable-output bridge."""

    _DEFAULT_MAX_CODEC_FRAMES = 8
    _DEFAULT_MAX_RETAINED_PCM_CHUNKS = 64

    def __init__(
        self,
        *,
        device: torch.device | str,
        samples_per_frame: int,
        codebook_count: int,
        pcm_dtype: torch.dtype,
        max_codec_frames: int = _DEFAULT_MAX_CODEC_FRAMES,
        max_retained_pcm_chunks: int = _DEFAULT_MAX_RETAINED_PCM_CHUNKS,
    ):
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError(f"CUDA codec processing requires a CUDA device, got {resolved_device}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; asynchronous CUDA codec processing cannot start")
        if resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        if samples_per_frame < 1:
            raise ValueError(f"samples_per_frame must be positive, got {samples_per_frame}")
        if codebook_count < 1:
            raise ValueError(f"codebook_count must be positive, got {codebook_count}")
        if not pcm_dtype.is_floating_point:
            raise TypeError(f"pcm_dtype must be floating point, got {pcm_dtype}")
        if max_codec_frames < 1:
            raise ValueError(f"max_codec_frames must be positive, got {max_codec_frames}")
        if max_retained_pcm_chunks < 1:
            raise ValueError(f"max_retained_pcm_chunks must be positive, got {max_retained_pcm_chunks}")

        self.device = resolved_device
        self.samples_per_frame = samples_per_frame
        self.codebook_count = codebook_count
        self.pcm_dtype = pcm_dtype
        self.max_codec_frames = max_codec_frames
        self.max_retained_pcm_chunks = max_retained_pcm_chunks
        self._slot_pool = _BoundedPcmSlotPool(max_retained_pcm_chunks)
        self._closed = False
        with torch.cuda.device(self.device):
            self.codec_stream = torch.cuda.Stream(device=self.device)
            slots = []
            for _ in range(max_retained_pcm_chunks):
                code_storage = torch.empty(
                    (1, codebook_count, max_codec_frames),
                    dtype=torch.long,
                    device=self.device,
                )
                host_audio_storage = torch.empty(
                    (1, max_codec_frames * samples_per_frame),
                    dtype=pcm_dtype,
                    device="cpu",
                    pin_memory=True,
                )
                slots.append(
                    _CudaCodecIoSlot(
                        code_views=tuple(
                            code_storage[:, :, :frame_count] for frame_count in range(1, max_codec_frames + 1)
                        ),
                        host_audio_views=tuple(
                            host_audio_storage[:, : frame_count * samples_per_frame]
                            for frame_count in range(1, max_codec_frames + 1)
                        ),
                        host_length_views=tuple(
                            torch.full(
                                (1,),
                                frame_count * samples_per_frame,
                                dtype=torch.long,
                                device="cpu",
                            )
                            for frame_count in range(1, max_codec_frames + 1)
                        ),
                        ready_event=torch.cuda.Event(blocking=False, enable_timing=False),
                        copy_complete_event=torch.cuda.Event(blocking=False, enable_timing=False),
                    )
                )
        self._slots = tuple(slots)

    def prepare(
        self,
        codes: torch.Tensor,
        producer_stream: Optional[torch.cuda.Stream],
    ) -> CodecSubmission:
        if codes.ndim != 3:
            raise ValueError(f"Codec frames must have shape [batch, codebooks, frames], got {tuple(codes.shape)}")
        if codes.dtype != torch.long:
            raise ValueError(f"Codec frames must use torch.int64, got {codes.dtype}")
        if codes.device.type != "cuda":
            raise ValueError(f"Codec frames must be CUDA tensors, got {codes.device}")
        if codes.device != self.device:
            raise ValueError(f"Codec frame device changed from {self.device} to {codes.device}")
        if codes.shape[0:2] != (1, self.codebook_count):
            raise ValueError(
                "Codec frame shape changed from the preallocated slot contract: "
                f"{tuple(codes.shape[0:2])} != {(1, self.codebook_count)}"
            )
        codec_frame_count = codes.shape[-1]
        if codec_frame_count < 1 or codec_frame_count > self.max_codec_frames:
            raise ValueError(f"Codec frame count must be in 1..{self.max_codec_frames}, got {codec_frame_count}")

        with torch.cuda.device(self.device):
            stream = producer_stream if producer_stream is not None else torch.cuda.current_stream(self.device)
            if stream.device != self.device:
                raise ValueError(f"Producer stream device {stream.device} does not match codec device {self.device}")
            slot_token = self._slot_pool.acquire()
            slot = self._slots[slot_token.slot_index]
            try:
                with torch.inference_mode(), torch.cuda.stream(stream):
                    # Copy into request-owned preallocated storage before the
                    # generator can overwrite its reusable prediction buffer.
                    slot.code_views[codec_frame_count - 1].copy_(codes)
                    slot.ready_event.record(stream)
            except BaseException:
                self._slot_pool.abandon_queued(slot_token)
                raise

        return _CudaCodecSubmission(
            slot_token=slot_token,
            codec_frame_count=codec_frame_count,
        )

    @abstractmethod
    def _decode_new(
        self,
        codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one immutable CUDA submission on ``codec_stream``."""

    def decode_to_host(self, submission: CodecSubmission) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(submission, _CudaCodecSubmission):
            raise TypeError(f"CUDA processor received {type(submission).__name__}")

        slot_token = submission.slot_token
        slot = self._slots[slot_token.slot_index]
        codec_frame_count = submission.codec_frame_count
        expected_sample_count = codec_frame_count * self.samples_per_frame
        self._slot_pool.mark_in_flight(slot_token)
        delivered = False
        try:
            with torch.cuda.device(self.device), torch.inference_mode(), torch.cuda.stream(self.codec_stream):
                self.codec_stream.wait_event(slot.ready_event)
                audio, audio_lengths = self._decode_new(
                    codes=slot.code_views[codec_frame_count - 1],
                )
                if audio.shape != (1, expected_sample_count):
                    raise RuntimeError(
                        "Rolling codec returned the wrong PCM suffix shape: "
                        f"{tuple(audio.shape)} != {(1, expected_sample_count)}"
                    )
                if audio_lengths.shape != (audio.shape[0],):
                    raise RuntimeError(
                        f"Rolling codec returned invalid length shape {tuple(audio_lengths.shape)} "
                        f"for batch {audio.shape[0]}"
                    )
                host_copy_target = slot.host_audio_views[codec_frame_count - 1]
                if host_copy_target.dtype != audio.dtype:
                    raise RuntimeError(
                        "Preallocated PCM slot dtype does not match codec output: "
                        f"{host_copy_target.dtype} != {audio.dtype}"
                    )
                host_copy_target.copy_(audio, non_blocking=True)
                slot.copy_complete_event.record(self.codec_stream)

            # Waiting and the callback both happen on the worker, never the
            # generation thread. The event covers codec kernels and D2H.
            slot.copy_complete_event.synchronize()

            # Create only Tensor metadata over preallocated pinned storage.
            # This exact Tensor object's lifetime is the immutable-output lease:
            # retaining it in or after the callback prevents slot reuse.
            host_audio = host_copy_target.as_strided(
                size=host_copy_target.shape,
                stride=host_copy_target.stride(),
            )
            host_length_storage = slot.host_length_views[codec_frame_count - 1]
            host_length_storage.fill_(expected_sample_count)
            host_lengths = host_length_storage.as_strided(
                size=host_length_storage.shape,
                stride=host_length_storage.stride(),
            )
            self._slot_pool.mark_delivered(slot_token)
            delivered = True
            _attach_pcm_slot_lifetimes(
                (host_audio, host_lengths),
                pool=self._slot_pool,
                token=slot_token,
            )
            return host_audio, host_lengths
        except BaseException:
            if delivered:
                self._slot_pool.release_delivered(slot_token)
            else:
                self._slot_pool.abandon_in_flight(slot_token)
            raise

    def discard(self, submission: CodecSubmission) -> None:
        if not isinstance(submission, _CudaCodecSubmission):
            raise TypeError(f"CUDA processor received {type(submission).__name__}")
        self._slot_pool.abandon_queued(submission.slot_token)

    def close(self) -> None:
        if self._closed:
            raise RuntimeError("CUDA codec processor was already closed")
        with torch.cuda.device(self.device):
            # Abandoned producer copies did not enter codec_stream. Explicitly
            # finish them before closing; their slots are never recycled.
            for slot_index in self._slot_pool.slot_indices(_PcmSlotState.ABANDONED):
                self._slots[slot_index].ready_event.synchronize()
            self.codec_stream.synchronize()
        self._slot_pool.close()
        self._closed = True


class CudaCodecChunkProcessor(_CudaCodecChunkProcessorBase):
    """CUDA bridge with a fresh eager causal state for one request."""

    def __init__(
        self,
        decoder: CausalCodecStreamingDecoder,
        device: torch.device | str,
        *,
        max_codec_frames: int = _CudaCodecChunkProcessorBase._DEFAULT_MAX_CODEC_FRAMES,
        max_retained_pcm_chunks: int = _CudaCodecChunkProcessorBase._DEFAULT_MAX_RETAINED_PCM_CHUNKS,
    ):
        first_parameter = next(decoder.codec_model.audio_decoder.parameters(), None)
        if first_parameter is None:
            raise RuntimeError("Streaming codec requires a parameterized audio decoder")
        super().__init__(
            device=device,
            samples_per_frame=decoder.samples_per_frame,
            codebook_count=decoder.input_codebook_count,
            pcm_dtype=first_parameter.dtype,
            max_codec_frames=max_codec_frames,
            max_retained_pcm_chunks=max_retained_pcm_chunks,
        )
        self.decoder = decoder
        self.state = CausalCodecStreamingState()
        with torch.cuda.device(self.device), torch.cuda.stream(self.codec_stream):
            self._device_lengths = preallocate_causal_codec_lengths(
                batch_size=1,
                max_codec_frames=max_codec_frames,
                samples_per_frame=decoder.samples_per_frame,
                device=self.device,
            )

    def _decode_new(
        self,
        codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = self._device_lengths[codes.shape[-1] - 1]
        return self.decoder.decode_new(
            codes=codes,
            state=self.state,
            lengths=lengths,
        )


class CudaGraphCodecChunkProcessor(_CudaCodecChunkProcessorBase):
    """CUDA bridge holding one exclusive lease on a pre-captured runtime."""

    def __init__(
        self,
        runtime: CausalCodecStreamingCudaGraphRuntime,
        device: torch.device | str,
        *,
        max_codec_frames: int = _CudaCodecChunkProcessorBase._DEFAULT_MAX_CODEC_FRAMES,
        max_retained_pcm_chunks: int = _CudaCodecChunkProcessorBase._DEFAULT_MAX_RETAINED_PCM_CHUNKS,
    ):
        if max_codec_frames != runtime.max_codec_frames:
            raise ValueError(
                "CUDA Graph codec processor max_codec_frames must match the captured runtime: "
                f"{max_codec_frames} != {runtime.max_codec_frames}"
            )
        resolved_device = torch.device(device)
        if resolved_device.index is None and resolved_device.type == "cuda":
            resolved_device = torch.device("cuda", torch.cuda.current_device())
        if resolved_device != runtime.device:
            raise ValueError(
                f"CUDA Graph runtime device {runtime.device} does not match requested device {resolved_device}"
            )
        self.runtime = runtime
        self.lease: CausalCodecStreamingCudaGraphLease = runtime.acquire()
        try:
            super().__init__(
                device=resolved_device,
                samples_per_frame=self.lease.samples_per_frame,
                codebook_count=runtime.decoder.input_codebook_count,
                pcm_dtype=runtime.decoder_dtype,
                max_codec_frames=max_codec_frames,
                max_retained_pcm_chunks=max_retained_pcm_chunks,
            )
        except BaseException:
            self.lease.release()
            raise
        self._closed = False

    def _decode_new(
        self,
        codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._closed:
            raise RuntimeError("CUDA Graph codec processor is closed")
        return self.lease.decode_new(codes)

    def close(self) -> None:
        if self._closed:
            raise RuntimeError("CUDA Graph codec processor was already closed")
        try:
            super().close()
        finally:
            try:
                self.lease.release()
            finally:
                self._closed = True


@dataclass(frozen=True)
class _CodecWorkItem:
    sequence_index: int
    first_codec_frame: int
    codec_frame_count: int
    final: bool
    submit_duration_seconds: float
    submitted_at: float
    submission: Optional[CodecSubmission]


@dataclass(frozen=True)
class _StopWork:
    pass


class _SessionStatus(Enum):
    RUNNING = "running"
    CANCELLING = "cancelling"
    CLOSED = "closed"
    FAILED = "failed"


PcmCallback = Callable[[StreamingPcmChunk], None]


class AsyncCodecSynthesisSession:
    """Ordered, non-blocking producer session for streaming codec decode.

    ``submit_chunk`` never waits for codec execution, D2H, or the callback.
    It can raise :class:`AsyncCodecBackpressureError` when the bounded queue is
    full; the caller must then stop or otherwise apply explicit upstream
    backpressure. There is deliberately no synchronous fallback or chunk drop.
    """

    def __init__(
        self,
        processor: CodecChunkProcessor,
        callback: PcmCallback,
        *,
        max_queued_chunks: int = 8,
        first_codec_frame: int = 0,
        worker_name: str = "async-codec-synthesis",
    ):
        if max_queued_chunks < 1:
            raise ValueError(f"max_queued_chunks must be at least 1, got {max_queued_chunks}")
        if first_codec_frame < 0:
            raise ValueError(f"first_codec_frame must be non-negative, got {first_codec_frame}")

        self._processor = processor
        self._callback = callback
        self._work_queue: queue.Queue[_CodecWorkItem | _StopWork] = queue.Queue(maxsize=max_queued_chunks)
        self._condition = threading.Condition()
        self._callback_lock = threading.Lock()
        self._status = _SessionStatus.RUNNING
        self._failure: Optional[BaseException] = None
        self._next_sequence_index = 0
        self._next_codec_frame = first_codec_frame
        self._accepted_chunks = 0
        self._terminal_chunks = 0
        self._final_submitted = False
        self._stop_enqueued = False
        self._empty_pcm = torch.empty((1, 0), dtype=torch.float32, device="cpu")
        self._empty_pcm_lengths = torch.zeros((1,), dtype=torch.long, device="cpu")
        self._worker = threading.Thread(
            target=self._worker_main,
            name=worker_name,
            daemon=True,
        )
        self._worker.start()

    @classmethod
    def for_cuda(
        cls,
        decoder: CausalCodecStreamingDecoder,
        callback: PcmCallback,
        *,
        device: torch.device | str,
        max_queued_chunks: int = 8,
        max_codec_frames: int = _CudaCodecChunkProcessorBase._DEFAULT_MAX_CODEC_FRAMES,
        max_retained_pcm_chunks: int = _CudaCodecChunkProcessorBase._DEFAULT_MAX_RETAINED_PCM_CHUNKS,
        first_codec_frame: int = 0,
        worker_name: str = "cuda-codec-synthesis",
    ) -> AsyncCodecSynthesisSession:
        """Create a CUDA-backed session with fresh rolling decoder state."""

        processor = CudaCodecChunkProcessor(
            decoder=decoder,
            device=device,
            max_codec_frames=max_codec_frames,
            max_retained_pcm_chunks=max_retained_pcm_chunks,
        )
        return cls(
            processor=processor,
            callback=callback,
            max_queued_chunks=max_queued_chunks,
            first_codec_frame=first_codec_frame,
            worker_name=worker_name,
        )

    @classmethod
    def for_cuda_graph(
        cls,
        runtime: CausalCodecStreamingCudaGraphRuntime,
        callback: PcmCallback,
        *,
        device: torch.device | str,
        max_queued_chunks: int = 8,
        max_codec_frames: int = _CudaCodecChunkProcessorBase._DEFAULT_MAX_CODEC_FRAMES,
        max_retained_pcm_chunks: int = _CudaCodecChunkProcessorBase._DEFAULT_MAX_RETAINED_PCM_CHUNKS,
        first_codec_frame: int = 0,
        worker_name: str = "cuda-graph-codec-synthesis",
    ) -> AsyncCodecSynthesisSession:
        """Lease one pre-captured graph runtime for this synthesis request."""

        processor = CudaGraphCodecChunkProcessor(
            runtime=runtime,
            device=device,
            max_codec_frames=max_codec_frames,
            max_retained_pcm_chunks=max_retained_pcm_chunks,
        )
        try:
            return cls(
                processor=processor,
                callback=callback,
                max_queued_chunks=max_queued_chunks,
                first_codec_frame=first_codec_frame,
                worker_name=worker_name,
            )
        except BaseException:
            processor.close()
            raise

    @property
    def next_codec_frame(self) -> int:
        with self._condition:
            return self._next_codec_frame

    @property
    def pending_chunks(self) -> int:
        with self._condition:
            return self._accepted_chunks - self._terminal_chunks

    def submit_chunk(
        self,
        chunk: StreamingCodecChunk,
        *,
        producer_stream: Optional[torch.cuda.Stream] = None,
    ) -> int:
        """Enqueue newly finalized code frames and return their sequence index.

        ``chunk.first_codec_frame`` must equal the first frame after the preceding
        accepted submission. This explicit continuity check rejects duplicate,
        cumulative-prefix, missing, and reordered submissions.
        """

        self._reject_worker_reentrancy("submit codec frames")
        submit_started = time.perf_counter()

        with self._condition:
            self._raise_for_status_locked()
            if self._final_submitted:
                raise AsyncCodecClosedError("The terminal codec chunk was already submitted")
            if chunk.first_codec_frame != self._next_codec_frame:
                raise ValueError(
                    "Codec frame submissions must be contiguous and contain new frames only: "
                    f"expected first frame {self._next_codec_frame}, got {chunk.first_codec_frame}"
                )
            if self._work_queue.full():
                raise AsyncCodecBackpressureError(f"Codec queue is full at {self._work_queue.maxsize} queued chunks")

            submission = (
                self._processor.prepare(codes=chunk.codes, producer_stream=producer_stream)
                if chunk.codec_frame_count > 0
                else None
            )
            sequence_index = self._next_sequence_index
            submitted_at = time.perf_counter()
            item = _CodecWorkItem(
                sequence_index=sequence_index,
                first_codec_frame=chunk.first_codec_frame,
                codec_frame_count=chunk.codec_frame_count,
                final=chunk.final,
                submit_duration_seconds=submitted_at - submit_started,
                submitted_at=submitted_at,
                submission=submission,
            )
            try:
                self._work_queue.put_nowait(item)
            except queue.Full as error:
                # Another producer may have raced after the explicit capacity
                # check. The submission was never accepted, and the error is
                # returned rather than dropping any previously accepted work.
                if submission is not None:
                    self._processor.discard(submission)
                raise AsyncCodecBackpressureError(
                    f"Codec queue is full at {self._work_queue.maxsize} queued chunks"
                ) from error

            self._next_sequence_index += 1
            self._next_codec_frame += chunk.codec_frame_count
            self._accepted_chunks += 1
            self._final_submitted = chunk.final
            self._condition.notify_all()
            return sequence_index

    def flush(self, timeout: Optional[float] = None) -> None:
        """Wait until every accepted chunk reached callback or terminal cancel."""

        self._reject_worker_reentrancy("flush the codec session")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            target_chunks = self._accepted_chunks
            while self._terminal_chunks < target_chunks and self._status != _SessionStatus.FAILED:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError(
                        f"Timed out with {target_chunks - self._terminal_chunks} codec chunks unfinished"
                    )
                self._condition.wait(timeout=remaining)

            self._raise_for_status_locked(allow_closed=True)

    def wait_for_sequence(self, sequence_index: int, timeout: Optional[float] = None) -> None:
        """Wait until one accepted sequence reached its ordered callback."""

        self._reject_worker_reentrancy("wait for a codec sequence")
        if sequence_index < 0:
            raise ValueError(f"sequence_index must be non-negative, got {sequence_index}")
        if timeout is not None and timeout < 0.0:
            raise ValueError(f"timeout must be non-negative, got {timeout}")
        deadline = None if timeout is None else time.monotonic() + timeout
        target_chunks = sequence_index + 1
        with self._condition:
            if target_chunks > self._accepted_chunks:
                raise ValueError(
                    f"Codec sequence {sequence_index} was not accepted; "
                    f"accepted sequence count is {self._accepted_chunks}"
                )
            while self._terminal_chunks < target_chunks and self._status != _SessionStatus.FAILED:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    raise TimeoutError(f"Timed out waiting for codec sequence {sequence_index}")
                self._condition.wait(timeout=remaining)

            self._raise_for_status_locked(allow_closed=True)

    def cancel(self, *, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Cancel the session and suppress callbacks for all later completions."""

        self._reject_worker_reentrancy("cancel the codec session")
        if timeout is not None and timeout < 0.0:
            raise ValueError(f"timeout must be non-negative, got {timeout}")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            if self._status == _SessionStatus.CLOSED:
                return
            if self._status == _SessionStatus.FAILED:
                self._raise_for_status_locked()
            already_cancelling = self._status == _SessionStatus.CANCELLING
            if not already_cancelling:
                self._status = _SessionStatus.CANCELLING
                self._condition.notify_all()

        if not already_cancelling:
            # The worker checks cancellation while holding this same lock before
            # invoking a callback. Acquiring it here forms a barrier: when cancel
            # returns, no callback can still be running or begin afterward.
            if deadline is None:
                callback_barrier_acquired = self._callback_lock.acquire()
            else:
                callback_barrier_acquired = self._callback_lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
            if callback_barrier_acquired:
                self._callback_lock.release()

            # Queue mutation and stop-sentinel insertion happen exactly once.
            # A repeated non-blocking cancel must not consume the sentinel and
            # leave the worker blocked forever in queue.get().
            self._discard_queued_after_cancel()
            self._enqueue_stop()
            if not callback_barrier_acquired:
                raise TimeoutError("Timed out waiting for the active codec callback to finish")
        if wait:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            self._join_worker(timeout=remaining)

    def close(self, *, timeout: Optional[float] = None) -> None:
        """Flush accepted work, stop the worker, and propagate worker failure."""

        self._reject_worker_reentrancy("close the codec session")
        with self._condition:
            self._raise_for_status_locked(allow_closed=True)
            if self._status != _SessionStatus.CLOSED and not self._final_submitted:
                raise AsyncCodecClosedError(
                    "Cannot close an incomplete codec stream; submit a chunk with final=True or cancel it"
                )
        try:
            self.flush(timeout=timeout)
        except AsyncCodecWorkerError:
            self._join_worker(timeout=timeout)
            raise

        with self._condition:
            if self._status == _SessionStatus.CLOSED:
                return
            if self._status == _SessionStatus.CANCELLING:
                raise AsyncCodecCancelledError("Codec synthesis session was cancelled")
            self._status = _SessionStatus.CLOSED
            self._condition.notify_all()
        self._enqueue_stop()
        self._join_worker(timeout=timeout)

    def __enter__(self) -> AsyncCodecSynthesisSession:
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        if exception_type is None:
            self.close()
        else:
            self.cancel()

    def _worker_main(self) -> None:
        try:
            while True:
                item = self._work_queue.get()
                if isinstance(item, _StopWork):
                    self._work_queue.task_done()
                    return

                try:
                    if item.submission is None:
                        samples = self._empty_pcm
                        sample_lengths = self._empty_pcm_lengths
                    else:
                        samples, sample_lengths = self._processor.decode_to_host(item.submission)
                    with self._callback_lock:
                        with self._condition:
                            should_callback = self._status == _SessionStatus.RUNNING
                        if should_callback:
                            chunk = StreamingPcmChunk(
                                sequence_index=item.sequence_index,
                                first_codec_frame=item.first_codec_frame,
                                codec_frame_count=item.codec_frame_count,
                                final=item.final,
                                samples=samples,
                                sample_lengths=sample_lengths,
                                submit_duration_seconds=item.submit_duration_seconds,
                                submitted_at=item.submitted_at,
                                ready_at=time.perf_counter(),
                            )
                            self._callback(chunk)
                            del chunk
                    del samples
                    del sample_lengths
                except BaseException as error:
                    self._record_worker_failure(error)
                    self._finish_item()
                    self._work_queue.task_done()
                    self._discard_queued_after_failure()
                    return

                self._finish_item()
                self._work_queue.task_done()
        finally:
            try:
                self._processor.close()
            except BaseException as error:
                self._record_worker_failure(error)

    def _finish_item(self) -> None:
        with self._condition:
            self._terminal_chunks += 1
            self._condition.notify_all()

    def _record_worker_failure(self, error: BaseException) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = error
            self._status = _SessionStatus.FAILED
            self._condition.notify_all()

    def _discard_queued_after_failure(self) -> None:
        discarded = 0
        while True:
            try:
                item = self._work_queue.get_nowait()
            except queue.Empty:
                break
            self._work_queue.task_done()
            if isinstance(item, _CodecWorkItem):
                if item.submission is not None:
                    self._processor.discard(item.submission)
                discarded += 1
        if discarded:
            with self._condition:
                self._terminal_chunks += discarded
                self._condition.notify_all()

    def _discard_queued_after_cancel(self) -> None:
        discarded = 0
        while True:
            try:
                item = self._work_queue.get_nowait()
            except queue.Empty:
                break
            self._work_queue.task_done()
            if isinstance(item, _CodecWorkItem):
                if item.submission is not None:
                    self._processor.discard(item.submission)
                discarded += 1
        if discarded:
            with self._condition:
                self._terminal_chunks += discarded
                self._condition.notify_all()

    def _enqueue_stop(self) -> None:
        with self._condition:
            if self._stop_enqueued or not self._worker.is_alive():
                return
            self._stop_enqueued = True
        self._work_queue.put(_StopWork())

    def _join_worker(self, timeout: Optional[float]) -> None:
        if threading.current_thread() is self._worker:
            raise RuntimeError("The codec callback cannot join its own worker thread")
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            raise TimeoutError("Timed out waiting for the codec worker to stop")
        with self._condition:
            if self._status == _SessionStatus.FAILED:
                self._raise_for_status_locked()

    def _reject_worker_reentrancy(self, operation: str) -> None:
        if threading.current_thread() is self._worker:
            raise RuntimeError(
                f"The PCM callback cannot {operation}; session lifecycle is owned by the generation thread"
            )

    def _raise_for_status_locked(self, *, allow_closed: bool = False) -> None:
        if self._status == _SessionStatus.FAILED:
            error = AsyncCodecWorkerError("Asynchronous codec worker failed")
            if self._failure is not None:
                raise error from self._failure
            raise error
        if self._status == _SessionStatus.CANCELLING:
            raise AsyncCodecCancelledError("Codec synthesis session was cancelled")
        if self._status == _SessionStatus.CLOSED and not allow_closed:
            raise AsyncCodecClosedError("Codec synthesis session is closed")
