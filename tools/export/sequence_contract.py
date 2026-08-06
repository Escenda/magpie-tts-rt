"""Fail-closed code/EOS accounting for the locked Sofia generation loop."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


ACTUAL_BATCH = 1
CODEBOOKS = 8
FRAMES_PER_STEP = 2
LOCAL_AR_POSITIONS = CODEBOOKS * FRAMES_PER_STEP
MIN_GENERATED_FRAMES = 4
MAX_DECODER_STEPS = 250


@dataclass(frozen=True)
class CodeMismatch:
    frame: int
    codebook: int
    actual: int | None
    expected: int | None


@dataclass(frozen=True)
class SequenceComparison:
    code_exact: bool
    first_mismatch: CodeMismatch | None
    generated_frames: int
    expected_frames: int
    generated_codes_sha256: str
    expected_codes_sha256: str
    terminal_decoder_step: int
    terminal_end_frame_index: int
    local_ar_invocations: int
    final_rng_counter: int


def _require_tensor(
    value: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(
            f"{name} must be {dtype} {shape}, got "
            f"{value.dtype} {tuple(value.shape)}"
        )


def _int64_payload(value: torch.Tensor) -> bytes:
    if value.dtype != torch.int64:
        raise ValueError(f"code tensor must be INT64, got {value.dtype}")
    return (
        value.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<i8", copy=False)
        .tobytes()
    )


class SequenceCodeTracker:
    """Track one batch-one generation and preserve only frames before EOS."""

    def __init__(self, *, initial_rng_counter: int = 0) -> None:
        if initial_rng_counter < 0:
            raise ValueError("initial_rng_counter must be non-negative")
        self.decoder_step = 0
        self.rng_counter = initial_rng_counter
        self.terminal_end_frame_index: int | None = None
        self.terminal_decoder_step: int | None = None
        self._chunks: list[torch.Tensor] = []

    @property
    def forbid_eos(self) -> bool:
        return (
            self.decoder_step * FRAMES_PER_STEP
            < MIN_GENERATED_FRAMES
        )

    @property
    def ended(self) -> bool:
        return self.terminal_decoder_step is not None

    @property
    def local_ar_invocations(self) -> int:
        return self.decoder_step + (1 if self.ended else 0)

    def accept_step(
        self,
        *,
        codec_tokens: torch.Tensor,
        updated_rng_counter: torch.Tensor,
        invalid_rows: torch.Tensor,
        end_frame_index: torch.Tensor,
    ) -> bool:
        if self.ended:
            raise RuntimeError("cannot accept a Local AR step after EOS")
        if self.decoder_step >= MAX_DECODER_STEPS:
            raise RuntimeError(
                f"decoder step limit is {MAX_DECODER_STEPS}"
            )
        _require_tensor(
            codec_tokens,
            name="codec_tokens",
            dtype=torch.int64,
            shape=(ACTUAL_BATCH, CODEBOOKS, FRAMES_PER_STEP),
        )
        _require_tensor(
            updated_rng_counter,
            name="updated_rng_counter",
            dtype=torch.int64,
            shape=(ACTUAL_BATCH,),
        )
        _require_tensor(
            invalid_rows,
            name="invalid_rows",
            dtype=torch.int32,
            shape=(ACTUAL_BATCH,),
        )
        _require_tensor(
            end_frame_index,
            name="end_frame_index",
            dtype=torch.int32,
            shape=(ACTUAL_BATCH,),
        )
        invalid = int(invalid_rows.item())
        if invalid != 0:
            raise RuntimeError(f"Local AR rejected the row: {invalid}")
        next_counter = int(updated_rng_counter.item())
        expected_counter = self.rng_counter + LOCAL_AR_POSITIONS
        if next_counter != expected_counter:
            raise RuntimeError(
                "Local AR RNG counter is not contiguous: "
                f"expected={expected_counter}, actual={next_counter}"
            )
        end_index = int(end_frame_index.item())
        if end_index not in (-1, 0, 1):
            raise RuntimeError(
                f"Local AR end_frame_index must be -1, 0, or 1, got {end_index}"
            )
        if self.forbid_eos and end_index != -1:
            raise RuntimeError(
                "Local AR reported EOS before min_generated_frames: "
                f"decoder_step={self.decoder_step}, end_frame_index={end_index}"
            )

        valid_frames = FRAMES_PER_STEP if end_index == -1 else end_index
        if valid_frames:
            self._chunks.append(
                codec_tokens[:, :, :valid_frames].detach().cpu().clone()
            )
        self.rng_counter = next_counter
        if end_index == -1:
            self.decoder_step += 1
            return False

        self.terminal_decoder_step = self.decoder_step
        self.terminal_end_frame_index = end_index
        return True

    def generated_codes(self) -> torch.Tensor:
        if not self._chunks:
            return torch.empty(
                (ACTUAL_BATCH, CODEBOOKS, 0),
                dtype=torch.int64,
            )
        return torch.cat(self._chunks, dim=2)

    def compare(self, expected_codes: torch.Tensor) -> SequenceComparison:
        if not self.ended:
            raise RuntimeError("cannot compare a sequence before EOS")
        if expected_codes.ndim != 3:
            raise ValueError(
                "expected_codes must have rank 3, got "
                f"{expected_codes.ndim}"
            )
        _require_tensor(
            expected_codes,
            name="expected_codes",
            dtype=torch.int64,
            shape=(
                ACTUAL_BATCH,
                CODEBOOKS,
                expected_codes.shape[-1],
            ),
        )
        generated = self.generated_codes()
        generated_frames = generated.shape[2]
        expected_frames = expected_codes.shape[2]
        overlap = min(generated_frames, expected_frames)
        first_mismatch: CodeMismatch | None = None
        if overlap:
            unequal = generated[:, :, :overlap] != expected_codes[:, :, :overlap].cpu()
            mismatch_indices = torch.nonzero(
                unequal.transpose(1, 2),
                as_tuple=False,
            )
            if mismatch_indices.numel():
                _, frame, codebook = mismatch_indices[0].tolist()
                first_mismatch = CodeMismatch(
                    frame=frame,
                    codebook=codebook,
                    actual=int(generated[0, codebook, frame]),
                    expected=int(expected_codes[0, codebook, frame]),
                )
        if first_mismatch is None and generated_frames != expected_frames:
            frame = overlap
            codebook = 0
            first_mismatch = CodeMismatch(
                frame=frame,
                codebook=codebook,
                actual=(
                    int(generated[0, codebook, frame])
                    if frame < generated_frames
                    else None
                ),
                expected=(
                    int(expected_codes[0, codebook, frame])
                    if frame < expected_frames
                    else None
                ),
            )
        generated_payload = _int64_payload(generated)
        expected_payload = _int64_payload(expected_codes)
        if self.terminal_decoder_step is None:
            raise AssertionError("ended sequence has no terminal decoder step")
        if self.terminal_end_frame_index is None:
            raise AssertionError("ended sequence has no terminal EOS frame")
        return SequenceComparison(
            code_exact=first_mismatch is None,
            first_mismatch=first_mismatch,
            generated_frames=generated_frames,
            expected_frames=expected_frames,
            generated_codes_sha256=hashlib.sha256(generated_payload).hexdigest(),
            expected_codes_sha256=hashlib.sha256(expected_payload).hexdigest(),
            terminal_decoder_step=self.terminal_decoder_step,
            terminal_end_frame_index=self.terminal_end_frame_index,
            local_ar_invocations=self.local_ar_invocations,
            final_rng_counter=self.rng_counter,
        )
