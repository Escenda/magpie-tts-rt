"""Exact Sofia alignment-prior controller used by the sequence validator.

The TensorRT Main Decoder only produces alignment scores. This state machine
implements the locked host-side policy that selects the next attended text
position and constructs the prior consumed by the next decoder step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


CFG_ROWS = 2
PRIOR_EPSILON = 0.1
INITIAL_ATTENDED = 1
IGNORED_TERMINAL_TOKENS = 3
SHORT_TEXT_NO_PRIOR_MAX_TOKENS = 5
LOOKAHEAD = 6
SINK_THRESHOLD = 4


@dataclass(frozen=True)
class AlignmentUpdate:
    prior: torch.Tensor
    attended: torch.Tensor


class SofiaAlignmentController:
    """Session-owned monotonic alignment state for one actual CFG batch row."""

    def __init__(
        self,
        *,
        text_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if text_length < 1:
            raise ValueError(f"text_length must be positive, got {text_length}")
        if dtype != torch.bfloat16:
            raise ValueError(f"Sofia alignment dtype must be BF16, got {dtype}")
        self.text_length = text_length
        self.device = device
        self.dtype = dtype
        self.last_attended = torch.full(
            (1,),
            min(INITIAL_ATTENDED, text_length - 1),
            dtype=torch.int64,
            device=device,
        )
        self.counters = torch.zeros(
            (1, text_length),
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.arange(
            text_length,
            dtype=torch.int64,
            device=device,
        ).unsqueeze(0)

    def update(self, alignment: torch.Tensor) -> AlignmentUpdate:
        expected_shape = (CFG_ROWS, self.text_length)
        if (
            alignment.shape != expected_shape
            or alignment.dtype != self.dtype
            or alignment.device != self.device
        ):
            raise ValueError(
                "alignment does not match controller state: "
                f"expected={expected_shape}/{self.dtype}/{self.device}, "
                f"actual={tuple(alignment.shape)}/{alignment.dtype}/{alignment.device}"
            )

        last_count = self.counters.gather(
            1,
            self.last_attended.unsqueeze(1),
        ).squeeze(1)
        search_start = self.last_attended + (
            last_count >= SINK_THRESHOLD
        ).to(torch.int64)
        search_start = search_start.clamp(min=0, max=self.text_length)
        window_end = torch.minimum(
            search_start + LOOKAHEAD,
            torch.full_like(
                search_start,
                self.text_length - IGNORED_TERMINAL_TOKENS,
            ),
        )
        valid_window = (
            (self.positions >= search_start.unsqueeze(1))
            & (self.positions < window_end.unsqueeze(1))
        )
        masked_scores = alignment[:1].masked_fill(~valid_window, float("-inf"))
        maximum = torch.argmax(masked_scores, dim=1)
        attended = torch.where(
            window_end > search_start,
            maximum,
            torch.full_like(maximum, self.text_length - 1),
        )
        self.last_attended.copy_(attended)
        self.counters.scatter_add_(
            1,
            attended.unsqueeze(1),
            torch.ones((1, 1), dtype=torch.int64, device=self.device),
        )

        prior = torch.full(
            (CFG_ROWS, 1, self.text_length),
            PRIOR_EPSILON,
            dtype=self.dtype,
            device=self.device,
        )
        conditional = prior[:1, 0]
        history_floor = min(INITIAL_ATTENDED, self.text_length - 1)
        prior_indices = [
            (attended - 1).clamp(
                min=history_floor,
                max=self.text_length - 1,
            ),
            attended,
        ]
        last_position = torch.full_like(attended, self.text_length - 1)
        for offset in range(1, LOOKAHEAD + 1):
            prior_indices.append(
                torch.minimum(attended + offset, last_position)
            )
        conditional.scatter_(
            1,
            torch.stack(prior_indices, dim=1),
            1.0,
        )
        if self.text_length <= SHORT_TEXT_NO_PRIOR_MAX_TOKENS:
            conditional.fill_(1.0)

        sink_positions = self.positions.masked_fill(
            self.counters < SINK_THRESHOLD,
            -1,
        )
        maximum_sink = sink_positions.amax(dim=1)
        conditional.masked_fill_(
            self.positions <= maximum_sink.unsqueeze(1),
            PRIOR_EPSILON,
        )
        return AlignmentUpdate(prior=prior, attended=attended.clone())
