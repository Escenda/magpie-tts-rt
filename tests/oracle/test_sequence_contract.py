from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[2]
    / "tools"
    / "export"
    / "sequence_contract.py"
)
SPEC = importlib.util.spec_from_file_location("sequence_contract", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def codes(first_frame_value: int) -> torch.Tensor:
    values = torch.arange(
        first_frame_value,
        first_frame_value + 16,
        dtype=torch.int64,
    )
    return values.reshape(1, 8, 2)


def scalar(value: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([value], dtype=dtype)


class SequenceCodeTrackerTests(unittest.TestCase):
    def accept(
        self,
        tracker,
        *,
        value: int,
        counter: int,
        end: int,
        invalid: int = 0,
    ) -> bool:
        return tracker.accept_step(
            codec_tokens=codes(value),
            updated_rng_counter=scalar(counter, torch.int64),
            invalid_rows=scalar(invalid, torch.int32),
            end_frame_index=scalar(end, torch.int32),
        )

    def test_odd_frame_terminal_sequence_is_exact(self) -> None:
        tracker = MODULE.SequenceCodeTracker()
        self.assertFalse(self.accept(tracker, value=0, counter=16, end=-1))
        self.assertFalse(self.accept(tracker, value=16, counter=32, end=-1))
        self.assertTrue(self.accept(tracker, value=32, counter=48, end=1))
        expected = torch.cat(
            [
                codes(0),
                codes(16),
                codes(32)[:, :, :1],
            ],
            dim=2,
        )

        comparison = tracker.compare(expected)

        self.assertTrue(comparison.code_exact)
        self.assertIsNone(comparison.first_mismatch)
        self.assertEqual(comparison.generated_frames, 5)
        self.assertEqual(comparison.terminal_decoder_step, 2)
        self.assertEqual(comparison.terminal_end_frame_index, 1)
        self.assertEqual(comparison.local_ar_invocations, 3)
        self.assertEqual(comparison.final_rng_counter, 48)

    def test_even_frame_terminal_sequence_keeps_no_eos_step_frames(self) -> None:
        tracker = MODULE.SequenceCodeTracker()
        self.accept(tracker, value=0, counter=16, end=-1)
        self.accept(tracker, value=16, counter=32, end=-1)
        self.assertTrue(self.accept(tracker, value=32, counter=48, end=0))
        comparison = tracker.compare(torch.cat([codes(0), codes(16)], dim=2))
        self.assertTrue(comparison.code_exact)
        self.assertEqual(comparison.generated_frames, 4)
        self.assertEqual(comparison.terminal_end_frame_index, 0)

    def test_reports_first_code_mismatch(self) -> None:
        tracker = MODULE.SequenceCodeTracker()
        self.accept(tracker, value=0, counter=16, end=-1)
        self.accept(tracker, value=16, counter=32, end=-1)
        self.accept(tracker, value=32, counter=48, end=0)
        expected = torch.cat([codes(0), codes(16)], dim=2)
        expected[0, 0, 3] = 888
        expected[0, 3, 2] = 999

        comparison = tracker.compare(expected)

        self.assertFalse(comparison.code_exact)
        self.assertEqual(
            comparison.first_mismatch,
            MODULE.CodeMismatch(
                frame=2,
                codebook=3,
                actual=int(codes(16)[0, 3, 0]),
                expected=999,
            ),
        )

    def test_reports_length_mismatch_after_common_prefix(self) -> None:
        tracker = MODULE.SequenceCodeTracker()
        self.accept(tracker, value=0, counter=16, end=-1)
        self.accept(tracker, value=16, counter=32, end=-1)
        self.accept(tracker, value=32, counter=48, end=1)
        expected = torch.cat([codes(0), codes(16)], dim=2)

        comparison = tracker.compare(expected)

        self.assertFalse(comparison.code_exact)
        self.assertEqual(comparison.first_mismatch.frame, 4)
        self.assertIsNone(comparison.first_mismatch.expected)

    def test_rejects_eos_while_forbidden(self) -> None:
        tracker = MODULE.SequenceCodeTracker()
        with self.assertRaisesRegex(RuntimeError, "before min_generated_frames"):
            self.accept(tracker, value=0, counter=16, end=0)

    def test_rejects_rng_counter_discontinuity(self) -> None:
        tracker = MODULE.SequenceCodeTracker()
        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            self.accept(tracker, value=0, counter=17, end=-1)

    def test_rejects_invalid_plugin_row(self) -> None:
        tracker = MODULE.SequenceCodeTracker()
        with self.assertRaisesRegex(RuntimeError, "rejected the row"):
            self.accept(
                tracker,
                value=0,
                counter=16,
                end=-1,
                invalid=1,
            )

    def test_rejects_compare_before_eos(self) -> None:
        tracker = MODULE.SequenceCodeTracker()
        with self.assertRaisesRegex(RuntimeError, "before EOS"):
            tracker.compare(torch.empty((1, 8, 0), dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
