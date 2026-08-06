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
    / "alignment_controller.py"
)
SPEC = importlib.util.spec_from_file_location("alignment_controller", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SofiaAlignmentControllerTests(unittest.TestCase):
    def test_selects_leftmost_maximum_and_constructs_cfg_prior(self) -> None:
        alignment = torch.zeros((2, 10), dtype=torch.bfloat16)
        alignment[0, 5] = 4.0
        alignment[0, 6] = 4.0
        controller = MODULE.SofiaAlignmentController(
            text_length=10,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        update = controller.update(alignment)

        self.assertEqual(update.attended.tolist(), [5])
        self.assertEqual(controller.counters[0, 5].item(), 1)
        expected_conditional = torch.full(
            (10,),
            MODULE.PRIOR_EPSILON,
            dtype=torch.bfloat16,
        )
        expected_conditional[4:] = 1.0
        self.assertTrue(torch.equal(update.prior[0, 0], expected_conditional))
        self.assertTrue(
            torch.equal(
                update.prior[1, 0],
                torch.full(
                    (10,),
                    MODULE.PRIOR_EPSILON,
                    dtype=torch.bfloat16,
                ),
            )
        )

    def test_sink_threshold_advances_search_and_suppresses_history(self) -> None:
        alignment = torch.zeros((2, 12), dtype=torch.bfloat16)
        alignment[0, 3] = 8.0
        alignment[0, 5] = 7.0
        controller = MODULE.SofiaAlignmentController(
            text_length=12,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        for _ in range(MODULE.SINK_THRESHOLD):
            update = controller.update(alignment)

        self.assertEqual(update.attended.tolist(), [3])
        self.assertEqual(controller.counters[0, 3].item(), MODULE.SINK_THRESHOLD)
        self.assertTrue(
            torch.equal(
                update.prior[0, 0, :4],
                torch.full(
                    (4,),
                    MODULE.PRIOR_EPSILON,
                    dtype=torch.bfloat16,
                ),
            )
        )

        advanced = controller.update(alignment)
        self.assertEqual(advanced.attended.tolist(), [5])
        self.assertEqual(controller.counters[0, 5].item(), 1)

    def test_short_text_uses_uniform_conditional_prior(self) -> None:
        alignment = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.bfloat16,
        )
        controller = MODULE.SofiaAlignmentController(
            text_length=5,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

        update = controller.update(alignment)

        self.assertTrue(
            torch.equal(
                update.prior[0, 0],
                torch.ones((5,), dtype=torch.bfloat16),
            )
        )
        self.assertTrue(
            torch.equal(
                update.prior[1, 0],
                torch.full(
                    (5,),
                    MODULE.PRIOR_EPSILON,
                    dtype=torch.bfloat16,
                ),
            )
        )

    def test_rejects_non_bf16_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be BF16"):
            MODULE.SofiaAlignmentController(
                text_length=10,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_rejects_mismatched_alignment(self) -> None:
        controller = MODULE.SofiaAlignmentController(
            text_length=10,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        with self.assertRaisesRegex(ValueError, "does not match controller"):
            controller.update(torch.zeros((1, 10), dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
