from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "tools" / "export" / "build_text_encoder_plan.py"
SPEC = importlib.util.spec_from_file_location("build_text_encoder_plan", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TextEncoderParityGateTests(unittest.TestCase):
    def passing_metrics(self) -> dict[str, float]:
        return {
            "max_absolute_error": MODULE.MAX_ABSOLUTE_ERROR,
            "mean_absolute_error": MODULE.MAX_MEAN_ABSOLUTE_ERROR,
            "p99_absolute_error": MODULE.MAX_P99_ABSOLUTE_ERROR,
            "cosine_similarity": MODULE.MIN_COSINE_SIMILARITY,
        }

    def test_accepts_every_declared_boundary(self) -> None:
        self.assertTrue(MODULE.parity_passes(self.passing_metrics()))

    def test_rejects_each_exceeded_error_metric(self) -> None:
        keys = (
            "max_absolute_error",
            "mean_absolute_error",
            "p99_absolute_error",
        )
        for key in keys:
            with self.subTest(key=key):
                metrics = self.passing_metrics()
                metrics[key] = metrics[key] * 1.01
                self.assertFalse(MODULE.parity_passes(metrics))

    def test_rejects_low_cosine_similarity(self) -> None:
        metrics = self.passing_metrics()
        metrics["cosine_similarity"] -= 0.00001
        self.assertFalse(MODULE.parity_passes(metrics))

    def test_rejects_non_finite_metrics(self) -> None:
        for key in self.passing_metrics():
            with self.subTest(key=key):
                metrics = self.passing_metrics()
                metrics[key] = float("nan")
                self.assertFalse(MODULE.parity_passes(metrics))

    def test_plugin_creator_contract_is_exact_and_ordered(self) -> None:
        self.assertEqual(
            MODULE.EXPECTED_PLUGIN_CREATORS,
            (
                ("MagpieLocalARSampling", "1", "magpie_tts_rt"),
                ("MagpieLocalAREos", "1", "magpie_tts_rt"),
                ("MagpieLayerNorm", "1", "magpie_tts_rt"),
                ("MagpieGeluTanh", "1", "magpie_tts_rt"),
                ("MagpieSoftmax", "1", "magpie_tts_rt"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
