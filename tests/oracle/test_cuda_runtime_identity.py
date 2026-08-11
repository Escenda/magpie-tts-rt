from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT = PROJECT_ROOT / "tools" / "export" / "cuda_runtime_identity.py"
SPEC = importlib.util.spec_from_file_location("cuda_runtime_identity", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CudaRuntimeIdentityTests(unittest.TestCase):
    def test_exact_identity_parses(self) -> None:
        identity = MODULE.parse_cuda_runtime_identity(
            {
                "cuda_driver_api_version_integer": 13020,
                "cuda_runtime_version_integer": 13020,
                "nvidia_driver_version": "595.78",
            },
            "runtime",
        )
        self.assertEqual(identity.cuda_driver_api_version_integer, 13020)
        self.assertEqual(identity.cuda_runtime_version_integer, 13020)
        self.assertEqual(identity.nvidia_driver_version, "595.78")

    def test_missing_field_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "keys mismatch"):
            MODULE.parse_cuda_runtime_identity(
                {
                    "cuda_runtime_version_integer": 13020,
                    "nvidia_driver_version": "595.78",
                },
                "runtime",
            )

    def test_non_numeric_driver_version_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "numeric NVIDIA driver"):
            MODULE.parse_cuda_runtime_identity(
                {
                    "cuda_driver_api_version_integer": 13020,
                    "cuda_runtime_version_integer": 13020,
                    "nvidia_driver_version": "latest",
                },
                "runtime",
            )


if __name__ == "__main__":
    unittest.main()
