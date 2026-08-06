from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[2]
EXPORT_TOOLS = ROOT / "tools" / "export"
if str(EXPORT_TOOLS) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS))

from nanocodec_contract import CANONICAL_STATE_BINDINGS
from sync_nanocodec_manifest_contract import (
    synchronize_fixture,
    synchronize_schema,
)


class NanoCodecCanonicalRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(
            (ROOT / "schemas" / "runtime-bundle-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.fixture = json.loads(
            (ROOT / "tests" / "manifest" / "fixtures" / "minimal-valid.json").read_text(
                encoding="utf-8"
            )
        )

    def test_registry_has_97_named_persistent_tensors(self) -> None:
        self.assertEqual(len(CANONICAL_STATE_BINDINGS), 97)
        self.assertEqual(
            len(
                {
                    binding.logical_name
                    for binding in CANONICAL_STATE_BINDINGS
                }
            ),
            97,
        )
        self.assertTrue(
            all(
                binding.dtype == "fp32"
                and binding.initial_output_binding.startswith("state_out.")
                and binding.steady_input_binding.startswith("state_in.")
                for binding in CANONICAL_STATE_BINDINGS
            )
        )

    def test_schema_is_synchronized(self) -> None:
        expected = copy.deepcopy(self.schema)
        synchronize_schema(expected)
        self.assertEqual(self.schema, expected)

    def test_manifest_fixture_is_synchronized(self) -> None:
        expected = copy.deepcopy(self.fixture)
        synchronize_fixture(expected)
        self.assertEqual(self.fixture, expected)

    def test_engine_binding_counts_are_exact(self) -> None:
        engines = {
            engine["role"]: engine for engine in self.fixture["engines"]
        }
        self.assertEqual(
            (
                len(engines["nanocodec_initial_4"]["inputs"]),
                len(engines["nanocodec_initial_4"]["outputs"]),
            ),
            (1, 99),
        )
        self.assertEqual(
            (
                len(engines["nanocodec_steady_8"]["inputs"]),
                len(engines["nanocodec_steady_8"]["outputs"]),
            ),
            (98, 99),
        )
        self.assertEqual(
            (
                len(engines["nanocodec_tail_1_8"]["inputs"]),
                len(engines["nanocodec_tail_1_8"]["outputs"]),
            ),
            (98, 99),
        )


if __name__ == "__main__":
    unittest.main()
