from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path


EXPORT_TOOLS = Path(__file__).parents[2] / "tools" / "export"
if str(EXPORT_TOOLS) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS))
SCRIPT = EXPORT_TOOLS / "mode8_class_table_identity.py"
SPEC = importlib.util.spec_from_file_location(
    "mode8_class_table_identity",
    SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


EXPECTED_FIXTURE_SHA256 = (
    "9dc9516d753da031147f50ff34296b2bf"
    "5f4498d4c789d18dc2fc19120190dbe"
)


def canonical_fixture() -> MODULE.JsonObject:
    classes: list[MODULE.JsonValue] = []
    for index in range(21):
        operation = "qk" if index < 7 else "pv"
        class_index = index if index < 7 else index - 7
        classes.append(
            {
                "operation": operation,
                "class_index": class_index,
                "function_name": f"kernel_{operation}_{class_index}",
                "parameter_transport": (
                    "kernel_params" if index % 2 == 0 else "extra"
                ),
                "block": [32 * (1 + index % 4), 1 + index % 2, 1],
                "shared_memory_bytes": index * 16,
                "parameter_offset": index * 8,
                "parameter_size": 8 + (index % 3) * 8,
            }
        )
    k_records: list[MODULE.JsonValue] = []
    for index in range(249):
        k_records.append(
            {
                "active_k": 219 + index,
                "qk": {
                    "class_index": index % 7,
                    "grid": [1 + index % 5, 1, 1],
                },
                "pv": {
                    "class_index": index % 14,
                    "grid": [1 + index % 3, 1 + index % 2, 1],
                },
            }
        )
    return {
        "schema_version": 1,
        "active_k_range": [219, 467],
        "qk_class_count": 7,
        "pv_class_count": 14,
        "classes": classes,
        "k_records": k_records,
    }


class Mode8ClassTableIdentityTests(unittest.TestCase):
    def test_cross_language_fixture_has_locked_digest(self) -> None:
        document = canonical_fixture()
        payload = MODULE.canonical_class_table_bytes(document)
        self.assertEqual(len(payload), 27_210)
        self.assertFalse(payload.endswith(b"\n"))
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            EXPECTED_FIXTURE_SHA256,
        )
        parsed = MODULE.parse_mode8_class_table_identity(
            document,
            EXPECTED_FIXTURE_SHA256,
            "fixture",
        )
        self.assertEqual(parsed.document, document)

    def test_mutation_cannot_reuse_digest(self) -> None:
        document = canonical_fixture()
        mutated = copy.deepcopy(document)
        classes = mutated["classes"]
        assert isinstance(classes, list)
        first = classes[0]
        assert isinstance(first, dict)
        first["block"] = [64, 1, 1]
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            MODULE.parse_mode8_class_table_identity(
                mutated,
                EXPECTED_FIXTURE_SHA256,
                "fixture",
            )

    def test_unused_class_fails_closed(self) -> None:
        document = canonical_fixture()
        records = document["k_records"]
        assert isinstance(records, list)
        for value in records:
            assert isinstance(value, dict)
            pv = value["pv"]
            assert isinstance(pv, dict)
            if pv["class_index"] == 13:
                pv["class_index"] = 12
        digest = hashlib.sha256(
            MODULE.canonical_class_table_bytes(document)
        ).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "unused kernel class"):
            MODULE.parse_mode8_class_table_identity(
                document,
                digest,
                "fixture",
            )


if __name__ == "__main__":
    unittest.main()
