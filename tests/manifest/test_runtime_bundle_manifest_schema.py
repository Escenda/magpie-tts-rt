from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).parents[2]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "runtime-bundle-manifest.schema.json"
FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "manifest" / "fixtures" / "minimal-valid.json"
)

type JsonValue = None | bool | int | float | str | list[JsonValue] | JsonObject
type JsonObject = dict[str, JsonValue]


class RuntimeBundleManifestSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        cls.validator = Draft202012Validator(schema)
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    @staticmethod
    def main_step(document: JsonObject) -> JsonObject:
        engine = next(
            engine
            for engine in document["engines"]
            if isinstance(engine, dict)
            if engine["role"] == "main_decoder_step"
        )
        return engine

    def test_canonical_device_position_fixture_is_valid(self) -> None:
        self.assertEqual(list(self.validator.iter_errors(self.fixture)), [])

    def test_host_position_is_rejected(self) -> None:
        candidate = copy.deepcopy(self.fixture)
        position = self.main_step(candidate)["inputs"][1]
        position["location"] = "host"
        self.assertTrue(list(self.validator.iter_errors(candidate)))

    def test_shape_inference_position_is_rejected(self) -> None:
        candidate = copy.deepcopy(self.fixture)
        position = self.main_step(candidate)["inputs"][1]
        position["shape_inference_io"] = True
        self.assertTrue(list(self.validator.iter_errors(candidate)))

    def test_legacy_position_value_profile_is_rejected(self) -> None:
        candidate = copy.deepcopy(self.fixture)
        profile = self.main_step(candidate)["profiles"][0]
        profile["input_values"] = [
            {
                "tensor_name": "position",
                "min": [218],
                "opt": [342],
                "max": [466],
            }
        ]
        self.assertTrue(list(self.validator.iter_errors(candidate)))

    def test_missing_execution_status_recurrence_is_rejected(self) -> None:
        candidate = copy.deepcopy(self.fixture)
        step = self.main_step(candidate)
        step["inputs"] = [
            value
            for value in step["inputs"]
            if value["name"] != "execution_status_in"
        ]
        step["outputs"] = [
            value
            for value in step["outputs"]
            if value["name"] != "execution_status_out"
        ]
        self.assertTrue(list(self.validator.iter_errors(candidate)))

    def test_execution_status_must_be_device_int32_scalar(self) -> None:
        candidate = copy.deepcopy(self.fixture)
        status = next(
            value
            for value in self.main_step(candidate)["inputs"]
            if value["name"] == "execution_status_in"
        )
        status["dtype"] = "int64"
        status["location"] = "host"
        status["shape"] = [1]
        self.assertTrue(list(self.validator.iter_errors(candidate)))


if __name__ == "__main__":
    unittest.main()
