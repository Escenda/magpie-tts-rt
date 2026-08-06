#!/usr/bin/env python3
"""Synchronize generated JSON contracts with the canonical NanoCodec registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanocodec_contract import CANONICAL_STATE_BINDINGS


REQUIRED_BINDING_FIELDS = [
    "logical_name",
    "dtype",
    "shape",
    "initial_output_binding",
    "steady_input_binding",
    "steady_output_binding",
    "tail_input_binding",
    "tail_output_binding",
]


def binding_schema(binding) -> dict:
    record = binding.manifest_record()
    return {
        "type": "object",
        "additionalProperties": False,
        "required": REQUIRED_BINDING_FIELDS,
        "properties": {
            key: {"const": value}
            for key, value in record.items()
        },
    }


def role_condition(engine_rule: dict) -> str | None:
    condition = engine_rule.get("if")
    if not isinstance(condition, dict):
        return None
    role = (
        condition.get("properties", {})
        .get("role", {})
        .get("const")
    )
    return role if isinstance(role, str) else None


def synchronize_schema(schema: dict) -> None:
    bindings = list(CANONICAL_STATE_BINDINGS)
    schema["$defs"]["codec"]["properties"]["state_bindings"] = {
        "type": "array",
        "minItems": len(bindings),
        "maxItems": len(bindings),
        "prefixItems": [
            binding_schema(binding) for binding in bindings
        ],
        "items": False,
    }
    engine_rules = schema["$defs"]["engine"]["allOf"]
    counts = {
        "nanocodec_initial_4": (1, 99),
        "nanocodec_steady_8": (98, 99),
        "nanocodec_tail_1_8": (98, 99),
    }
    found = set()
    for rule in engine_rules:
        role = role_condition(rule)
        if role not in counts:
            continue
        found.add(role)
        input_count, output_count = counts[role]
        properties = rule["then"]["properties"]
        properties["inputs"]["minItems"] = input_count
        properties["inputs"]["maxItems"] = input_count
        properties["outputs"]["minItems"] = output_count
        properties["outputs"]["maxItems"] = output_count
    if found != set(counts):
        raise RuntimeError(
            f"schema NanoCodec role rules differ: {sorted(found)}"
        )


def tensor(name: str, dtype: str, shape: list[int]) -> dict:
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "location": "device",
        "shape_inference_io": False,
    }


def synchronize_fixture(fixture: dict) -> None:
    bindings = list(CANONICAL_STATE_BINDINGS)
    fixture["codec"]["state_bindings"] = [
        binding.manifest_record() for binding in bindings
    ]
    engines = {
        engine["role"]: engine for engine in fixture["engines"]
    }
    initial = engines["nanocodec_initial_4"]
    initial["inputs"] = [
        tensor("codec_tokens", "int64", [1, 8, 4]),
    ]
    initial["outputs"] = [
        tensor("pcm", "fp32", [1, 4096]),
        tensor("valid_sample_length", "int64", [1]),
        *[
            tensor(
                binding.initial_output_binding,
                "fp32",
                list(binding.shape),
            )
            for binding in bindings
        ],
    ]
    steady = engines["nanocodec_steady_8"]
    steady["inputs"] = [
        tensor("codec_tokens", "int64", [1, 8, 8]),
        *[
            tensor(
                binding.steady_input_binding,
                "fp32",
                list(binding.shape),
            )
            for binding in bindings
        ],
    ]
    steady["outputs"] = [
        tensor("pcm", "fp32", [1, 8192]),
        tensor("valid_sample_length", "int64", [1]),
        *[
            tensor(
                binding.steady_output_binding,
                "fp32",
                list(binding.shape),
            )
            for binding in bindings
        ],
    ]
    tail = engines["nanocodec_tail_1_8"]
    tail["inputs"] = [
        tensor("codec_tokens", "int64", [1, 8, -1]),
        *[
            tensor(
                binding.tail_input_binding,
                "fp32",
                list(binding.shape),
            )
            for binding in bindings
        ],
    ]
    tail["outputs"] = [
        tensor("pcm", "fp32", [1, -1]),
        tensor("valid_sample_length", "int64", [1]),
        *[
            tensor(
                binding.tail_output_binding,
                "fp32",
                list(binding.shape),
            )
            for binding in bindings
        ],
    ]


def write_json(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = json.loads(
        args.schema.resolve(strict=True).read_text(encoding="utf-8")
    )
    fixture = json.loads(
        args.fixture.resolve(strict=True).read_text(encoding="utf-8")
    )
    synchronize_schema(schema)
    synchronize_fixture(fixture)
    write_json(args.schema, schema)
    write_json(args.fixture, fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
