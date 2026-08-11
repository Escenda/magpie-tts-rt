#!/usr/bin/env python3
"""Authenticate the stable mode-8 cuBLAS kernel-class table C ABI."""

from __future__ import annotations

import ctypes
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from cublas_runtime_identity import (
    CublasRuntimeIdentity,
    parse_cublas_runtime_identity,
)
from cuda_runtime_identity import (
    CudaRuntimeIdentity,
    parse_cuda_runtime_identity,
)


type JsonValue = None | bool | int | float | str | list[JsonValue] | JsonObject
type JsonObject = dict[str, JsonValue]

ABI_VERSION = 1
CLASS_COUNT = 21
QK_CLASS_COUNT = 7
PV_CLASS_COUNT = 14
K_COUNT = 249
MINIMUM_K = 219
MAXIMUM_K = 467
FUNCTION_NAME_CAPACITY = 256
STATUS_OK = 0
OPERATION_QK = 1
OPERATION_PV = 2
PARAMETER_KERNEL_PARAMS = 1
PARAMETER_EXTRA = 2


class _ClassRecord(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("operation", ctypes.c_uint32),
        ("class_index", ctypes.c_uint32),
        ("parameter_transport", ctypes.c_uint32),
        ("block_x", ctypes.c_uint32),
        ("block_y", ctypes.c_uint32),
        ("block_z", ctypes.c_uint32),
        ("shared_memory_bytes", ctypes.c_uint32),
        ("reserved_0", ctypes.c_uint32),
        ("parameter_offset", ctypes.c_uint64),
        ("parameter_size", ctypes.c_uint64),
        ("function_name", ctypes.c_char * FUNCTION_NAME_CAPACITY),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _KRecord(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("active_k", ctypes.c_int32),
        ("qk_class_index", ctypes.c_int32),
        ("qk_grid_x", ctypes.c_uint32),
        ("qk_grid_y", ctypes.c_uint32),
        ("qk_grid_z", ctypes.c_uint32),
        ("pv_class_index", ctypes.c_int32),
        ("pv_grid_x", ctypes.c_uint32),
        ("pv_grid_y", ctypes.c_uint32),
        ("pv_grid_z", ctypes.c_uint32),
        ("reserved_0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _ClassTable(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("class_count", ctypes.c_uint32),
        ("k_count", ctypes.c_uint32),
        ("classes", _ClassRecord * CLASS_COUNT),
        ("k_records", _KRecord * K_COUNT),
        ("reserved", ctypes.c_uint64 * 4),
    ]


@dataclass(frozen=True)
class Mode8ClassTableIdentity:
    document: JsonObject
    sha256: str


@dataclass(frozen=True)
class AuthenticatedMode8ComponentReceipt:
    receipt_sha256: str
    plugin_sha256: str
    plugin_size_bytes: int
    cuda_identity: CudaRuntimeIdentity
    cublas_identity: CublasRuntimeIdentity
    class_table: Mode8ClassTableIdentity


def canonical_class_table_bytes(document: JsonObject) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _require_integer(value: JsonValue | None, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def _require_positive_triplet(
    value: JsonValue | None,
    label: str,
) -> list[JsonValue]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
    ):
        raise RuntimeError(f"{label} must contain three positive integers")
    return value


def _validate_class_table_document(document: JsonObject, label: str) -> None:
    expected_keys = {
        "schema_version",
        "active_k_range",
        "qk_class_count",
        "pv_class_count",
        "classes",
        "k_records",
    }
    if set(document) != expected_keys:
        raise RuntimeError(f"{label} keys are not canonical")
    classes = document.get("classes")
    k_records = document.get("k_records")
    if (
        document.get("schema_version") != 1
        or document.get("active_k_range") != [MINIMUM_K, MAXIMUM_K]
        or document.get("qk_class_count") != QK_CLASS_COUNT
        or document.get("pv_class_count") != PV_CLASS_COUNT
        or not isinstance(classes, list)
        or len(classes) != CLASS_COUNT
        or not isinstance(k_records, list)
        or len(k_records) != K_COUNT
    ):
        raise RuntimeError(f"{label} dimensions are not canonical")
    class_keys = {
        "operation",
        "class_index",
        "function_name",
        "parameter_transport",
        "block",
        "shared_memory_bytes",
        "parameter_offset",
        "parameter_size",
    }
    for array_index, value in enumerate(classes):
        item_label = f"{label}.classes[{array_index}]"
        if not isinstance(value, dict) or set(value) != class_keys:
            raise RuntimeError(f"{item_label} keys are not canonical")
        expected_operation = "qk" if array_index < QK_CLASS_COUNT else "pv"
        expected_index = (
            array_index
            if expected_operation == "qk"
            else array_index - QK_CLASS_COUNT
        )
        function_name = value.get("function_name")
        if (
            value.get("operation") != expected_operation
            or value.get("class_index") != expected_index
            or not isinstance(function_name, str)
            or not function_name
            or len(function_name.encode("ascii", errors="ignore")) != len(function_name)
            or any(
                ord(character) < 0x21 or ord(character) > 0x7E
                for character in function_name
            )
            or value.get("parameter_transport")
            not in ("kernel_params", "extra")
        ):
            raise RuntimeError(f"{item_label} identity is not canonical")
        _require_positive_triplet(value.get("block"), f"{item_label}.block")
        shared = _require_integer(
            value.get("shared_memory_bytes"),
            f"{item_label}.shared_memory_bytes",
        )
        offset = _require_integer(
            value.get("parameter_offset"),
            f"{item_label}.parameter_offset",
        )
        size = _require_integer(
            value.get("parameter_size"),
            f"{item_label}.parameter_size",
        )
        if shared < 0 or offset < 0 or size <= 0:
            raise RuntimeError(f"{item_label} launch values are invalid")
    k_keys = {"active_k", "qk", "pv"}
    operation_keys = {"class_index", "grid"}
    used_qk: set[int] = set()
    used_pv: set[int] = set()
    for array_index, value in enumerate(k_records):
        item_label = f"{label}.k_records[{array_index}]"
        if not isinstance(value, dict) or set(value) != k_keys:
            raise RuntimeError(f"{item_label} keys are not canonical")
        if value.get("active_k") != MINIMUM_K + array_index:
            raise RuntimeError(f"{item_label}.active_k is not canonical")
        for operation, class_count, used in (
            ("qk", QK_CLASS_COUNT, used_qk),
            ("pv", PV_CLASS_COUNT, used_pv),
        ):
            operation_value = value.get(operation)
            if (
                not isinstance(operation_value, dict)
                or set(operation_value) != operation_keys
            ):
                raise RuntimeError(
                    f"{item_label}.{operation} keys are not canonical"
                )
            class_index = _require_integer(
                operation_value.get("class_index"),
                f"{item_label}.{operation}.class_index",
            )
            if not 0 <= class_index < class_count:
                raise RuntimeError(
                    f"{item_label}.{operation}.class_index is invalid"
                )
            _require_positive_triplet(
                operation_value.get("grid"),
                f"{item_label}.{operation}.grid",
            )
            used.add(class_index)
    if used_qk != set(range(QK_CLASS_COUNT)) or used_pv != set(range(PV_CLASS_COUNT)):
        raise RuntimeError(f"{label} contains an unused kernel class")


def _require_zero_reserved(
    values: ctypes.Array[ctypes.c_uint64],
    label: str,
) -> None:
    if any(int(value) != 0 for value in values):
        raise RuntimeError(f"{label} contains nonzero reserved fields")


def _decode_function_name(record: _ClassRecord, label: str) -> str:
    payload = ctypes.string_at(
        ctypes.addressof(record) + _ClassRecord.function_name.offset,
        FUNCTION_NAME_CAPACITY,
    )
    terminator = payload.find(b"\0")
    if terminator < 1:
        raise RuntimeError(f"{label}.function_name is not NUL-terminated")
    if any(payload[terminator + 1 :]):
        raise RuntimeError(f"{label}.function_name has nonzero trailing bytes")
    try:
        value = payload[:terminator].decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{label}.function_name is not ASCII") from error
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise RuntimeError(f"{label}.function_name has unsafe characters")
    return value


def _class_document(
    record: _ClassRecord,
    *,
    array_index: int,
) -> JsonObject:
    label = f"class_table.classes[{array_index}]"
    if record.struct_size != ctypes.sizeof(_ClassRecord):
        raise RuntimeError(f"{label}.struct_size mismatch")
    if record.abi_version != ABI_VERSION:
        raise RuntimeError(f"{label}.abi_version mismatch")
    expected_operation = (
        OPERATION_QK if array_index < QK_CLASS_COUNT else OPERATION_PV
    )
    expected_class_index = (
        array_index
        if expected_operation == OPERATION_QK
        else array_index - QK_CLASS_COUNT
    )
    if (
        record.operation != expected_operation
        or record.class_index != expected_class_index
    ):
        raise RuntimeError(f"{label} operation/class ordering mismatch")
    operation = "qk" if record.operation == OPERATION_QK else "pv"
    transport_names = {
        PARAMETER_KERNEL_PARAMS: "kernel_params",
        PARAMETER_EXTRA: "extra",
    }
    transport = transport_names.get(record.parameter_transport)
    if transport is None:
        raise RuntimeError(f"{label}.parameter_transport is invalid")
    block = [int(record.block_x), int(record.block_y), int(record.block_z)]
    if any(value <= 0 for value in block) or record.parameter_size <= 0:
        raise RuntimeError(f"{label} has an empty launch contract")
    if record.reserved_0 != 0:
        raise RuntimeError(f"{label}.reserved_0 is nonzero")
    _require_zero_reserved(record.reserved, f"{label}.reserved")
    return {
        "operation": operation,
        "class_index": int(record.class_index),
        "function_name": _decode_function_name(record, label),
        "parameter_transport": transport,
        "block": block,
        "shared_memory_bytes": int(record.shared_memory_bytes),
        "parameter_offset": int(record.parameter_offset),
        "parameter_size": int(record.parameter_size),
    }


def _k_document(record: _KRecord, *, array_index: int) -> JsonObject:
    label = f"class_table.k_records[{array_index}]"
    if record.struct_size != ctypes.sizeof(_KRecord):
        raise RuntimeError(f"{label}.struct_size mismatch")
    if record.abi_version != ABI_VERSION:
        raise RuntimeError(f"{label}.abi_version mismatch")
    expected_k = MINIMUM_K + array_index
    if record.active_k != expected_k:
        raise RuntimeError(f"{label}.active_k is not canonical")
    if not 0 <= record.qk_class_index < QK_CLASS_COUNT:
        raise RuntimeError(f"{label}.qk_class_index is invalid")
    if not 0 <= record.pv_class_index < PV_CLASS_COUNT:
        raise RuntimeError(f"{label}.pv_class_index is invalid")
    qk_grid = [int(record.qk_grid_x), int(record.qk_grid_y), int(record.qk_grid_z)]
    pv_grid = [int(record.pv_grid_x), int(record.pv_grid_y), int(record.pv_grid_z)]
    if any(value <= 0 for value in (*qk_grid, *pv_grid)):
        raise RuntimeError(f"{label} has an empty grid dimension")
    if record.reserved_0 != 0:
        raise RuntimeError(f"{label}.reserved_0 is nonzero")
    _require_zero_reserved(record.reserved, f"{label}.reserved")
    return {
        "active_k": int(record.active_k),
        "qk": {
            "class_index": int(record.qk_class_index),
            "grid": qk_grid,
        },
        "pv": {
            "class_index": int(record.pv_class_index),
            "grid": pv_grid,
        },
    }


def collect_mode8_class_table_identity(
    plugin_library: ctypes.CDLL,
) -> Mode8ClassTableIdentity:
    try:
        getter = plugin_library.mtt_plugin_get_main_device_position_class_table_v1
    except AttributeError as error:
        raise RuntimeError(
            "plugin does not export the mode-8 class-table C ABI"
        ) from error
    getter.argtypes = [ctypes.POINTER(_ClassTable)]
    getter.restype = ctypes.c_int32
    table = _ClassTable()
    table.struct_size = ctypes.sizeof(_ClassTable)
    table.abi_version = ABI_VERSION
    status = int(getter(ctypes.byref(table)))
    if status != STATUS_OK:
        raise RuntimeError(
            "mode-8 class-table C ABI is not ready or conflicted: "
            f"status={status}"
        )
    if (
        table.struct_size != ctypes.sizeof(_ClassTable)
        or table.abi_version != ABI_VERSION
        or table.class_count != CLASS_COUNT
        or table.k_count != K_COUNT
    ):
        raise RuntimeError("mode-8 class-table header mismatch")
    _require_zero_reserved(table.reserved, "class_table.reserved")
    classes = [
        _class_document(record, array_index=index)
        for index, record in enumerate(table.classes)
    ]
    k_records = [
        _k_document(record, array_index=index)
        for index, record in enumerate(table.k_records)
    ]
    used_qk = {
        int(record["qk"]["class_index"])
        for record in k_records
        if isinstance(record["qk"], dict)
    }
    used_pv = {
        int(record["pv"]["class_index"])
        for record in k_records
        if isinstance(record["pv"], dict)
    }
    if used_qk != set(range(QK_CLASS_COUNT)) or used_pv != set(range(PV_CLASS_COUNT)):
        raise RuntimeError("mode-8 class table contains an unused kernel class")
    document: JsonObject = {
        "schema_version": 1,
        "active_k_range": [MINIMUM_K, MAXIMUM_K],
        "qk_class_count": QK_CLASS_COUNT,
        "pv_class_count": PV_CLASS_COUNT,
        "classes": classes,
        "k_records": k_records,
    }
    _validate_class_table_document(document, "class_table")
    payload = canonical_class_table_bytes(document)
    return Mode8ClassTableIdentity(
        document=document,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def parse_mode8_class_table_identity(
    value: JsonValue | None,
    digest: JsonValue | None,
    label: str,
) -> Mode8ClassTableIdentity:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError(f"{label}_sha256 must be a lowercase SHA-256")
    _validate_class_table_document(value, label)
    actual_digest = hashlib.sha256(canonical_class_table_bytes(value)).hexdigest()
    if actual_digest != digest:
        raise RuntimeError(f"{label} digest mismatch")
    return Mode8ClassTableIdentity(document=value, sha256=digest)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(
    pairs: list[tuple[str, JsonValue]],
) -> JsonObject:
    document: JsonObject = {}
    for key, value in pairs:
        if key in document:
            raise RuntimeError(f"mode-8 receipt duplicates JSON key {key!r}")
        document[key] = value
    return document


def _require_mapping(value: JsonValue | None, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _require_sha256(value: JsonValue | None, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{label} must be a lowercase SHA-256")
    return value


def authenticate_mode8_component_receipt(
    path: Path,
) -> AuthenticatedMode8ComponentReceipt:
    if path.is_symlink():
        raise RuntimeError("mode-8 validation root must not be a symlink")
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("mode-8 validation root must be a directory")
    receipt_name = "validation-receipt.json"
    checksum_name = "validation-receipt.json.sha256"
    receipt_path = root / receipt_name
    checksum_path = root / checksum_name
    if receipt_path.is_symlink() or checksum_path.is_symlink():
        raise RuntimeError("mode-8 receipt files must not be symlinks")
    payload = receipt_path.read_bytes()
    receipt_sha256 = hashlib.sha256(payload).hexdigest()
    if checksum_path.read_bytes() != (
        f"{receipt_sha256}  {receipt_name}\n".encode("ascii")
    ):
        raise RuntimeError("mode-8 receipt checksum mismatch")
    raw_document = json.loads(
        payload,
        object_pairs_hook=_reject_duplicate_pairs,
    )
    document = _require_mapping(raw_document, "mode-8 receipt")
    if set(document) != {
        "schema_version",
        "artifact_role",
        "created_at_utc",
        "status",
        "reason",
        "contract",
        "source",
        "class_table",
        "class_table_sha256",
        "artifacts",
        "component",
        "cuda_graph",
        "execution_status",
        "runtime",
    }:
        raise RuntimeError("mode-8 receipt keys are not canonical")
    if document.get("schema_version") != 1:
        raise RuntimeError("unsupported mode-8 receipt schema")
    if (
        document.get("artifact_role")
        != "main_decoder_device_position_mode8_validation"
        or document.get("status") != "accepted"
    ):
        raise RuntimeError("mode-8 component receipt is not accepted")
    source = _require_mapping(document.get("source"), "mode-8 receipt.source")
    if set(source) != {
        "validator_sha256",
        "plugin_sha256",
        "plugin_size_bytes",
    }:
        raise RuntimeError("mode-8 receipt.source keys are not canonical")
    plugin_sha256 = _require_sha256(
        source.get("plugin_sha256"),
        "mode-8 receipt.source.plugin_sha256",
    )
    plugin_size_bytes = _require_integer(
        source.get("plugin_size_bytes"),
        "mode-8 receipt.source.plugin_size_bytes",
    )
    if plugin_size_bytes <= 0:
        raise RuntimeError("mode-8 plugin size must be positive")
    expected_validator = (
        Path(__file__).resolve().parent
        / "validate_main_step_device_position_plugin.py"
    ).resolve(strict=True)
    if _require_sha256(
        source.get("validator_sha256"),
        "mode-8 receipt.source.validator_sha256",
    ) != _sha256_file(expected_validator):
        raise RuntimeError("mode-8 receipt was produced by another validator")
    contract = _require_mapping(
        document.get("contract"),
        "mode-8 receipt.contract",
    )
    if contract != {
        "fixed_io": True,
        "position_location": "DEVICE",
        "position_is_shape_inference_io": False,
        "first_position": 218,
        "last_position": 466,
        "execution_status_dtype": "INT32",
        "execution_status_location": "DEVICE",
        "execution_status_is_shape_inference_io": False,
        "execution_status_sticky_first_error": True,
        "component_layer_index": 0,
    }:
        raise RuntimeError("mode-8 receipt contract is not canonical")
    component = _require_mapping(
        document.get("component"),
        "mode-8 receipt.component",
    )
    if set(component) != {
        "seeds",
        "mask_cases",
        "position_range",
        "case_count",
        "total_elements",
        "bit_mismatch_count",
        "bit_mismatch_ratio",
        "positions_with_mismatch",
        "position_mismatch_count",
        "worst_max_abs",
        "first_mismatch",
        "execution_status_mismatch_count",
        "bit_exact",
    }:
        raise RuntimeError("mode-8 receipt.component keys are not canonical")
    if (
        component.get("seeds") != [2026080601, 2026080602, 2026080603]
        or component.get("mask_cases")
        != ["all_valid", "staggered_holes", "batch_split_holes"]
        or component.get("position_range") != [218, 466]
        or component.get("bit_exact") is not True
        or component.get("case_count") != 2241
        or component.get("total_elements") != 3_442_176
        or component.get("bit_mismatch_count") != 0
        or component.get("bit_mismatch_ratio") != 0.0
        or component.get("positions_with_mismatch") != []
        or component.get("position_mismatch_count") != 0
        or component.get("worst_max_abs") != 0.0
        or component.get("first_mismatch") is not None
        or component.get("execution_status_mismatch_count") != 0
    ):
        raise RuntimeError("mode-8 component evidence is not exact")
    cuda_graph = _require_mapping(
        document.get("cuda_graph"),
        "mode-8 receipt.cuda_graph",
    )
    if set(cuda_graph) != {
        "single_capture",
        "device_position_changed_between_replays",
        "cases",
        "replay_completed",
    }:
        raise RuntimeError("mode-8 receipt.cuda_graph keys are not canonical")
    graph_cases = cuda_graph.get("cases")
    if (
        cuda_graph.get("single_capture") is not True
        or cuda_graph.get("device_position_changed_between_replays") is not True
        or cuda_graph.get("replay_completed") is not True
        or not isinstance(graph_cases, list)
        or [
            case.get("position") if isinstance(case, dict) else None
            for case in graph_cases
        ]
        != [218, 256, 300, 466]
        or any(
            not isinstance(case, dict)
            or set(case)
            != {
                "position",
                "bit_mismatch_count",
                "execution_status",
                "max_abs",
            }
            or case.get("bit_mismatch_count") != 0
            or case.get("execution_status") != 0
            or case.get("max_abs") != 0.0
            for case in graph_cases
        )
    ):
        raise RuntimeError("mode-8 graph evidence is not exact")
    execution_status = _require_mapping(
        document.get("execution_status"),
        "mode-8 receipt.execution_status",
    )
    if set(execution_status) != {
        "layer_index",
        "invalid_cases",
        "sticky_input_status",
        "sticky_output_status",
        "sticky_first_error_preserved",
    }:
        raise RuntimeError(
            "mode-8 receipt.execution_status keys are not canonical"
        )
    invalid_cases = execution_status.get("invalid_cases")
    expected_invalid_status = 1 << 28
    if (
        execution_status.get("layer_index") != 0
        or invalid_cases
        != [
            {"position": 217, "status": expected_invalid_status},
            {"position": 467, "status": expected_invalid_status},
        ]
        or execution_status.get("sticky_input_status")
        != expected_invalid_status
        or execution_status.get("sticky_output_status")
        != expected_invalid_status
        or execution_status.get("sticky_first_error_preserved") is not True
    ):
        raise RuntimeError(
            "mode-8 execution-status evidence is not canonical"
        )
    runtime = _require_mapping(document.get("runtime"), "mode-8 receipt.runtime")
    if set(runtime) != {
        "cuda",
        "cublas",
        "gpu_name",
        "gpu_compute_capability",
        "torch",
        "torch_cuda_build",
        "tensorrt",
    }:
        raise RuntimeError("mode-8 receipt.runtime keys are not canonical")
    if (
        runtime.get("gpu_name") != "NVIDIA Thor"
        or runtime.get("gpu_compute_capability") != [11, 0]
    ):
        raise RuntimeError("mode-8 receipt was not measured on Thor sm_110")
    cuda_identity = parse_cuda_runtime_identity(
        runtime.get("cuda"),
        "mode-8 receipt.runtime.cuda",
    )
    cublas_identity = parse_cublas_runtime_identity(
        runtime.get("cublas"),
        "mode-8 receipt.runtime.cublas",
    )
    class_table = parse_mode8_class_table_identity(
        document.get("class_table"),
        document.get("class_table_sha256"),
        "mode-8 receipt.class_table",
    )
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise RuntimeError("mode-8 receipt must authenticate exactly one plan")
    artifact = _require_mapping(artifacts[0], "mode-8 receipt.artifacts[0]")
    if set(artifact) != {"path", "sha256", "size_bytes"}:
        raise RuntimeError("mode-8 plan artifact keys are not canonical")
    relative = artifact.get("path")
    if (
        not isinstance(relative, str)
        or relative in {"", ".", ".."}
        or PurePosixPath(relative).is_absolute()
        or PurePosixPath(relative).as_posix() != relative
        or len(PurePosixPath(relative).parts) != 1
    ):
        raise RuntimeError("mode-8 plan artifact path is unsafe")
    plan_path = root / relative
    if plan_path.is_symlink() or not plan_path.resolve(strict=True).is_file():
        raise RuntimeError("mode-8 plan artifact is not a regular file")
    expected_size = _require_integer(
        artifact.get("size_bytes"),
        "mode-8 receipt.artifacts[0].size_bytes",
    )
    if (
        expected_size <= 0
        or plan_path.stat().st_size != expected_size
        or _sha256_file(plan_path)
        != _require_sha256(
            artifact.get("sha256"),
            "mode-8 receipt.artifacts[0].sha256",
        )
    ):
        raise RuntimeError("mode-8 plan artifact identity mismatch")
    actual_entries = {
        entry.relative_to(root).as_posix()
        for entry in root.rglob("*")
        if entry.is_file()
    }
    expected_entries = {receipt_name, checksum_name, relative}
    if actual_entries != expected_entries or any(
        entry.is_symlink() or not entry.is_file()
        for entry in root.rglob("*")
    ):
        raise RuntimeError("mode-8 receipt directory entry set mismatch")
    return AuthenticatedMode8ComponentReceipt(
        receipt_sha256=receipt_sha256,
        plugin_sha256=plugin_sha256,
        plugin_size_bytes=plugin_size_bytes,
        cuda_identity=cuda_identity,
        cublas_identity=cublas_identity,
        class_table=class_table,
    )
