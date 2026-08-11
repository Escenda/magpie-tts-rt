#!/usr/bin/env python3
"""Collect and validate the CUDA runtime environment used for acceptance."""

from __future__ import annotations

import ctypes
import re
from dataclasses import dataclass


type JsonValue = None | bool | int | float | str | list[JsonValue] | JsonObject
type JsonObject = dict[str, JsonValue]

CUDA_RUNTIME_SONAME = "libcudart.so.13"
NVML_SONAME = "libnvidia-ml.so.1"


@dataclass(frozen=True)
class CudaRuntimeIdentity:
    cuda_driver_api_version_integer: int
    cuda_runtime_version_integer: int
    nvidia_driver_version: str

    def to_json(self) -> JsonObject:
        return {
            "cuda_driver_api_version_integer": (
                self.cuda_driver_api_version_integer
            ),
            "cuda_runtime_version_integer": self.cuda_runtime_version_integer,
            "nvidia_driver_version": self.nvidia_driver_version,
        }


def _require_positive_integer(value: JsonValue | None, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _require_driver_version(value: JsonValue | None, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value) is None
    ):
        raise RuntimeError(f"{label} must be a numeric NVIDIA driver version")
    return value


def parse_cuda_runtime_identity(
    value: JsonValue | None,
    label: str,
) -> CudaRuntimeIdentity:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    expected_keys = {
        "cuda_driver_api_version_integer",
        "cuda_runtime_version_integer",
        "nvidia_driver_version",
    }
    if set(value) != expected_keys:
        raise RuntimeError(
            f"{label} keys mismatch: expected={sorted(expected_keys)}, "
            f"actual={sorted(value)}"
        )
    return CudaRuntimeIdentity(
        cuda_driver_api_version_integer=_require_positive_integer(
            value.get("cuda_driver_api_version_integer"),
            f"{label}.cuda_driver_api_version_integer",
        ),
        cuda_runtime_version_integer=_require_positive_integer(
            value.get("cuda_runtime_version_integer"),
            f"{label}.cuda_runtime_version_integer",
        ),
        nvidia_driver_version=_require_driver_version(
            value.get("nvidia_driver_version"),
            f"{label}.nvidia_driver_version",
        ),
    )


def _cuda_version(function, operation: str) -> int:
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    version = ctypes.c_int()
    status = int(function(ctypes.byref(version)))
    if status != 0 or version.value <= 0:
        raise RuntimeError(
            f"{operation} failed or returned an invalid version: "
            f"status={status}, version={version.value}"
        )
    return version.value


def _nvidia_driver_version() -> str:
    nvml = ctypes.CDLL(NVML_SONAME)
    nvml.nvmlInit_v2.argtypes = []
    nvml.nvmlInit_v2.restype = ctypes.c_int
    nvml.nvmlSystemGetDriverVersion.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_uint,
    ]
    nvml.nvmlSystemGetDriverVersion.restype = ctypes.c_int
    nvml.nvmlShutdown.argtypes = []
    nvml.nvmlShutdown.restype = ctypes.c_int
    init_status = int(nvml.nvmlInit_v2())
    if init_status != 0:
        raise RuntimeError(f"nvmlInit_v2 failed with status {init_status}")
    active_error: BaseException | None = None
    try:
        buffer = ctypes.create_string_buffer(80)
        status = int(
            nvml.nvmlSystemGetDriverVersion(buffer, len(buffer))
        )
        if status != 0:
            raise RuntimeError(
                "nvmlSystemGetDriverVersion failed with status "
                f"{status}"
            )
        try:
            value = buffer.value.decode("ascii")
        except UnicodeDecodeError as error:
            raise RuntimeError("NVIDIA driver version is not ASCII") from error
        return _require_driver_version(value, "NVIDIA driver version")
    except BaseException as error:
        active_error = error
        raise
    finally:
        shutdown_status = int(nvml.nvmlShutdown())
        if shutdown_status != 0 and active_error is None:
            raise RuntimeError(
                f"nvmlShutdown failed with status {shutdown_status}"
            )


def collect_cuda_runtime_identity() -> CudaRuntimeIdentity:
    runtime = ctypes.CDLL(CUDA_RUNTIME_SONAME)
    return CudaRuntimeIdentity(
        cuda_driver_api_version_integer=_cuda_version(
            runtime.cudaDriverGetVersion,
            "cudaDriverGetVersion",
        ),
        cuda_runtime_version_integer=_cuda_version(
            runtime.cudaRuntimeGetVersion,
            "cudaRuntimeGetVersion",
        ),
        nvidia_driver_version=_nvidia_driver_version(),
    )
