from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


EXPORT_TOOLS = Path(__file__).resolve().parents[2] / "tools" / "export"
if str(EXPORT_TOOLS) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS))

from export_main_decoder import expected_plan_contract, grouped_parity, inspect_plan


class FakeTensorRt:
    __version__ = "10.16-test"

    class DataType:
        BF16 = "bf16-enum"
        BOOL = "bool-enum"
        INT32 = "int32-enum"
        INT64 = "int64-enum"

    class TensorLocation:
        DEVICE = "device-enum"
        HOST = "host-enum"

    class TensorIOMode:
        INPUT = "input-enum"
        OUTPUT = "output-enum"

    class Logger:
        ERROR = "error"

        def __init__(self, severity: str) -> None:
            self.severity = severity

    engine = None

    class Runtime:
        def __init__(self, logger: FakeTensorRt.Logger) -> None:
            self.logger = logger

        def deserialize_cuda_engine(self, payload: bytes):
            if payload != b"plan":
                raise AssertionError("unexpected fake plan payload")
            return FakeTensorRt.engine


class FakeStepEngine:
    num_optimization_profiles = 1

    def __init__(self, *, status_dtype: str) -> None:
        self.contract = expected_plan_contract("step")
        self.names = list(self.contract)
        self.status_dtype = status_dtype
        self.num_io_tensors = len(self.names)

    def get_tensor_name(self, index: int) -> str:
        return self.names[index]

    def get_tensor_dtype(self, name: str) -> str:
        if name in {"execution_status_in", "execution_status_out"}:
            return self.status_dtype
        dtype = self.contract[name][0]
        return {
            "bf16": FakeTensorRt.DataType.BF16,
            "bool": FakeTensorRt.DataType.BOOL,
            "int32": FakeTensorRt.DataType.INT32,
            "int64": FakeTensorRt.DataType.INT64,
        }[dtype]

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.contract[name][1]

    def get_tensor_mode(self, name: str) -> str:
        return {
            "input": FakeTensorRt.TensorIOMode.INPUT,
            "output": FakeTensorRt.TensorIOMode.OUTPUT,
        }[self.contract[name][2]]

    def get_tensor_location(self, name: str) -> str:
        return FakeTensorRt.TensorLocation.DEVICE

    def is_shape_inference_io(self, name: str) -> bool:
        return False

    def get_tensor_profile_shape(
        self,
        name: str,
        profile_index: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        if profile_index != 0:
            raise AssertionError("unexpected optimization profile")
        shape = self.contract[name][1]
        return tuple(
            tuple(1 if value == -1 else value for value in shape)
            for _ in range(3)
        )


class MainDecoderPlanInspectionTests(unittest.TestCase):
    def inspect(self, status_dtype: str) -> dict:
        FakeTensorRt.engine = FakeStepEngine(status_dtype=status_dtype)
        with tempfile.TemporaryDirectory() as temporary_directory:
            plan = Path(temporary_directory) / "step.plan"
            plan.write_bytes(b"plan")
            return inspect_plan(FakeTensorRt, "step", plan)

    def test_scalar_device_int32_execution_status_is_accepted(self) -> None:
        inspection = self.inspect(FakeTensorRt.DataType.INT32)
        status = {
            record["name"]: record
            for record in inspection["tensors"]
            if record["name"].startswith("execution_status_")
        }
        self.assertEqual(set(status), {"execution_status_in", "execution_status_out"})
        for record in status.values():
            self.assertEqual(record["dtype"], "int32")
            self.assertEqual(record["shape"], [])
            self.assertEqual(record["location"], "device")
            self.assertFalse(record["shape_inference_io"])

    def test_wrong_execution_status_dtype_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "execution_status_in mismatch",
        ):
            self.inspect(FakeTensorRt.DataType.INT64)

    def test_zero_scalar_execution_status_parity_is_exact(self) -> None:
        metrics = grouped_parity(
            "step",
            {"execution_status_out": torch.zeros((), dtype=torch.int32)},
            {"execution_status_out": torch.zeros((), dtype=torch.int32)},
        )
        self.assertEqual(
            metrics["execution_status"],
            {"elements": 1, "mismatch_count": 0, "mismatch_ratio": 0.0},
        )

    def test_nonzero_execution_status_parity_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exact zero scalar"):
            grouped_parity(
                "step",
                {"execution_status_out": torch.ones((), dtype=torch.int32)},
                {"execution_status_out": torch.zeros((), dtype=torch.int32)},
            )

    def test_execution_status_dtype_or_shape_change_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "integer parity contract"):
            grouped_parity(
                "step",
                {"execution_status_out": torch.zeros((1,), dtype=torch.int32)},
                {"execution_status_out": torch.zeros((), dtype=torch.int32)},
            )


if __name__ == "__main__":
    unittest.main()
