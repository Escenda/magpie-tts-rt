from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "tools" / "oracle" / "verify_oracle_lock.py"
SPEC = importlib.util.spec_from_file_location("verify_oracle_lock", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class ModelConfigSelectionTests(unittest.TestCase):
    def test_last_tar_member_is_the_active_config(self) -> None:
        configs = [b"version: first\n", b"version: active\n"]
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "model.nemo"
            with tarfile.open(archive_path, "w") as archive:
                for config in configs:
                    member = tarfile.TarInfo("./model_config.yaml")
                    member.size = len(config)
                    archive.addfile(member, io.BytesIO(config))
            MODULE.require_model_configs(
                archive_path,
                [digest(config) for config in configs],
                "last_tar_member",
                digest(configs[-1]),
            )

    def test_reordered_config_occurrences_fail_closed(self) -> None:
        configs = [b"version: first\n", b"version: active\n"]
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "model.nemo"
            with tarfile.open(archive_path, "w") as archive:
                for config in reversed(configs):
                    member = tarfile.TarInfo("./model_config.yaml")
                    member.size = len(config)
                    archive.addfile(member, io.BytesIO(config))
            with self.assertRaisesRegex(RuntimeError, "occurrence mismatch"):
                MODULE.require_model_configs(
                    archive_path,
                    [digest(config) for config in configs],
                    "last_tar_member",
                    digest(configs[-1]),
                )


if __name__ == "__main__":
    unittest.main()
