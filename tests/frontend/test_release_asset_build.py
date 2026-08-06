from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "tools" / "frontend" / "build_pyopenjtalk_release_wheel"
BUILD_REQUIREMENTS = (
    ROOT
    / "tools"
    / "frontend"
    / "requirements-pyopenjtalk-wheel-build-aarch64-cp312.lock"
)
RUNTIME_REQUIREMENTS = (
    ROOT
    / "tools"
    / "frontend"
    / "requirements-aarch64-cp312.lock"
)


def python_heredoc_after(source: str, marker: str) -> str:
    marker_offset = source.index(marker)
    heredoc_offset = source.index("<<'PY'\n", marker_offset) + len("<<'PY'\n")
    heredoc_end = source.index("\nPY\n", heredoc_offset)
    return source[heredoc_offset:heredoc_end]


class ReleaseAssetBuildTests(unittest.TestCase):
    def test_script_is_valid_shell_and_disables_package_indexes(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(BUILD_SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        source = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--no-index", source)
        self.assertIn("--only-binary=:all:", source)
        self.assertIn("--require-hashes", source)
        self.assertIn("--no-build-isolation", source)
        self.assertIn("RENAME_NOREPLACE", source)

    def test_runtime_lock_accepts_only_the_release_wheel(self) -> None:
        build_lock = BUILD_REQUIREMENTS.read_text(encoding="utf-8")
        runtime_lock = RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")

        self.assertIn("setuptools-scm==9.2.2", build_lock)
        self.assertIn(
            "sha256:30e8f84d2ab1ba7cb0e653429b179395d0c33775d54807fc5f1dd6671801aef7",
            build_lock,
        )
        self.assertIn("pyopenjtalk==0.4.1", runtime_lock)
        self.assertIn(
            "sha256:deb7d40a8bc3ecdadafff4dfa40254352a11afe3cf00fda7f87072ea5d0807af",
            runtime_lock,
        )

    def test_release_asset_publication_never_replaces(self) -> None:
        source = BUILD_SCRIPT.read_text(encoding="utf-8")
        publisher = python_heredoc_after(
            source,
            'chmod 0755 "${STAGING_ROOT}"',
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            destination = root / "release"
            first_staging = root / "first"
            first_staging.mkdir()
            (first_staging / "identity").write_text("first", encoding="ascii")

            accepted = subprocess.run(
                ["python3", "-", str(first_staging), str(destination)],
                input=publisher,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(
                (destination / "identity").read_text(encoding="ascii"),
                "first",
            )

            second_staging = root / "second"
            second_staging.mkdir()
            (second_staging / "identity").write_text("second", encoding="ascii")
            rejected = subprocess.run(
                ["python3", "-", str(second_staging), str(destination)],
                input=publisher,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(
                "output already exists",
                rejected.stdout + rejected.stderr,
            )
            self.assertTrue(second_staging.is_dir())
            self.assertEqual(
                (destination / "identity").read_text(encoding="ascii"),
                "first",
            )


if __name__ == "__main__":
    unittest.main()
