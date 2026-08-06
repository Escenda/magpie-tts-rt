from __future__ import annotations

import json
import importlib.util
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/plugins/create_plugin_build_receipt.py"
SCHEMA = ROOT / "schemas/plugin-build-receipt.schema.json"
SPEC = importlib.util.spec_from_file_location("plugin_build_receipt", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PluginBuildReceiptTests(unittest.TestCase):
    def test_normalizes_source_and_build_roots(self) -> None:
        normalized = MODULE.normalize_build_command(
            "/tmp/build/nvcc -I/tmp/source/include -I/tmp/build/deps",
            source_root=Path("/tmp/source"),
            build_root=Path("/tmp/build"),
        )
        self.assertEqual(
            normalized,
            "${BUILD_ROOT}/nvcc -I${SOURCE_ROOT}/include "
            "-I${BUILD_ROOT}/deps",
        )

    def test_records_exact_sources_toolchain_flags_and_elf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            source_root = temporary / "source"
            build_root = temporary / "build"
            tools = temporary / "tools"
            source_root.mkdir()
            build_root.mkdir()
            tools.mkdir()
            for relative in MODULE.SOURCE_PATHS:
                source = source_root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(f"source:{relative}\n", encoding="utf-8")
            plugin_source = source_root / "plugins/local_ar_plugins.cu"
            plugin_source.write_text(
                "#include <stdio.h>\n"
                'int plugin_test(void) { return puts("plugin"); }\n',
                encoding="utf-8",
            )
            plugin = build_root / MODULE.PLUGIN_FILENAME
            subprocess.run(
                [
                    "/usr/bin/cc",
                    "-x",
                    "c",
                    "-shared",
                    "-Wl,-soname,libmagpie_tts_rt_plugins.so.0",
                    str(plugin_source),
                    "-o",
                    str(plugin),
                ],
                check=True,
            )
            fake_cxx = self._version_tool(tools / "cxx", "cxx-v1")
            fake_nvcc = self._version_tool(tools / "nvcc", "nvcc-v1")
            fake_linker = self._version_tool(tools / "ld", "ld-v1")
            fake_cmake = self._version_tool(tools / "cmake", "cmake-v1")
            fake_ninja = tools / "ninja"
            fake_ninja.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = \"--version\" ]; then\n"
                "  printf '%s\\n' 'ninja-v1'\n"
                "  exit 0\n"
                "fi\n"
                f"printf '%s\\n' '{fake_nvcc} "
                f"-I{source_root}/include -I{build_root}/_deps/cutlass "
                "-O3 -DNDEBUG -std=c++20 "
                "\"--generate-code=arch=compute_110,"
                "code=[compute_110,sm_110]\" "
                "-Xcompiler=-fPIC "
                "--frandom-seed=magpie_tts_rt_plugins_v1 "
                "-Xcompiler=-Wall,-Wextra -Xcompiler=-Werror "
                f"-c {plugin_source} -o plugin.o'\n"
                f"printf '%s\\n' ': && {fake_cxx} -Wl,--no-undefined "
                "-Wl,--strip-all "
                f"-Wl,--version-script={source_root}/cmake/"
                "magpie_tts_rt_plugins.map -shared "
                "-Wl,-soname,libmagpie_tts_rt_plugins.so.0 "
                f"-o {MODULE.PLUGIN_FILENAME} plugin.o && :'\n",
                encoding="utf-8",
            )
            fake_ninja.chmod(fake_ninja.stat().st_mode | stat.S_IXUSR)
            (build_root / "CMakeCache.txt").write_text(
                "\n".join(
                    (
                        f"CMAKE_HOME_DIRECTORY:INTERNAL={source_root}",
                        "CMAKE_GENERATOR:INTERNAL=Ninja",
                        "CMAKE_BUILD_TYPE:STRING=Release",
                        "CMAKE_CUDA_ARCHITECTURES:STRING=110",
                        "MAGPIE_TTS_RT_WARNINGS_AS_ERRORS:BOOL=ON",
                        f"CMAKE_CXX_COMPILER:FILEPATH={fake_cxx}",
                        f"CMAKE_CUDA_COMPILER:FILEPATH={fake_nvcc}",
                        f"CMAKE_LINKER:FILEPATH={fake_linker}",
                        f"CMAKE_MAKE_PROGRAM:FILEPATH={fake_ninja}",
                        f"CMAKE_COMMAND:INTERNAL={fake_cmake}",
                        "",
                    )
                ),
                encoding="utf-8",
            )
            output = temporary / "receipt.json"
            subprocess.run(
                [
                    "/usr/bin/python3",
                    str(TOOL),
                    "--plugin",
                    str(plugin),
                    "--source-root",
                    str(source_root),
                    "--build-directory",
                    str(build_root),
                    "--readelf",
                    "/usr/bin/readelf",
                    "--output",
                    str(output),
                ],
                check=True,
            )
            receipt = json.loads(output.read_text(encoding="utf-8"))
            schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
            validator = Draft202012Validator(schema)
            validator.validate(receipt)
            for unsafe in ("../outside", "a/../b", "a\\b", ".", "bad\u0000path"):
                candidate = json.loads(json.dumps(receipt))
                candidate["source"][0]["path"] = unsafe
                self.assertFalse(
                    validator.is_valid(candidate),
                    f"schema accepted unsafe/noncanonical source path {unsafe!r}",
                )
            self.assertEqual(
                [record["path"] for record in receipt["source"]],
                list(MODULE.SOURCE_PATHS),
            )
            self.assertEqual(receipt["toolchain"]["cxx_version"], "cxx-v1")
            self.assertEqual(receipt["toolchain"]["nvcc_version"], "nvcc-v1")
            self.assertEqual(receipt["toolchain"]["linker_version"], "ld-v1")
            self.assertEqual(receipt["toolchain"]["ninja_version"], "ninja-v1")
            self.assertEqual(receipt["toolchain"]["cmake_version"], "cmake-v1")
            self.assertEqual(
                receipt["artifact"]["soname"],
                MODULE.PLUGIN_SONAME,
            )
            self.assertIn(
                "${SOURCE_ROOT}/plugins/local_ar_plugins.cu",
                receipt["build"]["compile_command"],
            )
            self.assertIn(
                "${BUILD_ROOT}/_deps/cutlass",
                receipt["build"]["compile_command"],
            )
            self.assertIn(
                "-Wl,--strip-all",
                receipt["build"]["link_command"],
            )

            repeated = subprocess.run(
                [
                    "/usr/bin/python3",
                    str(TOOL),
                    "--plugin",
                    str(plugin),
                    "--source-root",
                    str(source_root),
                    "--build-directory",
                    str(build_root),
                    "--readelf",
                    "/usr/bin/readelf",
                    "--output",
                    str(output),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(repeated.returncode, 0)

    @staticmethod
    def _version_tool(path: Path, version: str) -> Path:
        path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{version}'\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path


if __name__ == "__main__":
    unittest.main()
