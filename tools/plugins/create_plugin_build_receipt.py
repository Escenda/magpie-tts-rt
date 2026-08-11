#!/usr/bin/env python3
"""Create an immutable build-provenance receipt for the runtime plugin."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_TOOLS = PROJECT_ROOT / "tools" / "export"
if str(EXPORT_TOOLS) not in sys.path:
    sys.path.insert(0, str(EXPORT_TOOLS))

from cublas_runtime_identity import (  # noqa: E402
    collect_cublas_runtime_identity,
)

SCHEMA_VERSION = "magpie-tts-rt.plugin-build.v1"
PLUGIN_FILENAME = "libmagpie_tts_rt_plugins.so.0.1.0"
PLUGIN_SONAME = "libmagpie_tts_rt_plugins.so.0"
CUTLASS_ARCHIVE_SHA256 = (
    "5288044d2d5e81632ac0c812b6b85c744901a7d3fd11c9119f18f71c3cef5f79"
)
SOURCE_PATHS = (
    "CMakeLists.txt",
    "cmake/magpie_tts_rt_plugins.map",
    "include/magpie_tts_rt/magpie_tts_rt_plugin.h",
    "plugins/local_ar_plugins.cu",
    "plugins/local_ar_plugins.hpp",
)
NORMALIZED_SOURCE_ROOT = "${SOURCE_ROOT}"
NORMALIZED_BUILD_ROOT = "${BUILD_ROOT}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_nonsymlink(path: Path, label: str) -> Path:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise RuntimeError(f"{label} path contains a symbolic link: {component}")
    if not absolute.is_file():
        raise RuntimeError(f"{label} must be a regular nonsymlink file: {path}")
    return absolute


def nonsymlink_directory(path: Path, label: str) -> Path:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise RuntimeError(f"{label} path contains a symbolic link: {component}")
    if not absolute.is_dir():
        raise RuntimeError(f"{label} must be a nonsymlink directory: {path}")
    return absolute


def executable_tool(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"{label} must resolve to an executable file: {path}")
    return resolved


def tool_version(path: Path) -> str:
    completed = subprocess.run(
        [str(path), "--version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError(f"tool returned an empty version: {path}")
    return value


def elf_contract(plugin: Path, readelf: Path) -> tuple[str, list[str]]:
    completed = subprocess.run(
        [str(readelf), "-dW", str(plugin)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sonames = re.findall(r"\(SONAME\).*\[([^]]+)\]", completed.stdout)
    if len(sonames) != 1:
        raise RuntimeError("plugin must contain exactly one ELF SONAME")
    needed = sorted(re.findall(r"\(NEEDED\).*\[([^]]+)\]", completed.stdout))
    if not needed:
        raise RuntimeError("plugin must contain at least one ELF dependency")
    return sonames[0], needed


def read_cmake_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    for line in regular_nonsymlink(path, "CMake cache").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        if ":" not in key_and_type:
            continue
        key, _ = key_and_type.split(":", 1)
        if key in cache:
            raise RuntimeError(f"duplicate CMake cache key: {key}")
        cache[key] = value
    return cache


def required_cache_value(cache: dict[str, str], key: str) -> str:
    value = cache.get(key)
    if value is None or not value:
        raise RuntimeError(f"CMake cache is missing {key}")
    return value


def normalize_build_command(
    command: str,
    *,
    source_root: Path,
    build_root: Path,
) -> str:
    if not command:
        raise RuntimeError("build command must not be empty")
    normalized = command.replace(str(build_root), NORMALIZED_BUILD_ROOT)
    normalized = normalized.replace(str(source_root), NORMALIZED_SOURCE_ROOT)
    if str(build_root) in normalized or str(source_root) in normalized:
        raise RuntimeError("build command path normalization failed")
    return normalized


def plugin_build_commands(
    *,
    ninja: Path,
    source_root: Path,
    build_root: Path,
) -> tuple[str, str]:
    completed = subprocess.run(
        [
            str(ninja),
            "-C",
            str(build_root),
            "-t",
            "commands",
            "magpie_tts_rt_plugins",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = tuple(line for line in completed.stdout.splitlines() if line)
    compile_candidates = tuple(
        line
        for line in lines
        if "plugins/local_ar_plugins.cu" in line and " -c " in line
    )
    link_candidates = tuple(
        line
        for line in lines
        if " -shared " in line and f"-o {PLUGIN_FILENAME}" in line
    )
    if len(compile_candidates) != 1 or len(link_candidates) != 1:
        raise RuntimeError(
            "Ninja must expose exactly one plugin compile and one plugin link command"
        )
    compile_command = normalize_build_command(
        compile_candidates[0],
        source_root=source_root,
        build_root=build_root,
    )
    link_command = normalize_build_command(
        link_candidates[0],
        source_root=source_root,
        build_root=build_root,
    )
    required_compile_fragments = (
        f"-c {NORMALIZED_SOURCE_ROOT}/plugins/local_ar_plugins.cu",
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
        "arch=compute_110,code=[compute_110,sm_110]",
        "-Xcompiler=-fPIC",
        "--frandom-seed=magpie_tts_rt_plugins_v1",
        "-Xcompiler=-Wall,-Wextra",
        "-Xcompiler=-Werror",
    )
    required_link_fragments = (
        "-Wl,--no-undefined",
        "-Wl,--strip-all",
        (
            "-Wl,--version-script="
            f"{NORMALIZED_SOURCE_ROOT}/cmake/magpie_tts_rt_plugins.map"
        ),
        "-shared",
        "-Wl,-soname,libmagpie_tts_rt_plugins.so.0",
        f"-o {PLUGIN_FILENAME}",
    )
    for fragment in required_compile_fragments:
        if fragment not in compile_command:
            raise RuntimeError(
                f"plugin compile command is missing required fragment: {fragment}"
            )
    for fragment in required_link_fragments:
        if fragment not in link_command:
            raise RuntimeError(
                f"plugin link command is missing required fragment: {fragment}"
            )
    return compile_command, link_command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--build-directory", required=True, type=Path)
    parser.add_argument("--readelf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    plugin = regular_nonsymlink(args.plugin, "plugin")
    source_root = nonsymlink_directory(args.source_root, "source root")
    build_root = nonsymlink_directory(args.build_directory, "build directory")
    cache = read_cmake_cache(build_root / "CMakeCache.txt")
    if Path(required_cache_value(cache, "CMAKE_HOME_DIRECTORY")) != source_root:
        raise RuntimeError("CMake cache source root does not match --source-root")
    if required_cache_value(cache, "CMAKE_GENERATOR") != "Ninja":
        raise RuntimeError("plugin provenance requires the Ninja generator")
    if required_cache_value(cache, "CMAKE_BUILD_TYPE") != "Release":
        raise RuntimeError("CMake build type must be exactly Release")
    if required_cache_value(cache, "CMAKE_CUDA_ARCHITECTURES") != "110":
        raise RuntimeError("CMake CUDA architecture must be exactly 110")
    if required_cache_value(cache, "MAGPIE_TTS_RT_WARNINGS_AS_ERRORS") != "ON":
        raise RuntimeError("plugin warnings-as-errors policy must be ON")

    sources = []
    for relative in SOURCE_PATHS:
        source = regular_nonsymlink(source_root / relative, "plugin source")
        sources.append(
            {
                "path": relative,
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )

    cxx = executable_tool(
        Path(required_cache_value(cache, "CMAKE_CXX_COMPILER")),
        "cxx",
    )
    nvcc = executable_tool(
        Path(required_cache_value(cache, "CMAKE_CUDA_COMPILER")),
        "nvcc",
    )
    linker = executable_tool(
        Path(required_cache_value(cache, "CMAKE_LINKER")),
        "linker",
    )
    ninja = executable_tool(
        Path(required_cache_value(cache, "CMAKE_MAKE_PROGRAM")),
        "ninja",
    )
    cmake = executable_tool(
        Path(required_cache_value(cache, "CMAKE_COMMAND")),
        "cmake",
    )
    readelf = executable_tool(args.readelf, "readelf")
    compile_command, link_command = plugin_build_commands(
        ninja=ninja,
        source_root=source_root,
        build_root=build_root,
    )
    built_plugins = tuple(
        candidate
        for candidate in build_root.glob("libmagpie_tts_rt_plugins.so.*.*.*")
        if candidate.is_file() and not candidate.is_symlink()
    )
    if len(built_plugins) != 1 or built_plugins[0].name != PLUGIN_FILENAME:
        raise RuntimeError("build directory must contain exactly one versioned plugin")
    built_plugin = regular_nonsymlink(built_plugins[0], "built plugin")
    if (
        built_plugin.stat().st_size != plugin.stat().st_size
        or sha256_file(built_plugin) != sha256_file(plugin)
    ):
        raise RuntimeError("plugin does not match the Ninja build artifact")
    soname, needed = elf_contract(plugin, readelf)
    if plugin.name != PLUGIN_FILENAME:
        raise RuntimeError(f"plugin filename must be exactly {PLUGIN_FILENAME}")
    if soname != PLUGIN_SONAME:
        raise RuntimeError("plugin has the wrong SONAME")
    plugin_library = ctypes.CDLL(str(plugin), mode=ctypes.RTLD_LOCAL)
    cublas_identity = collect_cublas_runtime_identity(plugin_library)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "recorded",
        "artifact": {
            "filename": plugin.name,
            "size_bytes": plugin.stat().st_size,
            "sha256": sha256_file(plugin),
            "soname": soname,
            "needed": needed,
        },
        "source": sources,
        "toolchain": {
            "cxx_path": str(cxx),
            "cxx_version": tool_version(cxx),
            "nvcc_path": str(nvcc),
            "nvcc_version": tool_version(nvcc),
            "linker_path": str(linker),
            "linker_version": tool_version(linker),
            "readelf_path": str(readelf),
            "ninja_path": str(ninja),
            "ninja_version": tool_version(ninja),
            "cmake_path": str(cmake),
            "cmake_version": tool_version(cmake),
            "cuda_architecture": "110",
        },
        "build": {
            "build_type": "Release",
            "tf32_policy": "disabled",
            "cutlass_archive_sha256": CUTLASS_ARCHIVE_SHA256,
            "compile_command": compile_command,
            "link_command": link_command,
            "cmake_definitions": [
                "CMAKE_BUILD_TYPE=Release",
                "CMAKE_CUDA_ARCHITECTURES=110",
                "MAGPIE_TTS_RT_WARNINGS_AS_ERRORS=ON",
            ],
        },
        "runtime_dependencies": {
            "cublas": cublas_identity.to_json(),
        },
    }
    payload = (
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    output = args.output.absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise RuntimeError("output must be a new file in an existing directory")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    print(sha256_file(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
