#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import re
import subprocess
import tempfile


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADER = PROJECT_ROOT / "include/magpie_tts_rt/magpie_tts_rt.h"
RUST_CONSTANTS = (
    PROJECT_ROOT / "rust/magpie-tts-rt-sys/src/constants.rs"
)
MACRO_PATTERN = re.compile(r"^#define (MTT_[A-Z0-9_]+)(?:\s|$)")
RUST_PATTERN = re.compile(
    r"^pub const (MTT_[A-Z0-9_]+):[^=]+="
    r"\s*([0-9][0-9_]*)\s*;$"
)


def c_macro_names() -> list[str]:
    result = subprocess.run(
        ["cc", "-dM", "-E", "-include", str(HEADER), "-"],
        input="",
        text=True,
        check=True,
        capture_output=True,
    )
    return sorted(
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := MACRO_PATTERN.match(line))
        if match.group(1) != "MTT_API"
    )


def c_values(names: list[str]) -> dict[str, int]:
    statements = "\n".join(
        f'  printf("{name}=%llu\\n", '
        f"(unsigned long long)({name}));"
        for name in names
    )
    source = (
        "#include <stdio.h>\n"
        f'#include "{HEADER}"\n'
        "int main(void) {\n"
        f"{statements}\n"
        "  return 0;\n"
        "}\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        temporary = pathlib.Path(directory)
        source_path = temporary / "constants.c"
        executable_path = temporary / "constants"
        source_path.write_text(source, encoding="utf-8")
        subprocess.run(
            [
                "cc",
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                str(source_path),
                "-o",
                str(executable_path),
            ],
            check=True,
        )
        output = subprocess.run(
            [str(executable_path)],
            text=True,
            check=True,
            capture_output=True,
        ).stdout
    return {
        name: int(value)
        for line in output.splitlines()
        for name, value in [line.split("=", maxsplit=1)]
    }


def rust_values() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in RUST_CONSTANTS.read_text(encoding="utf-8").splitlines():
        match = RUST_PATTERN.match(line)
        if match:
            values[match.group(1)] = int(match.group(2).replace("_", ""))
    return values


def main() -> None:
    names = c_macro_names()
    c_constants = c_values(names)
    rust_constants = rust_values()
    if c_constants != rust_constants:
        missing = sorted(c_constants.keys() - rust_constants.keys())
        extra = sorted(rust_constants.keys() - c_constants.keys())
        changed = sorted(
            name
            for name in c_constants.keys() & rust_constants.keys()
            if c_constants[name] != rust_constants[name]
        )
        raise SystemExit(
            "C/Rust ABI constant mismatch: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    print(f"verified {len(c_constants)} C/Rust ABI constants")


if __name__ == "__main__":
    main()
