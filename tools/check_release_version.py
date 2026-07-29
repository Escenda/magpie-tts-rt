#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import re
import tomllib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def cmake_version() -> str:
    contents = (PROJECT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(
        r"project\(\s*MagpieTTSRT\s+VERSION\s+"
        r"([0-9]+\.[0-9]+\.[0-9]+)",
        contents,
    )
    if match is None:
        raise SystemExit("unable to find the MagpieTTSRT CMake version")
    return match.group(1)


def cargo_version(manifest: pathlib.Path) -> str:
    with manifest.open("rb") as stream:
        document = tomllib.load(stream)
    return str(document["package"]["version"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    arguments = parser.parse_args()

    versions = {
        "CMake": cmake_version(),
        "magpie-tts-rt-sys": cargo_version(
            PROJECT_ROOT / "rust/magpie-tts-rt-sys/Cargo.toml"
        ),
        "magpie-tts-rt": cargo_version(
            PROJECT_ROOT / "rust/magpie-tts-rt/Cargo.toml"
        ),
    }
    unique_versions = set(versions.values())
    if len(unique_versions) != 1:
        raise SystemExit(f"release version mismatch: {versions}")

    version = unique_versions.pop()
    if arguments.tag is not None and arguments.tag != f"v{version}":
        raise SystemExit(
            f"tag {arguments.tag!r} does not match release version v{version}"
        )
    print(f"verified release version {version}")


if __name__ == "__main__":
    main()
