"""Fail-closed identity for the cuBLAS DSOs loaded by one plugin.

The private CUDA-kernel launch parameter contract used by MagpieTTS-RT is a
property of the actual cuBLAS and cuBLASLt bytes mapped into the process. A
library found through a fresh loader search is therefore not evidence. This
module resolves both version symbols through the already-loaded plugin scope,
matches each symbol mapping to a stable regular-file descriptor, and hashes
that exact artifact.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path


CUBLAS_SONAME = "libcublas.so.13"
CUBLAS_LT_SONAME = "libcublasLt.so.13"

type JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | list[JsonValue]
    | dict[str, JsonValue]
)


class CublasIdentityError(RuntimeError):
    """The loaded cuBLAS dependency identity could not be proven."""


@dataclass(frozen=True)
class LoadedLibraryIdentity:
    soname: str
    size_bytes: int
    sha256: str

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "soname": self.soname,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class CublasRuntimeIdentity:
    api_version_integer: int
    library: LoadedLibraryIdentity
    lt_library: LoadedLibraryIdentity

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "api_version_integer": self.api_version_integer,
            "library": self.library.to_json(),
            "lt_library": self.lt_library.to_json(),
        }


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


class _Elf64DynamicValue(ctypes.Union):
    _fields_ = [
        ("d_val", ctypes.c_uint64),
        ("d_ptr", ctypes.c_void_p),
    ]


class _Elf64DynamicEntry(ctypes.Structure):
    _fields_ = [
        ("d_tag", ctypes.c_int64),
        ("d_un", _Elf64DynamicValue),
    ]


class _LinkMap(ctypes.Structure):
    pass


_LinkMap._fields_ = [
    ("l_addr", ctypes.c_void_p),
    ("l_name", ctypes.c_char_p),
    ("l_ld", ctypes.POINTER(_Elf64DynamicEntry)),
    ("l_next", ctypes.POINTER(_LinkMap)),
    ("l_prev", ctypes.POINTER(_LinkMap)),
]


@dataclass(frozen=True)
class _MappingIdentity:
    device: int
    inode: int


def _require_positive_integer(value: JsonValue | None, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CublasIdentityError(f"{label} must be a positive integer")
    return value


def _require_sha256(value: JsonValue | None, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CublasIdentityError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _parse_library_identity(
    value: JsonValue | None,
    *,
    label: str,
    expected_soname: str,
) -> LoadedLibraryIdentity:
    if not isinstance(value, dict) or set(value) != {
        "soname",
        "size_bytes",
        "sha256",
    }:
        raise CublasIdentityError(
            f"{label} must contain exactly soname, size_bytes, and sha256"
        )
    soname = value["soname"]
    if soname != expected_soname:
        raise CublasIdentityError(
            f"{label}.soname must be {expected_soname}"
        )
    return LoadedLibraryIdentity(
        soname=expected_soname,
        size_bytes=_require_positive_integer(
            value["size_bytes"], f"{label}.size_bytes"
        ),
        sha256=_require_sha256(value["sha256"], f"{label}.sha256"),
    )


def parse_cublas_runtime_identity(
    value: JsonValue | None,
    label: str,
) -> CublasRuntimeIdentity:
    if not isinstance(value, dict) or set(value) != {
        "api_version_integer",
        "library",
        "lt_library",
    }:
        raise CublasIdentityError(
            f"{label} must contain exactly api_version_integer, library, "
            "and lt_library"
        )
    return CublasRuntimeIdentity(
        api_version_integer=_require_positive_integer(
            value["api_version_integer"],
            f"{label}.api_version_integer",
        ),
        library=_parse_library_identity(
            value["library"],
            label=f"{label}.library",
            expected_soname=CUBLAS_SONAME,
        ),
        lt_library=_parse_library_identity(
            value["lt_library"],
            label=f"{label}.lt_library",
            expected_soname=CUBLAS_LT_SONAME,
        ),
    )


def _mapping_identity(address: int) -> _MappingIdentity:
    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CublasIdentityError(
            f"unable to read /proc/self/maps: {error}"
        ) from error
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 5:
            raise CublasIdentityError("malformed entry in /proc/self/maps")
        try:
            start_text, end_text = fields[0].split("-", 1)
            start = int(start_text, 16)
            end = int(end_text, 16)
        except ValueError as error:
            raise CublasIdentityError(
                "malformed address range in /proc/self/maps"
            ) from error
        if not start <= address < end:
            continue
        try:
            major_text, minor_text = fields[3].split(":", 1)
            device = os.makedev(int(major_text, 16), int(minor_text, 16))
            inode = int(fields[4], 10)
        except ValueError as error:
            raise CublasIdentityError(
                "malformed device/inode in /proc/self/maps"
            ) from error
        if inode <= 0:
            raise CublasIdentityError(
                "cuBLAS symbol is not backed by a regular-file mapping"
            )
        return _MappingIdentity(device=device, inode=inode)
    raise CublasIdentityError(
        "cuBLAS provider symbol has no file-backed process mapping"
    )


def _stable_file_fields(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _loaded_provider_info(
    address: int,
    expected_soname: str,
) -> tuple[_DlInfo, str]:
    libdl = ctypes.CDLL("libdl.so.2", use_errno=True)
    dladdr1 = libdl.dladdr1
    dladdr1.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_DlInfo),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    dladdr1.restype = ctypes.c_int
    information = _DlInfo()
    link_map_pointer = ctypes.c_void_p()
    rtld_dl_linkmap = 2
    if (
        dladdr1(
            ctypes.c_void_p(address),
            ctypes.byref(information),
            ctypes.byref(link_map_pointer),
            rtld_dl_linkmap,
        )
        == 0
        or information.dli_fname is None
        or not link_map_pointer.value
    ):
        raise CublasIdentityError(
            f"dladdr did not identify the provider for {expected_soname}"
        )
    link_map = ctypes.cast(
        link_map_pointer,
        ctypes.POINTER(_LinkMap),
    ).contents
    if not link_map.l_ld:
        raise CublasIdentityError(
            f"loaded provider has no dynamic section for {expected_soname}"
        )
    string_table_address: int | None = None
    string_table_size: int | None = None
    soname_offset: int | None = None
    found_terminator = False
    for index in range(16 * 1024):
        entry = link_map.l_ld[index]
        if entry.d_tag == 0:
            found_terminator = True
            break
        if entry.d_tag == 5:
            string_table_address = entry.d_un.d_ptr
        elif entry.d_tag == 10:
            string_table_size = int(entry.d_un.d_val)
        elif entry.d_tag == 14:
            soname_offset = int(entry.d_un.d_val)
    if (
        not found_terminator
        or string_table_address is None
        or string_table_address <= 0
        or string_table_size is None
        or string_table_size <= 0
        or soname_offset is None
        or soname_offset < 0
        or soname_offset >= string_table_size
    ):
        raise CublasIdentityError(
            f"loaded provider has no bounded ELF SONAME for {expected_soname}"
        )
    encoded = ctypes.string_at(
        string_table_address + soname_offset,
        string_table_size - soname_offset,
    )
    terminator = encoded.find(b"\x00")
    if terminator < 0:
        raise CublasIdentityError(
            f"loaded provider has an unterminated ELF SONAME for "
            f"{expected_soname}"
        )
    try:
        actual_soname = encoded[:terminator].decode("ascii")
    except UnicodeDecodeError as error:
        raise CublasIdentityError(
            f"loaded provider has a non-ASCII ELF SONAME for "
            f"{expected_soname}"
        ) from error
    return information, actual_soname


def _loaded_library_identity(
    symbol: ctypes._CFuncPtr,
    expected_soname: str,
) -> LoadedLibraryIdentity:
    address = ctypes.cast(symbol, ctypes.c_void_p).value
    if address is None or address <= 0:
        raise CublasIdentityError(
            f"{expected_soname} provider symbol has an invalid address"
        )
    information, actual_soname = _loaded_provider_info(
        address,
        expected_soname,
    )
    if actual_soname != expected_soname:
        raise CublasIdentityError(
            f"loaded provider ELF SONAME differs from {expected_soname}: "
            f"{actual_soname}"
        )
    provider_path = information.dli_fname
    if provider_path is None:
        raise CublasIdentityError(
            f"dladdr returned no provider path for {expected_soname}"
        )
    loader_path = Path(os.fsdecode(provider_path))
    if loader_path.name != expected_soname:
        raise CublasIdentityError(
            f"loaded provider name differs from {expected_soname}: "
            f"{loader_path.name}"
        )
    try:
        canonical = loader_path.resolve(strict=True)
    except OSError as error:
        raise CublasIdentityError(
            f"unable to resolve loaded {expected_soname}: {error}"
        ) from error
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(canonical, flags)
    except OSError as error:
        raise CublasIdentityError(
            f"unable to open loaded {expected_soname}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise CublasIdentityError(
                f"loaded {expected_soname} is not a nonempty regular file"
            )
        mapping = _mapping_identity(address)
        if before.st_dev != mapping.device or before.st_ino != mapping.inode:
            raise CublasIdentityError(
                f"loaded {expected_soname} mapping does not match the "
                "resolved artifact device/inode"
            )
        digest = _hash_descriptor(descriptor)
        after = os.fstat(descriptor)
        if _stable_file_fields(before) != _stable_file_fields(after):
            raise CublasIdentityError(
                f"loaded {expected_soname} changed while being authenticated"
            )
        return LoadedLibraryIdentity(
            soname=expected_soname,
            size_bytes=before.st_size,
            sha256=digest,
        )
    finally:
        os.close(descriptor)


def _plugin_symbol(
    plugin: ctypes.CDLL,
    name: str,
) -> ctypes._CFuncPtr:
    try:
        return plugin[name]
    except AttributeError as error:
        raise CublasIdentityError(
            f"unable to resolve {name} through the plugin dependency scope"
        ) from error


def collect_cublas_runtime_identity(
    plugin: ctypes.CDLL,
) -> CublasRuntimeIdentity:
    create = _plugin_symbol(plugin, "cublasCreate_v2")
    destroy = _plugin_symbol(plugin, "cublasDestroy_v2")
    get_version = _plugin_symbol(plugin, "cublasGetVersion_v2")
    get_lt_version = _plugin_symbol(plugin, "cublasLtGetVersion")

    create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_int
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = ctypes.c_int
    get_version.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    get_lt_version.argtypes = []
    get_lt_version.restype = ctypes.c_size_t

    handle = ctypes.c_void_p()
    if int(create(ctypes.byref(handle))) != 0 or not handle.value:
        raise CublasIdentityError(
            "cublasCreate_v2 failed while collecting runtime identity"
        )
    version = ctypes.c_int()
    version_status = int(get_version(handle, ctypes.byref(version)))
    destroy_status = int(destroy(handle))
    if version_status != 0 or version.value <= 0:
        raise CublasIdentityError(
            "cublasGetVersion_v2 returned an invalid version"
        )
    if destroy_status != 0:
        raise CublasIdentityError(
            "cublasDestroy_v2 failed after runtime identity collection"
        )
    lt_version = int(get_lt_version())
    if lt_version <= 0 or lt_version != version.value:
        raise CublasIdentityError(
            "cuBLAS and cuBLASLt API versions differ or are invalid"
        )
    return CublasRuntimeIdentity(
        api_version_integer=version.value,
        library=_loaded_library_identity(get_version, CUBLAS_SONAME),
        lt_library=_loaded_library_identity(
            get_lt_version, CUBLAS_LT_SONAME
        ),
    )
