# Accepted NeMo oracle overlay

These 11 files are the exact source snapshot used by the accepted Sofia BF16
streaming receipt. They are stored at their NeMo-relative paths so a clean
checkout of revision `9ae3e66b7314b0358c96bce47fbac56d78728bcd` can be
materialized without relying on an uncommitted working tree.

Each source file retains its NVIDIA copyright and Apache-2.0 license header.
The expected SHA-256 values are in
[`../oracle-lock.json`](../oracle-lock.json). Do not edit these files in place.
A changed implementation is a new oracle snapshot and requires a new lock,
fixture ID, and acceptance receipt.

The overlay is reference/export input. It is not imported by the
MagpieTTS-RT C++ runtime.
