#!/usr/bin/env bash
set -euo pipefail

project_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
generated="$(mktemp)"
trap 'rm -f "$generated"' EXIT

bindgen "$project_root/include/magpie_tts_rt/magpie_tts_rt.h" \
  --allowlist-type 'mtt_.*' \
  --allowlist-function '^mtt_get_api$' \
  --allowlist-var '^MTT_.*' \
  --use-core \
  --ctypes-prefix core::ffi \
  --no-layout-tests \
  --no-doc-comments \
  --formatter rustfmt \
  --output "$generated" \
  -- -std=c11

diff -u \
  "$project_root/rust/magpie-tts-rt-sys/src/bindings.rs" \
  "$generated"
