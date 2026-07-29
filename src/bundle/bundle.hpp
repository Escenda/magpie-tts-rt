#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "manifest/manifest.hpp"

namespace magpie_tts_rt {

namespace detail {
class BundleVerifierAccess;
}

inline constexpr std::string_view kDefaultRuntimeBundleManifestPath =
    "runtime-bundle-manifest.json";

enum class BundleStage {
  bundle_root,
  manifest_file,
  artifact_path,
  artifact_file,
  sha256,
};

enum class BundleErrorCode {
  invalid_relative_path,
  not_found,
  not_directory,
  not_regular_file,
  path_escape,
  duplicate_canonical_path,
  duplicate_file_identity,
  io_error,
  size_limit_exceeded,
  size_mismatch,
  invalid_digest,
  digest_mismatch,
};

enum class BundleArtifactKind {
  manifest,
  model,
  export_artifact,
  tokenizer,
  plugin,
  engine,
  golden_receipt,
};

[[nodiscard]] std::string_view to_string(BundleStage stage) noexcept;
[[nodiscard]] std::string_view to_string(BundleErrorCode code) noexcept;
[[nodiscard]] std::string_view to_string(BundleArtifactKind kind) noexcept;

class BundleError final : public std::runtime_error {
 public:
  BundleError(
      BundleStage stage,
      BundleErrorCode code,
      std::filesystem::path path,
      std::string manifest_pointer,
      std::string detail);

  [[nodiscard]] BundleStage stage() const noexcept;
  [[nodiscard]] BundleErrorCode code() const noexcept;
  [[nodiscard]] const std::filesystem::path& path() const noexcept;
  [[nodiscard]] const std::string& manifest_pointer() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  BundleStage stage_;
  BundleErrorCode code_;
  std::filesystem::path path_;
  std::string manifest_pointer_;
  std::string detail_;
};

class VerifiedBundleArtifact final {
 public:
  ~VerifiedBundleArtifact();

  VerifiedBundleArtifact(const VerifiedBundleArtifact&) = delete;
  VerifiedBundleArtifact& operator=(const VerifiedBundleArtifact&) = delete;
  VerifiedBundleArtifact(VerifiedBundleArtifact&& other) noexcept;
  VerifiedBundleArtifact& operator=(VerifiedBundleArtifact&& other) noexcept;

  // Borrowed descriptor for an immutable sealed memfd containing the exact
  // bytes whose digest was verified. It remains valid until this artifact is
  // destroyed or moved from. Consumers must not close it. They must
  // deserialize, mmap, dlopen("/proc/self/fd/N"), or pread from this
  // descriptor; canonical_path is diagnostic metadata and must never be
  // reopened because doing so reintroduces replacement and mutation TOCTOU.
  [[nodiscard]] int verified_file_descriptor() const noexcept;

  BundleArtifactKind kind;
  std::string logical_name;
  std::string manifest_pointer;
  std::filesystem::path relative_path;
  std::filesystem::path canonical_path;
  std::string sha256;
  std::uint64_t size_bytes;

 private:
  friend class detail::BundleVerifierAccess;

  VerifiedBundleArtifact(
      BundleArtifactKind kind,
      std::string logical_name,
      std::string manifest_pointer,
      std::filesystem::path relative_path,
      std::filesystem::path canonical_path,
      std::string sha256,
      std::uint64_t size_bytes,
      int verified_file_descriptor);

  int verified_file_descriptor_;
};

struct VerifiedRuntimeBundle {
  VerifiedRuntimeBundle(
      std::filesystem::path canonical_root,
      std::filesystem::path canonical_manifest_path,
      VerifiedBundleArtifact manifest_snapshot,
      RuntimeBundleManifest manifest,
      std::vector<VerifiedBundleArtifact> artifacts);

  VerifiedRuntimeBundle(const VerifiedRuntimeBundle&) = delete;
  VerifiedRuntimeBundle& operator=(const VerifiedRuntimeBundle&) = delete;
  VerifiedRuntimeBundle(VerifiedRuntimeBundle&&) noexcept = default;
  VerifiedRuntimeBundle& operator=(VerifiedRuntimeBundle&&) noexcept = default;

  std::filesystem::path canonical_root;
  std::filesystem::path canonical_manifest_path;
  VerifiedBundleArtifact manifest_snapshot;
  RuntimeBundleManifest manifest;
  std::vector<VerifiedBundleArtifact> artifacts;
};

// Verifies the complete artifact set described by an already parsed manifest.
// All returned paths are canonical and confined to canonical_bundle_root.
[[nodiscard]] std::vector<VerifiedBundleArtifact> verify_runtime_bundle_files(
    const std::filesystem::path& bundle_root,
    const RuntimeBundleManifest& manifest);

// Loads the fixed default manifest, requires its exact trusted SHA-256, and
// verifies every referenced artifact from immutable snapshots.
[[nodiscard]] VerifiedRuntimeBundle load_and_verify_runtime_bundle(
    const std::filesystem::path& bundle_root,
    std::string_view expected_manifest_sha256);

// The explicit manifest path is relative to bundle_root and is subjected to the
// same canonical containment and regular-file checks as bundle artifacts.
[[nodiscard]] VerifiedRuntimeBundle load_and_verify_runtime_bundle(
    const std::filesystem::path& bundle_root,
    const std::filesystem::path& manifest_relative_path,
    std::string_view expected_manifest_sha256);

}  // namespace magpie_tts_rt
