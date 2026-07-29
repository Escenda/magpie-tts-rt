#include "bundle/bundle.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <system_error>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <linux/memfd.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace magpie_tts_rt {

namespace detail {

class BundleVerifierAccess final {
 public:
  [[nodiscard]] static VerifiedBundleArtifact adopt_verified_descriptor(
      const BundleArtifactKind kind,
      std::string logical_name,
      std::string manifest_pointer,
      std::filesystem::path relative_path,
      std::filesystem::path canonical_path,
      std::string sha256,
      const std::uint64_t size_bytes,
      const int verified_file_descriptor) {
    return VerifiedBundleArtifact(
        kind,
        std::move(logical_name),
        std::move(manifest_pointer),
        std::move(relative_path),
        std::move(canonical_path),
        std::move(sha256),
        size_bytes,
        verified_file_descriptor);
  }
};

}  // namespace detail

namespace {

constexpr std::size_t kSha256Bytes = 32;
constexpr std::size_t kReadBufferBytes = 64 * 1024;

[[nodiscard]] std::string build_error_message(
    const BundleStage stage,
    const BundleErrorCode code,
    const std::filesystem::path& path,
    const std::string_view manifest_pointer,
    const std::string_view detail) {
  std::ostringstream message;
  message << "runtime bundle verification error"
          << " [stage=" << to_string(stage) << ", code=" << to_string(code)
          << ", path=" << (path.empty() ? "<none>" : path.string())
          << ", manifest_pointer="
          << (manifest_pointer.empty() ? "/" : manifest_pointer) << "]: "
          << detail;
  return message.str();
}

[[noreturn]] void fail(
    const BundleStage stage,
    const BundleErrorCode code,
    const std::filesystem::path& path,
    const std::string& manifest_pointer,
    const std::string& detail) {
  throw BundleError(stage, code, path, manifest_pointer, detail);
}

[[nodiscard]] bool is_missing_error(const std::error_code& error) noexcept {
  return error == std::errc::no_such_file_or_directory ||
         error == std::errc::not_a_directory;
}

[[nodiscard]] std::string system_error_detail(
    const std::string_view operation,
    const int error_number) {
  std::ostringstream detail;
  detail << operation << " failed: " << std::strerror(error_number);
  return detail.str();
}

[[nodiscard]] std::string openssl_error_detail(
    const std::string_view operation) {
  std::array<char, 256> buffer{};
  const unsigned long error = ERR_get_error();
  if (error == 0) {
    return std::string(operation) + " failed without an OpenSSL error code";
  }
  ERR_error_string_n(error, buffer.data(), buffer.size());
  return std::string(operation) + " failed: " + buffer.data();
}

[[nodiscard]] bool is_strict_relative_path(
    const std::filesystem::path& path) {
  const std::string generic_path = path.generic_string();
  if (path.empty() || path.is_absolute() || path.has_root_name() ||
      path.has_root_directory() ||
      generic_path.find('\\') != std::string::npos) {
    return false;
  }
  for (const char raw_byte : generic_path) {
    const auto byte = static_cast<unsigned char>(raw_byte);
    if (byte <= 0x1FU || byte == 0x7FU) {
      return false;
    }
  }
  for (const std::filesystem::path& component : path) {
    if (component.empty() || component == "." || component == "..") {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool is_within_root(
    const std::filesystem::path& root,
    const std::filesystem::path& path) {
  auto root_component = root.begin();
  auto path_component = path.begin();
  for (; root_component != root.end();
       ++root_component, ++path_component) {
    if (path_component == path.end() || *root_component != *path_component) {
      return false;
    }
  }
  return path_component != path.end();
}

[[nodiscard]] std::filesystem::path canonical_bundle_root(
    const std::filesystem::path& bundle_root) {
  std::error_code status_error;
  const std::filesystem::file_status status =
      std::filesystem::status(bundle_root, status_error);
  if (status_error) {
    fail(
        BundleStage::bundle_root,
        is_missing_error(status_error) ? BundleErrorCode::not_found
                                       : BundleErrorCode::io_error,
        bundle_root,
        "/",
        status_error.message());
  }
  if (!std::filesystem::exists(status)) {
    fail(
        BundleStage::bundle_root,
        BundleErrorCode::not_found,
        bundle_root,
        "/",
        "bundle root does not exist");
  }
  if (!std::filesystem::is_directory(status)) {
    fail(
        BundleStage::bundle_root,
        BundleErrorCode::not_directory,
        bundle_root,
        "/",
        "bundle root must be a directory");
  }

  std::error_code canonical_error;
  const std::filesystem::path canonical =
      std::filesystem::canonical(bundle_root, canonical_error);
  if (canonical_error) {
    fail(
        BundleStage::bundle_root,
        BundleErrorCode::io_error,
        bundle_root,
        "/",
        canonical_error.message());
  }
  return canonical;
}

class FileDescriptor final {
 public:
  explicit FileDescriptor(const int value) noexcept : value_(value) {}

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  FileDescriptor(FileDescriptor&& other) noexcept
      : value_(std::exchange(other.value_, -1)) {}

  FileDescriptor& operator=(FileDescriptor&& other) noexcept {
    if (this != &other) {
      if (value_ >= 0) {
        static_cast<void>(::close(value_));
      }
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }

  ~FileDescriptor() {
    if (value_ >= 0) {
      static_cast<void>(::close(value_));
    }
  }

  [[nodiscard]] int get() const noexcept { return value_; }
  [[nodiscard]] int release() noexcept {
    return std::exchange(value_, -1);
  }

 private:
  int value_;
};

struct OpenedRegularFile {
  FileDescriptor descriptor;
  std::filesystem::path canonical_path;
  std::uintmax_t device_id;
  std::uintmax_t inode_id;
  std::uint64_t size_bytes;
};

[[nodiscard]] OpenedRegularFile open_regular_bundle_file(
    const std::filesystem::path& canonical_root,
    const std::filesystem::path& relative_path,
    const BundleStage stage,
    const std::string& manifest_pointer) {
  if (!is_strict_relative_path(relative_path)) {
    fail(
        stage == BundleStage::manifest_file ? BundleStage::manifest_file
                                            : BundleStage::artifact_path,
        BundleErrorCode::invalid_relative_path,
        relative_path,
        manifest_pointer,
        "path must be a non-empty relative path without control bytes, '.', '..', or backslash components");
  }

  const std::filesystem::path requested_path =
      canonical_root / relative_path;
  std::error_code canonical_error;
  const std::filesystem::path resolved_path =
      std::filesystem::canonical(requested_path, canonical_error);
  if (canonical_error) {
    fail(
        stage,
        is_missing_error(canonical_error) ? BundleErrorCode::not_found
                                          : BundleErrorCode::io_error,
        requested_path,
        manifest_pointer,
        canonical_error.message());
  }
  if (!is_within_root(canonical_root, resolved_path)) {
    fail(
        stage == BundleStage::manifest_file ? BundleStage::manifest_file
                                            : BundleStage::artifact_path,
        BundleErrorCode::path_escape,
        resolved_path,
        manifest_pointer,
        "canonical path escapes the bundle root");
  }

  const int descriptor = ::open(
      resolved_path.c_str(),
      O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (descriptor < 0) {
    const int open_error = errno;
    fail(
        stage,
        open_error == ENOENT || open_error == ENOTDIR
            ? BundleErrorCode::not_found
            : BundleErrorCode::io_error,
        resolved_path,
        manifest_pointer,
        system_error_detail("open", open_error));
  }
  FileDescriptor file(descriptor);

  struct stat file_status {};
  if (::fstat(file.get(), &file_status) != 0) {
    const int stat_error = errno;
    fail(
        stage,
        BundleErrorCode::io_error,
        resolved_path,
        manifest_pointer,
        system_error_detail("fstat", stat_error));
  }
  if (!S_ISREG(file_status.st_mode)) {
    fail(
        stage,
        BundleErrorCode::not_regular_file,
        resolved_path,
        manifest_pointer,
        "resolved path must identify a regular file");
  }
  if (file_status.st_size < 0) {
    fail(
        stage,
        BundleErrorCode::io_error,
        resolved_path,
        manifest_pointer,
        "regular file reported a negative size");
  }

  // Re-resolve the opened descriptor so a concurrent parent-directory or
  // symlink replacement cannot redirect the file outside the bundle after the
  // initial canonical() check.
  const std::filesystem::path descriptor_path =
      std::filesystem::path("/proc/self/fd") /
      std::to_string(file.get());
  std::error_code descriptor_error;
  const std::filesystem::path opened_path =
      std::filesystem::canonical(descriptor_path, descriptor_error);
  if (descriptor_error) {
    fail(
        stage,
        BundleErrorCode::io_error,
        resolved_path,
        manifest_pointer,
        "unable to resolve the opened file descriptor: " +
            descriptor_error.message());
  }
  if (!is_within_root(canonical_root, opened_path)) {
    fail(
        stage == BundleStage::manifest_file ? BundleStage::manifest_file
                                            : BundleStage::artifact_path,
        BundleErrorCode::path_escape,
        opened_path,
        manifest_pointer,
        "opened file escapes the bundle root");
  }

  return OpenedRegularFile{
      .descriptor = std::move(file),
      .canonical_path = opened_path,
      .device_id = static_cast<std::uintmax_t>(file_status.st_dev),
      .inode_id = static_cast<std::uintmax_t>(file_status.st_ino),
      .size_bytes = static_cast<std::uint64_t>(file_status.st_size),
  };
}

[[nodiscard]] bool is_lowercase_sha256(const std::string_view digest) {
  if (digest.size() != kSha256Bytes * 2) {
    return false;
  }
  for (const char character : digest) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

struct Sha256SnapshotResult {
  std::string digest;
  std::uint64_t size_bytes;
  FileDescriptor sealed_snapshot;
};

[[nodiscard]] Sha256SnapshotResult snapshot_and_sha256_file(
    const OpenedRegularFile& file,
    const std::string& manifest_pointer,
    const BundleStage snapshot_stage,
    const std::uint64_t maximum_bytes) {
  if (file.size_bytes > maximum_bytes) {
    fail(
        snapshot_stage,
        BundleErrorCode::size_limit_exceeded,
        file.canonical_path,
        manifest_pointer,
        "file exceeds the configured snapshot size limit");
  }
  using DigestContext =
      std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
  DigestContext context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (!context) {
    fail(
        BundleStage::sha256,
        BundleErrorCode::io_error,
        file.canonical_path,
        manifest_pointer,
        openssl_error_detail("EVP_MD_CTX_new"));
  }
  if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    fail(
        BundleStage::sha256,
        BundleErrorCode::io_error,
        file.canonical_path,
        manifest_pointer,
        openssl_error_detail("EVP_DigestInit_ex"));
  }

  const long snapshot_result = ::syscall(
      SYS_memfd_create,
      "magpie-tts-rt-artifact",
      MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (snapshot_result < 0) {
    const int snapshot_error = errno;
    fail(
        snapshot_stage,
        BundleErrorCode::io_error,
        file.canonical_path,
        manifest_pointer,
        system_error_detail("memfd_create", snapshot_error));
  }
  FileDescriptor snapshot(static_cast<int>(snapshot_result));

  std::array<unsigned char, kReadBufferBytes> buffer{};
  std::uint64_t size_bytes = 0;
  while (true) {
    const ssize_t read_size =
        ::read(file.descriptor.get(), buffer.data(), buffer.size());
    if (read_size == 0) {
      break;
    }
    if (read_size < 0) {
      const int read_error = errno;
      if (read_error == EINTR) {
        continue;
      }
      fail(
          BundleStage::sha256,
          BundleErrorCode::io_error,
          file.canonical_path,
          manifest_pointer,
          system_error_detail("read", read_error));
    }

    const auto chunk_size = static_cast<std::size_t>(read_size);
    if (static_cast<std::uint64_t>(chunk_size) > maximum_bytes ||
        size_bytes >
            maximum_bytes - static_cast<std::uint64_t>(chunk_size)) {
      fail(
          snapshot_stage,
          BundleErrorCode::size_limit_exceeded,
          file.canonical_path,
          manifest_pointer,
          "file grew beyond the configured snapshot size limit");
    }
    if (size_bytes >
        std::numeric_limits<std::uint64_t>::max() -
            static_cast<std::uint64_t>(chunk_size)) {
      fail(
          BundleStage::sha256,
          BundleErrorCode::io_error,
          file.canonical_path,
          manifest_pointer,
          "file size exceeds uint64_t");
    }
    size_bytes += static_cast<std::uint64_t>(chunk_size);
    std::size_t written = 0;
    while (written < chunk_size) {
      const ssize_t write_size = ::write(
          snapshot.get(),
          buffer.data() + written,
          chunk_size - written);
      if (write_size < 0) {
        const int write_error = errno;
        if (write_error == EINTR) {
          continue;
        }
        fail(
            snapshot_stage,
            BundleErrorCode::io_error,
            file.canonical_path,
            manifest_pointer,
            system_error_detail("write sealed snapshot", write_error));
      }
      if (write_size == 0) {
        fail(
            snapshot_stage,
            BundleErrorCode::io_error,
            file.canonical_path,
            manifest_pointer,
            "writing the artifact snapshot made no progress");
      }
      written += static_cast<std::size_t>(write_size);
    }
    if (EVP_DigestUpdate(context.get(), buffer.data(), chunk_size) != 1) {
      fail(
          BundleStage::sha256,
          BundleErrorCode::io_error,
          file.canonical_path,
          manifest_pointer,
          openssl_error_detail("EVP_DigestUpdate"));
    }
  }

  std::array<unsigned char, EVP_MAX_MD_SIZE> binary_digest{};
  unsigned int digest_size = 0;
  if (EVP_DigestFinal_ex(
          context.get(), binary_digest.data(), &digest_size) != 1) {
    fail(
        BundleStage::sha256,
        BundleErrorCode::io_error,
        file.canonical_path,
        manifest_pointer,
        openssl_error_detail("EVP_DigestFinal_ex"));
  }
  if (digest_size != kSha256Bytes) {
    fail(
        BundleStage::sha256,
        BundleErrorCode::io_error,
        file.canonical_path,
        manifest_pointer,
        "OpenSSL returned an unexpected SHA-256 digest length");
  }

  constexpr std::array<char, 16> hexadecimal{
      '0', '1', '2', '3', '4', '5', '6', '7',
      '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  std::string digest(kSha256Bytes * 2, '0');
  for (std::size_t index = 0; index < kSha256Bytes; ++index) {
    const unsigned char byte = binary_digest.at(index);
    digest.at(index * 2) = hexadecimal.at(byte >> 4U);
    digest.at(index * 2 + 1) = hexadecimal.at(byte & 0x0FU);
  }
  if (::lseek(snapshot.get(), 0, SEEK_SET) < 0) {
    const int seek_error = errno;
    fail(
        snapshot_stage,
        BundleErrorCode::io_error,
        file.canonical_path,
        manifest_pointer,
        system_error_detail("lseek sealed snapshot", seek_error));
  }
  constexpr int required_seals =
      F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;
  if (::fcntl(snapshot.get(), F_ADD_SEALS, required_seals) != 0) {
    const int seal_error = errno;
    fail(
        snapshot_stage,
        BundleErrorCode::io_error,
        file.canonical_path,
        manifest_pointer,
        system_error_detail("F_ADD_SEALS", seal_error));
  }

  return Sha256SnapshotResult{
      .digest = std::move(digest),
      .size_bytes = size_bytes,
      .sealed_snapshot = std::move(snapshot),
  };
}

[[nodiscard]] std::string read_complete_file(
    const int descriptor,
    const std::filesystem::path& diagnostic_path,
    const std::string& manifest_pointer) {
  if (::lseek(descriptor, 0, SEEK_SET) < 0) {
    const int seek_error = errno;
    fail(
        BundleStage::manifest_file,
        BundleErrorCode::io_error,
        diagnostic_path,
        manifest_pointer,
        system_error_detail("lseek manifest snapshot", seek_error));
  }
  std::string contents;
  std::array<char, kReadBufferBytes> buffer{};
  while (true) {
    const ssize_t read_size =
        ::read(descriptor, buffer.data(), buffer.size());
    if (read_size == 0) {
      return contents;
    }
    if (read_size < 0) {
      const int read_error = errno;
      if (read_error == EINTR) {
        continue;
      }
      fail(
          BundleStage::manifest_file,
          BundleErrorCode::io_error,
          diagnostic_path,
          manifest_pointer,
          system_error_detail("read", read_error));
    }
    const auto chunk_size = static_cast<std::size_t>(read_size);
    constexpr auto maximum_manifest_bytes =
        static_cast<std::size_t>(kMaximumRuntimeBundleManifestBytes);
    if (chunk_size > maximum_manifest_bytes ||
        contents.size() > maximum_manifest_bytes - chunk_size) {
      fail(
          BundleStage::manifest_file,
          BundleErrorCode::size_limit_exceeded,
          diagnostic_path,
          manifest_pointer,
          "manifest exceeds the 16 MiB size limit");
    }
    if (contents.size() > contents.max_size() - chunk_size) {
      fail(
          BundleStage::manifest_file,
          BundleErrorCode::io_error,
          diagnostic_path,
          manifest_pointer,
          "manifest is too large to represent in memory");
    }
    contents.append(buffer.data(), chunk_size);
  }
}

struct ArtifactRequest {
  BundleArtifactKind kind;
  std::string logical_name;
  std::filesystem::path relative_path;
  std::string expected_sha256;
  std::uint64_t expected_size_bytes;
  std::string path_pointer;
  std::string hash_pointer;
  std::string size_pointer;
};

[[nodiscard]] std::vector<ArtifactRequest> collect_artifacts(
    const RuntimeBundleManifest& manifest) {
  std::vector<ArtifactRequest> artifacts;
  artifacts.reserve(5 + manifest.engines.size());
  artifacts.push_back(ArtifactRequest{
      .kind = BundleArtifactKind::model,
      .logical_name = manifest.artifacts.model.model_id,
      .relative_path = manifest.artifacts.model.file.path,
      .expected_sha256 = manifest.artifacts.model.file.sha256,
      .expected_size_bytes = manifest.artifacts.model.file.size_bytes,
      .path_pointer = "/artifacts/model/file/path",
      .hash_pointer = "/artifacts/model/file/sha256",
      .size_pointer = "/artifacts/model/file/size_bytes",
  });
  artifacts.push_back(ArtifactRequest{
      .kind = BundleArtifactKind::export_artifact,
      .logical_name = manifest.artifacts.export_artifact.format,
      .relative_path = manifest.artifacts.export_artifact.file.path,
      .expected_sha256 = manifest.artifacts.export_artifact.file.sha256,
      .expected_size_bytes =
          manifest.artifacts.export_artifact.file.size_bytes,
      .path_pointer = "/artifacts/export/file/path",
      .hash_pointer = "/artifacts/export/file/sha256",
      .size_pointer = "/artifacts/export/file/size_bytes",
  });
  artifacts.push_back(ArtifactRequest{
      .kind = BundleArtifactKind::tokenizer,
      .logical_name = manifest.artifacts.tokenizer.kind,
      .relative_path = manifest.artifacts.tokenizer.file.path,
      .expected_sha256 = manifest.artifacts.tokenizer.file.sha256,
      .expected_size_bytes =
          manifest.artifacts.tokenizer.file.size_bytes,
      .path_pointer = "/artifacts/tokenizer/file/path",
      .hash_pointer = "/artifacts/tokenizer/file/sha256",
      .size_pointer = "/artifacts/tokenizer/file/size_bytes",
  });
  artifacts.push_back(ArtifactRequest{
      .kind = BundleArtifactKind::plugin,
      .logical_name = manifest.artifacts.plugin.name,
      .relative_path = manifest.artifacts.plugin.file.path,
      .expected_sha256 = manifest.artifacts.plugin.file.sha256,
      .expected_size_bytes = manifest.artifacts.plugin.file.size_bytes,
      .path_pointer = "/artifacts/plugin/file/path",
      .hash_pointer = "/artifacts/plugin/file/sha256",
      .size_pointer = "/artifacts/plugin/file/size_bytes",
  });
  for (std::size_t index = 0; index < manifest.engines.size(); ++index) {
    const EngineManifest& engine = manifest.engines.at(index);
    const std::string base_pointer =
        "/engines/" + std::to_string(index) + "/file";
    artifacts.push_back(ArtifactRequest{
        .kind = BundleArtifactKind::engine,
        .logical_name = engine.name,
        .relative_path = engine.file.path,
        .expected_sha256 = engine.file.sha256,
        .expected_size_bytes = engine.file.size_bytes,
        .path_pointer = base_pointer + "/path",
        .hash_pointer = base_pointer + "/sha256",
        .size_pointer = base_pointer + "/size_bytes",
    });
  }
  artifacts.push_back(ArtifactRequest{
      .kind = BundleArtifactKind::golden_receipt,
      .logical_name = "golden_receipt",
      .relative_path = manifest.golden_receipt.path,
      .expected_sha256 = manifest.golden_receipt.sha256,
      .expected_size_bytes = manifest.golden_receipt.size_bytes,
      .path_pointer = "/golden_receipt/path",
      .hash_pointer = "/golden_receipt/sha256",
      .size_pointer = "/golden_receipt/size_bytes",
  });
  return artifacts;
}

[[nodiscard]] std::vector<VerifiedBundleArtifact>
verify_runtime_bundle_files_at_canonical_root(
    const std::filesystem::path& canonical_root,
    const RuntimeBundleManifest& manifest) {
  const std::vector<ArtifactRequest> requests = collect_artifacts(manifest);
  std::uint64_t declared_snapshot_bytes = 0;
  for (const ArtifactRequest& request : requests) {
    if (request.expected_size_bytes >
        kMaximumBundleSnapshotBytes - declared_snapshot_bytes) {
      fail(
          BundleStage::artifact_file,
          BundleErrorCode::size_limit_exceeded,
          request.relative_path,
          request.size_pointer,
          "declared artifact sizes exceed the runtime bundle snapshot hard limit");
    }
    declared_snapshot_bytes += request.expected_size_bytes;
  }
  if (manifest.limits.maximum_bundle_snapshot_bytes >
      kMaximumBundleSnapshotBytes) {
    fail(
        BundleStage::artifact_file,
        BundleErrorCode::size_limit_exceeded,
        canonical_root,
        "/limits/maximum_bundle_snapshot_bytes",
        "bundle snapshot budget exceeds the runtime hard limit");
  }
  if (declared_snapshot_bytes !=
      manifest.limits.maximum_bundle_snapshot_bytes) {
    fail(
        BundleStage::artifact_file,
        BundleErrorCode::size_mismatch,
        canonical_root,
        "/limits/maximum_bundle_snapshot_bytes",
        "bundle snapshot budget must exactly equal the sum of declared artifact sizes");
  }
  std::vector<VerifiedBundleArtifact> verified;
  verified.reserve(requests.size());
  std::unordered_map<std::string, std::string> canonical_paths;
  canonical_paths.reserve(requests.size());
  std::unordered_map<std::string, std::string> file_identities;
  file_identities.reserve(requests.size());

  for (const ArtifactRequest& request : requests) {
    if (!is_lowercase_sha256(request.expected_sha256)) {
      fail(
          BundleStage::sha256,
          BundleErrorCode::invalid_digest,
          request.relative_path,
          request.hash_pointer,
          "expected SHA-256 must be exactly 64 lowercase hexadecimal characters");
    }

    OpenedRegularFile file = open_regular_bundle_file(
        canonical_root,
        request.relative_path,
        BundleStage::artifact_file,
        request.path_pointer);
    if (file.size_bytes != request.expected_size_bytes) {
      fail(
          BundleStage::artifact_file,
          BundleErrorCode::size_mismatch,
          file.canonical_path,
          request.size_pointer,
          "artifact size does not match the authenticated manifest value");
    }
    const std::string canonical_key = file.canonical_path.generic_string();
    const auto [existing, inserted] =
        canonical_paths.emplace(canonical_key, request.path_pointer);
    if (!inserted) {
      fail(
          BundleStage::artifact_path,
          BundleErrorCode::duplicate_canonical_path,
          file.canonical_path,
          request.path_pointer,
          "canonical artifact path is already referenced by " +
              existing->second);
    }
    const std::string identity_key =
        std::to_string(file.device_id) + ":" +
        std::to_string(file.inode_id);
    const auto [identity_existing, identity_inserted] =
        file_identities.emplace(identity_key, request.path_pointer);
    if (!identity_inserted) {
      fail(
          BundleStage::artifact_path,
          BundleErrorCode::duplicate_file_identity,
          file.canonical_path,
          request.path_pointer,
          "artifact is a hard-link alias already referenced by " +
              identity_existing->second);
    }

    Sha256SnapshotResult actual =
        snapshot_and_sha256_file(
            file,
            request.size_pointer,
            BundleStage::artifact_file,
            request.expected_size_bytes);
    if (actual.size_bytes != request.expected_size_bytes) {
      fail(
          BundleStage::artifact_file,
          BundleErrorCode::size_mismatch,
          file.canonical_path,
          request.size_pointer,
          "artifact size changed while the immutable snapshot was created");
    }
    if (actual.digest != request.expected_sha256) {
      fail(
          BundleStage::sha256,
          BundleErrorCode::digest_mismatch,
          file.canonical_path,
          request.hash_pointer,
          "expected " + request.expected_sha256 + ", computed " +
              actual.digest);
    }

    VerifiedBundleArtifact artifact =
        detail::BundleVerifierAccess::adopt_verified_descriptor(
        request.kind,
        request.logical_name,
        request.path_pointer,
        request.relative_path,
        file.canonical_path,
        actual.digest,
        actual.size_bytes,
        actual.sealed_snapshot.get());
    static_cast<void>(actual.sealed_snapshot.release());
    verified.push_back(std::move(artifact));
  }
  return verified;
}

}  // namespace

std::string_view to_string(const BundleStage stage) noexcept {
  switch (stage) {
    case BundleStage::bundle_root:
      return "bundle_root";
    case BundleStage::manifest_file:
      return "manifest_file";
    case BundleStage::artifact_path:
      return "artifact_path";
    case BundleStage::artifact_file:
      return "artifact_file";
    case BundleStage::sha256:
      return "sha256";
  }
  return "unknown";
}

std::string_view to_string(const BundleErrorCode code) noexcept {
  switch (code) {
    case BundleErrorCode::invalid_relative_path:
      return "invalid_relative_path";
    case BundleErrorCode::not_found:
      return "not_found";
    case BundleErrorCode::not_directory:
      return "not_directory";
    case BundleErrorCode::not_regular_file:
      return "not_regular_file";
    case BundleErrorCode::path_escape:
      return "path_escape";
    case BundleErrorCode::duplicate_canonical_path:
      return "duplicate_canonical_path";
    case BundleErrorCode::duplicate_file_identity:
      return "duplicate_file_identity";
    case BundleErrorCode::io_error:
      return "io_error";
    case BundleErrorCode::size_limit_exceeded:
      return "size_limit_exceeded";
    case BundleErrorCode::size_mismatch:
      return "size_mismatch";
    case BundleErrorCode::invalid_digest:
      return "invalid_digest";
    case BundleErrorCode::digest_mismatch:
      return "digest_mismatch";
  }
  return "unknown";
}

std::string_view to_string(const BundleArtifactKind kind) noexcept {
  switch (kind) {
    case BundleArtifactKind::manifest:
      return "manifest";
    case BundleArtifactKind::model:
      return "model";
    case BundleArtifactKind::export_artifact:
      return "export";
    case BundleArtifactKind::tokenizer:
      return "tokenizer";
    case BundleArtifactKind::plugin:
      return "plugin";
    case BundleArtifactKind::engine:
      return "engine";
    case BundleArtifactKind::golden_receipt:
      return "golden_receipt";
  }
  return "unknown";
}

BundleError::BundleError(
    const BundleStage stage,
    const BundleErrorCode code,
    std::filesystem::path path,
    std::string manifest_pointer,
    std::string detail)
    : std::runtime_error(
          build_error_message(stage, code, path, manifest_pointer, detail)),
      stage_(stage),
      code_(code),
      path_(std::move(path)),
      manifest_pointer_(std::move(manifest_pointer)),
      detail_(std::move(detail)) {}

BundleStage BundleError::stage() const noexcept { return stage_; }

BundleErrorCode BundleError::code() const noexcept { return code_; }

const std::filesystem::path& BundleError::path() const noexcept {
  return path_;
}

const std::string& BundleError::manifest_pointer() const noexcept {
  return manifest_pointer_;
}

const std::string& BundleError::detail() const noexcept { return detail_; }

VerifiedBundleArtifact::VerifiedBundleArtifact(
    const BundleArtifactKind kind_value,
    std::string logical_name_value,
    std::string manifest_pointer_value,
    std::filesystem::path relative_path_value,
    std::filesystem::path canonical_path_value,
    std::string sha256_value,
    const std::uint64_t size_bytes_value,
    const int verified_file_descriptor_value)
    : kind(kind_value),
      logical_name(std::move(logical_name_value)),
      manifest_pointer(std::move(manifest_pointer_value)),
      relative_path(std::move(relative_path_value)),
      canonical_path(std::move(canonical_path_value)),
      sha256(std::move(sha256_value)),
      size_bytes(size_bytes_value),
      verified_file_descriptor_(verified_file_descriptor_value) {
  if (verified_file_descriptor_ < 0) {
    throw std::invalid_argument(
        "verified artifact requires an open file descriptor");
  }
}

VerifiedBundleArtifact::~VerifiedBundleArtifact() {
  if (verified_file_descriptor_ >= 0) {
    static_cast<void>(::close(verified_file_descriptor_));
  }
}

VerifiedBundleArtifact::VerifiedBundleArtifact(
    VerifiedBundleArtifact&& other) noexcept
    : kind(other.kind),
      logical_name(std::move(other.logical_name)),
      manifest_pointer(std::move(other.manifest_pointer)),
      relative_path(std::move(other.relative_path)),
      canonical_path(std::move(other.canonical_path)),
      sha256(std::move(other.sha256)),
      size_bytes(other.size_bytes),
      verified_file_descriptor_(
          std::exchange(other.verified_file_descriptor_, -1)) {}

VerifiedBundleArtifact& VerifiedBundleArtifact::operator=(
    VerifiedBundleArtifact&& other) noexcept {
  if (this != &other) {
    if (verified_file_descriptor_ >= 0) {
      static_cast<void>(::close(verified_file_descriptor_));
    }
    kind = other.kind;
    logical_name = std::move(other.logical_name);
    manifest_pointer = std::move(other.manifest_pointer);
    relative_path = std::move(other.relative_path);
    canonical_path = std::move(other.canonical_path);
    sha256 = std::move(other.sha256);
    size_bytes = other.size_bytes;
    verified_file_descriptor_ =
        std::exchange(other.verified_file_descriptor_, -1);
  }
  return *this;
}

int VerifiedBundleArtifact::verified_file_descriptor() const noexcept {
  return verified_file_descriptor_;
}

VerifiedRuntimeBundle::VerifiedRuntimeBundle(
    std::filesystem::path canonical_root_value,
    std::filesystem::path canonical_manifest_path_value,
    VerifiedBundleArtifact manifest_snapshot_value,
    RuntimeBundleManifest manifest_value,
    std::vector<VerifiedBundleArtifact> artifacts_value)
    : canonical_root(std::move(canonical_root_value)),
      canonical_manifest_path(
          std::move(canonical_manifest_path_value)),
      manifest_snapshot(std::move(manifest_snapshot_value)),
      manifest(std::move(manifest_value)),
      artifacts(std::move(artifacts_value)) {}

std::vector<VerifiedBundleArtifact> verify_runtime_bundle_files(
    const std::filesystem::path& bundle_root,
    const RuntimeBundleManifest& manifest) {
  return verify_runtime_bundle_files_at_canonical_root(
      canonical_bundle_root(bundle_root), manifest);
}

VerifiedRuntimeBundle load_and_verify_runtime_bundle(
    const std::filesystem::path& bundle_root,
    const std::string_view expected_manifest_sha256) {
  return load_and_verify_runtime_bundle(
      bundle_root,
      std::filesystem::path(kDefaultRuntimeBundleManifestPath),
      expected_manifest_sha256);
}

VerifiedRuntimeBundle load_and_verify_runtime_bundle(
    const std::filesystem::path& bundle_root,
    const std::filesystem::path& manifest_relative_path,
    const std::string_view expected_manifest_sha256) {
  if (!is_lowercase_sha256(expected_manifest_sha256)) {
    fail(
        BundleStage::sha256,
        BundleErrorCode::invalid_digest,
        manifest_relative_path,
        "/",
        "trusted manifest SHA-256 must be exactly 64 lowercase hexadecimal characters");
  }
  const std::filesystem::path canonical_root =
      canonical_bundle_root(bundle_root);
  OpenedRegularFile manifest_file = open_regular_bundle_file(
      canonical_root,
      manifest_relative_path,
      BundleStage::manifest_file,
      "/");
  Sha256SnapshotResult manifest_hash = snapshot_and_sha256_file(
      manifest_file,
      "/",
      BundleStage::manifest_file,
      kMaximumRuntimeBundleManifestBytes);
  if (manifest_hash.digest != expected_manifest_sha256) {
    fail(
        BundleStage::sha256,
        BundleErrorCode::digest_mismatch,
        manifest_file.canonical_path,
        "/",
        "trusted manifest digest does not match the exact parsed snapshot");
  }
  RuntimeBundleManifest manifest = parse_runtime_bundle_manifest(
      read_complete_file(
          manifest_hash.sealed_snapshot.get(),
          manifest_file.canonical_path,
          "/"));
  if (::lseek(manifest_hash.sealed_snapshot.get(), 0, SEEK_SET) < 0) {
    const int seek_error = errno;
    fail(
        BundleStage::manifest_file,
        BundleErrorCode::io_error,
        manifest_file.canonical_path,
        "/",
        system_error_detail("lseek retained manifest snapshot", seek_error));
  }
  VerifiedBundleArtifact manifest_snapshot =
      detail::BundleVerifierAccess::adopt_verified_descriptor(
          BundleArtifactKind::manifest,
          "runtime_bundle_manifest",
          "/",
          manifest_relative_path,
          manifest_file.canonical_path,
          manifest_hash.digest,
          manifest_hash.size_bytes,
          manifest_hash.sealed_snapshot.get());
  static_cast<void>(manifest_hash.sealed_snapshot.release());
  std::vector<VerifiedBundleArtifact> artifacts =
      verify_runtime_bundle_files_at_canonical_root(
          canonical_root, manifest);
  return VerifiedRuntimeBundle(
      canonical_root,
      manifest_file.canonical_path,
      std::move(manifest_snapshot),
      std::move(manifest),
      std::move(artifacts));
}

}  // namespace magpie_tts_rt
