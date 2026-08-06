#include "bundle/bundle.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using Json = nlohmann::json;
using magpie_tts_rt::BundleArtifactKind;
using magpie_tts_rt::BundleError;
using magpie_tts_rt::BundleErrorCode;
using magpie_tts_rt::BundleStage;
using magpie_tts_rt::RuntimeBundleManifest;

constexpr std::string_view kFileContents = "abc";
constexpr std::string_view kFileSha256 =
    "ba7816bf8f01cfea414140de5dae2223"
    "b00361a396177a9cb410ff61f20015ad";
constexpr std::string_view kZeroSha256 =
    "00000000000000000000000000000000"
    "00000000000000000000000000000000";

static_assert(
    !std::is_copy_constructible_v<
        magpie_tts_rt::VerifiedBundleArtifact>);
static_assert(
    !std::is_copy_assignable_v<
        magpie_tts_rt::VerifiedBundleArtifact>);
static_assert(
    std::is_nothrow_move_constructible_v<
        magpie_tts_rt::VerifiedBundleArtifact>);
static_assert(
    std::is_nothrow_move_assignable_v<
        magpie_tts_rt::VerifiedBundleArtifact>);
static_assert(
    !std::is_copy_constructible_v<
        magpie_tts_rt::VerifiedRuntimeBundle>);
static_assert(
    std::is_nothrow_move_constructible_v<
        magpie_tts_rt::VerifiedRuntimeBundle>);

[[noreturn]] void test_failure(const std::string& detail) {
  throw std::runtime_error("test assertion failed: " + detail);
}

void require(const bool condition, const std::string& detail) {
  if (!condition) {
    test_failure(detail);
  }
}

[[nodiscard]] std::string read_file(
    const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream.is_open()) {
    throw std::runtime_error("unable to open fixture: " + path.string());
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  if (stream.bad()) {
    throw std::runtime_error("unable to read fixture: " + path.string());
  }
  return contents.str();
}

[[nodiscard]] std::string sha256_text(const std::string_view contents) {
  std::array<unsigned char, 32> digest{};
  std::size_t digest_size = digest.size();
  if (EVP_Q_digest(
          nullptr,
          "SHA256",
          nullptr,
          contents.data(),
          contents.size(),
          digest.data(),
          &digest_size) != 1 ||
      digest_size != digest.size()) {
    throw std::runtime_error("unable to compute test SHA-256");
  }
  constexpr std::array<char, 16> hexadecimal{
      '0', '1', '2', '3', '4', '5', '6', '7',
      '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  std::string result(digest.size() * 2, '0');
  for (std::size_t index = 0; index < digest.size(); ++index) {
    result.at(index * 2) = hexadecimal.at(digest.at(index) >> 4U);
    result.at(index * 2 + 1) =
        hexadecimal.at(digest.at(index) & 0x0FU);
  }
  return result;
}

[[nodiscard]] std::string sha256_file(
    const std::filesystem::path& path) {
  return sha256_text(read_file(path));
}

void write_file(
    const std::filesystem::path& path,
    const std::string_view contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream.is_open()) {
    throw std::runtime_error("unable to create test file: " + path.string());
  }
  stream.write(
      contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!stream) {
    throw std::runtime_error("unable to write test file: " + path.string());
  }
}

void write_oversized_manifest(const std::filesystem::path& path) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream.is_open()) {
    throw std::runtime_error(
        "unable to create oversized manifest: " + path.string());
  }
  stream.seekp(
      static_cast<std::streamoff>(
          magpie_tts_rt::kMaximumRuntimeBundleManifestBytes));
  stream.put('\n');
  if (!stream) {
    throw std::runtime_error(
        "unable to write oversized manifest: " + path.string());
  }
}

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "magpie-tts-rt-bundle-tests-XXXXXX")
            .string();
    char* const created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

void configure_file_artifact(Json& file) {
  file.at("sha256") = kFileSha256;
  file.at("size_bytes") = kFileContents.size();
}

[[nodiscard]] std::vector<Json*> json_file_artifacts(Json& document) {
  std::vector<Json*> files;
  files.reserve(15 + document.at("engines").size());
  files.push_back(
      &document.at("artifacts").at("source_model").at("acceptance_receipt"));
  files.push_back(
      &document.at("artifacts").at("export").at("export_receipt"));
  files.push_back(
      &document.at("artifacts").at("tokenizer").at("identity_receipt"));
  files.push_back(
      &document.at("artifacts").at("plugin").at("file"));
  files.push_back(
      &document.at("artifacts").at("plugin").at("build_receipt"));
  for (Json& license : document.at("licenses")) {
    files.push_back(&license.at("file"));
  }
  for (Json& engine : document.at("engines")) {
    files.push_back(&engine.at("file"));
  }
  files.push_back(&document.at("golden_fixture"));
  return files;
}

[[nodiscard]] RuntimeBundleManifest create_valid_bundle(
    const Json& fixture,
    const std::filesystem::path& root) {
  std::filesystem::create_directories(root);
  Json document = fixture;
  for (Json* const file : json_file_artifacts(document)) {
    configure_file_artifact(*file);
    write_file(
        root / file->at("path").get<std::string>(), kFileContents);
  }
  document.at("golden_receipt").at("sha256") = kFileSha256;
  document.at("golden_receipt").at("size_bytes") =
      kFileContents.size();
  document.at("limits").at("maximum_bundle_snapshot_bytes") =
      (document.at("engines").size() + 15U) * kFileContents.size();
  write_file(
      root /
          document.at("golden_receipt").at("path").get<std::string>(),
      kFileContents);
  write_file(
      root /
          std::filesystem::path(
              magpie_tts_rt::kDefaultRuntimeBundleManifestPath),
      document.dump(2));
  return magpie_tts_rt::parse_runtime_bundle_manifest(document.dump());
}

void expect_bundle_error(
    const std::string_view name,
    const BundleStage expected_stage,
    const BundleErrorCode expected_code,
    const std::string_view expected_manifest_pointer,
    const std::function<void()>& operation) {
  try {
    operation();
  } catch (const BundleError& error) {
    require(
        error.stage() == expected_stage,
        std::string(name) + ": unexpected stage " +
            std::string(magpie_tts_rt::to_string(error.stage())));
    require(
        error.code() == expected_code,
        std::string(name) + ": unexpected code " +
            std::string(magpie_tts_rt::to_string(error.code())));
    require(
        error.manifest_pointer() == expected_manifest_pointer,
        std::string(name) + ": unexpected manifest pointer " +
            error.manifest_pointer());
    require(!error.path().empty(), std::string(name) + ": empty error path");
    require(!error.detail().empty(), std::string(name) + ": empty error detail");
    return;
  }
  test_failure(std::string(name) + ": expected BundleError");
}

void test_known_hash_and_complete_artifact_set(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  const RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);

  const std::filesystem::path manifest_path =
      root /
      std::filesystem::path(
          magpie_tts_rt::kDefaultRuntimeBundleManifestPath);
  const std::string trusted_manifest_sha256 = sha256_file(manifest_path);
  const magpie_tts_rt::VerifiedRuntimeBundle bundle =
      magpie_tts_rt::load_and_verify_runtime_bundle(
          root, trusted_manifest_sha256);
  require(
      bundle.artifacts.size() == manifest.engines.size() + 15,
      "all artifact classes were verified");
  require(
      bundle.canonical_root == std::filesystem::canonical(root),
      "canonical bundle root");
  require(
      bundle.canonical_manifest_path ==
          std::filesystem::canonical(manifest_path),
      "canonical manifest path");
  require(
      bundle.manifest_snapshot.kind == BundleArtifactKind::manifest,
      "manifest has a dedicated immutable snapshot");
  require(
      bundle.manifest_snapshot.sha256 == trusted_manifest_sha256,
      "manifest snapshot matches the external trust anchor");

  std::size_t model_count = 0;
  std::size_t export_count = 0;
  std::size_t tokenizer_count = 0;
  std::size_t plugin_build_receipt_count = 0;
  std::size_t plugin_count = 0;
  std::size_t license_count = 0;
  std::size_t engine_count = 0;
  std::size_t fixture_count = 0;
  std::size_t receipt_count = 0;
  for (const auto& artifact : bundle.artifacts) {
    require(artifact.sha256 == kFileSha256, "known SHA-256");
    require(artifact.size_bytes == kFileContents.size(), "streamed byte count");
    require(
        std::filesystem::is_regular_file(artifact.canonical_path),
        "verified canonical file");
    switch (artifact.kind) {
      case BundleArtifactKind::manifest:
        test_failure("manifest must not be duplicated in artifact vector");
      case BundleArtifactKind::source_model_receipt:
        ++model_count;
        break;
      case BundleArtifactKind::export_receipt:
        ++export_count;
        break;
      case BundleArtifactKind::tokenizer_identity_receipt:
        ++tokenizer_count;
        break;
      case BundleArtifactKind::plugin_build_receipt:
        ++plugin_build_receipt_count;
        break;
      case BundleArtifactKind::plugin:
        ++plugin_count;
        break;
      case BundleArtifactKind::license:
        ++license_count;
        break;
      case BundleArtifactKind::engine:
        ++engine_count;
        break;
      case BundleArtifactKind::golden_fixture:
        ++fixture_count;
        break;
      case BundleArtifactKind::golden_receipt:
        ++receipt_count;
        break;
    }
  }
  require(model_count == 1, "model artifact");
  require(export_count == 1, "export artifact");
  require(tokenizer_count == 1, "tokenizer artifact");
  require(plugin_build_receipt_count == 1, "plugin build receipt artifact");
  require(plugin_count == 1, "plugin artifact");
  require(license_count == manifest.licenses.size(), "every license artifact");
  require(engine_count == manifest.engines.size(), "every engine artifact");
  require(fixture_count == 1, "golden fixture artifact");
  require(receipt_count == 1, "golden receipt artifact");

  const std::filesystem::path explicit_manifest =
      root / "metadata/runtime.json";
  std::filesystem::create_directories(explicit_manifest.parent_path());
  std::filesystem::rename(manifest_path, explicit_manifest);
  const magpie_tts_rt::VerifiedRuntimeBundle explicit_bundle =
      magpie_tts_rt::load_and_verify_runtime_bundle(
          root,
          "metadata/runtime.json",
          sha256_file(explicit_manifest));
  require(
      explicit_bundle.canonical_manifest_path ==
          std::filesystem::canonical(explicit_manifest),
      "explicit manifest path");
}

void test_exact_bundle_entry_inventory(const Json& fixture) {
  const auto expect_extra_entry =
      [&fixture](
          const std::string_view name,
          const std::function<void(const std::filesystem::path&)>& add_extra) {
        TemporaryDirectory temporary;
        const std::filesystem::path root = temporary.path() / "bundle";
        static_cast<void>(create_valid_bundle(fixture, root));
        add_extra(root);
        expect_bundle_error(
            name,
            BundleStage::artifact_path,
            BundleErrorCode::unexpected_entry,
            "/",
            [&root] {
              const std::filesystem::path manifest_path =
                  root /
                  std::filesystem::path(
                      magpie_tts_rt::kDefaultRuntimeBundleManifestPath);
              static_cast<void>(
                  magpie_tts_rt::load_and_verify_runtime_bundle(
                      root, sha256_file(manifest_path)));
            });
      };

  expect_extra_entry(
      "unlisted regular file",
      [](const std::filesystem::path& root) {
        write_file(root / "extra.bin", "extra");
      });
  expect_extra_entry(
      "unlisted empty directory",
      [](const std::filesystem::path& root) {
        std::filesystem::create_directory(root / "extra");
      });
  expect_extra_entry(
      "unlisted symbolic link",
      [](const std::filesystem::path& root) {
        std::filesystem::create_symlink(
            root /
                std::filesystem::path(
                    magpie_tts_rt::kDefaultRuntimeBundleManifestPath),
            root / "extra-link");
      });
  expect_extra_entry(
      "unlisted special file",
      [](const std::filesystem::path& root) {
        if (::mkfifo((root / "extra-fifo").c_str(), 0600) != 0) {
          throw std::runtime_error("mkfifo failed");
        }
      });
}

void test_optional_manifest_digest_transport(const Json& fixture) {
  {
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path() / "bundle";
    static_cast<void>(create_valid_bundle(fixture, root));
    const std::filesystem::path manifest_path =
        root /
        std::filesystem::path(
            magpie_tts_rt::kDefaultRuntimeBundleManifestPath);
    const std::string manifest_sha256 = sha256_file(manifest_path);
    write_file(
        root / "runtime-bundle-manifest.json.sha256",
        manifest_sha256 + "  runtime-bundle-manifest.json\n");
    static_cast<void>(
        magpie_tts_rt::load_and_verify_runtime_bundle(
            root, manifest_sha256));
  }
  {
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path() / "bundle";
    static_cast<void>(create_valid_bundle(fixture, root));
    const std::filesystem::path manifest_path =
        root /
        std::filesystem::path(
            magpie_tts_rt::kDefaultRuntimeBundleManifestPath);
    const std::string manifest_sha256 = sha256_file(manifest_path);
    write_file(
        root / "runtime-bundle-manifest.json.sha256",
        std::string(kZeroSha256) + "  runtime-bundle-manifest.json\n");
    expect_bundle_error(
        "incorrect manifest digest transport",
        BundleStage::sha256,
        BundleErrorCode::digest_mismatch,
        "/",
        [&root, &manifest_sha256] {
          static_cast<void>(
              magpie_tts_rt::load_and_verify_runtime_bundle(
                  root, manifest_sha256));
        });
  }
}

void test_missing_artifact(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  const RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
  std::filesystem::remove(root / manifest.artifacts.source_model.acceptance_receipt.path);
  expect_bundle_error(
      "missing artifact",
      BundleStage::artifact_file,
      BundleErrorCode::not_found,
      "/artifacts/source_model/acceptance_receipt/path",
      [&root, &manifest] {
        static_cast<void>(
            magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
      });
}

void test_missing_or_mutated_license_fails_closed(const Json& fixture) {
  {
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path() / "bundle";
    const RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
    std::filesystem::remove(root / manifest.licenses.front().file.path);
    expect_bundle_error(
        "missing license",
        BundleStage::artifact_file,
        BundleErrorCode::not_found,
        "/licenses/0/file/path",
        [&root, &manifest] {
          static_cast<void>(
              magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
        });
  }
  {
    TemporaryDirectory temporary;
    const std::filesystem::path root = temporary.path() / "bundle";
    const RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
    write_file(root / manifest.licenses.front().file.path, "xyz");
    expect_bundle_error(
        "mutated license",
        BundleStage::sha256,
        BundleErrorCode::digest_mismatch,
        "/licenses/0/file/sha256",
        [&root, &manifest] {
          static_cast<void>(
              magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
        });
  }
}

void test_manifest_trust_anchor_mismatch(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  static_cast<void>(create_valid_bundle(fixture, root));
  expect_bundle_error(
      "manifest trust anchor mismatch",
      BundleStage::sha256,
      BundleErrorCode::digest_mismatch,
      "/",
      [&root] {
        static_cast<void>(
            magpie_tts_rt::load_and_verify_runtime_bundle(
                root, kZeroSha256));
      });
}

struct DigestField {
  std::string* value;
  std::string pointer;
};

[[nodiscard]] std::vector<DigestField> digest_fields(
    RuntimeBundleManifest& manifest) {
  std::vector<DigestField> fields;
  fields.reserve(15 + manifest.engines.size());
  fields.push_back(
      {&manifest.artifacts.source_model.acceptance_receipt.sha256,
       "/artifacts/source_model/acceptance_receipt/sha256"});
  fields.push_back(
      {&manifest.artifacts.export_artifact.export_receipt.sha256,
       "/artifacts/export/export_receipt/sha256"});
  fields.push_back(
      {&manifest.artifacts.tokenizer.identity_receipt.sha256,
       "/artifacts/tokenizer/identity_receipt/sha256"});
  fields.push_back(
      {&manifest.artifacts.plugin.file.sha256,
       "/artifacts/plugin/file/sha256"});
  fields.push_back(
      {&manifest.artifacts.plugin.build_receipt.sha256,
       "/artifacts/plugin/build_receipt/sha256"});
  for (std::size_t index = 0; index < manifest.licenses.size(); ++index) {
    fields.push_back(
        {&manifest.licenses.at(index).file.sha256,
         "/licenses/" + std::to_string(index) + "/file/sha256"});
  }
  for (std::size_t index = 0; index < manifest.engines.size(); ++index) {
    fields.push_back(
        {&manifest.engines.at(index).file.sha256,
         "/engines/" + std::to_string(index) + "/file/sha256"});
  }
  fields.push_back(
      {&manifest.golden_fixture.sha256, "/golden_fixture/sha256"});
  fields.push_back(
      {&manifest.golden_receipt.sha256, "/golden_receipt/sha256"});
  return fields;
}

void test_every_digest_is_verified(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
  const std::vector<DigestField> fields = digest_fields(manifest);
  for (const DigestField& field : fields) {
    const std::string valid_digest = *field.value;
    *field.value = std::string(64, '0');
    expect_bundle_error(
        "digest mismatch " + field.pointer,
        BundleStage::sha256,
        BundleErrorCode::digest_mismatch,
        field.pointer,
        [&root, &manifest] {
          static_cast<void>(
              magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
        });
    *field.value = valid_digest;
  }
}

void test_authenticated_artifact_size_is_verified(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
  ++manifest.artifacts.source_model.acceptance_receipt.size_bytes;
  ++manifest.limits.maximum_bundle_snapshot_bytes;
  expect_bundle_error(
      "authenticated artifact size mismatch",
      BundleStage::artifact_file,
      BundleErrorCode::size_mismatch,
      "/artifacts/source_model/acceptance_receipt/size_bytes",
      [&root, &manifest] {
        static_cast<void>(
            magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
      });
}

void test_snapshot_budget_is_verified(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
  ++manifest.limits.maximum_bundle_snapshot_bytes;
  expect_bundle_error(
      "snapshot budget mismatch",
      BundleStage::artifact_file,
      BundleErrorCode::size_mismatch,
      "/limits/maximum_bundle_snapshot_bytes",
      [&root, &manifest] {
        static_cast<void>(
            magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
      });
}

void test_uppercase_digest_is_rejected(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
  manifest.artifacts.source_model.acceptance_receipt.sha256 =
      "BA7816BF8F01CFEA414140DE5DAE2223"
      "B00361A396177A9CB410FF61F20015AD";
  expect_bundle_error(
      "uppercase digest",
      BundleStage::sha256,
      BundleErrorCode::invalid_digest,
      "/artifacts/source_model/acceptance_receipt/sha256",
      [&root, &manifest] {
        static_cast<void>(
            magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
      });
}

void test_parent_component_is_rejected(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
  write_file(temporary.path() / "outside.bin", kFileContents);
  manifest.artifacts.source_model.acceptance_receipt.path = "../outside.bin";
  expect_bundle_error(
      "parent component",
      BundleStage::artifact_path,
      BundleErrorCode::invalid_relative_path,
      "/artifacts/source_model/acceptance_receipt/path",
      [&root, &manifest] {
        static_cast<void>(
            magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
      });
}

void test_embedded_nul_is_rejected(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
  std::string path_with_nul = "model/model.nemo";
  path_with_nul.push_back('\0');
  path_with_nul += "suffix";
  manifest.artifacts.source_model.acceptance_receipt.path = path_with_nul;
  expect_bundle_error(
      "embedded NUL",
      BundleStage::artifact_path,
      BundleErrorCode::invalid_relative_path,
      "/artifacts/source_model/acceptance_receipt/path",
      [&root, &manifest] {
        static_cast<void>(
            magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
      });
}

void test_symlink_escape_is_rejected(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  const RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
  const std::filesystem::path model_path =
      root / manifest.artifacts.source_model.acceptance_receipt.path;
  const std::filesystem::path outside_path =
      temporary.path() / "outside.bin";
  write_file(outside_path, kFileContents);
  std::filesystem::remove(model_path);
  std::filesystem::create_symlink(outside_path, model_path);
  expect_bundle_error(
      "symlink escape",
      BundleStage::artifact_path,
      BundleErrorCode::path_escape,
      "/artifacts/source_model/acceptance_receipt/path",
      [&root, &manifest] {
        static_cast<void>(
            magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
      });
}

void test_duplicate_canonical_path_is_rejected(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  const RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
  const std::filesystem::path model_path =
      root / manifest.artifacts.source_model.acceptance_receipt.path;
  const std::filesystem::path export_path =
      root / manifest.artifacts.export_artifact.export_receipt.path;
  std::filesystem::remove(export_path);
  std::filesystem::create_symlink(model_path, export_path);
  expect_bundle_error(
      "duplicate canonical path",
      BundleStage::artifact_path,
      BundleErrorCode::duplicate_canonical_path,
      "/artifacts/export/export_receipt/path",
      [&root, &manifest] {
        static_cast<void>(
            magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
      });
}

void test_hard_link_alias_is_rejected(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  const RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
  const std::filesystem::path model_path =
      root / manifest.artifacts.source_model.acceptance_receipt.path;
  const std::filesystem::path export_path =
      root / manifest.artifacts.export_artifact.export_receipt.path;
  std::filesystem::remove(export_path);
  std::filesystem::create_hard_link(model_path, export_path);
  expect_bundle_error(
      "hard-link alias",
      BundleStage::artifact_path,
      BundleErrorCode::duplicate_file_identity,
      "/artifacts/export/export_receipt/path",
      [&root, &manifest] {
        static_cast<void>(
            magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
      });
}

void test_non_regular_artifact_is_rejected(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  const RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
  const std::filesystem::path model_path =
      root / manifest.artifacts.source_model.acceptance_receipt.path;
  std::filesystem::remove(model_path);
  std::filesystem::create_directory(model_path);
  expect_bundle_error(
      "non-regular artifact",
      BundleStage::artifact_file,
      BundleErrorCode::not_regular_file,
      "/artifacts/source_model/acceptance_receipt/path",
      [&root, &manifest] {
        static_cast<void>(
            magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
      });
}

void test_fifo_artifact_is_rejected_without_blocking(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  const RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
  const std::filesystem::path model_path =
      root / manifest.artifacts.source_model.acceptance_receipt.path;
  std::filesystem::remove(model_path);
  if (::mkfifo(model_path.c_str(), 0600) != 0) {
    throw std::runtime_error("mkfifo failed");
  }
  expect_bundle_error(
      "FIFO artifact",
      BundleStage::artifact_file,
      BundleErrorCode::not_regular_file,
      "/artifacts/source_model/acceptance_receipt/path",
      [&root, &manifest] {
        static_cast<void>(
            magpie_tts_rt::verify_runtime_bundle_files(root, manifest));
      });
}

void test_verified_descriptor_survives_path_replacement(
    const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  const RuntimeBundleManifest manifest = create_valid_bundle(fixture, root);
  int retained_descriptor = -1;
  {
    std::vector<magpie_tts_rt::VerifiedBundleArtifact> artifacts =
        magpie_tts_rt::verify_runtime_bundle_files(root, manifest);
    const auto model = std::find_if(
        artifacts.begin(), artifacts.end(), [](const auto& artifact) {
          return artifact.kind == BundleArtifactKind::source_model_receipt;
        });
    require(model != artifacts.end(), "verified model descriptor");
    retained_descriptor = model->verified_file_descriptor();
    require(retained_descriptor >= 0, "open verified descriptor");

    write_file(model->canonical_path, "xyz");
    errno = 0;
    require(
        ::pwrite(retained_descriptor, "z", 1, 0) == -1 && errno == EPERM,
        "verified snapshot is sealed against in-place writes");

    std::filesystem::remove(model->canonical_path);
    write_file(model->canonical_path, "replacement");

    std::string verified_contents(kFileContents.size(), '\0');
    const ssize_t read_size = ::pread(
        retained_descriptor,
        verified_contents.data(),
        verified_contents.size(),
        0);
    require(
        read_size ==
            static_cast<ssize_t>(verified_contents.size()),
        "read retained verified descriptor");
    require(
        verified_contents == kFileContents,
        "sealed descriptor still contains the verified bytes");
    require(
        read_file(model->canonical_path) == "replacement",
        "diagnostic path now identifies replacement file");
  }

  errno = 0;
  require(
      ::fcntl(retained_descriptor, F_GETFD) == -1 && errno == EBADF,
      "verified descriptor closed with artifact lifetime");
}

void test_bundle_root_requirements(const Json& fixture) {
  TemporaryDirectory temporary;
  RuntimeBundleManifest manifest =
      magpie_tts_rt::parse_runtime_bundle_manifest(fixture.dump());
  const std::filesystem::path missing = temporary.path() / "missing";
  expect_bundle_error(
      "missing root",
      BundleStage::bundle_root,
      BundleErrorCode::not_found,
      "/",
      [&missing, &manifest] {
        static_cast<void>(
            magpie_tts_rt::verify_runtime_bundle_files(missing, manifest));
      });

  const std::filesystem::path file = temporary.path() / "not-a-directory";
  write_file(file, kFileContents);
  expect_bundle_error(
      "non-directory root",
      BundleStage::bundle_root,
      BundleErrorCode::not_directory,
      "/",
      [&file, &manifest] {
        static_cast<void>(
            magpie_tts_rt::verify_runtime_bundle_files(file, manifest));
      });
}

void test_manifest_path_requirements(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  static_cast<void>(create_valid_bundle(fixture, root));
  expect_bundle_error(
      "manifest parent component",
      BundleStage::manifest_file,
      BundleErrorCode::invalid_relative_path,
      "/",
      [&root] {
        static_cast<void>(
            magpie_tts_rt::load_and_verify_runtime_bundle(
                root,
                "../runtime-bundle-manifest.json",
                kZeroSha256));
      });

  const std::filesystem::path manifest_path =
      root /
      std::filesystem::path(
          magpie_tts_rt::kDefaultRuntimeBundleManifestPath);
  const std::string manifest_contents = read_file(manifest_path);
  const std::filesystem::path outside_manifest =
      temporary.path() / "outside-manifest.json";
  write_file(outside_manifest, manifest_contents);
  std::filesystem::remove(manifest_path);
  std::filesystem::create_symlink(outside_manifest, manifest_path);
  expect_bundle_error(
      "manifest symlink escape",
      BundleStage::manifest_file,
      BundleErrorCode::path_escape,
      "/",
      [&root] {
        static_cast<void>(
            magpie_tts_rt::load_and_verify_runtime_bundle(
                root, kZeroSha256));
      });
}

void test_manifest_size_limit(const Json& fixture) {
  TemporaryDirectory temporary;
  const std::filesystem::path root = temporary.path() / "bundle";
  static_cast<void>(create_valid_bundle(fixture, root));
  write_oversized_manifest(
      root /
      std::filesystem::path(
          magpie_tts_rt::kDefaultRuntimeBundleManifestPath));
  expect_bundle_error(
      "manifest size limit",
      BundleStage::manifest_file,
      BundleErrorCode::size_limit_exceeded,
      "/",
      [&root] {
        static_cast<void>(
            magpie_tts_rt::load_and_verify_runtime_bundle(
                root, kZeroSha256));
      });
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: bundle_tests /path/to/minimal-valid.json\n";
    return EXIT_FAILURE;
  }

  try {
    const Json fixture = Json::parse(read_file(argv[1]));
    test_known_hash_and_complete_artifact_set(fixture);
    test_exact_bundle_entry_inventory(fixture);
    test_optional_manifest_digest_transport(fixture);
    test_manifest_trust_anchor_mismatch(fixture);
    test_missing_artifact(fixture);
    test_missing_or_mutated_license_fails_closed(fixture);
    test_every_digest_is_verified(fixture);
    test_authenticated_artifact_size_is_verified(fixture);
    test_snapshot_budget_is_verified(fixture);
    test_uppercase_digest_is_rejected(fixture);
    test_parent_component_is_rejected(fixture);
    test_embedded_nul_is_rejected(fixture);
    test_symlink_escape_is_rejected(fixture);
    test_duplicate_canonical_path_is_rejected(fixture);
    test_hard_link_alias_is_rejected(fixture);
    test_non_regular_artifact_is_rejected(fixture);
    test_fifo_artifact_is_rejected_without_blocking(fixture);
    test_verified_descriptor_survives_path_replacement(fixture);
    test_bundle_root_requirements(fixture);
    test_manifest_path_requirements(fixture);
    test_manifest_size_limit(fixture);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "bundle verification tests passed\n";
  return EXIT_SUCCESS;
}
