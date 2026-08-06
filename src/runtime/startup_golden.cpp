#include "runtime/startup_golden.hpp"

#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <unistd.h>

#include "runtime/request_state.hpp"
#include "validation/character_validation.hpp"

namespace magpie_tts_rt {
namespace {

using Json = nlohmann::json;

[[nodiscard]] std::string error_message(
    const StartupGoldenErrorCode code,
    const std::string_view detail) {
  return "startup golden gate failed [code=" +
         std::string(to_string(code)) + "]: " + std::string(detail);
}

[[noreturn]] void fail(
    const StartupGoldenErrorCode code,
    const std::string& detail) {
  throw StartupGoldenError(code, detail);
}

[[nodiscard]] std::string openssl_error(
    const std::string_view operation) {
  const unsigned long code = ERR_get_error();
  if (code == 0) {
    return std::string(operation) + " failed";
  }
  std::array<char, 256> buffer{};
  ERR_error_string_n(code, buffer.data(), buffer.size());
  return std::string(operation) + " failed: " + buffer.data();
}

class DigestContext final {
 public:
  DigestContext()
      : context_(EVP_MD_CTX_new(), &EVP_MD_CTX_free) {
    if (context_ == nullptr) {
      fail(
          StartupGoldenErrorCode::invalid_capture,
          openssl_error("EVP_MD_CTX_new"));
    }
    if (EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1) {
      fail(
          StartupGoldenErrorCode::invalid_capture,
          openssl_error("EVP_DigestInit_ex"));
    }
  }

  void update(const std::span<const std::byte> bytes) {
    if (!bytes.empty() &&
        EVP_DigestUpdate(
            context_.get(), bytes.data(), bytes.size()) != 1) {
      fail(
          StartupGoldenErrorCode::invalid_capture,
          openssl_error("EVP_DigestUpdate"));
    }
  }

  [[nodiscard]] std::string finish() {
    std::array<unsigned char, 32> digest{};
    unsigned int digest_size = 0;
    if (EVP_DigestFinal_ex(
            context_.get(), digest.data(), &digest_size) != 1 ||
        digest_size != digest.size()) {
      fail(
          StartupGoldenErrorCode::invalid_capture,
          openssl_error("EVP_DigestFinal_ex"));
    }
    constexpr char digits[] = "0123456789abcdef";
    std::string encoded(digest.size() * 2U, '0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
      encoded[index * 2U] = digits[digest[index] >> 4U];
      encoded[index * 2U + 1U] = digits[digest[index] & 0x0FU];
    }
    return encoded;
  }

 private:
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_;
};

template <typename Unsigned>
void update_little_endian(
    DigestContext& digest,
    const Unsigned value) {
  static_assert(std::is_unsigned_v<Unsigned>);
  std::array<std::byte, sizeof(Unsigned)> bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::byte>(
        (value >> (index * 8U)) & static_cast<Unsigned>(0xFFU));
  }
  digest.update(bytes);
}

[[nodiscard]] std::string hash_int32_le(
    const std::span<const std::int32_t> values) {
  DigestContext digest;
  for (const std::int32_t value : values) {
    update_little_endian(
        digest, static_cast<std::uint32_t>(value));
  }
  return digest.finish();
}

[[nodiscard]] std::string hash_int64_le(
    const std::span<const std::int64_t> values) {
  DigestContext digest;
  for (const std::int64_t value : values) {
    update_little_endian(
        digest, static_cast<std::uint64_t>(value));
  }
  return digest.finish();
}

[[nodiscard]] std::string hash_float32_le(
    const std::span<const float> values) {
  static_assert(sizeof(float) == sizeof(std::uint32_t));
  DigestContext digest;
  for (const float value : values) {
    update_little_endian(
        digest, std::bit_cast<std::uint32_t>(value));
  }
  return digest.finish();
}

[[nodiscard]] std::string child_path(
    const std::string_view parent,
    const std::string_view child) {
  std::string result(parent);
  result.push_back('/');
  for (const char value : child) {
    if (value == '~') {
      result += "~0";
    } else if (value == '/') {
      result += "~1";
    } else {
      result.push_back(value);
    }
  }
  return result;
}

void require_exact_keys(
    const Json& value,
    const std::string& path,
    const std::initializer_list<std::string_view> required_keys) {
  if (!value.is_object()) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        path + ": expected an object");
  }
  std::set<std::string, std::less<>> expected;
  for (const std::string_view key : required_keys) {
    expected.emplace(key);
  }
  for (const auto& [key, unused] : value.items()) {
    static_cast<void>(unused);
    if (!expected.contains(key)) {
      fail(
          StartupGoldenErrorCode::invalid_fixture,
          child_path(path, key) + ": unknown schema-v1 field");
    }
  }
  for (const std::string_view key : required_keys) {
    if (!value.contains(key)) {
      fail(
          StartupGoldenErrorCode::invalid_fixture,
          child_path(path, key) + ": required field is missing");
    }
  }
}

[[nodiscard]] const Json& member(
    const Json& value,
    const std::string_view key) {
  return value.at(std::string(key));
}

[[nodiscard]] std::string parse_sha256(
    const Json& value,
    const std::string& path) {
  if (!value.is_string()) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        path + ": expected a SHA-256 string");
  }
  const std::string parsed = value.get<std::string>();
  if (!character_validation::is_lowercase_sha256(parsed)) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        path + ": expected 64 lowercase hexadecimal characters");
  }
  return parsed;
}

[[nodiscard]] std::string parse_identifier(
    const Json& value,
    const std::string& path) {
  if (!value.is_string()) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        path + ": expected an identifier string");
  }
  const std::string parsed = value.get<std::string>();
  if (!character_validation::is_identifier(parsed)) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        path + ": identifier does not match schema version 1");
  }
  return parsed;
}

template <typename Unsigned>
[[nodiscard]] Unsigned parse_unsigned(
    const Json& value,
    const std::string& path,
    const Unsigned minimum = 0) {
  static_assert(std::is_unsigned_v<Unsigned>);
  std::uint64_t parsed = 0;
  if (value.is_number_unsigned()) {
    parsed = value.get<std::uint64_t>();
  } else if (value.is_number_integer()) {
    const std::int64_t signed_value = value.get<std::int64_t>();
    if (signed_value < 0) {
      fail(
          StartupGoldenErrorCode::invalid_fixture,
          path + ": value must be non-negative");
    }
    parsed = static_cast<std::uint64_t>(signed_value);
  } else {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        path + ": expected an integer");
  }
  if (parsed < static_cast<std::uint64_t>(minimum) ||
      parsed >
          static_cast<std::uint64_t>(
              std::numeric_limits<Unsigned>::max())) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        path + ": integer is outside the accepted range");
  }
  return static_cast<Unsigned>(parsed);
}

void require_equal(
    const std::string_view actual,
    const std::string_view expected,
    const std::string_view field) {
  if (actual != expected) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        "golden fixture " + std::string(field) +
            " differs from the authenticated manifest");
  }
}

[[nodiscard]] std::string read_fixture_snapshot(
    const VerifiedBundleArtifact& artifact) {
  if (artifact.size_bytes == 0 ||
      artifact.size_bytes > kMaximumStartupGoldenFixtureBytes ||
      artifact.size_bytes >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::size_t>::max())) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        "golden fixture snapshot size is outside [1, 1 MiB]");
  }
  const int descriptor = artifact.verified_file_descriptor();
  if (descriptor < 0) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        "golden fixture has no retained verified descriptor");
  }
  std::string contents(
      static_cast<std::size_t>(artifact.size_bytes), '\0');
  std::size_t offset = 0;
  while (offset < contents.size()) {
    const ssize_t count = ::pread(
        descriptor,
        contents.data() + offset,
        contents.size() - offset,
        static_cast<off_t>(offset));
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      fail(
          StartupGoldenErrorCode::invalid_fixture,
          "unable to read retained golden fixture snapshot: errno=" +
              std::to_string(errno));
    }
    if (count == 0) {
      fail(
          StartupGoldenErrorCode::invalid_fixture,
          "retained golden fixture snapshot ended before its verified size");
    }
    offset += static_cast<std::size_t>(count);
  }
  return contents;
}

}  // namespace

std::string_view to_string(
    const StartupGoldenErrorCode code) noexcept {
  switch (code) {
    case StartupGoldenErrorCode::invalid_fixture:
      return "invalid_fixture";
    case StartupGoldenErrorCode::invalid_capture:
      return "invalid_capture";
    case StartupGoldenErrorCode::count_mismatch:
      return "count_mismatch";
    case StartupGoldenErrorCode::hash_mismatch:
      return "hash_mismatch";
  }
  return "unknown";
}

StartupGoldenError::StartupGoldenError(
    const StartupGoldenErrorCode code,
    std::string detail)
    : std::runtime_error(error_message(code, detail)),
      code_(code),
      detail_(std::move(detail)) {}

StartupGoldenErrorCode StartupGoldenError::code() const noexcept {
  return code_;
}

const std::string& StartupGoldenError::detail() const noexcept {
  return detail_;
}

StartupGoldenFixture parse_startup_golden_fixture(
    const std::string_view json_text,
    const RuntimeBundleManifest& manifest) {
  if (json_text.empty() ||
      json_text.size() > kMaximumStartupGoldenFixtureBytes) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        "golden fixture JSON size is outside [1, 1 MiB]");
  }

  Json root;
  std::unordered_map<int, std::set<std::string, std::less<>>>
      object_keys;
  const Json::parser_callback_t reject_duplicate_keys =
      [&object_keys](
          const int depth,
          const Json::parse_event_t event,
          Json& parsed) {
        if (event == Json::parse_event_t::object_start) {
          object_keys[depth].clear();
        } else if (event == Json::parse_event_t::key) {
          const int object_depth = depth - 1;
          const std::string key = parsed.get<std::string>();
          if (!object_keys[object_depth].emplace(key).second) {
            fail(
                StartupGoldenErrorCode::invalid_fixture,
                child_path("", key) +
                    ": duplicate JSON object key");
          }
        } else if (event == Json::parse_event_t::object_end) {
          object_keys.erase(depth);
        }
        return true;
      };
  try {
    root = Json::parse(
        json_text.begin(),
        json_text.end(),
        reject_duplicate_keys,
        true,
        false);
  } catch (const Json::parse_error& error) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        "invalid golden fixture JSON: " + std::string(error.what()));
  }

  require_exact_keys(
      root,
      "",
      {"schema_version",
       "fixture_id",
       "prepared_token_ids",
       "seed",
       "tokenizer_identity_sha256",
       "oracle_lock_sha256",
       "normalized_text_sha256",
       "token_ids_sha256",
       "baked_context_sha256",
       "expected"});
  const Json& expected_json = member(root, "expected");
  require_exact_keys(
      expected_json,
      "/expected",
      {"decoder_tokens_sha256",
       "codec_codes_sha256",
       "codec_frame_count",
       "pcm_f32le_sha256",
       "pcm_sample_count"});

  const std::uint32_t schema_version =
      parse_unsigned<std::uint32_t>(
          member(root, "schema_version"), "/schema_version", 1);
  if (schema_version != 1) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        "/schema_version: only schema version 1 is accepted");
  }
  const Json& prepared_json = member(root, "prepared_token_ids");
  if (!prepared_json.is_array() || prepared_json.empty() ||
      prepared_json.size() > manifest.limits.maximum_text_tokens) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        "/prepared_token_ids: token count is outside the authenticated limit");
  }
  std::vector<std::int32_t> prepared_token_ids;
  prepared_token_ids.reserve(prepared_json.size());
  for (std::size_t index = 0; index < prepared_json.size(); ++index) {
    const std::uint32_t token = parse_unsigned<std::uint32_t>(
        prepared_json.at(index),
        "/prepared_token_ids/" + std::to_string(index));
    prepared_token_ids.push_back(
        static_cast<std::int32_t>(token));
  }
  const PreparedTokenValidation token_validation =
      validate_prepared_token_ids(
          prepared_token_ids,
          manifest.artifacts.tokenizer.tokenizer_vocabulary_size,
          manifest.artifacts.tokenizer.eos_token_id);
  if (!token_validation.valid()) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        "/prepared_token_ids/" +
            std::to_string(token_validation.index) +
            ": sequence violates the authenticated prepared frontend "
            "contract");
  }

  StartupGoldenFixture fixture{
      .schema_version = schema_version,
      .fixture_id =
          parse_identifier(member(root, "fixture_id"), "/fixture_id"),
      .prepared_token_ids = std::move(prepared_token_ids),
      .seed = parse_unsigned<std::uint32_t>(
          member(root, "seed"), "/seed"),
      .tokenizer_identity_sha256 = parse_sha256(
          member(root, "tokenizer_identity_sha256"),
          "/tokenizer_identity_sha256"),
      .oracle_lock_sha256 = parse_sha256(
          member(root, "oracle_lock_sha256"),
          "/oracle_lock_sha256"),
      .normalized_text_sha256 = parse_sha256(
          member(root, "normalized_text_sha256"),
          "/normalized_text_sha256"),
      .token_ids_sha256 = parse_sha256(
          member(root, "token_ids_sha256"),
          "/token_ids_sha256"),
      .baked_context_sha256 = parse_sha256(
          member(root, "baked_context_sha256"),
          "/baked_context_sha256"),
      .expected =
          StartupGoldenExpected{
              .decoder_tokens_sha256 = parse_sha256(
                  member(expected_json, "decoder_tokens_sha256"),
                  "/expected/decoder_tokens_sha256"),
              .codec_codes_sha256 = parse_sha256(
                  member(expected_json, "codec_codes_sha256"),
                  "/expected/codec_codes_sha256"),
              .codec_frame_count =
                  parse_unsigned<std::uint64_t>(
                      member(expected_json, "codec_frame_count"),
                      "/expected/codec_frame_count",
                      1),
              .pcm_f32le_sha256 = parse_sha256(
                  member(expected_json, "pcm_f32le_sha256"),
                  "/expected/pcm_f32le_sha256"),
              .pcm_sample_count =
                  parse_unsigned<std::uint64_t>(
                      member(expected_json, "pcm_sample_count"),
                      "/expected/pcm_sample_count",
                      1),
          },
  };

  if (hash_int32_le(fixture.prepared_token_ids) !=
      fixture.token_ids_sha256) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        "prepared token bytes do not match token_ids_sha256");
  }
  require_equal(
      fixture.tokenizer_identity_sha256,
      manifest.artifacts.tokenizer.identity_sha256,
      "tokenizer identity");
  require_equal(
      fixture.normalized_text_sha256,
      manifest.golden_receipt.normalized_text_sha256,
      "normalized text digest");
  require_equal(
      fixture.token_ids_sha256,
      manifest.golden_receipt.token_ids_sha256,
      "token digest");
  require_equal(
      fixture.baked_context_sha256,
      manifest.golden_receipt.baked_context_sha256,
      "baked context digest");
  require_equal(
      fixture.expected.decoder_tokens_sha256,
      manifest.golden_receipt.decoder_tokens_sha256,
      "decoder token digest");
  require_equal(
      fixture.expected.codec_codes_sha256,
      manifest.golden_receipt.codec_codes_sha256,
      "codec code digest");
  require_equal(
      fixture.expected.pcm_f32le_sha256,
      manifest.golden_receipt.pcm_f32le_sha256,
      "PCM digest");
  if (fixture.seed != manifest.golden_receipt.seed ||
      fixture.expected.codec_frame_count !=
          manifest.golden_receipt.codec_frame_count ||
      fixture.expected.pcm_sample_count !=
          manifest.golden_receipt.sample_count) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        "golden fixture counts or seed differ from the authenticated receipt");
  }
  return fixture;
}

StartupGoldenFixture load_startup_golden_fixture(
    const VerifiedRuntimeBundle& bundle) {
  const VerifiedBundleArtifact* fixture_artifact = nullptr;
  for (const VerifiedBundleArtifact& artifact : bundle.artifacts) {
    if (artifact.kind != BundleArtifactKind::golden_fixture) {
      continue;
    }
    if (fixture_artifact != nullptr) {
      fail(
          StartupGoldenErrorCode::invalid_fixture,
          "verified bundle contains more than one golden fixture");
    }
    fixture_artifact = &artifact;
  }
  if (fixture_artifact == nullptr) {
    fail(
        StartupGoldenErrorCode::invalid_fixture,
        "verified bundle does not contain a golden fixture");
  }
  return parse_startup_golden_fixture(
      read_fixture_snapshot(*fixture_artifact), bundle.manifest);
}

StartupGoldenCapture::StartupGoldenCapture(
    const std::uint32_t codebooks_per_frame,
    const std::uint32_t codec_hop_length_samples,
    const std::uint32_t maximum_audio_frames)
    : codebooks_per_frame_(codebooks_per_frame),
      codec_hop_length_samples_(codec_hop_length_samples),
      maximum_audio_frames_(maximum_audio_frames),
      decoder_codebooks_(codebooks_per_frame) {
  if (codebooks_per_frame_ == 0 ||
      codec_hop_length_samples_ == 0 ||
      maximum_audio_frames_ == 0) {
    fail(
        StartupGoldenErrorCode::invalid_capture,
        "capture dimensions must be positive");
  }
  for (std::vector<std::int64_t>& codebook :
       decoder_codebooks_) {
    codebook.reserve(maximum_audio_frames_);
  }
  scheduled_codec_codes_.reserve(
      static_cast<std::size_t>(codebooks_per_frame_) *
      maximum_audio_frames_);
  pcm_.reserve(
      static_cast<std::size_t>(codec_hop_length_samples_) *
      maximum_audio_frames_);
}

void StartupGoldenCapture::record_chunk(
    const std::span<const std::int64_t> codec_codes,
    const std::uint32_t codec_frame_count,
    const std::span<const float> pcm) {
  if (codec_frame_count == 0 ||
      codec_frame_count >
          maximum_audio_frames_ - codec_frame_count_) {
    fail(
        StartupGoldenErrorCode::invalid_capture,
        "captured codec frames exceed the authenticated limit");
  }
  const std::uint64_t expected_codes =
      static_cast<std::uint64_t>(codebooks_per_frame_) *
      codec_frame_count;
  const std::uint64_t expected_samples =
      static_cast<std::uint64_t>(codec_hop_length_samples_) *
      codec_frame_count;
  if (codec_codes.size() != expected_codes ||
      pcm.size() != expected_samples) {
    fail(
        StartupGoldenErrorCode::invalid_capture,
        "captured code or PCM dimensions differ from the codec contract");
  }
  for (const float sample : pcm) {
    if (!std::isfinite(sample)) {
      fail(
          StartupGoldenErrorCode::invalid_capture,
          "captured PCM contains a non-finite sample");
    }
  }
  scheduled_codec_codes_.insert(
      scheduled_codec_codes_.end(),
      codec_codes.begin(),
      codec_codes.end());
  for (std::size_t codebook = 0;
       codebook < codebooks_per_frame_;
       ++codebook) {
    const std::size_t begin =
        codebook * codec_frame_count;
    decoder_codebooks_.at(codebook).insert(
        decoder_codebooks_.at(codebook).end(),
        codec_codes.begin() +
            static_cast<std::ptrdiff_t>(begin),
        codec_codes.begin() +
            static_cast<std::ptrdiff_t>(
                begin + codec_frame_count));
  }
  pcm_.insert(pcm_.end(), pcm.begin(), pcm.end());
  codec_frame_count_ += codec_frame_count;
}

StartupGoldenActual StartupGoldenCapture::finish() const {
  if (codec_frame_count_ == 0) {
    fail(
        StartupGoldenErrorCode::invalid_capture,
        "startup capture contains no codec frames");
  }
  std::vector<std::int64_t> decoder_tokens;
  decoder_tokens.reserve(
      static_cast<std::size_t>(codebooks_per_frame_) *
      static_cast<std::size_t>(codec_frame_count_));
  for (const std::vector<std::int64_t>& codebook :
       decoder_codebooks_) {
    if (codebook.size() != codec_frame_count_) {
      fail(
          StartupGoldenErrorCode::invalid_capture,
          "decoder codebook lengths are inconsistent");
    }
    decoder_tokens.insert(
        decoder_tokens.end(), codebook.begin(), codebook.end());
  }
  return StartupGoldenActual{
      .decoder_tokens_sha256 = hash_int64_le(decoder_tokens),
      .codec_codes_sha256 =
          hash_int64_le(scheduled_codec_codes_),
      .codec_frame_count = codec_frame_count_,
      .pcm_f32le_sha256 = hash_float32_le(pcm_),
      .pcm_sample_count = pcm_.size(),
  };
}

void require_startup_golden_match(
    const StartupGoldenFixture& expected,
    const StartupGoldenActual& actual) {
  if (actual.codec_frame_count !=
          expected.expected.codec_frame_count ||
      actual.pcm_sample_count !=
          expected.expected.pcm_sample_count) {
    fail(
        StartupGoldenErrorCode::count_mismatch,
        "generated codec-frame or PCM-sample count differs from golden");
  }
  if (actual.decoder_tokens_sha256 !=
      expected.expected.decoder_tokens_sha256) {
    fail(
        StartupGoldenErrorCode::hash_mismatch,
        "generated decoder token SHA-256 differs from golden");
  }
  if (actual.codec_codes_sha256 !=
      expected.expected.codec_codes_sha256) {
    fail(
        StartupGoldenErrorCode::hash_mismatch,
        "scheduled NanoCodec input SHA-256 differs from golden");
  }
  if (actual.pcm_f32le_sha256 !=
      expected.expected.pcm_f32le_sha256) {
    fail(
        StartupGoldenErrorCode::hash_mismatch,
        "generated PCM SHA-256 differs from golden");
  }
}

}  // namespace magpie_tts_rt
