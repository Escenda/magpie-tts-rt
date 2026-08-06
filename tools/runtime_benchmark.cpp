// Build the smoke gate's ABI/stream validators into this executable so the
// benchmark cannot silently drift to a weaker acceptance contract.
#define MTT_RUNTIME_SMOKE_TESTING
#include "runtime_smoke.cpp"

#include <sys/resource.h>

#include <openssl/evp.h>

#include <numeric>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_set>

namespace {

[[maybe_unused]] constexpr auto* kRuntimeSmokeRun = &run;

constexpr std::uint64_t kAcceptanceNormalCases = 108U;
constexpr std::uint64_t kAcceptanceLongRunIterations = 1'000U;
constexpr double kTargetWarmTtfaP95Ms = 100.0;
constexpr double kTargetGenerationRtfP95 = 0.30;
constexpr double kRegressionWarmTtfaP95Ms = 152.39;
constexpr double kRegressionGenerationRtfP95 = 0.401;
constexpr std::uint64_t kMaximumLongRunProcessRssGrowthBytes =
    64U * 1'024U * 1'024U;
constexpr double
    kMaximumLongRunProcessRssSlopeBytesPerIteration =
        64.0 * 1'024.0;

struct BenchmarkArguments {
  Arguments native;
  std::filesystem::path corpus;
  std::filesystem::path cuda_runtime_library;
  std::filesystem::path nvml_library;
  std::uint32_t nvml_device_index = 0U;
  std::uint64_t long_run_iterations = 0U;
  bool help = false;
};

struct CorpusCase {
  std::string case_id;
  std::string source_text;
  std::string source_text_sha256;
  std::vector<std::int64_t> token_ids;
  std::uint64_t seed = 0U;
};

struct BenchmarkCorpus {
  std::string corpus_id;
  std::string tokenizer_identity_sha256;
  std::string file_sha256;
  std::vector<CorpusCase> normal_cases;
  CorpusCase cancel_case;
};

struct MemoryPoint {
  std::uint64_t process_rss_bytes = 0U;
  std::optional<std::uint64_t> cuda_total_bytes;
  std::optional<std::uint64_t> cuda_free_bytes;
  std::optional<std::uint64_t> cuda_used_bytes;
};

struct ChunkTiming {
  std::uint64_t sequence = 0U;
  std::uint64_t first_sample_index = 0U;
  std::uint64_t sample_count = 0U;
  std::uint64_t codec_frames = 0U;
  double arrival_ms = 0.0;
  std::optional<double> interval_ms;
  double positive_playback_lateness_ms = 0.0;
  bool first = false;
  bool final = false;
};

struct CaseResult {
  std::string case_id;
  std::uint64_t prepared_token_count = 0U;
  std::uint64_t random_seed = 0U;
  double request_start_call_ms = 0.0;
  double request_start_to_first_audio_ms = 0.0;
  double ttfa_ms = 0.0;
  double generation_ms = 0.0;
  double total_ms = 0.0;
  std::uint64_t audio_samples = 0U;
  double audio_seconds = 0.0;
  double generation_rtf = 0.0;
  double total_rtf = 0.0;
  double maximum_positive_playback_lateness_ms = 0.0;
  std::vector<ChunkTiming> chunks;
};

struct ProcessCpuTimes {
  double user_seconds = 0.0;
  double system_seconds = 0.0;
};

struct CancelResult {
  std::string case_id;
  std::string injection_point;
  std::uint64_t audio_chunks_before_cancel = 0U;
  std::optional<double> first_audio_ms;
  double cancel_call_ms = 0.0;
  double cancel_latency_ms = 0.0;
  std::uint64_t audio_samples_before_cancel = 0U;
  std::uint64_t post_cancel_lease_count = 0U;
  bool terminal_cancelled = false;
};

struct Distribution {
  double median = 0.0;
  double p95 = 0.0;
  double maximum = 0.0;
};

struct LongRunAcceptanceGates {
  bool ttfa_regression_passed = false;
  bool generation_regression_passed = false;
  bool playback_lateness_regression_passed = false;
  bool process_rss_growth_passed = false;
  bool process_rss_slope_passed = false;
};

struct LongRunIteration {
  std::uint64_t iteration = 0U;
  std::string case_id;
  double ttfa_ms = 0.0;
  double generation_rtf = 0.0;
  double total_rtf = 0.0;
  double maximum_positive_playback_lateness_ms = 0.0;
  MemoryPoint memory;
};

struct LongRunFailure {
  std::uint64_t iteration = 0U;
  std::string case_id;
  std::string diagnostic_sha256;
};

struct BenchmarkResults {
  double startup_gate_ms = 0.0;
  MemoryPoint before_runtime;
  MemoryPoint after_startup;
  MemoryPoint after_cold;
  MemoryPoint after_warm;
  MemoryPoint after_cancel;
  MemoryPoint after_long_run;
  MemoryPoint after_native_cleanup;
  CaseResult cold;
  std::vector<CaseResult> warm;
  std::vector<CancelResult> cancellations;
  std::vector<LongRunIteration> long_run;
  std::optional<LongRunFailure> long_run_failure;
  bool native_cleanup_passed = false;
  bool cuda_cleanup_passed = false;
  bool nvml_cleanup_passed = false;
  ProcessCpuTimes cpu_before_runtime;
  ProcessCpuTimes cpu_after_cleanup;
  double benchmark_wall_seconds = 0.0;
};

[[nodiscard]] std::string benchmark_usage() {
  return
      "usage: mtt-runtime-benchmark "
      "--library /absolute/libmagpie_tts_rt.so "
      "--bundle /absolute/bundle "
      "--manifest-sha256 <64-lowercase-hex> "
      "--corpus /absolute/corpus.jsonl "
      "--cuda-device INDEX "
      "--cuda-runtime-library /absolute/libcudart.so.13.x "
      "--nvml-library /absolute/libnvidia-ml.so.1 "
      "--nvml-device INDEX "
      "--long-run-iterations COUNT "
      "--receipt-json /absolute/new-receipt.json "
      "[--request-timeout-ms MILLISECONDS]\n"
      "\n"
      "COUNT must be exactly 1000. The corpus must contain exactly 108 "
      "normal cases and one cancel case. The cancel case is exercised at "
      "three fixed injection points.";
}

[[nodiscard]] BenchmarkArguments parse_benchmark_arguments(
    const int argc, char** argv) {
  BenchmarkArguments arguments;
  bool have_library = false;
  bool have_bundle = false;
  bool have_manifest = false;
  bool have_corpus = false;
  bool have_cuda = false;
  bool have_cuda_runtime_library = false;
  bool have_nvml_library = false;
  bool have_nvml_device = false;
  bool have_long_run = false;
  bool have_receipt = false;
  bool have_timeout = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    auto reject_duplicate = [&](const bool present) {
      if (present) {
        throw std::invalid_argument(
            std::string(option) + " may be specified only once");
      }
    };
    if (option == "--library") {
      reject_duplicate(have_library);
      arguments.native.library =
          require_option_value(index, argc, argv, option);
      have_library = true;
    } else if (option == "--bundle") {
      reject_duplicate(have_bundle);
      arguments.native.bundle =
          require_option_value(index, argc, argv, option);
      have_bundle = true;
    } else if (option == "--manifest-sha256") {
      reject_duplicate(have_manifest);
      arguments.native.manifest_sha256_hex =
          require_option_value(index, argc, argv, option);
      arguments.native.manifest_sha256 =
          parse_sha256(arguments.native.manifest_sha256_hex);
      have_manifest = true;
    } else if (option == "--corpus") {
      reject_duplicate(have_corpus);
      arguments.corpus =
          require_option_value(index, argc, argv, option);
      have_corpus = true;
    } else if (option == "--cuda-device") {
      reject_duplicate(have_cuda);
      arguments.native.cuda_device_index = parse_integer<std::int32_t>(
          require_option_value(index, argc, argv, option), option);
      if (arguments.native.cuda_device_index < 0) {
        throw std::invalid_argument("--cuda-device must be non-negative");
      }
      have_cuda = true;
    } else if (option == "--cuda-runtime-library") {
      reject_duplicate(have_cuda_runtime_library);
      arguments.cuda_runtime_library =
          require_option_value(index, argc, argv, option);
      have_cuda_runtime_library = true;
    } else if (option == "--nvml-library") {
      reject_duplicate(have_nvml_library);
      arguments.nvml_library =
          require_option_value(index, argc, argv, option);
      have_nvml_library = true;
    } else if (option == "--nvml-device") {
      reject_duplicate(have_nvml_device);
      arguments.nvml_device_index = parse_integer<std::uint32_t>(
          require_option_value(index, argc, argv, option), option);
      have_nvml_device = true;
    } else if (option == "--long-run-iterations") {
      reject_duplicate(have_long_run);
      arguments.long_run_iterations = parse_integer<std::uint64_t>(
          require_option_value(index, argc, argv, option), option);
      if (arguments.long_run_iterations !=
          kAcceptanceLongRunIterations) {
        throw std::invalid_argument(
            "--long-run-iterations must be exactly 1000");
      }
      have_long_run = true;
    } else if (option == "--receipt-json") {
      reject_duplicate(have_receipt);
      arguments.native.receipt_json =
          require_option_value(index, argc, argv, option);
      have_receipt = true;
    } else if (option == "--request-timeout-ms") {
      reject_duplicate(have_timeout);
      const auto timeout = parse_integer<std::uint64_t>(
          require_option_value(index, argc, argv, option), option);
      if (timeout == 0U ||
          timeout >
              static_cast<std::uint64_t>(
                  std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument(
            "--request-timeout-ms must be a positive int64 value");
      }
      arguments.native.request_timeout = std::chrono::milliseconds(
          static_cast<std::chrono::milliseconds::rep>(timeout));
      have_timeout = true;
    } else if (option == "--help" || option == "-h") {
      arguments.help = true;
    } else {
      throw std::invalid_argument(
          "unknown option " + std::string(option) + "\n" +
          benchmark_usage());
    }
  }
  if (arguments.help) {
    return arguments;
  }
  if (!have_library || !have_bundle || !have_manifest || !have_corpus ||
      !have_cuda || !have_cuda_runtime_library ||
      !have_nvml_library || !have_nvml_device || !have_long_run ||
      !have_receipt) {
    throw std::invalid_argument(
        "missing required option\n" + benchmark_usage());
  }
  require_absolute_regular_file(arguments.native.library, "--library");
  require_absolute_directory(arguments.native.bundle, "--bundle");
  require_absolute_regular_file(arguments.corpus, "--corpus");
  require_absolute_regular_file(
      arguments.cuda_runtime_library, "--cuda-runtime-library");
  require_absolute_regular_file(
      arguments.nvml_library, "--nvml-library");
  require_new_absolute_receipt_path(*arguments.native.receipt_json);
  if (static_cast<std::uint32_t>(
          arguments.native.cuda_device_index) !=
      arguments.nvml_device_index) {
    throw std::invalid_argument(
        "--cuda-device and --nvml-device must identify the same index");
  }
  return arguments;
}

[[nodiscard]] std::array<std::uint8_t, MTT_SHA256_BYTES> sha256_bytes(
    const std::string_view bytes) {
  std::array<std::uint8_t, MTT_SHA256_BYTES> digest{};
  EVP_MD_CTX* const context = EVP_MD_CTX_new();
  if (context == nullptr) {
    throw std::runtime_error("EVP_MD_CTX_new failed");
  }
  unsigned int digest_size = 0U;
  const bool success =
      EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
      EVP_DigestUpdate(context, bytes.data(), bytes.size()) == 1 &&
      EVP_DigestFinal_ex(context, digest.data(), &digest_size) == 1;
  EVP_MD_CTX_free(context);
  if (!success || digest_size != digest.size()) {
    throw std::runtime_error("SHA-256 computation failed");
  }
  return digest;
}

[[nodiscard]] std::string digest_hex(
    const std::string_view bytes) {
  const auto digest = sha256_bytes(bytes);
  constexpr char kHex[] = "0123456789abcdef";
  std::string result(digest.size() * 2U, '0');
  for (std::size_t index = 0; index < digest.size(); ++index) {
    result[index * 2U] = kHex[digest[index] >> 4U];
    result[index * 2U + 1U] = kHex[digest[index] & 0x0fU];
  }
  return result;
}

[[nodiscard]] std::string file_digest_hex(
    const std::filesystem::path& path) {
  require_absolute_regular_file(path, "SHA-256 input");
  using DigestContext =
      std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
  DigestContext context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (!context) {
    throw std::runtime_error("EVP_MD_CTX_new failed");
  }
  if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("SHA-256 initialization failed");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "failed to open SHA-256 input " + path.string());
  }
  std::array<char, 1024U * 1024U> buffer{};
  while (input) {
    input.read(
        buffer.data(),
        static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0 &&
        EVP_DigestUpdate(
            context.get(), buffer.data(),
            static_cast<std::size_t>(count)) != 1) {
      throw std::runtime_error("SHA-256 update failed");
    }
  }
  if (!input.eof()) {
    throw std::runtime_error(
        "failed to read SHA-256 input " + path.string());
  }
  std::array<std::uint8_t, MTT_SHA256_BYTES> digest{};
  unsigned int digest_size = 0U;
  if (EVP_DigestFinal_ex(
          context.get(), digest.data(), &digest_size) != 1 ||
      digest_size != digest.size()) {
    throw std::runtime_error("SHA-256 finalization failed");
  }
  constexpr char kHex[] = "0123456789abcdef";
  std::string result(digest.size() * 2U, '0');
  for (std::size_t index = 0U; index < digest.size(); ++index) {
    result[index * 2U] = kHex[digest[index] >> 4U];
    result[index * 2U + 1U] =
        kHex[digest[index] & 0x0fU];
  }
  return result;
}

[[nodiscard]] std::string read_bounded_text_file(
    const std::filesystem::path& path) {
  require_absolute_regular_file(path, "benchmark corpus");
  const auto size = std::filesystem::file_size(path);
  if (size > kMaximumSmokeJsonBytes) {
    throw std::runtime_error(
        "benchmark corpus exceeds the 16 MiB input limit");
  }
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to open benchmark corpus");
  }
  std::string bytes{
      std::istreambuf_iterator<char>(stream),
      std::istreambuf_iterator<char>()};
  if (!stream.eof() && stream.fail()) {
    throw std::runtime_error("failed to read benchmark corpus");
  }
  if (bytes.find('\0') != std::string::npos) {
    throw std::runtime_error("benchmark corpus contains a NUL byte");
  }
  return bytes;
}

[[nodiscard]] nlohmann::json parse_strict_json_line(
    const std::string& line, const std::uint64_t line_number) {
  std::vector<std::unordered_set<std::string>> object_keys;
  const auto callback =
      [&](const int, const nlohmann::json::parse_event_t event,
          nlohmann::json& parsed) {
        if (event == nlohmann::json::parse_event_t::object_start) {
          object_keys.emplace_back();
        } else if (event == nlohmann::json::parse_event_t::key) {
          if (object_keys.empty()) {
            throw std::runtime_error(
                "JSON key appeared outside an object");
          }
          const auto key = parsed.get<std::string>();
          if (!object_keys.back().insert(key).second) {
            throw std::runtime_error(
                "duplicate JSON key on corpus line " +
                std::to_string(line_number) + ": " + key);
          }
        } else if (event == nlohmann::json::parse_event_t::object_end) {
          if (object_keys.empty()) {
            throw std::runtime_error(
                "JSON object stack underflow");
          }
          object_keys.pop_back();
        }
        return true;
      };
  try {
    return nlohmann::json::parse(line, callback, true, false);
  } catch (const nlohmann::json::exception& exception) {
    throw std::runtime_error(
        "invalid JSON on corpus line " +
        std::to_string(line_number) + ": " + exception.what());
  }
}

void require_exact_members(
    const nlohmann::json& record,
    const std::initializer_list<std::string_view> expected,
    const std::uint64_t line_number) {
  if (!record.is_object()) {
    throw std::runtime_error(
        "corpus line " + std::to_string(line_number) +
        " must be one JSON object");
  }
  std::set<std::string> actual;
  for (const auto& [key, value] : record.items()) {
    static_cast<void>(value);
    actual.insert(key);
  }
  std::set<std::string> required;
  for (const auto member : expected) {
    required.emplace(member);
  }
  if (actual != required) {
    throw std::runtime_error(
        "corpus line " + std::to_string(line_number) +
        " has unknown or missing members");
  }
}

[[nodiscard]] std::uint64_t require_uint(
    const nlohmann::json& record, const std::string_view member,
    const std::uint64_t line_number) {
  const auto found = record.find(member);
  if (found == record.end() ||
      (!found->is_number_unsigned() &&
       !found->is_number_integer())) {
    throw std::runtime_error(
        "corpus line " + std::to_string(line_number) + " member " +
        std::string(member) + " must be an unsigned integer");
  }
  if (found->is_number_integer() && found->get<std::int64_t>() < 0) {
    throw std::runtime_error(
        "corpus line " + std::to_string(line_number) + " member " +
        std::string(member) + " must be non-negative");
  }
  return found->get<std::uint64_t>();
}

[[nodiscard]] std::string require_string(
    const nlohmann::json& record, const std::string_view member,
    const std::uint64_t line_number) {
  const auto found = record.find(member);
  if (found == record.end() || !found->is_string()) {
    throw std::runtime_error(
        "corpus line " + std::to_string(line_number) + " member " +
        std::string(member) + " must be a string");
  }
  return found->get<std::string>();
}

[[nodiscard]] bool valid_identifier(const std::string_view value) {
  if (value.empty() || value.size() > 64U) {
    return false;
  }
  const auto allowed = [](const char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') ||
           character == '_' || character == '-';
  };
  const char first = value.front();
  const bool first_allowed =
      (first >= 'a' && first <= 'z') ||
      (first >= 'A' && first <= 'Z') ||
      (first >= '0' && first <= '9');
  return first_allowed &&
         std::all_of(value.begin(), value.end(), allowed);
}

[[nodiscard]] bool contains_japanese_codepoint(
    const std::string_view text) {
  std::size_t index = 0U;
  bool contains_japanese = false;
  while (index < text.size()) {
    const auto first = static_cast<std::uint8_t>(text[index]);
    std::uint32_t codepoint = 0U;
    std::size_t width = 0U;
    if (first < 0x80U) {
      codepoint = first;
      width = 1U;
    } else if ((first & 0xe0U) == 0xc0U) {
      codepoint = first & 0x1fU;
      width = 2U;
    } else if ((first & 0xf0U) == 0xe0U) {
      codepoint = first & 0x0fU;
      width = 3U;
    } else if ((first & 0xf8U) == 0xf0U) {
      codepoint = first & 0x07U;
      width = 4U;
    } else {
      throw std::runtime_error("source_text contains invalid UTF-8");
    }
    if (index + width > text.size()) {
      throw std::runtime_error("source_text contains truncated UTF-8");
    }
    for (std::size_t offset = 1U; offset < width; ++offset) {
      const auto continuation =
          static_cast<std::uint8_t>(text[index + offset]);
      if ((continuation & 0xc0U) != 0x80U) {
        throw std::runtime_error("source_text contains invalid UTF-8");
      }
      codepoint =
          (codepoint << 6U) | (continuation & 0x3fU);
    }
    if ((width == 2U && codepoint < 0x80U) ||
        (width == 3U && codepoint < 0x800U) ||
        (width == 4U && codepoint < 0x10000U) ||
        (codepoint >= 0xd800U && codepoint <= 0xdfffU) ||
        codepoint > 0x10ffffU) {
      throw std::runtime_error("source_text contains noncanonical UTF-8");
    }
    if ((codepoint >= 0x3040U && codepoint <= 0x30ffU) ||
        (codepoint >= 0x3400U && codepoint <= 0x9fffU)) {
      contains_japanese = true;
    }
    index += width;
  }
  return contains_japanese;
}

[[nodiscard]] CorpusCase parse_corpus_case(
    const nlohmann::json& record, const std::uint64_t line_number,
    const std::string_view required_record_type) {
  require_exact_members(
      record,
      {"record_type", "case_id", "source_text",
       "source_text_sha256", "prepared_token_ids", "random_seed"},
      line_number);
  if (require_string(record, "record_type", line_number) !=
      required_record_type) {
    throw std::runtime_error(
        "unexpected record_type on corpus line " +
        std::to_string(line_number));
  }
  CorpusCase result;
  result.case_id = require_string(record, "case_id", line_number);
  if (!valid_identifier(result.case_id)) {
    throw std::runtime_error(
        "invalid case_id on corpus line " +
        std::to_string(line_number));
  }
  result.source_text =
      require_string(record, "source_text", line_number);
  if (result.source_text.empty() || result.source_text.size() > 4'096U ||
      !contains_japanese_codepoint(result.source_text)) {
    throw std::runtime_error(
        "source_text must contain Japanese and be 1-4096 UTF-8 bytes");
  }
  result.source_text_sha256 =
      require_string(record, "source_text_sha256", line_number);
  if (result.source_text_sha256 != digest_hex(result.source_text)) {
    throw std::runtime_error(
        "source_text_sha256 mismatch on corpus line " +
        std::to_string(line_number));
  }
  const auto tokens = record.find("prepared_token_ids");
  if (tokens == record.end() || !tokens->is_array() || tokens->empty() ||
      tokens->size() > MTT_MAX_TEXT_TOKENS) {
    throw std::runtime_error(
        "prepared_token_ids must be a non-empty bounded array");
  }
  result.token_ids.reserve(tokens->size());
  for (const auto& token : *tokens) {
    if ((!token.is_number_unsigned() && !token.is_number_integer()) ||
        (token.is_number_integer() && token.get<std::int64_t>() < 0)) {
      throw std::runtime_error(
          "prepared_token_ids must contain non-negative integers");
    }
    const auto value = token.get<std::uint64_t>();
    if (value >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::int32_t>::max())) {
      throw std::runtime_error(
          "prepared token identifier exceeds INT32");
    }
    result.token_ids.push_back(static_cast<std::int64_t>(value));
  }
  result.seed = require_uint(record, "random_seed", line_number);
  if (result.seed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("random_seed must be a uint32");
  }
  return result;
}

[[nodiscard]] BenchmarkCorpus load_benchmark_corpus(
    const std::filesystem::path& path) {
  const auto bytes = read_bounded_text_file(path);
  std::istringstream lines(bytes);
  std::string line;
  std::vector<nlohmann::json> records;
  std::uint64_t line_number = 0U;
  while (std::getline(lines, line)) {
    ++line_number;
    if (line.empty() || line.back() == '\r') {
      throw std::runtime_error(
          "blank or CRLF corpus line " + std::to_string(line_number));
    }
    records.push_back(parse_strict_json_line(line, line_number));
  }
  if (records.size() != kAcceptanceNormalCases + 2U) {
    throw std::runtime_error(
        "benchmark corpus requires a header, exactly 108 normal cases, "
        "and one cancel case");
  }
  const auto& header = records.front();
  require_exact_members(
      header,
      {"record_type", "schema_version", "corpus_id",
       "tokenizer_identity_sha256", "normal_case_count"},
      1U);
  if (require_string(header, "record_type", 1U) != "header" ||
      require_string(header, "schema_version", 1U) !=
          "magpie-tts-rt.benchmark-corpus.v1") {
    throw std::runtime_error("unsupported benchmark corpus header");
  }
  BenchmarkCorpus corpus;
  corpus.corpus_id = require_string(header, "corpus_id", 1U);
  if (!valid_identifier(corpus.corpus_id)) {
    throw std::runtime_error("invalid benchmark corpus_id");
  }
  corpus.tokenizer_identity_sha256 =
      require_string(header, "tokenizer_identity_sha256", 1U);
  static_cast<void>(parse_sha256(corpus.tokenizer_identity_sha256));
  const auto declared_count =
      require_uint(header, "normal_case_count", 1U);
  if (declared_count != kAcceptanceNormalCases ||
      records.size() != declared_count + 2U) {
    throw std::runtime_error(
        "normal_case_count does not match the JSONL record count");
  }
  std::unordered_set<std::string> case_ids;
  corpus.normal_cases.reserve(
      static_cast<std::size_t>(declared_count));
  for (std::uint64_t index = 0U; index < declared_count; ++index) {
    auto parsed = parse_corpus_case(
        records[static_cast<std::size_t>(index + 1U)], index + 2U,
        "case");
    if (!case_ids.insert(parsed.case_id).second) {
      throw std::runtime_error("duplicate benchmark case_id");
    }
    corpus.normal_cases.push_back(std::move(parsed));
  }
  corpus.cancel_case = parse_corpus_case(
      records.back(), declared_count + 2U, "cancel_case");
  if (!case_ids.insert(corpus.cancel_case.case_id).second) {
    throw std::runtime_error("duplicate cancel case_id");
  }
  corpus.file_sha256 = digest_hex(bytes);
  return corpus;
}

using NvmlReturn = int;
using NvmlDevice = struct NvmlDeviceOpaque*;
constexpr NvmlReturn kNvmlSuccess = 0;
constexpr NvmlReturn kNvmlNotSupported = 3;
constexpr std::size_t kNvmlDriverBufferBytes = 256U;
constexpr std::size_t kNvmlDeviceNameBufferBytes = 256U;
constexpr std::size_t kNvmlUuidBufferBytes = 96U;

struct NvmlMemoryV2 {
  unsigned int version;
  unsigned long long total;
  unsigned long long reserved;
  unsigned long long free;
  unsigned long long used;
};

constexpr unsigned int kNvmlMemoryV2Version =
    static_cast<unsigned int>(
        sizeof(NvmlMemoryV2) | (2U << 24U));

[[nodiscard]] std::uint64_t process_rss_bytes() {
  std::ifstream status("/proc/self/status");
  if (!status) {
    throw std::runtime_error("failed to open /proc/self/status");
  }
  std::string line;
  while (std::getline(status, line)) {
    if (!line.starts_with("VmRSS:")) {
      continue;
    }
    std::istringstream fields(line.substr(6U));
    std::uint64_t kibibytes = 0U;
    std::string unit;
    std::string trailing;
    if (!(fields >> kibibytes >> unit) || unit != "kB" ||
        (fields >> trailing) ||
        kibibytes >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()) /
                1'024U) {
      throw std::runtime_error("invalid VmRSS in /proc/self/status");
    }
    return kibibytes * 1'024U;
  }
  throw std::runtime_error("VmRSS is absent from /proc/self/status");
}

[[nodiscard]] ProcessCpuTimes process_cpu_times() {
  rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0) {
    throw std::system_error(
        errno, std::generic_category(), "getrusage(RUSAGE_SELF)");
  }
  const auto seconds = [](const timeval& value) {
    if (value.tv_sec < 0 || value.tv_usec < 0 ||
        value.tv_usec >= 1'000'000) {
      throw std::runtime_error(
          "getrusage returned an invalid process CPU time");
    }
    return static_cast<double>(value.tv_sec) +
           static_cast<double>(value.tv_usec) / 1'000'000.0;
  };
  return ProcessCpuTimes{
      .user_seconds = seconds(usage.ru_utime),
      .system_seconds = seconds(usage.ru_stime),
  };
}

[[nodiscard]] MemoryPoint process_only_memory_point() {
  return MemoryPoint{
      .process_rss_bytes = process_rss_bytes(),
      .cuda_total_bytes = std::nullopt,
      .cuda_free_bytes = std::nullopt,
      .cuda_used_bytes = std::nullopt,
  };
}

class NvmlSampler {
 public:
  NvmlSampler(
      const std::filesystem::path& library,
      const std::uint32_t device_index) {
    library_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr) {
      const char* const message = dlerror();
      throw std::runtime_error(
          "NVML dlopen failed: " +
          std::string(message == nullptr ? "unknown" : message));
    }
    try {
      init_ = symbol<Init>("nvmlInit_v2");
      shutdown_ = symbol<Shutdown>("nvmlShutdown");
      get_device_ =
          symbol<GetDevice>("nvmlDeviceGetHandleByIndex_v2");
      get_memory_ =
          symbol<GetMemory>("nvmlDeviceGetMemoryInfo_v2");
      get_driver_ =
          symbol<GetDriver>("nvmlSystemGetDriverVersion");
      get_name_ = symbol<GetName>("nvmlDeviceGetName");
      get_uuid_ = symbol<GetUuid>("nvmlDeviceGetUUID");
      get_compute_capability_ = symbol<GetComputeCapability>(
          "nvmlDeviceGetCudaComputeCapability");
      require_nvml_ok("nvmlInit_v2", init_());
      initialized_ = true;
      require_nvml_ok(
          "nvmlDeviceGetHandleByIndex_v2",
          get_device_(device_index, &device_));
      if (device_ == nullptr) {
        throw std::runtime_error(
            "NVML returned a null device handle");
      }
      std::array<char, kNvmlDriverBufferBytes> driver{};
      require_nvml_ok(
          "nvmlSystemGetDriverVersion",
          get_driver_(
              driver.data(),
              static_cast<unsigned int>(driver.size())));
      const auto parsed =
          fixed_message(driver.data(), driver.size());
      if (!parsed.has_value() || parsed->empty()) {
        throw std::runtime_error(
            "NVML returned an invalid driver version");
      }
      driver_version_ = *parsed;

      std::array<char, kNvmlDeviceNameBufferBytes> name{};
      require_nvml_ok(
          "nvmlDeviceGetName",
          get_name_(
              device_, name.data(),
              static_cast<unsigned int>(name.size())));
      const auto parsed_name =
          fixed_message(name.data(), name.size());
      if (!parsed_name.has_value() || parsed_name->empty()) {
        throw std::runtime_error(
            "NVML returned an invalid device name");
      }
      device_name_ = *parsed_name;

      std::array<char, kNvmlUuidBufferBytes> uuid{};
      require_nvml_ok(
          "nvmlDeviceGetUUID",
          get_uuid_(
              device_, uuid.data(),
              static_cast<unsigned int>(uuid.size())));
      const auto parsed_uuid =
          fixed_message(uuid.data(), uuid.size());
      if (!parsed_uuid.has_value() || parsed_uuid->empty()) {
        throw std::runtime_error(
            "NVML returned an invalid device UUID");
      }
      device_uuid_ = *parsed_uuid;

      require_nvml_ok(
          "nvmlDeviceGetCudaComputeCapability",
          get_compute_capability_(
              device_, &compute_capability_major_,
              &compute_capability_minor_));
      if (compute_capability_major_ <= 0 ||
          compute_capability_minor_ < 0) {
        throw std::runtime_error(
            "NVML returned an invalid CUDA compute capability");
      }

      NvmlMemoryV2 memory{};
      memory.version = kNvmlMemoryV2Version;
      const auto memory_status = get_memory_(device_, &memory);
      if (memory_status == kNvmlNotSupported) {
        memory_api_status_ = "unsupported";
      } else {
        require_nvml_ok(
            "nvmlDeviceGetMemoryInfo_v2", memory_status);
        if (memory.version != kNvmlMemoryV2Version ||
            memory.total == 0U ||
            memory.reserved > memory.total ||
            memory.used > memory.total - memory.reserved ||
            memory.free !=
                memory.total - memory.reserved - memory.used) {
          throw std::runtime_error(
              "NVML memory values violate the v2 contract");
        }
        memory_api_status_ = "supported_not_used";
      }
    } catch (...) {
      if (initialized_) {
        static_cast<void>(shutdown_());
        initialized_ = false;
      }
      static_cast<void>(dlclose(library_));
      library_ = nullptr;
      throw;
    }
  }

  NvmlSampler(const NvmlSampler&) = delete;
  NvmlSampler& operator=(const NvmlSampler&) = delete;

  ~NvmlSampler() {
    if (initialized_) {
      static_cast<void>(shutdown_());
    }
    if (library_ != nullptr) {
      static_cast<void>(dlclose(library_));
    }
  }

  [[nodiscard]] const std::string& driver_version() const {
    return driver_version_;
  }

  [[nodiscard]] const std::string& device_name() const {
    return device_name_;
  }

  [[nodiscard]] const std::string& device_uuid() const {
    return device_uuid_;
  }

  [[nodiscard]] int compute_capability_major() const {
    return compute_capability_major_;
  }

  [[nodiscard]] int compute_capability_minor() const {
    return compute_capability_minor_;
  }

  [[nodiscard]] const std::string& memory_api_status() const {
    return memory_api_status_;
  }

  void close() {
    if (initialized_) {
      require_nvml_ok("nvmlShutdown", shutdown_());
      initialized_ = false;
    }
    if (library_ != nullptr) {
      if (dlclose(library_) != 0) {
        const char* const message = dlerror();
        throw std::runtime_error(
            "NVML dlclose failed: " +
            std::string(message == nullptr ? "unknown" : message));
      }
      library_ = nullptr;
    }
  }

 private:
  using Init = NvmlReturn (*)();
  using Shutdown = NvmlReturn (*)();
  using GetDevice = NvmlReturn (*)(unsigned int, NvmlDevice*);
  using GetMemory = NvmlReturn (*)(NvmlDevice, NvmlMemoryV2*);
  using GetDriver = NvmlReturn (*)(char*, unsigned int);
  using GetName =
      NvmlReturn (*)(NvmlDevice, char*, unsigned int);
  using GetUuid =
      NvmlReturn (*)(NvmlDevice, char*, unsigned int);
  using GetComputeCapability =
      NvmlReturn (*)(NvmlDevice, int*, int*);

  template <typename Function>
  [[nodiscard]] Function symbol(const char* const name) {
    dlerror();
    void* const raw = dlsym(library_, name);
    if (const char* const message = dlerror(); message != nullptr) {
      throw std::runtime_error(
          std::string("NVML dlsym failed for ") + name + ": " +
          message);
    }
    if (raw == nullptr) {
      throw std::runtime_error(
          std::string("NVML returned null symbol ") + name);
    }
    Function function = nullptr;
    static_assert(sizeof(function) == sizeof(raw));
    std::memcpy(&function, &raw, sizeof(function));
    return function;
  }

  static void require_nvml_ok(
      const std::string_view operation, const NvmlReturn status) {
    if (status != kNvmlSuccess) {
      throw std::runtime_error(
          std::string(operation) +
          " failed with NVML status " + std::to_string(status));
    }
  }

  void* library_ = nullptr;
  Init init_ = nullptr;
  Shutdown shutdown_ = nullptr;
  GetDevice get_device_ = nullptr;
  GetMemory get_memory_ = nullptr;
  GetDriver get_driver_ = nullptr;
  GetName get_name_ = nullptr;
  GetUuid get_uuid_ = nullptr;
  GetComputeCapability get_compute_capability_ = nullptr;
  NvmlDevice device_ = nullptr;
  std::string driver_version_;
  std::string device_name_;
  std::string device_uuid_;
  std::string memory_api_status_;
  int compute_capability_major_ = 0;
  int compute_capability_minor_ = 0;
  bool initialized_ = false;
};

using CudaReturn = int;
constexpr CudaReturn kCudaSuccess = 0;

class CudaMemorySampler {
 public:
  explicit CudaMemorySampler(
      const std::filesystem::path& library) {
    library_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr) {
      const char* const message = dlerror();
      throw std::runtime_error(
          "CUDART dlopen failed: " +
          std::string(message == nullptr ? "unknown" : message));
    }
    try {
      set_device_ = symbol<SetDevice>("cudaSetDevice");
      get_memory_ = symbol<GetMemory>("cudaMemGetInfo");
      get_driver_version_ =
          symbol<GetVersion>("cudaDriverGetVersion");
      get_runtime_version_ =
          symbol<GetVersion>("cudaRuntimeGetVersion");
      get_error_string_ =
          symbol<GetErrorString>("cudaGetErrorString");
    } catch (...) {
      static_cast<void>(dlclose(library_));
      library_ = nullptr;
      throw;
    }
  }

  CudaMemorySampler(const CudaMemorySampler&) = delete;
  CudaMemorySampler& operator=(const CudaMemorySampler&) = delete;

  ~CudaMemorySampler() {
    if (library_ != nullptr) {
      static_cast<void>(dlclose(library_));
    }
  }

  void activate(const std::int32_t device_index) {
    require_cuda_ok("cudaSetDevice", set_device_(device_index));
    device_index_ = device_index;
    require_cuda_ok(
        "cudaDriverGetVersion",
        get_driver_version_(&driver_version_));
    require_cuda_ok(
        "cudaRuntimeGetVersion",
        get_runtime_version_(&runtime_version_));
    if (driver_version_ <= 0 || runtime_version_ <= 0) {
      throw std::runtime_error(
          "CUDART returned an invalid version");
    }
    activated_ = true;
  }

  [[nodiscard]] int driver_version() const {
    require_activated();
    return driver_version_;
  }

  [[nodiscard]] int runtime_version() const {
    require_activated();
    return runtime_version_;
  }

  [[nodiscard]] MemoryPoint sample() const {
    require_activated();
    require_cuda_ok(
        "cudaSetDevice", set_device_(device_index_));
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    require_cuda_ok(
        "cudaMemGetInfo",
        get_memory_(&free_bytes, &total_bytes));
    if (total_bytes == 0U || free_bytes > total_bytes ||
        total_bytes >
            static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
      throw std::runtime_error(
          "cudaMemGetInfo returned invalid device-memory values");
    }
    const auto total =
        static_cast<std::uint64_t>(total_bytes);
    const auto free = static_cast<std::uint64_t>(free_bytes);
    return MemoryPoint{
        .process_rss_bytes = process_rss_bytes(),
        .cuda_total_bytes = total,
        .cuda_free_bytes = free,
        .cuda_used_bytes = total - free,
    };
  }

  void close() {
    if (library_ != nullptr) {
      if (dlclose(library_) != 0) {
        const char* const message = dlerror();
        throw std::runtime_error(
            "CUDART dlclose failed: " +
            std::string(message == nullptr ? "unknown" : message));
      }
      library_ = nullptr;
    }
  }

 private:
  using SetDevice = CudaReturn (*)(int);
  using GetMemory = CudaReturn (*)(std::size_t*, std::size_t*);
  using GetVersion = CudaReturn (*)(int*);
  using GetErrorString = const char* (*)(CudaReturn);

  template <typename Function>
  [[nodiscard]] Function symbol(const char* const name) {
    dlerror();
    void* const raw = dlsym(library_, name);
    if (const char* const message = dlerror(); message != nullptr) {
      throw std::runtime_error(
          std::string("CUDART dlsym failed for ") + name + ": " +
          message);
    }
    if (raw == nullptr) {
      throw std::runtime_error(
          std::string("CUDART returned null symbol ") + name);
    }
    Function function = nullptr;
    static_assert(sizeof(function) == sizeof(raw));
    std::memcpy(&function, &raw, sizeof(function));
    return function;
  }

  void require_cuda_ok(
      const std::string_view operation,
      const CudaReturn status) const {
    if (status == kCudaSuccess) {
      return;
    }
    const char* const raw_message =
        get_error_string_ == nullptr
            ? nullptr
            : get_error_string_(status);
    throw std::runtime_error(
        std::string(operation) + " failed with CUDA status " +
        std::to_string(status) + ": " +
        (raw_message == nullptr ? "unknown" : raw_message));
  }

  void require_activated() const {
    if (!activated_) {
      throw std::logic_error(
          "CUDART sampler used before post-startup activation");
    }
  }

  void* library_ = nullptr;
  SetDevice set_device_ = nullptr;
  GetMemory get_memory_ = nullptr;
  GetVersion get_driver_version_ = nullptr;
  GetVersion get_runtime_version_ = nullptr;
  GetErrorString get_error_string_ = nullptr;
  std::int32_t device_index_ = -1;
  int driver_version_ = 0;
  int runtime_version_ = 0;
  bool activated_ = false;
};

void validate_case_against_model(
    const CorpusCase& corpus_case,
    const mtt_model_info_v1_t& model_info) {
  if (corpus_case.token_ids.empty() ||
      corpus_case.token_ids.size() > model_info.maximum_text_tokens) {
    throw std::runtime_error(
        "case " + corpus_case.case_id +
        " exceeds the authenticated text-token limit");
  }
  if (corpus_case.token_ids.back() !=
      static_cast<std::int64_t>(model_info.eos_token_id)) {
    throw std::runtime_error(
        "case " + corpus_case.case_id +
        " does not end in authenticated EOS");
  }
  for (std::size_t index = 0;
       index + 1 < corpus_case.token_ids.size();
       ++index) {
    const std::int64_t token = corpus_case.token_ids[index];
    if (token < 0 ||
        token >= static_cast<std::int64_t>(
                     model_info.tokenizer_vocabulary_size)) {
      throw std::runtime_error(
          "case " + corpus_case.case_id +
          " contains a non-final token outside normal tokenizer rows");
    }
  }
}

void start_request(
    NativeObjects& native, const CorpusCase& corpus_case) {
  if (native.request != nullptr) {
    throw std::logic_error(
        "attempted to start a request while another is owned");
  }
  mtt_request_desc_v1_t descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = MTT_ABI_VERSION_1;
  descriptor.text_token_ids = corpus_case.token_ids.data();
  descriptor.text_token_count = corpus_case.token_ids.size();
  descriptor.random_seed = corpus_case.seed;
  auto error = fresh_error();
  require_ok(
      "request_start",
      native.api.request_start(
          native.session, &descriptor, &native.request, &error),
      error);
  if (native.request == nullptr) {
    throw std::runtime_error(
        "request_start succeeded without returning a request handle");
  }
}

void destroy_terminal_request(NativeObjects& native) {
  if (native.request == nullptr) {
    throw std::logic_error("no request is owned");
  }
  auto error = fresh_error();
  require_ok(
      "request_destroy",
      native.api.request_destroy(native.request, &error), error);
  native.request = nullptr;
}

[[nodiscard]] bool acquire_one_chunk(
    NativeObjects& native, const mtt_model_info_v1_t& model_info,
    const std::uint64_t request_token_count,
    const Clock::time_point request_started_at,
    StreamMetrics& contract_metrics,
    std::vector<ChunkTiming>& chunks,
    std::optional<Clock::time_point>& previous_arrival,
    std::optional<Clock::time_point>& first_arrival) {
  mtt_audio_lease_v1_t lease{};
  lease.struct_size = sizeof(lease);
  lease.abi_version = MTT_ABI_VERSION_1;
  auto error = fresh_error();
  const auto status = validate_call_status(
      "audio_acquire",
      native.api.audio_acquire(native.request, &lease, &error), error);
  if (status == MTT_STATUS_WOULD_BLOCK) {
    return false;
  }
  require_ok("audio_acquire", status, error);
  const auto arrived_at = Clock::now();
  double positive_playback_lateness_ms = 0.0;
  if (first_arrival.has_value()) {
    const auto observed_since_first =
        std::chrono::duration<double>(
            arrived_at - *first_arrival)
            .count();
    const auto ideal_since_first =
        static_cast<double>(lease.first_sample_index) /
        static_cast<double>(model_info.sample_rate_hz);
    positive_playback_lateness_ms = std::max(
        0.0,
        (observed_since_first - ideal_since_first) * 1'000.0);
  }
  ChunkTiming timing{
      .sequence = lease.sequence,
      .first_sample_index = lease.first_sample_index,
      .sample_count = lease.sample_count,
      .codec_frames = lease.sample_count / kCodecFrameSamples,
      .arrival_ms = milliseconds(arrived_at - request_started_at),
      .interval_ms =
          previous_arrival.has_value()
              ? std::optional<double>(
                    milliseconds(arrived_at - *previous_arrival))
              : std::nullopt,
      .positive_playback_lateness_ms =
          positive_playback_lateness_ms,
      .first = (lease.flags & MTT_AUDIO_FLAG_FIRST) != 0U,
      .final = (lease.flags & MTT_AUDIO_FLAG_FINAL) != 0U,
  };
  validate_and_release_lease(
      native, lease, model_info, request_token_count,
      contract_metrics);
  if (!first_arrival.has_value()) {
    first_arrival = arrived_at;
  }
  previous_arrival = arrived_at;
  chunks.push_back(timing);
  return true;
}

void drain_available_chunks(
    NativeObjects& native, const mtt_model_info_v1_t& model_info,
    const std::uint64_t request_token_count,
    const Clock::time_point request_started_at,
    StreamMetrics& contract_metrics,
    std::vector<ChunkTiming>& chunks,
    std::optional<Clock::time_point>& previous_arrival,
    std::optional<Clock::time_point>& first_arrival) {
  while (acquire_one_chunk(
      native, model_info, request_token_count, request_started_at,
      contract_metrics, chunks, previous_arrival, first_arrival)) {
  }
}

[[nodiscard]] std::string terminal_snapshot_message(
    const mtt_request_snapshot_v1_t& snapshot) {
  const auto message = fixed_message(
      snapshot.terminal_error_message, MTT_ERROR_MESSAGE_CAPACITY);
  return message.value_or("<not-NUL-terminated>");
}

[[nodiscard]] CaseResult run_completed_case(
    const BenchmarkArguments& arguments,
    const CorpusCase& corpus_case, NativeObjects& native,
    const mtt_model_info_v1_t& model_info) {
  validate_case_against_model(corpus_case, model_info);
  const auto started_at = Clock::now();
  start_request(native, corpus_case);
  const auto request_start_returned_at = Clock::now();
  StreamMetrics contract_metrics;
  std::vector<ChunkTiming> chunks;
  std::optional<Clock::time_point> previous_arrival;
  std::optional<Clock::time_point> first_arrival;
  std::optional<Clock::time_point> terminal_at;
  std::uint64_t revision = 0U;
  const auto deadline =
      started_at + arguments.native.request_timeout;
  for (;;) {
    drain_available_chunks(
        native, model_info, corpus_case.token_ids.size(), started_at,
        contract_metrics, chunks, previous_arrival, first_arrival);
    if (Clock::now() >= deadline) {
      throw std::runtime_error(
          "case " + corpus_case.case_id +
          " exceeded its request deadline");
    }
    mtt_request_snapshot_v1_t snapshot{};
    snapshot.struct_size = sizeof(snapshot);
    snapshot.abi_version = MTT_ABI_VERSION_1;
    auto error = fresh_error();
    const auto status = validate_call_status(
        "request_wait",
        native.api.request_wait(
            native.request, revision, 100'000'000U, &snapshot, &error),
        error);
    if (status == MTT_STATUS_TIMEOUT) {
      continue;
    }
    require_ok("request_wait", status, error);
    validate_snapshot(
        "request_wait", snapshot, corpus_case.token_ids.size(),
        model_info, revision, contract_metrics);
    revision = snapshot.revision;
    if (snapshot.state == MTT_REQUEST_STATE_RUNNING) {
      continue;
    }
    terminal_at = Clock::now();
    if (snapshot.state != MTT_REQUEST_STATE_COMPLETED) {
      throw std::runtime_error(
          "case " + corpus_case.case_id +
          " terminated unsuccessfully: status=" +
          std::to_string(snapshot.terminal_status) + " message=" +
          terminal_snapshot_message(snapshot));
    }
    break;
  }
  drain_available_chunks(
      native, model_info, corpus_case.token_ids.size(), started_at,
      contract_metrics, chunks, previous_arrival, first_arrival);
  mtt_request_snapshot_v1_t final_snapshot{};
  final_snapshot.struct_size = sizeof(final_snapshot);
  final_snapshot.abi_version = MTT_ABI_VERSION_1;
  auto error = fresh_error();
  require_ok(
      "request_poll",
      native.api.request_poll(native.request, &final_snapshot, &error),
      error);
  validate_snapshot(
      "request_poll", final_snapshot, corpus_case.token_ids.size(),
      model_info, std::nullopt, contract_metrics);
  if (final_snapshot.state != MTT_REQUEST_STATE_COMPLETED ||
      final_snapshot.available_audio_leases != 0U ||
      !contract_metrics.first_seen || !contract_metrics.final_seen ||
      contract_metrics.sample_count == 0U ||
      final_snapshot.published_samples !=
          contract_metrics.sample_count ||
      final_snapshot.generated_codec_frames !=
          contract_metrics.sample_count / kCodecFrameSamples ||
      final_snapshot.committed_text_tokens !=
          contract_metrics.last_alignment_token) {
    throw std::runtime_error(
        "case " + corpus_case.case_id +
        " final snapshot does not match its drained stream");
  }
  const auto completed_at = Clock::now();
  if (!first_arrival.has_value() || !terminal_at.has_value()) {
    throw std::runtime_error(
        "case " + corpus_case.case_id +
        " lacks first-audio or terminal timing");
  }
  destroy_terminal_request(native);
  const auto audio_seconds =
      static_cast<double>(contract_metrics.sample_count) /
      static_cast<double>(model_info.sample_rate_hz);
  const auto generation_seconds =
      std::chrono::duration<double>(*terminal_at - started_at).count();
  const auto total_seconds =
      std::chrono::duration<double>(completed_at - started_at).count();
  const auto maximum_playback_lateness =
      std::max_element(
          chunks.begin(), chunks.end(),
          [](const ChunkTiming& left, const ChunkTiming& right) {
            return left.positive_playback_lateness_ms <
                   right.positive_playback_lateness_ms;
          })
          ->positive_playback_lateness_ms;
  return CaseResult{
      .case_id = corpus_case.case_id,
      .prepared_token_count = corpus_case.token_ids.size(),
      .random_seed = corpus_case.seed,
      .request_start_call_ms =
          milliseconds(request_start_returned_at - started_at),
      .request_start_to_first_audio_ms =
          milliseconds(*first_arrival - request_start_returned_at),
      .ttfa_ms = milliseconds(*first_arrival - started_at),
      .generation_ms = generation_seconds * 1'000.0,
      .total_ms = total_seconds * 1'000.0,
      .audio_samples = contract_metrics.sample_count,
      .audio_seconds = audio_seconds,
      .generation_rtf = generation_seconds / audio_seconds,
      .total_rtf = total_seconds / audio_seconds,
      .maximum_positive_playback_lateness_ms =
          maximum_playback_lateness,
      .chunks = std::move(chunks),
};
}

enum class CancelInjectionPoint : std::uint64_t {
  kImmediatelyAfterRequestStart = 0U,
  kAfterFirstAudioChunk = 1U,
  kAfterSecondAudioChunk = 2U,
};

[[nodiscard]] std::string_view cancel_injection_point_name(
    const CancelInjectionPoint point) {
  switch (point) {
    case CancelInjectionPoint::kImmediatelyAfterRequestStart:
      return "immediately_after_request_start";
    case CancelInjectionPoint::kAfterFirstAudioChunk:
      return "after_first_audio_chunk";
    case CancelInjectionPoint::kAfterSecondAudioChunk:
      return "after_second_audio_chunk";
  }
  throw std::logic_error("unknown cancel injection point");
}

[[nodiscard]] CancelResult run_cancel_probe(
    const BenchmarkArguments& arguments,
    const CorpusCase& corpus_case, NativeObjects& native,
    const mtt_model_info_v1_t& model_info,
    const CancelInjectionPoint injection_point) {
  validate_case_against_model(corpus_case, model_info);
  const auto started_at = Clock::now();
  start_request(native, corpus_case);
  StreamMetrics contract_metrics;
  std::vector<ChunkTiming> chunks;
  std::optional<Clock::time_point> previous_arrival;
  std::optional<Clock::time_point> first_arrival;
  std::uint64_t revision = 0U;
  const auto deadline =
      started_at + arguments.native.request_timeout;
  const auto required_chunks =
      static_cast<std::uint64_t>(injection_point);
  while (chunks.size() < required_chunks) {
    if (acquire_one_chunk(
            native, model_info, corpus_case.token_ids.size(),
            started_at, contract_metrics, chunks, previous_arrival,
            first_arrival)) {
      if (contract_metrics.final_seen &&
          chunks.size() < required_chunks) {
        throw std::runtime_error(
            "cancel probe completed before its injection point");
      }
      continue;
    }
    if (Clock::now() >= deadline) {
      throw std::runtime_error(
          "cancel probe did not reach its injection point");
    }
    mtt_request_snapshot_v1_t snapshot{};
    snapshot.struct_size = sizeof(snapshot);
    snapshot.abi_version = MTT_ABI_VERSION_1;
    auto error = fresh_error();
    const auto status = validate_call_status(
        "request_wait",
        native.api.request_wait(
            native.request, revision, 100'000'000U, &snapshot, &error),
        error);
    if (status == MTT_STATUS_TIMEOUT) {
      continue;
    }
    require_ok("request_wait", status, error);
    validate_snapshot(
        "request_wait", snapshot, corpus_case.token_ids.size(),
        model_info, revision, contract_metrics);
    revision = snapshot.revision;
    if (snapshot.state != MTT_REQUEST_STATE_RUNNING) {
      throw std::runtime_error(
          "cancel probe terminated before cancellation");
    }
  }
  if (required_chunks > 0U && contract_metrics.final_seen) {
    throw std::runtime_error(
        "cancel probe completed at its injection point");
  }
  const auto audio_chunks_before_cancel = chunks.size();
  const auto audio_samples_before_cancel =
      contract_metrics.sample_count;
  const auto cancel_started_at = Clock::now();
  auto error = fresh_error();
  require_ok(
      "request_cancel",
      native.api.request_cancel(native.request, &error), error);
  const auto cancel_returned_at = Clock::now();
  std::uint64_t post_cancel_leases = 0U;
  std::optional<Clock::time_point> cancelled_at;
  for (;;) {
    while (acquire_one_chunk(
        native, model_info, corpus_case.token_ids.size(), started_at,
        contract_metrics, chunks, previous_arrival, first_arrival)) {
      ++post_cancel_leases;
    }
    if (Clock::now() >= deadline) {
      throw std::runtime_error(
          "cancel probe did not reach CANCELLED before its deadline");
    }
    mtt_request_snapshot_v1_t snapshot{};
    snapshot.struct_size = sizeof(snapshot);
    snapshot.abi_version = MTT_ABI_VERSION_1;
    error = fresh_error();
    const auto status = validate_call_status(
        "request_wait",
        native.api.request_wait(
            native.request, revision, 100'000'000U, &snapshot, &error),
        error);
    if (status == MTT_STATUS_TIMEOUT) {
      continue;
    }
    require_ok("request_wait", status, error);
    validate_snapshot(
        "request_wait", snapshot, corpus_case.token_ids.size(),
        model_info, revision, contract_metrics);
    revision = snapshot.revision;
    if (snapshot.state == MTT_REQUEST_STATE_RUNNING) {
      continue;
    }
    cancelled_at = Clock::now();
    if (snapshot.state != MTT_REQUEST_STATE_CANCELLED) {
      throw std::runtime_error(
          "cancel probe reached a non-CANCELLED terminal state");
    }
    break;
  }
  while (acquire_one_chunk(
      native, model_info, corpus_case.token_ids.size(), started_at,
      contract_metrics, chunks, previous_arrival, first_arrival)) {
    ++post_cancel_leases;
  }
  mtt_request_snapshot_v1_t final_snapshot{};
  final_snapshot.struct_size = sizeof(final_snapshot);
  final_snapshot.abi_version = MTT_ABI_VERSION_1;
  error = fresh_error();
  require_ok(
      "request_poll",
      native.api.request_poll(native.request, &final_snapshot, &error),
      error);
  validate_snapshot(
      "request_poll", final_snapshot, corpus_case.token_ids.size(),
      model_info, std::nullopt, contract_metrics);
  if (final_snapshot.state != MTT_REQUEST_STATE_CANCELLED ||
      final_snapshot.available_audio_leases != 0U ||
      final_snapshot.published_samples <
          contract_metrics.sample_count ||
      final_snapshot.generated_codec_frames <
          contract_metrics.sample_count / kCodecFrameSamples ||
      final_snapshot.committed_text_tokens <
          contract_metrics.last_alignment_token ||
      !cancelled_at.has_value()) {
    throw std::runtime_error(
        "cancel probe final snapshot is inconsistent");
  }
  destroy_terminal_request(native);
  std::optional<double> first_audio_ms;
  if (first_arrival.has_value()) {
    first_audio_ms = milliseconds(*first_arrival - started_at);
  }
  return CancelResult{
      .case_id = corpus_case.case_id,
      .injection_point =
          std::string(cancel_injection_point_name(injection_point)),
      .audio_chunks_before_cancel = audio_chunks_before_cancel,
      .first_audio_ms = first_audio_ms,
      .cancel_call_ms =
          milliseconds(cancel_returned_at - cancel_started_at),
      .cancel_latency_ms =
          milliseconds(*cancelled_at - cancel_started_at),
      .audio_samples_before_cancel =
          audio_samples_before_cancel,
      .post_cancel_lease_count = post_cancel_leases,
      .terminal_cancelled = true,
  };
}

[[nodiscard]] double percentile(
    std::vector<double> values, const double quantile) {
  if (values.empty() || quantile < 0.0 || quantile > 1.0) {
    throw std::invalid_argument("invalid percentile input");
  }
  std::sort(values.begin(), values.end());
  const double position =
      quantile * static_cast<double>(values.size() - 1U);
  const auto lower =
      static_cast<std::size_t>(std::floor(position));
  const auto upper =
      static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return values[lower] +
         (values[upper] - values[lower]) * fraction;
}

[[nodiscard]] Distribution distribution(
    const std::vector<double>& values) {
  if (values.empty()) {
    throw std::invalid_argument(
        "cannot summarize an empty distribution");
  }
  return Distribution{
      .median = percentile(values, 0.5),
      .p95 = percentile(values, 0.95),
      .maximum = *std::max_element(values.begin(), values.end()),
  };
}

[[nodiscard]] LongRunAcceptanceGates
evaluate_long_run_acceptance_gates(
    const std::optional<Distribution>& ttfa,
    const std::optional<Distribution>& generation,
    const std::optional<Distribution>& playback_lateness,
    const std::int64_t process_rss_growth,
    const std::optional<double> process_rss_slope) {
  return LongRunAcceptanceGates{
      .ttfa_regression_passed =
          ttfa.has_value() &&
          ttfa->p95 <= kRegressionWarmTtfaP95Ms,
      .generation_regression_passed =
          generation.has_value() &&
          generation->p95 <= kRegressionGenerationRtfP95,
      .playback_lateness_regression_passed =
          playback_lateness.has_value() &&
          playback_lateness->maximum <= 0.0,
      .process_rss_growth_passed =
          process_rss_growth <=
          static_cast<std::int64_t>(
              kMaximumLongRunProcessRssGrowthBytes),
      .process_rss_slope_passed =
          process_rss_slope.has_value() &&
          *process_rss_slope <=
              kMaximumLongRunProcessRssSlopeBytesPerIteration,
  };
}

[[nodiscard]] double linear_slope(
    const std::vector<std::uint64_t>& values) {
  if (values.size() < 2U) {
    return 0.0;
  }
  const double count = static_cast<double>(values.size());
  const double mean_x = (count - 1.0) / 2.0;
  const double mean_y =
      std::accumulate(
          values.begin(), values.end(), 0.0,
          [](const double sum, const std::uint64_t value) {
            return sum + static_cast<double>(value);
          }) /
      count;
  double numerator = 0.0;
  double denominator = 0.0;
  for (std::size_t index = 0U; index < values.size(); ++index) {
    const double x = static_cast<double>(index) - mean_x;
    numerator +=
        x * (static_cast<double>(values[index]) - mean_y);
    denominator += x * x;
  }
  if (denominator == 0.0) {
    throw std::logic_error("memory trend denominator is zero");
  }
  return numerator / denominator;
}

[[nodiscard]] std::int64_t signed_byte_delta(
    const std::uint64_t current,
    const std::uint64_t baseline) {
  if (current >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::int64_t>::max()) ||
      baseline >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::int64_t>::max())) {
    throw std::logic_error(
        "process RSS is outside the signed receipt range");
  }
  return static_cast<std::int64_t>(current) -
         static_cast<std::int64_t>(baseline);
}

[[nodiscard]] nlohmann::json memory_json(
    const MemoryPoint& memory) {
  nlohmann::json result{
      {"process_rss_bytes", memory.process_rss_bytes},
  };
  result["cuda_total_bytes"] =
      memory.cuda_total_bytes.has_value()
          ? nlohmann::json(*memory.cuda_total_bytes)
          : nlohmann::json(nullptr);
  result["cuda_free_bytes"] =
      memory.cuda_free_bytes.has_value()
          ? nlohmann::json(*memory.cuda_free_bytes)
          : nlohmann::json(nullptr);
  result["cuda_used_bytes"] =
      memory.cuda_used_bytes.has_value()
          ? nlohmann::json(*memory.cuda_used_bytes)
          : nlohmann::json(nullptr);
  return result;
}

[[nodiscard]] nlohmann::json distribution_json(
    const Distribution& summary) {
  return {
      {"median", summary.median},
      {"p95", summary.p95},
      {"maximum", summary.maximum},
  };
}

[[nodiscard]] nlohmann::json chunk_json(
    const ChunkTiming& chunk) {
  nlohmann::json result{
      {"sequence", chunk.sequence},
      {"first_sample_index", chunk.first_sample_index},
      {"sample_count", chunk.sample_count},
      {"codec_frames", chunk.codec_frames},
      {"arrival_ms", chunk.arrival_ms},
      {"positive_playback_lateness_ms",
       chunk.positive_playback_lateness_ms},
      {"first", chunk.first},
      {"final", chunk.final},
  };
  result["interval_ms"] =
      chunk.interval_ms.has_value()
          ? nlohmann::json(*chunk.interval_ms)
          : nlohmann::json(nullptr);
  return result;
}

[[nodiscard]] nlohmann::json case_json(
    const CaseResult& result) {
  nlohmann::json chunks = nlohmann::json::array();
  for (const auto& chunk : result.chunks) {
    chunks.push_back(chunk_json(chunk));
  }
  return {
      {"case_id", result.case_id},
      {"prepared_token_count", result.prepared_token_count},
      {"random_seed", result.random_seed},
      {"request_start_call_ms", result.request_start_call_ms},
      {"request_start_to_first_audio_ms",
       result.request_start_to_first_audio_ms},
      {"ttfa_ms", result.ttfa_ms},
      {"generation_ms", result.generation_ms},
      {"total_ms", result.total_ms},
      {"audio_samples", result.audio_samples},
      {"audio_seconds", result.audio_seconds},
      {"generation_rtf", result.generation_rtf},
      {"total_rtf", result.total_rtf},
      {"maximum_positive_playback_lateness_ms",
       result.maximum_positive_playback_lateness_ms},
      {"chunks", std::move(chunks)},
  };
}

[[nodiscard]] nlohmann::json cancel_json(
    const CancelResult& result) {
  nlohmann::json document{
      {"case_id", result.case_id},
      {"injection_point", result.injection_point},
      {"audio_chunks_before_cancel",
       result.audio_chunks_before_cancel},
      {"cancel_call_ms", result.cancel_call_ms},
      {"cancel_latency_ms", result.cancel_latency_ms},
      {"audio_samples_before_cancel",
       result.audio_samples_before_cancel},
      {"post_cancel_lease_count",
       result.post_cancel_lease_count},
      {"terminal_cancelled", result.terminal_cancelled},
  };
  document["first_audio_ms"] =
      result.first_audio_ms.has_value()
          ? nlohmann::json(*result.first_audio_ms)
          : nlohmann::json(nullptr);
  return document;
}

template <typename Projection>
[[nodiscard]] std::vector<double> projected_values(
    const std::vector<CaseResult>& results, Projection projection) {
  std::vector<double> values;
  values.reserve(results.size());
  for (const auto& result : results) {
    values.push_back(projection(result));
  }
  return values;
}

[[nodiscard]] std::vector<double> chunk_intervals(
    const std::vector<CaseResult>& results) {
  std::vector<double> values;
  for (const auto& result : results) {
    for (const auto& chunk : result.chunks) {
      if (chunk.interval_ms.has_value()) {
        values.push_back(*chunk.interval_ms);
      }
    }
  }
  return values;
}

[[nodiscard]] nlohmann::json memory_trend_json(
    const std::vector<LongRunIteration>& iterations,
    const bool process_rss) {
  std::vector<std::uint64_t> values;
  values.reserve(iterations.size());
  for (const auto& iteration : iterations) {
    if (process_rss) {
      values.push_back(iteration.memory.process_rss_bytes);
    } else if (iteration.memory.cuda_used_bytes.has_value()) {
      values.push_back(*iteration.memory.cuda_used_bytes);
    } else {
      throw std::logic_error(
          "long-run iteration lacks a CUDART memory sample");
    }
  }
  if (values.empty()) {
    return {
        {"sample_count", 0U},
        {"first_bytes", nullptr},
        {"last_bytes", nullptr},
        {"minimum_bytes", nullptr},
        {"maximum_bytes", nullptr},
        {"delta_bytes", nullptr},
        {"linear_slope_bytes_per_iteration", nullptr},
    };
  }
  const auto [minimum, maximum] =
      std::minmax_element(values.begin(), values.end());
  const auto first = values.front();
  const auto last = values.back();
  const auto delta = signed_byte_delta(last, first);
  return {
      {"sample_count", values.size()},
      {"first_bytes", first},
      {"last_bytes", last},
      {"minimum_bytes", *minimum},
      {"maximum_bytes", *maximum},
      {"delta_bytes", delta},
      {"linear_slope_bytes_per_iteration",
       values.size() >= 2U
           ? nlohmann::json(linear_slope(values))
           : nlohmann::json(nullptr)},
  };
}

[[nodiscard]] nlohmann::json build_benchmark_receipt(
    const BenchmarkArguments& arguments,
    const BenchmarkCorpus& corpus,
    const mtt_model_info_v1_t& model_info,
    const std::string& runtime_library_sha256,
    const std::string& cuda_runtime_library_sha256,
    const std::string& nvml_library_sha256,
    const NvmlSampler& nvml,
    const CudaMemorySampler& cuda_memory,
    const BenchmarkResults& results) {
  if (results.warm.empty()) {
    throw std::logic_error(
        "cannot build a benchmark receipt without warm cases");
  }
  const auto warm_ttfa = projected_values(
      results.warm,
      [](const CaseResult& result) { return result.ttfa_ms; });
  const auto warm_request_start_call = projected_values(
      results.warm,
      [](const CaseResult& result) {
        return result.request_start_call_ms;
      });
  const auto warm_request_start_to_first_audio = projected_values(
      results.warm,
      [](const CaseResult& result) {
        return result.request_start_to_first_audio_ms;
      });
  const auto warm_generation = projected_values(
      results.warm,
      [](const CaseResult& result) {
        return result.generation_rtf;
      });
  const auto warm_total = projected_values(
      results.warm,
      [](const CaseResult& result) { return result.total_rtf; });
  const auto warm_generation_ms = projected_values(
      results.warm,
      [](const CaseResult& result) {
        return result.generation_ms;
      });
  const auto warm_total_ms = projected_values(
      results.warm,
      [](const CaseResult& result) { return result.total_ms; });
  const auto warm_playback_lateness = projected_values(
      results.warm,
      [](const CaseResult& result) {
        return result.maximum_positive_playback_lateness_ms;
      });
  const auto intervals = chunk_intervals(results.warm);
  if (intervals.empty()) {
    throw std::logic_error(
        "warm benchmark did not contain inter-chunk intervals");
  }
  const auto ttfa_summary = distribution(warm_ttfa);
  const auto request_start_call_summary =
      distribution(warm_request_start_call);
  const auto request_start_to_first_audio_summary =
      distribution(warm_request_start_to_first_audio);
  const auto generation_summary = distribution(warm_generation);
  const auto total_summary = distribution(warm_total);
  const auto generation_ms_summary =
      distribution(warm_generation_ms);
  const auto total_ms_summary = distribution(warm_total_ms);
  const auto playback_lateness_summary =
      distribution(warm_playback_lateness);
  const auto interval_summary = distribution(intervals);

  std::vector<double> long_ttfa;
  std::vector<double> long_generation;
  std::vector<double> long_total;
  std::vector<double> long_playback_lateness;
  long_ttfa.reserve(results.long_run.size());
  long_generation.reserve(results.long_run.size());
  long_total.reserve(results.long_run.size());
  long_playback_lateness.reserve(results.long_run.size());
  for (const auto& iteration : results.long_run) {
    long_ttfa.push_back(iteration.ttfa_ms);
    long_generation.push_back(iteration.generation_rtf);
    long_total.push_back(iteration.total_rtf);
    long_playback_lateness.push_back(
        iteration.maximum_positive_playback_lateness_ms);
  }
  std::optional<Distribution> long_ttfa_summary;
  std::optional<Distribution> long_generation_summary;
  std::optional<Distribution> long_total_summary;
  std::optional<Distribution> long_playback_lateness_summary;
  if (!results.long_run.empty()) {
    long_ttfa_summary = distribution(long_ttfa);
    long_generation_summary = distribution(long_generation);
    long_total_summary = distribution(long_total);
    long_playback_lateness_summary =
        distribution(long_playback_lateness);
  }
  std::vector<std::uint64_t> long_process_rss;
  long_process_rss.reserve(results.long_run.size());
  for (const auto& iteration : results.long_run) {
    long_process_rss.push_back(iteration.memory.process_rss_bytes);
  }
  std::optional<double> long_process_rss_slope;
  if (long_process_rss.size() >= 2U) {
    long_process_rss_slope = linear_slope(long_process_rss);
  }
  const auto long_process_rss_growth =
      signed_byte_delta(
          results.after_long_run.process_rss_bytes,
          results.after_warm.process_rss_bytes);

  const bool warm_count_passed =
      results.warm.size() == kAcceptanceNormalCases;
  const bool ttfa_target_passed =
      ttfa_summary.p95 <= kTargetWarmTtfaP95Ms;
  const bool generation_target_passed =
      generation_summary.p95 <= kTargetGenerationRtfP95;
  const bool ttfa_regression_passed =
      ttfa_summary.p95 <= kRegressionWarmTtfaP95Ms;
  const bool generation_regression_passed =
      generation_summary.p95 <= kRegressionGenerationRtfP95;
  const bool playback_lateness_regression_passed =
      playback_lateness_summary.maximum <= 0.0;
  const std::array<std::string_view, 3U>
      required_cancel_injection_points{
          "immediately_after_request_start",
          "after_first_audio_chunk",
          "after_second_audio_chunk",
      };
  bool cancel_passed =
      results.cancellations.size() ==
      required_cancel_injection_points.size();
  for (std::size_t index = 0U;
       index < results.cancellations.size(); ++index) {
    const auto& cancellation = results.cancellations[index];
    cancel_passed =
        cancel_passed &&
        index < required_cancel_injection_points.size() &&
        cancellation.injection_point ==
            required_cancel_injection_points[index] &&
        cancellation.audio_chunks_before_cancel == index &&
        cancellation.terminal_cancelled &&
        cancellation.post_cancel_lease_count == 0U;
  }
  const bool long_run_passed =
      !results.long_run_failure.has_value() &&
      arguments.long_run_iterations ==
          kAcceptanceLongRunIterations &&
      results.long_run.size() ==
          kAcceptanceLongRunIterations;
  const auto long_run_acceptance =
      evaluate_long_run_acceptance_gates(
          long_ttfa_summary, long_generation_summary,
          long_playback_lateness_summary,
          long_process_rss_growth, long_process_rss_slope);
  const bool long_run_ttfa_regression_passed =
      long_run_acceptance.ttfa_regression_passed;
  const bool long_run_generation_regression_passed =
      long_run_acceptance.generation_regression_passed;
  const bool long_run_playback_lateness_regression_passed =
      long_run_acceptance.playback_lateness_regression_passed;
  const bool long_run_process_rss_growth_passed =
      long_run_acceptance.process_rss_growth_passed;
  const bool long_run_process_rss_slope_passed =
      long_run_acceptance.process_rss_slope_passed;
  const bool accepted =
      warm_count_passed && ttfa_target_passed &&
      generation_target_passed && cancel_passed &&
      playback_lateness_regression_passed && long_run_passed &&
      long_run_ttfa_regression_passed &&
      long_run_generation_regression_passed &&
      long_run_playback_lateness_regression_passed &&
      long_run_process_rss_growth_passed &&
      long_run_process_rss_slope_passed &&
      results.native_cleanup_passed &&
      results.cuda_cleanup_passed &&
      results.nvml_cleanup_passed;

  nlohmann::json warm_cases = nlohmann::json::array();
  for (const auto& result : results.warm) {
    warm_cases.push_back(case_json(result));
  }
  nlohmann::json cancellation_results =
      nlohmann::json::array();
  for (const auto& cancellation : results.cancellations) {
    cancellation_results.push_back(cancel_json(cancellation));
  }

  nlohmann::json long_failure = nullptr;
  if (results.long_run_failure.has_value()) {
    long_failure = {
        {"iteration", results.long_run_failure->iteration},
        {"case_id", results.long_run_failure->case_id},
        {"diagnostic", "request_exception"},
        {"diagnostic_sha256",
         results.long_run_failure->diagnostic_sha256},
    };
  }

  nlohmann::json long_distributions = nullptr;
  if (long_ttfa_summary.has_value() &&
      long_generation_summary.has_value() &&
      long_total_summary.has_value() &&
      long_playback_lateness_summary.has_value()) {
    long_distributions = {
        {"ttfa_ms", distribution_json(*long_ttfa_summary)},
        {"generation_rtf",
         distribution_json(*long_generation_summary)},
        {"total_rtf", distribution_json(*long_total_summary)},
        {"positive_playback_lateness_ms",
         distribution_json(*long_playback_lateness_summary)},
    };
  }

  const auto completed_at_unix_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  const auto user_cpu_delta =
      results.cpu_after_cleanup.user_seconds -
      results.cpu_before_runtime.user_seconds;
  const auto system_cpu_delta =
      results.cpu_after_cleanup.system_seconds -
      results.cpu_before_runtime.system_seconds;
  if (user_cpu_delta < 0.0 || system_cpu_delta < 0.0 ||
      results.benchmark_wall_seconds <= 0.0) {
    throw std::logic_error(
        "process CPU or wall time is non-monotonic");
  }
  const auto cpu_to_wall_ratio =
      (user_cpu_delta + system_cpu_delta) /
      results.benchmark_wall_seconds;
  return {
      {"schema_version",
       "magpie-tts-rt.acceptance-benchmark.v2"},
      {"status", accepted ? "accepted" : "rejected"},
      {"completed_at_unix_ms", completed_at_unix_ms},
      {"abi_version", MTT_ABI_VERSION_1},
      {"measurement_scope",
       {
           {"input_contract", "prepared_token_ids"},
           {"process_scope",
            "mtt-runtime-benchmark process and the native runtime "
            "loaded into that process"},
           {"measured",
            {
                "native C ABI request_start to first acquired PCM",
                "native stream generation and drain",
                "benchmark-process RSS and CPU time",
                "device-wide CUDA free and used memory",
            }},
           {"not_measured",
            {
                "text normalization and tokenization latency",
                "ROS transport and playback queue latency",
                "physical audio-device start or drain latency",
                "per-engine TensorRT latency",
                "process-isolated GPU utilization or VRAM",
            }},
       }},
      {"ttfa_stages",
       {
           {"frontend",
            {
                {"measured", false},
                {"value_ms", nullptr},
                {"boundary",
                 "source text input to prepared_token_ids"},
                {"reason",
                 "the pinned corpus already contains prepared token "
                 "IDs; no frontend executes in this process"},
            }},
           {"native_prepared_tokens",
            {
                {"measured", true},
                {"warm_ms", distribution_json(ttfa_summary)},
                {"boundary",
                 "immediately before request_start to the first "
                 "successful audio_acquire"},
            }},
           {"ros_transport",
            {
                {"measured", false},
                {"value_ms", nullptr},
                {"boundary",
                 "first C ABI PCM lease to first PCM accepted by the "
                 "ROS playback consumer"},
                {"reason",
                 "the native benchmark does not start ROS"},
            }},
           {"physical_output",
            {
                {"measured", false},
                {"value_ms", nullptr},
                {"boundary",
                 "first PCM accepted by the playback consumer to "
                 "first physical audio"},
                {"reason",
                 "the native benchmark does not open an audio device"},
            }},
       }},
      {"component_latency",
       {
           {"request_start_call_ms",
            distribution_json(request_start_call_summary)},
           {"request_start_to_first_audio_ms",
            distribution_json(
                request_start_to_first_audio_summary)},
           {"native_prepared_ttfa_ms",
            distribution_json(ttfa_summary)},
           {"engine_internal",
            {
                {"measured", false},
                {"reason",
                 "the C ABI v1 exposes no per-engine timestamps; "
                 "an nsys report is retained as companion evidence"},
            }},
       }},
      {"resource_scope",
       {
           {"process_rss",
            {
                {"measured", true},
                {"scope",
                 "benchmark process including the loaded TTS runtime"},
                {"source", "linux_procfs_vmrss"},
            }},
           {"process_cpu",
            {
                {"measured", true},
                {"scope",
                 "benchmark process including the loaded TTS runtime"},
                {"source", "getrusage(RUSAGE_SELF)"},
            }},
           {"cuda_memory",
            {
                {"measured", true},
                {"scope",
                 "device-wide CUDA memory; not process-isolated VRAM "
                 "on AGX Thor unified memory"},
                {"source", "cudaMemGetInfo"},
            }},
           {"process_isolated_vram",
            {
                {"measured", false},
                {"reason",
                 "cudaMemGetInfo is device-wide and NVML framebuffer "
                 "accounting is unsupported on AGX Thor UMA"},
            }},
           {"gpu_utilization",
            {
                {"measured", false},
                {"reason",
                 "GPU utilization is captured by the workflow's "
                 "companion nsys and tegrastats artifacts, not by "
                 "this in-process receipt"},
            }},
           {"companion_profile_manifest",
            {
                {"required", true},
                {"schema_version",
                 "magpie-tts-rt.thor-profile-evidence.v1"},
                {"generated_after_benchmark_exit", true},
                {"required_artifacts",
                 {
                     "benchmark_receipt",
                     "nsys_report",
                     "tegrastats_log",
                 }},
            }},
       }},
      {"inputs",
       {
           {"library_name",
            public_artifact_name(
                arguments.native.library, "runtime library")},
           {"library_sha256", runtime_library_sha256},
           {"library_size_bytes",
            public_regular_file_size(
                arguments.native.library, "runtime library")},
           {"bundle_name",
            public_artifact_name(
                arguments.native.bundle, "runtime bundle")},
           {"bundle_manifest_name", "runtime-bundle-manifest.json"},
           {"bundle_manifest_sha256",
            arguments.native.manifest_sha256_hex},
           {"bundle_manifest_size_bytes",
            public_regular_file_size(
                arguments.native.bundle /
                    "runtime-bundle-manifest.json",
                "runtime bundle manifest")},
           {"corpus_name",
            public_artifact_name(
                arguments.corpus, "benchmark corpus")},
           {"corpus_sha256", corpus.file_sha256},
           {"corpus_size_bytes",
            public_regular_file_size(
                arguments.corpus, "benchmark corpus")},
           {"corpus_id", corpus.corpus_id},
           {"cuda_device_index",
            arguments.native.cuda_device_index},
           {"cuda_runtime_library_name",
            public_artifact_name(
                arguments.cuda_runtime_library,
                "CUDA runtime library")},
           {"cuda_runtime_library_sha256",
            cuda_runtime_library_sha256},
           {"cuda_runtime_library_size_bytes",
            public_regular_file_size(
                arguments.cuda_runtime_library,
                "CUDA runtime library")},
           {"cuda_driver_version",
            cuda_memory.driver_version()},
           {"cuda_runtime_version",
            cuda_memory.runtime_version()},
           {"nvml_library_name",
            public_artifact_name(
                arguments.nvml_library, "NVML library")},
           {"nvml_library_sha256", nvml_library_sha256},
           {"nvml_library_size_bytes",
            public_regular_file_size(
                arguments.nvml_library, "NVML library")},
           {"nvml_device_index", arguments.nvml_device_index},
           {"nvml_driver_version", nvml.driver_version()},
           {"nvml_device_name", nvml.device_name()},
           {"nvml_device_uuid", nvml.device_uuid()},
           {"nvml_compute_capability_major",
            nvml.compute_capability_major()},
           {"nvml_compute_capability_minor",
            nvml.compute_capability_minor()},
           {"request_timeout_ms",
            arguments.native.request_timeout.count()},
           {"normal_case_count", corpus.normal_cases.size()},
           {"long_run_iterations",
            arguments.long_run_iterations},
       }},
      {"model",
       {
           {"tokenizer_vocabulary_size",
            model_info.tokenizer_vocabulary_size},
           {"text_embedding_rows",
            model_info.text_embedding_rows},
           {"bos_token_id", model_info.bos_token_id},
           {"eos_token_id", model_info.eos_token_id},
           {"japanese_global_pad_token_id",
            model_info.japanese_global_pad_token_id},
           {"maximum_text_tokens",
            model_info.maximum_text_tokens},
           {"maximum_audio_frames",
            model_info.maximum_audio_frames},
           {"tokenizer_identity_sha256",
            sha256_hex(
                model_info.tokenizer_identity_sha256)},
       }},
      {"stream_contract",
       {
           {"sample_rate_hz", model_info.sample_rate_hz},
           {"channels", model_info.channels},
           {"pcm_format", "f32_mono"},
           {"codec_frame_samples",
            model_info.codec_frame_samples},
           {"initial_frames", model_info.initial_frames},
           {"steady_frames", model_info.steady_frames},
           {"tail_min_frames",
            model_info.tail_min_frames},
           {"tail_max_frames",
            model_info.tail_max_frames},
       }},
      {"startup",
       {
           {"startup_gate_ms", results.startup_gate_ms},
           {"definition",
            "runtime/model/session creation including the mandatory "
            "session startup golden"},
       }},
      {"cold",
       {
           {"definition",
            "first user request after the mandatory session startup "
            "golden"},
           {"result", case_json(results.cold)},
       }},
      {"warm",
       {
           {"definition",
            "all pinned normal corpus cases, sequentially submitted "
            "through the same persistent session after the cold probe"},
           {"case_count", results.warm.size()},
           {"ttfa_ms", distribution_json(ttfa_summary)},
           {"generation_ms",
            distribution_json(generation_ms_summary)},
           {"total_ms",
            distribution_json(total_ms_summary)},
           {"generation_rtf",
            distribution_json(generation_summary)},
           {"total_rtf",
            distribution_json(total_summary)},
           {"chunk_interval_ms",
            distribution_json(interval_summary)},
           {"positive_playback_lateness_ms",
            distribution_json(playback_lateness_summary)},
           {"cases", std::move(warm_cases)},
      }},
      {"cancellation",
       {
           {"definition",
            "the same pinned cancel case is restarted and cancelled "
            "at each fixed injection point"},
           {"required_injection_points",
            {
                "immediately_after_request_start",
                "after_first_audio_chunk",
                "after_second_audio_chunk",
            }},
           {"results", std::move(cancellation_results)},
       }},
      {"long_run",
       {
           {"requested_iterations",
            arguments.long_run_iterations},
           {"completed_iterations", results.long_run.size()},
           {"failure", std::move(long_failure)},
           {"distributions", std::move(long_distributions)},
           {"process_rss_trend",
            memory_trend_json(results.long_run, true)},
           {"process_rss_stability",
            {
                {"source", "linux_procfs_vmrss"},
                {"acceptance_scope",
                 "benchmark process including the loaded TTS "
                 "runtime; device-wide CUDA memory is excluded"},
                {"baseline_phase", "after_warm"},
                {"final_phase", "after_long_run"},
                {"baseline_bytes",
                 results.after_warm.process_rss_bytes},
                {"final_bytes",
                 results.after_long_run.process_rss_bytes},
                {"growth_bytes", long_process_rss_growth},
                {"maximum_growth_bytes",
                 kMaximumLongRunProcessRssGrowthBytes},
                {"linear_slope_bytes_per_iteration",
                 long_process_rss_slope.has_value()
                     ? nlohmann::json(*long_process_rss_slope)
                     : nlohmann::json(nullptr)},
                {"maximum_linear_slope_bytes_per_iteration",
                 kMaximumLongRunProcessRssSlopeBytesPerIteration},
            }},
           {"cuda_used_memory_trend",
            memory_trend_json(results.long_run, false)},
       }},
      {"memory",
       {
           {"sources",
           {
                {"process_rss",
                 "linux_procfs_vmrss"},
                {"device_memory", "cudaMemGetInfo"},
                {"cuda_first_call_phase",
                 "after_startup_gate"},
                {"nvml_device_memory_status",
                 nvml.memory_api_status()},
            }},
           {"before_runtime",
            memory_json(results.before_runtime)},
           {"after_startup",
            memory_json(results.after_startup)},
           {"after_cold", memory_json(results.after_cold)},
           {"after_warm", memory_json(results.after_warm)},
           {"after_cancel",
            memory_json(results.after_cancel)},
           {"after_long_run",
            memory_json(results.after_long_run)},
           {"after_native_cleanup",
            memory_json(results.after_native_cleanup)},
       }},
      {"cpu",
       {
           {"source", "getrusage(RUSAGE_SELF)"},
           {"before_runtime",
            {
                {"user_seconds",
                 results.cpu_before_runtime.user_seconds},
                {"system_seconds",
                 results.cpu_before_runtime.system_seconds},
            }},
           {"after_cleanup",
            {
                {"user_seconds",
                 results.cpu_after_cleanup.user_seconds},
                {"system_seconds",
                 results.cpu_after_cleanup.system_seconds},
            }},
           {"delta_user_seconds", user_cpu_delta},
           {"delta_system_seconds", system_cpu_delta},
           {"measurement_wall_seconds",
            results.benchmark_wall_seconds},
           {"cpu_time_to_wall_ratio", cpu_to_wall_ratio},
       }},
      {"thresholds",
       {
           {"warm_case_count_exactly_108",
            {
                {"required", kAcceptanceNormalCases},
                {"observed", results.warm.size()},
                {"passed", warm_count_passed},
            }},
           {"warm_ttfa_p95_target_ms",
            {
                {"maximum", kTargetWarmTtfaP95Ms},
                {"observed", ttfa_summary.p95},
                {"passed", ttfa_target_passed},
            }},
           {"generation_rtf_p95_target",
            {
                {"maximum", kTargetGenerationRtfP95},
                {"observed", generation_summary.p95},
                {"passed", generation_target_passed},
            }},
           {"warm_ttfa_p95_regression_ceiling_ms",
            {
                {"maximum", kRegressionWarmTtfaP95Ms},
                {"observed", ttfa_summary.p95},
                {"passed", ttfa_regression_passed},
            }},
           {"generation_rtf_p95_regression_ceiling",
            {
                {"maximum", kRegressionGenerationRtfP95},
                {"observed", generation_summary.p95},
                {"passed", generation_regression_passed},
            }},
           {"maximum_positive_playback_lateness_regression_ceiling_ms",
            {
                {"maximum", 0.0},
                {"observed", playback_lateness_summary.maximum},
                {"passed", playback_lateness_regression_passed},
            }},
           {"all_cancel_points_terminal_without_post_cancel_pcm",
            {
                {"passed", cancel_passed},
            }},
           {"long_run_exactly_1000_without_failure",
            {
                {"required", kAcceptanceLongRunIterations},
                {"observed", results.long_run.size()},
                {"passed", long_run_passed},
            }},
           {"long_run_ttfa_p95_regression_ceiling_ms",
            {
                {"maximum", kRegressionWarmTtfaP95Ms},
                {"observed",
                 long_ttfa_summary.has_value()
                     ? nlohmann::json(long_ttfa_summary->p95)
                     : nlohmann::json(nullptr)},
                {"passed", long_run_ttfa_regression_passed},
            }},
           {"long_run_generation_rtf_p95_regression_ceiling",
            {
                {"maximum", kRegressionGenerationRtfP95},
                {"observed",
                 long_generation_summary.has_value()
                     ? nlohmann::json(
                           long_generation_summary->p95)
                     : nlohmann::json(nullptr)},
                {"passed",
                 long_run_generation_regression_passed},
            }},
           {"long_run_maximum_positive_playback_lateness_regression_ceiling_ms",
            {
                {"maximum", 0.0},
                {"observed",
                 long_playback_lateness_summary.has_value()
                     ? nlohmann::json(
                           long_playback_lateness_summary->maximum)
                     : nlohmann::json(nullptr)},
                {"passed",
                 long_run_playback_lateness_regression_passed},
            }},
           {"long_run_process_rss_growth_from_warm_maximum_bytes",
            {
                {"maximum",
                 kMaximumLongRunProcessRssGrowthBytes},
                {"observed", long_process_rss_growth},
                {"passed", long_run_process_rss_growth_passed},
            }},
           {"long_run_process_rss_linear_slope_maximum_bytes_per_iteration",
            {
                {"maximum",
                 kMaximumLongRunProcessRssSlopeBytesPerIteration},
                {"observed",
                 long_process_rss_slope.has_value()
                     ? nlohmann::json(*long_process_rss_slope)
                     : nlohmann::json(nullptr)},
                {"passed", long_run_process_rss_slope_passed},
            }},
       }},
      {"verification",
       {
           {"startup_golden", "passed"},
           {"corpus_tokenizer_identity", "passed"},
           {"stream_first_to_final",
            results.long_run_failure.has_value()
                ? "failed"
                : "passed"},
           {"snapshot_contract",
            results.long_run_failure.has_value()
                ? "failed"
                : "passed"},
           {"cancel_contract",
            cancel_passed ? "passed" : "failed"},
           {"long_run_performance_regression",
            long_run_ttfa_regression_passed &&
                    long_run_generation_regression_passed &&
                    long_run_playback_lateness_regression_passed
                ? "passed"
                : "failed"},
           {"long_run_process_rss_stability",
            long_run_process_rss_growth_passed &&
                    long_run_process_rss_slope_passed
                ? "passed"
                : "failed"},
           {"native_cleanup",
            results.native_cleanup_passed ? "passed" : "failed"},
           {"cuda_cleanup",
            results.cuda_cleanup_passed ? "passed" : "failed"},
           {"nvml_cleanup",
            results.nvml_cleanup_passed ? "passed" : "failed"},
       }},
  };
}

[[nodiscard]] int run_benchmark(const int argc, char** argv) {
  const auto arguments = parse_benchmark_arguments(argc, argv);
  if (arguments.help) {
    std::cout << benchmark_usage() << '\n';
    return 0;
  }
  const auto corpus = load_benchmark_corpus(arguments.corpus);
  const auto runtime_library_sha256 =
      file_digest_hex(arguments.native.library);
  const auto cuda_runtime_library_sha256 =
      file_digest_hex(arguments.cuda_runtime_library);
  const auto nvml_library_sha256 =
      file_digest_hex(arguments.nvml_library);
  NativeObjects native;
  BenchmarkResults results;
  mtt_model_info_v1_t model_info{};
  try {
    const auto benchmark_started_at = Clock::now();
    results.cpu_before_runtime = process_cpu_times();
    results.before_runtime = process_only_memory_point();
    const auto startup_started_at = Clock::now();
    load_native(arguments.native, native, model_info);
    results.startup_gate_ms =
        milliseconds(Clock::now() - startup_started_at);
    NvmlSampler nvml(
        arguments.nvml_library, arguments.nvml_device_index);
    CudaMemorySampler cuda_memory(
        arguments.cuda_runtime_library);
    cuda_memory.activate(arguments.native.cuda_device_index);
    if (corpus.tokenizer_identity_sha256 !=
        sha256_hex(model_info.tokenizer_identity_sha256)) {
      throw std::runtime_error(
          "corpus tokenizer identity does not match the "
          "authenticated model");
    }
    for (const auto& corpus_case : corpus.normal_cases) {
      validate_case_against_model(corpus_case, model_info);
    }
    validate_case_against_model(corpus.cancel_case, model_info);
    results.after_startup = cuda_memory.sample();

    results.cold = run_completed_case(
        arguments, corpus.normal_cases.front(), native, model_info);
    results.after_cold = cuda_memory.sample();

    results.warm.reserve(corpus.normal_cases.size());
    for (const auto& corpus_case : corpus.normal_cases) {
      results.warm.push_back(run_completed_case(
          arguments, corpus_case, native, model_info));
    }
    results.after_warm = cuda_memory.sample();

    constexpr std::array<CancelInjectionPoint, 3U>
        cancellation_points{
            CancelInjectionPoint::kImmediatelyAfterRequestStart,
            CancelInjectionPoint::kAfterFirstAudioChunk,
            CancelInjectionPoint::kAfterSecondAudioChunk,
        };
    results.cancellations.reserve(cancellation_points.size());
    for (const auto point : cancellation_points) {
      results.cancellations.push_back(run_cancel_probe(
          arguments, corpus.cancel_case, native, model_info, point));
    }
    results.after_cancel = cuda_memory.sample();

    results.long_run.reserve(
        static_cast<std::size_t>(arguments.long_run_iterations));
    for (std::uint64_t iteration = 0U;
         iteration < arguments.long_run_iterations; ++iteration) {
      const auto& corpus_case =
          corpus.normal_cases[static_cast<std::size_t>(
              iteration %
              static_cast<std::uint64_t>(
                  corpus.normal_cases.size()))];
      try {
        const auto result = run_completed_case(
            arguments, corpus_case, native, model_info);
        results.long_run.push_back(LongRunIteration{
            .iteration = iteration,
            .case_id = result.case_id,
            .ttfa_ms = result.ttfa_ms,
            .generation_rtf = result.generation_rtf,
            .total_rtf = result.total_rtf,
            .maximum_positive_playback_lateness_ms =
                result.maximum_positive_playback_lateness_ms,
            .memory = cuda_memory.sample(),
        });
      } catch (const std::exception& exception) {
        results.long_run_failure = LongRunFailure{
            .iteration = iteration,
            .case_id = corpus_case.case_id,
            .diagnostic_sha256 = digest_hex(exception.what()),
        };
        break;
      }
    }
    results.after_long_run = cuda_memory.sample();
    results.native_cleanup_passed = cleanup_native(native);
    results.after_native_cleanup = cuda_memory.sample();
    cuda_memory.close();
    results.cuda_cleanup_passed = true;
    nvml.close();
    results.nvml_cleanup_passed = true;
    results.cpu_after_cleanup = process_cpu_times();
    results.benchmark_wall_seconds =
        std::chrono::duration<double>(
            Clock::now() - benchmark_started_at)
            .count();

    const auto receipt = build_benchmark_receipt(
        arguments, corpus, model_info, runtime_library_sha256,
        cuda_runtime_library_sha256, nvml_library_sha256, nvml,
        cuda_memory, results);
    write_new_receipt(*arguments.native.receipt_json, receipt);
    const bool accepted = receipt.at("status") == "accepted";
    std::cout << "status="
              << (accepted ? "accepted" : "rejected") << '\n'
              << "corpus_sha256=" << corpus.file_sha256 << '\n'
              << "normal_cases=" << results.warm.size() << '\n'
              << "long_run_completed=" << results.long_run.size()
              << '\n'
              << "warm_ttfa_p95_ms="
              << receipt.at("warm").at("ttfa_ms").at("p95")
              << '\n'
              << "warm_generation_rtf_p95="
              << receipt.at("warm").at("generation_rtf").at("p95")
              << '\n'
              << "cancel_points="
              << results.cancellations.size() << '\n'
              << "receipt_json="
              << arguments.native.receipt_json->string() << '\n';
    return accepted ? 0 : 2;
  } catch (...) {
    const bool clean = cleanup_native(native);
    if (!clean) {
      std::cerr << "native cleanup also failed\n";
    }
    throw;
  }
}

}  // namespace

#ifndef MTT_RUNTIME_BENCHMARK_TESTING
int main(const int argc, char** argv) {
  try {
    return run_benchmark(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "runtime benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
#endif
