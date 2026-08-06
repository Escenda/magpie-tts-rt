#define MTT_RUNTIME_BENCHMARK_TESTING
#include "../../tools/runtime_benchmark.cpp"

#include <cstdlib>
#include <iostream>

#ifndef MTT_RUNTIME_BENCHMARK_MOCK_LIBRARY
#error "MTT_RUNTIME_BENCHMARK_MOCK_LIBRARY is required"
#endif

#ifndef MTT_RUNTIME_BENCHMARK_MOCK_NVML
#error "MTT_RUNTIME_BENCHMARK_MOCK_NVML is required"
#endif

#ifndef MTT_RUNTIME_BENCHMARK_MOCK_CUDA
#error "MTT_RUNTIME_BENCHMARK_MOCK_CUDA is required"
#endif

namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    std::array<char, 64U> pattern{};
    constexpr std::string_view prefix{
        "/tmp/magpie-tts-rt-benchmark-test.XXXXXX"};
    std::copy(prefix.begin(), prefix.end(), pattern.begin());
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

  [[nodiscard]] const std::filesystem::path& path() const {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

void require(const bool condition, const std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

[[nodiscard]] bool public_string_contains_local_path(
    const std::string_view value) {
  if (value.find("file://") != std::string_view::npos ||
      value.find("/home/") != std::string_view::npos) {
    return true;
  }
  if (value.size() >= 3U &&
      ((value[0] >= 'a' && value[0] <= 'z') ||
       (value[0] >= 'A' && value[0] <= 'Z')) &&
      value[1] == ':' && (value[2] == '/' || value[2] == '\\')) {
    return true;
  }
  for (std::size_t index = 0U; index < value.size(); ++index) {
    if (value[index] == '/' &&
        (index == 0U || value[index - 1U] == ' ' ||
         value[index - 1U] == '\t' || value[index - 1U] == '\n')) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool receipt_contains_local_path(
    const nlohmann::json& value) {
  if (value.is_string()) {
    return public_string_contains_local_path(
        value.get_ref<const std::string&>());
  }
  if (value.is_array()) {
    return std::any_of(
        value.begin(), value.end(), [](const nlohmann::json& item) {
          return receipt_contains_local_path(item);
        });
  }
  if (value.is_object()) {
    return std::any_of(
        value.begin(), value.end(), [](const nlohmann::json& item) {
          return receipt_contains_local_path(item);
        });
  }
  return false;
}

template <typename Function>
void require_failure(
    Function&& function, const std::string_view message) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

void write_text(
    const std::filesystem::path& path,
    const std::string_view content) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to create test file");
  }
  output.write(
      content.data(),
      static_cast<std::streamsize>(content.size()));
  if (!output) {
    throw std::runtime_error("failed to write test file");
  }
}

[[nodiscard]] std::string corpus_bytes(
    const bool add_unknown_header_member = false,
    const std::uint64_t normal_case_count = 108U) {
  nlohmann::json header{
      {"record_type", "header"},
      {"schema_version",
       "magpie-tts-rt.benchmark-corpus.v1"},
      {"corpus_id", "cpu-mock-ja"},
      {"tokenizer_identity_sha256", std::string(64U, '1')},
      {"normal_case_count", normal_case_count},
  };
  if (add_unknown_header_member) {
    header["unknown"] = true;
  }
  std::string bytes = header.dump() + "\n";
  for (std::uint64_t index = 0U; index < normal_case_count; ++index) {
    const auto text =
        std::string("アスパの音声試験") + std::to_string(index);
    const nlohmann::json record{
        {"record_type", "case"},
        {"case_id", "case-" + std::to_string(index)},
        {"source_text", text},
        {"source_text_sha256", digest_hex(text)},
        {"prepared_token_ids", {1, 2, 3, 4'097}},
        {"random_seed", index},
    };
    bytes += record.dump() + "\n";
  }
  const std::string cancel_text{"アスパ、停止して"};
  const nlohmann::json cancel{
      {"record_type", "cancel_case"},
      {"case_id", "cancel-case"},
      {"source_text", cancel_text},
      {"source_text_sha256", digest_hex(cancel_text)},
      {"prepared_token_ids", {1, 2, 3, 4'097}},
      {"random_seed", 4'294'967'295ULL},
  };
  bytes += cancel.dump() + "\n";
  return bytes;
}

[[nodiscard]] int invoke_benchmark(
    const std::filesystem::path& bundle,
    const std::filesystem::path& corpus,
    const std::filesystem::path& receipt,
    const std::filesystem::path& cuda_library =
        MTT_RUNTIME_BENCHMARK_MOCK_CUDA,
    const std::string_view long_run_iterations = "1000") {
  std::vector<std::string> values{
      "mtt-runtime-benchmark",
      "--library",
      MTT_RUNTIME_BENCHMARK_MOCK_LIBRARY,
      "--bundle",
      bundle.string(),
      "--manifest-sha256",
      std::string(64U, 'a'),
      "--corpus",
      corpus.string(),
      "--cuda-device",
      "0",
      "--cuda-runtime-library",
      cuda_library.string(),
      "--nvml-library",
      MTT_RUNTIME_BENCHMARK_MOCK_NVML,
      "--nvml-device",
      "0",
      "--long-run-iterations",
      std::string(long_run_iterations),
      "--receipt-json",
      receipt.string(),
      "--request-timeout-ms",
      "1000",
  };
  std::vector<char*> arguments;
  arguments.reserve(values.size());
  for (auto& value : values) {
    arguments.push_back(value.data());
  }
  return run_benchmark(
      static_cast<int>(arguments.size()), arguments.data());
}

void test_help() {
  char program[] = "mtt-runtime-benchmark";
  char help[] = "--help";
  char* arguments[] = {program, help};
  require(
      run_benchmark(2, arguments) == 0,
      "benchmark --help must succeed");
}

void test_corpus_schema_rejects_unknown_members(
    const TemporaryDirectory& temporary) {
  const auto invalid = temporary.path() / "invalid.jsonl";
  write_text(invalid, corpus_bytes(true));
  require_failure(
      [&]() {
        static_cast<void>(load_benchmark_corpus(invalid));
      },
      "unknown corpus members must fail closed");
  require_failure(
      [&]() {
        static_cast<void>(parse_strict_json_line(
            R"({"record_type":"header","record_type":"header"})",
            1U));
      },
      "duplicate JSON members must fail closed");
}

void test_acceptance_counts_fail_closed(
    const TemporaryDirectory& temporary) {
  const auto wrong_count =
      temporary.path() / "wrong-count.jsonl";
  write_text(wrong_count, corpus_bytes(false, 107U));
  require_failure(
      [&]() {
        static_cast<void>(load_benchmark_corpus(wrong_count));
      },
      "107 normal cases must not satisfy the 108-case gate");

  const auto bundle = temporary.path() / "wrong-long-run-bundle";
  std::filesystem::create_directory(bundle);
  const auto corpus =
      temporary.path() / "wrong-long-run-corpus.jsonl";
  const auto receipt =
      temporary.path() / "wrong-long-run-receipt.json";
  write_text(corpus, corpus_bytes());
  require_failure(
      [&]() {
        static_cast<void>(invoke_benchmark(
            bundle, corpus, receipt,
            MTT_RUNTIME_BENCHMARK_MOCK_CUDA, "999"));
      },
      "999 turns must not satisfy the 1000-turn gate");
  require(
      !std::filesystem::exists(receipt),
      "an invalid turn count must not publish a receipt");
}

void test_long_run_threshold_boundaries_fail_closed() {
  const Distribution ttfa{
      .median = 100.0,
      .p95 = 152.39,
      .maximum = 152.39,
  };
  const Distribution generation{
      .median = 0.30,
      .p95 = 0.401,
      .maximum = 0.401,
  };
  const Distribution playback_lateness{
      .median = 0.0,
      .p95 = 0.0,
      .maximum = 0.0,
  };
  const auto boundary = evaluate_long_run_acceptance_gates(
      ttfa, generation, playback_lateness, 67'108'864,
      65'536.0);
  require(
      boundary.ttfa_regression_passed &&
          boundary.generation_regression_passed &&
          boundary.playback_lateness_regression_passed &&
          boundary.process_rss_growth_passed &&
          boundary.process_rss_slope_passed,
      "exact long-run threshold boundaries must pass");

  auto changed_ttfa = ttfa;
  changed_ttfa.p95 = 152.390'001;
  require(
      !evaluate_long_run_acceptance_gates(
           changed_ttfa, generation, playback_lateness,
           67'108'864, 65'536.0)
           .ttfa_regression_passed,
      "TTFA above the fixed regression ceiling must fail");

  auto changed_generation = generation;
  changed_generation.p95 = 0.401'001;
  require(
      !evaluate_long_run_acceptance_gates(
           ttfa, changed_generation, playback_lateness,
           67'108'864, 65'536.0)
           .generation_regression_passed,
      "generation RTF above the fixed ceiling must fail");

  auto changed_lateness = playback_lateness;
  changed_lateness.maximum = 0.001;
  require(
      !evaluate_long_run_acceptance_gates(
           ttfa, generation, changed_lateness,
           67'108'864, 65'536.0)
           .playback_lateness_regression_passed,
      "positive long-run playback lateness must fail");

  require(
      !evaluate_long_run_acceptance_gates(
           ttfa, generation, playback_lateness,
           67'108'865, 65'536.0)
           .process_rss_growth_passed,
      "process RSS growth above 64 MiB must fail");
  require(
      !evaluate_long_run_acceptance_gates(
           ttfa, generation, playback_lateness,
           67'108'864, 65'536.001)
           .process_rss_slope_passed,
      "process RSS slope above 64 KiB per iteration must fail");

  const auto missing = evaluate_long_run_acceptance_gates(
      std::nullopt, std::nullopt, std::nullopt, 0,
      std::nullopt);
  require(
      !missing.ttfa_regression_passed &&
          !missing.generation_regression_passed &&
          !missing.playback_lateness_regression_passed &&
          !missing.process_rss_slope_passed,
      "missing long-run observations must fail closed");
}

void test_complete_benchmark_and_immutable_receipt(
    const TemporaryDirectory& temporary) {
  const auto bundle = temporary.path() / "bundle";
  std::filesystem::create_directory(bundle);
  write_text(
      bundle / "runtime-bundle-manifest.json",
      R"({"schema_version":1})");
  const auto corpus = temporary.path() / "corpus.jsonl";
  const auto receipt = temporary.path() / "receipt.json";
  write_text(corpus, corpus_bytes());

  require(
      invoke_benchmark(bundle, corpus, receipt) == 0,
      "deterministic CPU benchmark must be accepted");
  const auto document =
      read_json_file(receipt, "benchmark receipt");
  const std::set<std::string> expected_top_level{
      "schema_version",
      "status",
      "completed_at_unix_ms",
      "abi_version",
      "measurement_scope",
      "ttfa_stages",
      "component_latency",
      "resource_scope",
      "inputs",
      "model",
      "stream_contract",
      "startup",
      "cold",
      "warm",
      "cancellation",
      "long_run",
      "memory",
      "cpu",
      "thresholds",
      "verification",
  };
  std::set<std::string> actual_top_level;
  for (const auto& [key, value] : document.items()) {
    static_cast<void>(value);
    actual_top_level.insert(key);
  }
  require(
      actual_top_level == expected_top_level,
      "receipt top-level contract changed");
  require(
      document.at("schema_version") ==
          "magpie-tts-rt.acceptance-benchmark.v2",
      "receipt schema version");
  require(document.at("status") == "accepted", "accepted status");
  require(
      document.at("inputs").at("library_sha256") ==
          file_digest_hex(MTT_RUNTIME_BENCHMARK_MOCK_LIBRARY),
      "runtime library identity");
  require(
      document.at("inputs").at("library_name") ==
              std::filesystem::path(
                  MTT_RUNTIME_BENCHMARK_MOCK_LIBRARY)
                  .filename()
                  .string() &&
          document.at("inputs").at("bundle_name") == "bundle" &&
          document.at("inputs").at("corpus_name") ==
              "corpus.jsonl",
      "public benchmark inputs contain basenames");
  require(
      document.at("inputs").at("library_size_bytes") ==
              std::filesystem::file_size(
                  MTT_RUNTIME_BENCHMARK_MOCK_LIBRARY) &&
          document.at("inputs").at("corpus_size_bytes") ==
              std::filesystem::file_size(corpus) &&
          document.at("inputs").at("bundle_manifest_size_bytes") ==
              std::filesystem::file_size(
                  bundle / "runtime-bundle-manifest.json"),
      "public benchmark inputs contain artifact sizes");
  require(
      document.at("inputs").at("nvml_library_sha256") ==
          file_digest_hex(MTT_RUNTIME_BENCHMARK_MOCK_NVML),
      "NVML library identity");
  require(
      document.at("inputs").at("cuda_runtime_library_sha256") ==
          file_digest_hex(MTT_RUNTIME_BENCHMARK_MOCK_CUDA),
      "CUDART library identity");
  require(
      document.at("inputs").at("cuda_runtime_library_name") ==
              std::filesystem::path(
                  MTT_RUNTIME_BENCHMARK_MOCK_CUDA)
                  .filename()
                  .string() &&
          document.at("inputs").at("nvml_library_name") ==
              std::filesystem::path(
                  MTT_RUNTIME_BENCHMARK_MOCK_NVML)
                  .filename()
                  .string(),
      "public dependency identities contain basenames");
  require(
      document.dump().find(temporary.path().string()) ==
          std::string::npos,
      "public benchmark receipt must not contain the runner-local root");
  require(
      !receipt_contains_local_path(document),
      "public benchmark receipt must not contain path-shaped strings");
  require(
      document.at("inputs").at("nvml_device_name") ==
          "NVIDIA Thor",
      "NVML device identity");
  require(
      document.at("inputs").at("nvml_compute_capability_major") ==
              11 &&
          document.at("inputs").at("nvml_compute_capability_minor") ==
              0,
      "NVML compute capability");
  require(
      document.at("warm").at("case_count") == 108U,
      "warm corpus count");
  require(
      document.at("warm").at("cases").size() == 108U,
      "warm per-case records");
  require(
      document.at("long_run").at("completed_iterations") == 1000U,
      "long-run completion count");
  require(
      document.at("long_run").at("requested_iterations") == 1000U,
      "long-run requested count");
  require(
      document.at("long_run").at("failure").is_null(),
      "long-run failure must be null");
  require(
      document.at("long_run")
              .at("cuda_used_memory_trend")
              .at("sample_count") == 1000U,
      "one CUDART memory sample per long-run request");
  require(
      document.at("memory")
              .at("sources")
              .at("nvml_device_memory_status") == "unsupported",
      "Thor UMA NVML memory status");
  require(
      document.at("memory")
          .at("before_runtime")
          .at("cuda_used_bytes")
          .is_null(),
      "CUDART must not initialize before the startup gate");
  require(
      document.at("memory")
          .at("after_startup")
          .at("cuda_used_bytes")
          .is_number_unsigned(),
      "post-startup CUDART memory sample");
  require(
      document.at("warm")
              .at("positive_playback_lateness_ms")
              .at("maximum") == 0.0,
      "mock stream must stay ahead of its playback timeline");
  require(
      document.at("thresholds")
              .at("maximum_positive_playback_lateness_regression_ceiling_ms")
              .at("passed") == true,
      "zero-lateness regression ceiling");
  require(
      document.at("cpu").at("measurement_wall_seconds")
              .get<double>() > 0.0,
      "benchmark CPU wall interval");
  require(
      document.at("cpu").at("delta_user_seconds")
              .get<double>() >= 0.0 &&
          document.at("cpu").at("delta_system_seconds")
                  .get<double>() >= 0.0 &&
          document.at("cpu").at("cpu_time_to_wall_ratio")
                  .get<double>() >= 0.0,
      "process CPU user/system/wall metrics");
  require(
      document.at("cancellation").at("results").size() == 3U,
      "three cancellation injection points");
  const std::array<std::string_view, 3U> expected_cancel_points{
      "immediately_after_request_start",
      "after_first_audio_chunk",
      "after_second_audio_chunk",
  };
  for (std::size_t index = 0U;
       index < expected_cancel_points.size(); ++index) {
    const auto& cancellation =
        document.at("cancellation").at("results").at(index);
    require(
        cancellation.at("injection_point") ==
                expected_cancel_points[index] &&
            cancellation.at("audio_chunks_before_cancel") == index &&
            cancellation.at("post_cancel_lease_count") == 0U &&
            cancellation.at("terminal_cancelled") == true,
        "each cancel point must terminate without post-cancel PCM");
  }
  require(
      document.at("cancellation")
          .at("results")
          .at(0)
          .at("first_audio_ms")
          .is_null(),
      "immediate cancellation has no first-audio measurement");
  require(
      document.at("thresholds")
              .at("warm_case_count_exactly_108")
              .at("passed") == true &&
          document.at("thresholds")
                  .at("long_run_exactly_1000_without_failure")
                  .at("passed") == true,
      "108-case and 1000-turn gates must pass separately");
  require(
      document.at("thresholds")
              .at("long_run_ttfa_p95_regression_ceiling_ms")
              .at("maximum") == 152.39 &&
          document.at("thresholds")
                  .at("long_run_generation_rtf_p95_regression_ceiling")
                  .at("maximum") == 0.401 &&
          document.at("thresholds")
                  .at("long_run_maximum_positive_playback_lateness_regression_ceiling_ms")
                  .at("maximum") == 0.0,
      "long-run performance regression limits are fixed");
  require(
      document.at("thresholds")
              .at("long_run_ttfa_p95_regression_ceiling_ms")
              .at("passed") == true &&
          document.at("thresholds")
                  .at("long_run_generation_rtf_p95_regression_ceiling")
                  .at("passed") == true &&
          document.at("thresholds")
                  .at("long_run_maximum_positive_playback_lateness_regression_ceiling_ms")
                  .at("passed") == true,
      "long-run performance must remain below all regression ceilings");
  require(
      document.at("thresholds")
              .at("long_run_process_rss_growth_from_warm_maximum_bytes")
              .at("maximum") == 64U * 1'024U * 1'024U &&
          document.at("thresholds")
                  .at("long_run_process_rss_linear_slope_maximum_bytes_per_iteration")
                  .at("maximum") == 64.0 * 1'024.0,
      "process RSS stability limits are fixed");
  require(
      document.at("thresholds")
              .at("long_run_process_rss_growth_from_warm_maximum_bytes")
              .at("passed") == true &&
          document.at("thresholds")
                  .at("long_run_process_rss_linear_slope_maximum_bytes_per_iteration")
                  .at("passed") == true,
      "mock long-run process RSS must remain stable");
  require(
      document.at("long_run")
              .at("process_rss_stability")
              .at("baseline_phase") == "after_warm" &&
          document.at("long_run")
                  .at("process_rss_stability")
                  .at("final_phase") == "after_long_run" &&
          document.at("long_run")
                  .at("process_rss_stability")
                  .at("acceptance_scope")
                  .get<std::string>()
                  .find("device-wide CUDA memory is excluded") !=
              std::string::npos,
      "RSS gate scope excludes device-wide CUDA memory");
  require(
      document.at("ttfa_stages")
              .at("frontend")
              .at("measured") == false &&
          document.at("ttfa_stages")
                  .at("ros_transport")
                  .at("measured") == false &&
          document.at("ttfa_stages")
                  .at("physical_output")
                  .at("measured") == false &&
          document.at("ttfa_stages")
                  .at("native_prepared_tokens")
                  .at("measured") == true,
      "TTFA scope must not invent frontend, ROS, or physical timing");
  require(
      document.at("component_latency")
              .at("request_start_call_ms")
              .at("p95")
              .get<double>() >= 0.0 &&
          document.at("component_latency")
                  .at("request_start_to_first_audio_ms")
                  .at("p95")
                  .get<double>() >= 0.0,
      "observable native component latency distributions");
  require(
      document.at("resource_scope")
              .at("process_isolated_vram")
              .at("measured") == false &&
          document.at("resource_scope")
                  .at("gpu_utilization")
                  .at("measured") == false,
      "unsupported in-process resource scopes remain unmeasured");
  require(
      document.at("verification").at("native_cleanup") ==
          "passed",
      "native cleanup verification");
  require(
      document.at("verification").at("cuda_cleanup") ==
          "passed",
      "CUDART cleanup verification");
  require(
      document.at("verification").at("nvml_cleanup") ==
          "passed",
      "NVML cleanup verification");

  const auto original = read_bounded_text_file(receipt);
  require_failure(
      [&]() {
        static_cast<void>(
            invoke_benchmark(bundle, corpus, receipt));
      },
      "an existing receipt must never be overwritten");
  require(
      read_bounded_text_file(receipt) == original,
      "immutable receipt bytes");
}

void test_missing_cudart_symbol_fails_closed(
    const TemporaryDirectory& temporary) {
  const auto bundle = temporary.path() / "broken-cuda-bundle";
  std::filesystem::create_directory(bundle);
  const auto corpus = temporary.path() / "broken-cuda-corpus.jsonl";
  const auto receipt =
      temporary.path() / "broken-cuda-receipt.json";
  write_text(corpus, corpus_bytes());
  require_failure(
      [&]() {
        static_cast<void>(invoke_benchmark(
            bundle, corpus, receipt,
            MTT_RUNTIME_BENCHMARK_MOCK_NVML));
      },
      "a CUDART library missing required symbols must fail closed");
  require(
      !std::filesystem::exists(receipt),
      "a pre-measurement CUDART failure must not publish a receipt");
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    if (argc == 2 &&
        std::string_view(argv[1]) == "--threshold-contract-only") {
      test_long_run_threshold_boundaries_fail_closed();
      std::cout << "runtime benchmark threshold tests passed\n";
      return 0;
    }
    if (argc != 1) {
      throw std::runtime_error("unexpected test argument");
    }
    TemporaryDirectory temporary;
    test_help();
    test_corpus_schema_rejects_unknown_members(temporary);
    test_acceptance_counts_fail_closed(temporary);
    test_long_run_threshold_boundaries_fail_closed();
    test_missing_cudart_symbol_fails_closed(temporary);
    test_complete_benchmark_and_immutable_receipt(temporary);
    std::cout << "runtime benchmark tool tests passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "runtime benchmark tool tests failed: "
              << exception.what() << '\n';
    return 1;
  }
}
