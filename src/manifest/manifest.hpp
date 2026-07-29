#pragma once

// Internal C++ contract model. The supported application boundary is the C ABI.

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace magpie_tts_rt {

inline constexpr std::uint32_t kRuntimeBundleManifestSchemaVersion = 1;
inline constexpr std::uint64_t kMaximumRuntimeBundleManifestBytes =
    16U * 1024U * 1024U;
inline constexpr std::uint64_t kMaximumBundleSnapshotBytes =
    16ULL * 1024ULL * 1024ULL * 1024ULL;

enum class ManifestStage {
  io,
  json,
  top_level,
  runtime_fingerprint,
  artifacts,
  classifier_free_guidance,
  engine,
  tensor,
  kv_cache,
  alignment,
  sampling,
  local_ar,
  codec,
  limits,
  golden_receipt,
  runtime_compatibility,
};

enum class ManifestErrorCode {
  io_error,
  size_limit_exceeded,
  json_syntax_error,
  missing_field,
  unknown_field,
  type_mismatch,
  invalid_value,
  duplicate_value,
  invariant_violation,
  fingerprint_mismatch,
};

[[nodiscard]] std::string_view to_string(ManifestStage stage) noexcept;
[[nodiscard]] std::string_view to_string(ManifestErrorCode code) noexcept;

class ManifestError final : public std::runtime_error {
 public:
  ManifestError(
      ManifestStage stage,
      ManifestErrorCode code,
      std::string json_pointer,
      std::string detail);

  [[nodiscard]] ManifestStage stage() const noexcept;
  [[nodiscard]] ManifestErrorCode code() const noexcept;
  [[nodiscard]] const std::string& json_pointer() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  ManifestStage stage_;
  ManifestErrorCode code_;
  std::string json_pointer_;
  std::string detail_;
};

enum class Endianness {
  little,
  big,
};

enum class TensorDataType {
  fp32,
  fp16,
  bf16,
  int64,
  int32,
  int8,
  uint8,
  boolean,
};

enum class EngineRole {
  text_encoder,
  main_decoder_prefill,
  main_decoder_step,
  local_ar_16,
  nanocodec_initial_4,
  nanocodec_steady_8,
  nanocodec_tail_1_8,
};

[[nodiscard]] std::string_view to_string(Endianness value) noexcept;
[[nodiscard]] std::string_view to_string(TensorDataType value) noexcept;
[[nodiscard]] std::string_view to_string(EngineRole value) noexcept;

struct RuntimeFingerprint {
  std::string os_name;
  std::string os_version;
  std::string architecture;
  Endianness endianness;
  std::string cuda_version;
  std::string tensorrt_version;
  std::string driver_version;
  std::string gpu_name;
  std::string gpu_compute_capability;
  std::uint32_t plugin_abi_version;
};

struct FileArtifact {
  std::filesystem::path path;
  std::string sha256;
  std::uint64_t size_bytes;
};

struct ModelArtifact {
  std::string model_id;
  std::string revision;
  FileArtifact file;
};

struct ExportArtifact {
  std::string format;
  std::string source_revision;
  std::string voice_id;
  std::uint32_t baked_context_length;
  std::string baked_context_sha256;
  bool audio_bos_baked;
  FileArtifact file;
};

struct TokenizerArtifact {
  std::string kind;
  std::uint32_t vocabulary_size;
  FileArtifact file;
};

struct PluginArtifact {
  std::string name;
  std::uint32_t abi_version;
  FileArtifact file;
};

struct ArtifactsManifest {
  ModelArtifact model;
  ExportArtifact export_artifact;
  TokenizerArtifact tokenizer;
  PluginArtifact plugin;
};

struct ClassifierFreeGuidanceManifest {
  std::string row_order;
  std::uint32_t conditional_row_index;
  std::string conditional_condition_source;
  std::string conditional_mask_source;
  std::uint32_t unconditional_row_index;
  std::string unconditional_condition_source;
  std::string unconditional_mask_source;
};

struct TensorSpec {
  std::string name;
  TensorDataType dtype;
  std::vector<std::int64_t> shape;
};

struct TensorShapeRange {
  std::string tensor_name;
  std::vector<std::int64_t> minimum;
  std::vector<std::int64_t> optimum;
  std::vector<std::int64_t> maximum;
};

struct OptimizationProfile {
  std::string name;
  std::vector<TensorShapeRange> input_shapes;
};

struct EngineManifest {
  std::string name;
  EngineRole role;
  FileArtifact file;
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
  std::vector<OptimizationProfile> profiles;
};

struct KvLayerBindings {
  std::uint32_t layer_index;
  std::string prefill_self_key_output;
  std::string prefill_self_value_output;
  std::string prefill_self_mask_output;
  std::string prefill_cross_key_output;
  std::string prefill_cross_value_output;
  std::string step_self_key_input;
  std::string step_self_value_input;
  std::string step_self_mask_input;
  std::string step_cross_key_input;
  std::string step_cross_value_input;
  std::string step_self_key_output;
  std::string step_self_value_output;
  std::string step_self_mask_output;
};

struct KvCacheManifest {
  std::string layout;
  TensorDataType dtype;
  TensorDataType mask_dtype;
  std::uint32_t layers;
  std::uint32_t batch_size;
  std::uint32_t self_attention_heads;
  std::uint32_t self_attention_head_dimension;
  std::uint32_t cross_attention_heads;
  std::uint32_t cross_attention_head_dimension;
  std::uint32_t prefix_length;
  std::uint32_t maximum_generated_steps;
  std::uint32_t self_cache_capacity;
  std::string update_mode;
  // main_decoder_step invocation n (zero based) writes to
  // first_step_position + n. Prefill emits the first generated step, so valid
  // step positions are [first_step_position,
  // step_position_upper_bound_exclusive).
  std::string position_semantics;
  std::uint32_t first_step_position;
  std::uint32_t step_position_upper_bound_exclusive;
  std::vector<KvLayerBindings> layer_bindings;
};

struct AlignmentManifest {
  TensorDataType dtype;
  std::vector<std::uint32_t> source_decoder_layers;
  std::string prefill_output_binding;
  std::string step_prior_input_binding;
  std::string step_alignment_output_binding;
  std::string source_position_policy;
};

struct RngManifest {
  std::string algorithm;
  std::uint32_t seed_bits;
  std::uint32_t counter_bits;
  std::string state_location;
  std::string ownership;
  bool deterministic;
};

struct SamplingManifest {
  std::string algorithm;
  std::uint32_t top_k;
  double temperature;
  std::uint32_t eos_token_id;
  std::vector<std::uint32_t> forbidden_token_ids;
  std::string invalid_distribution_policy;
  std::string next_embedding_location;
  RngManifest rng;
};

struct LocalArManifest {
  std::string engine_name;
  std::string execution;
  std::uint32_t iterations;
  std::vector<std::uint32_t> positions;
  std::uint32_t codebooks_per_frame;
  std::uint32_t frames_per_decoder_step;
  std::string sampling_plugin_name;
  std::string invalid_rows_encoding;
  std::int32_t no_eos_frame_index;
};

struct CodecStateBinding {
  std::string logical_name;
  TensorDataType dtype;
  std::vector<std::int64_t> shape;
  std::string initial_output_binding;
  std::string steady_input_binding;
  std::string steady_output_binding;
  std::string tail_input_binding;
  std::string tail_output_binding;
};

struct CodecManifest {
  std::string initial_engine_name;
  std::string steady_engine_name;
  std::string tail_engine_name;
  std::uint32_t sample_rate_hz;
  std::uint32_t hop_length_samples;
  std::uint32_t channels;
  std::string pcm_format;
  bool stateful;
  std::uint32_t initial_frames;
  std::uint32_t steady_frames;
  std::uint32_t tail_min_frames;
  std::uint32_t tail_max_frames;
  std::vector<CodecStateBinding> state_bindings;
};

struct LimitsManifest {
  std::uint32_t maximum_text_tokens;
  std::uint32_t maximum_decoder_steps;
  std::uint32_t maximum_audio_frames;
  std::uint32_t maximum_sessions;
  std::uint32_t maximum_concurrent_requests;
  std::uint32_t pcm_ring_capacity_frames;
  std::uint64_t maximum_workspace_bytes;
  std::uint64_t maximum_device_memory_bytes;
  std::uint64_t maximum_bundle_snapshot_bytes;
};

struct GoldenReceiptManifest {
  std::uint32_t receipt_version;
  std::filesystem::path path;
  std::string sha256;
  std::uint64_t size_bytes;
  std::string created_at_utc;
  std::string normalized_text_sha256;
  std::string token_ids_sha256;
  std::string baked_context_sha256;
  std::uint64_t seed;
  std::string decoder_tokens_sha256;
  std::string codec_codes_sha256;
  std::string pcm_f32le_sha256;
  std::uint64_t sample_count;
  std::uint32_t initial_frames;
  std::uint32_t steady_frames;
  std::uint32_t tail_min_frames;
  std::uint32_t tail_max_frames;
};

struct RuntimeBundleManifest {
  std::uint32_t schema_version;
  std::string bundle_id;
  std::string created_at_utc;
  RuntimeFingerprint runtime;
  ArtifactsManifest artifacts;
  ClassifierFreeGuidanceManifest classifier_free_guidance;
  std::vector<EngineManifest> engines;
  KvCacheManifest kv_cache;
  AlignmentManifest alignment;
  SamplingManifest sampling;
  LocalArManifest local_ar;
  CodecManifest codec;
  LimitsManifest limits;
  GoldenReceiptManifest golden_receipt;
};

// Parses and validates every field and cross-field invariant. Unknown and
// missing fields are errors at every object boundary.
[[nodiscard]] RuntimeBundleManifest parse_runtime_bundle_manifest(
    std::string_view json_text);

// Reads at most kMaximumRuntimeBundleManifestBytes and delegates to
// parse_runtime_bundle_manifest(). A larger file fails closed before parsing.
[[nodiscard]] RuntimeBundleManifest load_runtime_bundle_manifest(
    const std::filesystem::path& manifest_path);

// Requires every fingerprint field to match exactly. This is intentionally not
// a compatibility check and does not admit version or hardware fallbacks.
void require_exact_runtime_fingerprint(
    const RuntimeFingerprint& expected,
    const RuntimeFingerprint& actual);

[[nodiscard]] const EngineManifest& require_engine(
    const RuntimeBundleManifest& manifest,
    EngineRole role);

}  // namespace magpie_tts_rt
