#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "manifest/manifest.hpp"
#include "runtime/cuda_memory.hpp"
#include "runtime/generation_diagnostics.hpp"
#include "runtime/pipeline_constants.hpp"

namespace magpie_tts_rt {

enum class SessionWorkspaceErrorCode {
  invalid_memory_limit,
  missing_engine,
  missing_tensor,
  incompatible_tensor_contract,
};

[[nodiscard]] std::string_view to_string(
    SessionWorkspaceErrorCode code) noexcept;

class SessionWorkspaceError final : public std::runtime_error {
 public:
  SessionWorkspaceError(
      SessionWorkspaceErrorCode code,
      std::string logical_name,
      std::string detail);

  [[nodiscard]] SessionWorkspaceErrorCode code() const noexcept;
  [[nodiscard]] const std::string& logical_name() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  SessionWorkspaceErrorCode code_;
  std::string logical_name_;
  std::string detail_;
};

struct DecoderLayerWorkspace {
  KvLayerBindings bindings;
  std::array<void*, 2> self_key;
  std::array<void*, 2> self_value;
  std::array<void*, 2> self_mask;
  void* cross_key;
  void* cross_value;
};

struct CodecStateWorkspace {
  CodecStateBinding binding;
  std::array<void*, 2> state;
};

// Per-session mutable device state. Every allocation is made once when a
// session is created, is bounded by the authenticated device-memory limit,
// and is reused only after the preceding request has reached a terminal state.
class SessionWorkspace final {
 public:
  SessionWorkspace(
      const RuntimeBundleManifest& manifest,
      std::uint64_t context_device_memory_bytes);
  ~SessionWorkspace();

  SessionWorkspace(const SessionWorkspace&) = delete;
  SessionWorkspace& operator=(const SessionWorkspace&) = delete;

  [[nodiscard]] void* text_token_ids() const noexcept;
  [[nodiscard]] void* text_mask() const noexcept;
  [[nodiscard]] void* text_condition() const noexcept;
  [[nodiscard]] void* cfg_condition() const noexcept;
  [[nodiscard]] void* condition_mask() const noexcept;
  [[nodiscard]] void* decoder_hidden() const noexcept;
  [[nodiscard]] std::int64_t* decoder_position() const noexcept;
  [[nodiscard]] std::int32_t* main_decoder_execution_status(
      std::size_t cache_index) const;
  [[nodiscard]] void* alignment_scores() const noexcept;
  [[nodiscard]] void* alignment_prior() const noexcept;
  [[nodiscard]] std::uint32_t* alignment_counters() const noexcept;
  [[nodiscard]] std::uint32_t* last_attended() const noexcept;
  [[nodiscard]] std::int64_t* attended_token_index(
      std::size_t step_slot) const;
  [[nodiscard]] bool* unfinished_text(std::size_t step_slot) const;
  [[nodiscard]] std::int32_t* alignment_invalid(
      std::size_t step_slot) const;
  [[nodiscard]] bool* generation_finished() const noexcept;
  [[nodiscard]] bool* forbid_eos() const noexcept;
  [[nodiscard]] std::int64_t* rng_seed() const noexcept;
  [[nodiscard]] std::int64_t* rng_counter(std::size_t index) const;
  [[nodiscard]] void* local_codec_tokens() const noexcept;
  [[nodiscard]] std::int32_t*
  canonical_local_invalid_rows() const noexcept;
  [[nodiscard]] std::int32_t*
  canonical_local_end_frame_index() const noexcept;
  [[nodiscard]] std::int32_t* local_invalid_rows(
      std::size_t step_slot) const;
  [[nodiscard]] std::int32_t* local_end_frame_index(
      std::size_t step_slot) const;
  [[nodiscard]] void* aggregate_codec_tokens(
      std::size_t batch_slot) const;
  [[nodiscard]] void* codec_input_tokens() const noexcept;
  [[nodiscard]] void* codec_pcm() const noexcept;
  [[nodiscard]] std::int64_t* codec_valid_sample_length() const noexcept;
  [[nodiscard]] float* pinned_pcm() const noexcept;
  [[nodiscard]] std::uint64_t pinned_pcm_capacity_samples() const noexcept;
  [[nodiscard]] std::int64_t* pinned_valid_sample_length() const noexcept;
  [[nodiscard]] GenerationBatchDiagnostics* generation_diagnostics(
      std::size_t batch_slot) const;
  [[nodiscard]] GenerationBatchDiagnostics* pinned_generation_diagnostics(
      std::size_t batch_slot) const;
  [[nodiscard]] std::int64_t* pinned_startup_codec_codes() const noexcept;
  [[nodiscard]] std::uint64_t
  pinned_startup_codec_code_capacity() const noexcept;

  [[nodiscard]] std::vector<DecoderLayerWorkspace>& decoder_layers() noexcept;
  [[nodiscard]] const std::vector<DecoderLayerWorkspace>&
  decoder_layers() const noexcept;
  [[nodiscard]] std::vector<CodecStateWorkspace>& codec_states() noexcept;
  [[nodiscard]] const std::vector<CodecStateWorkspace>&
  codec_states() const noexcept;

  [[nodiscard]] std::uint64_t allocated_device_memory_bytes() const noexcept;
  [[nodiscard]] std::uint64_t total_device_memory_bytes() const noexcept;

 private:
  DeviceMemoryRegistry memory_;
  std::uint64_t context_device_memory_bytes_;

  void* text_token_ids_{nullptr};
  void* text_mask_{nullptr};
  void* text_condition_{nullptr};
  void* cfg_condition_{nullptr};
  void* condition_mask_{nullptr};
  void* decoder_hidden_{nullptr};
  std::int64_t* decoder_position_{nullptr};
  std::array<std::int32_t*, 2> main_decoder_execution_status_{};
  void* alignment_scores_{nullptr};
  void* alignment_prior_{nullptr};
  std::uint32_t* alignment_counters_{nullptr};
  std::uint32_t* last_attended_{nullptr};
  std::array<std::int64_t*, kMaximumDecoderStepsPerEmission>
      attended_token_indices_{};
  std::array<bool*, kMaximumDecoderStepsPerEmission>
      unfinished_text_steps_{};
  std::array<std::int32_t*, kMaximumDecoderStepsPerEmission>
      alignment_invalid_steps_{};
  bool* generation_finished_{nullptr};
  bool* forbid_eos_{nullptr};
  std::int64_t* rng_seed_{nullptr};
  std::array<std::int64_t*, 2> rng_counter_{};
  void* local_codec_tokens_{nullptr};
  std::int32_t* canonical_local_invalid_rows_{nullptr};
  std::int32_t* canonical_local_end_frame_index_{nullptr};
  std::array<std::int32_t*, kMaximumDecoderStepsPerEmission>
      local_invalid_rows_steps_{};
  std::array<std::int32_t*, kMaximumDecoderStepsPerEmission>
      local_end_frame_indices_{};
  std::array<void*, kGenerationBatchSlotCount>
      aggregate_codec_tokens_{};
  void* codec_input_tokens_{nullptr};
  void* codec_pcm_{nullptr};
  std::int64_t* codec_valid_sample_length_{nullptr};
  std::unique_ptr<PinnedAllocation> pinned_pcm_;
  std::unique_ptr<PinnedAllocation> pinned_valid_sample_length_;
  std::array<GenerationBatchDiagnostics*, kGenerationBatchSlotCount>
      generation_diagnostics_{};
  std::array<
      std::unique_ptr<PinnedAllocation>,
      kGenerationBatchSlotCount>
      pinned_generation_diagnostics_{};
  std::unique_ptr<PinnedAllocation> pinned_startup_codec_codes_;
  std::uint64_t pinned_startup_codec_code_capacity_{0};
  std::uint64_t pinned_pcm_capacity_samples_{0};
  std::vector<DecoderLayerWorkspace> decoder_layers_;
  std::vector<CodecStateWorkspace> codec_states_;
};

}  // namespace magpie_tts_rt
