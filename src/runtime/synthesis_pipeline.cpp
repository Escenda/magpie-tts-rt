#include "runtime/synthesis_pipeline.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "runtime/alignment_kernel.hpp"
#include "runtime/async_pipeline_contract.hpp"
#include "runtime/engine_execution.hpp"
#include "runtime/eos_contract.hpp"
#include "runtime/main_device_position_status.hpp"
#include "runtime/session_workspace.hpp"
#include "runtime/synthesis_kernels.hpp"

namespace magpie_tts_rt {
namespace {

[[nodiscard]] std::string error_message(
    const SynthesisPipelineErrorCode code,
    const std::string_view detail) {
  return "streaming synthesis failed [code=" +
         std::string(to_string(code)) + "]: " +
         std::string(detail);
}

[[noreturn]] void fail(
    const SynthesisPipelineErrorCode code,
    const std::string& detail) {
  throw SynthesisPipelineError(code, detail);
}

void require_cuda(
    const cudaError_t status,
    const std::string_view operation) {
  if (status != cudaSuccess) {
    fail(
        SynthesisPipelineErrorCode::cuda_failure,
        std::string(operation) + ": " +
            cudaGetErrorString(status));
  }
}

[[nodiscard]] TensorAddressSet text_bindings(
    SessionWorkspace& workspace) {
  TensorAddressSet bindings;
  bindings.add("text_token_ids", workspace.text_token_ids());
  bindings.add("text_mask", workspace.text_mask());
  bindings.add("text_condition", workspace.text_condition());
  return bindings;
}

[[nodiscard]] TensorAddressSet prefill_bindings(
    const RuntimeBundleManifest& manifest,
    SessionWorkspace& workspace) {
  TensorAddressSet bindings;
  bindings.add("condition", workspace.cfg_condition());
  bindings.add("condition_mask", workspace.condition_mask());
  bindings.add("last_hidden", workspace.decoder_hidden());
  bindings.add(
      manifest.alignment.prefill_output_binding,
      workspace.alignment_scores());
  for (DecoderLayerWorkspace& layer :
       workspace.decoder_layers()) {
    bindings.add(
        layer.bindings.prefill_self_key_output,
        layer.self_key[0]);
    bindings.add(
        layer.bindings.prefill_self_value_output,
        layer.self_value[0]);
    bindings.add(
        layer.bindings.prefill_self_mask_output,
        layer.self_mask[0]);
    bindings.add(
        layer.bindings.prefill_cross_key_output,
        layer.cross_key);
    bindings.add(
        layer.bindings.prefill_cross_value_output,
        layer.cross_value);
  }
  return bindings;
}

[[nodiscard]] TensorAddressSet step_bindings(
    const RuntimeBundleManifest& manifest,
    SessionWorkspace& workspace,
    const std::size_t cache_input,
    const std::size_t cache_output) {
  TensorAddressSet bindings;
  bindings.add(
      "previous_codec_tokens",
      workspace.local_codec_tokens());
  bindings.add("position", workspace.decoder_position());
  bindings.add(
      "execution_status_in",
      workspace.main_decoder_execution_status(cache_input));
  bindings.add(
      manifest.alignment.step_prior_input_binding,
      workspace.alignment_prior());
  bindings.add("condition_mask", workspace.condition_mask());
  for (DecoderLayerWorkspace& layer :
       workspace.decoder_layers()) {
    bindings.add(
        layer.bindings.step_self_key_input,
        layer.self_key.at(cache_input));
    bindings.add(
        layer.bindings.step_self_value_input,
        layer.self_value.at(cache_input));
    bindings.add(
        layer.bindings.step_self_mask_input,
        layer.self_mask.at(cache_input));
    bindings.add(
        layer.bindings.step_cross_key_input,
        layer.cross_key);
    bindings.add(
        layer.bindings.step_cross_value_input,
        layer.cross_value);
  }
  bindings.add("decoder_hidden", workspace.decoder_hidden());
  bindings.add(
      manifest.alignment.step_alignment_output_binding,
      workspace.alignment_scores());
  bindings.add(
      "execution_status_out",
      workspace.main_decoder_execution_status(cache_output));
  for (DecoderLayerWorkspace& layer :
       workspace.decoder_layers()) {
    bindings.add(
        layer.bindings.step_self_key_output,
        layer.self_key.at(cache_output));
    bindings.add(
        layer.bindings.step_self_value_output,
        layer.self_value.at(cache_output));
    bindings.add(
        layer.bindings.step_self_mask_output,
        layer.self_mask.at(cache_output));
  }
  return bindings;
}

[[nodiscard]] TensorAddressSet local_bindings(
    SessionWorkspace& workspace) {
  TensorAddressSet bindings;
  bindings.add("decoder_hidden", workspace.decoder_hidden());
  bindings.add(
      "unfinished", workspace.unfinished_text(0));
  bindings.add(
      "finished", workspace.generation_finished());
  bindings.add("forbid_eos", workspace.forbid_eos());
  bindings.add("rng_seed", workspace.rng_seed());
  bindings.add(
      "rng_counter", workspace.rng_counter(0));
  bindings.add(
      "codec_tokens", workspace.local_codec_tokens());
  bindings.add(
      "updated_rng_counter",
      workspace.rng_counter(1));
  bindings.add(
      "invalid_rows",
      workspace.canonical_local_invalid_rows());
  bindings.add(
      "end_frame_index",
      workspace.canonical_local_end_frame_index());
  return bindings;
}

[[nodiscard]] TensorAddressSet codec_initial_bindings(
    SessionWorkspace& workspace) {
  TensorAddressSet bindings;
  bindings.add(
      "codec_tokens", workspace.codec_input_tokens());
  bindings.add("pcm", workspace.codec_pcm());
  bindings.add(
      "valid_sample_length",
      workspace.codec_valid_sample_length());
  for (CodecStateWorkspace& state : workspace.codec_states()) {
    bindings.add(
        state.binding.initial_output_binding, state.state[0]);
  }
  return bindings;
}

[[nodiscard]] TensorAddressSet codec_stateful_bindings(
    SessionWorkspace& workspace,
    const bool tail,
    const std::size_t state_input,
    const std::size_t state_output) {
  TensorAddressSet bindings;
  bindings.add(
      "codec_tokens", workspace.codec_input_tokens());
  for (CodecStateWorkspace& state : workspace.codec_states()) {
    bindings.add(
        tail ? state.binding.tail_input_binding
             : state.binding.steady_input_binding,
        state.state.at(state_input));
  }
  bindings.add("pcm", workspace.codec_pcm());
  bindings.add(
      "valid_sample_length",
      workspace.codec_valid_sample_length());
  for (CodecStateWorkspace& state : workspace.codec_states()) {
    bindings.add(
        tail ? state.binding.tail_output_binding
             : state.binding.steady_output_binding,
        state.state.at(state_output));
  }
  return bindings;
}

struct GenerationBatchResult {
  std::size_t batch_slot;
  std::uint32_t frame_count;
  bool final;
  // Both indices are local to this generation batch; -1 means no EOS.
  std::int32_t eos_step;
  std::int32_t eos_frame_index;
  std::vector<std::int64_t> attended_token_indices;
};

struct PendingGenerationBatch {
  std::size_t batch_slot;
  std::size_t step_count;
  std::uint64_t decoder_steps_after_batch;
};

struct PendingCodecBatch {
  GenerationBatchResult generation;
  std::uint64_t expected_samples;
  std::size_t startup_code_count;
  bool first;
};

class Pipeline final {
 public:
  Pipeline(
      const RuntimeBundleManifest& manifest,
      SessionResources& resources,
      const std::vector<std::int32_t>& text_token_ids,
      const std::uint32_t random_seed,
      StreamingRequestState& request_state,
      StartupGoldenCapture* startup_capture)
      : manifest_(manifest),
        resources_(resources),
        workspace_(resources.workspace()),
        text_token_ids_(text_token_ids),
        random_seed_(random_seed),
        request_state_(request_state),
        startup_capture_(startup_capture),
        shape_parameters_{
            .text_token_count =
                static_cast<std::uint32_t>(text_token_ids.size()),
            .codec_frame_count = 0,
            .codec_hop_length_samples =
                manifest.codec.hop_length_samples,
        },
        alignment_contract_{
            .text_length =
                static_cast<std::uint32_t>(text_token_ids.size()),
            .ignored_terminal_tokens =
                manifest.alignment.ignored_terminal_tokens,
            .short_text_no_prior_max_tokens =
                manifest.alignment.short_text_no_prior_max_tokens,
            .lookahead = manifest.alignment.lookahead,
            .sink_threshold = manifest.alignment.sink_threshold,
        } {}

  [[nodiscard]] PipelineSynchronizationMetrics run() {
    try {
      run_impl();
      settle_inflight();
    } catch (...) {
      try {
        settle_inflight();
      } catch (...) {
        resources_.abort_main_decoder_request();
        throw;
      }
      resources_.abort_main_decoder_request();
      throw;
    }
    if (request_state_.cancellation_requested() &&
        !request_state_.complete_cancellation_after_drain()) {
      fail(
          SynthesisPipelineErrorCode::invariant_failure,
          "accepted cancellation disappeared after CUDA drain");
    }
    return overlap_contract_.metrics();
  }

 private:
  void run_impl() {
    resources_.begin_main_decoder_request();
    require_initial_batch_capacity();
    initialize_request();
    run_text_encoder();
    run_prefill();

    PendingGenerationBatch pending =
        generate_first_four(0);
    GenerationBatchResult first =
        await_generation_batch(pending);
    if (first.final || first.frame_count != manifest_.codec.initial_frames) {
      fail(
          SynthesisPipelineErrorCode::invariant_failure,
          "EOS was admitted before the fixed initial four frames");
    }
    if (observe_cancellation()) {
      return;
    }
    PendingCodecBatch first_codec =
        schedule_codec(std::move(first), true);
    if (!await_codec_and_publish(std::move(first_codec))) {
      return;
    }
    if (observe_cancellation()) {
      return;
    }
    require_steady_batch_capacity();
    pending = generate_steady_batch(1);

    while (true) {
      if (!wait_for_output_capacity()) {
        return;
      }
      GenerationBatchResult batch =
          await_generation_batch(pending);
      if (observe_cancellation()) {
        return;
      }
      if (batch.frame_count == 0) {
        if (!publish_zero_frame_final(batch)) {
          return;
        }
        return;
      }
      const bool final = batch.final;
      if (!final) {
        require_steady_batch_capacity();
      }
      PendingCodecBatch codec =
          schedule_codec(std::move(batch), false);
      std::optional<PendingGenerationBatch> following;
      if (!final) {
        const std::size_t next_slot =
            1U - codec.generation.batch_slot;
        following.emplace(
            generate_steady_batch(next_slot));
        overlap_contract_.record_overlap_opportunity();
      }
      if (!await_codec_and_publish(std::move(codec))) {
        return;
      }
      if (final) {
        return;
      }
      if (observe_cancellation()) {
        return;
      }
      pending = *following;
    }
  }
  void initialize_request() {
    const cudaStream_t stream = resources_.generation_stream();
    static_assert(sizeof(bool) == sizeof(std::uint8_t));
    require_cuda(
        cudaMemcpyAsync(
            workspace_.text_token_ids(),
            text_token_ids_.data(),
            text_token_ids_.size() * sizeof(std::int32_t),
            cudaMemcpyHostToDevice,
            stream),
        "copy text token identifiers");
    require_cuda(
        cudaMemsetAsync(
            workspace_.text_mask(),
            1,
            text_token_ids_.size() * sizeof(bool),
            stream),
        "initialize text mask");
    require_cuda(
        cudaMemsetAsync(
            workspace_.alignment_counters(),
            0,
            text_token_ids_.size() * sizeof(std::uint32_t),
            stream),
        "clear alignment counters");
    const std::uint32_t initial_attended =
        manifest_.alignment.initial_attended;
    const std::int64_t seed =
        static_cast<std::int64_t>(random_seed_);
    const std::int64_t zero_counter = 0;
    if (manifest_.kv_cache.first_step_position == 0U) {
      fail(
          SynthesisPipelineErrorCode::invariant_failure,
          "Main Decoder first position must be greater than zero");
    }
    const std::int64_t position_before_first_step =
        static_cast<std::int64_t>(
            manifest_.kv_cache.first_step_position) -
        1;
    const bool false_value = false;
    require_cuda(
        cudaMemcpyAsync(
            workspace_.last_attended(),
            &initial_attended,
            sizeof(initial_attended),
            cudaMemcpyHostToDevice,
            stream),
        "initialize alignment position");
    require_cuda(
        cudaMemcpyAsync(
            workspace_.rng_seed(),
            &seed,
            sizeof(seed),
            cudaMemcpyHostToDevice,
            stream),
        "initialize random seed");
    require_cuda(
        cudaMemcpyAsync(
            workspace_.rng_counter(0),
            &zero_counter,
            sizeof(zero_counter),
            cudaMemcpyHostToDevice,
            stream),
        "initialize random counter");
    require_cuda(
        cudaMemcpyAsync(
            workspace_.decoder_position(),
            &position_before_first_step,
            sizeof(position_before_first_step),
            cudaMemcpyHostToDevice,
            stream),
        "initialize Main Decoder device position");
    for (std::size_t index = 0; index < 2; ++index) {
      require_cuda(
          cudaMemsetAsync(
              workspace_.main_decoder_execution_status(index),
              0,
              sizeof(std::int32_t),
              stream),
          "clear Main Decoder execution status");
    }
    require_cuda(
        cudaMemcpyAsync(
            workspace_.generation_finished(),
            &false_value,
            sizeof(false_value),
            cudaMemcpyHostToDevice,
            stream),
        "clear generation-finished latch");
  }

  void run_text_encoder() {
    enqueue_engine(
        require_engine(manifest_, EngineRole::text_encoder),
        resources_.context(EngineRole::text_encoder),
        shape_parameters_,
        text_bindings(workspace_),
        resources_.generation_stream(),
        nullptr);
    require_cuda(
        launch_prepare_cfg_inputs(
            workspace_.text_condition(),
            static_cast<const bool*>(workspace_.text_mask()),
            shape_parameters_.text_token_count,
            workspace_.cfg_condition(),
            static_cast<bool*>(workspace_.condition_mask()),
            resources_.generation_stream()),
        "prepare classifier-free guidance inputs");
  }

  void run_prefill() {
    enqueue_engine(
        require_engine(
            manifest_, EngineRole::main_decoder_prefill),
        resources_.context(
            EngineRole::main_decoder_prefill),
        shape_parameters_,
        prefill_bindings(manifest_, workspace_),
        resources_.generation_stream(),
        nullptr);
    cache_index_ = 0;
  }

  void advance_alignment(const std::size_t step_slot) {
    require_cuda(
        launch_alignment_controller(
            workspace_.alignment_scores(),
            alignment_contract_,
            workspace_.alignment_counters(),
            workspace_.last_attended(),
            workspace_.alignment_prior(),
            workspace_.attended_token_index(step_slot),
            workspace_.unfinished_text(step_slot),
            workspace_.alignment_invalid(step_slot),
            resources_.generation_stream()),
        "advance alignment controller");
  }

  void run_local(
      const std::size_t step_slot,
      const std::size_t batch_slot,
      const std::uint32_t aggregate_frame_offset) {
    if (step_slot != 0) {
      require_cuda(
          cudaMemcpyAsync(
              workspace_.unfinished_text(0),
              workspace_.unfinished_text(step_slot),
              sizeof(bool),
              cudaMemcpyDeviceToDevice,
              resources_.generation_stream()),
          "stage Local AR unfinished input");
    }
    const bool forbid_eos = generated_decoder_steps_ < 2;
    require_cuda(
        cudaMemcpyAsync(
            workspace_.forbid_eos(),
            &forbid_eos,
            sizeof(forbid_eos),
            cudaMemcpyHostToDevice,
            resources_.generation_stream()),
        "update EOS gate");
    if (!resources_.local_ar_graph_ready()) {
      if (startup_capture_ == nullptr) {
        fail(
            SynthesisPipelineErrorCode::invariant_failure,
            "Local AR CUDA graph was absent outside the startup gate");
      }
      // The first valid Local invocation warms TensorRT's deferred state. Its
      // output is deliberately discarded: rng_counter[0] is input-only, and
      // the graph-backed invocation below overwrites every canonical output.
      enqueue_engine(
          require_engine(manifest_, EngineRole::local_ar_16),
          resources_.context(EngineRole::local_ar_16),
          shape_parameters_,
          local_bindings(workspace_),
          resources_.generation_stream(),
          nullptr);
      resources_.capture_and_upload_local_ar_graph();
    }
    resources_.launch_local_ar_graph();
    require_cuda(
        launch_finalize_local_step(
            static_cast<const std::int64_t*>(
                workspace_.local_codec_tokens()),
            static_cast<std::int64_t*>(
                workspace_.aggregate_codec_tokens(batch_slot)),
            aggregate_frame_offset,
            workspace_.canonical_local_invalid_rows(),
            workspace_.canonical_local_end_frame_index(),
            workspace_.local_invalid_rows(step_slot),
            workspace_.local_end_frame_index(step_slot),
            workspace_.rng_counter(1),
            workspace_.rng_counter(0),
            workspace_.generation_finished(),
            resources_.generation_stream()),
        "finalize Local AR step");
    ++generated_decoder_steps_;
  }

  void run_decoder_step(const std::size_t step_slot) {
    if (generated_decoder_steps_ == 0 ||
        generated_decoder_steps_ >=
            manifest_.limits.maximum_decoder_steps) {
      fail(
          SynthesisPipelineErrorCode::context_exhausted,
          "Main Decoder step would exceed the authenticated context");
    }
    const std::uint64_t expected_position =
        static_cast<std::uint64_t>(
            manifest_.kv_cache.first_step_position) +
        generated_decoder_steps_ - 1U;
    if (expected_position >=
        manifest_.kv_cache.step_position_upper_bound_exclusive) {
      fail(
          SynthesisPipelineErrorCode::context_exhausted,
          "absolute Main Decoder position reached its upper bound");
    }
    const std::size_t next_cache = 1U - cache_index_;
    const EngineManifest& step_engine =
        require_engine(manifest_, EngineRole::main_decoder_step);
    std::optional<PreparedTensorAddressSet>& prepared_bindings =
        step_binding_cache_.at(cache_index_);
    if (!prepared_bindings.has_value()) {
      prepared_bindings.emplace(
          step_engine,
          step_bindings(
              manifest_,
              workspace_,
              cache_index_,
              next_cache));
    }
    nvinfer1::IExecutionContext& decoder_context =
        resources_.main_decoder_context(cache_index_);
    if (!resources_.main_decoder_warmed(cache_index_)) {
      // Exactly one eager invocation per direction flushes TensorRT's deferred
      // shape/context setup. Its output is the production result: A-to-B
      // produces the second pair of initial frames, while the next B-to-A
      // invocation occurs only after first audio has been published.
      require_cuda(
          launch_advance_decoder_position(
              workspace_.decoder_position(),
              resources_.generation_stream()),
          "advance Main Decoder device position for eager warmup");
      enqueue_engine(
          step_engine,
          decoder_context,
          shape_parameters_,
          *prepared_bindings,
          resources_.generation_stream(),
          nullptr);
      resources_.record_main_decoder_eager_warmup(cache_index_);
    } else if (!resources_.main_decoder_graph_ready(cache_index_)) {
      // Dynamic text length makes these graphs request-specific. The capture
      // includes the device-side +1 before enqueueV3, then the graph is
      // launched once and that result is admitted to the current request.
      std::int64_t* const decoder_position =
          workspace_.decoder_position();
      resources_.capture_and_upload_main_decoder_graph(
          cache_index_,
          [&decoder_context, decoder_position](
              const cudaStream_t stream) noexcept {
            if (launch_advance_decoder_position(
                    decoder_position, stream) != cudaSuccess) {
              return false;
            }
            return decoder_context.enqueueV3(stream);
          });
      resources_.launch_main_decoder_graph(cache_index_);
    } else {
      // No eager enqueue is admitted after capture. Missing or failed graphs
      // throw from SessionResources/CudaGraphExecutable and close the request.
      resources_.launch_main_decoder_graph(cache_index_);
    }
    cache_index_ = next_cache;
    advance_alignment(step_slot);
  }

  void require_steady_batch_capacity() const {
    const std::uint64_t steps =
        kMaximumDecoderStepsPerEmission;
    if (generated_decoder_steps_ == 0 ||
        generated_decoder_steps_ + steps >
            manifest_.limits.maximum_decoder_steps) {
      fail(
          SynthesisPipelineErrorCode::context_exhausted,
          "the next fixed four-step generation batch would exceed "
          "the authenticated decoder context");
    }
    const std::uint64_t final_position =
        static_cast<std::uint64_t>(
            manifest_.kv_cache.first_step_position) +
        generated_decoder_steps_ + steps - 2U;
    if (final_position >=
        manifest_.kv_cache.step_position_upper_bound_exclusive) {
      fail(
          SynthesisPipelineErrorCode::context_exhausted,
          "the next fixed four-step generation batch would exceed "
          "the authenticated position range");
    }
  }

  void prepare_generation_slot(const std::size_t batch_slot) {
    if (slot_requires_reuse_wait_.at(batch_slot)) {
      require_cuda(
          cudaStreamWaitEvent(
              resources_.generation_stream(),
              resources_.codes_consumed_event(batch_slot),
              0),
          "wait for codec to consume generation slot");
      overlap_contract_.queue_reuse_dependency(batch_slot);
      slot_requires_reuse_wait_.at(batch_slot) = false;
    }
    overlap_contract_.schedule_generation(batch_slot);
  }

  void require_initial_batch_capacity() const {
    if (manifest_.limits.maximum_decoder_steps < 2U) {
      fail(
          SynthesisPipelineErrorCode::context_exhausted,
          "the fixed first four codec frames require two decoder steps");
    }
    if (static_cast<std::uint64_t>(
            manifest_.kv_cache.first_step_position) >=
        manifest_.kv_cache.step_position_upper_bound_exclusive) {
      fail(
          SynthesisPipelineErrorCode::context_exhausted,
          "the first Main Decoder step is outside the authenticated "
          "position range");
    }
  }

  [[nodiscard]] PendingGenerationBatch generate_first_four(
      const std::size_t batch_slot) {
    prepare_generation_slot(batch_slot);
    advance_alignment(0);
    run_local(0, batch_slot, 0);
    run_decoder_step(1);
    run_local(1, batch_slot, 2);
    return finish_generation_batch(batch_slot, 2);
  }

  [[nodiscard]] PendingGenerationBatch generate_steady_batch(
      const std::size_t batch_slot) {
    require_steady_batch_capacity();
    prepare_generation_slot(batch_slot);
    for (std::size_t step_slot = 0;
         step_slot < kMaximumDecoderStepsPerEmission;
         ++step_slot) {
      run_decoder_step(step_slot);
      run_local(
          step_slot,
          batch_slot,
          static_cast<std::uint32_t>(
              step_slot *
              manifest_.local_ar.frames_per_decoder_step));
    }
    return finish_generation_batch(
        batch_slot, kMaximumDecoderStepsPerEmission);
  }

  [[nodiscard]] PendingGenerationBatch finish_generation_batch(
      const std::size_t batch_slot,
      const std::size_t step_count) {
    const cudaStream_t stream = resources_.generation_stream();
    GenerationDiagnosticSources sources{};
    sources.main_decoder_execution_status =
        workspace_.main_decoder_execution_status(cache_index_);
    for (std::size_t step = 0;
         step < kMaximumDecoderStepsPerEmission;
         ++step) {
      sources.attended_token_indices[step] =
          workspace_.attended_token_index(step);
      sources.alignment_invalid_steps[step] =
          workspace_.alignment_invalid(step);
      sources.local_invalid_rows[step] =
          workspace_.local_invalid_rows(step);
      sources.end_frame_indices[step] =
          workspace_.local_end_frame_index(step);
    }
    require_cuda(
        launch_pack_generation_diagnostics(
            sources,
            static_cast<std::uint32_t>(step_count),
            workspace_.generation_diagnostics(batch_slot),
            stream),
        "pack generation diagnostics");
    require_cuda(
        cudaMemcpyAsync(
            workspace_.pinned_generation_diagnostics(batch_slot),
            workspace_.generation_diagnostics(batch_slot),
            sizeof(GenerationBatchDiagnostics),
            cudaMemcpyDeviceToHost,
            stream),
        "copy generation diagnostics");
    require_cuda(
        cudaEventRecord(
            resources_.codes_ready_event(batch_slot), stream),
        "record codes-ready event");
    generation_pending_.at(batch_slot) = true;
    return PendingGenerationBatch{
        .batch_slot = batch_slot,
        .step_count = step_count,
        .decoder_steps_after_batch = generated_decoder_steps_,
    };
  }

  [[nodiscard]] GenerationBatchResult await_generation_batch(
      const PendingGenerationBatch& pending) {
    await_event(
        resources_.codes_ready_event(pending.batch_slot),
        HostWaitBoundary::terminal_diagnostics,
        "wait for generation diagnostics");
    generation_pending_.at(pending.batch_slot) = false;
    overlap_contract_.mark_generation_ready(
        pending.batch_slot);
    const GenerationBatchDiagnostics& diagnostics =
        *workspace_.pinned_generation_diagnostics(
            pending.batch_slot);
    const GenerationDiagnosticPayloadValidation validation =
        validate_generation_diagnostic_payload(
            diagnostics, pending.step_count);
    if (!validation.valid()) {
      fail(
          SynthesisPipelineErrorCode::invariant_failure,
          "packed generation diagnostics failed validation at step " +
              std::to_string(validation.step) +
              " with code " +
              std::to_string(static_cast<std::uint32_t>(validation.code)));
    }
    const std::int32_t main_decoder_status =
        diagnostics.main_decoder_execution_status;
    if (main_decoder_status != 0) {
      fail(
          SynthesisPipelineErrorCode::main_decoder_failure,
          describe_main_device_position_status(main_decoder_status));
    }
    std::vector<std::int64_t> attended;
    attended.reserve(pending.step_count);
    std::uint32_t emitted_frames = static_cast<std::uint32_t>(
        pending.step_count *
        manifest_.local_ar.frames_per_decoder_step);
    bool final = false;
    std::int32_t eos_step = -1;
    std::int32_t eos_frame_index = -1;
    for (std::size_t step = 0;
         step < pending.step_count;
         ++step) {
      const GenerationStepDiagnostics& step_diagnostics =
          diagnostics.steps[step];
      if (step_diagnostics.alignment_invalid != 0) {
        fail(
            SynthesisPipelineErrorCode::alignment_failure,
            "alignment controller rejected step " +
                std::to_string(step));
      }
      if (step_diagnostics.local_invalid_rows != 0) {
        fail(
            SynthesisPipelineErrorCode::local_ar_failure,
            "Local AR rejected step " + std::to_string(step));
      }
      const std::int64_t attended_index =
          step_diagnostics.attended_token_index;
      if (attended_index < 0 ||
          attended_index >=
              static_cast<std::int64_t>(
                  text_token_ids_.size())) {
        fail(
            SynthesisPipelineErrorCode::alignment_failure,
            "alignment produced an out-of-range token index");
      }
      const std::int32_t eos =
          step_diagnostics.end_frame_index;
      if (eos != -1 && eos != 0 && eos != 1) {
        fail(
            SynthesisPipelineErrorCode::local_ar_failure,
            "Local AR produced an invalid EOS frame index");
      }
      // All scheduled steps are checked above, including those after an
      // earlier EOS, but only the spoken prefix belongs to the chunk. The
      // previous implementation resized here and then appended later steps
      // again, which could make a three-frame terminal batch carry four
      // attended positions and create a lease-external alignment boundary.
      if (final) {
        continue;
      }
      attended.push_back(attended_index);
      if (eos >= 0) {
        emitted_frames =
            retained_codec_frames_before_eos(step, eos);
        final = true;
        eos_step = static_cast<std::int32_t>(step);
        eos_frame_index = eos;
      }
    }
    if (!final &&
        pending.decoder_steps_after_batch >=
            manifest_.limits.maximum_decoder_steps) {
      fail(
          SynthesisPipelineErrorCode::context_exhausted,
          "generation reached the maximum decoder steps without EOS");
    }
    return GenerationBatchResult{
        .batch_slot = pending.batch_slot,
        .frame_count = emitted_frames,
        .final = final,
        .eos_step = eos_step,
        .eos_frame_index = eos_frame_index,
        .attended_token_indices = std::move(attended),
    };
  }

  [[noreturn]] void fail_audio_chunk_publication(
      const RequestStateError& error,
      const AudioChunk& chunk,
      const GenerationBatchResult& batch) const {
    fail(
        SynthesisPipelineErrorCode::invariant_failure,
        describe_audio_chunk_validation_failure(
            error,
            chunk,
            AudioChunkOrigin{
                .random_seed = random_seed_,
                .eos_step = batch.eos_step,
                .eos_frame_index = batch.eos_frame_index,
                .attended_token_indices =
                    batch.attended_token_indices,
            }));
  }

  [[nodiscard]] bool publish_audio_chunk(
      AudioChunk chunk,
      const GenerationBatchResult& batch) {
    try {
      return request_state_.publish(std::move(chunk));
    } catch (const RequestStateError& error) {
      // publish() accepts an rvalue reference and performs validation before
      // moving into the queue, so the rejected chunk is intact here.
      fail_audio_chunk_publication(error, chunk, batch);
    }
  }

  void await_event(
      const cudaEvent_t event,
      const HostWaitBoundary boundary,
      const std::string_view operation) {
    overlap_contract_.record_host_event_synchronization(boundary);
    require_cuda(cudaEventSynchronize(event), operation);
  }

  void settle_inflight() {
    for (std::size_t slot = 0;
         slot < generation_pending_.size();
         ++slot) {
      if (!generation_pending_[slot]) {
        continue;
      }
      await_event(
          resources_.codes_ready_event(slot),
          HostWaitBoundary::terminal_diagnostics,
          "settle pending generation batch");
      generation_pending_[slot] = false;
      overlap_contract_.mark_generation_ready(slot);
    }
    if (audio_pending_) {
      await_event(
          resources_.audio_ready_event(),
          HostWaitBoundary::pcm_publication,
          "settle pending codec batch");
      audio_pending_ = false;
    }
  }

  [[nodiscard]] bool publish_zero_frame_final(
      const GenerationBatchResult& batch) {
    if (!batch.final || batch.frame_count != 0) {
      fail(
          SynthesisPipelineErrorCode::invariant_failure,
          "zero-frame publication requires a terminal generation batch");
    }
    if (observe_cancellation()) {
      return false;
    }
    if (!publish_audio_chunk(AudioChunk{
            .samples = AudioBuffer{},
            .first_sample_index = next_sample_index_,
            .sequence = sequence_,
            .codec_frame_count = 0,
            .first = false,
            .final = true,
            .alignment_valid = true,
            .committed_text_tokens =
                committed_text_tokens_,
            .alignment_events = {},
        }, batch)) {
      return false;
    }
    ++sequence_;
    return true;
  }

  [[nodiscard]] PendingCodecBatch schedule_codec(
      GenerationBatchResult batch,
      const bool first) {
    if (batch.frame_count >
            manifest_.codec.steady_frames ||
        (batch.frame_count == 0 &&
         (!batch.final || first))) {
      fail(
          SynthesisPipelineErrorCode::codec_failure,
          "zero codec frames are valid only for a non-FIRST FINAL "
          "control marker; decoded chunks must contain 1-8 frames");
    }
    if (batch.frame_count == 0) {
      fail(
          SynthesisPipelineErrorCode::invariant_failure,
          "zero-frame terminal markers do not enter NanoCodec");
    }
    if (audio_pending_) {
      fail(
          SynthesisPipelineErrorCode::invariant_failure,
          "the single PCM slot was reused before publication");
    }
    const cudaStream_t codec_stream = resources_.codec_stream();
    require_cuda(
        launch_pack_codec_frames(
            static_cast<const std::int64_t*>(
                workspace_.aggregate_codec_tokens(
                    batch.batch_slot)),
            static_cast<std::int64_t*>(
                workspace_.codec_input_tokens()),
            batch.frame_count,
            codec_stream),
        "pack NanoCodec input");
    require_cuda(
        cudaEventRecord(
            resources_.codes_consumed_event(batch.batch_slot),
            codec_stream),
        "record generation-slot consumption");
    overlap_contract_.schedule_codec(batch.batch_slot);
    slot_requires_reuse_wait_.at(batch.batch_slot) = true;

    shape_parameters_.codec_frame_count = batch.frame_count;
    if (first) {
      if (!resources_.nanocodec_initial_graph_ready()) {
        if (startup_capture_ == nullptr) {
          fail(
              SynthesisPipelineErrorCode::invariant_failure,
              "NanoCodec initial CUDA graph was absent outside the "
              "startup gate");
        }
        // Bind the immutable production addresses and flush TensorRT deferred
        // setup once. The eager state/PCM is discarded; the graph replay
        // below overwrites every output admitted by the startup golden.
        enqueue_engine(
            require_engine(
                manifest_, EngineRole::nanocodec_initial_4),
            resources_.context(
                EngineRole::nanocodec_initial_4),
            shape_parameters_,
            codec_initial_bindings(workspace_),
            codec_stream,
            nullptr);
        resources_.capture_and_upload_nanocodec_initial_graph();
      }
      resources_.launch_nanocodec_initial_graph();
      codec_state_index_ = 0;
    } else {
      const bool tail = batch.final;
      const EngineRole role =
          tail ? EngineRole::nanocodec_tail_1_8
               : EngineRole::nanocodec_steady_8;
      if (!tail &&
          batch.frame_count != manifest_.codec.steady_frames) {
        fail(
            SynthesisPipelineErrorCode::codec_failure,
            "a non-terminal codec call is not the fixed steady route");
      }
      const std::size_t next_state =
          1U - codec_state_index_;
      if (tail) {
        // Tail retains its authenticated 1..8 dynamic-shape enqueue. It is
        // never used as a fallback for a missing fixed steady graph.
        enqueue_engine(
            require_engine(manifest_, role),
            resources_.context(role),
            shape_parameters_,
            codec_stateful_bindings(
                workspace_,
                true,
                codec_state_index_,
                next_state),
            codec_stream,
            nullptr);
      } else {
        nvinfer1::IExecutionContext& steady_context =
            resources_.codec_steady_context(codec_state_index_);
        if (!resources_.nanocodec_steady_graph_ready(
                codec_state_index_)) {
          if (startup_capture_ == nullptr) {
            fail(
                SynthesisPipelineErrorCode::invariant_failure,
                "NanoCodec steady CUDA graph was absent outside the "
                "startup gate");
          }
          // Each recurrent direction binds a different context and fixed
          // A-to-B/B-to-A state addresses. The eager output is discarded and
          // overwritten by the production graph replay below.
          enqueue_engine(
              require_engine(manifest_, role),
              steady_context,
              shape_parameters_,
              codec_stateful_bindings(
                  workspace_,
                  false,
                  codec_state_index_,
                  next_state),
              codec_stream,
              nullptr);
          resources_.capture_and_upload_nanocodec_steady_graph(
              codec_state_index_);
        }
        resources_.launch_nanocodec_steady_graph(
            codec_state_index_);
      }
      codec_state_index_ = next_state;
    }

    const std::uint64_t expected_samples =
        static_cast<std::uint64_t>(batch.frame_count) *
        manifest_.codec.hop_length_samples;
    if (expected_samples >
        workspace_.pinned_pcm_capacity_samples()) {
      fail(
          SynthesisPipelineErrorCode::codec_failure,
          "PCM output exceeds the pinned emission slot");
    }
    require_cuda(
        cudaMemcpyAsync(
            workspace_.pinned_valid_sample_length(),
            workspace_.codec_valid_sample_length(),
            sizeof(std::int64_t),
            cudaMemcpyDeviceToHost,
            codec_stream),
        "copy NanoCodec valid sample length");
    require_cuda(
        cudaMemcpyAsync(
            workspace_.pinned_pcm(),
            workspace_.codec_pcm(),
            expected_samples * sizeof(float),
            cudaMemcpyDeviceToHost,
            codec_stream),
        "copy NanoCodec PCM");
    std::size_t startup_code_count = 0;
    if (startup_capture_ != nullptr) {
      startup_code_count =
          static_cast<std::size_t>(
              manifest_.local_ar.codebooks_per_frame) *
          batch.frame_count;
      if (startup_code_count >
          workspace_.pinned_startup_codec_code_capacity()) {
        fail(
            SynthesisPipelineErrorCode::invariant_failure,
            "startup codec capture exceeds its pinned slot");
      }
      require_cuda(
          cudaMemcpyAsync(
              workspace_.pinned_startup_codec_codes(),
              workspace_.codec_input_tokens(),
              startup_code_count * sizeof(std::int64_t),
              cudaMemcpyDeviceToHost,
              codec_stream),
          "copy startup golden codec codes");
    }
    require_cuda(
        cudaEventRecord(
            resources_.audio_ready_event(), codec_stream),
        "record audio-ready event");
    audio_pending_ = true;
    return PendingCodecBatch{
        .generation = std::move(batch),
        .expected_samples = expected_samples,
        .startup_code_count = startup_code_count,
        .first = first,
    };
  }

  [[nodiscard]] bool await_codec_and_publish(
      PendingCodecBatch pending) {
    await_event(
        resources_.audio_ready_event(),
        HostWaitBoundary::pcm_publication,
        "wait for codec output");
    audio_pending_ = false;
    const GenerationBatchResult& batch = pending.generation;
    const std::uint64_t expected_samples =
        pending.expected_samples;
    if (*workspace_.pinned_valid_sample_length() !=
        static_cast<std::int64_t>(expected_samples)) {
      fail(
          SynthesisPipelineErrorCode::codec_failure,
          "NanoCodec valid sample length differs from frame contract");
    }
    const std::span<const float> pcm(
        workspace_.pinned_pcm(),
        static_cast<std::size_t>(expected_samples));
    if (std::any_of(
            pcm.begin(), pcm.end(), [](const float sample) {
              return !std::isfinite(sample);
            })) {
      fail(
          SynthesisPipelineErrorCode::codec_failure,
          "NanoCodec produced non-finite PCM");
    }
    if (startup_capture_ != nullptr) {
      startup_capture_->record_chunk(
          std::span<const std::int64_t>(
              workspace_.pinned_startup_codec_codes(),
              pending.startup_code_count),
          batch.frame_count,
          pcm);
    }
    if (observe_cancellation()) {
      return false;
    }

    std::vector<AlignmentProgress> events;
    events.reserve(batch.attended_token_indices.size());
    for (std::size_t step = 0;
         step < batch.attended_token_indices.size();
         ++step) {
      const bool terminal_step =
          batch.final &&
          step + 1U ==
              batch.attended_token_indices.size();
      const std::uint64_t frames_before_step =
          static_cast<std::uint64_t>(step) *
          manifest_.local_ar.frames_per_decoder_step;
      if (terminal_step &&
          batch.frame_count < frames_before_step) {
        fail(
            SynthesisPipelineErrorCode::invariant_failure,
            "terminal attended prefix exceeds its codec frame count");
      }
      const std::uint32_t step_frames =
          terminal_step
              ? batch.frame_count -
                    static_cast<std::uint32_t>(
                        step *
                        manifest_.local_ar.frames_per_decoder_step)
              : manifest_.local_ar.frames_per_decoder_step;
      // An EOS at local frame zero terminates this decoder step before it
      // contributes audio. It therefore cannot advance sample-aligned text
      // progress; the preceding decoded steps remain the last committed
      // speech boundary.
      if (step_frames == 0) {
        continue;
      }
      const std::uint64_t boundary =
          next_sample_index_ +
          (static_cast<std::uint64_t>(step) *
               manifest_.local_ar.frames_per_decoder_step +
           step_frames) *
              manifest_.codec.hop_length_samples;
      const std::uint64_t committed = std::min<std::uint64_t>(
          static_cast<std::uint64_t>(
              batch.attended_token_indices[step]) +
              1U,
          text_token_ids_.size());
      if (committed > committed_text_tokens_) {
        events.push_back(AlignmentProgress{
            .sample_index = boundary,
            .committed_text_tokens = committed,
        });
        committed_text_tokens_ = committed;
      }
    }

    // The authenticated v1 manifest fixes the request PCM ring at exactly
    // eight codec frames. The next steady decode is admitted only by
    // wait_for_output_capacity(8), after the queue/live lease releases this
    // chunk's frames. Consequently this no-op owner cannot outlive the sole
    // pinned PCM slot's contents. Raising the ring capacity above eight would
    // invalidate that proof and requires a real pinned-slot pool.
    const std::shared_ptr<const float[]> samples(
        workspace_.pinned_pcm(),
        [](const float*) noexcept {});
    if (!publish_audio_chunk(AudioChunk{
        .samples = AudioBuffer(samples, expected_samples),
        .first_sample_index = next_sample_index_,
        .sequence = sequence_,
        .codec_frame_count = batch.frame_count,
        .first = pending.first,
        .final = batch.final,
        .alignment_valid = true,
        .committed_text_tokens = committed_text_tokens_,
        .alignment_events = std::move(events),
    }, batch)) {
      return false;
    }
    next_sample_index_ += expected_samples;
    ++sequence_;
    return true;
  }

  [[nodiscard]] bool wait_for_output_capacity() {
    return await_output_capacity(
        [&]() {
          return request_state_.can_publish(
              manifest_.codec.steady_frames);
        },
        [&]() {
          return observe_cancellation();
        },
        [&]() {
          return request_state_.snapshot().revision;
        },
        [&](const std::uint64_t after_revision) {
          RequestStateSnapshot updated{};
          // This timeout only bounds cancellation/error rechecks while the
          // request state is unchanged. Release-before/after-snapshot races
          // are covered by the capacity double-check and revision predicate.
          static_cast<void>(request_state_.wait_for_revision(
              after_revision,
              std::chrono::milliseconds(10),
              updated));
        });
  }

  [[nodiscard]] bool observe_cancellation() {
    return request_state_.cancellation_requested();
  }

  const RuntimeBundleManifest& manifest_;
  SessionResources& resources_;
  SessionWorkspace& workspace_;
  const std::vector<std::int32_t>& text_token_ids_;
  std::uint32_t random_seed_;
  StreamingRequestState& request_state_;
  StartupGoldenCapture* startup_capture_;
  EngineShapeParameters shape_parameters_;
  AlignmentKernelContract alignment_contract_;
  std::array<std::optional<PreparedTensorAddressSet>, 2>
      step_binding_cache_{};
  std::size_t cache_index_{0};
  std::size_t codec_state_index_{0};
  std::uint64_t generated_decoder_steps_{0};
  std::uint64_t next_sample_index_{0};
  std::uint64_t sequence_{0};
  std::uint64_t committed_text_tokens_{0};
  std::array<bool, kGenerationBatchSlotCount>
      generation_pending_{};
  std::array<bool, kGenerationBatchSlotCount>
      slot_requires_reuse_wait_{};
  bool audio_pending_{false};
  AsyncPipelineContract overlap_contract_;
};

}  // namespace

std::string_view to_string(
    const SynthesisPipelineErrorCode code) noexcept {
  switch (code) {
    case SynthesisPipelineErrorCode::cuda_failure:
      return "cuda_failure";
    case SynthesisPipelineErrorCode::engine_failure:
      return "engine_failure";
    case SynthesisPipelineErrorCode::main_decoder_failure:
      return "main_decoder_failure";
    case SynthesisPipelineErrorCode::alignment_failure:
      return "alignment_failure";
    case SynthesisPipelineErrorCode::local_ar_failure:
      return "local_ar_failure";
    case SynthesisPipelineErrorCode::codec_failure:
      return "codec_failure";
    case SynthesisPipelineErrorCode::context_exhausted:
      return "context_exhausted";
    case SynthesisPipelineErrorCode::invariant_failure:
      return "invariant_failure";
  }
  return "unknown";
}

SynthesisPipelineError::SynthesisPipelineError(
    const SynthesisPipelineErrorCode code,
    std::string detail)
    : std::runtime_error(error_message(code, detail)),
      code_(code),
      detail_(std::move(detail)) {}

SynthesisPipelineErrorCode
SynthesisPipelineError::code() const noexcept {
  return code_;
}

const std::string& SynthesisPipelineError::detail() const noexcept {
  return detail_;
}

void run_synthesis_pipeline(
    const RuntimeBundleManifest& manifest,
    SessionResources& resources,
    const std::vector<std::int32_t>& text_token_ids,
    const std::uint32_t random_seed,
    StreamingRequestState& request_state,
    StartupGoldenCapture* startup_capture,
    PipelineSynchronizationMetrics* synchronization_metrics) {
  try {
    const PipelineSynchronizationMetrics measured =
        Pipeline(
        manifest,
        resources,
        text_token_ids,
        random_seed,
        request_state,
        startup_capture)
            .run();
    if (synchronization_metrics != nullptr) {
      *synchronization_metrics = measured;
    }
  } catch (const EngineExecutionError& error) {
    fail(
        SynthesisPipelineErrorCode::engine_failure,
        error.what());
  } catch (const CudaGraphError& error) {
    fail(
        error.code() == CudaGraphErrorCode::captured_enqueue_failed
            ? SynthesisPipelineErrorCode::engine_failure
            : SynthesisPipelineErrorCode::cuda_failure,
        error.what());
  }
}

}  // namespace magpie_tts_rt
