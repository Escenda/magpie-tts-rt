#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include "manifest/manifest.hpp"

namespace magpie_tts_rt {

enum class EngineExecutionErrorCode {
  invalid_shape_parameter,
  missing_tensor_address,
  duplicate_tensor_address,
  unknown_tensor_address,
  input_shape_rejected,
  unresolved_shape,
  resolved_shape_mismatch,
  tensor_address_rejected,
  input_consumed_event_rejected,
  enqueue_failed,
};

[[nodiscard]] std::string_view to_string(
    EngineExecutionErrorCode code) noexcept;

class EngineExecutionError final : public std::runtime_error {
 public:
  EngineExecutionError(
      EngineExecutionErrorCode code,
      std::string engine_name,
      std::string tensor_name,
      std::string detail);

  [[nodiscard]] EngineExecutionErrorCode code() const noexcept;
  [[nodiscard]] const std::string& engine_name() const noexcept;
  [[nodiscard]] const std::string& tensor_name() const noexcept;
  [[nodiscard]] const std::string& detail() const noexcept;

 private:
  EngineExecutionErrorCode code_;
  std::string engine_name_;
  std::string tensor_name_;
  std::string detail_;
};

struct EngineShapeParameters {
  std::uint32_t text_token_count;
  std::uint32_t codec_frame_count;
  std::uint32_t codec_hop_length_samples;
};

struct TensorAddress {
  std::string name;
  void* address;
};

class TensorAddressSet final {
 public:
  void add(std::string name, void* address);
  [[nodiscard]] void* require(std::string_view name) const;
  [[nodiscard]] std::size_t size() const noexcept;

 private:
  std::vector<TensorAddress> addresses_;
};

// Resolves a complete named address set into manifest order once. Reusing this
// object avoids rebuilding strings and repeating O(N) name lookups on every
// invocation of a recurrent engine. The manifest remains the authority for
// tensor order; positional bindings are never accepted at the public edge.
class PreparedTensorAddressSet final {
 public:
  PreparedTensorAddressSet(
      const EngineManifest& manifest,
      const TensorAddressSet& addresses);

  [[nodiscard]] void* input(std::size_t index) const;
  [[nodiscard]] void* output(std::size_t index) const;

 private:
  std::vector<void*> inputs_;
  std::vector<void*> outputs_;
};

// Resolves only the dynamic dimensions admitted by the authenticated v1
// engine role. A dynamic dimension in any other position fails closed.
[[nodiscard]] std::vector<std::int64_t> resolve_tensor_shape(
    EngineRole role,
    const TensorSpec& tensor,
    const EngineShapeParameters& parameters);

[[nodiscard]] std::uint64_t tensor_storage_bytes(
    TensorDataType dtype,
    const std::vector<std::int64_t>& dimensions);

// Sets every dynamic input shape, requires a complete exact address set,
// verifies resolved output dimensions, and submits one enqueueV3 call. It
// never synchronizes the stream.
void enqueue_engine(
    const EngineManifest& manifest,
    nvinfer1::IExecutionContext& context,
    const EngineShapeParameters& parameters,
    const TensorAddressSet& addresses,
    cudaStream_t stream,
    cudaEvent_t input_consumed_event);

// Identical TensorRT validation and enqueue semantics with addresses already
// authenticated and flattened into manifest order.
void enqueue_engine(
    const EngineManifest& manifest,
    nvinfer1::IExecutionContext& context,
    const EngineShapeParameters& parameters,
    const PreparedTensorAddressSet& addresses,
    cudaStream_t stream,
    cudaEvent_t input_consumed_event);

}  // namespace magpie_tts_rt
