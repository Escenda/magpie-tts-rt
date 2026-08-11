#include "runtime/engine_execution.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace magpie_tts_rt {
namespace {

[[nodiscard]] std::string error_message(
    const EngineExecutionErrorCode code,
    const std::string_view engine_name,
    const std::string_view tensor_name,
    const std::string_view detail) {
  std::string message =
      "TensorRT execution failed [code=" +
      std::string(to_string(code)) + ", engine=" +
      std::string(engine_name);
  if (!tensor_name.empty()) {
    message += ", tensor=" + std::string(tensor_name);
  }
  message += "]: " + std::string(detail);
  return message;
}

[[noreturn]] void fail(
    const EngineExecutionErrorCode code,
    const std::string_view engine_name,
    const std::string_view tensor_name,
    const std::string& detail) {
  throw EngineExecutionError(
      code,
      std::string(engine_name),
      std::string(tensor_name),
      detail);
}

[[nodiscard]] std::uint64_t data_type_bytes(
    const TensorDataType dtype) {
  switch (dtype) {
    case TensorDataType::fp32:
    case TensorDataType::int32:
      return 4;
    case TensorDataType::fp16:
    case TensorDataType::bf16:
      return 2;
    case TensorDataType::int64:
      return 8;
    case TensorDataType::int8:
    case TensorDataType::uint8:
    case TensorDataType::boolean:
      return 1;
  }
  fail(
      EngineExecutionErrorCode::invalid_shape_parameter,
      "",
      "",
      "unknown tensor data type");
}

[[nodiscard]] std::string dimensions_string(
    const std::vector<std::int64_t>& dimensions) {
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << dimensions[index];
  }
  output << ']';
  return output.str();
}

[[nodiscard]] std::vector<std::int64_t> context_dimensions(
    const nvinfer1::Dims& dimensions,
    const std::string_view engine_name,
    const std::string_view tensor_name) {
  if (dimensions.nbDims < 0) {
    fail(
        EngineExecutionErrorCode::unresolved_shape,
        engine_name,
        tensor_name,
        "TensorRT returned an invalid dimension count");
  }
  std::vector<std::int64_t> resolved;
  resolved.reserve(static_cast<std::size_t>(dimensions.nbDims));
  for (std::int32_t index = 0; index < dimensions.nbDims; ++index) {
    if (dimensions.d[index] <= 0) {
      fail(
          EngineExecutionErrorCode::unresolved_shape,
          engine_name,
          tensor_name,
          "TensorRT retained a non-positive runtime dimension");
    }
    resolved.push_back(dimensions.d[index]);
  }
  return resolved;
}

[[nodiscard]] nvinfer1::Dims tensorrt_dimensions(
    const std::vector<std::int64_t>& dimensions,
    const std::string_view engine_name,
    const std::string_view tensor_name) {
  if (dimensions.size() >
      static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS)) {
    fail(
        EngineExecutionErrorCode::invalid_shape_parameter,
        engine_name,
        tensor_name,
        "tensor rank exceeds TensorRT Dims::MAX_DIMS");
  }
  nvinfer1::Dims output{};
  output.nbDims = static_cast<std::int32_t>(dimensions.size());
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    if (dimensions[index] <= 0 ||
        dimensions[index] >
            std::numeric_limits<std::int32_t>::max()) {
      fail(
          EngineExecutionErrorCode::invalid_shape_parameter,
          engine_name,
          tensor_name,
          "resolved dimension is outside positive INT32");
    }
    output.d[index] = static_cast<std::int32_t>(dimensions[index]);
  }
  return output;
}

[[nodiscard]] bool is_text_dynamic_role(const EngineRole role) {
  return role == EngineRole::text_encoder ||
         role == EngineRole::main_decoder_prefill ||
         role == EngineRole::main_decoder_step;
}

[[nodiscard]] std::int64_t resolve_dynamic_dimension(
    const EngineRole role,
    const TensorSpec& tensor,
    const std::size_t dimension_index,
    const EngineShapeParameters& parameters) {
  if (is_text_dynamic_role(role)) {
    if (parameters.text_token_count == 0) {
      fail(
          EngineExecutionErrorCode::invalid_shape_parameter,
          to_string(role),
          tensor.name,
          "text-token count must be positive");
    }
    return parameters.text_token_count;
  }
  if (role == EngineRole::nanocodec_tail_1_8) {
    if (parameters.codec_frame_count == 0 ||
        parameters.codec_frame_count > 8 ||
        parameters.codec_hop_length_samples == 0) {
      fail(
          EngineExecutionErrorCode::invalid_shape_parameter,
          to_string(role),
          tensor.name,
          "tail codec frames must be in [1,8] and hop length positive");
    }
    if (tensor.name == "codec_tokens" && dimension_index == 2) {
      return parameters.codec_frame_count;
    }
    if (tensor.name == "pcm" && dimension_index == 1) {
      const std::uint64_t samples =
          static_cast<std::uint64_t>(parameters.codec_frame_count) *
          parameters.codec_hop_length_samples;
      if (samples >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::int64_t>::max())) {
        fail(
            EngineExecutionErrorCode::invalid_shape_parameter,
            to_string(role),
            tensor.name,
            "tail PCM dimension overflow");
      }
      return static_cast<std::int64_t>(samples);
    }
  }
  fail(
      EngineExecutionErrorCode::invalid_shape_parameter,
      to_string(role),
      tensor.name,
      "dynamic dimension is not admitted at index " +
          std::to_string(dimension_index));
}

}  // namespace

std::string_view to_string(
    const EngineExecutionErrorCode code) noexcept {
  switch (code) {
    case EngineExecutionErrorCode::invalid_shape_parameter:
      return "invalid_shape_parameter";
    case EngineExecutionErrorCode::missing_tensor_address:
      return "missing_tensor_address";
    case EngineExecutionErrorCode::duplicate_tensor_address:
      return "duplicate_tensor_address";
    case EngineExecutionErrorCode::unknown_tensor_address:
      return "unknown_tensor_address";
    case EngineExecutionErrorCode::input_shape_rejected:
      return "input_shape_rejected";
    case EngineExecutionErrorCode::unresolved_shape:
      return "unresolved_shape";
    case EngineExecutionErrorCode::resolved_shape_mismatch:
      return "resolved_shape_mismatch";
    case EngineExecutionErrorCode::tensor_address_rejected:
      return "tensor_address_rejected";
    case EngineExecutionErrorCode::input_consumed_event_rejected:
      return "input_consumed_event_rejected";
    case EngineExecutionErrorCode::enqueue_failed:
      return "enqueue_failed";
  }
  return "unknown";
}

EngineExecutionError::EngineExecutionError(
    const EngineExecutionErrorCode code,
    std::string engine_name,
    std::string tensor_name,
    std::string detail)
    : std::runtime_error(
          error_message(code, engine_name, tensor_name, detail)),
      code_(code),
      engine_name_(std::move(engine_name)),
      tensor_name_(std::move(tensor_name)),
      detail_(std::move(detail)) {}

EngineExecutionErrorCode EngineExecutionError::code() const noexcept {
  return code_;
}

const std::string& EngineExecutionError::engine_name() const noexcept {
  return engine_name_;
}

const std::string& EngineExecutionError::tensor_name() const noexcept {
  return tensor_name_;
}

const std::string& EngineExecutionError::detail() const noexcept {
  return detail_;
}

void TensorAddressSet::add(std::string name, void* address) {
  if (name.empty() || address == nullptr) {
    fail(
        EngineExecutionErrorCode::missing_tensor_address,
        "",
        name,
        "tensor name and address must both be present");
  }
  const auto duplicate = std::find_if(
      addresses_.begin(),
      addresses_.end(),
      [&](const TensorAddress& candidate) {
        return candidate.name == name;
      });
  if (duplicate != addresses_.end()) {
    fail(
        EngineExecutionErrorCode::duplicate_tensor_address,
        "",
        name,
        "tensor address was added more than once");
  }
  addresses_.push_back(
      TensorAddress{.name = std::move(name), .address = address});
}

void* TensorAddressSet::require(const std::string_view name) const {
  const auto found = std::find_if(
      addresses_.begin(),
      addresses_.end(),
      [&](const TensorAddress& candidate) {
        return candidate.name == name;
      });
  if (found == addresses_.end()) {
    fail(
        EngineExecutionErrorCode::missing_tensor_address,
        "",
        name,
        "tensor has no bound address");
  }
  return found->address;
}

std::size_t TensorAddressSet::size() const noexcept {
  return addresses_.size();
}

PreparedTensorAddressSet::PreparedTensorAddressSet(
    const EngineManifest& manifest,
    const TensorAddressSet& addresses) {
  const std::size_t expected_address_count =
      manifest.inputs.size() + manifest.outputs.size();
  if (addresses.size() != expected_address_count) {
    fail(
        addresses.size() < expected_address_count
            ? EngineExecutionErrorCode::missing_tensor_address
            : EngineExecutionErrorCode::unknown_tensor_address,
        manifest.name,
        "",
        "expected exactly " +
            std::to_string(expected_address_count) +
            " tensor addresses, got " +
            std::to_string(addresses.size()));
  }
  inputs_.reserve(manifest.inputs.size());
  outputs_.reserve(manifest.outputs.size());
  for (const TensorSpec& input : manifest.inputs) {
    inputs_.push_back(addresses.require(input.name));
  }
  for (const TensorSpec& output : manifest.outputs) {
    outputs_.push_back(addresses.require(output.name));
  }
}

void* PreparedTensorAddressSet::input(const std::size_t index) const {
  return inputs_.at(index);
}

void* PreparedTensorAddressSet::output(const std::size_t index) const {
  return outputs_.at(index);
}

std::vector<std::int64_t> resolve_tensor_shape(
    const EngineRole role,
    const TensorSpec& tensor,
    const EngineShapeParameters& parameters) {
  std::vector<std::int64_t> resolved = tensor.shape;
  for (std::size_t index = 0; index < resolved.size(); ++index) {
    if (resolved[index] == -1) {
      resolved[index] = resolve_dynamic_dimension(
          role, tensor, index, parameters);
    } else if (resolved[index] <= 0) {
      fail(
          EngineExecutionErrorCode::invalid_shape_parameter,
          to_string(role),
          tensor.name,
          "manifest shape contains a non-positive static dimension");
    }
  }
  return resolved;
}

std::uint64_t tensor_storage_bytes(
    const TensorDataType dtype,
    const std::vector<std::int64_t>& dimensions) {
  std::uint64_t elements = 1;
  for (const std::int64_t dimension : dimensions) {
    if (dimension <= 0) {
      fail(
          EngineExecutionErrorCode::invalid_shape_parameter,
          "",
          "",
          "storage shape contains a non-positive dimension");
    }
    const std::uint64_t unsigned_dimension =
        static_cast<std::uint64_t>(dimension);
    if (elements >
        std::numeric_limits<std::uint64_t>::max() /
            unsigned_dimension) {
      fail(
          EngineExecutionErrorCode::invalid_shape_parameter,
          "",
          "",
          "tensor element count overflow");
    }
    elements *= unsigned_dimension;
  }
  const std::uint64_t bytes_per_element = data_type_bytes(dtype);
  if (elements >
      std::numeric_limits<std::uint64_t>::max() /
          bytes_per_element) {
    fail(
        EngineExecutionErrorCode::invalid_shape_parameter,
        "",
        "",
        "tensor storage byte count overflow");
  }
  return elements * bytes_per_element;
}

template <typename AddressProvider>
void enqueue_engine_with_addresses(
    const EngineManifest& manifest,
    nvinfer1::IExecutionContext& context,
    const EngineShapeParameters& parameters,
    AddressProvider&& address,
    const cudaStream_t stream,
    const cudaEvent_t input_consumed_event) {
  for (const TensorSpec& input : manifest.inputs) {
    const std::vector<std::int64_t> resolved =
        resolve_tensor_shape(manifest.role, input, parameters);
    if (std::find(input.shape.begin(), input.shape.end(), -1) !=
            input.shape.end() &&
        !input.shape_inference_io &&
        !context.setInputShape(
            input.name.c_str(),
            tensorrt_dimensions(
                resolved, manifest.name, input.name))) {
      fail(
          EngineExecutionErrorCode::input_shape_rejected,
          manifest.name,
          input.name,
          "TensorRT rejected shape " +
              dimensions_string(resolved));
    }
  }

  const auto validate_shape = [&](const TensorSpec& tensor) {
    const std::vector<std::int64_t> expected =
        resolve_tensor_shape(manifest.role, tensor, parameters);
    const std::vector<std::int64_t> actual =
        context_dimensions(
            context.getTensorShape(tensor.name.c_str()),
            manifest.name,
            tensor.name);
    if (actual != expected) {
      fail(
          EngineExecutionErrorCode::resolved_shape_mismatch,
          manifest.name,
          tensor.name,
          "expected " + dimensions_string(expected) +
              ", got " + dimensions_string(actual));
    }
  };
  const auto bind = [&context, &manifest](
                        const TensorSpec& tensor,
                        void* const tensor_address) {
    if (!context.setTensorAddress(
            tensor.name.c_str(), tensor_address)) {
      fail(
          EngineExecutionErrorCode::tensor_address_rejected,
          manifest.name,
          tensor.name,
          "TensorRT rejected the authenticated tensor address");
    }
  };
  // TensorRT shape inference reads HOST shape-input values through their
  // bound addresses. All input addresses therefore have to be present before
  // inferShapes(); the caller owns those buffers through enqueue completion.
  for (std::size_t index = 0;
       index < manifest.inputs.size();
       ++index) {
    const TensorSpec& input = manifest.inputs[index];
    validate_shape(input);
    bind(input, address(true, index, input));
  }

  const std::int32_t unresolved = context.inferShapes(0, nullptr);
  if (unresolved != 0) {
    fail(
        EngineExecutionErrorCode::unresolved_shape,
        manifest.name,
        "",
        unresolved < 0
            ? "TensorRT shape inference failed"
            : std::to_string(unresolved) +
                  " input shapes remain unresolved");
  }

  for (std::size_t index = 0;
       index < manifest.outputs.size();
       ++index) {
    const TensorSpec& output = manifest.outputs[index];
    validate_shape(output);
    bind(output, address(false, index, output));
  }
  if (!context.setInputConsumedEvent(input_consumed_event)) {
    fail(
        EngineExecutionErrorCode::input_consumed_event_rejected,
        manifest.name,
        "",
        "TensorRT rejected the input-consumed event");
  }
  if (!context.enqueueV3(stream)) {
    fail(
        EngineExecutionErrorCode::enqueue_failed,
        manifest.name,
        "",
        "enqueueV3 returned false");
  }
}

void enqueue_engine(
    const EngineManifest& manifest,
    nvinfer1::IExecutionContext& context,
    const EngineShapeParameters& parameters,
    const TensorAddressSet& addresses,
    const cudaStream_t stream,
    const cudaEvent_t input_consumed_event) {
  const std::size_t expected_address_count =
      manifest.inputs.size() + manifest.outputs.size();
  if (addresses.size() != expected_address_count) {
    fail(
        addresses.size() < expected_address_count
            ? EngineExecutionErrorCode::missing_tensor_address
            : EngineExecutionErrorCode::unknown_tensor_address,
        manifest.name,
        "",
        "expected exactly " +
            std::to_string(expected_address_count) +
            " tensor addresses, got " +
            std::to_string(addresses.size()));
  }
  enqueue_engine_with_addresses(
      manifest,
      context,
      parameters,
      [&addresses](
          const bool,
          const std::size_t,
          const TensorSpec& tensor) {
        return addresses.require(tensor.name);
      },
      stream,
      input_consumed_event);
}

void enqueue_engine(
    const EngineManifest& manifest,
    nvinfer1::IExecutionContext& context,
    const EngineShapeParameters& parameters,
    const PreparedTensorAddressSet& addresses,
    const cudaStream_t stream,
    const cudaEvent_t input_consumed_event) {
  enqueue_engine_with_addresses(
      manifest,
      context,
      parameters,
      [&addresses](
          const bool input,
          const std::size_t index,
          const TensorSpec&) {
        return input ? addresses.input(index) : addresses.output(index);
      },
      stream,
      input_consumed_event);
}

}  // namespace magpie_tts_rt
