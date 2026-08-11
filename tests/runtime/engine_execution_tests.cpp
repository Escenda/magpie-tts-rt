#include "runtime/engine_execution.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using magpie_tts_rt::EngineExecutionError;
using magpie_tts_rt::EngineManifest;
using magpie_tts_rt::EngineRole;
using magpie_tts_rt::EngineShapeParameters;
using magpie_tts_rt::TensorDataType;
using magpie_tts_rt::TensorMemoryLocation;
using magpie_tts_rt::TensorAddressSet;
using magpie_tts_rt::TensorSpec;

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_text_shapes() {
  const TensorSpec tensor{
      .name = "condition",
      .dtype = TensorDataType::bf16,
      .shape = {2, -1, 768},
      .location = TensorMemoryLocation::device,
      .shape_inference_io = false,
  };
  const auto shape = magpie_tts_rt::resolve_tensor_shape(
      EngineRole::main_decoder_prefill,
      tensor,
      EngineShapeParameters{
          .text_token_count = 37,
          .codec_frame_count = 0,
          .codec_hop_length_samples = 1024,
      });
  require(
      shape == std::vector<std::int64_t>({2, 37, 768}),
      "text shape was not resolved");
  require(
      magpie_tts_rt::tensor_storage_bytes(
          TensorDataType::bf16, shape) ==
          2ULL * 37ULL * 768ULL * 2ULL,
      "BF16 storage byte count mismatch");
}

void test_tail_shapes() {
  const EngineShapeParameters parameters{
      .text_token_count = 0,
      .codec_frame_count = 7,
      .codec_hop_length_samples = 1024,
  };
  const auto codes = magpie_tts_rt::resolve_tensor_shape(
      EngineRole::nanocodec_tail_1_8,
      TensorSpec{
          .name = "codec_tokens",
          .dtype = TensorDataType::int64,
          .shape = {1, 8, -1},
          .location = TensorMemoryLocation::device,
          .shape_inference_io = false,
      },
      parameters);
  const auto pcm = magpie_tts_rt::resolve_tensor_shape(
      EngineRole::nanocodec_tail_1_8,
      TensorSpec{
          .name = "pcm",
          .dtype = TensorDataType::fp32,
          .shape = {1, -1},
          .location = TensorMemoryLocation::device,
          .shape_inference_io = false,
      },
      parameters);
  require(
      codes == std::vector<std::int64_t>({1, 8, 7}),
      "tail code shape mismatch");
  require(
      pcm == std::vector<std::int64_t>({1, 7168}),
      "tail PCM shape mismatch");
}

void test_unknown_dynamic_shape_fails() {
  bool rejected = false;
  try {
    static_cast<void>(magpie_tts_rt::resolve_tensor_shape(
        EngineRole::local_ar_16,
        TensorSpec{
            .name = "decoder_hidden",
            .dtype = TensorDataType::bf16,
            .shape = {2, -1},
            .location = TensorMemoryLocation::device,
            .shape_inference_io = false,
        },
        EngineShapeParameters{
            .text_token_count = 9,
            .codec_frame_count = 0,
            .codec_hop_length_samples = 1024,
        }));
  } catch (const EngineExecutionError&) {
    rejected = true;
  }
  require(rejected, "unknown Local AR dynamic shape was accepted");
}

void test_scalar_host_shape_input_storage() {
  const TensorSpec position{
      .name = "position",
      .dtype = TensorDataType::int64,
      .shape = {},
      .location = TensorMemoryLocation::host,
      .shape_inference_io = true,
  };
  const auto shape = magpie_tts_rt::resolve_tensor_shape(
      EngineRole::main_decoder_step,
      position,
      EngineShapeParameters{
          .text_token_count = 39,
          .codec_frame_count = 0,
          .codec_hop_length_samples = 1024,
      });
  require(shape.empty(), "scalar position shape changed rank");
  require(
      magpie_tts_rt::tensor_storage_bytes(
          TensorDataType::int64, shape) == sizeof(std::int64_t),
      "scalar position storage byte count mismatch");
}

void test_prepared_addresses_follow_manifest_order() {
  const TensorSpec input{
      .name = "input",
      .dtype = TensorDataType::bf16,
      .shape = {1},
      .location = TensorMemoryLocation::device,
      .shape_inference_io = false,
  };
  const TensorSpec output{
      .name = "output",
      .dtype = TensorDataType::bf16,
      .shape = {1},
      .location = TensorMemoryLocation::device,
      .shape_inference_io = false,
  };
  EngineManifest manifest{};
  manifest.name = "prepared-address-test";
  manifest.inputs = {input};
  manifest.outputs = {output};
  auto* const input_address = reinterpret_cast<void*>(0x1000U);
  auto* const output_address = reinterpret_cast<void*>(0x2000U);
  TensorAddressSet named;
  named.add("output", output_address);
  named.add("input", input_address);
  const magpie_tts_rt::PreparedTensorAddressSet prepared(
      manifest, named);
  require(
      prepared.input(0) == input_address &&
          prepared.output(0) == output_address,
      "prepared addresses did not follow manifest order");

  TensorAddressSet missing;
  missing.add("input", input_address);
  bool rejected = false;
  try {
    static_cast<void>(magpie_tts_rt::PreparedTensorAddressSet(
        manifest, missing));
  } catch (const EngineExecutionError&) {
    rejected = true;
  }
  require(rejected, "incomplete prepared address set was accepted");
}

}  // namespace

int main() {
  try {
    test_text_shapes();
    test_tail_shapes();
    test_unknown_dynamic_shape_fails();
    test_scalar_host_shape_input_storage();
    test_prepared_addresses_follow_manifest_order();
  } catch (const std::exception& error) {
    std::cerr << "engine execution test failed: "
              << error.what() << '\n';
    return 1;
  }
  return 0;
}
