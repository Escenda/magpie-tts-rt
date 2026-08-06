cmake_minimum_required(VERSION 3.24)

project(
  MagpieTTSRT
  VERSION 0.1.0
  DESCRIPTION "Low-latency TensorRT runtime for MagpieTTS"
  LANGUAGES C CXX
)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "MagpieTTS-RT currently supports Linux only")
endif()

include(CMakePackageConfigHelpers)
include(FetchContent)
include(GNUInstallDirs)

function(mtt_require_elf_soname library_path expected_soname)
  if(NOT CMAKE_READELF)
    find_program(CMAKE_READELF NAMES readelf REQUIRED)
  endif()
  execute_process(
    COMMAND "${CMAKE_READELF}" -d "${library_path}"
    RESULT_VARIABLE _mtt_readelf_result
    OUTPUT_VARIABLE _mtt_readelf_output
    ERROR_VARIABLE _mtt_readelf_error
  )
  if(NOT _mtt_readelf_result EQUAL 0)
    message(
      FATAL_ERROR
      "Unable to inspect ${library_path}: ${_mtt_readelf_error}"
    )
  endif()
  string(
    REGEX MATCH
    "\\(SONAME\\)[^\n]*"
    _mtt_soname_line
    "${_mtt_readelf_output}"
  )
  string(
    FIND
    "${_mtt_soname_line}"
    "[${expected_soname}]"
    _mtt_expected_soname
  )
  if(_mtt_soname_line STREQUAL "" OR _mtt_expected_soname EQUAL -1)
    message(
      FATAL_ERROR
      "${library_path} must have ELF SONAME ${expected_soname}"
    )
  endif()
endfunction()

option(
  MAGPIE_TTS_RT_BUILD_RUNTIME
  "Build the CUDA and TensorRT shared runtime"
  ON
)
option(MAGPIE_TTS_RT_BUILD_TESTS "Build contract and runtime tests" ON)
option(
  MAGPIE_TTS_RT_BUILD_GPU_TESTS
  "Build tests that create a CUDA and TensorRT runtime on the current GPU"
  OFF
)
option(MAGPIE_TTS_RT_WARNINGS_AS_ERRORS "Treat project warnings as errors" ON)

find_package(nlohmann_json 3.11 CONFIG REQUIRED)
find_package(OpenSSL 3 REQUIRED COMPONENTS Crypto)
mtt_require_elf_soname("${OPENSSL_CRYPTO_LIBRARY}" "libcrypto.so.3")

add_library(
  magpie_tts_rt_contracts
  STATIC
    src/manifest/manifest.cpp
    src/bundle/bundle.cpp
    src/runtime/alignment_controller.cpp
    src/runtime/request_state.cpp
    src/runtime/startup_golden.cpp
)
target_compile_features(magpie_tts_rt_contracts PUBLIC cxx_std_20)
target_include_directories(
  magpie_tts_rt_contracts
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
)
target_link_libraries(
  magpie_tts_rt_contracts
  PUBLIC nlohmann_json::nlohmann_json OpenSSL::Crypto
)
set_target_properties(
  magpie_tts_rt_contracts
  PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN YES
)

function(mtt_enable_strict_warnings target)
  if(MSVC)
    target_compile_options("${target}" PRIVATE /W4)
    if(MAGPIE_TTS_RT_WARNINGS_AS_ERRORS)
      target_compile_options("${target}" PRIVATE /WX)
    endif()
  else()
    target_compile_options(
      "${target}"
      PRIVATE
        "$<$<COMPILE_LANGUAGE:C,CXX>:-Wall>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:-Wconversion>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:-Wsign-conversion>"
    )
    if(MAGPIE_TTS_RT_WARNINGS_AS_ERRORS)
      target_compile_options(
        "${target}"
        PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Werror>"
      )
    endif()
  endif()
endfunction()

mtt_enable_strict_warnings(magpie_tts_rt_contracts)

if(MAGPIE_TTS_RT_BUILD_RUNTIME)
  find_package(CUDAToolkit 13 REQUIRED)
  FetchContent_Declare(
    mtt_cutlass
    URL
      "https://github.com/NVIDIA/cutlass/archive/0d2b201e8c1c4a03efa6e9c468161916e2334725.tar.gz"
    URL_HASH
      "SHA256=5288044d2d5e81632ac0c812b6b85c744901a7d3fd11c9119f18f71c3cef5f79"
  )
  FetchContent_GetProperties(mtt_cutlass)
  if(NOT mtt_cutlass_POPULATED)
    FetchContent_Populate(mtt_cutlass)
  endif()
  set(
    CMAKE_CUDA_ARCHITECTURES
    110
    CACHE STRING
    "CUDA architectures supported by the MagpieTTS-RT plugins"
  )
  enable_language(CUDA)
  find_library(
    MAGPIE_TTS_RT_CUDART_LIBRARY
    NAMES cudart libcudart.so.13
    NAMES_PER_DIR
    HINTS "${CUDAToolkit_LIBRARY_DIR}"
    REQUIRED
  )
  find_library(
    MAGPIE_TTS_RT_CUBLAS_LIBRARY
    NAMES cublas libcublas.so.13
    NAMES_PER_DIR
    HINTS "${CUDAToolkit_LIBRARY_DIR}"
    PATH_SUFFIXES
      lib
      lib64
      targets/sbsa-linux/lib
      targets/aarch64-linux/lib
      targets/x86_64-linux/lib
    REQUIRED
  )
  find_path(
    TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATH_SUFFIXES aarch64-linux-gnu x86_64-linux-gnu
    REQUIRED
  )
  find_library(
    TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer libnvinfer.so.10
    NAMES_PER_DIR
    REQUIRED
  )
  find_library(
    MAGPIE_TTS_RT_NVML_LIBRARY
    NAMES nvidia-ml libnvidia-ml.so.1
    NAMES_PER_DIR
    HINTS
      "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia"
      "/usr/lib64/nvidia"
    REQUIRED
  )
  mtt_require_elf_soname(
    "${MAGPIE_TTS_RT_CUDART_LIBRARY}"
    "libcudart.so.13"
  )
  mtt_require_elf_soname(
    "${MAGPIE_TTS_RT_CUBLAS_LIBRARY}"
    "libcublas.so.13"
  )
  mtt_require_elf_soname(
    "${TENSORRT_NVINFER_LIBRARY}"
    "libnvinfer.so.10"
  )
  mtt_require_elf_soname(
    "${MAGPIE_TTS_RT_NVML_LIBRARY}"
    "libnvidia-ml.so.1"
  )

  add_library(MagpieTTSRT_tensorrt_nvinfer SHARED IMPORTED)
  set_target_properties(
    MagpieTTSRT_tensorrt_nvinfer
    PROPERTIES
      IMPORTED_LOCATION "${TENSORRT_NVINFER_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
  )

  add_library(MagpieTTSRT_cuda_cudart SHARED IMPORTED)
  set_target_properties(
    MagpieTTSRT_cuda_cudart
    PROPERTIES
      IMPORTED_LOCATION "${MAGPIE_TTS_RT_CUDART_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}"
  )

  add_library(MagpieTTSRT_cuda_cublas SHARED IMPORTED)
  set_target_properties(
    MagpieTTSRT_cuda_cublas
    PROPERTIES
      IMPORTED_LOCATION "${MAGPIE_TTS_RT_CUBLAS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}"
  )

  add_library(MagpieTTSRT_nvidia_nvml SHARED IMPORTED)
  set_target_properties(
    MagpieTTSRT_nvidia_nvml
    PROPERTIES
      IMPORTED_LOCATION "${MAGPIE_TTS_RT_NVML_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}"
  )

  add_library(
    magpie_tts_rt
    SHARED
      src/api/magpie_tts_rt.cpp
      src/runtime/alignment_kernel.cu
      src/runtime/cuda_memory.cpp
      src/runtime/engine_execution.cpp
      src/runtime/fingerprint.cpp
      src/runtime/model_loader.cpp
      src/runtime/session_resources.cpp
      src/runtime/session_workspace.cpp
      src/runtime/startup_gate.cpp
      src/runtime/synthesis_pipeline.cpp
      src/runtime/synthesis_kernels.cu
  )
  add_library(MagpieTTSRT::runtime ALIAS magpie_tts_rt)

  add_library(
    magpie_tts_rt_plugins
    SHARED
      plugins/local_ar_plugins.cu
  )
  add_library(MagpieTTSRT::plugins ALIAS magpie_tts_rt_plugins)
  target_compile_features(magpie_tts_rt_plugins PRIVATE cxx_std_20)
  target_include_directories(
    magpie_tts_rt_plugins
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/plugins"
      "${TENSORRT_INCLUDE_DIR}"
  )
  target_include_directories(
    magpie_tts_rt_plugins
    SYSTEM PRIVATE "${mtt_cutlass_SOURCE_DIR}/include"
  )
  target_compile_definitions(
    magpie_tts_rt_plugins
    PRIVATE MAGPIE_TTS_RT_BUILDING_PLUGIN_LIBRARY
  )
  target_link_libraries(
    magpie_tts_rt_plugins
    PRIVATE
      MagpieTTSRT_cuda_cublas
      MagpieTTSRT_cuda_cudart
      MagpieTTSRT_tensorrt_nvinfer
  )
  target_compile_options(
    magpie_tts_rt_plugins
    PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:--frandom-seed=magpie_tts_rt_plugins_v1>"
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
  )
  if(MAGPIE_TTS_RT_WARNINGS_AS_ERRORS)
    target_compile_options(
      magpie_tts_rt_plugins
      PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>"
    )
  endif()
  target_link_options(
    magpie_tts_rt_plugins
    PRIVATE
      "LINKER:--no-undefined"
      "LINKER:--strip-all"
      "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/magpie_tts_rt_plugins.map"
  )
  set_target_properties(
    magpie_tts_rt_plugins
    PROPERTIES
      CUDA_ARCHITECTURES 110
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED YES
      CXX_VISIBILITY_PRESET hidden
      EXPORT_NAME plugins
      POSITION_INDEPENDENT_CODE ON
      VISIBILITY_INLINES_HIDDEN YES
      OUTPUT_NAME magpie_tts_rt_plugins
      LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/magpie_tts_rt_plugins.map"
      VERSION "${PROJECT_VERSION}"
      SOVERSION 0
  )

  target_compile_features(magpie_tts_rt PRIVATE cxx_std_20)
  target_compile_options(
    magpie_tts_rt
    PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
  )
  if(MAGPIE_TTS_RT_WARNINGS_AS_ERRORS)
    target_compile_options(
      magpie_tts_rt
      PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>"
    )
  endif()
  set_target_properties(
    magpie_tts_rt
    PROPERTIES
      CUDA_ARCHITECTURES 110
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED YES
  )
  target_compile_definitions(
    magpie_tts_rt
    PRIVATE MAGPIE_TTS_RT_BUILDING_LIBRARY
  )
  target_include_directories(
    magpie_tts_rt
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  )
  target_link_libraries(
    magpie_tts_rt
    PRIVATE
      MagpieTTSRT_cuda_cudart
      MagpieTTSRT_nvidia_nvml
      MagpieTTSRT_tensorrt_nvinfer
      magpie_tts_rt_contracts
      "${CMAKE_DL_LIBS}"
  )
  target_include_directories(
    magpie_tts_rt
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_options(
    magpie_tts_rt
    PRIVATE
      "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/magpie_tts_rt.map"
  )
  set_target_properties(
    magpie_tts_rt
    PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      EXPORT_NAME runtime
      VISIBILITY_INLINES_HIDDEN YES
      OUTPUT_NAME magpie_tts_rt
      LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/magpie_tts_rt.map"
      VERSION "${PROJECT_VERSION}"
      SOVERSION 0
  )
  mtt_enable_strict_warnings(magpie_tts_rt)

  add_executable(mtt-runtime-smoke tools/runtime_smoke.cpp)
  target_compile_features(mtt-runtime-smoke PRIVATE cxx_std_20)
  target_include_directories(
    mtt-runtime-smoke
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  target_link_libraries(
    mtt-runtime-smoke
    PRIVATE nlohmann_json::nlohmann_json "${CMAKE_DL_LIBS}"
  )
  mtt_enable_strict_warnings(mtt-runtime-smoke)
elseif(MAGPIE_TTS_RT_BUILD_GPU_TESTS)
  message(
    FATAL_ERROR
    "MAGPIE_TTS_RT_BUILD_GPU_TESTS requires MAGPIE_TTS_RT_BUILD_RUNTIME"
  )
endif()

add_executable(mtt-runtime-benchmark tools/runtime_benchmark.cpp)
target_compile_features(mtt-runtime-benchmark PRIVATE cxx_std_20)
target_include_directories(
  mtt-runtime-benchmark
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(
  mtt-runtime-benchmark
  PRIVATE
    nlohmann_json::nlohmann_json
    OpenSSL::Crypto
    "${CMAKE_DL_LIBS}"
)
mtt_enable_strict_warnings(mtt-runtime-benchmark)

add_executable(mtt-validate-manifest tools/validate_manifest.cpp)
target_compile_features(mtt-validate-manifest PRIVATE cxx_std_20)
target_link_libraries(mtt-validate-manifest PRIVATE magpie_tts_rt_contracts)
mtt_enable_strict_warnings(mtt-validate-manifest)

add_executable(mtt-validate-bundle tools/validate_bundle.cpp)
target_compile_features(mtt-validate-bundle PRIVATE cxx_std_20)
target_link_libraries(mtt-validate-bundle PRIVATE magpie_tts_rt_contracts)
mtt_enable_strict_warnings(mtt-validate-bundle)

if(MAGPIE_TTS_RT_BUILD_TESTS)
  enable_testing()

  add_library(mtt_c_header_compile_test OBJECT tests/c_api/c_header_test.c)
  target_compile_definitions(
    mtt_c_header_compile_test
    PRIVATE MTT_HEADER_COMPILE_ONLY
  )
  target_include_directories(
    mtt_c_header_compile_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  mtt_enable_strict_warnings(mtt_c_header_compile_test)

  add_executable(mtt_manifest_tests tests/manifest/manifest_tests.cpp)
  target_compile_features(mtt_manifest_tests PRIVATE cxx_std_20)
  target_link_libraries(mtt_manifest_tests PRIVATE magpie_tts_rt_contracts)
  mtt_enable_strict_warnings(mtt_manifest_tests)
  add_test(
    NAME manifest
    COMMAND
      mtt_manifest_tests
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/manifest/fixtures/minimal-valid.json"
  )
  # The sanitizer workflows run the complete manifest mutation contract with
  # UAR disabled, then enable UAR for the bounded runtime-state subset. The
  # manifest parser is deliberately absent from that second pass because its
  # aarch64 FakeStack cost is not a bounded CI gate.
  set_tests_properties(manifest PROPERTIES TIMEOUT 120)

  add_executable(mtt_bundle_tests tests/bundle/bundle_tests.cpp)
  target_compile_features(mtt_bundle_tests PRIVATE cxx_std_20)
  target_link_libraries(mtt_bundle_tests PRIVATE magpie_tts_rt_contracts)
  mtt_enable_strict_warnings(mtt_bundle_tests)
  add_test(
    NAME bundle
    COMMAND
      mtt_bundle_tests
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/manifest/fixtures/minimal-valid.json"
  )
  set_tests_properties(bundle PROPERTIES TIMEOUT 900)

  add_executable(
    mtt_request_state_tests
    tests/runtime/request_state_tests.cpp
  )
  target_compile_features(mtt_request_state_tests PRIVATE cxx_std_20)
  target_link_libraries(
    mtt_request_state_tests
    PRIVATE magpie_tts_rt_contracts
  )
  mtt_enable_strict_warnings(mtt_request_state_tests)
  add_test(NAME request_state COMMAND mtt_request_state_tests)

  add_executable(
    mtt_async_pipeline_contract_tests
    tests/runtime/async_pipeline_contract_tests.cpp
  )
  target_compile_features(
    mtt_async_pipeline_contract_tests
    PRIVATE cxx_std_20
  )
  target_link_libraries(
    mtt_async_pipeline_contract_tests
    PRIVATE magpie_tts_rt_contracts
  )
  mtt_enable_strict_warnings(mtt_async_pipeline_contract_tests)
  add_test(
    NAME async_pipeline_contract
    COMMAND mtt_async_pipeline_contract_tests
  )

  add_executable(
    mtt_alignment_controller_tests
    tests/runtime/alignment_controller_tests.cpp
  )
  target_compile_features(
    mtt_alignment_controller_tests
    PRIVATE cxx_std_20
  )
  target_link_libraries(
    mtt_alignment_controller_tests
    PRIVATE magpie_tts_rt_contracts
  )
  mtt_enable_strict_warnings(mtt_alignment_controller_tests)
  add_test(
    NAME alignment_controller
    COMMAND mtt_alignment_controller_tests
  )

  add_executable(
    mtt_startup_golden_tests
    tests/runtime/startup_golden_tests.cpp
  )
  target_compile_features(
    mtt_startup_golden_tests
    PRIVATE cxx_std_20
  )
  target_link_libraries(
    mtt_startup_golden_tests
    PRIVATE magpie_tts_rt_contracts
  )
  mtt_enable_strict_warnings(mtt_startup_golden_tests)
  add_test(
    NAME startup_golden
    COMMAND
      mtt_startup_golden_tests
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/manifest/fixtures/minimal-valid.json"
  )

  add_library(
    mtt_runtime_benchmark_mock
    SHARED tests/runtime/runtime_benchmark_mock.cpp
  )
  target_compile_features(
    mtt_runtime_benchmark_mock PRIVATE cxx_std_20
  )
  target_include_directories(
    mtt_runtime_benchmark_mock
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  mtt_enable_strict_warnings(mtt_runtime_benchmark_mock)

  add_library(
    mtt_runtime_benchmark_nvml_mock
    SHARED tests/runtime/runtime_benchmark_nvml_mock.cpp
  )
  target_compile_features(
    mtt_runtime_benchmark_nvml_mock PRIVATE cxx_std_20
  )
  mtt_enable_strict_warnings(mtt_runtime_benchmark_nvml_mock)

  add_library(
    mtt_runtime_benchmark_cuda_mock
    SHARED tests/runtime/runtime_benchmark_cuda_mock.cpp
  )
  target_compile_features(
    mtt_runtime_benchmark_cuda_mock PRIVATE cxx_std_20
  )
  mtt_enable_strict_warnings(mtt_runtime_benchmark_cuda_mock)

  add_executable(
    mtt_runtime_benchmark_tool_tests
    tests/runtime/runtime_benchmark_tool_tests.cpp
  )
  target_compile_features(
    mtt_runtime_benchmark_tool_tests PRIVATE cxx_std_20
  )
  target_include_directories(
    mtt_runtime_benchmark_tool_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  target_compile_definitions(
    mtt_runtime_benchmark_tool_tests
    PRIVATE
      "MTT_RUNTIME_BENCHMARK_MOCK_LIBRARY=\"$<TARGET_FILE:mtt_runtime_benchmark_mock>\""
      "MTT_RUNTIME_BENCHMARK_MOCK_NVML=\"$<TARGET_FILE:mtt_runtime_benchmark_nvml_mock>\""
      "MTT_RUNTIME_BENCHMARK_MOCK_CUDA=\"$<TARGET_FILE:mtt_runtime_benchmark_cuda_mock>\""
  )
  target_link_libraries(
    mtt_runtime_benchmark_tool_tests
    PRIVATE
      nlohmann_json::nlohmann_json
      OpenSSL::Crypto
      "${CMAKE_DL_LIBS}"
  )
  add_dependencies(
    mtt_runtime_benchmark_tool_tests
    mtt_runtime_benchmark_mock
    mtt_runtime_benchmark_nvml_mock
    mtt_runtime_benchmark_cuda_mock
  )
  mtt_enable_strict_warnings(mtt_runtime_benchmark_tool_tests)
  add_test(
    NAME runtime_benchmark_tool
    COMMAND mtt_runtime_benchmark_tool_tests
  )
  add_test(
    NAME runtime_benchmark_thresholds
    COMMAND
      mtt_runtime_benchmark_tool_tests
      --threshold-contract-only
  )
  add_test(
    NAME runtime_benchmark_help
    COMMAND mtt-runtime-benchmark --help
  )
  set_tests_properties(
    runtime_benchmark_tool
    runtime_benchmark_thresholds
    runtime_benchmark_help
    PROPERTIES TIMEOUT 60
  )

  if(MAGPIE_TTS_RT_BUILD_RUNTIME)
    add_executable(
      mtt_runtime_smoke_tool_tests
      tests/runtime/runtime_smoke_tool_tests.cpp
    )
    target_compile_features(
      mtt_runtime_smoke_tool_tests
      PRIVATE cxx_std_20
    )
    target_include_directories(
      mtt_runtime_smoke_tool_tests
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(
      mtt_runtime_smoke_tool_tests
      PRIVATE nlohmann_json::nlohmann_json "${CMAKE_DL_LIBS}"
    )
    mtt_enable_strict_warnings(mtt_runtime_smoke_tool_tests)
    add_test(
      NAME runtime_smoke_tool
      COMMAND mtt_runtime_smoke_tool_tests
    )
    add_test(
      NAME runtime_smoke_help
      COMMAND mtt-runtime-smoke --help
    )

    add_executable(
      mtt_engine_execution_tests
      tests/runtime/engine_execution_tests.cpp
      src/runtime/engine_execution.cpp
    )
    target_compile_features(
      mtt_engine_execution_tests
      PRIVATE cxx_std_20
    )
    target_include_directories(
      mtt_engine_execution_tests
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(
      mtt_engine_execution_tests
      PRIVATE
        MagpieTTSRT_cuda_cudart
        MagpieTTSRT_tensorrt_nvinfer
        magpie_tts_rt_contracts
    )
    mtt_enable_strict_warnings(mtt_engine_execution_tests)
    add_test(
      NAME engine_execution
      COMMAND mtt_engine_execution_tests
    )

    add_executable(
      mtt_fingerprint_tests
      tests/runtime/fingerprint_tests.cpp
      src/runtime/fingerprint.cpp
    )
    target_compile_features(mtt_fingerprint_tests PRIVATE cxx_std_20)
    target_include_directories(
      mtt_fingerprint_tests
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(
      mtt_fingerprint_tests
      PRIVATE
        MagpieTTSRT_cuda_cudart
        MagpieTTSRT_nvidia_nvml
        MagpieTTSRT_tensorrt_nvinfer
        magpie_tts_rt_contracts
    )
    mtt_enable_strict_warnings(mtt_fingerprint_tests)
    add_test(NAME fingerprint COMMAND mtt_fingerprint_tests)

    add_executable(mtt_c_header_test tests/c_api/c_header_test.c)
    target_link_libraries(mtt_c_header_test PRIVATE MagpieTTSRT::runtime)
    mtt_enable_strict_warnings(mtt_c_header_test)
    add_test(NAME c_header COMMAND mtt_c_header_test)

    add_executable(
      mtt_plugin_abi_tests
      tests/plugins/plugin_abi_tests.cpp
    )
    target_compile_features(mtt_plugin_abi_tests PRIVATE cxx_std_20)
    target_compile_definitions(
      mtt_plugin_abi_tests
      PRIVATE
        "MTT_PLUGIN_TEST_LIBRARY=\"$<TARGET_FILE:magpie_tts_rt_plugins>\""
    )
    target_include_directories(
      mtt_plugin_abi_tests
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(mtt_plugin_abi_tests PRIVATE "${CMAKE_DL_LIBS}")
    mtt_enable_strict_warnings(mtt_plugin_abi_tests)
    add_test(NAME plugin_abi COMMAND mtt_plugin_abi_tests)
  endif()

  if(MAGPIE_TTS_RT_BUILD_RUNTIME AND MAGPIE_TTS_RT_BUILD_GPU_TESTS)
    add_executable(
      mtt_plugin_owner_gpu_tests
      tests/runtime/plugin_owner_gpu_tests.cpp
      src/runtime/model_loader.cpp
      src/runtime/fingerprint.cpp
    )
    target_compile_features(
      mtt_plugin_owner_gpu_tests PRIVATE cxx_std_20
    )
    target_compile_definitions(
      mtt_plugin_owner_gpu_tests
      PRIVATE MAGPIE_TTS_RT_PLUGIN_OWNER_TESTING
    )
    target_include_directories(
      mtt_plugin_owner_gpu_tests
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(
      mtt_plugin_owner_gpu_tests
      PRIVATE
        MagpieTTSRT_cuda_cudart
        MagpieTTSRT_nvidia_nvml
        MagpieTTSRT_tensorrt_nvinfer
        magpie_tts_rt_contracts
        "${CMAKE_DL_LIBS}"
    )
    mtt_enable_strict_warnings(mtt_plugin_owner_gpu_tests)

    add_executable(
      mtt_multi_runtime_plugin_gpu_tests
      tests/c_api/multi_runtime_plugin_gpu_tests.cpp
    )
    target_compile_features(
      mtt_multi_runtime_plugin_gpu_tests PRIVATE cxx_std_20
    )
    target_include_directories(
      mtt_multi_runtime_plugin_gpu_tests
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(
      mtt_multi_runtime_plugin_gpu_tests
      PRIVATE MagpieTTSRT::runtime
    )
    mtt_enable_strict_warnings(
      mtt_multi_runtime_plugin_gpu_tests
    )

    add_executable(
      mtt_local_ar_plugin_gpu_tests
      plugins/local_ar_plugins.cu
      tests/plugins/local_ar_plugin_tests.cu
    )
    target_compile_features(mtt_local_ar_plugin_gpu_tests PRIVATE cxx_std_20)
    target_compile_definitions(
      mtt_local_ar_plugin_gpu_tests
      PRIVATE
        MAGPIE_TTS_RT_BUILDING_PLUGIN_LIBRARY
        MAGPIE_TTS_RT_PLUGIN_TESTING
    )
    target_include_directories(
      mtt_local_ar_plugin_gpu_tests
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/plugins"
        "${TENSORRT_INCLUDE_DIR}"
    )
    target_include_directories(
      mtt_local_ar_plugin_gpu_tests
      SYSTEM PRIVATE "${mtt_cutlass_SOURCE_DIR}/include"
    )
    target_link_libraries(
      mtt_local_ar_plugin_gpu_tests
      PRIVATE
        CUDA::cublas
        MagpieTTSRT_cuda_cudart
        MagpieTTSRT_tensorrt_nvinfer
    )
    target_compile_options(
      mtt_local_ar_plugin_gpu_tests
      PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
    )
    if(MAGPIE_TTS_RT_WARNINGS_AS_ERRORS)
      target_compile_options(
        mtt_local_ar_plugin_gpu_tests
        PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>"
      )
    endif()
    set_target_properties(
      mtt_local_ar_plugin_gpu_tests
      PROPERTIES
        CUDA_ARCHITECTURES 110
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED YES
    )
    add_test(
      NAME local_ar_plugin_gpu
      COMMAND mtt_local_ar_plugin_gpu_tests
    )
    set_tests_properties(
      local_ar_plugin_gpu
      PROPERTIES LABELS "gpu;thor"
    )

    add_executable(
      mtt_alignment_kernel_gpu_tests
      tests/runtime/alignment_kernel_gpu_tests.cu
      src/runtime/alignment_kernel.cu
    )
    target_compile_features(
      mtt_alignment_kernel_gpu_tests
      PRIVATE cxx_std_20
    )
    target_include_directories(
      mtt_alignment_kernel_gpu_tests
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(
      mtt_alignment_kernel_gpu_tests
      PRIVATE
        MagpieTTSRT_cuda_cudart
        magpie_tts_rt_contracts
    )
    target_compile_options(
      mtt_alignment_kernel_gpu_tests
      PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
    )
    if(MAGPIE_TTS_RT_WARNINGS_AS_ERRORS)
      target_compile_options(
        mtt_alignment_kernel_gpu_tests
        PRIVATE
          "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>"
      )
    endif()
    set_target_properties(
      mtt_alignment_kernel_gpu_tests
      PROPERTIES
        CUDA_ARCHITECTURES 110
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED YES
    )
    add_test(
      NAME alignment_kernel_gpu
      COMMAND mtt_alignment_kernel_gpu_tests
    )
    set_tests_properties(
      alignment_kernel_gpu
      PROPERTIES LABELS "gpu;thor"
    )

    add_executable(
      mtt_session_workspace_gpu_tests
      tests/runtime/session_workspace_gpu_tests.cpp
      src/runtime/cuda_memory.cpp
      src/runtime/engine_execution.cpp
      src/runtime/session_workspace.cpp
    )
    target_compile_features(
      mtt_session_workspace_gpu_tests
      PRIVATE cxx_std_20
    )
    target_include_directories(
      mtt_session_workspace_gpu_tests
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(
      mtt_session_workspace_gpu_tests
      PRIVATE
        MagpieTTSRT_cuda_cudart
        MagpieTTSRT_tensorrt_nvinfer
        magpie_tts_rt_contracts
    )
    mtt_enable_strict_warnings(mtt_session_workspace_gpu_tests)
    add_test(
      NAME session_workspace_gpu
      COMMAND
        mtt_session_workspace_gpu_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/manifest/fixtures/minimal-valid.json"
    )
    set_tests_properties(
      session_workspace_gpu
      PROPERTIES LABELS "gpu;thor"
    )

    add_executable(
      mtt_synthesis_kernels_gpu_tests
      tests/runtime/synthesis_kernels_gpu_tests.cu
      src/runtime/synthesis_kernels.cu
    )
    target_compile_features(
      mtt_synthesis_kernels_gpu_tests
      PRIVATE cxx_std_20
    )
    target_include_directories(
      mtt_synthesis_kernels_gpu_tests
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(
      mtt_synthesis_kernels_gpu_tests
      PRIVATE MagpieTTSRT_cuda_cudart
    )
    target_compile_options(
      mtt_synthesis_kernels_gpu_tests
      PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
    )
    if(MAGPIE_TTS_RT_WARNINGS_AS_ERRORS)
      target_compile_options(
        mtt_synthesis_kernels_gpu_tests
        PRIVATE
          "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>"
      )
    endif()
    set_target_properties(
      mtt_synthesis_kernels_gpu_tests
      PROPERTIES
        CUDA_ARCHITECTURES 110
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED YES
    )
    add_test(
      NAME synthesis_kernels_gpu
      COMMAND mtt_synthesis_kernels_gpu_tests
    )
    set_tests_properties(
      synthesis_kernels_gpu
      PROPERTIES LABELS "gpu;thor"
    )

    add_executable(
      mtt_fingerprint_gpu_tests
      tests/runtime/fingerprint_gpu_tests.cpp
      src/runtime/fingerprint.cpp
    )
    target_compile_features(mtt_fingerprint_gpu_tests PRIVATE cxx_std_20)
    target_include_directories(
      mtt_fingerprint_gpu_tests
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_libraries(
      mtt_fingerprint_gpu_tests
      PRIVATE
        MagpieTTSRT_cuda_cudart
        MagpieTTSRT_nvidia_nvml
        MagpieTTSRT_tensorrt_nvinfer
        magpie_tts_rt_contracts
    )
    mtt_enable_strict_warnings(mtt_fingerprint_gpu_tests)
    add_test(NAME fingerprint_gpu COMMAND mtt_fingerprint_gpu_tests)

    add_executable(mtt_c_api_gpu_tests tests/c_api/c_api_tests.cpp)
    target_compile_features(mtt_c_api_gpu_tests PRIVATE cxx_std_20)
    target_link_libraries(mtt_c_api_gpu_tests PRIVATE MagpieTTSRT::runtime)
    mtt_enable_strict_warnings(mtt_c_api_gpu_tests)
    add_test(NAME c_api_gpu COMMAND mtt_c_api_gpu_tests)
    set_tests_properties(c_api_gpu PROPERTIES LABELS "gpu;thor")
  endif()
endif()

configure_package_config_file(
  cmake/MagpieTTSRTConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/MagpieTTSRTConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MagpieTTSRT"
  PATH_VARS CMAKE_INSTALL_LIBDIR
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/MagpieTTSRTConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY ExactVersion
)

install(
  TARGETS
    mtt-runtime-benchmark
    mtt-validate-manifest
    mtt-validate-bundle
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(
  FILES
    include/magpie_tts_rt/magpie_tts_rt.h
    include/magpie_tts_rt/magpie_tts_rt_plugin.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/magpie_tts_rt"
)
install(
  FILES
    schemas/golden-fixture.schema.json
    schemas/plugin-build-receipt.schema.json
    schemas/runtime-bundle-manifest.schema.json
    schemas/runtime-benchmark-receipt.schema.json
    schemas/source-model-acceptance-receipt.schema.json
    schemas/thor-profile-evidence.schema.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/magpie-tts-rt/schemas"
)
install(
  FILES
    tools/frontend/__init__.py
    tools/frontend/japanese_frontend.py
    tools/frontend/japanese_frontend_server.py
    tools/frontend/requirements-aarch64-cp312.lock
  DESTINATION "${CMAKE_INSTALL_DATADIR}/magpie-tts-rt/tools/frontend"
)
install(
  PROGRAMS tools/frontend/generate_benchmark_corpus.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/magpie-tts-rt/tools/frontend"
)
install(
  PROGRAMS tools/plugins/create_plugin_build_receipt.py
  DESTINATION "${CMAKE_INSTALL_DATADIR}/magpie-tts-rt/tools/plugins"
)
install(
  FILES reference/oracle-lock.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/magpie-tts-rt/reference"
)
install(
  FILES
    reference/benchmark/README.md
    reference/benchmark/thor-ja-v1-source.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/magpie-tts-rt/reference/benchmark"
)
install(
  FILES LICENSE NOTICE
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/magpie-tts-rt"
)
install(
  FILES third_party/pytorch/LICENSE
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/magpie-tts-rt/third_party/pytorch"
)
install(
  FILES
    third_party/nlohmann-json/LICENSE.MIT
    third_party/nlohmann-json/SOURCE
  DESTINATION
    "${CMAKE_INSTALL_DATADIR}/doc/magpie-tts-rt/third_party/nlohmann-json"
)
if(MAGPIE_TTS_RT_BUILD_RUNTIME)
  install(
    FILES third_party/cutlass/LICENSE.txt
    DESTINATION
      "${CMAKE_INSTALL_DATADIR}/doc/magpie-tts-rt/third_party/cutlass"
  )
  install(
    FILES
      third_party/cuda/EULA.txt
      third_party/cuda/NOTICE
    DESTINATION
      "${CMAKE_INSTALL_DATADIR}/doc/magpie-tts-rt/third_party/cuda"
  )
  install(
    FILES
      third_party/nvidia-open-model-license/NVIDIA-Open-Model-License-Agreement-2025-10-24.pdf
      third_party/nvidia-open-model-license/NOTICE
      third_party/nvidia-open-model-license/SOURCE
    DESTINATION
      "${CMAKE_INSTALL_DATADIR}/doc/magpie-tts-rt/third_party/nvidia-open-model-license"
  )
  set(
    _mtt_pc_directory
    "/${CMAKE_INSTALL_LIBDIR}/pkgconfig"
  )
  file(
    RELATIVE_PATH
    MAGPIE_TTS_RT_PC_PREFIX_RELATIVE
    "${_mtt_pc_directory}"
    "/"
  )
  configure_file(
    cmake/magpie-tts-rt.pc.in
    "${CMAKE_CURRENT_BINARY_DIR}/magpie-tts-rt.pc"
    @ONLY
  )
  install(
    TARGETS magpie_tts_rt magpie_tts_rt_plugins
    EXPORT MagpieTTSRTTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )
  install(
    TARGETS mtt-runtime-smoke
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  )
  install(
    FILES
      "${CMAKE_CURRENT_BINARY_DIR}/MagpieTTSRTConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/MagpieTTSRTConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MagpieTTSRT"
  )
  install(
    EXPORT MagpieTTSRTTargets
    NAMESPACE MagpieTTSRT::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MagpieTTSRT"
  )
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/magpie-tts-rt.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
  )
endif()
