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

add_library(
  magpie_tts_rt_contracts
  STATIC
    src/manifest/manifest.cpp
    src/bundle/bundle.cpp
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
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
    )
    if(MAGPIE_TTS_RT_WARNINGS_AS_ERRORS)
      target_compile_options("${target}" PRIVATE -Werror)
    endif()
  endif()
endfunction()

mtt_enable_strict_warnings(magpie_tts_rt_contracts)

if(MAGPIE_TTS_RT_BUILD_RUNTIME)
  find_package(CUDAToolkit 13 REQUIRED)
  find_library(
    MAGPIE_TTS_RT_CUDART_LIBRARY
    NAMES cudart libcudart.so.13
    NAMES_PER_DIR
    HINTS "${CUDAToolkit_LIBRARY_DIR}"
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
  mtt_require_elf_soname(
    "${MAGPIE_TTS_RT_CUDART_LIBRARY}"
    "libcudart.so.13"
  )
  mtt_require_elf_soname(
    "${TENSORRT_NVINFER_LIBRARY}"
    "libnvinfer.so.10"
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

  add_library(magpie_tts_rt SHARED src/api/magpie_tts_rt.cpp)
  add_library(MagpieTTSRT::runtime ALIAS magpie_tts_rt)

  target_compile_features(magpie_tts_rt PRIVATE cxx_std_20)
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
      MagpieTTSRT_tensorrt_nvinfer
      magpie_tts_rt_contracts
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
elseif(MAGPIE_TTS_RT_BUILD_GPU_TESTS)
  message(
    FATAL_ERROR
    "MAGPIE_TTS_RT_BUILD_GPU_TESTS requires MAGPIE_TTS_RT_BUILD_RUNTIME"
  )
endif()

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
  set_tests_properties(manifest PROPERTIES TIMEOUT 180)

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
  set_tests_properties(bundle PROPERTIES TIMEOUT 180)

  if(MAGPIE_TTS_RT_BUILD_RUNTIME)
    add_executable(mtt_c_header_test tests/c_api/c_header_test.c)
    target_link_libraries(mtt_c_header_test PRIVATE MagpieTTSRT::runtime)
    mtt_enable_strict_warnings(mtt_c_header_test)
    add_test(NAME c_header COMMAND mtt_c_header_test)
  endif()

  if(MAGPIE_TTS_RT_BUILD_RUNTIME AND MAGPIE_TTS_RT_BUILD_GPU_TESTS)
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
  TARGETS mtt-validate-manifest mtt-validate-bundle
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(
  FILES include/magpie_tts_rt/magpie_tts_rt.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/magpie_tts_rt"
)
install(
  FILES schemas/runtime-bundle-manifest.schema.json
  DESTINATION "${CMAKE_INSTALL_DATADIR}/magpie-tts-rt/schemas"
)
install(
  FILES LICENSE NOTICE
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/magpie-tts-rt"
)
if(MAGPIE_TTS_RT_BUILD_RUNTIME)
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
    TARGETS magpie_tts_rt
    EXPORT MagpieTTSRTTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
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
