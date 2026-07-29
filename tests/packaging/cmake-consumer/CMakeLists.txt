cmake_minimum_required(VERSION 3.24)

project(MagpieTTSRTPackageConsumer LANGUAGES C)

find_package(
  MagpieTTSRT 0.1.0 EXACT
  REQUIRED
  COMPONENTS runtime
)

add_executable(mtt_cmake_package_consumer main.c)
target_link_libraries(
  mtt_cmake_package_consumer
  PRIVATE MagpieTTSRT::runtime
)
