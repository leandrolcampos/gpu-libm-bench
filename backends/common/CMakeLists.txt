# Defines shared support for native GPU backends.

add_library(GPULibmBenchCommon INTERFACE)

target_compile_features(GPULibmBenchCommon
  INTERFACE
    cxx_std_17)

target_include_directories(GPULibmBenchCommon
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}")
