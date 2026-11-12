# Defines the native CUDA backend library.

add_library(GPULibmBenchCUDA STATIC
  Context.cpp
  Memory.cpp
  Module.cpp
  Utils.cpp)

set_target_properties(GPULibmBenchCUDA PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

target_include_directories(GPULibmBenchCUDA
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}")

target_compile_definitions(GPULibmBenchCUDA
  PUBLIC
    GPU_LIBM_BENCH_BACKEND=cuda)

target_link_libraries(GPULibmBenchCUDA
  PUBLIC
    GPULibmBenchCommon
  PRIVATE
    CUDA::cuda_driver)

target_compile_options(GPULibmBenchCUDA
  PRIVATE
    "$<$<CXX_COMPILER_ID:Clang,GNU>:-fno-exceptions>")
