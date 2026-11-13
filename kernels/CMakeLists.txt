# Builds device images for GPU math libraries.

cmake_minimum_required(VERSION 3.31)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(GPULibmBenchToolchain)

project(GPULibmBenchKernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GPULibmBenchImageRules)

add_subdirectory(backends)

set(supported_libraries llvm-libm)
if(GPU_LIBM_BENCH_LIBRARY)
  if(NOT GPU_LIBM_BENCH_LIBRARY IN_LIST supported_libraries)
    message(FATAL_ERROR "Unsupported math library: ${GPU_LIBM_BENCH_LIBRARY}")
  endif()

  add_subdirectory("${GPU_LIBM_BENCH_LIBRARY}")
endif()

option(BUILD_TESTING "Build device validation kernels" OFF)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
