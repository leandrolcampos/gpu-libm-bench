# Builds device images for kernel libraries.

cmake_minimum_required(VERSION 3.31)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(GPULibmBenchToolchain)

project(GPULibmBenchKernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GPULibmBenchImageRules)

if(GPU_LIBM_BENCH_LIBRARY STREQUAL "support")
  add_subdirectory(support)
elseif(GPU_LIBM_BENCH_LIBRARY)
  add_subdirectory(libm)
endif()

option(BUILD_TESTING "Build device validation kernels" OFF)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
