# Builds the CUDA backend.

cmake_minimum_required(VERSION 3.31)

project(GPULibmBenchCUDABackend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(GPULibmBenchDependencies)

add_subdirectory(../common "${CMAKE_CURRENT_BINARY_DIR}/common")
add_subdirectory(lib)
add_subdirectory(python)

option(BUILD_TESTING "Build native backend tests" OFF)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(../common/tests "${CMAKE_CURRENT_BINARY_DIR}/tests/common")
endif()
