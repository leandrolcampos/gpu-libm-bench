# Discovers CUDA Toolkit paths for kernel compilation.

cmake_minimum_required(VERSION 3.31)

project(GPULibmBenchCUDAProbe LANGUAGES NONE)

find_package(CUDAToolkit REQUIRED)
find_program(PTXAS_EXECUTABLE
  NAMES ptxas
  PATHS "${CUDAToolkit_BIN_DIR}"
  NO_DEFAULT_PATH
  REQUIRED)

set(root "${CUDAToolkit_LIBRARY_ROOT}")
set(version "${CUDAToolkit_VERSION}")
set(ptxas "${PTXAS_EXECUTABLE}")
set(metadata "{}")

foreach(key IN ITEMS root version ptxas)
  string(REPLACE "\\" "\\\\" value "${${key}}")
  string(REPLACE "\"" "\\\"" value "${value}")
  string(JSON metadata SET "${metadata}" "${key}" "\"${value}\"")
endforeach()

file(WRITE "${CMAKE_BINARY_DIR}/cuda-toolkit.json" "${metadata}\n")
