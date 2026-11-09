# Discovers LLVM tools and library directories.

cmake_minimum_required(VERSION 3.31)

project(GPULibmBenchLLVMProbe LANGUAGES C CXX)

find_package(LLVM CONFIG REQUIRED PATHS "${LLVM_ROOT}" NO_DEFAULT_PATH)

find_program(CLANG_EXECUTABLE NAMES clang++
  PATHS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_program(LLVM_DIS_EXECUTABLE NAMES llvm-dis
  PATHS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_program(LLVM_NM_EXECUTABLE NAMES llvm-nm
  PATHS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)

function(json_string output value)
  string(REPLACE "\\" "\\\\" value "${value}")
  string(REPLACE "\"" "\\\"" value "${value}")
  set(${output} "\"${value}\"" PARENT_SCOPE)
endfunction()

set(clang "${CLANG_EXECUTABLE}")
set(llvm_dis "${LLVM_DIS_EXECUTABLE}")
set(llvm_nm "${LLVM_NM_EXECUTABLE}")
set(metadata "{}")
foreach(key IN ITEMS clang llvm_dis llvm_nm)
  json_string(encoded "${${key}}")
  string(JSON metadata SET "${metadata}" "${key}" "${encoded}")
endforeach()

set(library_dirs "[]")
set(index 0)
foreach(directory IN LISTS LLVM_LIBRARY_DIRS)
  json_string(encoded "${directory}")
  string(JSON library_dirs SET "${library_dirs}" ${index} "${encoded}")
  math(EXPR index "${index} + 1")
endforeach()
string(JSON metadata SET "${metadata}" library_dirs "${library_dirs}")

file(WRITE "${CMAKE_BINARY_DIR}/llvm-toolchain.json" "${metadata}\n")
