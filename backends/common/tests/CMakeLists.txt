# Builds CPU tests for shared native backend support.

cmake_minimum_required(VERSION 3.31)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(GPULibmBenchCommonTests LANGUAGES CXX)

  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  add_subdirectory(.. "${CMAKE_CURRENT_BINARY_DIR}/common")

  enable_testing()
endif()

add_executable(ExpectedTest
  ExpectedTest.cpp)

target_link_libraries(ExpectedTest
  PRIVATE
    GPULibmBenchCommon)

target_compile_options(ExpectedTest
  PRIVATE
    "$<$<CXX_COMPILER_ID:Clang,GNU>:-fno-exceptions>")

add_test(NAME ExpectedTest COMMAND ExpectedTest)
