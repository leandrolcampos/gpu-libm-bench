# Builds a math library device image from the selected kernel sources.

if(NOT GPU_LIBM_BENCH_BITCODE_PATH)
  message(FATAL_ERROR
    "GPU_LIBM_BENCH_BITCODE_PATH must be provided for a math library")
endif()
if(NOT GPU_LIBM_BENCH_FUNCTIONS)
  message(FATAL_ERROR
    "No kernel functions selected for ${GPU_LIBM_BENCH_LIBRARY}")
endif()

add_executable(libm-image)
set_target_properties(libm-image PROPERTIES OUTPUT_NAME math)
target_include_directories(libm-image PRIVATE "${CMAKE_BINARY_DIR}/generated")

set(source_expf Expf.cpp)

foreach(function_name IN LISTS GPU_LIBM_BENCH_FUNCTIONS)
  if(NOT DEFINED source_${function_name})
    message(FATAL_ERROR
      "No kernel source registered for function: ${function_name}")
  endif()
  gpu_libm_bench_add_function(libm-image "${function_name}"
    "${source_${function_name}}")
endforeach()

# Internal controls are available in every math library image.
target_sources(libm-image PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/Idf16.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/Idf.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/Id.cpp")

gpu_libm_bench_configure_library(libm-image)
