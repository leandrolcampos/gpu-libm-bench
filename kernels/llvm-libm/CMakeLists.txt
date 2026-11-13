# Builds the LLVM-libm device image.

if(NOT GPU_LIBM_BENCH_BITCODE_PATH)
  message(FATAL_ERROR "GPU_LIBM_BENCH_BITCODE_PATH must be provided for LLVM-libm")
endif()

add_executable(llvm-libm-image)
set_target_properties(llvm-libm-image PROPERTIES OUTPUT_NAME math)

gpu_libm_bench_add_function(llvm-libm-image expf Expf.cpp)

gpu_libm_bench_configure_library(llvm-libm-image)
