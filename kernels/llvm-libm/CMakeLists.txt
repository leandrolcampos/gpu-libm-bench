# Builds the LLVM-libm device image.

add_executable(llvm-libm)
gpu_libm_bench_add_function(llvm-libm expf Expf.cpp)
gpu_libm_bench_configure_library(llvm-libm)
