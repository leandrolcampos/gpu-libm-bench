# Builds the LLVM-libm device image.

add_executable(llvm-libm-image)
gpu_libm_bench_add_function(llvm-libm-image expf Expf.cpp)
gpu_libm_bench_configure_image(llvm-libm-image)
