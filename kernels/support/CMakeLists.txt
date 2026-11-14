# Builds device images for backend support kernels.

add_executable(support-image Profiling.cpp)
set_target_properties(support-image PROPERTIES OUTPUT_NAME support)

gpu_libm_bench_configure_library(support-image)
