# Builds device images for validation kernels.

add_executable(timer-image Timer.cpp)
set_target_properties(timer-image PROPERTIES OUTPUT_NAME timer)
gpu_libm_bench_configure_device_image(timer-image "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(latency-image Latency.cpp FPBits.cpp)
set_target_properties(latency-image PROPERTIES OUTPUT_NAME latency)
gpu_libm_bench_configure_device_image(latency-image "${CMAKE_CURRENT_BINARY_DIR}")
