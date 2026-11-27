# Builds compile-time checks and device images for validation kernels.

add_library(fpbits-check OBJECT FPBitsTest.cpp)
target_include_directories(fpbits-check PRIVATE "${PROJECT_SOURCE_DIR}/common")
target_compile_options(fpbits-check PRIVATE
  -nostdlibinc -nostdinc++ "-march=${GPU_LIBM_BENCH_ARCHITECTURE}")

add_executable(timer-image TimerTest.cpp)
set_target_properties(timer-image PROPERTIES OUTPUT_NAME timer)
gpu_libm_bench_configure_device_image(timer-image "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(latency-image LatencyTest.cpp)
set_target_properties(latency-image PROPERTIES OUTPUT_NAME latency)
gpu_libm_bench_configure_device_image(latency-image "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(throughput-image ThroughputTest.cpp)
set_target_properties(throughput-image PROPERTIES OUTPUT_NAME throughput)
gpu_libm_bench_configure_device_image(throughput-image "${CMAKE_CURRENT_BINARY_DIR}")
