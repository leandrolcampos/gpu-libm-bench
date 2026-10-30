# Builds device validation images.

add_executable(timer-image Timer.cpp)
set_target_properties(timer-image PROPERTIES OUTPUT_NAME timer)
gpu_libm_bench_configure_device_image(timer-image tests)

add_executable(latency-image Latency.cpp FPBits.cpp)
set_target_properties(latency-image PROPERTIES OUTPUT_NAME latency)
gpu_libm_bench_configure_device_image(latency-image tests)
