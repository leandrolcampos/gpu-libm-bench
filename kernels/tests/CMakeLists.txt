# Builds device validation images.

add_executable(timer-image Timer.cpp)
set_target_properties(timer-image PROPERTIES OUTPUT_NAME timer)
gpu_libm_bench_configure_device_image(timer-image tests)
