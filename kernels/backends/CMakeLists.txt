add_executable(profiling-image Profiling.cpp)
set_target_properties(profiling-image PROPERTIES OUTPUT_NAME profiling)
gpu_libm_bench_configure_device_image(profiling-image "${CMAKE_CURRENT_BINARY_DIR}")
