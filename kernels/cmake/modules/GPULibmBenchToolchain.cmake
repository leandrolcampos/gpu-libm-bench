# Configures the device compilation toolchain.

include_guard(GLOBAL)

foreach(variable IN ITEMS CLANG TARGET_TRIPLE)
  if(NOT GPU_LIBM_BENCH_${variable})
    message(FATAL_ERROR "GPU_LIBM_BENCH_${variable} must be provided")
  endif()
endforeach()

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_CXX_COMPILER "${GPU_LIBM_BENCH_CLANG}")
set(CMAKE_CXX_COMPILER_TARGET "${GPU_LIBM_BENCH_TARGET_TRIPLE}")
set(CMAKE_CXX_FLAGS_INIT "-flto -ffreestanding -nogpulib")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
