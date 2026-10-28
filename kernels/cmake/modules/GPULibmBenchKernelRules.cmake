# Defines device image build rules for math libraries.

include_guard(GLOBAL)
include(GPULibmBenchKernelMetadata)

function(gpu_libm_bench_add_function library function_name source_file)
  target_sources(${library} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
  set_property(TARGET ${library} APPEND PROPERTY
    GPU_LIBM_BENCH_FUNCTIONS "${function_name}")
endfunction()

function(_gpu_libm_bench_configure_cuda_image library)
  set(directory "${CMAKE_BINARY_DIR}/cuda/${GPU_LIBM_BENCH_ARCHITECTURE}/${library}")
  set(ptx "${directory}/math.ptx")
  set_target_properties(${library} PROPERTIES
    SUFFIX .cubin RUNTIME_OUTPUT_DIRECTORY "${directory}"
    GPU_LIBM_BENCH_BACKEND cuda GPU_LIBM_BENCH_PTX "${ptx}")
  target_compile_options(${library} PRIVATE
    "-march=${GPU_LIBM_BENCH_ARCHITECTURE}")
  target_link_options(${library} PRIVATE
    "-march=${GPU_LIBM_BENCH_ARCHITECTURE}"
    "--cuda-path=${GPU_LIBM_BENCH_SDK_ROOT}"
    -Wl,--save-temps)
  add_custom_command(TARGET ${library} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${CMAKE_BINARY_DIR}/math.s" "${ptx}"
    BYPRODUCTS "${ptx}"
    VERBATIM)
endfunction()

function(gpu_libm_bench_configure_library library)
  foreach(variable IN ITEMS SDK_ROOT ARCHITECTURE BITCODE_FILE)
    if(NOT GPU_LIBM_BENCH_${variable})
      message(FATAL_ERROR "GPU_LIBM_BENCH_${variable} must be provided")
    endif()
  endforeach()
  set(bitcode_file "${GPU_LIBM_BENCH_BITCODE_FILE}")
  if(NOT EXISTS "${bitcode_file}" OR IS_DIRECTORY "${bitcode_file}")
    message(FATAL_ERROR "Library bitcode was not found: ${bitcode_file}")
  endif()

  get_target_property(source_files ${library} SOURCES)
  set_property(SOURCE ${source_files} APPEND PROPERTY
    OBJECT_DEPENDS "${bitcode_file}")

  set_target_properties(${library} PROPERTIES OUTPUT_NAME math)
  target_include_directories(${library} PRIVATE "${PROJECT_SOURCE_DIR}/common")
  target_compile_options(${library} PRIVATE
    -nostdlibinc -nostdinc++ -fno-builtin -fno-exceptions -fno-rtti -O3
    "SHELL:-Xclang -mlink-builtin-bitcode -Xclang \"${bitcode_file}\"")
  target_link_options(${library} PRIVATE -O3)

  if(GPU_LIBM_BENCH_TARGET_TRIPLE STREQUAL "nvptx64-nvidia-cuda")
    _gpu_libm_bench_configure_cuda_image(${library})
  else()
    message(FATAL_ERROR
      "Unsupported device target: ${GPU_LIBM_BENCH_TARGET_TRIPLE}")
  endif()

  get_target_property(directory ${library} RUNTIME_OUTPUT_DIRECTORY)
  gpu_libm_bench_write_kernel_metadata(${library} "${directory}")
endfunction()
