# Defines device image build rules for math libraries.

include_guard(GLOBAL)
include(GPULibmBenchImageMetadata)

function(gpu_libm_bench_add_function image_target function_name source_file)
  target_sources(${image_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
  set_property(TARGET ${image_target} APPEND PROPERTY
    GPU_LIBM_BENCH_FUNCTIONS "${function_name}")
endfunction()

function(_gpu_libm_bench_configure_cuda_image image_target)
  set(directory
    "${CMAKE_BINARY_DIR}/cuda/${GPU_LIBM_BENCH_ARCHITECTURE}/${GPU_LIBM_BENCH_LIBRARY}")
  set(ptx "${directory}/math.ptx")
  set_target_properties(${image_target} PROPERTIES
    SUFFIX .cubin RUNTIME_OUTPUT_DIRECTORY "${directory}"
    GPU_LIBM_BENCH_BACKEND cuda GPU_LIBM_BENCH_PTX "${ptx}")
  target_compile_options(${image_target} PRIVATE
    "-march=${GPU_LIBM_BENCH_ARCHITECTURE}")
  target_link_options(${image_target} PRIVATE
    "-march=${GPU_LIBM_BENCH_ARCHITECTURE}"
    "--cuda-path=${GPU_LIBM_BENCH_SDK_ROOT}"
    -Wl,--save-temps)
  add_custom_command(TARGET ${image_target} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${CMAKE_BINARY_DIR}/math.s" "${ptx}"
    BYPRODUCTS "${ptx}"
    VERBATIM)
endfunction()

function(gpu_libm_bench_configure_image image_target)
  foreach(variable IN ITEMS SDK_ROOT ARCHITECTURE BITCODE_FILE)
    if(NOT GPU_LIBM_BENCH_${variable})
      message(FATAL_ERROR "GPU_LIBM_BENCH_${variable} must be provided")
    endif()
  endforeach()
  set(bitcode_file "${GPU_LIBM_BENCH_BITCODE_FILE}")
  if(NOT EXISTS "${bitcode_file}" OR IS_DIRECTORY "${bitcode_file}")
    message(FATAL_ERROR "Library bitcode was not found: ${bitcode_file}")
  endif()

  get_target_property(source_files ${image_target} SOURCES)
  set_property(SOURCE ${source_files} APPEND PROPERTY
    OBJECT_DEPENDS "${bitcode_file}")

  set_target_properties(${image_target} PROPERTIES OUTPUT_NAME math)
  target_include_directories(${image_target} PRIVATE "${PROJECT_SOURCE_DIR}/common")
  target_compile_options(${image_target} PRIVATE
    -nostdlibinc -nostdinc++ -fno-builtin -fno-exceptions -fno-rtti -O3
    "SHELL:-Xclang -mlink-builtin-bitcode -Xclang \"${bitcode_file}\"")
  target_link_options(${image_target} PRIVATE -O3)

  if(GPU_LIBM_BENCH_TARGET_TRIPLE STREQUAL "nvptx64-nvidia-cuda")
    _gpu_libm_bench_configure_cuda_image(${image_target})
  else()
    message(FATAL_ERROR
      "Unsupported device target: ${GPU_LIBM_BENCH_TARGET_TRIPLE}")
  endif()

  get_target_property(directory ${image_target} RUNTIME_OUTPUT_DIRECTORY)
  gpu_libm_bench_write_image_metadata(${image_target} "${directory}")
endfunction()
