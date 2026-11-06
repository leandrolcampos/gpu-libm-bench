# Defines device image build rules.

include_guard(GLOBAL)
include(GPULibmBenchImageMetadata)

function(gpu_libm_bench_add_function library_target function_name source_file)
  target_sources(${library_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
  set_property(TARGET ${library_target} APPEND PROPERTY
    GPU_LIBM_BENCH_FUNCTIONS "${function_name}")
endfunction()

function(gpu_libm_bench_configure_library library_target)
  if(NOT GPU_LIBM_BENCH_BITCODE_PATH)
    message(FATAL_ERROR "GPU_LIBM_BENCH_BITCODE_PATH must be provided")
  endif()

  set(bitcode_path "${GPU_LIBM_BENCH_BITCODE_PATH}")
  if(NOT EXISTS "${bitcode_path}" OR IS_DIRECTORY "${bitcode_path}")
    message(FATAL_ERROR "Library bitcode was not found: ${bitcode_path}")
  endif()

  get_target_property(source_files ${library_target} SOURCES)
  set_property(SOURCE ${source_files} APPEND PROPERTY
    OBJECT_DEPENDS "${bitcode_path}")

  set_target_properties(${library_target} PROPERTIES OUTPUT_NAME math)
  gpu_libm_bench_configure_device_image(${library_target} "${GPU_LIBM_BENCH_LIBRARY}")
  target_compile_options(${library_target} PRIVATE
    "SHELL:-Xclang -mlink-builtin-bitcode -Xclang \"${bitcode_path}\"")

  get_target_property(directory ${library_target} RUNTIME_OUTPUT_DIRECTORY)
  gpu_libm_bench_write_device_image_metadata(${library_target} "${directory}")
endfunction()

function(gpu_libm_bench_configure_device_image image_target subdirectory)
  foreach(variable IN ITEMS SDK_ROOT ARCHITECTURE)
    if(NOT GPU_LIBM_BENCH_${variable})
      message(FATAL_ERROR "GPU_LIBM_BENCH_${variable} must be provided")
    endif()
  endforeach()

  target_include_directories(${image_target} PRIVATE "${PROJECT_SOURCE_DIR}/common")
  target_compile_options(${image_target} PRIVATE
    -nostdlibinc -nostdinc++ -fno-builtin -fno-exceptions -fno-rtti -O3)
  target_link_options(${image_target} PRIVATE -O3)

  if(GPU_LIBM_BENCH_TARGET_TRIPLE STREQUAL "nvptx64-nvidia-cuda")
    _gpu_libm_bench_configure_cuda_image(${image_target} "${subdirectory}")
  else()
    message(FATAL_ERROR
      "Unsupported device target: ${GPU_LIBM_BENCH_TARGET_TRIPLE}")
  endif()
endfunction()

function(_gpu_libm_bench_configure_cuda_image image_target subdirectory)
  set(directory
    "${CMAKE_BINARY_DIR}/cuda/${GPU_LIBM_BENCH_ARCHITECTURE}/${subdirectory}")
  get_target_property(output_name ${image_target} OUTPUT_NAME)
  set(ptx "${directory}/${output_name}.ptx")

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
      "${CMAKE_BINARY_DIR}/${output_name}.s" "${ptx}"
    BYPRODUCTS "${ptx}"
    VERBATIM)
endfunction()
