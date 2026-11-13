# Records device image identity and compilation settings.

include_guard(GLOBAL)

function(gpu_libm_bench_write_device_image_metadata image_target directory)
  set(library "${GPU_LIBM_BENCH_LIBRARY}")
  get_target_property(backend ${image_target} GPU_LIBM_BENCH_BACKEND)
  set(target_triple "${GPU_LIBM_BENCH_TARGET_TRIPLE}")
  set(architecture "${GPU_LIBM_BENCH_ARCHITECTURE}")
  set(image_path "$<TARGET_FILE:${image_target}>")

  set(compiler "${CMAKE_CXX_COMPILER}")
  set(llvm_root "${GPU_LIBM_BENCH_LLVM_ROOT}")
  set(sdk_root "${GPU_LIBM_BENCH_SDK_ROOT}")
  set(sdk_version "${GPU_LIBM_BENCH_SDK_VERSION}")

  set(compile_commands "${CMAKE_BINARY_DIR}/compile_commands.json")
  set(compile_flags "${CMAKE_CXX_FLAGS}")
  set(link_flags "${CMAKE_EXE_LINKER_FLAGS}")

  set(metadata "{}")
  foreach(field IN ITEMS library backend target_triple architecture image_path
      compiler llvm_root sdk_root sdk_version compile_commands
      compile_flags link_flags)
    _gpu_libm_bench_json_string(encoded "${${field}}")
    string(JSON metadata SET "${metadata}" "${field}" "${encoded}")
  endforeach()

  set(encoded null)
  if(GPU_LIBM_BENCH_BITCODE_PATH)
    _gpu_libm_bench_json_string(encoded "${GPU_LIBM_BENCH_BITCODE_PATH}")
  endif()
  string(JSON metadata SET "${metadata}" bitcode_path "${encoded}")

  get_target_property(ptx ${image_target} GPU_LIBM_BENCH_PTX)
  if(ptx)
    _gpu_libm_bench_json_string(encoded "${ptx}")
    string(JSON metadata SET "${metadata}" ptx "${encoded}")
  endif()

  foreach(field IN ITEMS functions sources compile_options link_options
      include_directories)
    if(field STREQUAL "functions")
      get_target_property(values ${image_target} GPU_LIBM_BENCH_FUNCTIONS)
    else()
      string(TOUPPER "${field}" property)
      get_target_property(values ${image_target} "${property}")
    endif()
    if(NOT values)
      set(values "")
    endif()

    _gpu_libm_bench_json_array(encoded ${values})
    string(JSON metadata SET "${metadata}" "${field}" "${encoded}")
  endforeach()

  execute_process(COMMAND "${compiler}" --version OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr COMMAND_ERROR_IS_FATAL ANY)
  string(STRIP "${stdout}${stderr}" version)
  _gpu_libm_bench_json_string(encoded "${version}")
  string(JSON metadata SET "${metadata}" compiler_version "${encoded}")

  set(build_plan "${directory}/build-plan.json")
  get_target_property(output_name ${image_target} OUTPUT_NAME)
  set(manifest_path "${directory}/${output_name}.manifest.json")
  file(GENERATE OUTPUT "${build_plan}" CONTENT "${metadata}\n")
  set_property(TARGET ${image_target} APPEND PROPERTY LINK_DEPENDS "${build_plan}")

  add_custom_command(TARGET ${image_target} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${build_plan}" "${manifest_path}"
    BYPRODUCTS "${manifest_path}"
    VERBATIM)
endfunction()

function(_gpu_libm_bench_json_string output value)
  string(REPLACE "\\" "\\\\" value "${value}")
  string(REPLACE "\"" "\\\"" value "${value}")
  string(REPLACE "\n" "\\n" value "${value}")
  string(REPLACE "\r" "\\r" value "${value}")
  string(REPLACE "\t" "\\t" value "${value}")
  set(${output} "\"${value}\"" PARENT_SCOPE)
endfunction()

function(_gpu_libm_bench_json_array output)
  set(result "[]")
  set(index 0)

  foreach(value IN LISTS ARGN)
    _gpu_libm_bench_json_string(encoded "${value}")
    string(JSON result SET "${result}" ${index} "${encoded}")
    math(EXPR index "${index} + 1")
  endforeach()

  set(${output} "${result}" PARENT_SCOPE)
endfunction()
