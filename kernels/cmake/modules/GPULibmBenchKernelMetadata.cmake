# Records device image identity and compilation settings.

include_guard(GLOBAL)

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

function(gpu_libm_bench_write_kernel_metadata library directory)
  get_target_property(backend ${library} GPU_LIBM_BENCH_BACKEND)
  set(target_triple "${GPU_LIBM_BENCH_TARGET_TRIPLE}")
  set(architecture "${GPU_LIBM_BENCH_ARCHITECTURE}")
  set(image "$<TARGET_FILE:${library}>")
  set(compiler "${CMAKE_CXX_COMPILER}")
  set(llvm_root "${GPU_LIBM_BENCH_LLVM_ROOT}")
  set(sdk_root "${GPU_LIBM_BENCH_SDK_ROOT}")
  set(sdk_version "${GPU_LIBM_BENCH_SDK_VERSION}")
  set(bitcode_file "${GPU_LIBM_BENCH_BITCODE_FILE}")
  set(compile_commands "${CMAKE_BINARY_DIR}/compile_commands.json")
  set(compile_flags "${CMAKE_CXX_FLAGS}")
  set(link_flags "${CMAKE_EXE_LINKER_FLAGS}")

  set(metadata "{}")
  foreach(field IN ITEMS library backend target_triple architecture image
      compiler llvm_root sdk_root sdk_version bitcode_file compile_commands
      compile_flags link_flags)
    _gpu_libm_bench_json_string(encoded "${${field}}")
    string(JSON metadata SET "${metadata}" "${field}" "${encoded}")
  endforeach()
  get_target_property(ptx ${library} GPU_LIBM_BENCH_PTX)
  if(ptx)
    _gpu_libm_bench_json_string(encoded "${ptx}")
    string(JSON metadata SET "${metadata}" ptx "${encoded}")
  endif()
  foreach(field IN ITEMS functions sources compile_options link_options
      include_directories)
    if(field STREQUAL "functions")
      get_target_property(values ${library} GPU_LIBM_BENCH_FUNCTIONS)
    else()
      string(TOUPPER "${field}" property)
      get_target_property(values ${library} "${property}")
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
  set(manifest "${directory}/math.manifest.json")
  file(GENERATE OUTPUT "${build_plan}" CONTENT "${metadata}\n")
  set_property(TARGET ${library} APPEND PROPERTY LINK_DEPENDS "${build_plan}")
  add_custom_command(TARGET ${library} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${build_plan}" "${manifest}"
    BYPRODUCTS "${manifest}"
    VERBATIM)
endfunction()
