# Resolves the Python binding and CUDA Driver API dependencies of the backend.

include_guard(GLOBAL)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_cmake_directory
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

find_package(pybind11 3 CONFIG REQUIRED HINTS "${pybind11_cmake_directory}")
find_package(CUDAToolkit REQUIRED)
