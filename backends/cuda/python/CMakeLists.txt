# Exposes the CUDA backend as a private Python extension.

pybind11_add_module(_cuda MODULE
  ../../common/python/Bindings.cpp)

target_compile_definitions(_cuda
  PRIVATE
    GPU_LIBM_BENCH_PYTHON_MODULE=_cuda)

target_link_libraries(_cuda
  PRIVATE
    GPULibmBenchCUDA)
