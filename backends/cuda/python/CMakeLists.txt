# Exposes the CUDA backend as a private Python extension.

pybind11_add_module(_cuda MODULE
  Bindings.cpp)

target_link_libraries(_cuda
  PRIVATE
    GPULibmBenchCUDA)
