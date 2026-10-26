// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Python bindings for the CUDA backend.

#include "DeviceContext.hpp"

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <utility>

namespace py = pybind11;
using namespace gpu_libm_bench;

template <typename T> static T unwrap(Expected<T> Result) {
  if (Result)
    return *std::move(Result);
  if (Result.error().Kind == ErrorKind::InvalidArgument)
    throw py::value_error(Result.error().Message);
  throw std::runtime_error(Result.error().Message);
}

PYBIND11_MODULE(_cuda, Module) {
  Module.def("device_count", [] { return unwrap(getDeviceCount()); });
  Module.def(
      "get_device_info",
      [](int DeviceID) {
        DeviceInfo Info = unwrap(getDeviceInfo(DeviceID));
        py::dict Properties;
        Properties["device_id"] = Info.DeviceID;
        Properties["name"] = Info.Name;
        Properties["architecture"] = Info.Architecture;
        Properties["total_memory_bytes"] = Info.TotalMemoryBytes;
        return Properties;
      },
      py::arg("device_id") = 0);
}
