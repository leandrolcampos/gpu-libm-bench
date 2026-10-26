// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Python bindings for the CUDA backend.

#include "DeviceContext.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace gpu_libm_bench;

PYBIND11_MODULE(_cuda, Module) {
  Module.def("device_count", &getDeviceCount);
  Module.def(
      "get_device_info",
      [](int DeviceID) {
        DeviceInfo Info = getDeviceInfo(DeviceID);
        py::dict Properties;
        Properties["device_id"] = Info.DeviceID;
        Properties["name"] = Info.Name;
        Properties["architecture"] = Info.Architecture;
        Properties["total_memory_bytes"] = Info.TotalMemoryBytes;
        return Properties;
      },
      py::arg("device_id") = 0);
}
