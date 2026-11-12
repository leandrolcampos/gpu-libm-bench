// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Python bindings for native GPU execution backends.

#include "Context.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifndef GPU_LIBM_BENCH_PYTHON_MODULE
#error "GPU_LIBM_BENCH_PYTHON_MODULE must be defined"
#endif

namespace py = pybind11;
using namespace gpu_libm_bench;
using namespace gpu_libm_bench::GPU_LIBM_BENCH_BACKEND;

[[noreturn]] static void raiseError(const Error &Failure) {
  if (Failure.Kind == ErrorKind::InvalidArgument)
    throw py::value_error(Failure.Message);

  throw std::runtime_error(Failure.Message);
}

template <typename T> static T unwrap(Expected<T> Result) {
  if (!Result)
    raiseError(Result.error());

  return *std::move(Result);
}

static void unwrap(const Error &Failure) {
  if (Failure)
    raiseError(Failure);
}

static py::array requireArray(py::handle Object, bool Writable) {
  if (!py::isinstance<py::array>(Object))
    throw py::type_error("copy endpoint must be a NumPy array");

  auto Array = py::reinterpret_borrow<py::array>(Object);
  auto Flags = static_cast<unsigned int>(Array.flags());

  if (Array.ndim() != 1 ||
      !(Flags & static_cast<unsigned int>(py::array::c_style)))
    throw py::value_error(
        "copy endpoint must be a contiguous one-dimensional array");

  if (Writable && !Array.writeable())
    throw py::value_error("copy destination must be writable");

  if (Array.dtype().attr("hasobject").cast<bool>())
    throw py::type_error("copy endpoint must not contain Python objects");

  return Array;
}

PYBIND11_MODULE(GPU_LIBM_BENCH_PYTHON_MODULE, Module) {
  Module.def("device_count", [] { return unwrap(getDeviceCount()); });

  Module.def(
      "get_device_info",
      [](int DeviceID) {
        DeviceInfoTy DeviceInfo = unwrap(getDeviceInfo(DeviceID));

        py::dict TimerProperties;
        TimerProperties["source"] = DeviceInfo.TimerInfo.Source;
        TimerProperties["counter_bits"] = DeviceInfo.TimerInfo.CounterBits;
        TimerProperties["frequency_hz"] = DeviceInfo.TimerInfo.FrequencyHz;

        py::dict Properties;
        Properties["id"] = DeviceInfo.ID;
        Properties["name"] = DeviceInfo.Name;
        Properties["architecture"] = DeviceInfo.Architecture;
        Properties["total_memory_bytes"] = DeviceInfo.TotalMemoryBytes;
        Properties["timer_info"] = TimerProperties;

        return Properties;
      },
      py::arg("device_id") = 0);

  py::class_<DeviceBuffer>(Module, "DeviceBuffer")
      .def_property_readonly("nbytes", &DeviceBuffer::getByteCount);

  // NOLINTNEXTLINE(bugprone-unused-raii): Register the Python class.
  py::class_<DeviceKernel>(Module, "DeviceKernel");

  py::class_<DeviceModule>(Module, "DeviceModule")
      .def("get_kernel",
           [](const DeviceModule &LoadedModule, const std::string &Symbol) {
             return unwrap(LoadedModule.getKernel(Symbol));
           });

  py::class_<DeviceContext>(Module, "DeviceContext")
      .def(py::init([](int DeviceID) {
             return unwrap(DeviceContext::create(DeviceID));
           }),
           py::arg("device_id") = 0)

      .def_property_readonly("released", &DeviceContext::isReleased)
      .def("release", [](DeviceContext &Context) { unwrap(Context.release()); })

      .def("create_buffer",
           [](const DeviceContext &Context, py::ssize_t ByteCount) {
             if (ByteCount < 0)
               throw py::value_error(
                   "device buffer byte size must be nonnegative");

             return unwrap(
                 Context.createBuffer(static_cast<std::size_t>(ByteCount)));
           })

      .def("load_module",
           [](const DeviceContext &Context, const std::string &Path) {
             return unwrap(Context.loadModule(Path));
           })

      .def("copy_to_device",
           [](const DeviceContext &Context, DeviceBuffer &Destination,
              py::handle Source) {
             auto Array = requireArray(Source, false);

             unwrap(Context.copyToDevice(
                 Destination, Array.data(),
                 static_cast<std::size_t>(Array.nbytes())));
           })

      .def("copy_to_host",
           [](const DeviceContext &Context, py::handle Destination,
              const DeviceBuffer &Source) {
             auto Array = requireArray(Destination, true);
             auto ByteCount = Source.getByteCount();

             if (ByteCount > static_cast<std::size_t>(Array.nbytes()))
               throw py::value_error(
                   "copy exceeds destination host array capacity");

             unwrap(
                 Context.copyToHost(Array.mutable_data(), Source, ByteCount));
           })

      .def("launch", [](const DeviceContext &Context,
                        const DeviceKernel &Kernel, const py::tuple &Arguments,
                        const std::array<std::uint32_t, 3> &Grid,
                        const std::array<std::uint32_t, 3> &Block,
                        std::size_t SharedMemoryBytes) {
        std::vector<LaunchArgument> NativeArguments;
        std::vector<py::array> Scalars;
        NativeArguments.reserve(Arguments.size());
        Scalars.reserve(Arguments.size());

        auto NumPyScalar = py::module_::import("numpy").attr("generic");

        for (py::handle Argument : Arguments) {
          if (py::isinstance<DeviceBuffer>(Argument)) {
            NativeArguments.emplace_back(
                &Argument.cast<const DeviceBuffer &>());
          } else {
            if (!py::isinstance(Argument, NumPyScalar))
              throw py::type_error(
                  "kernel arguments must be device buffers or NumPy scalars");

            auto Array = py::array::ensure(Argument);

            if (!Array || Array.dtype().attr("hasobject").cast<bool>())
              throw py::type_error(
                  "kernel scalar storage must not contain Python objects");

            Scalars.push_back(std::move(Array));
            NativeArguments.emplace_back(Scalars.back().data());
          }
        }

        LaunchConfig Config{{Grid[0], Grid[1], Grid[2]},
                            {Block[0], Block[1], Block[2]},
                            SharedMemoryBytes};

        unwrap(Context.launch(Kernel, NativeArguments, Config));
      });
}
