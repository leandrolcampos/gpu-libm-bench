// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Python bindings for native GPU execution backends.

#include "Context.hpp"

#include <pybind11/native_enum.h>
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

  if (Failure.Kind == ErrorKind::NotImplemented) {
    PyErr_SetString(PyExc_NotImplementedError, Failure.Message.c_str());
    throw py::error_already_set();
  }

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

namespace {

struct LaunchArguments {
  std::vector<KernelArgument> KernelArguments;
  LaunchConfig Config;
  std::vector<py::array> ScalarStorages;

  LaunchArguments(const py::tuple &Values,
                  const std::array<std::uint32_t, 3> &Grid,
                  const std::array<std::uint32_t, 3> &Block,
                  std::size_t SharedMemoryBytes)
      : Config{{Grid[0], Grid[1], Grid[2]},
               {Block[0], Block[1], Block[2]},
               SharedMemoryBytes} {
    KernelArguments.reserve(Values.size());
    ScalarStorages.reserve(Values.size());

    auto NumPyScalar = py::module_::import("numpy").attr("generic");

    for (py::handle Value : Values) {
      if (py::isinstance<DeviceBuffer>(Value)) {
        KernelArguments.emplace_back(&Value.cast<const DeviceBuffer &>());
      } else {
        if (!py::isinstance(Value, NumPyScalar))
          throw py::type_error(
              "kernel arguments must be device buffers or NumPy scalars");

        auto ScalarStorage = py::array::ensure(Value);
        if (!ScalarStorage ||
            ScalarStorage.dtype().attr("hasobject").cast<bool>())
          throw py::type_error(
              "kernel scalar storage must not contain Python objects");

        ScalarStorages.push_back(std::move(ScalarStorage));
        KernelArguments.emplace_back(ScalarStorages.back().data());
      }
    }
  }
};

} // namespace

PYBIND11_MODULE(GPU_LIBM_BENCH_PYTHON_MODULE, Module) {
  py::native_enum<KernelMetric>(Module, "KernelMetric", "enum.Enum")
      .value("CLOCK_FREQUENCY", KernelMetric::ClockFrequency)
      .value("DURATION", KernelMetric::Duration)
      .finalize();

  Module.def("device_count", [] { return unwrap(getDeviceCount()); });

  Module.def(
      "get_device_info",
      [](int DeviceID) {
        DeviceInfoTy DeviceInfo = unwrap(getDeviceInfo(DeviceID));

        py::dict TimerInfo;
        TimerInfo["source"] = DeviceInfo.TimerInfo.Source;
        TimerInfo["counter_bits"] = DeviceInfo.TimerInfo.CounterBits;
        TimerInfo["frequency_hz"] = DeviceInfo.TimerInfo.FrequencyHz;

        py::dict Properties;
        Properties["id"] = DeviceInfo.ID;
        Properties["name"] = DeviceInfo.Name;
        Properties["architecture"] = DeviceInfo.Architecture;
        Properties["total_memory_bytes"] = DeviceInfo.TotalMemoryBytes;
        Properties["timer_info"] = TimerInfo;

        return Properties;
      },
      py::arg("device_id") = 0);

  Module.def(
      "_set_images_build_root",
      [](const std::string &Root) {
        unwrap(
            gpu_libm_bench::GPU_LIBM_BENCH_BACKEND::detail::setImagesBuildRoot(
                Root));
      },
      py::arg("root"));

  py::class_<DeviceBuffer>(Module, "DeviceBuffer")
      .def_property_readonly("nbytes", &DeviceBuffer::getByteCount);

  // NOLINTNEXTLINE(bugprone-unused-raii): Register the Python class.
  py::class_<DeviceKernel>(Module, "DeviceKernel");

  py::class_<DeviceModule>(Module, "DeviceModule")
      .def(
          "get_kernel",
          [](const DeviceModule &LoadedModule, const std::string &Symbol) {
            return unwrap(LoadedModule.getKernel(Symbol));
          },
          py::arg("symbol"));

  py::class_<DeviceContext>(Module, "DeviceContext")
      .def(py::init([](int DeviceID) {
             return unwrap(DeviceContext::create(DeviceID));
           }),
           py::arg("device_id") = 0)

      .def_property_readonly("released", &DeviceContext::isReleased)
      .def("release", [](DeviceContext &Context) { unwrap(Context.release()); })

      .def(
          "create_buffer",
          [](const DeviceContext &Context, py::ssize_t ByteCount) {
            if (ByteCount < 0)
              throw py::value_error(
                  "device buffer byte size must be nonnegative");

            return unwrap(
                Context.createBuffer(static_cast<std::size_t>(ByteCount)));
          },
          py::arg("byte_count"))

      .def(
          "load_module",
          [](const DeviceContext &Context, const std::string &Path) {
            return unwrap(Context.loadModule(Path));
          },
          py::arg("path"))

      .def(
          "copy_to_device",
          [](const DeviceContext &Context, DeviceBuffer &Destination,
             py::handle Source) {
            auto Array = requireArray(Source, false);

            unwrap(
                Context.copyToDevice(Destination, Array.data(),
                                     static_cast<std::size_t>(Array.nbytes())));
          },
          py::arg("destination"), py::arg("source"))

      .def(
          "copy_to_host",
          [](const DeviceContext &Context, py::handle Destination,
             const DeviceBuffer &Source) {
            auto Array = requireArray(Destination, true);
            auto ByteCount = Source.getByteCount();

            if (ByteCount > static_cast<std::size_t>(Array.nbytes()))
              throw py::value_error(
                  "copy exceeds destination host array capacity");

            unwrap(Context.copyToHost(Array.mutable_data(), Source, ByteCount));
          },
          py::arg("destination"), py::arg("source"))

      .def(
          "launch",
          [](const DeviceContext &Context, const DeviceKernel &Kernel,
             const py::tuple &KernelArguments,
             const std::array<std::uint32_t, 3> &Grid,
             const std::array<std::uint32_t, 3> &Block,
             std::size_t SharedMemoryBytes) {
            LaunchArguments Arguments(KernelArguments, Grid, Block,
                                      SharedMemoryBytes);
            unwrap(Context.launch(Kernel, Arguments.KernelArguments,
                                  Arguments.Config));
          },
          py::arg("kernel"), py::arg("kernel_arguments"), py::arg("grid"),
          py::arg("block"), py::arg("shared_memory_bytes"))

      .def(
          "profile",
          [](const DeviceContext &Context, const DeviceKernel &Kernel,
             const py::tuple &KernelArguments,
             const std::array<std::uint32_t, 3> &Grid,
             const std::array<std::uint32_t, 3> &Block,
             std::size_t SharedMemoryBytes,
             const std::set<KernelMetric> &Metrics) {
            LaunchArguments Arguments(KernelArguments, Grid, Block,
                                      SharedMemoryBytes);
            KernelProfile Profile = unwrap(Context.profile(
                Kernel, Arguments.KernelArguments, Arguments.Config, Metrics));

            py::dict Result;
            Result["clock_frequency_hz"] = Profile.ClockFrequencyHz;
            Result["duration_ms"] = Profile.DurationMs;

            return Result;
          },
          py::arg("kernel"), py::arg("kernel_arguments"), py::arg("grid"),
          py::arg("block"), py::arg("shared_memory_bytes"), py::arg("metrics"));
}
