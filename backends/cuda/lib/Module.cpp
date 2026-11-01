// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Implements device image loading and kernel resolution.

#include "Module.hpp"
#include "Utils.hpp"

namespace gpu_libm_bench::cuda {

namespace detail {

class ModuleState {
public:
  ModuleState(std::shared_ptr<DeviceState> Device, CUmodule Module)
      : Device(std::move(Device)), Module(Module) {}

  ~ModuleState() noexcept {
    auto Guard = CurrentContextGuard::create(Device->getContext());
    if (!Guard) {
      reportCleanupError(Guard.error());
      return;
    }

    reportCleanupError(check(cuModuleUnload(Module), "cuModuleUnload"));
  }

  ModuleState(const ModuleState &) = delete;
  ModuleState &operator=(const ModuleState &) = delete;

  std::shared_ptr<DeviceState> Device;
  CUmodule Module;
};

} // namespace detail

DeviceKernel::DeviceKernel(std::shared_ptr<detail::ModuleState> Module,
                           void *Function)
    : Module(std::move(Module)), Function(Function) {}

const std::shared_ptr<detail::DeviceState> &DeviceKernel::getDevice() const {
  return Module->Device;
}

DeviceModule::DeviceModule(std::shared_ptr<detail::ModuleState> Storage)
    : Storage(std::move(Storage)) {}

Expected<DeviceModule>
DeviceModule::load(std::shared_ptr<detail::DeviceState> Device,
                   const std::string &Path) {
  auto Guard = detail::CurrentContextGuard::create(Device->getContext());
  if (!Guard)
    return std::move(Guard).error();

  CUmodule Module = nullptr;
  std::string Operation = "cuModuleLoad(" + Path + ")";
  if (auto Err = check(cuModuleLoad(&Module, Path.c_str()), Operation.c_str()))
    return Err;

  return DeviceModule(
      std::make_shared<detail::ModuleState>(std::move(Device), Module));
}

Expected<DeviceKernel>
DeviceModule::getKernel(const std::string &Symbol) const {
  if (!Storage)
    return Error{ErrorKind::InvalidArgument,
                 "device module has no loaded image"};
  if (Symbol.empty())
    return Error{ErrorKind::InvalidArgument, "invalid kernel symbol"};

  auto Guard =
      detail::CurrentContextGuard::create(Storage->Device->getContext());
  if (!Guard)
    return std::move(Guard).error();

  CUfunction Function = nullptr;
  std::string Operation = "cuModuleGetFunction(" + Symbol + ")";
  if (auto Err =
          check(cuModuleGetFunction(&Function, Storage->Module, Symbol.c_str()),
                Operation.c_str()))
    return Err;

  return DeviceKernel(Storage, Function);
}

} // namespace gpu_libm_bench::cuda
