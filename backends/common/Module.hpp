// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Loaded device modules and kernel resources.

#ifndef GPU_LIBM_BENCH_COMMON_MODULE_HPP
#define GPU_LIBM_BENCH_COMMON_MODULE_HPP

#include "Expected.hpp"

#include <memory>

#ifndef GPU_LIBM_BENCH_BACKEND
#error "GPU_LIBM_BENCH_BACKEND must be defined"
#endif

namespace gpu_libm_bench::GPU_LIBM_BENCH_BACKEND {

namespace detail {
class DeviceState;
class ModuleState;
} // namespace detail

class DeviceKernel {
public:
  DeviceKernel(const DeviceKernel &) = delete;
  DeviceKernel &operator=(const DeviceKernel &) = delete;
  DeviceKernel(DeviceKernel &&) noexcept = default;
  DeviceKernel &operator=(DeviceKernel &&) = delete;

private:
  friend class DeviceContext;
  friend class DeviceModule;

  DeviceKernel(std::shared_ptr<detail::ModuleState> Module, void *Function);
  const std::shared_ptr<detail::DeviceState> &getDevice() const;

  std::shared_ptr<detail::ModuleState> Module;
  void *Function;
};

class DeviceModule {
public:
  DeviceModule(const DeviceModule &) = delete;
  DeviceModule &operator=(const DeviceModule &) = delete;
  DeviceModule(DeviceModule &&) noexcept = default;
  DeviceModule &operator=(DeviceModule &&) = delete;

  Expected<DeviceKernel> getKernel(const std::string &Symbol) const;

private:
  friend class DeviceContext;

  explicit DeviceModule(std::shared_ptr<detail::ModuleState> Storage);
  static Expected<DeviceModule>
  load(std::shared_ptr<detail::DeviceState> Device, const std::string &Path);

  std::shared_ptr<detail::ModuleState> Storage;
};

} // namespace gpu_libm_bench::GPU_LIBM_BENCH_BACKEND

#endif // GPU_LIBM_BENCH_COMMON_MODULE_HPP
