// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Internal CUDA device state, context activation, and diagnostics.

#ifndef GPU_LIBM_BENCH_CUDA_UTILS_HPP
#define GPU_LIBM_BENCH_CUDA_UTILS_HPP

#include "Expected.hpp"

#include <cuda.h>

#include <memory>

namespace gpu_libm_bench::cuda {

Error check(CUresult Result, const char *Operation);
void reportCleanupError(const Error &Failure) noexcept;

namespace detail {

class CurrentContextGuard {
public:
  ~CurrentContextGuard() noexcept;
  CurrentContextGuard(const CurrentContextGuard &) = delete;
  CurrentContextGuard &operator=(const CurrentContextGuard &) = delete;
  CurrentContextGuard(CurrentContextGuard &&Other) noexcept;
  CurrentContextGuard &operator=(CurrentContextGuard &&) = delete;

  static Expected<CurrentContextGuard> create(CUcontext Context);

private:
  CurrentContextGuard() = default;
  bool Active = true;
};

class DeviceStateTy {
public:
  ~DeviceStateTy() noexcept;
  DeviceStateTy(const DeviceStateTy &) = delete;
  DeviceStateTy &operator=(const DeviceStateTy &) = delete;

  static Expected<std::shared_ptr<DeviceStateTy>> create(int DeviceID);
  CUcontext getContext() const noexcept { return Context; }

private:
  DeviceStateTy(CUdevice Device, CUcontext Context)
      : Device(Device), Context(Context) {}

  CUdevice Device;
  CUcontext Context;
};

} // namespace detail
} // namespace gpu_libm_bench::cuda

#endif // GPU_LIBM_BENCH_CUDA_UTILS_HPP
