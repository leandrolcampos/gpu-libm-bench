//===----------------------------------------------------------------------===//
// Copyright 2026 Leandro Lacerda Campos
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Internal CUDA device state, context activation, and diagnostics.
//
//===----------------------------------------------------------------------===//

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
