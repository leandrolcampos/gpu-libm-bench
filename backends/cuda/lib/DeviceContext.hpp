// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// CUDA device information and execution context interfaces.

#ifndef GPU_LIBM_BENCH_CUDA_DEVICECONTEXT_HPP
#define GPU_LIBM_BENCH_CUDA_DEVICECONTEXT_HPP

#include <cstddef>
#include <string>

namespace gpu_libm_bench {

struct DeviceInfo {
  int DeviceID;
  std::string Name;
  std::string Architecture;
  std::size_t TotalMemoryBytes;
};

int getDeviceCount();
DeviceInfo getDeviceInfo(int DeviceID);

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_CUDA_DEVICECONTEXT_HPP
