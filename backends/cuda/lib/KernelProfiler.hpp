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
// Internal CUDA kernel profiling resources and measurements.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_LIBM_BENCH_CUDA_KERNELPROFILER_HPP
#define GPU_LIBM_BENCH_CUDA_KERNELPROFILER_HPP

#include "Expected.hpp"
#include "Module.hpp"
#include "Profiling.hpp"

#include <cuda.h>

#include <functional>
#include <memory>
#include <set>

namespace gpu_libm_bench::cuda::detail {

class KernelProfiler {
public:
  static Expected<std::unique_ptr<KernelProfiler>>
  create(CUcontext Context, CUstream Stream, const DeviceModule &SupportModule);

  ~KernelProfiler() noexcept;
  KernelProfiler(const KernelProfiler &) = delete;
  KernelProfiler &operator=(const KernelProfiler &) = delete;

  static Error validateMetrics(const std::set<KernelMetric> &Metrics);

  Expected<KernelProfile> profile(const std::function<Error()> &EnqueueKernel,
                                  const std::set<KernelMetric> &Metrics);

private:
  struct StreamControl;
  struct ClockCounters;
  struct ClockFrequencyResources;
  struct DurationResources;

  KernelProfiler(CUcontext Context, CUstream Stream,
                 const DeviceModule &SupportModule);

  Error prepareClockFrequency();
  Error prepareDuration();

  Error recordClockCounters(CUdeviceptr Destination);
  Error unblock();

  CUcontext Context;
  CUstream Stream;
  DeviceModule SupportModule;

  CUfunction BlockStream = nullptr;
  StreamControl *HostStreamControl = nullptr;
  CUdeviceptr DeviceStreamControl = 0;

  std::unique_ptr<ClockFrequencyResources> CachedClockFrequency;
  std::unique_ptr<DurationResources> CachedDuration;

  bool NeedsUnblock = false;
};

} // namespace gpu_libm_bench::cuda::detail

#endif // GPU_LIBM_BENCH_CUDA_KERNELPROFILER_HPP
