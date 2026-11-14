// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Internal CUDA kernel profiling resources and measurements.

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

  Error readClockCounters(CUdeviceptr Destination);
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
