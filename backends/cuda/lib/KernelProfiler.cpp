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
// Implements CUDA kernel profiling.
//
//===----------------------------------------------------------------------===//

#include "KernelProfiler.hpp"
#include "Utils.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace gpu_libm_bench::cuda::detail {

//===----------------------------------------------------------------------===//
// Profiling resources
//===----------------------------------------------------------------------===//

struct KernelProfiler::StreamControl {
  std::uint32_t Released = 0;
  std::uint32_t TimedOut = 0;
};

struct KernelProfiler::ClockCounters {
  std::uint64_t Steady = 0;
  std::uint64_t Cycles = 0;
  std::uint32_t Recorded = 0;
};

struct KernelProfiler::ClockFrequencyResources {
  struct ClockCounterReadings {
    ClockCounters Start;
    ClockCounters Stop;
  };

  explicit ClockFrequencyResources(CUcontext Context) : Context(Context) {}

  ~ClockFrequencyResources() noexcept {
    if (!HostReadings)
      return;

    auto Guard = CurrentContextGuard::create(Context);
    if (!Guard) {
      reportCleanupError(Guard.error());
      return;
    }

    reportCleanupError(
        check(cuMemFreeHost(HostReadings), "cuMemFreeHost(clock counters)"));
  }

  ClockFrequencyResources(const ClockFrequencyResources &) = delete;
  ClockFrequencyResources &operator=(const ClockFrequencyResources &) = delete;

  CUcontext Context;
  CUfunction RecordClockCounters = nullptr;
  int SMCount = 0;
  ClockCounterReadings *HostReadings = nullptr;
  CUdeviceptr DeviceStartCounters = 0;
  CUdeviceptr DeviceStopCounters = 0;
};

struct KernelProfiler::DurationResources {
  explicit DurationResources(CUcontext Context) : Context(Context) {}

  ~DurationResources() noexcept {
    if (!StartEvent && !StopEvent)
      return;

    auto Guard = CurrentContextGuard::create(Context);
    if (!Guard) {
      reportCleanupError(Guard.error());
      return;
    }

    if (StopEvent)
      reportCleanupError(
          check(cuEventDestroy(StopEvent), "cuEventDestroy(stop)"));
    if (StartEvent)
      reportCleanupError(
          check(cuEventDestroy(StartEvent), "cuEventDestroy(start)"));
  }

  DurationResources(const DurationResources &) = delete;
  DurationResources &operator=(const DurationResources &) = delete;

  CUcontext Context;
  CUevent StartEvent = nullptr;
  CUevent StopEvent = nullptr;
};

//===----------------------------------------------------------------------===//
// Profiler lifecycle
//===----------------------------------------------------------------------===//

Expected<std::unique_ptr<KernelProfiler>>
KernelProfiler::create(CUcontext Context, CUstream Stream,
                       const DeviceModule &SupportModule) {
  auto Profiler = std::unique_ptr<KernelProfiler>(
      new KernelProfiler(Context, Stream, SupportModule));

  auto Kernel = SupportModule.getKernel("blockStream");
  if (!Kernel)
    return std::move(Kernel).error();
  Profiler->BlockStream = static_cast<CUfunction>(Kernel->Function);

  void *Storage = nullptr;
  if (auto Err = check(cuMemHostAlloc(&Storage, sizeof(StreamControl),
                                      CU_MEMHOSTALLOC_DEVICEMAP),
                       "cuMemHostAlloc(stream control)"))
    return Err;
  Profiler->HostStreamControl = new (Storage) StreamControl;

  if (auto Err = check(
          cuMemHostGetDevicePointer(&Profiler->DeviceStreamControl, Storage, 0),
          "cuMemHostGetDevicePointer(stream control)"))
    return Err;

  return Profiler;
}

KernelProfiler::~KernelProfiler() noexcept {
  auto Guard = CurrentContextGuard::create(Context);
  if (!Guard) {
    reportCleanupError(Guard.error());
    return;
  }

  reportCleanupError(unblock());
  CachedDuration.reset();
  CachedClockFrequency.reset();

  if (HostStreamControl)
    reportCleanupError(check(cuMemFreeHost(HostStreamControl),
                             "cuMemFreeHost(stream control)"));
}

//===----------------------------------------------------------------------===//
// Metric validation and collection
//===----------------------------------------------------------------------===//

Error KernelProfiler::validateMetrics(const std::set<KernelMetric> &Metrics) {
  if (Metrics.empty())
    return Error{ErrorKind::InvalidArgument,
                 "kernel profiling requires at least one metric"};

  for (auto Metric : Metrics)
    if (Metric != KernelMetric::ClockFrequency &&
        Metric != KernelMetric::Duration)
      return Error{ErrorKind::InvalidArgument,
                   "unsupported kernel profiling metric"};

  return Error::success();
}

Expected<KernelProfile>
KernelProfiler::profile(const std::function<Error()> &EnqueueKernel,
                        const std::set<KernelMetric> &Metrics) {
  if (auto Err = validateMetrics(Metrics))
    return Err;

  const bool ClockFrequency = Metrics.count(KernelMetric::ClockFrequency) != 0;
  const bool Duration = Metrics.count(KernelMetric::Duration) != 0;

  // Complete resource preparation before blocking the stream.
  if (ClockFrequency) {
    if (auto Err = prepareClockFrequency())
      return Err;
  }
  if (Duration) {
    if (auto Err = prepareDuration())
      return Err;
  }

  if (auto Err = check(cuStreamSynchronize(Stream), "cuStreamSynchronize"))
    return Err;

  *HostStreamControl = {};
  if (ClockFrequency)
    *CachedClockFrequency->HostReadings = {};

  // Hold the stream until the kernel and its measurement operations are queued.
  struct ProfilingGuard {
    KernelProfiler &Profiler;
    ~ProfilingGuard() { reportCleanupError(Profiler.unblock()); }
  } Guard{*this};

  CUdeviceptr Released =
      DeviceStreamControl + offsetof(StreamControl, Released);
  CUdeviceptr TimedOut =
      DeviceStreamControl + offsetof(StreamControl, TimedOut);
  std::uint64_t TimeoutTicks = 5'000'000'000;
  void *Arguments[] = {&Released, &TimedOut, &TimeoutTicks};

  NeedsUnblock = true;
  if (auto Err = check(cuLaunchKernel(BlockStream, 1, 1, 1, 1, 1, 1, 0, Stream,
                                      Arguments, nullptr),
                       "cuLaunchKernel(blockStream)"))
    return Err;

  // Keep counter kernels outside the timing events.
  if (ClockFrequency) {
    if (auto Err =
            recordClockCounters(CachedClockFrequency->DeviceStartCounters))
      return Err;
  }
  if (Duration) {
    if (auto Err = check(cuEventRecord(CachedDuration->StartEvent, Stream),
                         "cuEventRecord(start)"))
      return Err;
  }

  if (auto Err = EnqueueKernel())
    return Err;

  if (Duration) {
    if (auto Err = check(cuEventRecord(CachedDuration->StopEvent, Stream),
                         "cuEventRecord(stop)"))
      return Err;
  }
  if (ClockFrequency) {
    if (auto Err =
            recordClockCounters(CachedClockFrequency->DeviceStopCounters))
      return Err;
  }

  // Wait for GPU writes before reading the control flags and counter readings.
  if (auto Err = unblock())
    return Err;
  if (HostStreamControl->TimedOut)
    return Error{ErrorKind::Runtime,
                 "kernel profiling blocking kernel timed out"};

  KernelProfile Profile;
  if (ClockFrequency) {
    const auto &[Start, Stop] = *CachedClockFrequency->HostReadings;
    if (!Start.Recorded || !Stop.Recorded || Stop.Steady <= Start.Steady ||
        Stop.Cycles <= Start.Cycles) {
      Profile.ClockFrequencyHz = std::numeric_limits<double>::quiet_NaN();
    } else {
      auto ElapsedTicks = Stop.Steady - Start.Steady;
      auto ElapsedCycles = Stop.Cycles - Start.Cycles;
      Profile.ClockFrequencyHz = static_cast<double>(ElapsedCycles) /
                                 static_cast<double>(ElapsedTicks) * 1e9;
    }
  }
  if (Duration) {
    float Milliseconds = 0;
    if (auto Err =
            check(cuEventElapsedTime(&Milliseconds, CachedDuration->StartEvent,
                                     CachedDuration->StopEvent),
                  "cuEventElapsedTime"))
      return Err;
    Profile.DurationMs = static_cast<double>(Milliseconds);
  }

  return Profile;
}

//===----------------------------------------------------------------------===//
// Resource preparation
//===----------------------------------------------------------------------===//

KernelProfiler::KernelProfiler(CUcontext Context, CUstream Stream,
                               const DeviceModule &SupportModule)
    : Context(Context), Stream(Stream), SupportModule(SupportModule.Storage) {}

Error KernelProfiler::prepareClockFrequency() {
  if (CachedClockFrequency)
    return Error::success();

  auto Resources = std::make_unique<ClockFrequencyResources>(Context);
  auto Kernel = SupportModule.getKernel("recordClockCounters");
  if (!Kernel)
    return std::move(Kernel).error();
  Resources->RecordClockCounters = static_cast<CUfunction>(Kernel->Function);

  CUdevice Device;
  if (auto Err = check(cuCtxGetDevice(&Device), "cuCtxGetDevice"))
    return Err;
  if (auto Err = check(cuDeviceGetAttribute(
                           &Resources->SMCount,
                           CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, Device),
                       "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)"))
    return Err;

  using ClockCounterReadings = ClockFrequencyResources::ClockCounterReadings;
  void *Storage = nullptr;
  if (auto Err = check(cuMemHostAlloc(&Storage, sizeof(ClockCounterReadings),
                                      CU_MEMHOSTALLOC_DEVICEMAP),
                       "cuMemHostAlloc(clock counters)"))
    return Err;
  Resources->HostReadings = new (Storage) ClockCounterReadings;

  CUdeviceptr DeviceReadings = 0;
  if (auto Err = check(cuMemHostGetDevicePointer(&DeviceReadings, Storage, 0),
                       "cuMemHostGetDevicePointer(clock counters)"))
    return Err;

  Resources->DeviceStartCounters =
      DeviceReadings + offsetof(ClockCounterReadings, Start);
  Resources->DeviceStopCounters =
      DeviceReadings + offsetof(ClockCounterReadings, Stop);

  CachedClockFrequency = std::move(Resources);
  return Error::success();
}

Error KernelProfiler::prepareDuration() {
  if (CachedDuration)
    return Error::success();

  auto Resources = std::make_unique<DurationResources>(Context);
  if (auto Err = check(cuEventCreate(&Resources->StartEvent, CU_EVENT_DEFAULT),
                       "cuEventCreate(start)"))
    return Err;
  if (auto Err = check(cuEventCreate(&Resources->StopEvent, CU_EVENT_DEFAULT),
                       "cuEventCreate(stop)"))
    return Err;

  CachedDuration = std::move(Resources);
  return Error::success();
}

//===----------------------------------------------------------------------===//
// Stream operations
//===----------------------------------------------------------------------===//

Error KernelProfiler::recordClockCounters(CUdeviceptr Destination) {
  CUdeviceptr SteadyCounter = Destination + offsetof(ClockCounters, Steady);
  CUdeviceptr CycleCounter = Destination + offsetof(ClockCounters, Cycles);
  CUdeviceptr Recorded = Destination + offsetof(ClockCounters, Recorded);
  void *Arguments[] = {&SteadyCounter, &CycleCounter, &Recorded};

  return check(
      cuLaunchKernel(CachedClockFrequency->RecordClockCounters,
                     static_cast<unsigned int>(CachedClockFrequency->SMCount),
                     1, 1, 1, 1, 1, 0, Stream, Arguments, nullptr),
      "cuLaunchKernel(recordClockCounters)");
}

Error KernelProfiler::unblock() {
  if (!NeedsUnblock)
    return Error::success();

  volatile std::uint32_t &Released = HostStreamControl->Released;
  Released = 1;
  auto Failure = check(cuStreamSynchronize(Stream), "cuStreamSynchronize");
  NeedsUnblock = false;
  return Failure;
}

} // namespace gpu_libm_bench::cuda::detail
