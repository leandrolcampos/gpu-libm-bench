// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Implements CUDA kernel profiling.

#include "KernelProfiler.hpp"
#include "Utils.hpp"

#include <cstddef>
#include <cstdint>
#include <new>

namespace gpu_libm_bench::cuda::detail {

//
// Profiling resources
//

struct KernelProfiler::StreamControl {
  std::uint32_t Release = 0;
  std::uint32_t TimedOut = 0;
};

struct KernelProfiler::ClockCounters {
  std::uint64_t Steady = 0;
  std::uint64_t Cycles = 0;
};

struct KernelProfiler::ClockFrequencyResources {
  struct Samples {
    ClockCounters Start;
    ClockCounters Stop;
  };

  explicit ClockFrequencyResources(CUcontext Context) : Context(Context) {}

  ~ClockFrequencyResources() noexcept {
    if (!HostSamples)
      return;

    auto Guard = CurrentContextGuard::create(Context);
    if (!Guard) {
      reportCleanupError(Guard.error());
      return;
    }

    reportCleanupError(
        check(cuMemFreeHost(HostSamples), "cuMemFreeHost(clock counters)"));
  }

  ClockFrequencyResources(const ClockFrequencyResources &) = delete;
  ClockFrequencyResources &operator=(const ClockFrequencyResources &) = delete;

  CUcontext Context;
  CUfunction ReadClockCounters = nullptr;
  int SMCount = 0;
  Samples *HostSamples = nullptr;
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

//
// Profiler lifecycle
//

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

//
// Metric validation and collection
//

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
    *CachedClockFrequency->HostSamples = {};

  // Hold the stream until the kernel and its measurement operations are queued.
  struct ProfilingGuard {
    KernelProfiler &Profiler;
    ~ProfilingGuard() { reportCleanupError(Profiler.unblock()); }
  } Guard{*this};

  CUdeviceptr Release = DeviceStreamControl + offsetof(StreamControl, Release);
  CUdeviceptr TimedOut =
      DeviceStreamControl + offsetof(StreamControl, TimedOut);
  std::uint64_t TimeoutTicks = 5'000'000'000;
  void *Arguments[] = {&Release, &TimedOut, &TimeoutTicks};

  NeedsUnblock = true;
  if (auto Err = check(cuLaunchKernel(BlockStream, 1, 1, 1, 1, 1, 1, 0, Stream,
                                      Arguments, nullptr),
                       "cuLaunchKernel(blockStream)"))
    return Err;

  // Keep counter kernels outside the timing events.
  if (ClockFrequency) {
    if (auto Err = readClockCounters(CachedClockFrequency->DeviceStartCounters))
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
    if (auto Err = readClockCounters(CachedClockFrequency->DeviceStopCounters))
      return Err;
  }

  // Wait for GPU writes before reading the control flags and counter samples.
  if (auto Err = unblock())
    return Err;
  if (HostStreamControl->TimedOut)
    return Error{ErrorKind::Runtime,
                 "kernel profiling blocking kernel timed out"};

  KernelProfile Profile;
  if (ClockFrequency) {
    const auto &[Start, Stop] = *CachedClockFrequency->HostSamples;
    if (!Start.Steady || !Stop.Steady || Stop.Steady <= Start.Steady ||
        Stop.Cycles <= Start.Cycles)
      return Error{ErrorKind::Runtime, "invalid SM 0 clock counter readings"};

    auto ElapsedTicks = Stop.Steady - Start.Steady;
    auto ElapsedCycles = Stop.Cycles - Start.Cycles;
    Profile.ClockFrequencyHz = static_cast<double>(ElapsedCycles) /
                               static_cast<double>(ElapsedTicks) * 1e9;
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

//
// Resource preparation
//

KernelProfiler::KernelProfiler(CUcontext Context, CUstream Stream,
                               const DeviceModule &SupportModule)
    : Context(Context), Stream(Stream), SupportModule(SupportModule.Storage) {}

Error KernelProfiler::prepareClockFrequency() {
  if (CachedClockFrequency)
    return Error::success();

  auto Resources = std::make_unique<ClockFrequencyResources>(Context);
  auto Kernel = SupportModule.getKernel("readClockCounters");
  if (!Kernel)
    return std::move(Kernel).error();
  Resources->ReadClockCounters = static_cast<CUfunction>(Kernel->Function);

  CUdevice Device;
  if (auto Err = check(cuCtxGetDevice(&Device), "cuCtxGetDevice"))
    return Err;
  if (auto Err = check(cuDeviceGetAttribute(
                           &Resources->SMCount,
                           CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, Device),
                       "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)"))
    return Err;

  using Samples = ClockFrequencyResources::Samples;
  void *Storage = nullptr;
  if (auto Err = check(
          cuMemHostAlloc(&Storage, sizeof(Samples), CU_MEMHOSTALLOC_DEVICEMAP),
          "cuMemHostAlloc(clock counters)"))
    return Err;
  Resources->HostSamples = new (Storage) Samples;

  CUdeviceptr DeviceSamples = 0;
  if (auto Err = check(cuMemHostGetDevicePointer(&DeviceSamples, Storage, 0),
                       "cuMemHostGetDevicePointer(clock counters)"))
    return Err;

  Resources->DeviceStartCounters = DeviceSamples + offsetof(Samples, Start);
  Resources->DeviceStopCounters = DeviceSamples + offsetof(Samples, Stop);

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

//
// Stream operations
//

Error KernelProfiler::readClockCounters(CUdeviceptr Destination) {
  CUdeviceptr SteadyCounter = Destination + offsetof(ClockCounters, Steady);
  CUdeviceptr CycleCounter = Destination + offsetof(ClockCounters, Cycles);
  void *Arguments[] = {&SteadyCounter, &CycleCounter};

  return check(
      cuLaunchKernel(CachedClockFrequency->ReadClockCounters,
                     static_cast<unsigned int>(CachedClockFrequency->SMCount),
                     1, 1, 1, 1, 1, 0, Stream, Arguments, nullptr),
      "cuLaunchKernel(readClockCounters)");
}

Error KernelProfiler::unblock() {
  if (!NeedsUnblock)
    return Error::success();

  volatile std::uint32_t &Release = HostStreamControl->Release;
  Release = 1;
  auto Failure = check(cuStreamSynchronize(Stream), "cuStreamSynchronize");
  NeedsUnblock = false;
  return Failure;
}

} // namespace gpu_libm_bench::cuda::detail
