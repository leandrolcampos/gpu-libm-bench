// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Device information and execution context interfaces.

#ifndef GPU_LIBM_BENCH_COMMON_CONTEXT_HPP
#define GPU_LIBM_BENCH_COMMON_CONTEXT_HPP

#include "Launch.hpp"
#include "Memory.hpp"
#include "Module.hpp"
#include "Profiling.hpp"

#include <set>
#include <vector>

#ifndef GPU_LIBM_BENCH_BACKEND
#error "GPU_LIBM_BENCH_BACKEND must be defined"
#endif

namespace gpu_libm_bench::GPU_LIBM_BENCH_BACKEND {

struct TimerInfoTy {
  std::string Source;
  std::uint32_t CounterBits;
  std::uint64_t FrequencyHz;
};

struct DeviceInfoTy {
  int ID;
  std::string Name;
  std::string Architecture;
  std::size_t TotalMemoryBytes;
  TimerInfoTy TimerInfo;
};

Expected<int> getDeviceCount();
Expected<DeviceInfoTy> getDeviceInfo(int DeviceID);

using KernelArgument = std::variant<const DeviceBuffer *, const void *>;

class DeviceContext {
public:
  ~DeviceContext() noexcept;
  DeviceContext(const DeviceContext &) = delete;
  DeviceContext &operator=(const DeviceContext &) = delete;
  DeviceContext(DeviceContext &&Other) noexcept;
  DeviceContext &operator=(DeviceContext &&) = delete;

  static Expected<DeviceContext> create(int DeviceID);

  bool isReleased() const noexcept { return !Device; }
  Error release();

  Expected<DeviceBuffer> createBuffer(std::size_t ByteCount) const;
  Expected<DeviceModule> loadModule(const std::string &Path) const;

  Error copyToDevice(DeviceBuffer &Destination, const void *Source,
                     std::size_t ByteCount) const;
  Error copyToHost(void *Destination, const DeviceBuffer &Source,
                   std::size_t ByteCount) const;

  Error launch(const DeviceKernel &Kernel,
               const std::vector<KernelArgument> &KernelArguments,
               const LaunchConfig &Config) const;

  Expected<KernelProfile>
  profile(const DeviceKernel &Kernel,
          const std::vector<KernelArgument> &KernelArguments,
          const LaunchConfig &Config,
          const std::set<KernelMetric> &Metrics) const;

private:
  DeviceContext(std::shared_ptr<detail::DeviceState> Device, void *Stream);

  Error ensureActive() const;
  Error validateBuffer(const DeviceBuffer &Buffer) const;
  Error validateKernel(const DeviceKernel &Kernel) const;
  Error validateKernelArguments(
      const std::vector<KernelArgument> &KernelArguments) const;

  std::shared_ptr<detail::DeviceState> Device;
  void *Stream;
};

} // namespace gpu_libm_bench::GPU_LIBM_BENCH_BACKEND

#endif // GPU_LIBM_BENCH_COMMON_CONTEXT_HPP
