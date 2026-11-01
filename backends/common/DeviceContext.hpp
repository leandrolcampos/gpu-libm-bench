// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Device information and execution context interfaces.

#ifndef GPU_LIBM_BENCH_COMMON_DEVICECONTEXT_HPP
#define GPU_LIBM_BENCH_COMMON_DEVICECONTEXT_HPP

#include "Launch.hpp"
#include "Memory.hpp"
#include "Module.hpp"

#include <vector>

#ifndef GPU_LIBM_BENCH_BACKEND
#error "GPU_LIBM_BENCH_BACKEND must be defined"
#endif

namespace gpu_libm_bench::GPU_LIBM_BENCH_BACKEND {

struct TimerInfo {
  std::string Source;
  std::uint32_t CounterBits;
  std::uint64_t FrequencyHz;
};

struct DeviceInfo {
  int DeviceID;
  std::string Name;
  std::string Architecture;
  std::size_t TotalMemoryBytes;
  TimerInfo Timer;
};

Expected<int> getDeviceCount();
Expected<DeviceInfo> getDeviceInfo(int DeviceID);

using LaunchArgument = std::variant<const DeviceBuffer *, const void *>;

class DeviceContext {
public:
  ~DeviceContext() noexcept;
  DeviceContext(const DeviceContext &) = delete;
  DeviceContext &operator=(const DeviceContext &) = delete;
  DeviceContext(DeviceContext &&Other) noexcept;
  DeviceContext &operator=(DeviceContext &&) = delete;

  static Expected<DeviceContext> create(int DeviceID);

  bool isClosed() const noexcept { return !Device; }
  Error close();

  Expected<DeviceBuffer> createBuffer(std::size_t ByteCount) const;
  Expected<DeviceModule> loadModule(const std::string &Path) const;

  Error copyToDevice(DeviceBuffer &Destination, const void *Source,
                     std::size_t ByteCount) const;
  Error copyToHost(void *Destination, const DeviceBuffer &Source,
                   std::size_t ByteCount) const;

  Error launch(const DeviceKernel &Kernel,
               const std::vector<LaunchArgument> &Arguments,
               const LaunchConfig &Config) const;

private:
  DeviceContext(std::shared_ptr<detail::DeviceState> Device, void *Stream);

  Error ensureOpen() const;
  Error validateBuffer(const DeviceBuffer &Buffer) const;
  Error validateKernel(const DeviceKernel &Kernel) const;

  std::shared_ptr<detail::DeviceState> Device;
  void *Stream;
};

} // namespace gpu_libm_bench::GPU_LIBM_BENCH_BACKEND

#endif // GPU_LIBM_BENCH_COMMON_DEVICECONTEXT_HPP
