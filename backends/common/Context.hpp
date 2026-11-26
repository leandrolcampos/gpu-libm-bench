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
// Device information and execution context interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_LIBM_BENCH_COMMON_CONTEXT_HPP
#define GPU_LIBM_BENCH_COMMON_CONTEXT_HPP

#include "Launch.hpp"
#include "Memory.hpp"
#include "Module.hpp"
#include "Profiling.hpp"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#ifndef GPU_LIBM_BENCH_BACKEND
#error "GPU_LIBM_BENCH_BACKEND must be defined"
#endif

namespace gpu_libm_bench::GPU_LIBM_BENCH_BACKEND {

struct LaunchLimitsTy {
  Dim3 MaxGridDim;
  Dim3 MaxBlockDim;
  std::uint32_t MaxThreadsPerBlock;
  std::size_t MaxSharedMemoryBytesPerBlock;
};

struct TimerInfoTy {
  std::string Source;
  std::uint32_t CounterBits;
  std::uint64_t FrequencyHz;
};

struct DeviceInfoTy {
  int ID;
  std::string Name;
  std::string Architecture;
  std::uint32_t MultiprocessorCount;
  std::uint32_t WarpSize;
  std::uint64_t DefaultClockFrequencyHz;
  std::size_t TotalMemoryBytes;
  std::size_t L2CacheSizeBytes;
  LaunchLimitsTy LaunchLimits;
  TimerInfoTy TimerInfo;
};

namespace detail {
class KernelProfiler;
Error setImagesBuildRoot(const std::string &Root);
} // namespace detail

using KernelArgument = std::variant<const DeviceBuffer *, const void *>;

class DeviceContext {
public:
  DeviceContext(DeviceContext &&Other) noexcept;
  ~DeviceContext() noexcept;

  DeviceContext(const DeviceContext &) = delete;
  DeviceContext &operator=(const DeviceContext &) = delete;
  DeviceContext &operator=(DeviceContext &&) = delete;

  static Expected<DeviceContext> create(int DeviceID);

  Error release();
  bool isReleased() const noexcept { return !DeviceState; }

  Expected<DeviceBuffer> createBuffer(std::size_t ByteCount) const;
  Error copyToDevice(DeviceBuffer &Destination, const void *Source,
                     std::size_t ByteCount) const;
  Error copyToHost(void *Destination, const DeviceBuffer &Source,
                   std::size_t ByteCount) const;

  Expected<DeviceModule> loadModule(const std::string &Path) const;

  Error launch(const DeviceKernel &Kernel,
               const std::vector<KernelArgument> &KernelArguments,
               const LaunchConfig &Config) const;

  Expected<KernelProfile>
  profile(const DeviceKernel &Kernel,
          const std::vector<KernelArgument> &KernelArguments,
          const LaunchConfig &Config,
          const std::set<KernelMetric> &Metrics) const;

private:
  DeviceContext(std::shared_ptr<detail::DeviceStateTy> DeviceState,
                void *Stream, const std::string &Architecture);

  Expected<const DeviceModule *> getSupportModule() const;

  Error enqueueKernel(const DeviceKernel &Kernel,
                      const std::vector<KernelArgument> &KernelArguments,
                      const LaunchConfig &Config) const;

  Error ensureActive() const;
  Error validateBuffer(const DeviceBuffer &Buffer) const;
  Error validateKernel(const DeviceKernel &Kernel) const;
  Error validateKernelArguments(
      const std::vector<KernelArgument> &KernelArguments) const;

  std::shared_ptr<detail::DeviceStateTy> DeviceState;
  void *Stream;
  std::string SupportImagePath;
  mutable std::optional<DeviceModule> SupportModule;
  mutable std::unique_ptr<detail::KernelProfiler> Profiler;
};

Expected<int> getDeviceCount();
Expected<DeviceInfoTy> getDeviceInfo(int DeviceID);

} // namespace gpu_libm_bench::GPU_LIBM_BENCH_BACKEND

#endif // GPU_LIBM_BENCH_COMMON_CONTEXT_HPP
