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
// Device memory allocations and byte capacities.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_LIBM_BENCH_COMMON_MEMORY_HPP
#define GPU_LIBM_BENCH_COMMON_MEMORY_HPP

#include "Expected.hpp"

#include <cstddef>
#include <memory>

#ifndef GPU_LIBM_BENCH_BACKEND
#error "GPU_LIBM_BENCH_BACKEND must be defined"
#endif

namespace gpu_libm_bench::GPU_LIBM_BENCH_BACKEND {

namespace detail {
class DeviceStateTy;
} // namespace detail

class DeviceBuffer {
public:
  ~DeviceBuffer() noexcept;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&Other) noexcept;
  DeviceBuffer &operator=(DeviceBuffer &&) = delete;

  std::size_t getByteCount() const noexcept { return ByteCount; }

private:
  friend class DeviceContext;

  static Expected<DeviceBuffer>
  create(std::shared_ptr<detail::DeviceStateTy> DeviceState,
         std::size_t ByteCount);
  DeviceBuffer(std::shared_ptr<detail::DeviceStateTy> DeviceState,
               std::size_t ByteCount);

  std::shared_ptr<detail::DeviceStateTy> DeviceState;
  std::uint64_t Pointer;
  std::size_t ByteCount;
};

} // namespace gpu_libm_bench::GPU_LIBM_BENCH_BACKEND

#endif // GPU_LIBM_BENCH_COMMON_MEMORY_HPP
