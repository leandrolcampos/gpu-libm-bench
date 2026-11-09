// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Device memory allocations and byte capacities.

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
class DeviceState;
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
  create(std::shared_ptr<detail::DeviceState> Device, std::size_t ByteCount);
  DeviceBuffer(std::shared_ptr<detail::DeviceState> Device,
               std::size_t ByteCount);

  std::shared_ptr<detail::DeviceState> Device;
  std::uint64_t Pointer;
  std::size_t ByteCount;
};

} // namespace gpu_libm_bench::GPU_LIBM_BENCH_BACKEND

#endif // GPU_LIBM_BENCH_COMMON_MEMORY_HPP
