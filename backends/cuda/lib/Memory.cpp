// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Implements device memory allocation and release.

#include "Memory.hpp"
#include "Utils.hpp"

namespace gpu_libm_bench::cuda {

DeviceBuffer::DeviceBuffer(std::shared_ptr<detail::DeviceState> Device,
                           std::size_t ByteCount)
    : Device(std::move(Device)), Pointer(0), ByteCount(ByteCount) {}

DeviceBuffer::DeviceBuffer(DeviceBuffer &&Other) noexcept
    : Device(std::move(Other.Device)), Pointer(std::exchange(Other.Pointer, 0)),
      ByteCount(std::exchange(Other.ByteCount, 0)) {}

DeviceBuffer::~DeviceBuffer() noexcept {
  if (!Pointer)
    return;

  auto Guard = detail::CurrentContextGuard::create(Device->getContext());
  if (!Guard) {
    reportCleanupError(Guard.error());
    return;
  }

  reportCleanupError(check(cuMemFree(Pointer), "cuMemFree"));
}

Expected<DeviceBuffer>
DeviceBuffer::create(std::shared_ptr<detail::DeviceState> Device,
                     std::size_t ByteCount) {
  DeviceBuffer Buffer(std::move(Device), ByteCount);
  auto Guard = detail::CurrentContextGuard::create(Buffer.Device->getContext());
  if (!Guard)
    return std::move(Guard).error();

  CUdeviceptr Pointer = 0;
  if (ByteCount) {
    if (auto Err = check(cuMemAlloc(&Pointer, ByteCount), "cuMemAlloc"))
      return Err;
  }

  Buffer.Pointer = Pointer;
  return Buffer;
}

} // namespace gpu_libm_bench::cuda
