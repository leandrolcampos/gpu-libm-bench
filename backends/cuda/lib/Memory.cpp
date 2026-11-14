// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Implements CUDA device memory allocation and release.

#include "Memory.hpp"
#include "Utils.hpp"

namespace gpu_libm_bench::cuda {

DeviceBuffer::DeviceBuffer(std::shared_ptr<detail::DeviceStateTy> DeviceState,
                           std::size_t ByteCount)
    : DeviceState(std::move(DeviceState)), Pointer(0), ByteCount(ByteCount) {}

DeviceBuffer::DeviceBuffer(DeviceBuffer &&Other) noexcept
    : DeviceState(std::move(Other.DeviceState)),
      Pointer(std::exchange(Other.Pointer, 0)),
      ByteCount(std::exchange(Other.ByteCount, 0)) {}

DeviceBuffer::~DeviceBuffer() noexcept {
  if (!Pointer)
    return;

  auto Guard = detail::CurrentContextGuard::create(DeviceState->getContext());
  if (!Guard) {
    reportCleanupError(Guard.error());
    return;
  }

  reportCleanupError(check(cuMemFree(Pointer), "cuMemFree"));
}

Expected<DeviceBuffer>
DeviceBuffer::create(std::shared_ptr<detail::DeviceStateTy> DeviceState,
                     std::size_t ByteCount) {
  DeviceBuffer Buffer(std::move(DeviceState), ByteCount);
  auto Guard =
      detail::CurrentContextGuard::create(Buffer.DeviceState->getContext());
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
