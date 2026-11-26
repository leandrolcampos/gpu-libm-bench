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
// Implements CUDA device memory allocation and release.
//
//===----------------------------------------------------------------------===//

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
