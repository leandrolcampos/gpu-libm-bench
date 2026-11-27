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
// Implements support kernels for kernel profiling.
//
//===----------------------------------------------------------------------===//

#include "GPU.hpp"

#include <gpuintrin.h>

#include <stdint.h>

using namespace gpu_libm_bench;

extern "C" __gpu_kernel void blockStream(const volatile uint32_t *Released,
                                         volatile uint32_t *TimedOut,
                                         uint64_t TimeoutTicks) {
  if (gpu::globalThreadId() != 0)
    return;

  *TimedOut = 0;
  uint64_t Start = gpu::readSteadyCounter();

  while (!*Released) {
    if (gpu::readSteadyCounter() - Start >= TimeoutTicks) {
      *TimedOut = 1;
      break;
    }
  }

  gpu::atomicThreadFence<gpu::MemoryOrder::SeqCst, gpu::MemoryScope::System>();
}

extern "C" __gpu_kernel void recordClockCounters(uint64_t *SteadyCounter,
                                                 uint64_t *CycleCounter,
                                                 uint32_t *Recorded) {
  if (gpu::threadId() != 0 || gpu::multiprocessorId() != 0)
    return;

  if (gpu::atomicExchange<gpu::MemoryOrder::Relaxed, gpu::MemoryScope::Device>(
          Recorded, 1U) != 0)
    return;

  uint64_t SteadyReading = gpu::readSteadyCounter();
  uint64_t CycleReading = gpu::readCycleCounter();

  *SteadyCounter = SteadyReading;
  *CycleCounter = CycleReading;
}
