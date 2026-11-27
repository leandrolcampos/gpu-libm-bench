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
// Timer validation kernel for consecutive steady counter reads.
//
//===----------------------------------------------------------------------===//

#include "GPU.hpp"

extern "C" __gpu_kernel void timerReadings(uint64_t *Readings, uint32_t Count) {
  if (gpu_libm_bench::gpu::globalThreadId() != 0)
    return;

#pragma clang loop unroll(disable)
  for (uint32_t Index = 0; Index < Count; ++Index) {
    uint64_t Start = gpu_libm_bench::gpu::readSteadyCounter();
    uint64_t Stop = gpu_libm_bench::gpu::readSteadyCounter();

    Readings[2 * static_cast<uint64_t>(Index)] = Start;
    Readings[2 * static_cast<uint64_t>(Index) + 1] = Stop;
  }
}
