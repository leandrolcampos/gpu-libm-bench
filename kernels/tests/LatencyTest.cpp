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
// Latency validation kernels for dependent calls.
//
//===----------------------------------------------------------------------===//

#include "Latency.hpp"
#include "FPUtil.hpp"
#include "GPU.hpp"

#include <gpuintrin.h>

#include <stdint.h>

// One thread per probe may access this state. Probes in the same module must
// run serially because they share it.
static uint64_t *Trace;
static uint32_t CallCount;

template <typename InputType> static uint64_t asUInt(InputType Input) {
  if constexpr (__is_integral(InputType))
    return static_cast<__make_unsigned(InputType)>(Input);
  else
    return gpu_libm_bench::fputil::FPBits(Input).asUInt();
}

template <typename FloatType, typename... OtherInputTypes>
static FloatType observeInputs(FloatType FirstInput,
                               OtherInputTypes... OtherInputs) {
  const uint64_t Call = CallCount++;
  uint64_t *Entry = Trace + Call * (1 + sizeof...(OtherInputs));
  *Entry++ = asUInt(FirstInput);
  ((*Entry++ = asUInt(OtherInputs)), ...);

  return FirstInput;
}

template <typename StorageType, typename FloatType, typename... OtherInputTypes>
static void probeInputs(uint32_t PassCount, uint32_t ChainLength,
                        const StorageType *ZeroMask, uint64_t *ElapsedTicks,
                        StorageType *Output, uint64_t *ObservedInputs,
                        uint32_t *ObservedCount, const FloatType *FirstInput,
                        const OtherInputTypes *...OtherInputs) {
  if (gpu_libm_bench::gpu::globalThreadId() != 0)
    return;

  Trace = ObservedInputs;
  CallCount = 0;

  gpu_libm_bench::measureLatency<observeInputs<FloatType, OtherInputTypes...>>(
      PassCount, ChainLength, ZeroMask, ElapsedTicks, Output, FirstInput,
      OtherInputs...);

  ObservedCount[0] = CallCount;
}

extern "C" __gpu_kernel void
latencyProbe(uint32_t PassCount, uint32_t ChainLength, const uint32_t *ZeroMask,
             uint64_t *ElapsedTicks, uint32_t *Output, uint64_t *ObservedInputs,
             uint32_t *ObservedCount, const float *X) {
  probeInputs(PassCount, ChainLength, ZeroMask, ElapsedTicks, Output,
              ObservedInputs, ObservedCount, X);
}

extern "C" __gpu_kernel void halfBinaryLatencyProbe(
    uint32_t PassCount, uint32_t ChainLength, const uint16_t *ZeroMask,
    uint64_t *ElapsedTicks, uint16_t *Output, uint64_t *ObservedInputs,
    uint32_t *ObservedCount, const _Float16 *X, const _Float16 *Y) {
  probeInputs(PassCount, ChainLength, ZeroMask, ElapsedTicks, Output,
              ObservedInputs, ObservedCount, X, Y);
}

extern "C" __gpu_kernel void doubleIntLatencyProbe(
    uint32_t PassCount, uint32_t ChainLength, const uint64_t *ZeroMask,
    uint64_t *ElapsedTicks, uint64_t *Output, uint64_t *ObservedInputs,
    uint32_t *ObservedCount, const double *X, const int *Exponent) {
  probeInputs(PassCount, ChainLength, ZeroMask, ElapsedTicks, Output,
              ObservedInputs, ObservedCount, X, Exponent);
}
