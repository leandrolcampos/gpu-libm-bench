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
// Throughput validation kernels for independent calls.
//
//===----------------------------------------------------------------------===//

#include "Throughput.hpp"
#include "FPUtil.hpp"

#include <gpuintrin.h>

#include <stdint.h>

// One thread per probe may access this state. Probes in the same module must
// run serially because they share it.
static uint64_t *Trace;
static uint64_t CallCount;
static uint64_t TraceCapacity;

template <typename InputType> static uint64_t asUInt(InputType Input) {
  if constexpr (__is_integral(InputType))
    return static_cast<__make_unsigned(InputType)>(Input);
  else
    return gpu_libm_bench::fputil::FPBits(Input).asUInt();
}

template <typename... InputTypes>
static void observeInputs(InputTypes... Inputs) {
  const uint64_t Call = CallCount++;
  if (Call >= TraceCapacity)
    return;

  uint64_t *Entry = Trace + Call * sizeof...(Inputs);
  ((*Entry++ = asUInt(Inputs)), ...);
}

static _Float16 observeHalfInputs(_Float16 X, _Float16 Y) {
  observeInputs(X, Y);
  return Y;
}

static double observeMixedInputs(long X, unsigned int Y, int Z, double W) {
  observeInputs(X, Y, Z, W);
  return W;
}

template <auto Function, typename StorageType, typename... InputTypes>
static void probeInputs(uint32_t PassCount, uint32_t ZeroMask, uint64_t Count,
                        StorageType *Checksum, uint64_t *ObservedInputs,
                        uint64_t *ObservedCount, const InputTypes *...Inputs) {
  Trace = ObservedInputs;
  CallCount = 0;
  TraceCapacity = Count * PassCount;

  gpu_libm_bench::measureThroughput<Function, 2>(PassCount, ZeroMask, Count,
                                                 Checksum, Inputs...);

  ObservedCount[0] = CallCount;
}

extern "C" __gpu_kernel void
halfBinaryThroughputProbe(uint32_t PassCount, uint32_t ZeroMask, uint64_t Count,
                          uint16_t *Checksum, uint64_t *ObservedInputs,
                          uint64_t *ObservedCount, const _Float16 *X,
                          const _Float16 *Y) {
  probeInputs<observeHalfInputs>(PassCount, ZeroMask, Count, Checksum,
                                 ObservedInputs, ObservedCount, X, Y);
}

extern "C" __gpu_kernel void
mixedThroughputProbe(uint32_t PassCount, uint32_t ZeroMask, uint64_t Count,
                     uint64_t *Checksum, uint64_t *ObservedInputs,
                     uint64_t *ObservedCount, const long *X,
                     const unsigned int *Y, const int *Z, const double *W) {
  probeInputs<observeMixedInputs>(PassCount, ZeroMask, Count, Checksum,
                                  ObservedInputs, ObservedCount, X, Y, Z, W);
}
