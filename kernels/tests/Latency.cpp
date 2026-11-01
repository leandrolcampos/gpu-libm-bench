// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Dependent-call latency validation kernels.

#include "Latency.hpp"

static uint64_t *Trace;
static uint32_t CallCount;

template <typename InputType>
static void observeInput(uint64_t Index, InputType Input) {
  if constexpr (__is_same(InputType, int))
    Trace[Index] = static_cast<uint32_t>(Input);
  else
    Trace[Index] = gpu_libm_bench::fputil::FPBits(Input).asUInt();
}

template <typename FloatType, typename... OtherInputTypes>
static FloatType observeInputs(FloatType FirstInput,
                               OtherInputTypes... OtherInputs) {
  uint64_t Index = CallCount++ * (1 + sizeof...(OtherInputs));
  observeInput(Index++, FirstInput);
  (observeInput(Index++, OtherInputs), ...);
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
  gpu_libm_bench::latency<observeInputs<FloatType, OtherInputTypes...>>(
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
