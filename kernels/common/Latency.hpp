// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Latency measurement wrappers for GPU math functions.

#ifndef GPU_LIBM_BENCH_KERNELS_COMMON_LATENCY_HPP
#define GPU_LIBM_BENCH_KERNELS_COMMON_LATENCY_HPP

#include "FPUtil.hpp"
#include "GPU.hpp"

namespace gpu_libm_bench {

namespace detail {

template <typename StorageType>
inline uint64_t beginTiming(StorageType &State, StorageType DependencyMask) {
  uint64_t Timestamp = gpu::readTimer();
  StorageType Dependency = static_cast<StorageType>(Timestamp) & DependencyMask;
  State ^= Dependency;
  return Timestamp;
}

template <typename StorageType>
inline uint64_t endTiming(StorageType &State, StorageType DependencyMask) {
  // Keep the result consumer ordered before the timer.
#if defined(__NVPTX__)
  State ^= DependencyMask;
  asm volatile("" : : "llr"(State) : "memory");
#elif defined(__AMDGCN__)
  if constexpr (sizeof(StorageType) == 8) {
    uint32_t Consumed;
    asm volatile("v_or_b32 %0, %1, %2"
                 : "=v"(Consumed)
                 : "v"(static_cast<uint32_t>(State)),
                   "v"(static_cast<uint32_t>(State >> 32))
                 : "memory");
  } else
    asm volatile("v_or_b32 %0, 0, %0"
                 :
                 : "v"(static_cast<uint32_t>(State))
                 : "memory");
#else
#error "unsupported GPU target"
#endif

  return gpu::readTimer();
}

template <auto Function, typename StorageType, typename FloatType,
          typename... OtherInputTypes>
void latencyChain(uint32_t PassCount, uint32_t ChainLength,
                  StorageType DependencyMask, uint64_t *ElapsedTicks,
                  StorageType *Output, FloatType FirstInput,
                  OtherInputTypes... OtherInputs) {
  static_assert(
      __is_same(FloatType, decltype(Function(FirstInput, OtherInputs...))),
      "return type must match the first input type");
  static_assert(
      __is_same(StorageType, typename fputil::FPBits<FloatType>::StorageType),
      "mask and output storage types must match the first input type");

  const auto XBits = fputil::FPBits(FirstInput).asUInt();
  auto State = XBits;
  uint64_t Start = 0;
  uint64_t Stop = 0;

#pragma clang loop unroll(disable)
  for (uint32_t Pass = 0; Pass < PassCount; ++Pass) {
    Start = beginTiming(State, DependencyMask);

#pragma clang loop unroll(disable)
    for (uint32_t Step = 0; Step < ChainLength; ++Step) {
      StorageType DependentInput = XBits ^ StorageType(State & DependencyMask);
      auto Result =
          Function(fputil::FPBits(DependentInput).getValue(), OtherInputs...);
      State = fputil::FPBits(Result).asUInt();
    }

    Stop = endTiming(State, DependencyMask);
  }

  ElapsedTicks[0] = Stop - Start;
  Output[0] = State;
}

} // namespace detail

template <auto Function, typename StorageType, typename FloatType,
          typename... OtherInputTypes>
void latency(uint32_t PassCount, uint32_t ChainLength,
             const StorageType *ZeroMask, uint64_t *ElapsedTicks,
             StorageType *Output, const FloatType *FirstInput,
             const OtherInputTypes *...OtherInputs) {
  if (gpu::globalThreadId() != 0)
    return;

  detail::latencyChain<Function>(PassCount, ChainLength, ZeroMask[0],
                                 ElapsedTicks, Output, FirstInput[0],
                                 OtherInputs[0]...);
}

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_KERNELS_COMMON_LATENCY_HPP
