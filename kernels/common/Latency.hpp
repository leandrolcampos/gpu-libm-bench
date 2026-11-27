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
// Latency measurement wrappers for GPU math functions.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_LIBM_BENCH_KERNELS_COMMON_LATENCY_HPP
#define GPU_LIBM_BENCH_KERNELS_COMMON_LATENCY_HPP

#include "FPUtil.hpp"
#include "GPU.hpp"

#include <stdint.h>

namespace gpu_libm_bench {

namespace detail {

template <typename FirstType, typename... OtherTypes>
constexpr FirstType &&getFirstArgument(FirstType &&First,
                                       OtherTypes &&...) noexcept {
  return static_cast<FirstType &&>(First);
}

// Ensure the start timestamp is ready before the chain starts.
template <typename StorageType>
inline uint64_t beginTiming(StorageType &State, StorageType DependencyMask) {
  const uint64_t Timestamp = gpu::readSteadyCounter();
  const StorageType Dependency =
      static_cast<StorageType>(Timestamp) & DependencyMask;
  State ^= Dependency;

  return Timestamp;
}

// Keep each step dependent on the previous result even for the identity.
// A zero mask selects every bit from the original input.
inline uint32_t makeDependentWord(uint32_t Input, uint32_t State,
                                  uint32_t Mask) {
  uint32_t Selected;

#if defined(__AMDGCN__)
  asm volatile("v_bfi_b32 %0, %1, %2, %3"
               : "=v"(Selected)
               : "s"(Mask), "v"(State), "v"(Input));
#elif defined(__NVPTX__)
  asm volatile("lop3.b32 %0, %1, %2, %3, 0xca;"
               : "=r"(Selected)
               : "r"(Mask), "r"(State), "r"(Input));
#else
#error "unsupported GPU target"
#endif

  return Selected;
}

template <typename StorageType>
inline StorageType makeDependent(StorageType Input, StorageType State,
                                 StorageType Mask) {
  const uint32_t Low =
      makeDependentWord(uint32_t(Input), uint32_t(State), uint32_t(Mask));

  if constexpr (sizeof(StorageType) == sizeof(uint64_t)) {
    const uint32_t High =
        makeDependentWord(Input >> 32U, State >> 32U, Mask >> 32U);
    return (uint64_t(High) << 32U) | Low;
  }

  return static_cast<StorageType>(Low);
}

// Ensure the end timestamp is read after the chain finishes.
template <typename StorageType>
inline uint64_t endTiming(StorageType &State,
                          [[maybe_unused]] StorageType DependencyMask) {
#if defined(__AMDGCN__)
  if constexpr (sizeof(StorageType) == sizeof(uint64_t)) {
    uint32_t Consumed;
    asm volatile("v_or_b32 %0, %1, %2"
                 : "=v"(Consumed)
                 : "v"(uint32_t(State)), "v"(uint32_t(State >> 32U))
                 : "memory");
  } else {
    asm volatile("v_or_b32 %0, 0, %0" : : "v"(uint32_t(State)) : "memory");
  }
#elif defined(__NVPTX__)
  // Use a runtime zero mask because ptxas can eliminate the AMD-style OR.
  State ^= DependencyMask;

  // NOLINTNEXTLINE(bugprone-branch-clone)
  if constexpr (sizeof(StorageType) == sizeof(uint64_t)) {
    asm volatile("" : : "l"(State) : "memory");
  } else if constexpr (sizeof(StorageType) == sizeof(uint32_t)) {
    asm volatile("" : : "r"(State) : "memory");
  } else {
    asm volatile("" : : "h"(State) : "memory");
  }
#else
#error "unsupported GPU target"
#endif

  return gpu::readSteadyCounter();
}

template <auto Function, typename StorageType, typename... InputTypes>
void measureChain(uint32_t PassCount, uint32_t ChainLength,
                  StorageType DependencyMask, uint64_t *ElapsedTicks,
                  StorageType *Output, InputTypes... Inputs) {
  auto &FirstInput = getFirstArgument(Inputs...);
  using ReturnType = decltype(Function(Inputs...));

  static_assert(__is_same(decltype(FirstInput), ReturnType &),
                "return type must match the first input type");
  static_assert(__is_same(ReturnType, _Float16) ||
                    __is_same(ReturnType, float) ||
                    __is_same(ReturnType, double),
                "first input type must be _Float16, float, or double");

  using FPBits = fputil::FPBits<ReturnType>;
  static_assert(__is_same(StorageType, typename FPBits::StorageType),
                "mask and output storage type must match the first input type");

  const auto FirstInputUInt = FPBits(FirstInput).asUInt();
  auto State = FirstInputUInt;

  uint64_t Start = 0;
  uint64_t Stop = 0;

#pragma clang loop unroll(disable)
  for (uint32_t Pass = 0; Pass < PassCount; ++Pass) {
    Start = beginTiming(State, DependencyMask);

#pragma clang loop unroll_count(4)
    for (uint32_t Step = 0; Step < ChainLength; ++Step) {
      const auto DependentInput =
          makeDependent(FirstInputUInt, State, DependencyMask);
      FirstInput = FPBits(DependentInput).getValue();
      State = FPBits(Function(Inputs...)).asUInt();
    }

    Stop = endTiming(State, DependencyMask);
  }

  ElapsedTicks[0] = Stop - Start;
  Output[0] = State;
}

} // namespace detail

// The mixed ABI is intentional: ZeroMask is used on every chain step, so keep
// it indirect to encourage a single load before the timed loop.
template <auto Function, typename StorageType, typename... InputTypes>
void measureLatency(uint32_t PassCount, uint32_t ChainLength,
                    const StorageType *ZeroMask, uint64_t *ElapsedTicks,
                    StorageType *Output, const InputTypes *...Inputs) {
  if (gpu::globalThreadId() != 0)
    return;

  detail::measureChain<Function>(PassCount, ChainLength, ZeroMask[0],
                                 ElapsedTicks, Output, Inputs[0]...);
}

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_KERNELS_COMMON_LATENCY_HPP
