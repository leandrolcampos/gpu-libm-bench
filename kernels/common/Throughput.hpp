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
// Throughput measurement wrappers for GPU math functions.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_LIBM_BENCH_KERNELS_COMMON_THROUGHPUT_HPP
#define GPU_LIBM_BENCH_KERNELS_COMMON_THROUGHPUT_HPP

#include "FPUtil.hpp"
#include "GPU.hpp"

#include <stdint.h>

namespace gpu_libm_bench {

namespace detail {

// Load each group's inputs once so all passes can reuse them.
template <typename T, unsigned Size> class InputGroup {
public:
  InputGroup(const T *Data, uint64_t Stride) {
#pragma clang loop unroll(full)
    for (unsigned Index = 0; Index < Size; ++Index)
      Values[Index] = Data[Index * Stride];
  }

  T operator[](unsigned Index) const { return Values[Index]; }

private:
  T Values[Size];
};

template <typename T> T makeDependent(T Value, uint32_t Mask) {
  if constexpr (__is_same(T, int) || __is_same(T, unsigned int) ||
                __is_same(T, long)) {
    using StorageType = __make_unsigned(T);
    const auto ValueUInt = __builtin_bit_cast(StorageType, Value);
    return __builtin_bit_cast(T, ValueUInt ^ static_cast<StorageType>(Mask));
  } else if constexpr (__is_same(T, _Float16) || __is_same(T, float) ||
                       __is_same(T, double)) {
    using FPBits = fputil::FPBits<T>;
    using StorageType = typename FPBits::StorageType;

    const auto ValueUInt = FPBits(Value).storage();
    const auto DependentUInt =
        static_cast<StorageType>(ValueUInt ^ static_cast<StorageType>(Mask));
    return FPBits(DependentUInt).value();
  } else {
    static_assert(!__is_same(T, T), "unsupported throughput input type");
  }
}

template <auto Function, unsigned Size, typename StorageType,
          typename... InputTypes>
void evaluateGroup(uint32_t PassCount, uint32_t ZeroMask,
                   StorageType (&Checksums)[Size],
                   const InputGroup<InputTypes, Size> &...Inputs) {
  using FPBits = fputil::FPBits<decltype(Function(Inputs[0]...))>;
  static_assert(__is_same(StorageType, typename FPBits::StorageType),
                "checksum storage type must match the storage type of the "
                "function result");

  // Keep one sum per input so the unrolled calls have independent accumulation
  // chains. And use addition across passes because XOR would cancel identical
  // results in pairs.
  StorageType Sums[Size] = {};

#pragma clang loop unroll(disable)
  for (uint32_t Pass = 0; Pass < PassCount; ++Pass) {
    // The runtime zero mask preserves the input and keeps the evaluations
    // inside the loop after device linking and assembly.
    const uint32_t PassMask = Pass & ZeroMask;

#pragma clang loop unroll(full)
    for (unsigned Index = 0; Index < Size; ++Index)
      Sums[Index] +=
          FPBits(Function(makeDependent(Inputs[Index], PassMask)...)).storage();
  }

#pragma clang loop unroll(full)
  for (unsigned Index = 0; Index < Size; ++Index)
    Checksums[Index] ^= Sums[Index];
}

} // namespace detail

template <auto Function, unsigned UnrollFactor, typename StorageType,
          typename... InputTypes>
void measureThroughput(uint32_t PassCount, uint32_t ZeroMask, uint64_t Count,
                       StorageType *Checksums, const InputTypes *...Inputs) {
  static_assert(UnrollFactor > 0, "unroll factor must be positive");
  static_assert(sizeof...(InputTypes) > 0, "at least one input is required");

  const uint64_t ThreadId = gpu::globalThreadId();
  const uint64_t Stride = gpu::gridSize() * gpu::blockSize();

  // Give each unrolled call its own accumulator to avoid a shared dependency
  // chain.
  StorageType Accumulators[UnrollFactor] = {};
  uint64_t Index = ThreadId;

#pragma clang loop unroll(disable)
  for (; Index < Count && Count - Index > (UnrollFactor - 1) * Stride;
       Index += UnrollFactor * Stride)
    detail::evaluateGroup<Function>(
        PassCount, ZeroMask, Accumulators,
        detail::InputGroup<InputTypes, UnrollFactor>(Inputs + Index,
                                                     Stride)...);

  StorageType Checksum[1] = {};
#pragma clang loop unroll(full)
  for (unsigned AccumulatorIndex = 0; AccumulatorIndex < UnrollFactor;
       ++AccumulatorIndex)
    Checksum[0] ^= Accumulators[AccumulatorIndex];

#pragma clang loop unroll(disable)
  for (; Index < Count; Index += Stride)
    detail::evaluateGroup<Function>(
        PassCount, ZeroMask, Checksum,
        detail::InputGroup<InputTypes, 1>(Inputs + Index, Stride)...);

  Checksums[ThreadId] = Checksum[0];
}

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_KERNELS_COMMON_THROUGHPUT_HPP
