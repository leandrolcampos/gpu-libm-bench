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
// Floating-point representation utilities for GPU kernels.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_LIBM_BENCH_KERNELS_COMMON_FPUTIL_HPP
#define GPU_LIBM_BENCH_KERNELS_COMMON_FPUTIL_HPP

#include <stdint.h>

namespace gpu_libm_bench::fputil {

namespace detail {

template <typename FloatType> struct FPStorageType;

template <> struct FPStorageType<_Float16> {
  using Type = uint16_t;
};

template <> struct FPStorageType<float> {
  using Type = uint32_t;
};

template <> struct FPStorageType<double> {
  using Type = uint64_t;
};

} // namespace detail

template <typename FloatType> class FPBits {
public:
  using StorageType = typename detail::FPStorageType<FloatType>::Type;

  constexpr explicit FPBits(FloatType Value)
      : Bits(__builtin_bit_cast(StorageType, Value)) {}

  constexpr explicit FPBits(StorageType Bits) : Bits(Bits) {}

  template <typename ValueType> explicit FPBits(ValueType) = delete;

  constexpr FloatType value() const {
    return __builtin_bit_cast(FloatType, Bits);
  }

  constexpr StorageType storage() const { return Bits; }

private:
  StorageType Bits;
};

FPBits(uint16_t) -> FPBits<_Float16>;
FPBits(uint32_t) -> FPBits<float>;
FPBits(uint64_t) -> FPBits<double>;

} // namespace gpu_libm_bench::fputil

#endif // GPU_LIBM_BENCH_KERNELS_COMMON_FPUTIL_HPP
