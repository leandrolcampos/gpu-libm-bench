// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Recoverable errors shared by native GPU backends.

#ifndef GPU_LIBM_BENCH_COMMON_ERROR_HPP
#define GPU_LIBM_BENCH_COMMON_ERROR_HPP

#include <cstdint>
#include <string>

namespace gpu_libm_bench {

enum class ErrorKind : std::uint8_t { Success, InvalidArgument, Runtime };

struct [[nodiscard]] Error {
  ErrorKind Kind = ErrorKind::Success;
  std::string Message;

  static Error success() { return {}; }

  explicit operator bool() const noexcept { return Kind != ErrorKind::Success; }
};

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_COMMON_ERROR_HPP
