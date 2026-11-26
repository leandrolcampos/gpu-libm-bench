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
// Error categories and messages shared by native GPU backends.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_LIBM_BENCH_COMMON_ERROR_HPP
#define GPU_LIBM_BENCH_COMMON_ERROR_HPP

#include <cstdint>
#include <string>

namespace gpu_libm_bench {

enum class ErrorKind : std::uint8_t {
  Success,
  InvalidArgument,
  Runtime,
  NotImplemented,
};

struct [[nodiscard]] Error {
  ErrorKind Kind = ErrorKind::Success;
  std::string Message;

  static Error success() { return {}; }

  explicit operator bool() const noexcept { return Kind != ErrorKind::Success; }
};

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_COMMON_ERROR_HPP
