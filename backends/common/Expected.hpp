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
// Value-or-error results for native GPU backends.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_LIBM_BENCH_COMMON_EXPECTED_HPP
#define GPU_LIBM_BENCH_COMMON_EXPECTED_HPP

#include "Error.hpp"

#include <cassert>
#include <utility>
#include <variant>

namespace gpu_libm_bench {

template <typename T> class [[nodiscard]] Expected {
  std::variant<T, Error> Storage;

public:
  // NOLINTNEXTLINE(misc-explicit-constructor): Allow direct value returns.
  Expected(T Value) : Storage(std::in_place_index<0>, std::move(Value)) {}

  // NOLINTNEXTLINE(misc-explicit-constructor): Allow direct error propagation.
  Expected(Error Failure)
      : Storage(std::in_place_index<1>, std::move(Failure)) {
    assert(*std::get_if<1>(&Storage) &&
           "Expected<T> requires a value or a failure.");
  }

  explicit operator bool() const noexcept { return Storage.index() == 0; }

  T &operator*() & {
    assert(*this);
    return *std::get_if<0>(&Storage);
  }

  const T &operator*() const & {
    assert(*this);
    return *std::get_if<0>(&Storage);
  }

  T &&operator*() && { return std::move(**this); }

  T *operator->() {
    assert(*this);
    return std::get_if<0>(&Storage);
  }

  const T *operator->() const {
    assert(*this);
    return std::get_if<0>(&Storage);
  }

  Error &error() & {
    assert(!*this);
    return *std::get_if<1>(&Storage);
  }

  const Error &error() const & {
    assert(!*this);
    return *std::get_if<1>(&Storage);
  }

  Error &&error() && { return std::move(error()); }
};

template <> class [[nodiscard]] Expected<void> {
  Error Status = Error::success();

public:
  Expected() = default;
  // NOLINTNEXTLINE(misc-explicit-constructor): Allow direct error propagation.
  Expected(Error Status) : Status(std::move(Status)) {}

  explicit operator bool() const noexcept { return !Status; }

  Error &error() & {
    assert(Status);
    return Status;
  }

  const Error &error() const & {
    assert(Status);
    return Status;
  }

  Error &&error() && { return std::move(error()); }
};

} // namespace gpu_libm_bench

#endif // GPU_LIBM_BENCH_COMMON_EXPECTED_HPP
