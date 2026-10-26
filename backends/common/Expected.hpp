// Copyright 2026 Leandro A. Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Value-or-error results for native GPU backends.

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
