// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Tests for native result propagation and resource ownership.

#include "Expected.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <type_traits>

using namespace gpu_libm_bench;

static void check(bool Condition, const char *Message) {
  if (!Condition) {
    std::fprintf(stderr, "%s\n", Message);
    std::exit(EXIT_FAILURE);
  }
}

struct CountingDeleter {
  int *Releases;

  void operator()(int *Pointer) const {
    ++*Releases;
    delete Pointer;
  }
};

using Resource = std::unique_ptr<int, CountingDeleter>;

static_assert(!std::is_copy_constructible_v<Expected<Resource>>);
static_assert(!std::is_copy_assignable_v<Expected<Resource>>);
static_assert(std::is_nothrow_move_constructible_v<Expected<Resource>>);
static_assert(std::is_nothrow_move_assignable_v<Expected<Resource>>);

static void testErrorPropagation() {
  Error Success = Error::success();
  check(!Success, "a successful status must not signal an error");
  Expected<void> Completed = Success;
  check(static_cast<bool>(Completed),
        "a successful status must produce a successful void result");

  Error Err{ErrorKind::InvalidArgument, "invalid device"};
  check(static_cast<bool>(Err), "a failed status must signal an error");

  Expected<void> Failure = std::move(Err);
  check(!Failure, "an error must produce a failed void result");

  Expected<Resource> Propagated = std::move(Failure).error();
  check(!Propagated, "propagation must preserve failure for a resource result");
  check(Propagated.error().Kind == ErrorKind::InvalidArgument &&
            Propagated.error().Message == "invalid device",
        "propagation must preserve the error kind and message");
}

static void testUnclaimedResource() {
  int Releases = 0;

  {
    Expected<Resource> Result =
        Resource(new int(7), CountingDeleter{&Releases});
    check(Result && **Result == 7, "a resource result must contain its value");
  }

  check(Releases == 1, "an unclaimed resource must be released exactly once");
}

static void testResourceTransfer() {
  int Releases = 0;
  Resource Owner;

  {
    Expected<Resource> Original =
        Resource(new int(7), CountingDeleter{&Releases});
    Expected<Resource> Moved = std::move(Original);
    Owner = *std::move(Moved);
  }

  check(Releases == 0 && Owner && *Owner == 7,
        "moving results and extracting a resource must transfer ownership");

  Owner.reset();
  check(Releases == 1, "the extracted resource must be released exactly once");
}

static void testResourceReplacement() {
  int Releases = 0;

  {
    Expected<Resource> First = Resource(new int(1), CountingDeleter{&Releases});
    Expected<Resource> Second =
        Resource(new int(2), CountingDeleter{&Releases});

    Second = std::move(First);
    check(Releases == 1 && Second && **Second == 1,
          "move assignment must release the replaced resource");

    Second = Error{ErrorKind::Runtime, "operation failed"};
    check(Releases == 2 && !Second,
          "replacing a value with an error must release its resource");
  }

  check(Releases == 2,
        "moved and replaced resources must not be released twice");
}

int main() {
  testErrorPropagation();
  testUnclaimedResource();
  testResourceTransfer();
  testResourceReplacement();
}
