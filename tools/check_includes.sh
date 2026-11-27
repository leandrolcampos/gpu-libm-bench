#!/bin/sh
# Reports include changes for C++ sources using the selected compilation database.

set -eu

build_dir=$1
shift

find "$@" -type f -name '*.cpp' -exec sh -ec '
  build_dir=$1
  shift
  for source_file in "$@"; do
    printf "%s\n" "$source_file"
    clang-include-cleaner --print=changes -p "$build_dir" "$source_file"
  done
' sh "$build_dir" '{}' +
