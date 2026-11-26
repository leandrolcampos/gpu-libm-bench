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
// Kernel entry points for the _Float16 identity.
//
//===----------------------------------------------------------------------===//

extern "C" inline _Float16 gpu_libm_bench_idf16(_Float16 X) { return X; }

#define LIBM_NAME idf16
#define LIBM_SYMBOL gpu_libm_bench_idf16
#define LIBM_ARG_TYPE _Float16
#define LIBM_RETURN_TYPE _Float16

#include "UnaryKernels.inc" // IWYU pragma: keep
