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
// Helper macros for math kernels.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_LIBM_BENCH_KERNELS_LIBM_MACROS_HPP
#define GPU_LIBM_BENCH_KERNELS_LIBM_MACROS_HPP

#define LIBM_KERNEL_NAME_IMPL(Name, Suffix) Name##Suffix
#define LIBM_KERNEL_NAME(Name, Suffix) LIBM_KERNEL_NAME_IMPL(Name, Suffix)

#endif // GPU_LIBM_BENCH_KERNELS_LIBM_MACROS_HPP
