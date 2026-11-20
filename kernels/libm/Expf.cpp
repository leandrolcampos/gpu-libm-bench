// Copyright 2026 Leandro Lacerda Campos
// SPDX-License-Identifier: Apache-2.0
//
// Kernel entry points for expf.

#define LIBM_BIND_EXPF
#include "LibraryBindings.def"

#ifndef LIBM_NAME
#error "Missing expf binding in LibraryBindings.def"
#endif

#include "UnaryKernels.inc" // IWYU pragma: keep
