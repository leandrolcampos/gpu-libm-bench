# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Throughput execution and result aggregation for experiments."""

from ._experiment import (
    ThroughputDomainResult,
    ThroughputResults,
    run,
    validate_results,
)

__all__ = [
    "ThroughputDomainResult",
    "ThroughputResults",
    "run",
    "validate_results",
]
