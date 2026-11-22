# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Throughput search and confirmation for benchmark experiments."""

from ._confirmation import CONFIRMATION_REPETITIONS, PLATEAU_MARGIN
from ._experiment import (
    ThroughputDomainResult,
    ThroughputResults,
    run,
    validate_results,
)

__all__ = [
    "CONFIRMATION_REPETITIONS",
    "PLATEAU_MARGIN",
    "ThroughputDomainResult",
    "ThroughputResults",
    "run",
    "validate_results",
]
