# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Probability distributions for benchmark inputs."""

from .base import Distribution
from .uniform import Uniform

__all__ = ["Distribution", "Uniform"]
