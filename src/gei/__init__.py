# Copyright 2026 Hossein Jamali and Frederick C. Harris, Jr.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# SPDX-License-Identifier: Apache-2.0
"""Godwit Endurance Inference (GEI).

An AI model that shrinks its own weights during a single generation, on a
schedule planned before launch, to free memory for its growing context.

Inspired by the Bar-tailed Godwit, which converts its own body tissue to fuel
mid-flight and decides before departure whether the crossing is possible.

Layout
------
envelope      shared memory/energy envelope; the W + K <= M constraint
length_prior  belief over task length T (unknown at launch)
schedule      dynamic program for the burn schedule
launch        launch check and go/no-go
elastic       nested-model backend interface, CPU mock, HF backend
controller    in-flight controller with fuel-mix switching
metrics       switch shock, accuracy, calibration
"""

from .controller import Binding, FlightLog, binding_constraint, fly
from .elastic import ElasticModel, HFElasticModel, KVPolicy, MockElasticModel
from .envelope import GB, MB, Envelope, ModelConfig, range_gain, validate_family
from .launch import LaunchDecision, launch_check
from .length_prior import LengthPrior
from .schedule import BurnSchedule, reactive_schedule, solve_schedule

__version__ = "0.1.0"

__all__ = [
    "GB",
    "MB",
    "Binding",
    "BurnSchedule",
    "ElasticModel",
    "Envelope",
    "FlightLog",
    "HFElasticModel",
    "KVPolicy",
    "LaunchDecision",
    "LengthPrior",
    "MockElasticModel",
    "ModelConfig",
    "binding_constraint",
    "fly",
    "launch_check",
    "range_gain",
    "reactive_schedule",
    "solve_schedule",
    "validate_family",
]
