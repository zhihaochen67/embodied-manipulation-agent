"""Structured deterministic scenarios; batch benchmarking is deferred."""

from .scenarios import (
    CubeSpec,
    Scenario,
    TraySpec,
    generate_scenario,
    validate_scenario,
)

__all__ = [
    "CubeSpec",
    "Scenario",
    "TraySpec",
    "generate_scenario",
    "validate_scenario",
]
