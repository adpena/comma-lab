"""Comma-lab adapters for canonical ``tac.preflight`` checks."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "check_42_train_inference_parity",
    "check_dispatch_cli_shell_hazards",
    "check_feature_flags_have_live_objective_effect",
    "check_no_mps_fallback_default",
    "check_public_release_hygiene",
    "emit_catalog",
    "preflight_all",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    strict_checks = import_module("comma_lab.preflight.strict_checks")
    return getattr(strict_checks, name)


def __dir__() -> list[str]:
    return sorted(__all__)
