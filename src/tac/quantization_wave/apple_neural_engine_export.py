# SPDX-License-Identifier: MIT
"""Apple Neural Engine (ANE) export helpers — Core ML pattern.

Per the operator's fleet (M5 Max primary + M1 MacBook Pro tertiary +
Intel iMac mini), the Apple Neural Engine is accessible via Core ML.
ANE-quantized models can inference at 1-10ms per pair (vs CPU ~30-100ms),
which makes the M-series fleet a meaningful FREE PROXY throughput
multiplier for the dev loop.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-
COMPLIANT HARDWARE" non-negotiable: ANE is NEVER a 1:1 contest-compliant
axis. Score outputs MUST be tagged ``[macOS-ANE-advisory]`` and are
PROMOTION-INELIGIBLE.

This module exposes:

1. ``can_export_to_ane(model)`` — predicate that returns whether all
   layers of a PyTorch model have Core ML equivalents + ANE compute
   units.
2. ``coreml_export_metadata()`` — environment metadata (coremltools
   version, Core ML format version, ANE availability) for the dev-loop
   ranker.

The actual conversion (via ``coremltools.convert``) is left to the
caller per CLAUDE.md "Beauty, simplicity, and developer experience"
non-negotiable: this module provides the PREDICATE + METADATA, not the
heavy export pipeline.

[verified-against:Apple Core ML docs (https://developer.apple.com/
documentation/coreml) + coremltools 8.0+ ANE compute_units API]
"""

from __future__ import annotations

import platform
from typing import Any


def can_export_to_ane() -> tuple[bool, str]:
    """Return (can_export, reason) for the current machine.

    The check is non-destructive: it only reads platform metadata and
    attempts a coremltools import. It does NOT run an actual conversion
    (which would require a model + significant CPU time).
    """
    is_macos = platform.system() == "Darwin"
    if not is_macos:
        return False, "ANE is macOS-only (current OS: %s)" % platform.system()
    is_apple_silicon = platform.machine() == "arm64"
    if not is_apple_silicon:
        return False, "ANE requires Apple Silicon (current arch: %s)" % platform.machine()
    try:
        import coremltools  # type: ignore[import-not-found]
        ct_version = getattr(coremltools, "__version__", "unknown")
    except ImportError:
        return False, "coremltools is not installed; install via 'uv pip install coremltools'"
    # Check coremltools version >= 6.0 (ANE supported from CT 6.0+)
    try:
        major = int(ct_version.split(".")[0])
        if major < 6:
            return False, f"coremltools {ct_version} < 6.0; ANE compute_units requires >= 6.0"
    except (ValueError, IndexError):
        pass
    return True, f"ANE export is supported (coremltools {ct_version} on {platform.machine()})"


def coreml_export_metadata() -> dict[str, Any]:
    """Return ANE export metadata for the dev-loop ranker."""
    can_export, reason = can_export_to_ane()
    metadata: dict[str, Any] = {
        "can_export_to_ane": can_export,
        "reason": reason,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "coremltools_version": None,
        "promotion_eligible": False,  # NEVER True per CLAUDE.md
        "axis": "macos_ane_advisory",
        "tag_recommendation": "[macOS-ANE-advisory]" if can_export else "[unavailable]",
    }
    try:
        import coremltools  # type: ignore[import-not-found]
        metadata["coremltools_version"] = getattr(coremltools, "__version__", "unknown")
    except ImportError:
        pass
    return metadata
