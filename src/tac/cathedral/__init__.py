# SPDX-License-Identifier: MIT
"""Canonical cathedral autopilot consumer contract namespace.

Per operator directive 2026-05-19 verbatim: *"What if we change the paradigm
by making cathedral autopilot ingest by default if within a certain directory
and exposing/respecting a certain contract or schema. Fix permanently and self
protect against"*.

This package exposes the **canonical contract** every package in
``src/tac/cathedral_consumers/`` must satisfy in order to be auto-discovered
and auto-registered by ``tools/cathedral_autopilot_autonomous_loop.py``.

The PARADIGM SHIFT: convention-over-configuration extincts the orphan-signal
class permanently. NEW packages land in ``src/tac/cathedral_consumers/`` +
expose ``CathedralConsumerContract`` + auto-discovery loop ingests them
WITHOUT manual wire-in. STRICT preflight gate Catalog #335 refuses
non-compliant landings.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable: this is THE structural extinction of the
orphan-signal-at-cathedral-autopilot bug class (12 NEW tac.* namespaces this
window; ZERO consumed by cathedral_autopilot — per wiring+integration audit
commit 3821cfb6b 2026-05-19).

Cross-references
----------------
- :mod:`tac.cathedral.consumer_contract` — the canonical Protocol + dataclasses
- :mod:`tac.cathedral_consumers` — the directory where consumers land
- :mod:`tools.cathedral_autopilot_autonomous_loop` — the auto-discovery loop
- CLAUDE.md "Catalog #265 canonical contract pattern" (sister design)
- CLAUDE.md "Catalog #125 6-hook wire-in" (this contract IS hook #4)
- CLAUDE.md "Meta-Lagrangian/Pareto solver" (no orphan signals discipline)
- ``feedback_cathedral_auto_ingest_paradigm_shift_landed_20260519``
"""
from __future__ import annotations

from tac.cathedral.consumer_contract import (
    CathedralConsumerContract,
    CathedralConsumerContractError,
    ConsumerRegistration,
    HookNumber,
    WAIVER_TOKEN,
    discover_waiver_in_init,
    validate_consumer_module,
)

__all__ = [
    "CathedralConsumerContract",
    "CathedralConsumerContractError",
    "ConsumerRegistration",
    "HookNumber",
    "WAIVER_TOKEN",
    "discover_waiver_in_init",
    "validate_consumer_module",
]
