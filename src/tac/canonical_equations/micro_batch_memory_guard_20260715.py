# SPDX-License-Identifier: MIT
"""Canonical equation surface for the V9 micro-batch launch-memory guard.

This is a launch-safety equation, not a memory measurement model.  #261 measured complete process
peaks of 5,907 MiB at B=1 and 5,918 MiB at B=4 on an n=8 diagnostic.  A current-V9 B=2 n600 actual
RSS observation is not available under the $0/no-GPU gate.  Rather than reuse the serial B=1
projection, the guard charges the *entire* measured B=4 process peak for every additional pair.
That is intentionally much larger than the observed 11 MiB B4-B1 delta.

Equation::

    G(B) = max(0, B - 1) * (5918 MiB / 1024)
    M_guarded(B) = M_serial + G(B)

The dependency-free implementation lives in :mod:`tac.micro_batch_memory_guard` so the standalone
governor does not pull the canonical-equations registry's optional scientific dependencies into its
preflight process. The equation certifies only a conservative projected envelope. It cannot be
called an actual B2 RSS measurement or a current-V9 throughput result.
"""

from __future__ import annotations

from tac.micro_batch_memory_guard import (
    EQUATION_ID,
    MEASURED_B1_N8_PROCESS_PEAK_MIB,
    MEASURED_B4_N8_PROCESS_PEAK_MIB,
    MICRO_BATCH_EXTRA_PAIR_GUARD_GIB,
    MICRO_BATCH_GUARD_PROVENANCE,
    REQ_R,
    VERDICT_SCOPE,
    MicroBatchMemoryGuardReceipt,
    build_guard_receipt,
    micro_batch_guard_gib,
)

__all__ = [
    "EQUATION_ID",
    "MEASURED_B1_N8_PROCESS_PEAK_MIB",
    "MEASURED_B4_N8_PROCESS_PEAK_MIB",
    "MICRO_BATCH_EXTRA_PAIR_GUARD_GIB",
    "MICRO_BATCH_GUARD_PROVENANCE",
    "REQ_R",
    "VERDICT_SCOPE",
    "MicroBatchMemoryGuardReceipt",
    "build_guard_receipt",
    "micro_batch_guard_gib",
]
