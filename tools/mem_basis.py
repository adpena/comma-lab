"""Canonical reclaimable-aware memory basis for safety guards (CLASS 1 fix, 2026-07-17).

WHY (operator P0 bug-class sweep 2026-07-17; sister of
``admission_gate_naive_counts_reclaimable_as_committed_20260716``): dozens of probe/apply/harvest
tools guard themselves with ``free = psutil.virtual_memory().available; if free < FLOOR: abort`` to
protect the live trainer. On macOS ``psutil.virtual_memory().available`` = (free + inactive) — it
counts DIRTY ANONYMOUS pages parked in the inactive queue as "available" even though evicting them
needs swap. MEASURED live 2026-07-17 while the 76-GiB trainer ran: psutil ``.available`` = 57.3 GiB
but the truly reclaimable-without-swap figure = 13.7 GiB. A guard trusting 57.3 would ADMIT a probe
that pushes the box into swap-thrash / jetsam of the live trainer — the dangerous UNDER-protection
direction. (The opposite direction also bites: on an idle file-cache-heavy box ``.available`` can
UNDERcount reclaimable file cache, causing spurious rc=7 aborts + relaunch churn.)

THE CANONICAL BASIS is the governor's kernel-queue decomposition (``read_system_memory_snapshot`` →
``available_reclaimable_gib`` = free + file_backed + purgeable, min'd against total − committed, with
the anon/file queue-identity + non-wired bound validations). This module is a THIN, side-effect-free
adapter so every guard callsite routes through ONE reclaimable-aware source instead of raw psutil.
It degrades gracefully: governor snapshot → psutil ``.available`` → caller-supplied default, so a
non-macOS / psutil-absent host behaves exactly like the pre-fix code.

This module does NOT edit the control-plane governor; it only READS its public pure snapshot. Callers
that need refuse/admit decisions use ``conservative_free_gib``; the legacy raw psutil basis is what
the CLASS-1 preflight gate refuses outside this module.
"""

from __future__ import annotations

_GIB = 1024.0 ** 3


def _governor_snapshot():
    """Return the governor's reclaimable-aware snapshot, or None if unavailable. Lazy + isolated:
    any import / measurement failure (non-macOS, vm_stat absent, parse error) returns None so the
    caller falls through to psutil. No import-time side effects."""
    try:
        from tools.system_memory_governor import read_system_memory_snapshot
    except Exception:
        try:
            # Support ``python tools/foo.py`` invocations where ``tools`` isn't a package on the path.
            from system_memory_governor import read_system_memory_snapshot  # type: ignore
        except Exception:
            return None
    try:
        return read_system_memory_snapshot()
    except Exception:
        return None


def _psutil_available_gib() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().available) / _GIB
    except Exception:
        return None


def _psutil_used_gib() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().used) / _GIB
    except Exception:
        return None


def conservative_free_gib(default: float = float("inf")) -> float:
    """Reclaimable-without-swap free RAM in GiB — the CANONICAL safety-guard basis.

    Precedence: governor reclaimable-aware snapshot (``available_reclaimable_gib`` when
    ``reclaimable_ok``, else its validated ``available_gib``) → raw psutil ``.available`` → ``default``.
    ``default`` preserves each caller's prior "measurement unavailable" semantics (most probes used
    ``float('inf')`` = permissive-on-failure; pass ``0.0`` for fail-closed-on-failure)."""
    snap = _governor_snapshot()
    if snap is not None:
        try:
            if getattr(snap, "reclaimable_ok", False):
                return float(snap.available_reclaimable_gib)
            return float(snap.available_gib)
        except Exception:
            pass
    ps = _psutil_available_gib()
    if ps is not None:
        return ps
    return default


def true_committed_gib(default: float = 0.0) -> float:
    """Reclaimable-aware TRUE committed (needs-swap-to-evict) used RAM in GiB.

    Precedence: governor snapshot (``used_committed_gib`` when ``reclaimable_ok``, else ``used_gib``)
    → raw psutil ``.used`` → ``default``."""
    snap = _governor_snapshot()
    if snap is not None:
        try:
            if getattr(snap, "reclaimable_ok", False):
                return float(snap.used_committed_gib)
            return float(snap.used_gib)
        except Exception:
            pass
    ps = _psutil_used_gib()
    if ps is not None:
        return ps
    return default
