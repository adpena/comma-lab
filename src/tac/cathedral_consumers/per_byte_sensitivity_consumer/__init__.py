# SPDX-License-Identifier: MIT
"""Cathedral consumer for per-byte sensitivity from the master-gradient ledger.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Closes the producer→consumer orphan-signal loop for the per-byte sensitivity
signal emitted by the master-gradient extractor (Catalog #327,
``tac.master_gradient``). Sister of the 8 Cable D per-pair consumer packages
landed 2026-05-19 (commit ``418698227``) per
``feedback_master_gradient_consumer_cathedral_wire_in_landed_20260519.md``.

The producer (``tac.master_gradient``) appends per-archive rows to
``.omx/state/master_gradient_anchors.jsonl`` with
``gradient_tensor_kind="aggregate_per_byte_v1"`` and ``gradient_array_path``
pointing at an (N_bytes, 3) ``.npy`` array. The Cable D consumers wire the
per-PAIR signals (Pareto envelope / KKT residuals / Lagrangian λ_R / LoRA
supervision / coding-budget / engineered correction / Volterra cross terms
/ gradient-informed decoder pruning). THIS consumer wires the per-BYTE
signal (which bytes carry the score weight; canonical L1-sum-of-abs
importance per ``master_gradient_consumers._compute_aggregate_per_byte_importance``).

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" + Catalog #287 / #323:
the cathedral contribution is observability-only ([predicted] axis,
promotable=False, predicted_delta_adjustment=0.0). Per-byte sensitivity
INFORMS ranking (which bytes a future mutation campaign should target)
but does NOT directly adjust score predictions — that contribution is
deferred to the sister cascade pattern (Catalog #319 / #322 deliverability
proof + composition_alpha v2 cascade) when an empirical anchor binds the
per-byte signal to a measured ΔS.

THIS wrapper exposes the canonical CathedralConsumerContract so the
auto-discovery loop ingests the surface (Catalog #335 paradigm shift; no
manual cathedral_autopilot edits required).

Hook numbers per Catalog #125 6-hook wire-in declaration:

- Hook #1 SENSITIVITY_MAP — per-byte L1-importance IS the sensitivity-map
  contribution (which bytes carry score weight per CLAUDE.md "Apples-to-
  apples evidence discipline" annotation surface).
- Hook #3 BIT_ALLOCATOR — top-K sensitive byte indices ORDER allocation
  priority for any future mutation campaign (canonical fan-out target).
- Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH — this wrapper surfaces the signal
  to the autopilot ranker as observability annotation.

Hooks #2 PARETO_CONSTRAINT / #5 CONTINUAL_LEARNING_POSTERIOR / #6
PROBE_DISAMBIGUATOR = N/A (per-pair consumers cover #2; the producer's
canonical persistence covers #5; the per-byte signal is not a probe
disambiguator on its own — that role belongs to KKT residuals consumer #12).

Sister of:

- :mod:`tac.master_gradient_per_byte_consumer` — canonical reader helper
- :mod:`tac.master_gradient` — producer of anchors (Catalog #327)
- :mod:`tac.cathedral_consumers.per_pair_pareto_envelope_consumer` —
  per-pair sister (consumer template reference)
- :mod:`tac.cathedral_consumers.atom_consumer` — canonical observability
  pattern reference
- :mod:`tac.cathedral.consumer_contract` — Catalog #265 / #335 contract
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "per_byte_sensitivity_consumer"
CONSUMER_VERSION = "1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Per-byte sensitivity anchors are already persisted via the canonical
    ``.omx/state/master_gradient_anchors.jsonl`` fcntl-locked JSONL store
    (``tac.master_gradient.update_from_anchor``). No additional posterior
    update is required here. NO-OP by design per the Cable D pattern.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Routing logic:

    1. Extract ``archive_sha256`` from the candidate (any of common fields:
       ``archive_sha256`` / ``archive_sha`` / ``sha256`` / ``sha``).
    2. If no SHA available → return no-signal verdict (rationale cites
       missing SHA so the operator can audit which candidates lack the
       routing key).
    3. Route through :func:`tac.master_gradient_per_byte_consumer.
       load_per_byte_sensitivity_for_archive` to pull the latest per-byte
       payload for this archive (returns ``None`` if no anchor exists).
    4. If no payload → return no-signal verdict citing the absent anchor
       (per CLAUDE.md "Apples-to-apples evidence discipline": absence of
       signal is itself a signal — the operator can prioritize extracting
       a master-gradient anchor for this archive).
    5. Surface payload summary stats + top-K byte indices in the verdict's
       rationale + ``per_byte_sensitivity`` notes block.

    Per Catalog #287 / #323: ``predicted_delta_adjustment=0.0`` ALWAYS
    (observability-only contribution); ``promotable=False`` ALWAYS; the
    payload's ``measurement_axis`` is propagated into the verdict's
    ``axis_tag`` so downstream consumers (autopilot ranker / per-substrate
    dispatch router) honor it for routing decisions.

    Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #127 / #192: when the
    payload's hardware substrate is advisory (macOS / darwin / MPS), the
    verdict's ``axis_tag`` becomes ``[macOS-CPU advisory]`` rather than
    ``[predicted]`` so the routing decision is honest about the source.
    """
    # Step 1: extract archive SHA from candidate via common field aliases.
    archive_sha = _extract_archive_sha(candidate)
    if not archive_sha:
        return _no_signal_verdict(
            "candidate missing archive_sha256 / archive_sha / sha256 field; "
            "per-byte sensitivity routing key unavailable"
        )

    # Step 2: route through the canonical reader.
    try:
        from tac.master_gradient_per_byte_consumer import (
            load_per_byte_sensitivity_for_archive,
            summarize_payload,
        )
    except ImportError:
        return _no_signal_verdict(
            "tac.master_gradient_per_byte_consumer unavailable (numpy not "
            "installed or canonical helper missing); per-byte signal unrouteable"
        )

    payload = load_per_byte_sensitivity_for_archive(archive_sha, top_k=100)
    if payload is None:
        return _no_signal_verdict(
            f"no per-byte sensitivity anchor found for archive "
            f"{archive_sha[:12]}; consider extracting via "
            "tools/extract_master_gradient.py + master_gradient_anchors.jsonl"
        )

    # Step 3: derive axis_tag honoring the payload's hardware substrate.
    axis_tag = _resolve_axis_tag(payload.measurement_hardware, payload.measurement_axis)

    summary = summarize_payload(payload)
    rationale = (
        f"per-byte sensitivity available for {archive_sha[:12]}: "
        f"{summary['n_bytes']} bytes ({summary['n_bytes_above_zero']} non-zero, "
        f"{summary['sparsity_pct']:.1f}% sparse), top-{summary['top_k_count']} "
        f"indices ranked by L1-sum-of-abs importance "
        f"(aggregate_sum={summary['aggregate_l1_importance_sum']:.4g}); "
        f"hardware={payload.measurement_hardware} axis={payload.measurement_axis} "
        f"[predicted]"
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": axis_tag,
        "promotable": False,
        "confidence": 0.0,
        "notes": {
            "per_byte_sensitivity": {
                "archive_sha256_prefix": archive_sha[:12],
                "n_bytes": payload.n_bytes,
                "n_bytes_above_zero": payload.n_bytes_above_zero,
                "sparsity_pct": summary["sparsity_pct"],
                "aggregate_l1_importance_sum": payload.aggregate_l1_importance_sum,
                "top_k_count": len(payload.top_k_sensitivity_indices),
                "top_k_indices_first_10": list(
                    payload.top_k_sensitivity_indices[:10]
                ),
                "measurement_axis": payload.measurement_axis,
                "measurement_hardware": payload.measurement_hardware,
                "measurement_method": payload.measurement_method,
                "measurement_utc": payload.measurement_utc,
                "gradient_array_path": payload.gradient_array_path,
            }
        },
    }


# ─────────────────────────────────────────────────────────────────────────
# Helpers (private to this consumer)
# ─────────────────────────────────────────────────────────────────────────


_CANDIDATE_SHA_FIELDS = (
    "archive_sha256",
    "archive_sha",
    "sha256",
    "sha",
    "scored_archive_sha256",
)


def _extract_archive_sha(candidate: Mapping[str, Any]) -> str | None:
    """Extract a usable archive SHA from a cathedral candidate dict."""
    if not isinstance(candidate, Mapping):
        return None
    for field_name in _CANDIDATE_SHA_FIELDS:
        value = candidate.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _resolve_axis_tag(hardware: str, axis: str) -> str:
    """Resolve the verdict's axis_tag honoring hardware advisory class.

    Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #127 / #192: if the
    underlying anchor was extracted on macOS / darwin / MPS hardware the
    verdict carries ``[macOS-CPU advisory]`` rather than the producer's
    declared axis. This keeps the routing honest about the source so a
    downstream promotion guard refuses to treat advisory data as contest-
    axis evidence.
    """
    hw_lower = (hardware or "").lower()
    advisory_tokens = ("darwin", "macos", "mps", "advisory")
    if any(token in hw_lower for token in advisory_tokens):
        return "[macOS-CPU advisory]"
    axis_stripped = (axis or "").strip()
    if axis_stripped:
        return axis_stripped
    return "[predicted]"


def _no_signal_verdict(reason: str) -> dict[str, Any]:
    """Canonical zero-signal verdict for the no-payload / no-SHA branches."""
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": f"per_byte_sensitivity_consumer: {reason} [predicted]",
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }


__all__ = (
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
    "consume_candidate",
    "update_from_anchor",
)
