# SPDX-License-Identifier: MIT
"""Cathedral consumer for per-pair bit-allocation guidance.

Per Catalog #335 + :mod:`tac.cathedral.consumer_contract`, this package is
auto-discoverable by the cathedral autopilot consumer sweep. The consumer
surfaces per-pair master-gradient sensitivity as hook #3 bit-allocator
observability only: it ranks byte indices by absolute sensitivity so a later
bit-allocation pass can decide where extra bytes are most defensible.

The directive cited ``tac.master_gradient_consumers.load_master_gradient_for_archive``.
That helper is not present in this checkout, so this implementation adapts to
the live canonical helper:
``tac.master_gradient_consumers.load_per_pair_gradient_from_anchor``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "bit_allocator_per_pair_consumer"
CONSUMER_VERSION = "1.0"
CONSUMER_HOOK_NUMBERS = (HookNumber.BIT_ALLOCATOR,)

DEFAULT_TOP_K_BYTES = 16
_REQUESTED_HELPER_NAME = "load_master_gradient_for_archive"
_FALLBACK_HELPER_NAME = "load_per_pair_gradient_from_anchor"

_CANDIDATE_SHA_FIELDS = (
    "archive_sha256",
    "archive_sha",
    "sha256",
    "sha",
    "scored_archive_sha256",
)

_LAST_UPDATE_FROM_ANCHOR: dict[str, Any] | None = None


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 update hook callable surface.

    This consumer does not persist a separate posterior. It opportunistically
    reads the per-pair master-gradient anchor so invocation failures are visible
    during update paths without creating a score claim or changing dispatch
    ranking.
    """
    global _LAST_UPDATE_FROM_ANCHOR

    archive_sha = _extract_archive_sha(anchor)
    if not archive_sha:
        _LAST_UPDATE_FROM_ANCHOR = {
            "loaded": False,
            "reason": "anchor missing archive_sha256",
        }
        return

    loaded = _load_per_pair_gradient_for_archive(archive_sha)
    if loaded is None:
        _LAST_UPDATE_FROM_ANCHOR = {
            "loaded": False,
            "archive_sha256_prefix": archive_sha[:12],
            "requested_helper_missing": True,
            "fallback_helper": _FALLBACK_HELPER_NAME,
        }
        return

    gradient, source_anchor, source_helper, requested_helper_missing = loaded
    _LAST_UPDATE_FROM_ANCHOR = {
        "loaded": True,
        "archive_sha256_prefix": archive_sha[:12],
        "gradient_shape": tuple(int(dim) for dim in gradient.shape),
        "measurement_axis": str(source_anchor.get("measurement_axis") or "[predicted]"),
        "measurement_hardware": str(source_anchor.get("measurement_hardware") or "unknown"),
        "requested_helper_missing": requested_helper_missing,
        "source_helper": source_helper,
    }


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return bit-allocator guidance for a candidate.

    The returned mapping is intentionally observability-only:
    ``predicted_delta_adjustment=0.0``, ``axis_tag="[predicted]"``,
    ``promotable=False``, and ``score_claim=False``. Top-K notes are stable:
    ties sort by lower byte/pair index.
    """
    archive_sha = _extract_archive_sha(candidate)
    if not archive_sha:
        return _no_signal_verdict(
            "candidate missing archive_sha256 / archive_sha / sha256 field; "
            "per-pair bit-allocation routing key unavailable"
        )

    loaded = _load_per_pair_gradient_for_archive(archive_sha)
    if loaded is None:
        return _no_signal_verdict(
            f"no per-pair master-gradient anchor found for archive {archive_sha[:12]}; "
            f"requested helper `{_REQUESTED_HELPER_NAME}` absent, fallback "
            f"`{_FALLBACK_HELPER_NAME}` found no usable anchor"
        )

    gradient, anchor, source_helper, requested_helper_missing = loaded
    try:
        notes = _build_bit_allocator_notes(
            gradient,
            archive_sha256=archive_sha,
            anchor=anchor,
            source_helper=source_helper,
            requested_helper_missing=requested_helper_missing,
            top_k=DEFAULT_TOP_K_BYTES,
        )
    except ValueError as exc:
        return _no_signal_verdict(
            f"invalid per-pair master-gradient payload for archive {archive_sha[:12]}: {exc}"
        )
    top_k_count = len(notes["top_k_byte_indices_by_abs_sensitivity"])
    n_pairs = notes["n_pairs"]
    n_bytes = notes["n_bytes"]

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            f"per-pair master-gradient bit-allocation guidance available for "
            f"{archive_sha[:12]}: top-{top_k_count} byte indices ranked by "
            f"absolute sensitivity across {n_pairs} pairs and {n_bytes} bytes; "
            "observability-only hook #3 [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "score_claim": False,
        "promotion_eligible": False,
        "confidence": 0.0,
        "authority": "predicted_observability_only",
        "notes": {"bit_allocator_per_pair_consumer": notes},
    }


def _load_per_pair_gradient_for_archive(
    archive_sha256: str,
) -> tuple[Any, Mapping[str, Any], str, bool] | None:
    """Load a per-pair master-gradient tensor via the live helper surface."""
    try:
        from tac import master_gradient_consumers as consumers
    except ImportError:
        return None

    requested = getattr(consumers, _REQUESTED_HELPER_NAME, None)
    if callable(requested):
        try:
            loaded = requested(archive_sha256=archive_sha256)
        except (FileNotFoundError, ValueError, OSError):
            return None
        if isinstance(loaded, tuple) and len(loaded) == 2:
            gradient, anchor = loaded
            return gradient, anchor, _REQUESTED_HELPER_NAME, False
        return None

    fallback = getattr(consumers, _FALLBACK_HELPER_NAME, None)
    if not callable(fallback):
        return None
    try:
        loaded = fallback(archive_sha256=archive_sha256)
    except (FileNotFoundError, ValueError, OSError):
        return None
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        return None
    gradient, anchor = loaded
    return gradient, anchor, _FALLBACK_HELPER_NAME, True


def _build_bit_allocator_notes(
    gradient: Any,
    *,
    archive_sha256: str,
    anchor: Mapping[str, Any],
    source_helper: str,
    requested_helper_missing: bool,
    top_k: int,
) -> dict[str, Any]:
    """Build deterministic top-K byte and pair rankings from a per-pair tensor."""
    import numpy as np

    arr = np.asarray(gradient)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"per-pair master gradient must have shape (N_bytes, N_pairs, 3); got {arr.shape}"
        )

    n_bytes = int(arr.shape[0])
    n_pairs = int(arr.shape[1])
    byte_scores = np.abs(arr).sum(axis=(1, 2))
    pair_scores = np.abs(arr).sum(axis=(0, 2))
    per_pair_byte_scores = np.abs(arr).sum(axis=2)
    k_bytes = max(0, min(int(top_k), n_bytes))

    top_bytes = _rank_indices(byte_scores, k_bytes)
    top_pairs = _rank_indices(pair_scores, min(k_bytes, n_pairs))
    per_pair_top_bytes = [
        {
            "pair_index": int(pair_idx),
            "top_byte_indices": [
                {
                    "byte_index": int(byte_idx),
                    "absolute_sensitivity": float(per_pair_byte_scores[byte_idx, pair_idx]),
                }
                for byte_idx in _rank_indices(per_pair_byte_scores[:, pair_idx], k_bytes)
            ],
        }
        for pair_idx in range(n_pairs)
    ]

    return {
        "archive_sha256_prefix": archive_sha256[:12],
        "source_helper": source_helper,
        "requested_helper_missing": requested_helper_missing,
        "requested_helper": _REQUESTED_HELPER_NAME,
        "n_bytes": n_bytes,
        "n_pairs": n_pairs,
        "top_k": k_bytes,
        "measurement_axis": str(anchor.get("measurement_axis") or "[predicted]"),
        "measurement_hardware": str(anchor.get("measurement_hardware") or "unknown"),
        "measurement_method": str(anchor.get("measurement_method") or "unknown"),
        "measurement_utc": str(anchor.get("measurement_utc") or ""),
        "top_k_byte_indices_by_abs_sensitivity": [
            {
                "byte_index": int(idx),
                "absolute_sensitivity": float(byte_scores[idx]),
            }
            for idx in top_bytes
        ],
        "top_k_pair_indices_by_abs_sensitivity": [
            {
                "pair_index": int(idx),
                "absolute_sensitivity": float(pair_scores[idx]),
            }
            for idx in top_pairs
        ],
        "per_pair_top_k_byte_indices": per_pair_top_bytes,
    }


def _rank_indices(scores: Any, k: int) -> list[int]:
    """Rank indices by descending score, tie-breaking by ascending index."""
    if k <= 0:
        return []
    return [
        int(idx)
        for idx in sorted(
            range(len(scores)),
            key=lambda i: (-float(scores[i]), int(i)),
        )[:k]
    ]


def _extract_archive_sha(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    for field_name in _CANDIDATE_SHA_FIELDS:
        value = payload.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _no_signal_verdict(reason: str) -> dict[str, Any]:
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": f"bit_allocator_per_pair_consumer: {reason} [predicted]",
        "axis_tag": "[predicted]",
        "promotable": False,
        "score_claim": False,
        "promotion_eligible": False,
        "confidence": 0.0,
        "authority": "predicted_observability_only",
        "notes": {
            "bit_allocator_per_pair_consumer": {
                "loaded": False,
                "reason": reason,
                "requested_helper_missing": True,
                "requested_helper": _REQUESTED_HELPER_NAME,
                "fallback_helper": _FALLBACK_HELPER_NAME,
            }
        },
    }


__all__ = (
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "consume_candidate",
    "update_from_anchor",
)
