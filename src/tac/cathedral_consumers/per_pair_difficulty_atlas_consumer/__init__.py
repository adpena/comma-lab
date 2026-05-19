# SPDX-License-Identifier: MIT
"""Cathedral consumer for per-pair difficulty-atlas posterior payloads.

Per Catalog #335 this package exposes the canonical
``CathedralConsumerContract`` surface so the cathedral consumer auto-discovery
path can see the per-pair difficulty atlas without a manual dispatcher edit.

The source signal is the per-pair master-gradient tensor:
``gradient_tensor_kind="per_pair_per_byte_v1"`` in
``.omx/state/master_gradient_anchors.jsonl``.  For each pair ``p`` this
consumer computes:

    difficulty_p = ||g_p||_2 * pair_score_p

where ``pair_score_p`` is taken from an explicit per-pair score vector when one
is present on the candidate or anchor.  If no pair-score vector exists, the
consumer uses unit weights and marks that fact in the payload.  The output is a
``[predicted]`` routing/learning payload, never an empirical score claim.

Hook assignments per Catalog #125:

* #1 sensitivity map — per-pair difficulty is a sensitivity/routing signal.
* #4 cathedral autopilot dispatch — auto-discovery can present the signal.
* #5 continual-learning posterior — ACTIVE as a payload builder.  The current
  ``tac.continual_learning.posterior_update_locked`` helper accepts
  ``ContestResult`` auth-eval objects and has no per-pair predicted-anchor
  schema, so this consumer deliberately does not call it with fake empirical
  data or mutate posterior state directly.  The payload records the exact
  mismatch for later canonical-helper extension.
"""
from __future__ import annotations

import importlib
import inspect
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "per_pair_difficulty_atlas_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)

_CANDIDATE_SHA_FIELDS = (
    "archive_sha256",
    "archive_sha",
    "sha256",
    "sha",
    "scored_archive_sha256",
)
_ANCHOR_PATH_FIELDS = (
    "master_gradient_anchor_path",
    "master_gradient_ledger_path",
    "ledger_path",
    "anchor_path",
)
_PAIR_SCORE_FIELDS = (
    "pair_scores",
    "per_pair_scores",
    "pair_score_p",
    "pair_score",
    "pair_score_by_index",
    "per_pair_score_by_index",
    "per_pair_score_marginals",
    "pair_score_marginals",
    "per_pair_score_contributions",
)
_NO_POSTERIOR_MUTATION_STATUS = "payload_only_canonical_helper_mismatch"


def update_from_anchor(anchor: Any) -> Mapping[str, Any]:
    """Catalog #125 hook #5: build predicted CL payloads from a new anchor.

    The current canonical posterior helper is auth-eval ``ContestResult``
    oriented.  This function therefore returns a payload-only verdict instead
    of creating a fake score result or directly appending to any posterior file.
    """
    if not isinstance(anchor, Mapping):
        return _fail_closed_update(
            "anchor is not a mapping; per-pair difficulty atlas update refused"
        )
    archive_sha = _extract_archive_sha(anchor)
    if archive_sha is None:
        return _fail_closed_update(
            "anchor missing archive_sha256 / archive_sha / sha256 key"
        )
    try:
        gradient, loaded_anchor = _load_gradient_from_anchor_row(anchor)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return _fail_closed_update(str(exc), archive_sha=archive_sha)
    try:
        payload = build_predicted_difficulty_payload(
            gradient,
            archive_sha256=archive_sha,
            source_anchor=loaded_anchor,
            candidate=anchor,
        )
    except ValueError as exc:
        return _fail_closed_update(str(exc), archive_sha=archive_sha)
    return {
        "accepted": False,
        "status": _NO_POSTERIOR_MUTATION_STATUS,
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "posterior_wire_status": posterior_wire_status(),
        "predicted_anchor_payload": payload,
    }


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4: present per-pair difficulty as predicted metadata."""
    if not isinstance(candidate, Mapping):
        return _no_signal_verdict(
            "candidate is not a mapping; per-pair difficulty routing key unavailable"
        )
    archive_sha = _extract_archive_sha(candidate)
    if archive_sha is None:
        return _no_signal_verdict(
            "candidate missing archive_sha256 / archive_sha / sha256 field"
        )

    anchor_path = _extract_anchor_path(candidate)
    try:
        from tac.master_gradient_consumers import load_per_pair_gradient_from_anchor
    except ImportError as exc:
        return _no_signal_verdict(
            f"tac.master_gradient_consumers unavailable: {exc}"
        )

    try:
        gradient, anchor = load_per_pair_gradient_from_anchor(
            archive_sha256=archive_sha,
            anchor_path=anchor_path,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return _no_signal_verdict(
            f"no usable per-pair master-gradient anchor for {archive_sha[:12]}: {exc}"
        )

    try:
        payload = build_predicted_difficulty_payload(
            gradient,
            archive_sha256=archive_sha,
            source_anchor=anchor,
            candidate=candidate,
        )
    except ValueError as exc:
        return _no_signal_verdict(
            f"per-pair difficulty payload refused for {archive_sha[:12]}: {exc}"
        )

    hardest = payload["top_pairs"][0] if payload["top_pairs"] else None
    if isinstance(hardest, Mapping):
        hard_text = (
            f"hardest_pair={hardest['pair_index']} "
            f"difficulty={hardest['difficulty_score']:.6g}"
        )
    else:
        hard_text = "no pairs ranked"

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            f"per-pair difficulty atlas predicted payload ready for "
            f"{archive_sha[:12]} ({payload['n_pairs']} pairs; {hard_text}); "
            f"posterior_status={_NO_POSTERIOR_MUTATION_STATUS} [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "notes": {"per_pair_difficulty_atlas": payload},
    }


def build_predicted_difficulty_payload(
    per_pair_gradient: Any,
    *,
    archive_sha256: str,
    source_anchor: Mapping[str, Any],
    candidate: Mapping[str, Any] | None = None,
    top_k: int = 50,
) -> dict[str, Any]:
    """Compute deterministic ``[predicted]`` per-pair difficulty payload."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - covered in minimal envs
        raise RuntimeError("numpy required for per-pair difficulty atlas") from exc

    arr = np.asarray(per_pair_gradient, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"per_pair_gradient must have shape (N_bytes, N_pairs, 3); got {arr.shape}"
        )
    n_bytes, n_pairs, _ = arr.shape
    if n_pairs <= 0:
        raise ValueError("per_pair_gradient must contain at least one pair")

    pair_scores, pair_score_source = _extract_pair_scores(
        candidate or {},
        source_anchor,
        n_pairs=n_pairs,
    )
    norms = np.sqrt((arr * arr).sum(axis=(0, 2)))
    seg_l1 = np.abs(arr[:, :, 0]).sum(axis=0)
    pose_l1 = np.abs(arr[:, :, 1]).sum(axis=0)
    rate_l1 = np.abs(arr[:, :, 2]).sum(axis=0)
    difficulty_scores = norms * np.asarray(pair_scores, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    sorted_pair_indices = sorted(
        range(n_pairs),
        key=lambda idx: (-float(difficulty_scores[idx]), int(idx)),
    )
    for rank, pair_index in enumerate(sorted_pair_indices):
        rows.append(
            {
                "pair_index": int(pair_index),
                "difficulty_rank": int(rank),
                "gradient_norm_l2": float(norms[pair_index]),
                "pair_score": float(pair_scores[pair_index]),
                "difficulty_score": float(difficulty_scores[pair_index]),
                "seg_axis_contribution_l1": float(seg_l1[pair_index]),
                "pose_axis_contribution_l1": float(pose_l1[pair_index]),
                "rate_axis_contribution_l1": float(rate_l1[pair_index]),
            }
        )

    top_n = _coerce_top_k((candidate or {}).get("top_k"), top_k, n_pairs)
    posterior_status = posterior_wire_status()
    return {
        "schema": "cathedral_per_pair_difficulty_atlas_predicted_anchor_v1",
        "consumer_id": CONSUMER_NAME,
        "archive_sha256": archive_sha256,
        "source_gradient_tensor_kind": source_anchor.get("gradient_tensor_kind"),
        "source_measurement_axis": source_anchor.get("measurement_axis"),
        "source_measurement_hardware": source_anchor.get("measurement_hardware"),
        "source_measurement_method": source_anchor.get("measurement_method"),
        "source_measurement_call_id": source_anchor.get("measurement_call_id"),
        "source_measurement_utc": source_anchor.get("measurement_utc"),
        "n_bytes": int(n_bytes),
        "n_pairs": int(n_pairs),
        "pair_score_source": pair_score_source,
        "difficulty_formula": "difficulty_p = l2_norm(g_p) * pair_score_p",
        "axis_tag": "[predicted]",
        "evidence_grade": "predicted",
        "empirical_anchor": False,
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "posterior_mutation_performed": False,
        "posterior_update_status": _NO_POSTERIOR_MUTATION_STATUS,
        "posterior_wire_status": posterior_status,
        "top_pairs": rows[:top_n],
        "all_pairs": rows,
    }


def posterior_wire_status() -> dict[str, Any]:
    """Return the current canonical-helper compatibility verdict.

    This intentionally re-checks the helper dynamically so tests and future
    callers see the live import surface.  It does not call the helper.
    """
    try:
        module = importlib.import_module("tac.continual_learning")
    except ImportError as exc:
        return {
            "helper": "tac.continual_learning.posterior_update_locked",
            "helper_available": False,
            "helper_callable": False,
            "predicted_per_pair_anchor_supported": False,
            "status": "fail_closed_helper_missing",
            "reason": str(exc),
            "direct_posterior_jsonl_mutation": False,
        }
    helper = getattr(module, "posterior_update_locked", None)
    if not callable(helper):
        return {
            "helper": "tac.continual_learning.posterior_update_locked",
            "helper_available": False,
            "helper_callable": False,
            "predicted_per_pair_anchor_supported": False,
            "status": "fail_closed_helper_missing",
            "reason": "posterior_update_locked is absent or not callable",
            "direct_posterior_jsonl_mutation": False,
        }
    signature = str(inspect.signature(helper))
    return {
        "helper": "tac.continual_learning.posterior_update_locked",
        "helper_available": True,
        "helper_callable": True,
        "helper_signature": signature,
        "predicted_per_pair_anchor_supported": False,
        "status": _NO_POSTERIOR_MUTATION_STATUS,
        "reason": (
            "posterior_update_locked accepts ContestResult auth-eval anchors; "
            "this consumer builds per-pair [predicted] difficulty anchors and "
            "therefore must not coerce them into empirical score results"
        ),
        "direct_posterior_jsonl_mutation": False,
    }


def _extract_archive_sha(candidate: Mapping[str, Any]) -> str | None:
    for field_name in _CANDIDATE_SHA_FIELDS:
        value = candidate.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _extract_anchor_path(candidate: Mapping[str, Any]) -> Path | None:
    for field_name in _ANCHOR_PATH_FIELDS:
        value = candidate.get(field_name)
        if isinstance(value, (str, Path)) and str(value).strip():
            return Path(value)
    return None


def _load_gradient_from_anchor_row(
    anchor: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any]]:
    if anchor.get("gradient_tensor_kind") != "per_pair_per_byte_v1":
        raise ValueError(
            "anchor is not gradient_tensor_kind='per_pair_per_byte_v1'"
        )
    gradient_path_raw = anchor.get("gradient_array_path")
    if not isinstance(gradient_path_raw, str) or not gradient_path_raw.strip():
        raise ValueError("anchor missing gradient_array_path")
    gradient_path = Path(gradient_path_raw)
    if not gradient_path.is_absolute():
        gradient_path = Path.cwd() / gradient_path
    if not gradient_path.is_file():
        raise FileNotFoundError(f"per-pair gradient array not found: {gradient_path}")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - covered in minimal envs
        raise RuntimeError("numpy required for per-pair difficulty atlas") from exc
    return np.load(gradient_path), anchor


def _extract_pair_scores(
    candidate: Mapping[str, Any],
    anchor: Mapping[str, Any],
    *,
    n_pairs: int,
) -> tuple[tuple[float, ...], str]:
    for source_name, source in (("candidate", candidate), ("anchor", anchor)):
        for field_name in _PAIR_SCORE_FIELDS:
            if field_name not in source:
                continue
            parsed = _parse_pair_scores(source[field_name], n_pairs=n_pairs)
            return parsed, f"{source_name}.{field_name}"
    return tuple(1.0 for _ in range(n_pairs)), "unit_default_no_pair_score_field"


def _parse_pair_scores(value: Any, *, n_pairs: int) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        scores: list[float] = []
        for idx in range(n_pairs):
            raw = value.get(idx, value.get(str(idx)))
            if raw is None:
                raise ValueError(f"pair score mapping missing pair index {idx}")
            scores.append(_finite_nonnegative_float(raw, field=f"pair_scores[{idx}]"))
        return tuple(scores)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != n_pairs:
            raise ValueError(
                f"pair score vector length {len(value)} != n_pairs {n_pairs}"
            )
        return tuple(
            _finite_nonnegative_float(raw, field=f"pair_scores[{idx}]")
            for idx, raw in enumerate(value)
        )
    raise ValueError("pair score payload must be a sequence or mapping")


def _finite_nonnegative_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, got bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    if parsed < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return parsed


def _coerce_top_k(value: Any, default: int, n_pairs: int) -> int:
    if isinstance(value, bool) or value is None:
        parsed = default
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
    return max(0, min(int(parsed), n_pairs))


def _no_signal_verdict(reason: str) -> dict[str, Any]:
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": f"{CONSUMER_NAME}: {reason} [predicted]",
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _fail_closed_update(
    reason: str, *, archive_sha: str | None = None
) -> dict[str, Any]:
    return {
        "accepted": False,
        "status": "fail_closed",
        "reason": reason,
        "archive_sha256": archive_sha,
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "posterior_mutation_performed": False,
        "posterior_wire_status": posterior_wire_status(),
    }


__all__ = (
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "build_predicted_difficulty_payload",
    "consume_candidate",
    "posterior_wire_status",
    "update_from_anchor",
)
