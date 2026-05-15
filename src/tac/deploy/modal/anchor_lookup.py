# SPDX-License-Identifier: MIT
"""Canonical promotable-anchor lookup for paired Modal CPU+CUDA dispatches.

Empirical anchor (2026-05-15)
-----------------------------
Z3 v2 FULL paired dispatch (sister ``a06e09e0``) re-fired CUDA on Modal T4
even though the trainer's inline ``gate_auth_eval_call`` had already produced
a contest-CUDA anchor (``score=0.19869``, ``archive_sha=b6c4a6f1...``) on the
same archive bytes. The re-run cost ~$0.40 and added marginal value
(advisory to promotion-grade tag + determinism sanity check) but is
structurally inefficient.

Per the standing directive
``feedback_consolidate_everything_into_meta_layer_or_canonical_helpers_standing_directive_20260515.md``:
canonical helpers should auto-detect existing anchors and skip redundant
re-runs unless explicitly forced.

What this module does
---------------------
``find_promotable_anchor_for_axis_and_sha(axis, archive_sha256, repo_root)``
returns the dict for an existing promotable anchor on the requested axis +
archive sha, or ``None`` if no eligible anchor exists. Lookup priority:

1. **PRIMARY**: ``tac.deploy.modal.call_id_ledger.load_call_ids`` filtered
   by ``status="harvested"`` + ``archive_sha256`` + ``score_axis``. The
   ledger is the canonical primary index per
   ``feedback_modal_call_id_ledger_canonical_landed_20260515.md``.
2. **FALLBACK**: filesystem scan of
   ``experiments/results/lane_*_modal/harvested_artifacts/contest_auth_eval*.json``
   + ``experiments/results/modal_auth_eval/*/modal_*_auth_eval_result.json``
   + the older ``experiments/results/lane_*_modal/contest_auth_eval*.json``.
   Used when the ledger is missing or absent.

Custody validation per CLAUDE.md "Apples-to-apples evidence discipline" +
Catalog #127 (custody validator) + Catalog #221 (auth_eval result
fail-closed):

- ``evidence_grade`` in ``AUTHORITATIVE_TAGS`` matches the requested axis
  (case-insensitive normalize). ``[contest-CUDA]`` only counts for cuda;
  ``[contest-CPU]``-family only for cpu. Advisory grades
  (``macOS-CPU-advisory``, ``macos_cpu_advisory_only``,
  ``training-only``, etc.) are REFUSED.
- ``score_claim_valid=True`` OR ``score_claim=True`` is REQUIRED so we
  never reuse an advisory / training-only / diagnostic JSON as the
  promotable anchor.
- ``archive_sha256`` must be present and exact-match the requested sha
  (case-insensitive). Partial-prefix matches are REFUSED to fail-safe
  toward firing a fresh dispatch.
- ``score`` must be a finite numeric (rejects ``None`` / ``NaN``).

Fail-safe semantics
-------------------
Per the apples-to-apples discipline: if ANY custody invariant is unclear,
return ``None`` and let the caller fire a fresh dispatch. The cost of a
false-negative skip-decision is one extra ~$0.40 dispatch; the cost of a
false-positive skip-decision is a corrupt promotion claim. The latter is
catastrophic per Catalog #127 / #221; the former is recoverable.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Axes accepted as input. We normalize to these canonical strings.
_AXIS_CUDA = "contest_cuda"
_AXIS_CPU = "contest_cpu"

# Mapping of canonical axis -> set of accepted evidence_grade tokens
# (case-insensitive). Per CLAUDE.md AUTHORITATIVE_TAGS frozenset in
# tac.continual_learning.
_PROMOTABLE_GRADES_PER_AXIS: dict[str, frozenset[str]] = {
    _AXIS_CUDA: frozenset(
        {
            "contest-cuda",
            "[contest-cuda]",
        }
    ),
    _AXIS_CPU: frozenset(
        {
            "contest-cpu",
            "[contest-cpu]",
            "[contest-cpu gha linux x86_64]",
            "[contest-cpu gha]",
            "contest-cpu-gha-linux-x86-64",
        }
    ),
}

# Substrings that DEFINITIVELY mark a payload as advisory / non-promotable
# regardless of any other field. Per CLAUDE.md FORBIDDEN_PATTERNS +
# Catalog #192.
_ADVISORY_GRADE_REJECT_SUBSTRINGS: frozenset[str] = frozenset(
    {
        "advisory",
        "macos",
        "macos-cpu",
        "training-only",
        "training_only",
        "diagnostic",
        "mps",
        "byte-anchor",
        "byte_anchor",
        "predicted",
        "proxy",
    }
)


def _normalize_axis(axis: str) -> str | None:
    """Return ``contest_cuda`` / ``contest_cpu`` or ``None`` if unrecognized."""
    if not isinstance(axis, str):
        return None
    s = axis.strip().lower().replace("-", "_")
    if s in {"cuda", "contest_cuda", "contest_cuda_axis"}:
        return _AXIS_CUDA
    if s in {"cpu", "contest_cpu", "contest_cpu_axis"}:
        return _AXIS_CPU
    return None


def _normalize_grade(grade: Any) -> str:
    """Lowercase + strip whitespace; return empty string for non-strings."""
    if not isinstance(grade, str):
        return ""
    return grade.strip().lower()


def _grade_is_advisory(grade_norm: str) -> bool:
    """True iff the normalized grade matches any advisory-reject substring."""
    if not grade_norm:
        return False
    return any(reject in grade_norm for reject in _ADVISORY_GRADE_REJECT_SUBSTRINGS)


def _grade_matches_axis(grade: Any, canonical_axis: str) -> bool:
    """Custody check 1: evidence_grade must match the requested axis.

    Refuses advisory / training-only / mps / byte-anchor / proxy tokens.
    """
    grade_norm = _normalize_grade(grade)
    if not grade_norm:
        return False
    if _grade_is_advisory(grade_norm):
        return False
    accepted = _PROMOTABLE_GRADES_PER_AXIS.get(canonical_axis, frozenset())
    return grade_norm in accepted


def _score_is_finite_numeric(value: Any) -> bool:
    """Custody check 4: score must be a finite numeric (not None / NaN / inf)."""
    if isinstance(value, bool):
        # bool is an int subclass; explicitly reject.
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _archive_sha_matches(payload_sha: Any, requested_sha: str) -> bool:
    """Custody check 3: exact-match on archive_sha256 (case-insensitive).

    Partial prefixes are REJECTED to fail-safe toward a fresh dispatch.
    """
    if not isinstance(payload_sha, str) or not payload_sha.strip():
        return False
    if not isinstance(requested_sha, str) or not requested_sha.strip():
        return False
    return payload_sha.strip().lower() == requested_sha.strip().lower()


def _score_claim_present(payload: dict[str, Any]) -> bool:
    """Custody check 2: score_claim_valid OR score_claim must be True.

    Per Catalog #221: tolerate either field name; require explicit ``True``.
    """
    if payload.get("score_claim_valid") is True:
        return True
    return payload.get("score_claim") is True


def _runtime_tree_sha256(payload: dict[str, Any]) -> str:
    """Return the best available runtime-tree SHA from auth-eval custody JSON."""
    if not isinstance(payload, dict):
        return ""
    provenance = _provenance(payload)
    for container in (payload, provenance):
        for key in (
            "runtime_tree_sha256",
            "inflate_runtime_tree_sha256",
            "expected_runtime_tree_sha256",
        ):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        for key in (
            "inflate_runtime_manifest",
            "runtime_manifest",
            "submission_runtime",
        ):
            nested = container.get(key)
            if not isinstance(nested, dict):
                continue
            for nested_key in (
                "runtime_tree_sha256",
                "portable_runtime_tree_sha256",
                "tree_sha256",
            ):
                value = nested.get(nested_key)
                if isinstance(value, str) and value.strip():
                    return value.strip().lower()
    return ""


def _runtime_tree_sha_matches(payload: dict[str, Any], expected_runtime_tree_sha256: str) -> bool:
    """True iff payload runtime SHA exactly matches the expected runtime SHA."""
    expected = expected_runtime_tree_sha256.strip().lower()
    if not expected:
        return True
    actual = _runtime_tree_sha256(payload)
    return bool(actual) and actual == expected


def _provenance(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("provenance")
    return value if isinstance(value, dict) else {}


def _first_present(*pairs: tuple[dict[str, Any], str]) -> Any:
    """Return the first key that exists, preserving falsey values like 0."""
    for payload, key in pairs:
        if key in payload:
            return payload[key]
    return None


def _payload_matches_axis_contract(payload: dict[str, Any], canonical_axis: str) -> bool:
    """Require explicit axis/sample/device custody before reusing an anchor."""
    provenance = _provenance(payload)
    payload_axis = _normalize_axis(
        str(_first_present((payload, "score_axis"), (provenance, "score_axis")) or "")
    )
    if payload_axis != canonical_axis:
        return False
    samples = _first_present((payload, "n_samples"), (payload, "samples"), (provenance, "n_samples"))
    if samples != 600:
        return False
    device = str(
        _first_present(
            (payload, "scorer_device"),
            (payload, "device"),
            (provenance, "scorer_device"),
            (provenance, "device"),
        )
        or ""
    ).strip().lower()
    if canonical_axis == _AXIS_CUDA:
        return device == "cuda"
    if canonical_axis == _AXIS_CPU:
        platform_system = str(
            _first_present((payload, "platform_system"), (provenance, "platform_system")) or ""
        ).strip().lower()
        platform_machine = str(
            _first_present((payload, "platform_machine"), (provenance, "platform_machine")) or ""
        ).strip().lower()
        hardware = str(
            _first_present((payload, "hardware"), (provenance, "hardware")) or ""
        ).strip().lower()
        linux_x86_64 = (
            platform_system == "linux"
            and platform_machine in {"x86_64", "amd64"}
        ) or (
            "linux x86_64" in hardware
            or "github-actions-ubuntu-latest_x86_64" in hardware
        )
        return device == "cpu" and linux_x86_64
    return False


def _validate_payload_for_axis_and_sha(
    payload: dict[str, Any],
    *,
    canonical_axis: str,
    archive_sha256: str,
    expected_runtime_tree_sha256: str = "",
) -> bool:
    """Apply archive, axis, score, and optional runtime custody checks."""
    if not isinstance(payload, dict):
        return False
    grade = payload.get("evidence_grade")
    if not _grade_matches_axis(grade, canonical_axis):
        return False
    if not _payload_matches_axis_contract(payload, canonical_axis):
        return False
    if not _score_claim_present(payload):
        return False
    payload_sha = (
        payload.get("archive_sha256")
        or payload.get("expected_archive_sha256")
        or payload.get("submission_archive_sha256")
    )
    if not _archive_sha_matches(payload_sha, archive_sha256):
        return False
    if not _runtime_tree_sha_matches(payload, expected_runtime_tree_sha256):
        return False
    score = (
        payload.get("score")
        if "score" in payload
        else payload.get("final_score")
        if "final_score" in payload
        else payload.get("canonical_score")
    )
    return _score_is_finite_numeric(score)


def _build_anchor_dict(
    payload: dict[str, Any],
    *,
    canonical_axis: str,
    archive_sha256: str,
    result_path: Path | str,
    source: str,
) -> dict[str, Any]:
    """Construct the public-API return dict for a verified anchor."""
    score = (
        payload.get("score")
        if "score" in payload
        else payload.get("final_score")
        if "final_score" in payload
        else payload.get("canonical_score")
    )
    return {
        "axis": canonical_axis,
        "archive_sha256": str(archive_sha256).strip().lower(),
        "archive_bytes": payload.get("archive_size_bytes")
        or payload.get("archive_bytes")
        or payload.get("expected_archive_size_bytes"),
        "score": float(score) if score is not None else None,
        "evidence_grade": str(payload.get("evidence_grade") or ""),
        "hardware_substrate": str(payload.get("hardware_substrate") or payload.get("scorer_device") or ""),
        "runtime_tree_sha256": _runtime_tree_sha256(payload) or None,
        "result_path": str(result_path),
        "dispatched_at_utc": str(
            payload.get("dispatched_at_utc") or (payload.get("provenance") or {}).get("started_at_utc") or ""
        ),
        "source": source,
        "custody_match": True,
    }


def _lookup_via_ledger(
    *,
    canonical_axis: str,
    archive_sha256: str,
    expected_runtime_tree_sha256: str,
    repo_root: Path,
) -> dict[str, Any] | None:
    """PRIMARY lookup: query the canonical Modal call_id ledger."""
    try:
        from tac.deploy.modal.call_id_ledger import load_call_ids
    except ImportError:
        return None

    # The canonical constant is absolute for the current checkout. For lookup
    # against a requested repo_root, always resolve the repo-relative ledger
    # location explicitly; do not depend on the caller's current working dir.
    ledger_path = repo_root / ".omx" / "state" / "modal_call_id_ledger.jsonl"

    try:
        rows = load_call_ids(ledger_path)
    except Exception:
        return None

    requested_axis_short = "cuda" if canonical_axis == _AXIS_CUDA else "cpu"

    # Walk rows in reverse so we prefer the most recent harvested anchor
    # for the same (axis, sha) pair.
    for row in reversed(rows):
        if not isinstance(row, dict):
            continue
        if row.get("status") != "harvested":
            continue
        harvest_result = row.get("harvest_result")
        if not isinstance(harvest_result, dict):
            harvest_result = {}
        # Ledger rows may carry auth-eval fields either at top level or inside
        # harvest_result, depending on whether they were registered live or
        # backfilled from older Modal artifacts.
        row_axis = _first_present(
            (row, "score_axis"),
            (row, "expected_axis"),
            (harvest_result, "score_axis"),
            (harvest_result, "scorer_device"),
        )
        if not isinstance(row_axis, str):
            continue
        row_axis_norm = row_axis.strip().lower().replace("-", "_")
        if row_axis_norm not in {requested_axis_short, canonical_axis}:
            continue
        row_archive_sha = _first_present(
            (row, "archive_sha256"),
            (harvest_result, "archive_sha256"),
            (harvest_result, "expected_archive_sha256"),
            (harvest_result, "submission_archive_sha256"),
        )
        if not _archive_sha_matches(row_archive_sha, archive_sha256):
            continue
        # Synthesize a payload-shape dict so the same custody validator runs.
        synthesized = {
            "evidence_grade": _first_present(
                (row, "evidence_grade"),
                (harvest_result, "evidence_grade"),
                (harvest_result, "tag"),
            ),
            "score_claim_valid": _first_present(
                (row, "score_claim_valid"),
                (harvest_result, "score_claim_valid"),
            ),
            "score_claim": _first_present(
                (row, "score_claim"),
                (harvest_result, "score_claim"),
            ),
            "archive_sha256": row_archive_sha,
            "archive_size_bytes": _first_present(
                (row, "archive_bytes"),
                (harvest_result, "archive_size_bytes"),
                (harvest_result, "eval_archive_size_bytes"),
            ),
            "score": _first_present(
                (row, "score"),
                (harvest_result, "score"),
            ),
            "score_axis": _first_present(
                (row, "score_axis"),
                (harvest_result, "score_axis"),
            ),
            "n_samples": _first_present(
                (row, "n_samples"),
                (row, "samples"),
                (harvest_result, "n_samples"),
                (harvest_result, "samples"),
            ),
            "scorer_device": _first_present(
                (row, "scorer_device"),
                (row, "device"),
                (harvest_result, "scorer_device"),
                (harvest_result, "device"),
            )
            or requested_axis_short,
            "platform_system": _first_present(
                (row, "platform_system"),
                (harvest_result, "platform_system"),
            ),
            "platform_machine": _first_present(
                (row, "platform_machine"),
                (harvest_result, "platform_machine"),
            ),
            "runtime_tree_sha256": _first_present(
                (row, "runtime_tree_sha256"),
                (row, "inflate_runtime_tree_sha256"),
                (harvest_result, "runtime_tree_sha256"),
                (harvest_result, "inflate_runtime_tree_sha256"),
            ),
            "inflate_runtime_manifest": _first_present(
                (row, "inflate_runtime_manifest"),
                (harvest_result, "inflate_runtime_manifest"),
                (harvest_result, "runtime_manifest"),
            ),
            "dispatched_at_utc": row.get("dispatched_at_utc"),
        }
        if not _validate_payload_for_axis_and_sha(
            synthesized,
            canonical_axis=canonical_axis,
            archive_sha256=archive_sha256,
            expected_runtime_tree_sha256=expected_runtime_tree_sha256,
        ):
            continue
        return _build_anchor_dict(
            synthesized,
            canonical_axis=canonical_axis,
            archive_sha256=archive_sha256,
            result_path=str(ledger_path) + f"#call_id={row.get('call_id')}",
            source="modal_call_id_ledger",
        )

    return None


def _candidate_anchor_paths(repo_root: Path) -> Iterable[Path]:
    """FALLBACK lookup: yield every JSON file that may contain a contest anchor.

    Per CLAUDE.md "Forbidden /tmp paths": these all live under
    ``experiments/results/`` (DERIVED_OUTPUT per Catalog #113).
    """
    results_root = repo_root / "experiments" / "results"
    if not results_root.is_dir():
        return

    # Most-canonical layout: lane_*_modal/harvested_artifacts/contest_auth_eval*.json
    yield from results_root.glob("lane_*_modal/harvested_artifacts/contest_auth_eval*.json")
    # Alternate layout: lane_*_modal/contest_auth_eval*.json
    yield from results_root.glob("lane_*_modal/contest_auth_eval*.json")
    # Modal CUDA result dispatcher layout
    yield from results_root.glob("modal_auth_eval/*/modal_cuda_auth_eval_result.json")
    yield from results_root.glob("modal_auth_eval_cpu/*/modal_cpu_auth_eval_result.json")
    # Older paired-dispatch layout
    yield from results_root.glob("modal_auth_eval/*/contest_auth_eval*.json")
    yield from results_root.glob("modal_auth_eval_cpu/*/contest_auth_eval*.json")


def _lookup_via_filesystem(
    *,
    canonical_axis: str,
    archive_sha256: str,
    expected_runtime_tree_sha256: str,
    repo_root: Path,
) -> dict[str, Any] | None:
    """FALLBACK lookup: scan canonical anchor JSON paths."""
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in _candidate_anchor_paths(repo_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if not _validate_payload_for_axis_and_sha(
            payload,
            canonical_axis=canonical_axis,
            archive_sha256=archive_sha256,
            expected_runtime_tree_sha256=expected_runtime_tree_sha256,
        ):
            continue
        candidates.append((path, payload))

    if not candidates:
        return None

    # Prefer the most recently modified file as the canonical anchor for
    # this (axis, sha) pair.
    candidates.sort(key=lambda pair: pair[0].stat().st_mtime, reverse=True)
    path, payload = candidates[0]
    return _build_anchor_dict(
        payload,
        canonical_axis=canonical_axis,
        archive_sha256=archive_sha256,
        result_path=str(path),
        source="filesystem_scan",
    )


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


def find_promotable_anchor_for_axis_and_sha(
    axis: str,
    archive_sha256: str,
    repo_root: Path | str = ".",
    *,
    expected_runtime_tree_sha256: str = "",
) -> dict[str, Any] | None:
    """Find an existing promotable anchor for ``(axis, archive_sha256)``.

    Parameters
    ----------
    axis
        ``"cuda"`` / ``"contest_cuda"`` or ``"cpu"`` / ``"contest_cpu"``.
        Other inputs return ``None`` (fail-safe).
    archive_sha256
        Hex sha256 string (case-insensitive). Empty / non-string returns
        ``None`` (fail-safe).
    repo_root
        Repository root; defaults to ``"."``. Tests pass a tmp dir.
    expected_runtime_tree_sha256
        Optional runtime-tree sha256. When provided, an anchor is promotable
        only if its runtime custody exactly matches this SHA. This prevents
        paired dispatch skip-logic from reusing old scores after ``inflate.py``
        or ``inflate.sh`` changed while archive bytes stayed constant.

    Returns
    -------
    dict | None
        On hit::

            {
                "axis": "contest_cuda" | "contest_cpu",
                "archive_sha256": <lowercase hex>,
                "archive_bytes": int | None,
                "score": float,
                "evidence_grade": str,
                "hardware_substrate": str,
                "result_path": str,
                "dispatched_at_utc": str,
                "source": "modal_call_id_ledger" | "filesystem_scan",
                "custody_match": True,
            }

        ``None`` if no anchor satisfies all 4 custody checks.

    Lookup priority
    ---------------
    1. ``tac.deploy.modal.call_id_ledger`` (PRIMARY) - the canonical primary
       index per the 2026-05-15 ledger landing.
    2. Filesystem scan of canonical anchor JSON paths under
       ``experiments/results/`` (FALLBACK) - used when the ledger is empty
       or absent. Both paths apply the same 4-check custody validator.
    """
    canonical_axis = _normalize_axis(axis)
    if canonical_axis is None:
        return None
    if not isinstance(archive_sha256, str) or not archive_sha256.strip():
        return None

    root = Path(repo_root).resolve()
    if not root.is_dir():
        return None

    # PRIMARY: ledger lookup
    hit = _lookup_via_ledger(
        canonical_axis=canonical_axis,
        archive_sha256=archive_sha256,
        expected_runtime_tree_sha256=expected_runtime_tree_sha256,
        repo_root=root,
    )
    if hit is not None:
        return hit

    # FALLBACK: filesystem scan
    return _lookup_via_filesystem(
        canonical_axis=canonical_axis,
        archive_sha256=archive_sha256,
        expected_runtime_tree_sha256=expected_runtime_tree_sha256,
        repo_root=root,
    )


__all__ = [
    "find_promotable_anchor_for_axis_and_sha",
]
