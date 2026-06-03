# SPDX-License-Identifier: MIT
"""Receiver-closed modelsize/fc_dim ladder producer for NeRV-family carriers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from tac.analysis.nerv_receiver_proof_identity import (
    bind_nerv_receiver_proof_identity,
)
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    build_modelsize_budget_plan,
)

SCHEMA = "nerv_receiver_closed_modelsize_ladder.v1"
AXIS_TAG = "[planning/control]"
FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "production_hardened_claim": False,
    "ready_for_exact_eval_dispatch": False,
}

_ARCHIVE_BYTE_KEYS = (
    "measured_archive_bytes",
    "archive_bytes",
    "archive_zip_bytes",
    "candidate_archive_bytes",
    "archive_path_stat_bytes",
)
_PROJECTED_ARCHIVE_BYTE_KEYS = ("projected_archive_bytes_600pair",)
_AUTHORITY_TRUE_KEYS = (
    "score_claim",
    "score_claim_valid",
    "frontier_score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "production_hardened_claim",
    "ready_for_exact_eval_dispatch",
)
_NONRATE_ADVISORY_KEYS = {"nonrate_score_advisory"}
_ADVISORY_AXIS_TOKENS = (
    "advisory",
    "projected",
    "predicted",
    "proxy",
    "research-signal",
    "macos",
    "mps",
    "planning",
)
_CONTEST_AUTH_AXIS_PREFIXES = ("[contest-cpu", "[contest-cuda")


class NervReceiverClosedModelsizeLadderError(ValueError):
    """Raised when the receiver-closed modelsize ladder input is malformed."""


@dataclass(frozen=True)
class _NormalizedRow:
    row: dict[str, Any]
    budget_row: dict[str, Any] | None


def build_nerv_receiver_closed_modelsize_ladder(
    rows: Sequence[Mapping[str, Any]],
    *,
    carrier_id: str,
    baseline_id: str = "pr95_hnerv",
    source_artifact_path: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize candidate rows and emit a fail-closed receiver-byte ladder."""

    carrier = str(carrier_id).strip()
    if not carrier:
        raise NervReceiverClosedModelsizeLadderError("carrier_id must be non-empty")

    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    carrier_aliases = _carrier_aliases(carrier)
    normalized = [
        _normalize_row(index, row, repo_root=root, carrier_aliases=carrier_aliases)
        for index, row in enumerate(_mapping_rows(rows))
    ]
    budget_rows = [row.budget_row for row in normalized if row.budget_row is not None]
    plan = build_modelsize_budget_plan(
        budget_rows,
        carrier_id=carrier,
        baseline_id=baseline_id,
    )
    receiver_ready = plan.get("status") == "receiver_closed_modelsize_budget_selected"
    blocked_rows = [row.row for row in normalized if row.row["blockers"]]
    receiver_closed_rows = [
        row.row for row in normalized if row.row["receiver_closed_modelsize_row"]
    ]

    blocker_records = _build_blocker_records(
        carrier=carrier,
        carrier_aliases=carrier_aliases,
        receiver_ready=receiver_ready,
        receiver_closed_rows=receiver_closed_rows,
        budget_rows=budget_rows,
        blocked_rows=blocked_rows,
        modelsize_budget_blockers=plan.get("blockers", []),
    )
    blockers = _ordered_unique(_blocker_label(record) for record in blocker_records)

    return {
        "schema": SCHEMA,
        "axis_tag": AXIS_TAG,
        "carrier_id": carrier,
        "baseline_id": str(baseline_id),
        "source_artifact_path": source_artifact_path,
        "status": (
            "receiver_closed_modelsize_ladder_ready"
            if receiver_ready
            else "receiver_closed_modelsize_ladder_blocked"
        ),
        "verdict": (
            "GO_LOCAL_CARRIER_TRAINING_GATE_INPUT__NO_GO_SCORE_OR_EXACT_AUTH"
            if receiver_ready
            else "NO_GO_CARRIER_TRAINING_GATE_INPUT__RECEIVER_CLOSED_LADDER_MISSING"
        ),
        "row_count": len(normalized),
        "budget_row_count": len(budget_rows),
        "receiver_closed_row_count": len(receiver_closed_rows),
        "blocked_row_count": len(blocked_rows),
        "normalized_rows": [row.row for row in normalized],
        "modelsize_budget_rows": budget_rows,
        "modelsize_budget_plan": plan,
        "receiver_closed_selected_archive_bytes": plan.get(
            "receiver_closed_selected_archive_bytes"
        ),
        "receiver_closed_selected_modelsize_mparams": _selected_source_field(
            plan, "modelsize_mparams"
        ),
        "receiver_closed_selected_fc_dim": _selected_source_field(plan, "fc_dim"),
        "ready_for_carrier_training_plan": receiver_ready,
        "ready_for_score_aware_training": False,
        "ready_for_exact_eval_dispatch": False,
        "recommended_next_actions": _recommended_next_actions(receiver_ready),
        "blockers": blockers,
        "blocker_records": blocker_records,
        **FALSE_AUTHORITY,
    }


def build_nerv_receiver_closed_modelsize_ladder_from_iterable(
    rows: Iterable[Mapping[str, Any]],
    *,
    carrier_id: str,
    baseline_id: str = "pr95_hnerv",
    source_artifact_path: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a ladder from any row iterable."""

    return build_nerv_receiver_closed_modelsize_ladder(
        list(rows),
        carrier_id=carrier_id,
        baseline_id=baseline_id,
        source_artifact_path=source_artifact_path,
        repo_root=repo_root,
    )


def _normalize_row(
    index: int,
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    carrier_aliases: set[str],
) -> _NormalizedRow:
    source = dict(row)
    row_id = str(source.get("row_id") or source.get("id") or f"modelsize_{index:04d}")
    source_family = (
        source.get("family")
        or source.get("carrier_id")
        or _lookup(source, ("solved_budget", "family"))
    )
    family_key = _carrier_key(source_family)
    family_mismatch = family_key is not None and family_key not in carrier_aliases
    modelsize = _first_float(
        source,
        (
            "modelsize_mparams",
            "modelsize",
            ("official_controls", "--modelsize"),
            ("solved_budget", "modelsize_mparams"),
            ("solved_budget", "official_controls", "--modelsize"),
        ),
    )
    fc_dim = _first_int(
        source,
        (
            "fc_dim",
            ("derived", "fc_dim"),
            ("solved_budget", "derived", "fc_dim"),
        ),
    )
    archive_bytes, archive_key = _archive_bytes(source, repo_root=repo_root)
    nonrate_score, nonrate_score_key = _first_float_with_key(
        source,
        ("nonrate_score", "nonrate_score_value", "nonrate_score_advisory"),
    )
    if nonrate_score is None:
        nonrate_score = _nonrate_from_components(source)
        if nonrate_score is not None:
            nonrate_score_key = "component_distortions"
    archive_sha = _archive_sha(source, repo_root=repo_root)
    receiver_proof_identity = bind_nerv_receiver_proof_identity(
        (source,),
        repo_root=repo_root,
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha,
    )
    receiver_proof = bool(
        receiver_proof_identity["bound"]
        and receiver_proof_identity["proof_passed"]
    )
    authority_blockers = _authority_claim_blockers(source)
    advisory_nonrate = nonrate_score_key in _NONRATE_ADVISORY_KEYS
    source_axis_tag = _string_or_none(
        source.get("axis_tag") or source.get("score_axis") or source.get("evidence_axis")
    )
    source_axis_authorized = _axis_is_receiver_closed_authority(source_axis_tag)

    blockers: list[str] = []
    if modelsize is None and fc_dim is None:
        blockers.append("modelsize_or_fc_dim_missing")
    if family_mismatch:
        blockers.append("carrier_family_mismatch")
    if archive_bytes is None:
        blockers.append("measured_archive_bytes_missing")
    if archive_key in _PROJECTED_ARCHIVE_BYTE_KEYS:
        blockers.append("projected_archive_bytes_not_receiver_closed")
    if nonrate_score is None:
        blockers.append("nonrate_score_or_component_distortions_missing")
    if advisory_nonrate:
        blockers.append("advisory_nonrate_score_not_receiver_closed")
    if not source_axis_authorized:
        blockers.append("source_axis_not_receiver_closed_contest_authority")
    if not receiver_proof:
        blockers.append("receiver_closed_byte_proof_missing")
        if not receiver_proof_identity["bound"]:
            blockers.append("receiver_proof_identity_missing")
            blockers.extend(receiver_proof_identity["blockers"])
    blockers.extend(authority_blockers)

    complete_budget_shape = (
        (modelsize is not None or fc_dim is not None)
        and not family_mismatch
        and archive_bytes is not None
        and nonrate_score is not None
        and not advisory_nonrate
        and source_axis_authorized
        and not authority_blockers
    )
    receiver_closed_modelsize_row = complete_budget_shape and receiver_proof and (
        archive_key in _ARCHIVE_BYTE_KEYS
    )
    eligible_for_modelsize_budget_plan = complete_budget_shape
    normalized = {
        "row_id": row_id,
        "source_family": source_family,
        "source_family_key": family_key,
        "modelsize_mparams": modelsize,
        "fc_dim": fc_dim,
        "archive_bytes": archive_bytes,
        "archive_bytes_key": archive_key,
        "archive_byte_evidence_kind": (
            "projected_or_lower_bound"
            if archive_key in _PROJECTED_ARCHIVE_BYTE_KEYS
            else "measured_receiver_archive"
            if archive_key in _ARCHIVE_BYTE_KEYS
            else "missing"
        ),
        "archive_sha256": archive_sha,
        "archive_path": _string_or_none(
            source.get("archive_path")
            or source.get("archive_zip_path")
            or source.get("candidate_archive_path")
        ),
        "receiver_proof_identity_bound": bool(receiver_proof_identity["bound"]),
        "receiver_proof_identity": receiver_proof_identity,
        "receiver_proof_path": receiver_proof_identity["proof_path"],
        "receiver_proof_sha256": receiver_proof_identity["proof_sha256"],
        "byte_closed_receiver_proof": receiver_proof,
        "nonrate_score": nonrate_score,
        "source_axis_tag": source_axis_tag,
        "source_axis_receiver_closed_authority": source_axis_authorized,
        "nonrate_score_key": nonrate_score_key,
        "nonrate_score_evidence_kind": (
            "advisory"
            if advisory_nonrate
            else "measured_or_component_distortion"
            if nonrate_score is not None
            else "missing"
        ),
        "receiver_proof_passed": receiver_proof,
        "eligible_for_modelsize_budget_plan": eligible_for_modelsize_budget_plan,
        "receiver_closed_modelsize_row": receiver_closed_modelsize_row,
        "blockers": blockers,
        "source": source,
    }
    if not eligible_for_modelsize_budget_plan:
        return _NormalizedRow(row=normalized, budget_row=None)

    budget_row: dict[str, Any] = {
        "row_id": row_id,
        "nonrate_score": float(nonrate_score),
        "modelsize_mparams": modelsize,
        "fc_dim": fc_dim,
        "archive_sha256": archive_sha,
        "axis_tag": source_axis_tag,
        "num_pairs": _first_int(
            source,
            (
                "num_pairs",
                "n_pairs",
                "pair_count",
                "sample_pair_count",
                "sample_pairs",
                "sample_count",
                "n_samples",
            ),
        ),
        "full_video_coverage": bool(
            source.get("full_video_coverage")
            or source.get("full600_coverage")
            or source.get("full_sample_coverage")
        ),
        "receiver_proof_passed": receiver_closed_modelsize_row,
        "receiver_closed": receiver_closed_modelsize_row,
        "byte_closed_receiver_proof": receiver_closed_modelsize_row,
        "receiver_proof_identity_bound": bool(receiver_proof_identity["bound"]),
        "receiver_proof_path": receiver_proof_identity["proof_path"],
        "receiver_proof_sha256": receiver_proof_identity["proof_sha256"],
        "source_row_id": row_id,
    }
    if archive_key in _PROJECTED_ARCHIVE_BYTE_KEYS:
        budget_row["projected_archive_bytes_600pair"] = int(archive_bytes)
        budget_row["lower_bound_only"] = True
    else:
        budget_row["archive_bytes"] = int(archive_bytes)
    return _NormalizedRow(row=normalized, budget_row=budget_row)


def _archive_bytes(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[int | None, str | None]:
    for key in (*_ARCHIVE_BYTE_KEYS, *_PROJECTED_ARCHIVE_BYTE_KEYS):
        value = _finite_int(row.get(key))
        if value is not None and value > 0:
            return value, key
    archive_path = _archive_path(row, repo_root=repo_root)
    if archive_path is not None and archive_path.exists() and archive_path.is_file():
        return archive_path.stat().st_size, "archive_path_stat_bytes"
    return None, None


def _archive_sha(row: Mapping[str, Any], *, repo_root: Path) -> str | None:
    explicit = _string_or_none(
        row.get("archive_sha256")
        or row.get("archive_zip_sha256")
        or row.get("candidate_archive_sha256")
    )
    if explicit:
        return explicit
    archive_path = _archive_path(row, repo_root=repo_root)
    if archive_path is None or not archive_path.exists() or not archive_path.is_file():
        return None
    return _sha256_file(archive_path)


def _archive_path(row: Mapping[str, Any], *, repo_root: Path) -> Path | None:
    value = _string_or_none(
        row.get("archive_path")
        or row.get("archive_zip_path")
        or row.get("candidate_archive_path")
    )
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def _nonrate_from_components(row: Mapping[str, Any]) -> float | None:
    seg = _first_float(row, ("avg_segnet_dist", "d_seg"))
    pose = _first_float(row, ("avg_posenet_dist", "d_pose"))
    if seg is None or pose is None:
        return None
    from tac.auth_eval_schema import contest_formula_score

    return float(contest_formula_score(seg_dist=seg, pose_dist=pose, archive_bytes=0))


def _selected_source_field(plan: Mapping[str, Any], field: str) -> Any:
    selected = plan.get("receiver_closed_selected_point")
    if not isinstance(selected, Mapping):
        return None
    source = selected.get("source")
    if not isinstance(source, Mapping):
        return None
    return source.get(field)


def _recommended_next_actions(receiver_ready: bool) -> list[str]:
    if receiver_ready:
        return [
            "feed_receiver_closed_modelsize_budget_rows_to_carrier_training_plan",
            "run_byte_closed_local_replay_gate_before_exact_auth",
            "preserve_ladder_rows_as_rate_distortion_curve_evidence",
        ]
    return [
        "materialize_at_least_two_trained_receiver_archive_rows_with_modelsize_or_fc_dim",
        "prove_receiver_inflate_or_archive_replay_for_each_ladder_row",
        "attach_nonrate_score_or_seg_pose_component_deltas_to_each_ladder_row",
    ]


def _build_blocker_records(
    *,
    carrier: str,
    carrier_aliases: set[str],
    receiver_ready: bool,
    receiver_closed_rows: Sequence[Mapping[str, Any]],
    budget_rows: Sequence[Mapping[str, Any]],
    blocked_rows: Sequence[Mapping[str, Any]],
    modelsize_budget_blockers: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not receiver_ready:
        records.append(
            {
                "scope": "ladder",
                "reason": "receiver_closed_modelsize_ladder_not_ready",
                "carrier_id": carrier,
            }
        )
    if len(receiver_closed_rows) < 2:
        records.append(
            {
                "scope": "ladder",
                "reason": "receiver_closed_modelsize_ladder_has_fewer_than_two_points",
                "carrier_id": carrier,
                "receiver_closed_row_count": len(receiver_closed_rows),
                "minimum_receiver_closed_rows": 2,
            }
        )
    if not budget_rows:
        records.append(
            {
                "scope": "ladder",
                "reason": "no_rows_eligible_for_modelsize_budget_plan",
                "carrier_id": carrier,
            }
        )

    for row in blocked_rows:
        for blocker in row.get("blockers", []):
            reason = str(blocker)
            record: dict[str, Any] = {
                "scope": "row",
                "row_id": row.get("row_id"),
                "reason": reason,
                "carrier_id": carrier,
            }
            if reason == "carrier_family_mismatch":
                record.update(
                    {
                        "source_family": row.get("source_family"),
                        "source_family_key": row.get("source_family_key"),
                        "accepted_carrier_aliases": sorted(carrier_aliases),
                    }
                )
            records.append(record)

    for blocker in _sequence_or_empty(modelsize_budget_blockers):
        records.append(
            {
                "scope": "modelsize_budget",
                "reason": str(blocker),
                "carrier_id": carrier,
            }
        )

    records.append(
        {
            "scope": "authority",
            "reason": "nerv_receiver_closed_modelsize_ladder_is_false_authority",
            "carrier_id": carrier,
        }
    )
    records.append(
        {
            "scope": "dispatch",
            "reason": "exact_or_full_video_cuda_blocked_until_PR101_and_Z5_terminal",
            "carrier_id": carrier,
        }
    )
    return records


def _blocker_label(record: Mapping[str, Any]) -> str:
    scope = str(record.get("scope") or "")
    reason = str(record.get("reason") or "")
    if scope == "row":
        return f"row:{record.get('row_id')}:{reason}"
    if scope == "modelsize_budget":
        return f"modelsize_budget:{reason}"
    return reason


def _mapping_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    out = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise NervReceiverClosedModelsizeLadderError(
                f"all ladder rows must be mappings, got {type(row).__name__}"
            )
        out.append(row)
    return out


def _first_float(row: Mapping[str, Any], keys: Sequence[Any]) -> float | None:
    parsed, _key = _first_float_with_key(row, keys)
    return parsed


def _first_float_with_key(
    row: Mapping[str, Any],
    keys: Sequence[Any],
) -> tuple[float | None, str | None]:
    for key in keys:
        value = _lookup(row, key)
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(out):
            return out, _path_label(key)
    return None, None


def _first_int(row: Mapping[str, Any], keys: Sequence[Any]) -> int | None:
    for key in keys:
        value = _finite_int(_lookup(row, key))
        if value is not None:
            return value
    return None


def _finite_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def _lookup(row: Mapping[str, Any], key: Any) -> Any:
    if isinstance(key, tuple):
        value: Any = row
        for part in key:
            if not isinstance(value, Mapping):
                return None
            value = value.get(part)
        return value
    return row.get(key)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _authority_claim_blockers(row: Mapping[str, Any]) -> list[str]:
    return _ordered_unique(
        f"source_authority_flag_true:{key}"
        for key in _AUTHORITY_TRUE_KEYS
        if _truthy(row.get(key))
    )


def _axis_is_receiver_closed_authority(axis_tag: str | None) -> bool:
    if axis_tag is None:
        return False
    text = axis_tag.strip().lower()
    if any(token in text for token in _ADVISORY_AXIS_TOKENS):
        return False
    return text.startswith(_CONTEST_AUTH_AXIS_PREFIXES)


def _path_label(path: Any) -> str:
    if isinstance(path, tuple):
        return ".".join(str(part) for part in path)
    return str(path)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sequence_or_empty(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return value
    return (value,)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _carrier_aliases(carrier_id: str) -> set[str]:
    key = _carrier_key(carrier_id) or ""
    aliases = {key}
    if key in {"snerv", "snervt"}:
        aliases.update({"snerv", "snervt", "snerv_t"})
    if key in {"hinerv", "hi_nerv"}:
        aliases.update({"hinerv", "hi_nerv", "hi-nerv"})
    if key == "hnerv":
        aliases.add("hnerv")
    return aliases


def _carrier_key(value: Any) -> str | None:
    text = _string_or_none(value)
    if text is None:
        return None
    return text.strip().lower().replace("-", "_")


__all__ = [
    "SCHEMA",
    "NervReceiverClosedModelsizeLadderError",
    "build_nerv_receiver_closed_modelsize_ladder",
    "build_nerv_receiver_closed_modelsize_ladder_from_iterable",
]
