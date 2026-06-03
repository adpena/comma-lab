# SPDX-License-Identifier: MIT
"""Harvest measured NeRV receiver rows without granting false ladder authority."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from tac.auth_eval_schema import FULL_CONTEST_SAMPLE_COUNT, contest_formula_score

SCHEMA = "nerv_receiver_closed_ladder_row_harvest.v1"
AXIS_TAG = "[planning/control:false-authority]"
FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "production_hardened_claim": False,
    "ready_for_exact_eval_dispatch": False,
}
_AUTHORITY_TRUE_KEYS = (
    "score_claim",
    "score_claim_valid",
    "frontier_score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "production_hardened_claim",
    "ready_for_exact_eval_dispatch",
)
_ADVISORY_AXIS_TOKENS = (
    "advisory",
    "projected",
    "predicted",
    "proxy",
    "research-signal",
    "macos",
    "mps",
)


class NervReceiverClosedLadderRowHarvestError(ValueError):
    """Raised when row-harvest inputs are malformed."""


@dataclass(frozen=True)
class ReceiverRowSource:
    """A JSON artifact plus optional path provenance."""

    payload: Mapping[str, Any]
    path: str | None = None


def build_nerv_receiver_closed_ladder_row_harvest(
    sources: Sequence[ReceiverRowSource | Mapping[str, Any]],
    *,
    carrier_id: str,
    full_pair_count: int = FULL_CONTEST_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Return ladder-ingestable rows while preserving local-smoke blockers."""

    carrier = str(carrier_id).strip()
    if not carrier:
        raise NervReceiverClosedLadderRowHarvestError("carrier_id must be non-empty")
    if full_pair_count <= 0:
        raise NervReceiverClosedLadderRowHarvestError("full_pair_count must be positive")

    normalized_sources = [_normalize_source(source) for source in sources]
    harvested_rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source_index, source in enumerate(normalized_sources):
        payload = dict(source.payload)
        source_path = source.path
        source_schema = _string_or_none(payload.get("schema"))
        source_axis_tag = _string_or_none(payload.get("axis_tag"))
        source_family = _source_family(payload)
        candidates = _candidate_rows(payload)
        source_summaries.append(
            {
                "source_index": source_index,
                "source_artifact_path": source_path,
                "source_schema": source_schema,
                "source_axis_tag": source_axis_tag,
                "source_family": source_family,
                "candidate_count": len(candidates),
            }
        )
        for candidate_index, candidate in enumerate(candidates):
            harvested_rows.append(
                _harvest_row(
                    source_index=source_index,
                    candidate_index=candidate_index,
                    source_path=source_path,
                    source_schema=source_schema,
                    source_axis_tag=source_axis_tag,
                    source_family=source_family,
                    source_payload=payload,
                    candidate=candidate,
                    carrier_id=carrier,
                    full_pair_count=full_pair_count,
                )
            )

    full_scope_rows = [
        row for row in harvested_rows if row["sample_scope"] == "full600_or_better"
    ]
    receiver_proof_rows = [row for row in harvested_rows if row["receiver_proof_passed"]]
    local_replay_rows = [
        row for row in harvested_rows if row["local_receiver_archive_replay_verified"]
    ]
    modelsize_present_rows = [
        row
        for row in harvested_rows
        if _row_has_capacity_axis(row)
    ]
    ladder_candidate_rows = [
        row
        for row in harvested_rows
        if row["receiver_proof_passed"]
        and not row["harvest_blockers"]
        and _row_has_capacity_axis(row)
        and row.get("archive_bytes") is not None
        and row.get("nonrate_score") is not None
    ]
    blockers = _ordered_unique(
        [
            *(
                []
                if len(ladder_candidate_rows) >= 2
                else ["fewer_than_two_full600_receiver_modelsize_rows"]
            ),
            *([] if receiver_proof_rows else ["no_full600_receiver_proof_rows"]),
            *([] if modelsize_present_rows else ["modelsize_or_fc_dim_missing_for_all_rows"]),
            *[
                f"row:{row['row_id']}:{blocker}"
                for row in harvested_rows
                for blocker in row["harvest_blockers"]
            ],
            "harvest_rows_are_false_authority_until_ladder_and_full600_replay_pass",
        ]
    )
    ready = len(ladder_candidate_rows) >= 2
    return {
        "schema": SCHEMA,
        "axis_tag": AXIS_TAG,
        "carrier_id": carrier,
        "full_pair_count": int(full_pair_count),
        "status": (
            "receiver_closed_ladder_rows_ready"
            if ready
            else "receiver_closed_ladder_rows_blocked"
        ),
        "verdict": (
            "GO_LADDER_INPUT__NO_GO_SCORE_OR_EXACT_AUTH"
            if ready
            else "NO_GO_LADDER_INPUT__FULL600_RECEIVER_ROWS_MISSING"
        ),
        "source_count": len(normalized_sources),
        "source_summaries": source_summaries,
        "harvested_row_count": len(harvested_rows),
        "full_scope_row_count": len(full_scope_rows),
        "local_receiver_replay_row_count": len(local_replay_rows),
        "receiver_proof_row_count": len(receiver_proof_rows),
        "modelsize_present_row_count": len(modelsize_present_rows),
        "ladder_candidate_row_count": len(ladder_candidate_rows),
        "harvested_rows": harvested_rows,
        "blockers": blockers,
        "ready_for_receiver_closed_modelsize_ladder": ready,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def build_nerv_receiver_closed_ladder_row_harvest_from_iterable(
    sources: Iterable[ReceiverRowSource | Mapping[str, Any]],
    *,
    carrier_id: str,
    full_pair_count: int = FULL_CONTEST_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Build a row harvest from any source iterable."""

    return build_nerv_receiver_closed_ladder_row_harvest(
        list(sources), carrier_id=carrier_id, full_pair_count=full_pair_count
    )


def _harvest_row(
    *,
    source_index: int,
    candidate_index: int,
    source_path: str | None,
    source_schema: str | None,
    source_axis_tag: str | None,
    source_family: str | None,
    source_payload: Mapping[str, Any],
    candidate: Mapping[str, Any],
    carrier_id: str,
    full_pair_count: int,
) -> dict[str, Any]:
    row_id = _row_id(source_index, candidate_index, source_path, candidate)
    pair_count = _first_int(
        candidate,
        source_payload,
        keys=("pair_count", "n_pairs", "num_pairs", "sample_count", "n_samples"),
    )
    sample_scope = (
        "full600_or_better"
        if pair_count is not None and pair_count >= full_pair_count
        else "local_pair_smoke"
    )
    archive_bytes = _first_int(
        candidate,
        source_payload,
        keys=(
            "measured_archive_bytes",
            "archive_bytes",
            "archive_bytes_total",
            "archive_zip_bytes",
            "archive_size_bytes",
            "bytes",
        ),
    )
    archive_sha = _first_string(
        candidate,
        source_payload,
        keys=("archive_sha256", "receiver_archive_sha256", "archive_zip_sha256", "sha256"),
    )
    d_seg = _first_float(
        candidate,
        source_payload,
        keys=(
            "d_seg",
            "d_seg_linf",
            "d_seg_mean_linf",
            "avg_segnet_dist",
            "seg_distortion",
            ("score_aware_training", "d_seg_mean_linf"),
            ("score_aware_training", "d_seg"),
            ("score_aware_training", "d_seg_linf"),
        ),
    )
    d_pose = _first_float(
        candidate,
        source_payload,
        keys=(
            "d_pose",
            "d_pose_linf",
            "d_pose_mean_linf",
            "avg_posenet_dist",
            "pose_distortion",
            ("score_aware_training", "d_pose_mean_linf"),
            ("score_aware_training", "d_pose"),
            ("score_aware_training", "d_pose_linf"),
        ),
    )
    nonrate_score = _first_float(
        candidate,
        source_payload,
        keys=("nonrate_score", "nonrate_score_value", "score_linf_without_rate"),
    )
    if nonrate_score is None and d_seg is not None and d_pose is not None:
        nonrate_score = float(
            contest_formula_score(seg_dist=d_seg, pose_dist=d_pose, archive_bytes=0)
        )
    modelsize = _first_float(
        candidate,
        source_payload,
        keys=(
            "modelsize_mparams",
            "modelsize",
            ("official_controls", "--modelsize"),
            ("solved_budget", "modelsize_mparams"),
            ("solved_budget", "official_controls", "--modelsize"),
            ("modelsize_candidate_selection", "candidate", "modelsize_mparams"),
        ),
    )
    modelsize_scale = _first_float(
        candidate,
        source_payload,
        keys=(
            "modelsize_scale",
            ("solved_budget", "modelsize_scale"),
            ("modelsize_candidate_selection", "candidate", "modelsize_scale"),
        ),
    )
    fc_dim = _first_int(
        candidate,
        source_payload,
        keys=(
            "fc_dim",
            ("derived", "fc_dim"),
            ("solved_budget", "derived", "fc_dim"),
            ("modelsize_candidate_selection", "candidate", "fc_dim"),
        ),
    )
    snerv_levels = _first_int(
        candidate,
        source_payload,
        keys=(
            "levels",
            ("modelsize_candidate_selection", "candidate", "levels"),
            ("score_aware_training", "levels"),
        ),
    )
    snerv_bits_per_coeff = _first_float(
        candidate,
        source_payload,
        keys=(
            "bits_per_coeff",
            "target_bits_per_coeff",
            ("modelsize_candidate_selection", "candidate", "bits_per_coeff"),
            ("score_aware_training", "target_bits_per_coeff"),
        ),
    )
    snerv_step_map_bits_per_coeff = _first_float(
        candidate,
        source_payload,
        keys=(
            "step_map_bits_per_coeff",
            ("modelsize_candidate_selection", "candidate", "step_map_bits_per_coeff"),
            ("score_aware_training", "step_map_waterfill_bits_per_coeff"),
        ),
    )
    decoder_payload_codec = _first_string(
        candidate,
        source_payload,
        keys=(
            "decoder_payload_codec",
            ("modelsize_candidate_selection", "candidate", "decoder_payload_codec"),
            ("score_aware_training", "decoder_payload_codec"),
        ),
    )
    local_receiver_replay = _truthy_first(
        candidate,
        source_payload,
        keys=(
            "receiver_archive_replay_verified",
            "receiver_contract_satisfied",
            "runtime_consumption_proof_ready",
            "receiver_matches_direct",
            ("score_aware_training", "receiver_archive_replay_verified"),
            ("score_aware_training", "receiver_contract_satisfied"),
            ("score_aware_training", "mlx_native_receiver_proof_passed"),
            ("snerv_mlx_native_export", "receiver_proof_passed"),
            ("snerv_mlx_native_export", "receiver_contract_satisfied"),
            ("hi_nerv_mlx_export", "receiver_proof_passed"),
            ("hi_nerv_mlx_export", "receiver_contract_satisfied"),
        ),
    )
    accepted = _accepted(candidate)
    family = _string_or_none(
        candidate.get("family") or candidate.get("carrier_id") or source_family
    )
    axis_blocked = _axis_is_advisory_or_projected(source_axis_tag)
    authority_blockers = _authority_claim_blockers(candidate, source_payload)
    full600_receiver_proof = bool(
        local_receiver_replay
        and sample_scope == "full600_or_better"
        and not axis_blocked
        and not authority_blockers
    )

    blockers: list[str] = []
    if family is not None and _carrier_key(family) != _carrier_key(carrier_id):
        blockers.append("carrier_family_mismatch")
    if not accepted:
        blockers.append("source_row_not_accepted")
    if archive_bytes is None:
        blockers.append("measured_archive_bytes_missing")
    if archive_sha is None:
        blockers.append("archive_sha256_missing")
    if nonrate_score is None:
        blockers.append("nonrate_score_or_component_distortions_missing")
    has_capacity_axis = _capacity_axis_present(
        modelsize_mparams=modelsize,
        modelsize_scale=modelsize_scale,
        fc_dim=fc_dim,
        snerv_levels=snerv_levels,
        snerv_bits_per_coeff=snerv_bits_per_coeff,
    )
    if not has_capacity_axis:
        blockers.append("modelsize_or_fc_dim_missing")
    if not local_receiver_replay:
        blockers.append("receiver_replay_or_contract_missing")
    if sample_scope != "full600_or_better":
        blockers.append("local_smoke_only_not_full600_receiver_proof")
    if axis_blocked:
        blockers.append(
            "source_axis_advisory_or_projected_not_receiver_closed_ladder_authority"
        )
    blockers.extend(authority_blockers)

    return {
        "row_id": row_id,
        "carrier_id": family or carrier_id,
        "family": family or carrier_id,
        "source_artifact_path": source_path,
        "source_schema": source_schema,
        "source_axis_tag": source_axis_tag,
        "source_candidate_index": candidate_index,
        "source_label": _string_or_none(
            candidate.get("sweep_label")
            or candidate.get("label")
            or candidate.get("source_artifact")
        ),
        "sample_pair_count": pair_count,
        "sample_scope": sample_scope,
        "modelsize_mparams": modelsize,
        "modelsize_scale": modelsize_scale,
        "fc_dim": fc_dim,
        "snerv_levels": snerv_levels,
        "snerv_bits_per_coeff": snerv_bits_per_coeff,
        "snerv_step_map_bits_per_coeff": snerv_step_map_bits_per_coeff,
        "decoder_payload_codec": decoder_payload_codec,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "nonrate_score": nonrate_score,
        "accepted": accepted,
        "source_axis_is_advisory_or_projected": axis_blocked,
        "local_receiver_archive_replay_verified": bool(local_receiver_replay),
        "full600_receiver_proof": full600_receiver_proof,
        "receiver_proof_passed": full600_receiver_proof,
        "receiver_closed": full600_receiver_proof,
        "receiver_archive_replay_verified_local_only": bool(local_receiver_replay),
        "harvest_blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _candidate_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    explicit = payload.get("rows")
    if isinstance(explicit, list):
        rows.extend(row for row in explicit if isinstance(row, Mapping))
    archive_rows = payload.get("archive_rows")
    if isinstance(archive_rows, list):
        rows.extend(row for row in archive_rows if isinstance(row, Mapping))
    profiles = payload.get("profile_records")
    if isinstance(profiles, list):
        rows.extend(row for row in profiles if isinstance(row, Mapping))
    for key in ("best", "baseline"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            row = dict(value)
            row.setdefault("sweep_label", key)
            rows.append(row)
    if not rows:
        rows.append(payload)
    return rows


def _normalize_source(source: ReceiverRowSource | Mapping[str, Any]) -> ReceiverRowSource:
    if isinstance(source, ReceiverRowSource):
        if not isinstance(source.payload, Mapping):
            raise NervReceiverClosedLadderRowHarvestError("source payload must be a mapping")
        return source
    if not isinstance(source, Mapping):
        raise NervReceiverClosedLadderRowHarvestError(
            f"source must be a mapping, got {type(source).__name__}"
        )
    payload = source.get("payload")
    if isinstance(payload, Mapping):
        return ReceiverRowSource(payload=payload, path=_string_or_none(source.get("path")))
    return ReceiverRowSource(payload=source, path=_string_or_none(source.get("path")))


def _source_family(payload: Mapping[str, Any]) -> str | None:
    explicit = _string_or_none(payload.get("family") or payload.get("carrier_id"))
    if explicit is not None:
        return explicit
    schema = _string_or_none(payload.get("schema"))
    if schema is None:
        return None
    key = schema.lower().replace("-", "_")
    if key.startswith("snerv"):
        return "snerv"
    if key.startswith("hinerv") or key.startswith("hi_nerv"):
        return "hinerv"
    if key.startswith("hnerv"):
        return "hnerv"
    return None


def _row_has_capacity_axis(row: Mapping[str, Any]) -> bool:
    return _capacity_axis_present(
        modelsize_mparams=row.get("modelsize_mparams"),
        modelsize_scale=row.get("modelsize_scale"),
        fc_dim=row.get("fc_dim"),
        snerv_levels=row.get("snerv_levels"),
        snerv_bits_per_coeff=row.get("snerv_bits_per_coeff"),
    )


def _capacity_axis_present(
    *,
    modelsize_mparams: Any,
    modelsize_scale: Any,
    fc_dim: Any,
    snerv_levels: Any,
    snerv_bits_per_coeff: Any,
) -> bool:
    return (
        modelsize_mparams is not None
        or modelsize_scale is not None
        or fc_dim is not None
        or (snerv_levels is not None and snerv_bits_per_coeff is not None)
    )


def _row_id(
    source_index: int,
    candidate_index: int,
    source_path: str | None,
    candidate: Mapping[str, Any],
) -> str:
    explicit = _string_or_none(candidate.get("row_id") or candidate.get("id"))
    if explicit:
        return explicit
    stem = Path(source_path).stem if source_path else f"source_{source_index:04d}"
    label = _string_or_none(candidate.get("sweep_label") or candidate.get("label"))
    suffix = label or f"row_{candidate_index:04d}"
    return f"{stem}:{suffix}"


def _accepted(candidate: Mapping[str, Any]) -> bool:
    if "accepted" in candidate:
        return candidate.get("accepted") is True
    if "accepted_improvement" in candidate:
        return candidate.get("accepted_improvement") is True
    return True


def _first_float(
    primary: Mapping[str, Any],
    fallback: Mapping[str, Any],
    *,
    keys: Sequence[Any],
) -> float | None:
    for key in keys:
        for row in (primary, fallback):
            value = _lookup(row, key)
            try:
                out = float(value)
            except (TypeError, ValueError):
                continue
            if isfinite(out):
                return out
    return None


def _first_int(
    primary: Mapping[str, Any],
    fallback: Mapping[str, Any],
    *,
    keys: Sequence[Any],
) -> int | None:
    parsed = _first_float(primary, fallback, keys=keys)
    if parsed is None or int(parsed) != parsed:
        return None
    return int(parsed)


def _first_string(
    primary: Mapping[str, Any],
    fallback: Mapping[str, Any],
    *,
    keys: Sequence[Any],
) -> str | None:
    for key in keys:
        for row in (primary, fallback):
            value = _string_or_none(_lookup(row, key))
            if value is not None:
                return value
    return None


def _truthy_first(
    primary: Mapping[str, Any],
    fallback: Mapping[str, Any],
    *,
    keys: Sequence[Any],
) -> bool:
    for key in keys:
        for row in (primary, fallback):
            if _truthy(_lookup(row, key)):
                return True
    return False


def _lookup(row: Mapping[str, Any], key: Any) -> Any:
    if isinstance(key, tuple):
        value: Any = row
        for part in key:
            if not isinstance(value, Mapping):
                return None
            value = value.get(part)
        return value
    return row.get(key)


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _axis_is_advisory_or_projected(axis_tag: str | None) -> bool:
    if axis_tag is None:
        return False
    text = axis_tag.strip().lower()
    return any(token in text for token in _ADVISORY_AXIS_TOKENS)


def _authority_claim_blockers(*rows: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for row in rows:
        for key in _AUTHORITY_TRUE_KEYS:
            if _truthy(row.get(key)):
                blockers.append(f"source_authority_flag_true:{key}")
    return _ordered_unique(blockers)


def _carrier_key(value: Any) -> str | None:
    text = _string_or_none(value)
    if text is None:
        return None
    return text.strip().lower().replace("-", "_")


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


__all__ = [
    "SCHEMA",
    "NervReceiverClosedLadderRowHarvestError",
    "ReceiverRowSource",
    "build_nerv_receiver_closed_ladder_row_harvest",
    "build_nerv_receiver_closed_ladder_row_harvest_from_iterable",
]
