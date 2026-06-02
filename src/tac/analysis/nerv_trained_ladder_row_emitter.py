# SPDX-License-Identifier: MIT
"""Emit receiver-closed trained-row payloads for SNeRV/HiNeRV ladders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

from tac.auth_eval_schema import FULL_CONTEST_SAMPLE_COUNT, contest_formula_score
from tac.repo_io import sha256_file

SCHEMA = "nerv_trained_ladder_row_payload.v1"
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

FAMILY_ALIASES = {
    "snerv": {"snerv", "snerv_t", "snervt"},
    "hinerv": {"hinerv", "hi_nerv", "hi-nerv"},
}

FAMILY_REQUIRED_FIELDS: dict[str, tuple[Any, ...]] = {
    "snerv": (
        ("official_controls", "--modelsize"),
        ("official_controls", "fc_dim"),
        ("official_controls", "emb_size"),
        ("official_controls", "wavelet"),
        ("official_controls", "levels"),
        ("official_controls", "mfu_enabled"),
        ("official_controls", "hfr_enabled"),
        ("official_controls", "snerv_t_enabled"),
        "receiver_codec_mode",
        "lf_payload_codec",
        "decoder_precision_mode",
        "step_map_codec",
        "target_bits_per_coeff",
        "qat_bits",
    ),
    "hinerv": (
        ("official_controls", "config_name"),
        ("official_controls", "patch_mode"),
        ("official_controls", "hierarchical_grid_shapes"),
        ("official_controls", "decoder_channels"),
        ("official_controls", "prune_config"),
        ("official_controls", "quant_config"),
        ("official_controls", "bitstream_q"),
        "receiver_codec_mode",
        "decoder_precision_mode",
        "latent_precision_mode",
        "hierarchical_grid_precision_modes",
        "bitstream_codec",
    ),
}

_PAIR_COUNT_KEYS = ("pair_count", "n_pairs", "sample_pair_count", "n_samples", "sample_count")
_ARCHIVE_BYTE_KEYS = (
    "archive_bytes",
    "archive_zip_bytes",
    "archive_bytes_total",
    "measured_archive_bytes",
    "archive_size_bytes",
)
_ARCHIVE_SHA_KEYS = (
    "archive_sha256",
    "archive_zip_sha256",
    "receiver_archive_sha256",
    "candidate_archive_sha256",
    "sha256",
)
_SEG_KEYS = ("d_seg", "d_seg_linf", "d_seg_mean_linf", "avg_segnet_dist", "seg_distortion")
_POSE_KEYS = (
    "d_pose",
    "d_pose_linf",
    "d_pose_mean_linf",
    "avg_posenet_dist",
    "pose_distortion",
)
_RECEIVER_PROOF_KEYS = (
    "receiver_archive_replay_verified",
    "receiver_contract_satisfied",
    "byte_closed_receiver_proof",
    "runtime_consumption_proof_ready",
    "receiver_matches_direct",
)


class NervTrainedLadderRowEmitterError(ValueError):
    """Raised when trained-row emission inputs are malformed."""


def build_nerv_trained_ladder_row_payload(
    *,
    family: str,
    archive_path: str | Path | None = None,
    trainer_metadata: Mapping[str, Any] | None = None,
    receiver_proof: Mapping[str, Any] | None = None,
    scorer_eval: Mapping[str, Any] | None = None,
    official_controls: Mapping[str, Any] | None = None,
    row_id: str | None = None,
    n_pairs: int | None = None,
    modelsize_mparams: float | None = None,
    fc_dim: int | None = None,
    full_pair_count: int = FULL_CONTEST_SAMPLE_COUNT,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build one ladder-row payload from trainer/export/eval/proof metadata."""

    family_key = _family_key(family)
    if family_key not in FAMILY_ALIASES:
        raise NervTrainedLadderRowEmitterError(
            f"family must be one of {sorted(FAMILY_ALIASES)}, got {family!r}"
        )
    if full_pair_count <= 0:
        raise NervTrainedLadderRowEmitterError("full_pair_count must be positive")

    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    trainer = dict(trainer_metadata or {})
    proof = dict(receiver_proof or {})
    eval_payload = dict(scorer_eval or {})
    controls = _merged_controls(
        official_controls=official_controls,
        trainer=trainer,
        proof=proof,
        eval_payload=eval_payload,
    )
    archive = _resolve_archive_path(
        archive_path
        or _first_string(
            trainer,
            proof,
            eval_payload,
            keys=("archive_path", "archive_zip_path", "candidate_archive_path"),
        ),
        repo_root=root,
    )
    archive_bytes, archive_sha, archive_blockers = _archive_custody(
        archive,
        trainer=trainer,
        proof=proof,
        eval_payload=eval_payload,
    )

    pair_count = _first_int_value(
        n_pairs,
        *_values_for_keys((trainer, proof, eval_payload), _PAIR_COUNT_KEYS),
    )
    modelsize = _first_float_value(
        modelsize_mparams,
        trainer.get("modelsize_mparams"),
        trainer.get("modelsize"),
        controls.get("--modelsize"),
        _lookup(trainer, ("solved_budget", "modelsize_mparams")),
        _lookup(trainer, ("solved_budget", "official_controls", "--modelsize")),
    )
    fc = _first_int_value(
        fc_dim,
        trainer.get("fc_dim"),
        controls.get("fc_dim"),
        _lookup(trainer, ("derived", "fc_dim")),
        _lookup(trainer, ("solved_budget", "derived", "fc_dim")),
    )
    d_seg = _first_float_value(*_values_for_keys((eval_payload, trainer, proof), _SEG_KEYS))
    d_pose = _first_float_value(*_values_for_keys((eval_payload, trainer, proof), _POSE_KEYS))
    nonrate_score = _first_float_value(
        trainer.get("nonrate_score"),
        eval_payload.get("nonrate_score"),
        eval_payload.get("nonrate_score_value"),
        eval_payload.get("score_linf_without_rate"),
    )
    if nonrate_score is None and d_seg is not None and d_pose is not None:
        nonrate_score = float(
            contest_formula_score(seg_dist=d_seg, pose_dist=d_pose, archive_bytes=0)
        )
    receiver_replay = _truthy_first((proof, trainer, eval_payload), _RECEIVER_PROOF_KEYS)
    source_axis_tag = _first_string(
        eval_payload,
        trainer,
        proof,
        keys=("axis_tag", "axis_label", "evidence_axis", "score_axis"),
    )

    row = {
        "row_id": row_id or _row_id(family_key, trainer, eval_payload, archive),
        "family": family_key,
        "carrier_id": family_key,
        "sample_pair_count": pair_count,
        "n_pairs": pair_count,
        "archive_path": archive.as_posix() if archive is not None else None,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "modelsize_mparams": modelsize,
        "fc_dim": fc,
        "official_controls": controls,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "avg_segnet_dist": d_seg,
        "avg_posenet_dist": d_pose,
        "nonrate_score": nonrate_score,
        "receiver_archive_replay_verified": bool(receiver_replay),
        "receiver_contract_satisfied": bool(receiver_replay),
        "byte_closed_receiver_proof": bool(receiver_replay),
        "receiver_proof_passed": bool(
            receiver_replay
            and pair_count is not None
            and int(pair_count) >= int(full_pair_count)
        ),
        "receiver_closed": bool(
            receiver_replay
            and pair_count is not None
            and int(pair_count) >= int(full_pair_count)
        ),
        "source_axis_tag": source_axis_tag,
        "receiver_codec_mode": _first_present(
            "receiver_codec_mode", trainer, controls, proof, eval_payload
        ),
        "lf_payload_codec": _first_present(
            "lf_payload_codec", trainer, controls, proof, eval_payload
        ),
        "decoder_precision_mode": _first_present(
            "decoder_precision_mode", trainer, controls, proof, eval_payload
        ),
        "step_map_codec": _first_present(
            "step_map_codec", trainer, controls, proof, eval_payload
        ),
        "target_bits_per_coeff": _first_present(
            "target_bits_per_coeff", trainer, controls, proof, eval_payload
        ),
        "qat_bits": _first_present("qat_bits", trainer, controls, proof, eval_payload),
        "latent_precision_mode": _first_present(
            "latent_precision_mode", trainer, controls, proof, eval_payload
        ),
        "hierarchical_grid_precision_modes": _first_present(
            "hierarchical_grid_precision_modes", trainer, controls, proof, eval_payload
        ),
        "bitstream_codec": _first_present(
            "bitstream_codec", trainer, controls, proof, eval_payload
        ),
        **FALSE_AUTHORITY,
    }

    blockers = _ordered_unique(
        [
            *archive_blockers,
            *(["sample_pair_count_missing"] if pair_count is None else []),
            *(
                ["sample_pair_count_below_full600"]
                if pair_count is not None and int(pair_count) < int(full_pair_count)
                else []
            ),
            *(["modelsize_or_fc_dim_missing"] if modelsize is None and fc is None else []),
            *(["nonrate_score_or_component_distortions_missing"] if nonrate_score is None else []),
            *(["receiver_replay_or_contract_missing"] if not receiver_replay else []),
            *_required_field_blockers(family_key, row),
        ]
    )
    accepted = not blockers
    row["accepted"] = accepted
    row["emission_blockers"] = blockers

    return {
        "schema": SCHEMA,
        "axis_tag": AXIS_TAG,
        "family": family_key,
        "status": (
            "trained_ladder_row_ready"
            if accepted
            else "trained_ladder_row_blocked"
        ),
        "verdict": (
            "GO_HARVEST_INPUT__NO_GO_SCORE_OR_EXACT_AUTH"
            if accepted
            else "NO_GO_HARVEST_INPUT__TRAINED_ROW_PROOF_INCOMPLETE"
        ),
        "full_pair_count": int(full_pair_count),
        "archive_custody": {
            "archive_path": archive.as_posix() if archive is not None else None,
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha,
            "computed_from_file": archive_bytes is not None and archive_sha is not None,
        },
        "source_schemas": {
            "trainer_metadata_schema": trainer.get("schema"),
            "receiver_proof_schema": proof.get("schema"),
            "scorer_eval_schema": eval_payload.get("schema"),
        },
        "rows": [row],
        "ready_for_receiver_closed_ladder_harvest": accepted,
        "ready_for_receiver_closed_modelsize_ladder": False,
        "blockers": blockers,
        "next_actions": _next_actions(family_key, blockers),
        **FALSE_AUTHORITY,
    }


def _archive_custody(
    archive: Path | None,
    *,
    trainer: Mapping[str, Any],
    proof: Mapping[str, Any],
    eval_payload: Mapping[str, Any],
) -> tuple[int | None, str | None, list[str]]:
    blockers: list[str] = []
    if archive is None:
        return None, None, ["archive_path_missing"]
    if not archive.is_file():
        return None, None, [f"archive_path_not_file:{archive.as_posix()}"]
    actual_bytes = int(archive.stat().st_size)
    actual_sha = sha256_file(archive)
    for value in _values_for_keys((trainer, proof, eval_payload), _ARCHIVE_BYTE_KEYS):
        parsed = _finite_int(value)
        if parsed is not None and parsed != actual_bytes:
            blockers.append("archive_bytes_metadata_mismatch")
            break
    for value in _values_for_keys((trainer, proof, eval_payload), _ARCHIVE_SHA_KEYS):
        text = _string_or_none(value)
        if text is not None and text != actual_sha:
            blockers.append("archive_sha256_metadata_mismatch")
            break
    return actual_bytes, actual_sha, blockers


def _required_field_blockers(family: str, row: Mapping[str, Any]) -> list[str]:
    blockers = []
    for key in FAMILY_REQUIRED_FIELDS[family]:
        if _lookup(row, key) is None:
            blockers.append(f"required_emission_field_missing:{_path_label(key)}")
    return blockers


def _next_actions(family: str, blockers: Sequence[str]) -> list[str]:
    actions = []
    labels = set(blockers)
    if any(label.startswith("archive_") for label in labels):
        actions.append(f"{family}: export a real receiver archive and pass its path")
    if "sample_pair_count_below_full600" in labels or "sample_pair_count_missing" in labels:
        actions.append(f"{family}: emit full600 pair_count/n_pairs metadata")
    if "modelsize_or_fc_dim_missing" in labels:
        actions.append(f"{family}: emit source-bound modelsize_mparams or fc_dim")
    if "receiver_replay_or_contract_missing" in labels:
        actions.append(f"{family}: attach receiver archive replay proof")
    if "nonrate_score_or_component_distortions_missing" in labels:
        actions.append(f"{family}: attach SegNet/PoseNet component deltas")
    if any(label.startswith("required_emission_field_missing:") for label in labels):
        actions.append(f"{family}: emit official source controls and receiver codec controls")
    return _ordered_unique(actions)


def _merged_controls(
    *,
    official_controls: Mapping[str, Any] | None,
    trainer: Mapping[str, Any],
    proof: Mapping[str, Any],
    eval_payload: Mapping[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source in (trainer, proof, eval_payload):
        value = source.get("official_controls")
        if isinstance(value, Mapping):
            out.update(dict(value))
    if official_controls is not None:
        out.update(dict(official_controls))
    return out


def _resolve_archive_path(value: str | Path | None, *, repo_root: Path) -> Path | None:
    text = _string_or_none(value)
    if text is None:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def _row_id(
    family: str,
    trainer: Mapping[str, Any],
    eval_payload: Mapping[str, Any],
    archive: Path | None,
) -> str:
    explicit = _string_or_none(
        trainer.get("row_id")
        or trainer.get("id")
        or eval_payload.get("row_id")
        or eval_payload.get("id")
    )
    if explicit:
        return explicit
    stem = archive.stem if archive is not None else "missing_archive"
    return f"{family}_trained_ladder_row_{stem}"


def _first_string(*rows: Mapping[str, Any], keys: Sequence[Any]) -> str | None:
    for value in _values_for_keys(rows, keys):
        text = _string_or_none(value)
        if text is not None:
            return text
    return None


def _values_for_keys(rows: Sequence[Mapping[str, Any]], keys: Sequence[Any]) -> list[Any]:
    return [_lookup(row, key) for key in keys for row in rows]


def _first_present(key: str, *rows: Mapping[str, Any]) -> Any:
    for row in rows:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _lookup(row: Mapping[str, Any], key: Any) -> Any:
    if isinstance(key, tuple):
        value: Any = row
        for part in key:
            if not isinstance(value, Mapping):
                return None
            value = value.get(part)
        return value
    return row.get(key)


def _truthy_first(rows: Sequence[Mapping[str, Any]], keys: Sequence[Any]) -> bool:
    return any(_truthy(_lookup(row, key)) for row in rows for key in keys)


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _first_int_value(*values: Any) -> int | None:
    for value in values:
        parsed = _finite_int(value)
        if parsed is not None:
            return parsed
    return None


def _first_float_value(*values: Any) -> float | None:
    for value in values:
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed
    return None


def _finite_int(value: Any) -> int | None:
    parsed = _finite_float(value)
    if parsed is None or int(parsed) != parsed:
        return None
    return int(parsed)


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _family_key(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text == "hi_nerv":
        return "hinerv"
    return text


def _path_label(path: Any) -> str:
    if isinstance(path, tuple):
        return ".".join(str(part) for part in path)
    return str(path)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ordered_unique(values: Sequence[str]) -> list[str]:
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
    "NervTrainedLadderRowEmitterError",
    "build_nerv_trained_ladder_row_payload",
]
