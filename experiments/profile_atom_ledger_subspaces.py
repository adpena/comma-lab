#!/usr/bin/env python3
"""Profile planning-only atom ledgers into active subspace summaries.

This tool is intentionally non-promotable. It turns CMG/Yousfi-Fridrich atom
ledgers into deterministic summaries that can guide archive construction, exact
CUDA dispatch, and paper figures without becoming score evidence by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _as_int(value: Any, *, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _first_int(values: Any) -> int | None:
    if isinstance(values, list) and values:
        return int(values[0])
    if isinstance(values, int):
        return int(values)
    return None


def _hist_keys(histogram: Any) -> tuple[int, ...]:
    if not isinstance(histogram, Mapping):
        return ()
    return tuple(sorted(int(key) for key in histogram.keys()))


def _atom_feature(atom: Mapping[str, Any], *, rank: int) -> dict[str, Any]:
    identity = atom.get("identity") if isinstance(atom.get("identity"), Mapping) else {}
    lagrangian = atom.get("lagrangian") if isinstance(atom.get("lagrangian"), Mapping) else {}
    cost = atom.get("cost_model") if isinstance(atom.get("cost_model"), Mapping) else {}
    bbox = atom.get("bbox_xyxy") if isinstance(atom.get("bbox_xyxy"), list) else None
    class_ids = atom.get("class_ids")

    pair_index = identity.get("pair_index")
    frame_index = identity.get("frame_index")
    class_id = identity.get("class_id")
    if pair_index is None:
        pair_index = _first_int(atom.get("pair_indices"))
    if frame_index is None:
        frame_index = _first_int(atom.get("frame_indices"))
    if class_id is None:
        class_id = _first_int(class_ids)

    estimated_bytes = _as_float(cost.get("estimated_charged_bytes"), default=0.0)
    score_saved = _as_float(lagrangian.get("estimated_marginal_score_saved_proxy"), default=0.0)
    net_proxy = _as_float(lagrangian.get("estimated_lagrangian_net_proxy"), default=0.0)
    residual_pixels = _as_int(atom.get("residual_pixels"), default=0)

    centroid = None
    if bbox is not None and len(bbox) == 4:
        centroid = {
            "x": (float(bbox[0]) + float(bbox[2])) / 2.0,
            "y": (float(bbox[1]) + float(bbox[3])) / 2.0,
        }

    return {
        "atom_family": str(atom.get("atom_family", "unknown")),
        "atom_id": str(atom.get("atom_id", f"rank_{rank}")),
        "candidate_classes": _hist_keys(atom.get("candidate_class_histogram_pixels")),
        "centroid": centroid,
        "class_id": None if class_id is None else int(class_id),
        "estimated_charged_bytes": estimated_bytes,
        "estimated_lagrangian_net_proxy": net_proxy,
        "estimated_marginal_score_saved_proxy": score_saved,
        "estimated_score_saved_per_charged_byte": 0.0
        if estimated_bytes <= 0
        else score_saved / estimated_bytes,
        "frame_index": None if frame_index is None else int(frame_index),
        "pair_index": None if pair_index is None else int(pair_index),
        "rank": rank,
        "residual_pixels": residual_pixels,
        "source_classes": _hist_keys(atom.get("source_class_histogram_pixels")),
    }


def _accumulate(
    table: dict[str, dict[str, Any]],
    key: str,
    feature: Mapping[str, Any],
) -> None:
    entry = table.setdefault(
        key,
        {
            "estimated_charged_bytes_sum": 0.0,
            "estimated_lagrangian_net_proxy_sum": 0.0,
            "estimated_marginal_score_saved_proxy_sum": 0.0,
            "hit_count": 0,
            "rank_sum": 0.0,
            "residual_pixels_sum": 0,
            "weighted_centroid_x_sum": 0.0,
            "weighted_centroid_y_sum": 0.0,
            "weight_sum_for_centroid": 0.0,
        },
    )
    score_saved = _as_float(feature.get("estimated_marginal_score_saved_proxy"))
    entry["estimated_charged_bytes_sum"] += _as_float(feature.get("estimated_charged_bytes"))
    entry["estimated_lagrangian_net_proxy_sum"] += _as_float(
        feature.get("estimated_lagrangian_net_proxy")
    )
    entry["estimated_marginal_score_saved_proxy_sum"] += score_saved
    entry["hit_count"] += 1
    entry["rank_sum"] += _as_float(feature.get("rank"))
    entry["residual_pixels_sum"] += _as_int(feature.get("residual_pixels"))

    centroid = feature.get("centroid")
    if isinstance(centroid, Mapping):
        weight = max(score_saved, 0.0)
        entry["weighted_centroid_x_sum"] += _as_float(centroid.get("x")) * weight
        entry["weighted_centroid_y_sum"] += _as_float(centroid.get("y")) * weight
        entry["weight_sum_for_centroid"] += weight


def _finalize_table(table: Mapping[str, Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, entry in table.items():
        hit_count = _as_int(entry.get("hit_count"))
        byte_sum = _as_float(entry.get("estimated_charged_bytes_sum"))
        score_sum = _as_float(entry.get("estimated_marginal_score_saved_proxy_sum"))
        weight_sum = _as_float(entry.get("weight_sum_for_centroid"))
        row = {
            "estimated_charged_bytes_sum": round(byte_sum, 12),
            "estimated_lagrangian_net_proxy_sum": round(
                _as_float(entry.get("estimated_lagrangian_net_proxy_sum")), 12
            ),
            "estimated_marginal_score_saved_proxy_sum": round(score_sum, 12),
            "estimated_score_saved_per_charged_byte": 0.0
            if byte_sum <= 0
            else round(score_sum / byte_sum, 12),
            "hit_count": hit_count,
            "key": key,
            "mean_rank": None if hit_count == 0 else round(_as_float(entry.get("rank_sum")) / hit_count, 6),
            "residual_pixels_sum": _as_int(entry.get("residual_pixels_sum")),
            "weighted_centroid": None
            if weight_sum <= 0
            else {
                "x": round(_as_float(entry.get("weighted_centroid_x_sum")) / weight_sum, 6),
                "y": round(_as_float(entry.get("weighted_centroid_y_sum")) / weight_sum, 6),
            },
        }
        rows.append(row)

    rows.sort(
        key=lambda item: (
            -float(item["estimated_lagrangian_net_proxy_sum"]),
            -float(item["estimated_marginal_score_saved_proxy_sum"]),
            int(item["hit_count"]),
            str(item["key"]),
        )
    )
    return rows[:limit]


def _profile_one(path: Path, payload: Mapping[str, Any], *, top_k: int, table_limit: int) -> dict[str, Any]:
    atoms = payload.get("top_atoms")
    if not isinstance(atoms, list):
        raise ValueError(f"{path} has no top_atoms list")
    selected = atoms[:top_k]
    features = [_atom_feature(atom, rank=rank) for rank, atom in enumerate(selected, start=1)]

    families: dict[str, dict[str, Any]] = {}
    pairs: dict[str, dict[str, Any]] = {}
    frames: dict[str, dict[str, Any]] = {}
    classes: dict[str, dict[str, Any]] = {}
    source_to_candidate: dict[str, dict[str, Any]] = {}

    for feature in features:
        _accumulate(families, str(feature["atom_family"]), feature)
        if feature["pair_index"] is not None:
            _accumulate(pairs, str(feature["pair_index"]), feature)
        if feature["frame_index"] is not None:
            _accumulate(frames, str(feature["frame_index"]), feature)
        if feature["class_id"] is not None:
            _accumulate(classes, str(feature["class_id"]), feature)
        for source in feature["source_classes"]:
            for candidate in feature["candidate_classes"]:
                _accumulate(source_to_candidate, f"{source}->{candidate}", feature)

    total_bytes = sum(_as_float(item["estimated_charged_bytes"]) for item in features)
    total_score_saved = sum(_as_float(item["estimated_marginal_score_saved_proxy"]) for item in features)
    total_net = sum(_as_float(item["estimated_lagrangian_net_proxy"]) for item in features)

    return {
        "atom_count_profiled": len(features),
        "evidence_grade": str(payload.get("evidence_grade", "unknown")),
        "input": {
            "path": str(path),
            "sha256": _sha256_file(path),
            "schema": payload.get("schema"),
            "score_claim": _as_bool(payload.get("score_claim", False)),
        },
        "score_claim": False,
        "subspaces": {
            "classes": _finalize_table(classes, limit=table_limit),
            "families": _finalize_table(families, limit=table_limit),
            "frames": _finalize_table(frames, limit=table_limit),
            "pairs": _finalize_table(pairs, limit=table_limit),
            "source_to_candidate": _finalize_table(source_to_candidate, limit=table_limit),
        },
        "totals": {
            "estimated_charged_bytes_sum": round(total_bytes, 12),
            "estimated_lagrangian_net_proxy_sum": round(total_net, 12),
            "estimated_marginal_score_saved_proxy_sum": round(total_score_saved, 12),
            "estimated_score_saved_per_charged_byte": 0.0
            if total_bytes <= 0
            else round(total_score_saved / total_bytes, 12),
        },
        "top_atom_ids": [str(feature["atom_id"]) for feature in features],
        "top_pair_indices": [
            int(row["key"]) for row in _finalize_table(pairs, limit=table_limit) if row["key"].isdigit()
        ],
    }


def _overlap_rows(profiles: list[Mapping[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(profiles):
        for right_index in range(left_index + 1, len(profiles)):
            right = profiles[right_index]
            left_ids = set(left.get("top_atom_ids", [])[:top_k])
            right_ids = set(right.get("top_atom_ids", [])[:top_k])
            left_pairs = set(left.get("top_pair_indices", [])[:top_k])
            right_pairs = set(right.get("top_pair_indices", [])[:top_k])
            atom_union = left_ids | right_ids
            pair_union = left_pairs | right_pairs
            rows.append(
                {
                    "atom_id_overlap": len(left_ids & right_ids),
                    "atom_id_union": len(atom_union),
                    "atom_id_jaccard": 1.0 if not atom_union else round(len(left_ids & right_ids) / len(atom_union), 12),
                    "left_path": left["input"]["path"],
                    "pair_overlap": len(left_pairs & right_pairs),
                    "pair_union": len(pair_union),
                    "pair_jaccard": 1.0
                    if not pair_union
                    else round(len(left_pairs & right_pairs) / len(pair_union), 12),
                    "right_path": right["input"]["path"],
                }
            )
    rows.sort(key=lambda item: (item["left_path"], item["right_path"]))
    return rows


def _accumulate_profile_row(
    table: dict[str, dict[str, Any]],
    *,
    key: str,
    row: Mapping[str, Any],
    profile_path: str,
) -> None:
    entry = table.setdefault(
        key,
        {
            "estimated_charged_bytes_sum": 0.0,
            "estimated_lagrangian_net_proxy_sum": 0.0,
            "estimated_marginal_score_saved_proxy_sum": 0.0,
            "hit_count": 0,
            "profile_paths": set(),
            "residual_pixels_sum": 0,
            "weighted_centroid_x_sum": 0.0,
            "weighted_centroid_y_sum": 0.0,
            "weight_sum_for_centroid": 0.0,
        },
    )
    score_saved = _as_float(row.get("estimated_marginal_score_saved_proxy_sum"))
    entry["estimated_charged_bytes_sum"] += _as_float(row.get("estimated_charged_bytes_sum"))
    entry["estimated_lagrangian_net_proxy_sum"] += _as_float(
        row.get("estimated_lagrangian_net_proxy_sum")
    )
    entry["estimated_marginal_score_saved_proxy_sum"] += score_saved
    entry["hit_count"] += _as_int(row.get("hit_count"))
    entry["profile_paths"].add(profile_path)
    entry["residual_pixels_sum"] += _as_int(row.get("residual_pixels_sum"))

    centroid = row.get("weighted_centroid")
    if isinstance(centroid, Mapping):
        weight = max(score_saved, 0.0)
        entry["weighted_centroid_x_sum"] += _as_float(centroid.get("x")) * weight
        entry["weighted_centroid_y_sum"] += _as_float(centroid.get("y")) * weight
        entry["weight_sum_for_centroid"] += weight


def _finalize_aggregate_table(
    table: Mapping[str, Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, entry in table.items():
        byte_sum = _as_float(entry.get("estimated_charged_bytes_sum"))
        score_sum = _as_float(entry.get("estimated_marginal_score_saved_proxy_sum"))
        profile_paths = sorted(str(path) for path in entry.get("profile_paths", set()))
        weight_sum = _as_float(entry.get("weight_sum_for_centroid"))
        rows.append(
            {
                "estimated_charged_bytes_sum": round(byte_sum, 12),
                "estimated_lagrangian_net_proxy_sum": round(
                    _as_float(entry.get("estimated_lagrangian_net_proxy_sum")), 12
                ),
                "estimated_marginal_score_saved_proxy_sum": round(score_sum, 12),
                "estimated_score_saved_per_charged_byte": 0.0
                if byte_sum <= 0
                else round(score_sum / byte_sum, 12),
                "hit_count": _as_int(entry.get("hit_count")),
                "key": key,
                "profile_hit_count": len(profile_paths),
                "profile_paths": profile_paths,
                "residual_pixels_sum": _as_int(entry.get("residual_pixels_sum")),
                "weighted_centroid": None
                if weight_sum <= 0
                else {
                    "x": round(_as_float(entry.get("weighted_centroid_x_sum")) / weight_sum, 6),
                    "y": round(_as_float(entry.get("weighted_centroid_y_sum")) / weight_sum, 6),
                },
            }
        )

    rows.sort(
        key=lambda item: (
            -int(item["profile_hit_count"]),
            -float(item["estimated_lagrangian_net_proxy_sum"]),
            -float(item["estimated_marginal_score_saved_proxy_sum"]),
            str(item["key"]),
        )
    )
    return rows[:limit]


def _aggregate_subspaces(
    profiles: list[Mapping[str, Any]],
    *,
    table_limit: int,
) -> dict[str, Any]:
    tables = {
        "classes": {},
        "families": {},
        "frames": {},
        "pairs": {},
        "source_to_candidate": {},
    }
    for profile in profiles:
        profile_path = str(profile["input"]["path"])
        subspaces = profile.get("subspaces")
        if not isinstance(subspaces, Mapping):
            continue
        for section, table in tables.items():
            rows = subspaces.get(section)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping) or "key" not in row:
                    continue
                _accumulate_profile_row(
                    table,
                    key=str(row["key"]),
                    row=row,
                    profile_path=profile_path,
                )
    return {
        section: _finalize_aggregate_table(table, limit=table_limit)
        for section, table in tables.items()
    }


def _overlap_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "comparison_count": 0,
            "mean_atom_id_jaccard": None,
            "mean_pair_jaccard": None,
            "min_pair_jaccard": None,
        }
    atom_values = [_as_float(row.get("atom_id_jaccard")) for row in rows]
    pair_values = [_as_float(row.get("pair_jaccard")) for row in rows]
    return {
        "comparison_count": len(rows),
        "mean_atom_id_jaccard": round(sum(atom_values) / len(atom_values), 12),
        "mean_pair_jaccard": round(sum(pair_values) / len(pair_values), 12),
        "min_pair_jaccard": round(min(pair_values), 12),
    }


def _signal_surface(
    *,
    aggregate_subspaces: Mapping[str, Any],
    overlaps: list[Mapping[str, Any]],
) -> dict[str, Any]:
    pairs = list(aggregate_subspaces.get("pairs", []))
    classes = list(aggregate_subspaces.get("classes", []))
    confusions = list(aggregate_subspaces.get("source_to_candidate", []))
    return {
        "action_rules": [
            "Use consensus subspaces to propose charged repair, multimask, or learned-selector policies.",
            "Use byte-saving exact negatives as cliff maps; do not promote or rank from profiler output.",
            "Dispatch only after a deterministic archive builder charges the selected atoms and exact CUDA auth eval closes custody.",
        ],
        "canonical_score_source_required": "archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
        "low_dimensional_consensus": {
            "top_class_confusions": [str(row["key"]) for row in confusions[:8]],
            "top_classes": [str(row["key"]) for row in classes[:8]],
            "top_pairs": [int(row["key"]) for row in pairs[:16] if str(row["key"]).isdigit()],
        },
        "overlap_summary": _overlap_summary(overlaps),
        "score_claim": False,
    }


def build_profile(
    *,
    ledger_paths: Iterable[Path],
    output_json: Path,
    top_k: int = 256,
    table_limit: int = 32,
    overlap_top_k: int = 64,
    allow_score_claim_inputs: bool = False,
) -> dict[str, Any]:
    paths = [Path(path) for path in ledger_paths]
    if not paths:
        raise ValueError("at least one --ledger is required")
    profiles = [
        _profile_one(path, _load_json(path), top_k=top_k, table_limit=table_limit)
        for path in paths
    ]

    score_claim_inputs = [profile["input"]["path"] for profile in profiles if profile["input"]["score_claim"]]
    if score_claim_inputs and not allow_score_claim_inputs:
        joined = ", ".join(score_claim_inputs)
        raise ValueError(
            "score-claim input ledgers are rejected by default; this profiler "
            f"is planning-only. Offending inputs: {joined}"
        )

    overlaps = _overlap_rows(profiles, top_k=overlap_top_k)
    aggregate_subspaces = _aggregate_subspaces(profiles, table_limit=table_limit)
    payload = {
        "aggregate_subspaces": aggregate_subspaces,
        "canonical_score_source_required": "archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
        "evidence_grade": "planning_only",
        "fridrich_yousfi_signal_surface": _signal_surface(
            aggregate_subspaces=aggregate_subspaces,
            overlaps=overlaps,
        ),
        "input_count": len(profiles),
        "no_score_claim": True,
        "overlap_top_k": overlap_top_k,
        "pairwise_overlaps": overlaps,
        "profiles": profiles,
        "promotion_eligible": False,
        "schema": "atom_ledger_active_subspace_profile_v1",
        "score_claim": False,
        "score_claim_inputs_detected": score_claim_inputs,
        "table_limit": table_limit,
        "top_k": top_k,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(_json_dumps(payload), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", action="append", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--table-limit", type=int, default=32)
    parser.add_argument("--overlap-top-k", type=int, default=64)
    parser.add_argument(
        "--allow-score-claim-inputs",
        action="store_true",
        help="debug-only escape hatch; output remains non-promotable planning evidence",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_profile(
        ledger_paths=args.ledger,
        output_json=args.output_json,
        top_k=args.top_k,
        table_limit=args.table_limit,
        overlap_top_k=args.overlap_top_k,
        allow_score_claim_inputs=args.allow_score_claim_inputs,
    )
    print(
        _json_dumps(
            {
                "evidence_grade": payload["evidence_grade"],
                "input_count": payload["input_count"],
                "output_json": str(args.output_json),
                "schema": payload["schema"],
                "score_claim": payload["score_claim"],
            }
        ),
        end="",
    )


if __name__ == "__main__":
    main()
