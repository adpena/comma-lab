#!/usr/bin/env python3
"""Build PR85 randmulti group-policy archive candidates.

This turns the planning-only output from
``plan_pr85_randmulti_group_waterfill.py`` into byte-closed PR85 single-member
``x`` archives. It only mutates the charged ``randmulti`` segment: selected
groups are preserved from the source archive, unselected groups are encoded as
zero rows, then the PR85 v5 bundle is re-packed deterministically.

The emitted archives are exact-eval candidates, not score claims. Remote
dispatch still requires an active lane claim plus the usual PR85 runtime gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Sequence

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "experiments" / "plan_pr85_randmulti_group_waterfill.py"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_POLICY_JSON = (
    REPO_ROOT
    / "experiments/results/pr85_randmulti_group_waterfill_20260504_codex/candidate_policies.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_randmulti_group_policy_candidates_20260504_codex"
TOOL = "experiments/build_pr85_randmulti_group_policy_candidates.py"
SCHEMA = "pr85_randmulti_group_policy_archive_candidates_v1"
MANIFEST_SCHEMA = "pr85_randmulti_group_policy_archive_candidate_v1"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489


def _load_plan_module() -> Any:
    spec = importlib.util.spec_from_file_location("pr85_randmulti_policy_builder_plan", PLAN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load planner helper from {PLAN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan = _load_plan_module()
recode = plan.recode


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _zip_info(name: str = "x") -> zipfile.ZipInfo:
    recode._safe_zip_member(name)
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info("x"), payload)


def _archive_info(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"expected one archive member, found {len(infos)}")
        info = infos[0]
        raw = zf.read(info)
    return {
        "archive_path": _rel(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": _sha256_file(path),
        "member_name": info.filename,
        "member_bytes": int(len(raw)),
        "member_sha256": _sha256(raw),
        "zip_stored": info.compress_type == zipfile.ZIP_STORED,
    }


def _read_policy_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "policies" not in payload:
        raise ValueError(f"{path} is not a randmulti policy JSON")
    return payload


def _selected_policy_rows(
    payload: dict[str, Any],
    policy_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    policies = payload.get("policies")
    if not isinstance(policies, list):
        raise ValueError("policy JSON field 'policies' must be a list")
    by_id = {}
    for row in policies:
        if not isinstance(row, dict) or not isinstance(row.get("candidate_policy_id"), str):
            raise ValueError("every policy row must contain candidate_policy_id")
        by_id[row["candidate_policy_id"]] = row
    selected_ids = list(policy_ids or [
        "waterfill_top001",
        "waterfill_top002",
        "waterfill_top004",
        "waterfill_top008",
        "waterfill_top016",
        "waterfill_best_prefix_by_net",
        "isolated_net_positive_groups",
    ])
    missing = [policy_id for policy_id in selected_ids if policy_id not in by_id]
    if missing:
        raise ValueError(f"unknown policy id(s): {missing}")
    return [by_id[policy_id] for policy_id in selected_ids]


def _group_semantics_report(
    *,
    source_groups: Sequence[Any],
    candidate_decoded_raw: bytes,
    selected_group_ids: set[int],
) -> dict[str, Any]:
    candidate_groups = plan._decode_pr85_randmulti_groups(candidate_decoded_raw)
    if len(candidate_groups) != len(source_groups):
        return {
            "status": "failed",
            "reason": "group_count_mismatch",
            "source_group_count": len(source_groups),
            "candidate_group_count": len(candidate_groups),
        }
    mismatches: list[dict[str, Any]] = []
    zero_group_ids: list[int] = []
    preserved_group_ids: list[int] = []
    group_profiles: list[dict[str, Any]] = []
    for source, candidate in zip(source_groups, candidate_groups, strict=True):
        source_rows = tuple(source.rows)
        candidate_rows = tuple(candidate.rows)
        source_nonzero = sum(sum(1 for value in row if value) for row in source_rows)
        candidate_nonzero = sum(sum(1 for value in row if value) for row in candidate_rows)
        source_value_sum = sum(sum(int(value) for value in row) for row in source_rows)
        candidate_value_sum = sum(sum(int(value) for value in row) for row in candidate_rows)
        candidate_raw_payload_bytes = sum(plan._encode_sparse_row(row).__len__() for row in candidate_rows)
        group_profiles.append(
            {
                "amplitude": int(source.amplitude),
                "candidate_nonzero_choice_count": int(candidate_nonzero),
                "candidate_raw_payload_bytes": int(candidate_raw_payload_bytes),
                "candidate_value_sum": int(candidate_value_sum),
                "group_index": int(source.group_index),
                "height": int(source.height),
                "scount": int(source.scount),
                "selected": int(source.group_index) in selected_group_ids,
                "source_nonzero_choice_count": int(source_nonzero),
                "source_raw_payload_bytes": int(source.raw_payload_bytes),
                "source_value_sum": int(source_value_sum),
                "width": int(source.width),
            }
        )
        if int(source.group_index) in selected_group_ids:
            if candidate_rows == source_rows:
                preserved_group_ids.append(int(source.group_index))
            else:
                mismatches.append({"group_index": int(source.group_index), "expected": "source"})
        else:
            zero_rows = tuple(bytes(plan.PAIR_COUNT) for _ in range(int(source.scount)))
            if candidate_rows == zero_rows:
                zero_group_ids.append(int(source.group_index))
            else:
                mismatches.append({"group_index": int(source.group_index), "expected": "zero"})
    return {
        "status": "passed" if not mismatches else "failed",
        "selected_group_count": len(selected_group_ids),
        "preserved_group_ids": preserved_group_ids,
        "zero_group_ids": zero_group_ids,
        "zero_group_count": len(zero_group_ids),
        "group_profiles": group_profiles,
        "source_nonzero_choice_total": int(
            sum(row["source_nonzero_choice_count"] for row in group_profiles)
        ),
        "candidate_nonzero_choice_total": int(
            sum(row["candidate_nonzero_choice_count"] for row in group_profiles)
        ),
        "source_raw_payload_bytes_total": int(
            sum(row["source_raw_payload_bytes"] for row in group_profiles)
        ),
        "candidate_raw_payload_bytes_total": int(
            sum(row["candidate_raw_payload_bytes"] for row in group_profiles)
        ),
        "mismatched_groups": mismatches,
    }


def _build_one(
    *,
    policy: dict[str, Any],
    source_archive: dict[str, Any],
    source_bundle: dict[str, Any],
    source_segments: dict[str, bytes],
    source_groups: Sequence[Any],
    out_dir: Path,
    policy_payload_path: Path,
) -> dict[str, Any]:
    policy_id = str(policy["candidate_policy_id"])
    selected_group_ids = {int(value) for value in policy.get("selected_group_ids", [])}
    if len(selected_group_ids) != len(policy.get("selected_group_ids", [])):
        raise ValueError(f"policy {policy_id} contains duplicate selected_group_ids")
    invalid = sorted(group_id for group_id in selected_group_ids if group_id < 0 or group_id >= len(source_groups))
    if invalid:
        raise ValueError(f"policy {policy_id} selected group ids outside source schedule: {invalid}")

    encoded_raw = plan._encode_selected_headerless_raw(source_groups, selected_group_ids)
    compressed_randmulti, brotli_params = recode._brotli_best(encoded_raw)
    decoded_check = brotli.decompress(compressed_randmulti)
    if decoded_check != encoded_raw:
        raise ValueError(f"policy {policy_id} randmulti Brotli roundtrip failed")

    candidate_segments = dict(source_segments)
    candidate_segments["randmulti"] = compressed_randmulti
    payload = recode._pack_bundle(candidate_segments, header_mode="v5")
    validation = recode._validate_candidate_bundle(payload, candidate_segments)
    if validation["status"] != "passed":
        raise ValueError(f"policy {policy_id} failed PR85 bundle validation")

    semantic_report = _group_semantics_report(
        source_groups=source_groups,
        candidate_decoded_raw=decoded_check,
        selected_group_ids=selected_group_ids,
    )
    if semantic_report["status"] != "passed":
        raise ValueError(f"policy {policy_id} failed group semantics validation")

    candidate_dir = out_dir / policy_id
    archive_path = candidate_dir / "archive.zip"
    _write_archive(archive_path, payload)
    candidate = _archive_info(archive_path)
    byte_delta = int(candidate["archive_bytes"] - source_archive["bytes"])
    dispatchable = byte_delta < 0 and bool(selected_group_ids)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tool": TOOL,
        "policy_id": policy_id,
        "score_claim": False,
        "dispatch_performed": False,
        "evidence_grade": "empirical_archive_build_from_exact_negative_policy",
        "source_archive": source_archive,
        "source_bundle": {
            "format": source_bundle["format"],
            "header_bytes": int(source_bundle["header_bytes"]),
            "segment_lengths": source_bundle["segment_lengths"],
            "fixed_length_segments": source_bundle.get("fixed_length_segments", {}),
        },
        "candidate": candidate,
        "candidate_bundle_validation": validation,
        "source_policy": {
            "policy_json": _rel(policy_payload_path),
            "candidate_policy_id": policy_id,
            "policy_hash": policy.get("policy_hash"),
            "selected_group_ids": sorted(selected_group_ids),
            "estimated_component_score_rescue": policy.get("estimated_component_score_rescue"),
            "estimated_net_score_rescue_after_rate": policy.get("estimated_net_score_rescue_after_rate"),
            "planning_only": policy.get("planning_only", True),
        },
        "randmulti_transform": {
            "source_segment_bytes": int(len(source_segments["randmulti"])),
            "source_segment_sha256": _sha256(source_segments["randmulti"]),
            "candidate_segment_bytes": int(len(compressed_randmulti)),
            "candidate_segment_sha256": _sha256(compressed_randmulti),
            "segment_byte_delta": int(len(compressed_randmulti) - len(source_segments["randmulti"])),
            "candidate_decoded_bytes": int(len(decoded_check)),
            "candidate_decoded_sha256": _sha256(decoded_check),
            "brotli_params": brotli_params,
            "group_semantics": semantic_report,
        },
        "byte_delta_vs_source_archive": byte_delta,
        "formula_only_rate_score_delta_vs_source": byte_delta * RATE_SCORE_PER_BYTE,
        "dispatch_gate": (
            "eligible_for_cuda_auth_eval_after_lane_claim"
            if dispatchable
            else "planning_only/no_remote_dispatch"
        ),
        "next_gate": (
            "Before exact eval, claim the lane with tools/claim_lane_dispatch.py and run PR85 public-runtime CUDA auth eval on this exact archive."
        ),
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def build_candidates(
    *,
    archive: Path = DEFAULT_ARCHIVE,
    policy_json: Path = DEFAULT_POLICY_JSON,
    out_dir: Path = DEFAULT_OUT_DIR,
    policy_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    policy_payload = _read_policy_payload(policy_json)
    source_archive, raw = recode._read_pr85_archive(archive)
    source_bundle, source_segments = recode._parse_bundle(raw)
    decoded_randmulti = brotli.decompress(source_segments["randmulti"])
    source_groups = plan._decode_pr85_randmulti_groups(decoded_randmulti)
    rows = [
        _build_one(
            policy=policy,
            source_archive=source_archive,
            source_bundle=source_bundle,
            source_segments=source_segments,
            source_groups=source_groups,
            out_dir=out_dir,
            policy_payload_path=policy_json,
        )
        for policy in _selected_policy_rows(policy_payload, policy_ids)
    ]
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "evidence_grade": "empirical_archive_build_from_exact_negative_policy",
        "source_archive": source_archive,
        "policy_json": _rel(policy_json),
        "candidate_count": len(rows),
        "dispatchable_candidate_count": sum(
            1 for row in rows if row["dispatch_gate"] == "eligible_for_cuda_auth_eval_after_lane_claim"
        ),
        "candidates": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "candidate_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--policy-json", type=Path, default=DEFAULT_POLICY_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy", action="append", dest="policies")
    args = parser.parse_args(argv)

    payload = build_candidates(
        archive=args.archive,
        policy_json=args.policy_json,
        out_dir=args.out_dir,
        policy_ids=args.policies,
    )
    print(_json_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
