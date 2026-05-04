#!/usr/bin/env python3
"""Build PR85 post/motion group-policy archive candidates.

This is a deterministic byte-closed builder for PR85 partial side-channel
atoms. It leaves the mask/model/pose/bias/region/randmulti streams untouched,
then preserves selected post/motion groups while neutralizing unselected
post/motion groups to the public PR85 runtime defaults.

The emitted archives are exact-eval candidates, not score claims. Promotion
requires CUDA auth eval on the exact archive/runtime pair after a lane claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
RECODE_PATH = REPO_ROOT / "experiments" / "build_pr85_sidechannel_recode_candidates.py"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_OUT_DIR = (
    REPO_ROOT / "experiments/results/pr85_post_motion_group_policy_candidates_20260504_codex"
)
TOOL = "experiments/build_pr85_post_motion_group_policy_candidates.py"
SCHEMA = "pr85_post_motion_group_policy_archive_candidates_v1"
MANIFEST_SCHEMA = "pr85_post_motion_group_policy_archive_candidate_v1"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PAIR_COUNT = 600
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
PR85_FRONTIER_SCORE = 0.25806611029397786
PR85_FRONTIER_BYTES = 236_328
PR85_FRONTIER_SHA256 = "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e"

POST_GROUPS = ("post_stage1", "post_stage2", "post_stage3", "post_stage4")
MOTION_GROUPS = ("motion_shift", "motion_frac", "motion_frac2", "motion_frac3")
GROUPS = POST_GROUPS + MOTION_GROUPS

POLICIES: dict[str, tuple[str, ...]] = {
    "preserve_post123_motion": (
        "post_stage1",
        "post_stage2",
        "post_stage3",
        "motion_shift",
        "motion_frac",
        "motion_frac2",
        "motion_frac3",
    ),
    "preserve_post23_motion": (
        "post_stage2",
        "post_stage3",
        "motion_shift",
        "motion_frac",
        "motion_frac2",
        "motion_frac3",
    ),
    "preserve_post_all_shift": (
        "post_stage1",
        "post_stage2",
        "post_stage3",
        "post_stage4",
        "motion_shift",
    ),
    "preserve_post_all_shift_frac2_frac3": (
        "post_stage1",
        "post_stage2",
        "post_stage3",
        "post_stage4",
        "motion_shift",
        "motion_frac2",
        "motion_frac3",
    ),
    "preserve_motion_only": MOTION_GROUPS,
}

WHOLE_STREAM_NEGATIVES = {
    "minus_post": {
        "score": 0.3093560838497741,
        "role": "exact T4 negative; post stream carries high-value signal",
    },
    "minus_motion_stack": {
        "score": 0.363610914426998,
        "role": "exact T4 negative; motion streams carry high-value signal",
    },
    "minus_randmulti": {
        "score": 0.282674,
        "role": "exact negative context only; this builder does not mutate randmulti",
    },
}


@dataclass(frozen=True)
class SegmentTransform:
    segment: str
    source_segment: bytes
    candidate_segment: bytes
    selected_groups: tuple[str, ...]
    neutralized_groups: tuple[str, ...]
    semantic_report: dict[str, Any]
    brotli_params: dict[str, int | str] | None


def _load_recode_module() -> Any:
    spec = importlib.util.spec_from_file_location("pr85_post_motion_group_recode", RECODE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load PR85 recode helper from {RECODE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


recode = _load_recode_module()


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


def _validate_selected_groups(selected_groups: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(group) for group in selected_groups)
    duplicates = sorted({group for group in selected if selected.count(group) > 1})
    if duplicates:
        raise ValueError(f"duplicate selected group id(s): {duplicates}")
    unknown = sorted(group for group in selected if group not in GROUPS)
    if unknown:
        raise ValueError(f"unknown selected group id(s): {unknown}")
    return tuple(sorted(selected, key=GROUPS.index))


def _policy_rows(policy_ids: Sequence[str] | None) -> list[tuple[str, tuple[str, ...]]]:
    selected = list(policy_ids or POLICIES)
    missing = [policy_id for policy_id in selected if policy_id not in POLICIES]
    if missing:
        raise ValueError(f"unknown policy id(s): {missing}")
    return [(policy_id, _validate_selected_groups(POLICIES[policy_id])) for policy_id in selected]


def _post_stage_records(semantic: bytes) -> list[tuple[int, bytes]]:
    pos = 0
    stages: list[tuple[int, bytes]] = []
    while pos < len(semantic):
        if pos + 3 > len(semantic):
            raise ValueError("truncated post semantic stage record")
        stage_id = semantic[pos]
        pos += 1
        count = int.from_bytes(semantic[pos : pos + 2], "little")
        pos += 2
        choices = semantic[pos : pos + count]
        pos += count
        if len(choices) != count:
            raise ValueError("truncated post semantic choices")
        if count != PAIR_COUNT:
            raise ValueError(f"post stage {stage_id} has {count} choices, expected {PAIR_COUNT}")
        stages.append((stage_id, bytes(choices)))
    if pos != len(semantic):
        raise ValueError("post semantic parser desynchronized")
    expected = list(range(1, len(stages) + 1))
    observed = [stage_id for stage_id, _choices in stages]
    if observed != expected:
        raise ValueError(f"post stages are not ordered for headerless PR85: {observed}")
    return stages


def _post_semantic_from_stages(stages: Sequence[tuple[int, bytes]]) -> bytes:
    out = bytearray()
    for stage_id, choices in stages:
        out.append(stage_id)
        out += len(choices).to_bytes(2, "little")
        out += choices
    return bytes(out)


def _post_transform(source_segment: bytes, selected_groups: set[str]) -> SegmentTransform:
    decoded = recode._decode_segment_raw("post", source_segment)
    source_semantic = recode._segment_semantics("post", decoded)
    stages = _post_stage_records(source_semantic)
    selected = tuple(f"post_stage{stage_id}" for stage_id, _ in stages if f"post_stage{stage_id}" in selected_groups)
    neutralized = tuple(
        f"post_stage{stage_id}" for stage_id, _ in stages if f"post_stage{stage_id}" not in selected_groups
    )
    candidate_stages = [
        (stage_id, choices if f"post_stage{stage_id}" in selected_groups else bytes(len(choices)))
        for stage_id, choices in stages
    ]
    candidate_semantic = _post_semantic_from_stages(candidate_stages)
    if not neutralized:
        candidate_segment = source_segment
        params = None
    else:
        candidate_raw = recode._post_raw_from_semantics(candidate_semantic, variant="headerless")
        candidate_segment, params = recode._brotli_best(candidate_raw)
    decoded_check = recode._decode_segment_raw("post", candidate_segment)
    check_stages = _post_stage_records(recode._segment_semantics("post", decoded_check))
    mismatches = []
    profiles = []
    for (stage_id, source_choices), (_check_id, candidate_choices) in zip(stages, check_stages, strict=True):
        group_id = f"post_stage{stage_id}"
        source_nonzero = sum(1 for value in source_choices if value)
        candidate_nonzero = sum(1 for value in candidate_choices if value)
        profiles.append(
            {
                "group_id": group_id,
                "selected": group_id in selected_groups,
                "source_nonzero_choice_count": int(source_nonzero),
                "candidate_nonzero_choice_count": int(candidate_nonzero),
                "source_semantic_sha256": _sha256(source_choices),
                "candidate_semantic_sha256": _sha256(candidate_choices),
            }
        )
        if group_id in selected_groups and candidate_choices != source_choices:
            mismatches.append({"group_id": group_id, "expected": "source_choices"})
        if group_id not in selected_groups and candidate_choices != bytes(PAIR_COUNT):
            mismatches.append({"group_id": group_id, "expected": "zero_choices"})
    return SegmentTransform(
        segment="post",
        source_segment=source_segment,
        candidate_segment=candidate_segment,
        selected_groups=selected,
        neutralized_groups=neutralized,
        semantic_report={
            "status": "passed" if not mismatches else "failed",
            "stage_count": int(len(stages)),
            "group_profiles": profiles,
            "mismatched_groups": mismatches,
        },
        brotli_params=params,
    )


def _motion_raw_from_values(name: str, values: bytes) -> bytes:
    if name == "shift":
        return recode._encode_delta_choice(b"SD4", values, default_choice=40)
    if name == "frac":
        return recode._encode_sparse_choice(b"FV1", values, default_choice=4)
    if name == "frac2":
        return b"FH2" + values
    if name == "frac3":
        return recode._encode_delta_choice(b"FD3", values, default_choice=4)
    raise ValueError(f"unsupported motion segment {name!r}")


def _motion_default(name: str) -> int:
    if name == "shift":
        return 40
    if name in {"frac", "frac2", "frac3"}:
        return 4
    raise ValueError(f"unsupported motion segment {name!r}")


def _motion_transform(name: str, source_segment: bytes, selected_groups: set[str]) -> SegmentTransform:
    group_id = f"motion_{name}"
    decoded = recode._decode_segment_raw(name, source_segment)
    source_values = recode._decode_choice_semantics(name, decoded)
    if len(source_values) != PAIR_COUNT:
        raise ValueError(f"{name} decoded {len(source_values)} choices, expected {PAIR_COUNT}")
    selected = group_id in selected_groups
    candidate_values = source_values if selected else bytes([_motion_default(name)]) * PAIR_COUNT
    if selected:
        candidate_segment = source_segment
        params = None
    else:
        candidate_raw = _motion_raw_from_values(name, candidate_values)
        candidate_segment, params = recode._brotli_best(candidate_raw)
    decoded_check = recode._decode_segment_raw(name, candidate_segment)
    check_values = recode._decode_choice_semantics(name, decoded_check)
    expected_values = source_values if selected else bytes([_motion_default(name)]) * PAIR_COUNT
    mismatches = [] if check_values == expected_values else [{"group_id": group_id, "expected": "source" if selected else "neutral"}]
    source_default = _motion_default(name)
    return SegmentTransform(
        segment=name,
        source_segment=source_segment,
        candidate_segment=candidate_segment,
        selected_groups=(group_id,) if selected else (),
        neutralized_groups=() if selected else (group_id,),
        semantic_report={
            "status": "passed" if not mismatches else "failed",
            "group_profiles": [
                {
                    "group_id": group_id,
                    "selected": selected,
                    "neutral_default_choice": int(source_default),
                    "source_nondefault_choice_count": int(
                        sum(1 for value in source_values if value != source_default)
                    ),
                    "candidate_nondefault_choice_count": int(
                        sum(1 for value in check_values if value != source_default)
                    ),
                    "source_semantic_sha256": _sha256(source_values),
                    "candidate_semantic_sha256": _sha256(check_values),
                }
            ],
            "mismatched_groups": mismatches,
        },
        brotli_params=params,
    )


def _transform_record(transform: SegmentTransform) -> dict[str, Any]:
    source_semantic = recode._segment_semantics(
        transform.segment,
        recode._decode_segment_raw(transform.segment, transform.source_segment),
    )
    candidate_semantic = recode._segment_semantics(
        transform.segment,
        recode._decode_segment_raw(transform.segment, transform.candidate_segment),
    )
    return {
        "segment": transform.segment,
        "source_segment_bytes": int(len(transform.source_segment)),
        "source_segment_sha256": _sha256(transform.source_segment),
        "candidate_segment_bytes": int(len(transform.candidate_segment)),
        "candidate_segment_sha256": _sha256(transform.candidate_segment),
        "segment_byte_delta": int(len(transform.candidate_segment) - len(transform.source_segment)),
        "source_semantic_sha256": _sha256(source_semantic),
        "candidate_semantic_sha256": _sha256(candidate_semantic),
        "selected_groups": list(transform.selected_groups),
        "neutralized_groups": list(transform.neutralized_groups),
        "noop_segment": transform.source_segment == transform.candidate_segment,
        "brotli_params": transform.brotli_params,
        "semantic_report": transform.semantic_report,
    }


def _build_one(
    *,
    policy_id: str,
    selected_groups: tuple[str, ...],
    source_archive: dict[str, Any],
    source_bundle: dict[str, Any],
    source_segments: dict[str, bytes],
    out_dir: Path,
) -> dict[str, Any]:
    selected_set = set(selected_groups)
    transforms = [
        _post_transform(source_segments["post"], selected_set),
        _motion_transform("shift", source_segments["shift"], selected_set),
        _motion_transform("frac", source_segments["frac"], selected_set),
        _motion_transform("frac2", source_segments["frac2"], selected_set),
        _motion_transform("frac3", source_segments["frac3"], selected_set),
    ]
    for transform in transforms:
        if transform.semantic_report["status"] != "passed":
            raise ValueError(f"policy {policy_id} failed semantic validation for {transform.segment}")

    candidate_segments = dict(source_segments)
    for transform in transforms:
        candidate_segments[transform.segment] = transform.candidate_segment
    header_mode = recode._header_mode_for_segments(source_segments, candidate_segments)
    payload = recode._pack_bundle(candidate_segments, header_mode=header_mode)
    validation = recode._validate_candidate_bundle(payload, candidate_segments)
    if validation["status"] != "passed":
        raise ValueError(f"policy {policy_id} failed PR85 bundle parse validation")

    candidate_dir = out_dir / policy_id
    archive_path = candidate_dir / "archive.zip"
    _write_archive(archive_path, payload)
    candidate = _archive_info(archive_path)
    transform_records = [_transform_record(transform) for transform in transforms]
    changed = [row["segment"] for row in transform_records if not row["noop_segment"]]
    neutralized_groups = sorted(
        (group for group in GROUPS if group not in selected_set),
        key=GROUPS.index,
    )
    byte_delta = int(candidate["archive_bytes"] - source_archive["bytes"])
    eval_ready = bool(changed) and validation["status"] == "passed" and candidate["member_name"] == "x"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tool": TOOL,
        "policy_id": policy_id,
        "score_claim": False,
        "dispatch_performed": False,
        "promotion_status": "non_promotable_pending_exact_cuda_eval",
        "evidence_grade": "empirical_partial_sidechannel_atom_candidate",
        "source_frontier": {
            "score": PR85_FRONTIER_SCORE,
            "archive_bytes": PR85_FRONTIER_BYTES,
            "archive_sha256": PR85_FRONTIER_SHA256,
            "score_source": "exact T4 PR85 replay; comparison context only",
        },
        "whole_stream_negative_context": WHOLE_STREAM_NEGATIVES,
        "source_archive": source_archive,
        "source_member": {
            "member_name": source_archive["member_name"],
            "member_bytes": int(source_archive["member_file_size"]),
            "member_sha256": source_archive["member_sha256"],
        },
        "source_bundle": {
            "format": source_bundle["format"],
            "header_bytes": int(source_bundle["header_bytes"]),
            "segment_lengths": source_bundle["segment_lengths"],
            "fixed_length_segments": source_bundle.get("fixed_length_segments", {}),
        },
        "candidate": candidate,
        "candidate_bundle_validation": validation,
        "header_mode": header_mode,
        "selected_groups": list(selected_groups),
        "neutralized_groups": neutralized_groups,
        "preserved_non_target_segments": ["mask", "model", "pose", "bias", "region", "randmulti"],
        "transforms": transform_records,
        "changed_segments": changed,
        "byte_delta_vs_source_archive": byte_delta,
        "formula_only_rate_score_delta_vs_source": byte_delta * RATE_SCORE_PER_BYTE,
        "eval_ready": eval_ready,
        "dispatch_gate": (
            "eligible_for_cuda_auth_eval_after_lane_claim"
            if eval_ready
            else "planning_only/no_remote_dispatch"
        ),
        "next_gate": (
            "Before exact eval, claim the lane with tools/claim_lane_dispatch.py and run PR85 public-runtime CUDA auth eval on this exact archive."
        ),
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def build_candidates(
    archive: Path = DEFAULT_ARCHIVE,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    policy_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    source_archive, raw = recode._read_pr85_archive(archive)
    source_bundle, source_segments = recode._parse_bundle(raw)
    rows = [
        _build_one(
            policy_id=policy_id,
            selected_groups=selected_groups,
            source_archive=source_archive,
            source_bundle=source_bundle,
            source_segments=source_segments,
            out_dir=out_dir,
        )
        for policy_id, selected_groups in _policy_rows(policy_ids)
    ]
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "promotion_status": "non_promotable_pending_exact_cuda_eval",
        "evidence_grade": "empirical_partial_sidechannel_atom_candidate",
        "source_archive": source_archive,
        "source_member": {
            "member_name": source_archive["member_name"],
            "member_bytes": int(source_archive["member_file_size"]),
            "member_sha256": source_archive["member_sha256"],
        },
        "group_universe": list(GROUPS),
        "policy_count": len(rows),
        "candidate_count": len(rows),
        "eval_ready_candidate_count": sum(1 for row in rows if row["eval_ready"]),
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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy", action="append", dest="policies", choices=sorted(POLICIES))
    args = parser.parse_args(argv)

    payload = build_candidates(args.archive, args.out_dir, policy_ids=args.policies)
    print(_json_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
