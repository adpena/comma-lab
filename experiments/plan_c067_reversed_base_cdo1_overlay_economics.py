#!/usr/bin/env python3
"""Plan reversed-base CDO1 overlay economics for C067 mask compression.

This local-only planner asks whether a smaller decoded-mask base (for example
CMG3 nonzero-row runs) can be repaired back toward the C067 decoded-mask basin
with a charged CDO1 overlay. It emits no archive and no score claim.
"""
from __future__ import annotations

import argparse
import json
import lzma
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tac.repo_io import json_text, read_json, sha256_bytes, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.plan_c067_decoded_delta_overlay_mask_topology import (
    PAIR_INDEX_BASIS_AUTO,
    PAIR_INDEX_BASIS_CHOICES,
    PAIR_INDEX_BASIS_VIDEO_FRAME,
    RUN_STRUCT_NAME,
    _encode_overlay_payload,
    _load_mask_array,
    _mask_tensor_sha256,
    _pair_frame_mask,
    _resolve_pair_index_basis,
    _runs_from_selected,
    _selected_pair_indices_from_frame_indices,
)


TOOL = "experiments/plan_c067_reversed_base_cdo1_overlay_economics.py"
SCHEMA = "c067_reversed_base_cdo1_overlay_economics_v1"
ORIGINAL_VIDEO_BYTES = 37_545_489
C067_FRONTIER_ARCHIVE_BYTES = 276_214
C067_MASK_STREAM_BYTES = 219_472
C067_NON_MASK_PACKED_BYTES = C067_FRONTIER_ARCHIVE_BYTES - C067_MASK_STREAM_BYTES
UNCHANGED_DISTORTION_SUB0300_BYTE_GATE = 252_759
UNCHANGED_DISTORTION_SUB0240_BYTE_GATE = 162_650
DEFAULT_MAX_RESIDUAL_DISAGREEMENT_FRACTION = 0.0010
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/c067_reversed_base_cdo1_overlay_economics_20260502"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "c067_reversed_base_cdo1_overlay_economics.json"
DEFAULT_TARGET_MASK_ARRAY = (
    REPO_ROOT
    / "experiments/results/c063_trace_weighted_mask_grammar_plan_20260502_codex/decoded_mask_array.npy"
)
DEFAULT_TRUST_PLAN_JSON = (
    REPO_ROOT
    / "experiments/results/c067_postdecode_mask_repair_candidate_20260502/"
    "c067_postdecode_mask_repair_waterfill_pair_class_plan.json"
)


@dataclass(frozen=True)
class BaseCandidate:
    label: str
    decoded_mask_array: Path
    payload_path: Path


DEFAULT_BASE_CANDIDATES: tuple[BaseCandidate, ...] = (
    BaseCandidate(
        "cmg3_nonzero_top1",
        REPO_ROOT / "experiments/results/c067_multimask_reconciliation_20260502/cmg3_nonzero_top1.decoded_mask_array.npz",
        REPO_ROOT / "experiments/results/c067_multimask_reconciliation_20260502/cmg3_nonzero_top1.masks.cmg3",
    ),
    BaseCandidate(
        "cmg3_nonzero_top2",
        REPO_ROOT / "experiments/results/c067_multimask_reconciliation_20260502/cmg3_nonzero_top2.decoded_mask_array.npz",
        REPO_ROOT / "experiments/results/c067_multimask_reconciliation_20260502/cmg3_nonzero_top2.masks.cmg3",
    ),
    BaseCandidate(
        "cmg3a_body200",
        REPO_ROOT / "experiments/results/c067_multimask_reconciliation_20260502/cmg3a_body200.decoded_mask_array.npz",
        REPO_ROOT / "experiments/results/c067_multimask_reconciliation_20260502/cmg3a_body200.masks.cmg3",
    ),
    BaseCandidate(
        "cmg3_rowspan_stride1",
        REPO_ROOT / "experiments/results/c067_multimask_reconciliation_20260502/cmg3_rowspan_stride1.decoded_mask_array.npz",
        REPO_ROOT / "experiments/results/c067_multimask_reconciliation_20260502/cmg3_rowspan_stride1.masks.cmg3",
    ),
)


class ReversedBasePlannerError(ValueError):
    """Raised for malformed reversed-base planner inputs."""


def _json_bytes(payload: Any) -> bytes:
    return json_text(payload).encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except (OSError, ValueError):
        return str(path)


def _compressed_payloads(raw_payload: bytes) -> dict[str, dict[str, Any]]:
    zlib_payload = zlib.compress(raw_payload, level=9)
    lzma_payload = lzma.compress(raw_payload, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    return {
        "raw": {"bytes": len(raw_payload), "sha256": sha256_bytes(raw_payload)},
        "zlib": {"bytes": len(zlib_payload), "sha256": sha256_bytes(zlib_payload)},
        "lzma_xz": {"bytes": len(lzma_payload), "sha256": sha256_bytes(lzma_payload)},
    }


def _best_compressor(payloads: dict[str, dict[str, Any]]) -> str:
    return min(payloads, key=lambda key: (int(payloads[key]["bytes"]), key))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except json.JSONDecodeError as exc:
        raise ReversedBasePlannerError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ReversedBasePlannerError(f"{path} must contain a JSON object")
    return payload


def _policy_rows_from_trust_plan(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "policy_id": "full_repair_to_c067_decoded_mask",
            "selection": "all_base_target_differences",
            "pair_indices": [],
            "class_ids": [],
            "source": "synthetic_full_geometry_control",
        }
    ]
    if not path.exists():
        return rows
    payload = _read_json(path)
    policies = payload.get("budget_policies")
    if not isinstance(policies, list):
        raise ReversedBasePlannerError(f"{path} lacks budget_policies")
    for policy in policies:
        if not isinstance(policy, dict):
            continue
        pair_indices: set[int] = set()
        class_ids: set[int] = set()
        selected_atoms: list[dict[str, Any]] = []
        for atom in policy.get("selected_atoms") or []:
            if not isinstance(atom, dict):
                continue
            pairs = [int(value) for value in atom.get("pair_indices") or [] if isinstance(value, int)]
            pair_indices.update(pairs)
            class_id = atom.get("class_id")
            if isinstance(class_id, int):
                class_ids.add(int(class_id))
            selected_atoms.append(
                {
                    "atom_id": atom.get("atom_id"),
                    "pair_indices": pairs,
                    "class_id": int(class_id) if isinstance(class_id, int) else None,
                    "changed_pixels": atom.get("changed_pixels"),
                    "expected_component_score_improvement_first_order": atom.get(
                        "expected_component_score_improvement_first_order"
                    ),
                }
            )
        rows.append(
            {
                "policy_id": str(policy.get("policy_id") or f"budget_{policy.get('budget_payload_bytes')}"),
                "selection": "trust_plan_pair_target_class",
                "budget_payload_bytes": policy.get("budget_payload_bytes"),
                "pair_indices": sorted(pair_indices),
                "class_ids": sorted(class_ids),
                "selected_atoms": selected_atoms,
                "source": _display_path(path, REPO_ROOT),
            }
        )
    return rows


def _longest_diff_run_threshold_selection(
    base: np.ndarray,
    target: np.ndarray,
    *,
    max_residual_disagreement_fraction: float,
) -> np.ndarray:
    """Select whole diff runs until the residual disagreement gate is met.

    This is a deterministic lower-bound proxy for CDO1 repair economics: it
    avoids fractional pixels and chooses the longest changed runs first, which
    minimizes CDO1 header/run overhead before LZMA/Brotli effects are measured.
    """

    diff = base != target
    diff_count = int(diff.sum())
    total = int(diff.size)
    allowed_residual = int(np.floor(max_residual_disagreement_fraction * total))
    required_selected = max(0, diff_count - allowed_residual)
    selected = np.zeros(diff.shape, dtype=bool)
    if required_selected == 0:
        return selected

    run_rows: list[tuple[int, int, int, int, int]] = []
    frame_count, height, _width = diff.shape
    for frame in range(frame_count):
        frame_diff = diff[frame]
        for y in range(height):
            xs = np.flatnonzero(frame_diff[y])
            if xs.size == 0:
                continue
            start = int(xs[0])
            prev = start
            for raw_x in xs[1:]:
                x = int(raw_x)
                if x == prev + 1:
                    prev = x
                    continue
                run_rows.append((prev - start + 1, frame, y, start, prev + 1))
                start = prev = x
            run_rows.append((prev - start + 1, frame, y, start, prev + 1))
    run_rows.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    selected_count = 0
    for length, frame, y, x0, x1 in run_rows:
        selected[frame, y, x0:x1] = True
        selected_count += int(length)
        if selected_count >= required_selected:
            break
    return selected


def _selected_pixels_for_policy(
    base: np.ndarray,
    target: np.ndarray,
    policy: dict[str, Any],
    *,
    max_residual_disagreement_fraction: float,
    pair_index_basis: str = PAIR_INDEX_BASIS_AUTO,
) -> np.ndarray:
    diff = base != target
    if policy["selection"] == "all_base_target_differences":
        return diff
    if policy["selection"] == "longest_diff_runs_to_residual_gate":
        return _longest_diff_run_threshold_selection(
            base,
            target,
            max_residual_disagreement_fraction=max_residual_disagreement_fraction,
        )
    selected = diff.copy()
    pairs = [int(value) for value in policy.get("pair_indices") or []]
    if pairs:
        selected &= _pair_frame_mask(
            int(base.shape[0]),
            set(pairs),
            pair_index_basis=pair_index_basis,
        )[:, None, None]
    else:
        selected &= False
    class_ids = [int(value) for value in policy.get("class_ids") or []]
    if class_ids:
        selected &= np.isin(target, class_ids)
    return selected


def _validate_policy_pair_indices(
    pair_indices: Iterable[int],
    *,
    frame_count: int,
    pair_index_basis: str,
    policy_id: str,
) -> list[int]:
    resolved_basis = _resolve_pair_index_basis(frame_count, pair_index_basis)
    max_allowed = frame_count - 1
    if resolved_basis == PAIR_INDEX_BASIS_VIDEO_FRAME:
        max_allowed = max(0, (frame_count - 1) // 2)
    out: list[int] = []
    for raw in pair_indices:
        value = int(raw)
        if value < 0 or value > max_allowed:
            raise ReversedBasePlannerError(
                f"{policy_id} pair index {value} outside {resolved_basis} range "
                f"[0,{max_allowed}] for frame_count={frame_count}"
            )
        out.append(value)
    return sorted(set(out))


def _pair_indices_for_mask(
    selected: np.ndarray,
    *,
    pair_index_basis: str,
) -> list[int]:
    return _selected_pair_indices_from_frame_indices(
        np.nonzero(selected)[0],
        frame_count=int(selected.shape[0]),
        pair_index_basis=pair_index_basis,
    )


def _candidate_summary(
    *,
    base: BaseCandidate,
    base_array: np.ndarray,
    target: np.ndarray,
    target_sha: str,
    policy: dict[str, Any],
    max_residual_disagreement_fraction: float,
    pair_index_basis: str,
    repo_root: Path,
) -> dict[str, Any]:
    if base_array.shape != target.shape:
        raise ReversedBasePlannerError(
            f"{base.label} shape {base_array.shape} differs from target {target.shape}"
        )
    diff = base_array != target
    policy_pair_indices = _validate_policy_pair_indices(
        [int(value) for value in policy.get("pair_indices") or []],
        frame_count=int(target.shape[0]),
        pair_index_basis=pair_index_basis,
        policy_id=str(policy.get("policy_id") or "unknown_policy"),
    )
    if policy_pair_indices != list(policy.get("pair_indices") or []):
        policy = dict(policy)
        policy["pair_indices"] = policy_pair_indices
    selected = _selected_pixels_for_policy(
        base_array,
        target,
        policy,
        max_residual_disagreement_fraction=max_residual_disagreement_fraction,
        pair_index_basis=pair_index_basis,
    )
    final = base_array.copy()
    final[selected] = target[selected]
    residual = final != target
    runs = _runs_from_selected(base_array, target, selected)
    base_sha = _mask_tensor_sha256(base_array)
    final_sha = _mask_tensor_sha256(final)
    resolved_pair_index_basis = _resolve_pair_index_basis(
        int(target.shape[0]),
        pair_index_basis,
    )
    header = {
        "schema": "c067_decoded_delta_overlay_payload_v1",
        "producer": TOOL,
        "score_claim": False,
        "base_mask_tensor_sha256": base_sha,
        "target_mask_tensor_sha256": target_sha,
        "reconstructed_mask_u8_sha256": final_sha,
        "shape": [int(value) for value in target.shape],
        "pair_index_basis": resolved_pair_index_basis,
        "run_struct": RUN_STRUCT_NAME,
        "run_count": len(runs),
        "selected_pixel_count": int(selected.sum()),
        "selected_pair_indices": _pair_indices_for_mask(
            selected,
            pair_index_basis=resolved_pair_index_basis,
        ),
        "policy_id": policy["policy_id"],
    }
    raw_payload = _encode_overlay_payload(runs=runs, header=header)
    payloads = _compressed_payloads(raw_payload)
    best = _best_compressor(payloads)
    base_payload_bytes = int(base.payload_path.stat().st_size)
    best_overlay_bytes = int(payloads[best]["bytes"])
    estimated_archive_bytes = C067_NON_MASK_PACKED_BYTES + base_payload_bytes + best_overlay_bytes
    residual_count = int(residual.sum())
    total = int(target.size)
    residual_fraction = residual_count / float(total)
    selected_count = int(selected.sum())
    byte_gate_sub0300 = estimated_archive_bytes <= UNCHANGED_DISTORTION_SUB0300_BYTE_GATE
    byte_gate_sub0240 = estimated_archive_bytes <= UNCHANGED_DISTORTION_SUB0240_BYTE_GATE
    geometry_gate = residual_fraction <= max_residual_disagreement_fraction
    return {
        "candidate_id": f"{base.label}__{policy['policy_id']}",
        "score_claim": False,
        "promotion_eligible": False,
        "base": {
            "label": base.label,
            "decoded_mask_array": _display_path(base.decoded_mask_array, repo_root),
            "payload_path": _display_path(base.payload_path, repo_root),
            "payload_bytes": base_payload_bytes,
            "payload_sha256": sha256_file(base.payload_path),
            "decoded_mask_tensor_sha256": base_sha,
        },
        "policy": policy,
        "mask_disagreement": {
            "pair_index_basis": resolved_pair_index_basis,
            "base_vs_target_count": int(diff.sum()),
            "base_vs_target_fraction": round(int(diff.sum()) / float(total), 12),
            "base_vs_target_pair_indices": _pair_indices_for_mask(
                diff,
                pair_index_basis=resolved_pair_index_basis,
            ),
            "selected_overlay_pixels": selected_count,
            "selected_overlay_fraction": round(selected_count / float(total), 12),
            "selected_overlay_pair_indices": header["selected_pair_indices"],
            "residual_vs_target_count_after_overlay": residual_count,
            "residual_vs_target_fraction_after_overlay": round(residual_fraction, 12),
            "residual_vs_target_pair_indices_after_overlay": _pair_indices_for_mask(
                residual,
                pair_index_basis=resolved_pair_index_basis,
            ),
            "max_residual_disagreement_fraction": max_residual_disagreement_fraction,
        },
        "cdo1_payload": {
            "raw_bytes": len(raw_payload),
            "raw_sha256": sha256_bytes(raw_payload),
            "run_count": len(runs),
            "payload_header": header,
            "compressed_payloads": payloads,
            "recommended_compressor": best,
        },
        "estimated_archive": {
            "model": "c067_packed_non_mask_bytes_plus_base_payload_plus_cdo1_payload",
            "c067_frontier_archive_bytes": C067_FRONTIER_ARCHIVE_BYTES,
            "c067_mask_stream_bytes_removed": C067_MASK_STREAM_BYTES,
            "c067_non_mask_packed_bytes": C067_NON_MASK_PACKED_BYTES,
            "base_payload_bytes": base_payload_bytes,
            "best_overlay_payload_bytes": best_overlay_bytes,
            "estimated_archive_bytes": estimated_archive_bytes,
            "estimated_delta_vs_c067": estimated_archive_bytes - C067_FRONTIER_ARCHIVE_BYTES,
            "estimated_rate_delta_vs_c067": 25 * (estimated_archive_bytes - C067_FRONTIER_ARCHIVE_BYTES) / ORIGINAL_VIDEO_BYTES,
            "unchanged_distortion_sub0300_byte_gate": UNCHANGED_DISTORTION_SUB0300_BYTE_GATE,
            "unchanged_distortion_sub0240_byte_gate": UNCHANGED_DISTORTION_SUB0240_BYTE_GATE,
        },
        "gates": {
            "byte_gate_sub0300_if_distortion_unchanged": byte_gate_sub0300,
            "byte_gate_sub0240_if_distortion_unchanged": byte_gate_sub0240,
            "residual_geometry_gate": geometry_gate,
            "joint_sub0300_geometry_gate": byte_gate_sub0300 and geometry_gate,
            "joint_sub0240_geometry_gate": byte_gate_sub0240 and geometry_gate,
        },
    }


def build_plan(
    *,
    repo_root: Path,
    target_mask_array: Path,
    base_candidates: Iterable[BaseCandidate],
    trust_plan_json: Path,
    max_residual_disagreement_fraction: float = DEFAULT_MAX_RESIDUAL_DISAGREEMENT_FRACTION,
    pair_index_basis: str = PAIR_INDEX_BASIS_AUTO,
) -> dict[str, Any]:
    if max_residual_disagreement_fraction <= 0.0:
        raise ReversedBasePlannerError("max_residual_disagreement_fraction must be positive")
    target = _load_mask_array(target_mask_array)
    resolved_pair_index_basis = _resolve_pair_index_basis(
        int(target.shape[0]),
        pair_index_basis,
    )
    target_sha = _mask_tensor_sha256(target)
    policies = _policy_rows_from_trust_plan(trust_plan_json)
    policies.insert(
        1,
        {
            "policy_id": "geometry_threshold_longest_runs_to_residual_gate",
            "selection": "longest_diff_runs_to_residual_gate",
            "pair_indices": [],
            "class_ids": [],
            "source": "deterministic_largest_run_waterfill_lower_bound",
            "max_residual_disagreement_fraction": max_residual_disagreement_fraction,
        },
    )
    candidates: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for base in base_candidates:
        if not base.decoded_mask_array.exists() or not base.payload_path.exists():
            missing.append(
                {
                    "label": base.label,
                    "decoded_mask_array": _display_path(base.decoded_mask_array, repo_root),
                    "payload_path": _display_path(base.payload_path, repo_root),
                }
            )
            continue
        base_array = _load_mask_array(base.decoded_mask_array)
        for policy in policies:
            candidates.append(
                _candidate_summary(
                    base=base,
                    base_array=base_array,
                    target=target,
                    target_sha=target_sha,
                    policy=policy,
                    max_residual_disagreement_fraction=max_residual_disagreement_fraction,
                    pair_index_basis=resolved_pair_index_basis,
                    repo_root=repo_root,
                )
            )
    candidates.sort(
        key=lambda row: (
            not bool(row["gates"]["joint_sub0240_geometry_gate"]),
            not bool(row["gates"]["joint_sub0300_geometry_gate"]),
            not bool(row["gates"]["byte_gate_sub0240_if_distortion_unchanged"]),
            float(row["mask_disagreement"]["residual_vs_target_fraction_after_overlay"]),
            int(row["estimated_archive"]["estimated_archive_bytes"]),
            str(row["candidate_id"]),
        )
    )
    joint_sub0240 = [row for row in candidates if row["gates"]["joint_sub0240_geometry_gate"]]
    joint_sub0300 = [row for row in candidates if row["gates"]["joint_sub0300_geometry_gate"]]
    byte_only = [
        row for row in candidates
        if row["gates"]["byte_gate_sub0240_if_distortion_unchanged"]
        and not row["gates"]["residual_geometry_gate"]
    ]
    geometry_only = [
        row for row in candidates
        if row["gates"]["residual_geometry_gate"]
        and not row["gates"]["byte_gate_sub0300_if_distortion_unchanged"]
    ]
    if joint_sub0240:
        decision = "joint_sub0240_geometry_candidate_requires_archive_build_and_exact_eval"
    elif joint_sub0300:
        decision = "joint_sub0300_geometry_candidate_requires_archive_build_and_exact_eval"
    elif byte_only:
        decision = "byte_headroom_but_geometry_blocked"
    elif geometry_only:
        decision = "geometry_safe_but_byte_regressive"
    else:
        decision = "no_reversed_base_cdo1_candidate_passes_joint_gates"
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_jobs_dispatched": False,
        "target_decoded_mask": {
            "path": _display_path(target_mask_array, repo_root),
            "tensor_sha256": target_sha,
            "shape": [int(value) for value in target.shape],
            "dtype": str(target.dtype),
        },
        "gates": {
            "pair_index_basis": resolved_pair_index_basis,
            "max_residual_disagreement_fraction": max_residual_disagreement_fraction,
            "unchanged_distortion_sub0300_byte_gate": UNCHANGED_DISTORTION_SUB0300_BYTE_GATE,
            "unchanged_distortion_sub0240_byte_gate": UNCHANGED_DISTORTION_SUB0240_BYTE_GATE,
        },
        "candidate_count": len(candidates),
        "missing_candidate_inputs": missing,
        "joint_sub0240_geometry_count": len(joint_sub0240),
        "joint_sub0300_geometry_count": len(joint_sub0300),
        "byte_headroom_geometry_blocked_count": len(byte_only),
        "geometry_safe_byte_regressive_count": len(geometry_only),
        "decision": decision,
        "best_candidates": candidates[:12],
        "all_candidates": candidates,
        "required_before_remote_eval": [
            "build deterministic byte-closed archive only if a candidate passes residual geometry and byte gates",
            "claim lane with tools/claim_lane_dispatch.py before remote exact eval",
            "run exact CUDA auth eval; this planner makes no score claim",
        ],
    }


def _parse_base_candidate(raw: str, repo_root: Path) -> BaseCandidate:
    try:
        label, decoded, payload = raw.split("=", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "base candidate must be label=decoded_mask_array=payload_path"
        ) from exc
    decoded_path = Path(decoded)
    payload_path = Path(payload)
    if not decoded_path.is_absolute():
        decoded_path = repo_root / decoded_path
    if not payload_path.is_absolute():
        payload_path = repo_root / payload_path
    return BaseCandidate(label=label, decoded_mask_array=decoded_path, payload_path=payload_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--target-mask-array", type=Path, default=DEFAULT_TARGET_MASK_ARRAY)
    parser.add_argument("--trust-plan-json", type=Path, default=DEFAULT_TRUST_PLAN_JSON)
    parser.add_argument("--base-candidate", action="append", default=[])
    parser.add_argument(
        "--max-residual-disagreement-fraction",
        type=float,
        default=DEFAULT_MAX_RESIDUAL_DISAGREEMENT_FRACTION,
    )
    parser.add_argument(
        "--pair-index-basis",
        choices=PAIR_INDEX_BASIS_CHOICES,
        default=PAIR_INDEX_BASIS_AUTO,
        help=(
            "How trust-plan pair indices map to mask tensor rows. auto uses "
            "half_frame_pair_index for 600-row half-frame C067 masks and "
            "video_frame_pair_index otherwise."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_json
    if output_json.exists() and not args.force:
        raise SystemExit(f"{output_json} exists; pass --force to overwrite")
    base_candidates = (
        tuple(_parse_base_candidate(raw, args.repo_root) for raw in args.base_candidate)
        if args.base_candidate
        else DEFAULT_BASE_CANDIDATES
    )
    plan = build_plan(
        repo_root=args.repo_root,
        target_mask_array=args.target_mask_array,
        base_candidates=base_candidates,
        trust_plan_json=args.trust_plan_json,
        max_residual_disagreement_fraction=args.max_residual_disagreement_fraction,
        pair_index_basis=args.pair_index_basis,
    )
    _write_json(output_json, plan)
    print(
        json.dumps(
            {
                "output_json": _display_path(output_json, args.repo_root),
                "decision": plan["decision"],
                "candidate_count": plan["candidate_count"],
                "joint_sub0240_geometry_count": plan["joint_sub0240_geometry_count"],
                "joint_sub0300_geometry_count": plan["joint_sub0300_geometry_count"],
                "byte_headroom_geometry_blocked_count": plan["byte_headroom_geometry_blocked_count"],
                "geometry_safe_byte_regressive_count": plan["geometry_safe_byte_regressive_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
