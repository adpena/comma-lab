#!/usr/bin/env python3
"""Local pose-safety preflight for renderer transplants.

This profiler compares the source runtime renderer against a transplanted
renderer archive on the same charged masks and optimized poses. It is a local
fail-closed gate only: it emits no score claim, no promotion, and no remote
dispatch command.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SCHEMA = "renderer_transplant_pose_safety_preflight_v1"
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip"
)
DEFAULT_CANDIDATE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/trained_renderer_blockfp_preflight_20260503_qfaithful_qzs3_2146/"
    "trained_qbf1_b0512/archive.zip"
)
RENDERER_MEMBER = "renderer.bin"
MASK_MEMBER = "masks.mkv"
POSE_MEMBER_NAMES = ("optimized_poses.bin", "optimized_poses.qp1")
OPTIONAL_UNCHANGED_MEMBERS = (
    "seg_tile_actions.bin",
    "seg_tile_action_dict.bin",
    "zoom_scalars.bin",
)
ALLOWED_LOGICAL_MEMBERS = (
    RENDERER_MEMBER,
    MASK_MEMBER,
    *POSE_MEMBER_NAMES,
    *OPTIONAL_UNCHANGED_MEMBERS,
)
PAYLOAD_MEMBER_NAMES = ("p", "renderer_payload.bin", "renderer_payload.bin.br")
DEFAULT_MAX_PAIRS = 32
DEFAULT_MAX_MEAN_ABS_DELTA = 0.05
DEFAULT_MAX_RMS_DELTA = 0.08
DEFAULT_MAX_MAX_ABS_DELTA = 1.5


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BLOCKFP_BUILDER = _load_module(
    REPO_ROOT / "experiments" / "build_blockfp_c067_archive.py",
    "_pose_safety_blockfp_builder",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _inspect_zip_contract(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
    names = [info.filename for info in infos]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    unsafe_names = [
        name
        for name in names
        if not name
        or name.startswith("/")
        or "\\" in name
        or any(part in {"", ".", ".."} for part in Path(name).parts)
        or name.startswith("__MACOSX/")
        or Path(name).name.startswith("._")
    ]
    payload_members = [name for name in names if name in PAYLOAD_MEMBER_NAMES]
    return {
        "zip_members": names,
        "duplicate_names": duplicate_names,
        "unsafe_names": unsafe_names,
        "payload_members": payload_members,
        "exactly_one_payload_member": len(payload_members) == 1,
        "safe_zip_names": not duplicate_names and not unsafe_names,
    }


def extract_runtime_contract(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Extract runtime members and enforce archive contract invariants."""

    path = path.resolve()
    zip_contract = _inspect_zip_contract(path)
    members, packaging = BLOCKFP_BUILDER.extract_runtime_members(path)
    logical_names = list(members)
    missing_core = [
        name for name in (RENDERER_MEMBER, MASK_MEMBER) if name not in members
    ]
    present_pose_members = [name for name in POSE_MEMBER_NAMES if name in members]
    missing = [*missing_core]
    if not present_pose_members:
        missing.append("optimized_pose_payload")
    extras = sorted(name for name in logical_names if name not in ALLOWED_LOGICAL_MEMBERS)
    ok = (
        zip_contract["safe_zip_names"]
        and not missing
        and not extras
        and (
            packaging.get("source_archive_packaging") != "packed_renderer_payload"
            or zip_contract["exactly_one_payload_member"]
        )
    )
    contract = {
        "path": str(path),
        "repo_relative_path": _repo_rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "ok": ok,
        "missing_required_members": missing,
        "extra_logical_members": extras,
        "pose_members": present_pose_members,
        "logical_members": {
            name: {
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "magic4": payload[:4].hex(),
            }
            for name, payload in members.items()
        },
        **zip_contract,
        **packaging,
    }
    return members, contract


def validate_transplant_contract(
    source_members: dict[str, bytes],
    candidate_members: dict[str, bytes],
) -> dict[str, Any]:
    """Verify the candidate changed only renderer bytes."""

    required_missing = {
        "source": [name for name in (RENDERER_MEMBER, MASK_MEMBER) if name not in source_members],
        "candidate": [name for name in (RENDERER_MEMBER, MASK_MEMBER) if name not in candidate_members],
    }
    source_pose_members = [name for name in POSE_MEMBER_NAMES if name in source_members]
    candidate_pose_members = [name for name in POSE_MEMBER_NAMES if name in candidate_members]
    if not source_pose_members:
        required_missing["source"].append("optimized_pose_payload")
    if not candidate_pose_members:
        required_missing["candidate"].append("optimized_pose_payload")

    source_names = set(source_members)
    candidate_names = set(candidate_members)
    non_renderer_name_mismatch = sorted(
        (source_names ^ candidate_names) - {RENDERER_MEMBER}
    )
    comparisons: dict[str, Any] = {}
    for name in sorted(source_names | candidate_names):
        if name not in source_members or name not in candidate_members:
            continue
        comparisons[name] = {
            "source_bytes": len(source_members[name]),
            "candidate_bytes": len(candidate_members[name]),
            "source_sha256": _sha256_bytes(source_members[name]),
            "candidate_sha256": _sha256_bytes(candidate_members[name]),
            "same_bytes": source_members[name] == candidate_members[name],
        }
    masks_same = bool(comparisons.get(MASK_MEMBER, {}).get("same_bytes"))
    pose_names_match = source_pose_members == candidate_pose_members
    poses_same = bool(
        pose_names_match
        and source_pose_members
        and all(comparisons.get(name, {}).get("same_bytes") for name in source_pose_members)
    )
    unchanged_aux_same = all(
        bool(comparisons.get(name, {}).get("same_bytes"))
        for name in sorted((source_names | candidate_names) - {RENDERER_MEMBER, MASK_MEMBER, *POSE_MEMBER_NAMES})
    )
    renderer_changed = not bool(comparisons.get(RENDERER_MEMBER, {}).get("same_bytes"))
    ok = (
        not required_missing["source"]
        and not required_missing["candidate"]
        and not non_renderer_name_mismatch
        and masks_same
        and poses_same
        and unchanged_aux_same
        and renderer_changed
    )
    failures: list[str] = []
    if required_missing["source"] or required_missing["candidate"]:
        failures.append("missing_required_runtime_member")
    if non_renderer_name_mismatch:
        failures.append("non_renderer_member_set_changed")
    if not masks_same:
        failures.append("mask_payload_changed")
    if not poses_same:
        failures.append("pose_payload_changed")
    if not unchanged_aux_same:
        failures.append("aux_payload_changed")
    if not renderer_changed:
        failures.append("renderer_payload_unchanged_or_surrogate")
    return {
        "ok": ok,
        "required_missing": required_missing,
        "source_pose_members": source_pose_members,
        "candidate_pose_members": candidate_pose_members,
        "non_renderer_name_mismatch": non_renderer_name_mismatch,
        "comparisons": comparisons,
        "failures": failures,
    }


def _select_pair_indices(pair_count: int, max_pairs: int) -> list[int]:
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if max_pairs >= pair_count:
        return list(range(pair_count))
    values = {
        int(round(i * (pair_count - 1) / (max_pairs - 1)))
        for i in range(max_pairs)
    }
    return sorted(values)


def _frame_hash(tensor: Any) -> str:
    torch = _require_torch()
    as_u8 = torch.clamp(tensor.detach().cpu().round(), 0, 255).to(torch.uint8)
    return _sha256_bytes(as_u8.contiguous().numpy().tobytes())


def _summarize_frames(label: str, frames: Any) -> dict[str, Any]:
    return {
        "label": label,
        "shape": list(frames.shape),
        "uint8_sha256": _frame_hash(frames),
        "mean": float(frames.float().mean().item()),
        "std": float(frames.float().std(unbiased=False).item()),
        "min": float(frames.float().min().item()),
        "max": float(frames.float().max().item()),
    }


def compare_frame_batches(
    source_frames: Any,
    target_frames: Any,
    *,
    max_mean_abs_delta: float,
    max_rms_delta: float,
    max_max_abs_delta: float,
) -> dict[str, Any]:
    """Compare two rendered batches and return a fail-closed verdict."""

    if tuple(source_frames.shape) != tuple(target_frames.shape):
        return {
            "ok": False,
            "failure_class": "render_shape_mismatch",
            "source_shape": list(source_frames.shape),
            "target_shape": list(target_frames.shape),
        }
    diff = target_frames.float() - source_frames.float()
    mean_abs = float(diff.abs().mean().item())
    rms = float(torch_square_mean(diff))
    max_abs = float(diff.abs().max().item())
    same_hash = _frame_hash(source_frames) == _frame_hash(target_frames)
    ok = (
        mean_abs <= max_mean_abs_delta
        and rms <= max_rms_delta
        and max_abs <= max_max_abs_delta
    )
    failures: list[str] = []
    if mean_abs > max_mean_abs_delta:
        failures.append("mean_abs_delta_exceeds_threshold")
    if rms > max_rms_delta:
        failures.append("rms_delta_exceeds_threshold")
    if max_abs > max_max_abs_delta:
        failures.append("max_abs_delta_exceeds_threshold")
    return {
        "ok": ok,
        "same_uint8_hash": same_hash,
        "mean_abs_delta": mean_abs,
        "rms_delta": rms,
        "max_abs_delta": max_abs,
        "thresholds": {
            "max_mean_abs_delta": max_mean_abs_delta,
            "max_rms_delta": max_rms_delta,
            "max_max_abs_delta": max_max_abs_delta,
        },
        "failures": failures,
    }


def torch_square_mean(tensor: Any) -> float:
    return math.sqrt(float((tensor.float() * tensor.float()).mean().item()))


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - project dependency.
        raise RuntimeError("renderer pose-safety preflight requires torch") from exc
    return torch


def _write_members(directory: Path, members: dict[str, bytes]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in members.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _load_runtime_state(directory: Path, members: dict[str, bytes]) -> dict[str, Any]:
    inflate = _load_module(
        REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py",
        f"_pose_safety_inflate_runtime_{hash(directory)}",
    )
    _write_members(directory, members)
    renderer = inflate._load_renderer(str(directory / "renderer.bin"), "cpu")
    masks = inflate._load_archive_masks_with_optional_amr1_repair(
        directory,
        directory / "masks.mkv",
    )
    from tac.submission_archive import load_optimized_poses

    pose_dim = int(getattr(renderer, "pose_dim", 6) or 6)
    optimized_bin = directory / "optimized_poses.bin"
    optimized_qp1 = directory / "optimized_poses.qp1"
    if optimized_bin.exists():
        poses = load_optimized_poses(optimized_bin, pose_dim=pose_dim)
        pose_source = "optimized_poses.bin"
    elif optimized_qp1.exists():
        poses = inflate._decode_qp1_poses_float32(optimized_qp1, pose_dim=pose_dim)
        pose_source = "optimized_poses.qp1"
    else:
        raise FileNotFoundError("missing optimized_poses.bin or optimized_poses.qp1")
    return {
        "renderer": renderer,
        "masks": masks,
        "poses": poses,
        "pose_dim": pose_dim,
        "pose_source": pose_source,
        "half_frame_masks": bool(getattr(masks, "_half_frame_only", False)),
        "mask_shape": list(masks.shape),
        "pose_shape": list(poses.shape),
    }


def _render_pair_batch(
    *,
    renderer: Any,
    masks: Any,
    poses: Any,
    pair_indices: list[int],
) -> Any:
    torch = _require_torch()
    frames = []
    with torch.inference_mode():
        for pair_index in pair_indices:
            if bool(getattr(masks, "_half_frame_only", False)):
                mask_t = masks[pair_index]
                mask_t1 = masks[pair_index]
            else:
                mask_t = masks[2 * pair_index]
                mask_t1 = masks[2 * pair_index + 1]
            pose = poses[pair_index].to(dtype=torch.float32).view(1, -1)
            pair = renderer(
                mask_t.view(1, *mask_t.shape).to(dtype=torch.long),
                mask_t1.view(1, *mask_t1.shape).to(dtype=torch.long),
                pose=pose,
            )
            frames.append(pair.detach().cpu())
    return torch.cat(frames, dim=0)


def _build_output_parity(
    *,
    source_state: dict[str, Any],
    candidate_state: dict[str, Any],
    max_pairs: int,
    max_mean_abs_delta: float,
    max_rms_delta: float,
    max_max_abs_delta: float,
) -> dict[str, Any]:
    source_masks = source_state["masks"]
    candidate_masks = candidate_state["masks"]
    source_poses = source_state["poses"]
    candidate_poses = candidate_state["poses"]
    if list(source_masks.shape) != list(candidate_masks.shape):
        return {
            "ok": False,
            "failure_class": "mask_shape_mismatch",
            "source_mask_shape": list(source_masks.shape),
            "candidate_mask_shape": list(candidate_masks.shape),
        }
    if list(source_poses.shape) != list(candidate_poses.shape):
        return {
            "ok": False,
            "failure_class": "pose_shape_mismatch",
            "source_pose_shape": list(source_poses.shape),
            "candidate_pose_shape": list(candidate_poses.shape),
        }
    pair_count = int(source_poses.shape[0])
    if bool(getattr(source_masks, "_half_frame_only", False)):
        pair_count = min(pair_count, int(source_masks.shape[0]))
    else:
        pair_count = min(pair_count, int(source_masks.shape[0]) // 2)
    pair_indices = _select_pair_indices(pair_count, max_pairs)
    source_frames = _render_pair_batch(
        renderer=source_state["renderer"],
        masks=source_masks,
        poses=source_poses,
        pair_indices=pair_indices,
    )
    candidate_frames = _render_pair_batch(
        renderer=candidate_state["renderer"],
        masks=candidate_masks,
        poses=candidate_poses,
        pair_indices=pair_indices,
    )
    aggregate = compare_frame_batches(
        source_frames,
        candidate_frames,
        max_mean_abs_delta=max_mean_abs_delta,
        max_rms_delta=max_rms_delta,
        max_max_abs_delta=max_max_abs_delta,
    )
    per_pair = []
    for offset, pair_index in enumerate(pair_indices):
        comparison = compare_frame_batches(
            source_frames[offset : offset + 1],
            candidate_frames[offset : offset + 1],
            max_mean_abs_delta=max_mean_abs_delta,
            max_rms_delta=max_rms_delta,
            max_max_abs_delta=max_max_abs_delta,
        )
        per_pair.append({"pair_index": pair_index, **comparison})
    return {
        "ok": bool(aggregate.get("ok")),
        "failure_class": None if aggregate.get("ok") else "render_output_parity_unsafe",
        "pair_count_available": pair_count,
        "sampled_pair_indices": pair_indices,
        "source_output_summary": _summarize_frames("source", source_frames),
        "candidate_output_summary": _summarize_frames("candidate", candidate_frames),
        "aggregate": aggregate,
        "per_pair": per_pair,
    }


def build_pose_safety_preflight(
    *,
    source_archive: Path,
    candidate_archive: Path,
    output_json: Path,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    max_mean_abs_delta: float = DEFAULT_MAX_MEAN_ABS_DELTA,
    max_rms_delta: float = DEFAULT_MAX_RMS_DELTA,
    max_max_abs_delta: float = DEFAULT_MAX_MAX_ABS_DELTA,
) -> dict[str, Any]:
    """Build the local no-score renderer transplant safety report."""

    source_members, source_contract = extract_runtime_contract(source_archive)
    candidate_members, candidate_contract = extract_runtime_contract(candidate_archive)
    transplant_contract = validate_transplant_contract(source_members, candidate_members)
    with tempfile.TemporaryDirectory(prefix="renderer_pose_safety_") as tmp:
        tmp_root = Path(tmp)
        source_state = _load_runtime_state(tmp_root / "source", source_members)
        candidate_state = _load_runtime_state(tmp_root / "candidate", candidate_members)
        output_parity = _build_output_parity(
            source_state=source_state,
            candidate_state=candidate_state,
            max_pairs=max_pairs,
            max_mean_abs_delta=max_mean_abs_delta,
            max_rms_delta=max_rms_delta,
            max_max_abs_delta=max_max_abs_delta,
        )
    safe_for_exact_eval = bool(
        source_contract["ok"]
        and candidate_contract["ok"]
        and transplant_contract["ok"]
        and output_parity["ok"]
    )
    fail_closed_reasons: list[str] = []
    if not source_contract["ok"]:
        fail_closed_reasons.append("source_archive_contract_failed")
    if not candidate_contract["ok"]:
        fail_closed_reasons.append("candidate_archive_contract_failed")
    if not transplant_contract["ok"]:
        fail_closed_reasons.extend(transplant_contract["failures"])
    if not output_parity["ok"]:
        fail_closed_reasons.append(output_parity.get("failure_class") or "output_parity_failed")
    report = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "safe_for_exact_eval_dispatch": safe_for_exact_eval,
        "evidence_grade": "empirical_local_preflight_no_score",
        "failure_class": None if safe_for_exact_eval else "renderer_transplant_pose_safety_failed",
        "fail_closed_reasons": sorted(set(fail_closed_reasons)),
        "source_archive": source_contract,
        "candidate_archive": candidate_contract,
        "transplant_contract": transplant_contract,
        "runtime_state": {
            "source": {
                "mask_shape": source_state["mask_shape"],
                "pose_shape": source_state["pose_shape"],
                "pose_dim": source_state["pose_dim"],
                "pose_source": source_state["pose_source"],
                "half_frame_masks": source_state["half_frame_masks"],
            },
            "candidate": {
                "mask_shape": candidate_state["mask_shape"],
                "pose_shape": candidate_state["pose_shape"],
                "pose_dim": candidate_state["pose_dim"],
                "pose_source": candidate_state["pose_source"],
                "half_frame_masks": candidate_state["half_frame_masks"],
            },
        },
        "output_parity": output_parity,
        "provenance": {
            "tool": "experiments/preflight_renderer_transplant_pose_safety.py",
            "canonical_score_source_required": (
                "archive.zip -> inflate.sh -> upstream/evaluate.py via "
                "experiments/contest_auth_eval.py --device cuda"
            ),
            "no_score_no_promotion_reason": (
                "local renderer output parity is a dispatch preflight only; "
                "CUDA auth eval remains the score truth"
            ),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_bytes(_json_bytes(report))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--candidate-archive", type=Path, default=DEFAULT_CANDIDATE_ARCHIVE)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument(
        "--max-mean-abs-delta",
        type=float,
        default=DEFAULT_MAX_MEAN_ABS_DELTA,
    )
    parser.add_argument("--max-rms-delta", type=float, default=DEFAULT_MAX_RMS_DELTA)
    parser.add_argument(
        "--max-max-abs-delta",
        type=float,
        default=DEFAULT_MAX_MAX_ABS_DELTA,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_pose_safety_preflight(
        source_archive=args.source_archive,
        candidate_archive=args.candidate_archive,
        output_json=args.output_json,
        max_pairs=args.max_pairs,
        max_mean_abs_delta=args.max_mean_abs_delta,
        max_rms_delta=args.max_rms_delta,
        max_max_abs_delta=args.max_max_abs_delta,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["safe_for_exact_eval_dispatch"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
