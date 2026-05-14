#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build byte-closed C067 multimask reconciliation candidates.

This bridges ``plan_multimask_reconciliation_atoms.py`` to an existing contest
runtime contract.  It materializes a deterministic fused mask from a ranked
multimask policy, then re-encodes that fused tensor as ``masks.cmg3`` through
the reviewed CMG3A nonzero-row-run path.  The resulting archive is a byte
screen only; it is not score evidence until exact CUDA auth eval runs on the
archive bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER_PATH = REPO_ROOT / "experiments" / "plan_multimask_reconciliation_atoms.py"
CMG3A_BUILDER_PATH = REPO_ROOT / "experiments" / "build_cmg3_adaptive_runs_candidate.py"
PACKER_PATH = REPO_ROOT / "experiments" / "build_renderer_packed_payload_archive.py"

SCHEMA = "c067_multimask_reconciler_cmg3a_candidate_v1"
DEFAULT_FRONTIER_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/c067_breakthrough_candidate_matrix_20260502T1030Z/"
    / "line_search_source_c067_fixedslice/archive.zip"
)
DEFAULT_PLAN_JSON = (
    REPO_ROOT
    / "experiments/results/c067_multimask_reconciliation_20260502/"
    / "multimask_reconciliation_plan.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/results/c067_multimask_reconciliation_20260502_cmg3a_reconciler"
)
DEFAULT_TARGET_BODY_BYTES: tuple[int, ...] = ()
DEFAULT_TARGET_EXTRA_RUNS = (55_000, 65_000, 72_000)
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)


class ReconcilerBuildError(ValueError):
    """Raised when a multimask candidate cannot be built safely."""


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PLANNER = _load_module(PLANNER_PATH, "_c067_multimask_reconciler_planner")
CMG3A = _load_module(CMG3A_BUILDER_PATH, "_c067_multimask_reconciler_cmg3a")
PACKER = _load_module(PACKER_PATH, "_c067_multimask_reconciler_packer")


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _mask_u8_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    return _sha256_bytes(contiguous.tobytes(order="C"))


def _diff_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        raise ReconcilerBuildError(f"shape mismatch: {a.shape} != {b.shape}")
    changed = a != b
    count = int(np.count_nonzero(changed))
    total = int(changed.size)
    return {
        "disagreement_count": count,
        "disagreement_fraction": round(count / total, 12),
        "agreement_fraction": round(1.0 - count / total, 12),
    }


def _load_mask_array(path: Path) -> np.ndarray:
    path = path.resolve()
    if path.suffix.lower() == ".npy":
        raw = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        try:
            keys = list(archive.files)
            preferred = [key for key in ("masks", "mask", "decoded_masks", "array") if key in keys]
            if len(preferred) == 1:
                raw = archive[preferred[0]]
            elif len(keys) == 1:
                raw = archive[keys[0]]
            else:
                raise ReconcilerBuildError(f"{path} must contain one array or a masks/decoded_masks array")
        finally:
            archive.close()
    else:
        raise ReconcilerBuildError(f"{path} must be a .npy or .npz decoded mask array")
    array = np.asarray(raw)
    if array.ndim != 3 or array.dtype != np.uint8:
        raise ReconcilerBuildError(f"{path} must be uint8 rank-3 masks, got {array.shape} {array.dtype}")
    if array.shape[1:] != (CMG3A.HEIGHT, CMG3A.WIDTH):
        raise ReconcilerBuildError(
            f"{path} must be {CMG3A.HEIGHT}x{CMG3A.WIDTH}, got {array.shape[1:]}"
        )
    if int(array.min()) < 0 or int(array.max()) >= CMG3A.CLASS_COUNT:
        raise ReconcilerBuildError(
            f"{path} classes must be in [0,{CMG3A.CLASS_COUNT}), got [{int(array.min())},{int(array.max())}]"
        )
    return np.ascontiguousarray(array)


def _read_plan(plan_json: Path) -> dict[str, Any]:
    try:
        payload = json.loads(plan_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReconcilerBuildError(f"{plan_json} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ReconcilerBuildError(f"{plan_json} must contain a JSON object")
    if payload.get("score_claim") is not False:
        raise ReconcilerBuildError("multimask plan must have score_claim=false")
    policies = payload.get("candidate_policies")
    if not isinstance(policies, list) or not policies:
        raise ReconcilerBuildError("multimask plan must contain candidate_policies")
    return payload


def select_policy(plan: dict[str, Any], policy_id: str | None = None) -> dict[str, Any]:
    policies = plan.get("candidate_policies")
    if not isinstance(policies, list) or not policies:
        raise ReconcilerBuildError("plan has no candidate policies")
    if policy_id is not None:
        for policy in policies:
            if isinstance(policy, dict) and policy.get("policy_id") == policy_id:
                if policy.get("score_claim") is not False:
                    raise ReconcilerBuildError(f"policy {policy_id!r} must have score_claim=false")
                return policy
        raise ReconcilerBuildError(f"policy_id {policy_id!r} not found")
    for policy in policies:
        if not isinstance(policy, dict):
            continue
        relevance = policy.get("dispatch_relevance")
        if (
            policy.get("score_claim") is False
            and isinstance(relevance, dict)
            and relevance.get("dispatchable_byte_model") is True
            and relevance.get("no_op_vs_source") is False
        ):
            return policy
    raise ReconcilerBuildError("no dispatchable non-no-op multimask policy found")


def _input_paths_from_plan(plan: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
    inputs = plan.get("inputs")
    if not isinstance(inputs, dict):
        raise ReconcilerBuildError("plan inputs must be an object")
    source = inputs.get("source")
    candidates = inputs.get("candidates")
    if not isinstance(source, dict) or not isinstance(candidates, list):
        raise ReconcilerBuildError("plan inputs must contain source and candidates")
    source_path = Path(str(source.get("path"))).resolve()
    candidate_paths: dict[str, Path] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ReconcilerBuildError("candidate input records must be objects")
        name = str(candidate.get("family_name"))
        path = Path(str(candidate.get("path"))).resolve()
        if not name or name in candidate_paths:
            raise ReconcilerBuildError(f"invalid or duplicate candidate family name: {name!r}")
        candidate_paths[name] = path
    return source_path, candidate_paths


def materialize_fused_mask(
    *,
    plan: dict[str, Any],
    policy: dict[str, Any],
    source: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> np.ndarray:
    fusion = policy.get("fusion_reconciliation_policy")
    if not isinstance(fusion, dict):
        raise ReconcilerBuildError("policy lacks fusion_reconciliation_policy")
    name = fusion.get("name")
    if name == "majority_vote":
        inputs = fusion.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ReconcilerBuildError("majority_vote policy requires input family list")
        source_name = str(plan.get("source_family_name", "source"))
        arrays: list[np.ndarray] = []
        for family in inputs:
            family_name = str(family)
            if family_name == source_name:
                arrays.append(source)
            elif family_name in candidates:
                arrays.append(candidates[family_name])
            else:
                raise ReconcilerBuildError(f"majority_vote references unknown family {family_name!r}")
        return np.ascontiguousarray(PLANNER._majority_vote(arrays, source=source).astype(np.uint8, copy=False))
    if name == "priority_order":
        priority = fusion.get("priority")
        if not isinstance(priority, list) or not priority:
            raise ReconcilerBuildError("priority_order policy requires priority list")
        first = str(priority[0])
        if first not in candidates:
            raise ReconcilerBuildError(f"priority_order first family {first!r} is not a candidate")
        return np.ascontiguousarray(candidates[first])
    if name == "disagreement_gated_veto":
        threshold = float(fusion.get("candidate_consensus_threshold", 1.0))
        family_names = [str(v) for v in policy.get("candidate_family_names", [])]
        if not family_names:
            raise ReconcilerBuildError("disagreement_gated_veto requires candidate_family_names")
        arrays = [candidates[name] for name in family_names]
        consensus, mask = PLANNER._candidate_consensus(arrays, threshold=threshold)
        return np.ascontiguousarray(np.where(mask, consensus, source).astype(np.uint8, copy=False))
    if name == "cheap_residual_over_base":
        family = str(fusion.get("residual_family"))
        if family not in candidates:
            raise ReconcilerBuildError(f"cheap residual references unknown family {family!r}")
        return np.ascontiguousarray(candidates[family])
    raise ReconcilerBuildError(f"unsupported fusion policy: {name!r}")


def _write_source_archive(path: Path, *, members: dict[str, bytes], cmg3_payload: bytes) -> None:
    CMG3A.BASE._write_source_archive(  # noqa: SLF001
        path,
        [
            ("renderer.bin", members["renderer.bin"]),
            ("masks.cmg3", cmg3_payload),
            ("optimized_poses.bin", members["optimized_poses.bin"]),
        ],
    )


def _build_one_archive(
    *,
    output_dir: Path,
    frontier_archive: Path,
    frontier_members: dict[str, bytes],
    source: np.ndarray,
    fused: np.ndarray,
    target_body_bytes: int | None,
    target_extra_runs: int | None,
    base_runs_per_row: int,
    adaptive_max_runs_per_row: int,
    compressor: str,
    body_search_mode: str,
) -> dict[str, Any]:
    if target_body_bytes is None and target_extra_runs is None:
        raise ReconcilerBuildError("candidate requires target_body_bytes or target_extra_runs")
    if target_body_bytes is not None and target_extra_runs is not None:
        raise ReconcilerBuildError("pass only one of target_body_bytes or target_extra_runs")
    candidate_label = (
        f"body{int(target_body_bytes):06d}"
        if target_body_bytes is not None
        else f"extra{int(target_extra_runs):06d}"
    )
    candidate_dir = output_dir / candidate_label
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    run_stream, recon, run_stats, cmg3_policy = CMG3A.encode_adaptive_run_stream(
        fused,
        base_runs_per_row=base_runs_per_row,
        target_extra_runs=target_extra_runs,
        target_body_bytes=target_body_bytes,
        adaptive_max_runs_per_row=adaptive_max_runs_per_row,
        compressor=compressor,
        body_search_mode=body_search_mode,
    )
    fused_sha = _mask_u8_sha256(fused)
    recon_sha = _mask_u8_sha256(recon)
    cmg3_payload, cmg3_header = CMG3A.encode_cmg3a_payload(
        run_stream,
        frame_count=int(fused.shape[0]),
        max_runs_per_row=int(run_stats["max_selected_runs_per_row"]),
        source_mask_sha256=fused_sha,
        reconstructed_mask_sha256=recon_sha,
        pixel_disagreement=float(run_stats["pixel_disagreement"]),
        pixel_disagreement_count=int(run_stats["pixel_disagreement_count"]),
        policy=cmg3_policy,
        compressor=compressor,
    )

    source_archive = candidate_dir / "multimask_reconciler_source_members.zip"
    _write_source_archive(source_archive, members=frontier_members, cmg3_payload=cmg3_payload)
    archive_path = candidate_dir / "archive.zip"
    packed_meta = PACKER.build_packed_archive(
        source_archive,
        archive_path,
        brotli_quality=11,
        pose_codec=PACKER.POSE_QP1_CODEC,
        payload_member_name=PACKER.SHORT_PAYLOAD_MEMBER_NAME,
        payload_format=PACKER.PAYLOAD_FORMAT_RPK1_JSON,
    )

    frontier_bytes = frontier_archive.stat().st_size
    archive_bytes = archive_path.stat().st_size
    record = {
        "schema": "c067_multimask_reconciler_candidate_record_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_byte_screen_archive_candidate_until_exact_cuda",
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "target_body_bytes": None if target_body_bytes is None else int(target_body_bytes),
        "target_extra_runs": None if target_extra_runs is None else int(target_extra_runs),
        "frontier_archive_bytes": int(frontier_bytes),
        "archive": {
            "path": str(archive_path),
            "bytes": int(archive_bytes),
            "sha256": _sha256_file(archive_path),
            "delta_bytes_vs_frontier": int(archive_bytes - frontier_bytes),
            "formula_only_rate_delta_vs_frontier": round(
                25.0 * float(archive_bytes - frontier_bytes) / float(CMG3A.ORIGINAL_VIDEO_BYTES),
                12,
            ),
        },
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_archive.stat().st_size,
            "sha256": _sha256_file(source_archive),
        },
        "runtime_contract": {
            "mask_member": "masks.cmg3",
            "mask_mode": "nonzero_row_runs_topk_v1",
            "payload_member": "p",
            "sidecars_required": False,
            "zip_safe_single_member_archive": True,
        },
        "cmg3": {
            **cmg3_header,
            "payload_bytes": len(cmg3_payload),
            "payload_sha256": _sha256_bytes(cmg3_payload),
            "run_stats_vs_fused_input": run_stats,
        },
        "mask_metrics": {
            "fused_vs_source": _diff_metrics(fused, source),
            "reconstructed_vs_fused": _diff_metrics(recon, fused),
            "reconstructed_vs_source": _diff_metrics(recon, source),
            "fused_mask_sha256": fused_sha,
            "reconstructed_mask_sha256": recon_sha,
        },
        "packed_payload": packed_meta,
    }
    (candidate_dir / "candidate_manifest.json").write_bytes(_json_bytes(record))
    return record


def build_candidates(
    *,
    plan_json: Path,
    frontier_archive: Path,
    output_dir: Path,
    target_body_bytes: tuple[int, ...],
    target_extra_runs: tuple[int, ...] = (),
    policy_id: str | None = None,
    base_runs_per_row: int = 1,
    adaptive_max_runs_per_row: int = 8,
    compressor: str = "auto",
    body_search_mode: str = "coarse",
    force: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = _read_plan(plan_json.resolve())
    policy = select_policy(plan, policy_id=policy_id)
    source_path, candidate_paths = _input_paths_from_plan(plan)
    source = _load_mask_array(source_path)
    candidate_arrays = {name: _load_mask_array(path) for name, path in candidate_paths.items()}
    for name, array in candidate_arrays.items():
        if array.shape != source.shape:
            raise ReconcilerBuildError(f"candidate {name} shape {array.shape} differs from source {source.shape}")

    fused = materialize_fused_mask(plan=plan, policy=policy, source=source, candidates=candidate_arrays)
    fused_path = output_dir / f"{policy['policy_id']}.fused_mask_array.npy"
    np.save(fused_path, fused)

    frontier_archive = frontier_archive.resolve()
    frontier_members = CMG3A._extract_frontier_members(frontier_archive)  # noqa: SLF001
    candidate_specs = [(int(target), None) for target in target_body_bytes]
    candidate_specs.extend((None, int(target)) for target in target_extra_runs)
    if not candidate_specs:
        raise ReconcilerBuildError("at least one target body byte or extra-run candidate is required")

    records = [
        _build_one_archive(
            output_dir=output_dir,
            frontier_archive=frontier_archive,
            frontier_members=frontier_members,
            source=source,
            fused=fused,
            target_body_bytes=body_target,
            target_extra_runs=extra_target,
            base_runs_per_row=base_runs_per_row,
            adaptive_max_runs_per_row=adaptive_max_runs_per_row,
            compressor=compressor,
            body_search_mode=body_search_mode,
        )
        for body_target, extra_target in candidate_specs
    ]
    records.sort(
        key=lambda item: (
            item["archive"]["bytes"] >= item["frontier_archive_bytes"],
            item["mask_metrics"]["reconstructed_vs_source"]["disagreement_count"],
            item["archive"]["bytes"],
        )
    )
    best = records[0]
    summary = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_byte_screen_archive_candidate_until_exact_cuda",
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "plan_json": {
            "path": str(plan_json.resolve()),
            "sha256": _sha256_file(plan_json.resolve()),
            "schema_version": plan.get("schema_version"),
        },
        "frontier_archive": {
            "path": str(frontier_archive),
            "bytes": frontier_archive.stat().st_size,
            "sha256": _sha256_file(frontier_archive),
        },
        "selected_multimask_policy": {
            "policy_id": policy.get("policy_id"),
            "policy_family": policy.get("policy_family"),
            "rank": policy.get("rank"),
            "planner_estimated_charged_bytes": policy.get("estimated_charged_bytes"),
            "planner_rank_cost_proxy": policy.get("rank_cost_proxy"),
            "fusion_reconciliation_policy": policy.get("fusion_reconciliation_policy"),
        },
        "fused_mask_array": {
            "path": str(fused_path),
            "bytes": fused_path.stat().st_size,
            "sha256": _sha256_file(fused_path),
            "array_sha256": _mask_u8_sha256(fused),
            "typed_array_sha256": _array_sha256(fused),
            "shape": [int(v) for v in fused.shape],
            "dtype": str(fused.dtype),
        },
        "candidate_records": records,
        "best_candidate": best,
        "exact_eval_dispatch_recommended": bool(
            best["archive"]["bytes"] < frontier_archive.stat().st_size
            and best["runtime_contract"]["sidecars_required"] is False
        ),
        "dispatch_note": (
            "Recommended only as exact CUDA auth eval, not as a score claim, because the archive is "
            "byte-closed and smaller than the C067 frontier while using an existing masks.cmg3 runtime."
            if best["archive"]["bytes"] < frontier_archive.stat().st_size
            else "Do not dispatch: no candidate beat C067 bytes in the local byte screen."
        ),
    }
    (output_dir / "byte_screen_summary.json").write_bytes(_json_bytes(summary))
    return summary


def _parse_target_body_bytes(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("target body bytes must be a comma-separated list of positive integers")
    return values


def _parse_target_extra_runs(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("target extra runs must be a comma-separated list of nonnegative integers")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--frontier-archive", type=Path, default=DEFAULT_FRONTIER_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy-id", default=None)
    parser.add_argument(
        "--target-body-bytes",
        type=_parse_target_body_bytes,
        default=DEFAULT_TARGET_BODY_BYTES,
        help="Comma-separated CMG3 body byte targets to screen.",
    )
    parser.add_argument(
        "--target-extra-runs",
        type=_parse_target_extra_runs,
        default=DEFAULT_TARGET_EXTRA_RUNS,
        help="Comma-separated explicit CMG3 extra-run counts to screen; much faster than body search.",
    )
    parser.add_argument("--base-runs-per-row", type=int, default=1)
    parser.add_argument("--adaptive-max-runs-per-row", type=int, default=8)
    parser.add_argument("--compressor", choices=("auto", "bz2", "raw", "zlib", "lzma_xz"), default="auto")
    parser.add_argument("--body-search-mode", choices=("auto", "exhaustive", "coarse"), default="coarse")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = build_candidates(
            plan_json=args.plan_json,
            frontier_archive=args.frontier_archive,
            output_dir=args.output_dir,
            target_body_bytes=args.target_body_bytes,
            target_extra_runs=args.target_extra_runs,
            policy_id=args.policy_id,
            base_runs_per_row=args.base_runs_per_row,
            adaptive_max_runs_per_row=args.adaptive_max_runs_per_row,
            compressor=args.compressor,
            body_search_mode=args.body_search_mode,
            force=args.force,
        )
    except (FileNotFoundError, ReconcilerBuildError, ValueError) as exc:
        parser.exit(2, f"{Path(__file__).name}: error: {exc}\n")
    best = summary["best_candidate"]["archive"]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "best_archive": best,
                "score_claim": False,
                "exact_eval_dispatch_recommended": summary["exact_eval_dispatch_recommended"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
