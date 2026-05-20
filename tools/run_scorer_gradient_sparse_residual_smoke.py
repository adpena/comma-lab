#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a charged sparse residual candidate from scorer-gradient pixels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.inflate_postprocess_surface import RawVideoShape  # noqa: E402
from tac.optimization.scorer_gradient_sparse_residual import (  # noqa: E402
    GradientAlignedSelection,
    ScorerGradientSparseConfig,
    build_plan_from_gradient_selection,
    compute_pair_component_distortions,
    compute_pair_scorer_gradient,
    local_pair_eval_worse_or_null,
    pair_component_delta,
    select_gradient_aligned_residuals,
)
from tac.optimization.sparse_residual_oracle import (  # noqa: E402
    authority_payload,
    sha256_file,
    write_charge_proxy_archive,
    write_sparse_residual_candidate,
)
from tac.scorer import load_differentiable_scorers  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe[:96] if safe else "unnamed"


def _run_advisory(
    *,
    raw: Path,
    archive: Path,
    output_dir: Path,
    axis_label: str,
    batch_size: int,
    num_threads: int,
    timeout: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_raw_advisory_eval.py"),
        "--raw",
        str(raw),
        "--archive",
        str(archive),
        "--output-dir",
        str(output_dir),
        "--axis-label",
        axis_label,
        "--batch-size",
        str(batch_size),
        "--num-threads",
        str(num_threads),
        "--timeout",
        str(timeout),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "advisory_wrapper.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "advisory_wrapper.stderr.log").write_text(proc.stderr, encoding="utf-8")
    payload_path = output_dir / "raw_advisory_eval.json"
    if payload_path.is_file():
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    else:
        payload = {"returncode": proc.returncode, "blockers": ["raw_advisory_eval_json_missing"]}
    payload["wrapper_returncode"] = proc.returncode
    payload["wrapper_cmd"] = cmd
    return payload


def _load_pair(raw: np.memmap, pair_index: int) -> np.ndarray:
    frame0 = pair_index * 2
    return np.asarray(raw[frame0 : frame0 + 2], dtype=np.uint8)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    shape = RawVideoShape(args.frames, args.height, args.width, args.channels)
    config = ScorerGradientSparseConfig(
        top_k_pixels=args.top_k_pixels,
        max_abs_delta=args.max_abs_delta,
        component=args.component,
        quantize_bits=args.quantize_bits,
        compression=args.compression,
        rate_cap_bytes=args.rate_cap_bytes,
    )
    config.validate()
    output_root = args.output_root.resolve()
    baseline_raw = args.baseline_raw.resolve()
    target_raw = args.target_raw.resolve()
    archive = args.archive.resolve()
    target_hash = sha256_file(target_raw)
    candidate_id = _candidate_slug(args.candidate_id or f"target_{target_hash[:12]}")
    pair_label = "_".join(str(i) for i in args.pair_index)
    candidate_dir = output_root / (
        f"{args.component}_pairs_{pair_label}_k{args.top_k_pixels}_d{args.max_abs_delta}_{candidate_id}"
    )
    if candidate_dir.exists() and any(candidate_dir.iterdir()) and not args.overwrite_candidate:
        raise SystemExit(
            f"candidate directory already exists and is non-empty: {candidate_dir}. "
            "Pass --overwrite-candidate only for an intentional rerun."
        )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    baseline = np.memmap(
        baseline_raw,
        dtype=np.uint8,
        mode="r",
        shape=(shape.frames, shape.height, shape.width, shape.channels),
    )
    target = np.memmap(
        target_raw,
        dtype=np.uint8,
        mode="r",
        shape=(shape.frames, shape.height, shape.width, shape.channels),
    )
    device = args.device
    posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)
    for scorer in (posenet, segnet):
        scorer.eval()
        for param in scorer.parameters():
            param.requires_grad_(False)

    selections = []
    pair_metrics: list[dict[str, Any]] = []
    per_pair_top_k = max(config.top_k_pixels, args.per_pair_candidate_pool)
    for pair_index in args.pair_index:
        baseline_pair = _load_pair(baseline, pair_index)
        target_pair = _load_pair(target, pair_index)
        grad, metrics = compute_pair_scorer_gradient(
            baseline_pair_hwc=baseline_pair,
            target_pair_hwc=target_pair,
            posenet=posenet,
            segnet=segnet,
            device=device,
            component=config.component,
            seg_ce_weight=args.seg_ce_weight,
        )
        residual = target_pair.astype(np.int16) - baseline_pair.astype(np.int16)
        selection = select_gradient_aligned_residuals(
            gradient=grad,
            residual=residual,
            shape=shape,
            frame_indices=[pair_index * 2, pair_index * 2 + 1],
            top_k_pixels=per_pair_top_k,
            max_abs_delta=config.max_abs_delta,
        )
        selections.append(selection)
        pair_metrics.append(
            {
                "pair_index": int(pair_index),
                "frame_indices": [int(pair_index * 2), int(pair_index * 2 + 1)],
                "gradient_metrics": metrics,
                "selection": selection.as_dict(),
            }
        )

    selected = [sel for sel in selections if sel.indices.size]
    if selected:
        indices = np.concatenate([sel.indices for sel in selected])
        values = np.concatenate([sel.values for sel in selected])
        gains = np.concatenate([sel.gains for sel in selected])
    else:
        indices = np.asarray([], dtype=np.uint32)
        values = np.zeros((0, shape.channels), dtype=np.int16)
        gains = np.asarray([], dtype=np.float32)
    if indices.size:
        order = np.argsort(-gains, kind="stable")[: config.top_k_pixels]
        indices = indices[order]
        values = values[order]
        gains = gains[order]
    plan = build_plan_from_gradient_selection(
        selection=GradientAlignedSelection(
            indices=indices,
            values=values,
            gains=gains,
            candidate_count=sum(sel.candidate_count for sel in selections),
            rejected_non_descent_count=sum(sel.rejected_non_descent_count for sel in selections),
        ),
        shape=shape,
        config=config,
    )

    correction_bin = candidate_dir / "scorer_gradient_sparse_residual.bin"
    candidate_raw = candidate_dir / "0.raw"
    charge_archive = candidate_dir / "archive_charge_proxy.zip"
    apply_result = write_sparse_residual_candidate(
        baseline_raw=baseline_raw,
        output_raw=candidate_raw,
        correction_bin=correction_bin,
        plan=plan,
        shape=shape,
    )
    candidate = np.memmap(
        candidate_raw,
        dtype=np.uint8,
        mode="r",
        shape=(shape.frames, shape.height, shape.width, shape.channels),
    )
    local_pair_evals = []
    for pair_index in args.pair_index:
        before = compute_pair_component_distortions(
            candidate_pair_hwc=_load_pair(baseline, pair_index),
            target_pair_hwc=_load_pair(target, pair_index),
            posenet=posenet,
            segnet=segnet,
            device=device,
        )
        after = compute_pair_component_distortions(
            candidate_pair_hwc=_load_pair(candidate, pair_index),
            target_pair_hwc=_load_pair(target, pair_index),
            posenet=posenet,
            segnet=segnet,
            device=device,
        )
        delta = pair_component_delta(before, after)
        local_pair_evals.append(
            {
                "pair_index": int(pair_index),
                "before": before,
                "after": after,
                "delta": delta,
                "worse_or_null": local_pair_eval_worse_or_null(delta, eps=args.local_veto_eps),
            }
        )
    del candidate
    archive_charge = write_charge_proxy_archive(
        baseline_archive=archive,
        correction_payload=plan.packed,
        output_archive=charge_archive,
    )
    row: dict[str, Any] = {
        "schema": "scorer_gradient_sparse_residual_candidate.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/run_scorer_gradient_sparse_residual_smoke.py",
        "inputs": {
            "baseline_raw": str(baseline_raw),
            "baseline_raw_sha256": sha256_file(baseline_raw),
            "target_raw": str(target_raw),
            "target_raw_sha256": target_hash,
            "archive": str(archive),
            "archive_sha256": sha256_file(archive),
            "shape": shape.as_dict(),
            "pair_indices": [int(i) for i in args.pair_index],
            "device": device,
            "candidate_id": candidate_id,
        },
        "config": config.as_dict(),
        "pair_metrics": pair_metrics,
        "local_pair_evals": local_pair_evals,
        "plan": plan.as_dict(),
        "candidate": apply_result.as_dict(),
        "archive_charge": archive_charge,
        "authority": authority_payload(),
    }
    local_veto = bool(args.skip_advisory_if_local_worse and all(item["worse_or_null"] for item in local_pair_evals))
    row["local_pair_veto"] = {
        "enabled": bool(args.skip_advisory_if_local_worse),
        "triggered": local_veto,
        "eps": float(args.local_veto_eps),
    }
    if local_veto:
        row["advisory_eval"] = {"skipped": True, "reason": "local_pair_component_veto"}
    elif args.run_advisory and apply_result.passed_visible_change:
        advisory = _run_advisory(
            raw=candidate_raw,
            archive=charge_archive,
            output_dir=candidate_dir / "advisory_eval",
            axis_label=args.axis_label,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            timeout=args.timeout,
        )
        row["advisory_eval"] = advisory
        if args.baseline_score is not None and advisory.get("canonical_score") is not None:
            row["delta_vs_baseline_score"] = float(advisory["canonical_score"]) - float(args.baseline_score)
    elif args.run_advisory:
        row["advisory_eval"] = {"skipped": True, "reason": "no_visible_raw_change"}

    advisory_payload = row.get("advisory_eval")
    advisory_terminal = (
        not args.run_advisory
        or isinstance(advisory_payload, dict)
        and (advisory_payload.get("returncode") == 0 or advisory_payload.get("skipped") is True)
    )
    if args.cleanup_candidate_raw and candidate_raw.is_file() and advisory_terminal:
        row["cleanup"] = {
            "candidate_raw_deleted": True,
            "candidate_raw_sha256_before_delete": sha256_file(candidate_raw),
            "candidate_raw_bytes_before_delete": candidate_raw.stat().st_size,
        }
        candidate_raw.unlink()
    else:
        row["cleanup"] = {"candidate_raw_deleted": False}
    _write_json(candidate_dir / "scorer_gradient_sparse_residual_manifest.json", row)

    summary = {
        "schema": "scorer_gradient_sparse_residual_smoke.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "candidate_manifest": str(candidate_dir / "scorer_gradient_sparse_residual_manifest.json"),
        "summary": {
            "component": args.component,
            "pair_indices": [int(i) for i in args.pair_index],
            "n_kept": int(plan.sparse["n_kept"]),
            "packed_bytes": plan.packed_bytes,
            "changed_pixel_count": apply_result.changed_pixel_count,
            "changed_byte_count": apply_result.changed_byte_count,
            "changed_frame_count": apply_result.changed_frame_count,
            "advisory_score": row.get("advisory_eval", {}).get("canonical_score")
            if isinstance(row.get("advisory_eval"), dict)
            else None,
            "delta_vs_baseline_score": row.get("delta_vs_baseline_score"),
            "score_claim": False,
        },
        "candidate": row,
        "authority": authority_payload(),
    }
    _write_json(args.output.resolve(), summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--target-raw", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-id",
        help=(
            "Optional stable slug appended to the candidate directory. "
            "Defaults to a target-raw SHA prefix so repeated runs with different "
            "decode semantics cannot overwrite each other."
        ),
    )
    parser.add_argument(
        "--overwrite-candidate",
        action="store_true",
        help="Allow overwriting a non-empty candidate directory for intentional reruns.",
    )
    parser.add_argument("--pair-index", type=int, action="append", required=True)
    parser.add_argument("--top-k-pixels", type=int, required=True)
    parser.add_argument("--per-pair-candidate-pool", type=int, default=2048)
    parser.add_argument("--max-abs-delta", type=int, default=1)
    parser.add_argument("--component", choices=["pose", "seg", "combined"], default="pose")
    parser.add_argument("--seg-ce-weight", type=float, default=0.05)
    parser.add_argument("--quantize-bits", type=int, choices=[4, 8, 16], default=8)
    parser.add_argument("--compression", choices=["zlib", "none"], default="zlib")
    parser.add_argument("--rate-cap-bytes", type=int)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--height", type=int, default=874)
    parser.add_argument("--width", type=int, default=1164)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--run-advisory", action="store_true")
    parser.add_argument("--skip-advisory-if-local-worse", action="store_true")
    parser.add_argument("--local-veto-eps", type=float, default=0.0)
    parser.add_argument("--baseline-score", type=float)
    parser.add_argument("--axis-label", default="[macOS-CPU advisory scorer-gradient-sparse-residual]")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cleanup-candidate-raw", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = run_smoke(parse_args(argv))
    print(json.dumps({"summary": payload["summary"], "score_claim": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
