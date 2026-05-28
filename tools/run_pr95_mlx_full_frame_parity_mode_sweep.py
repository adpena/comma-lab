#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sweep PR95 MLX full-frame parity modes against the public runtime."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    FALSE_AUTHORITY,
    PR95_MLX_CONV2D_ACCUMULATION_MODES,
    PR95_MLX_CONV2D_ACCUMULATION_OVERRIDE_PRESETS,
    pr95_mlx_conv2d_accumulation_overrides_from_preset,
    pr95_mlx_conv2d_scope_search_candidates,
)
from tac.repo_io import write_json_artifact  # noqa: E402

DEFAULT_INFLATE_SH = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/inflate.sh"
)
DEFAULT_CANDIDATES = (
    "optimized:none",
    "fixed_fp32:none",
    "kahan_fp32:none",
    "optimized:blocks02_kahan_fp32",
)
CPU_EXTRA_CANDIDATES = ("fixed_fp64:none",)
SCHEMA = "pr95_hnerv_mlx_full_frame_parity_mode_sweep.v1"
PLAN_SCHEMA = "pr95_hnerv_mlx_full_frame_parity_mode_sweep_plan.v1"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def _override_items(overrides: dict[str, str]) -> list[str]:
    return [f"{name}={mode}" for name, mode in sorted(overrides.items())]


def _append_candidate(
    rows: list[dict[str, Any]],
    seen: set[tuple[str, tuple[tuple[str, str], ...]]],
    *,
    candidate_id: str,
    mode: str,
    preset: str,
    overrides: dict[str, str],
    source: str,
    scope_search_kind: str | None = None,
) -> None:
    key = (mode, tuple(sorted(overrides.items())))
    if key in seen:
        return
    seen.add(key)
    rows.append(
        {
            "candidate_id": candidate_id,
            "conv2d_accumulation_mode": mode,
            "conv2d_override_preset": preset,
            "conv2d_accumulation_overrides": overrides,
            "conv2d_override_items": _override_items(overrides),
            "candidate_source": source,
            "scope_search_kind": scope_search_kind,
        }
    )


def _candidate_specs(
    raw_specs: list[str] | None,
    *,
    mlx_device: str,
    include_scope_search_candidates: bool,
    scope_block_count: int,
    scope_include_presets: bool,
    scope_include_single_blocks: bool,
    scope_include_prefix_blocks: bool,
    scope_include_individual_modules: bool,
    scope_include_pair_blocks: bool,
    scope_search_candidate_limit: int,
) -> list[dict[str, Any]]:
    specs = list(raw_specs or DEFAULT_CANDIDATES)
    if raw_specs is None and mlx_device == "cpu":
        specs.extend(CPU_EXTRA_CANDIDATES)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for raw in specs:
        parts = raw.split(":")
        if len(parts) != 2:
            raise ValueError(f"candidate must be '<mode>:<preset>', got {raw!r}")
        mode, preset = (part.strip() for part in parts)
        if mode not in PR95_MLX_CONV2D_ACCUMULATION_MODES:
            raise ValueError(f"unknown Conv2d accumulation mode {mode!r}")
        if preset not in PR95_MLX_CONV2D_ACCUMULATION_OVERRIDE_PRESETS:
            raise ValueError(f"unknown Conv2d override preset {preset!r}")
        if mlx_device == "gpu" and mode == "fixed_fp64":
            raise ValueError("fixed_fp64 is unsupported on MLX GPU")
        _append_candidate(
            rows,
            seen,
            candidate_id=f"{_safe_id(mode)}__{_safe_id(preset)}",
            mode=mode,
            preset=preset,
            overrides=pr95_mlx_conv2d_accumulation_overrides_from_preset(preset),
            source="explicit",
        )

    if include_scope_search_candidates:
        if scope_block_count < 1:
            raise ValueError("--scope-block-count must be >= 1")
        scope_candidates = pr95_mlx_conv2d_scope_search_candidates(
            block_count=scope_block_count,
            include_presets=scope_include_presets,
            include_single_blocks=scope_include_single_blocks,
            include_prefix_blocks=scope_include_prefix_blocks,
            include_individual_modules=scope_include_individual_modules,
            include_pair_blocks=scope_include_pair_blocks,
        )
        if scope_search_candidate_limit > 0:
            scope_candidates = scope_candidates[:scope_search_candidate_limit]
        for candidate in scope_candidates:
            _append_candidate(
                rows,
                seen,
                candidate_id=(
                    f"scope__{_safe_id(str(candidate['kind']))}__"
                    f"{_safe_id(str(candidate['candidate_id']))}"
                ),
                mode="optimized",
                preset="none",
                overrides=dict(candidate["conv2d_accumulation_overrides"]),
                source="canonical_scope_search",
                scope_search_kind=str(candidate["kind"]),
            )
    return rows


def _command_for_candidate(
    candidate: dict[str, Any],
    *,
    archive_zip: Path,
    inflate_sh: Path,
    output_json: Path,
    work_dir: Path,
    mlx_device: str,
    timeout_seconds: float,
    max_output_bytes: int,
    max_mismatch_samples: int,
    allow_large_output: bool,
    skip_torch_direct_reference: bool,
    allow_existing_output_dir: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "prove_pr95_public_archive_full_frame_parity.py"),
        "--archive-zip",
        str(archive_zip),
        "--inflate-sh",
        str(inflate_sh),
        "--output-json",
        str(output_json),
        "--work-dir",
        str(work_dir),
        "--mlx-device",
        mlx_device,
        "--conv2d-accumulation-mode",
        candidate["conv2d_accumulation_mode"],
        "--conv2d-override-preset",
        candidate["conv2d_override_preset"],
        "--timeout-seconds",
        str(timeout_seconds),
        "--max-output-bytes",
        str(max_output_bytes),
        "--max-mismatch-samples",
        str(max_mismatch_samples),
    ]
    for item in candidate.get("conv2d_override_items", []):
        command.extend(["--conv2d-override", str(item)])
    if allow_large_output:
        command.append("--allow-large-output")
    if skip_torch_direct_reference:
        command.append("--skip-torch-direct-reference")
    if allow_existing_output_dir and output_json.exists():
        command.extend(
            [
                "--allow-overwrite",
                "--expected-existing-sha256",
                _sha256_file(output_json),
            ]
        )
    return command


def _failure_row(
    candidate: dict[str, Any],
    *,
    proof_path: Path,
    returncode: int | None,
    elapsed_seconds: float,
    verdict: str,
    stderr_tail: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "conv2d_accumulation_mode": candidate["conv2d_accumulation_mode"],
        "conv2d_override_preset": candidate["conv2d_override_preset"],
        "conv2d_accumulation_overrides": candidate.get(
            "conv2d_accumulation_overrides", {}
        ),
        "conv2d_override_items": candidate.get("conv2d_override_items", []),
        "candidate_source": candidate.get("candidate_source"),
        "scope_search_kind": candidate.get("scope_search_kind"),
        "proof_path": _rel(proof_path),
        "returncode": returncode,
        "elapsed_seconds": float(elapsed_seconds),
        "byte_exact": False,
        "changed_byte_count": 10**18,
        "max_abs_uint8": 10**9,
        "mean_abs_uint8": 10**9,
        "drift_localization_verdict": verdict,
        "stderr_tail": stderr_tail[-2000:],
        **FALSE_AUTHORITY,
    }


def _row_from_proof(
    candidate: dict[str, Any],
    *,
    proof_path: Path,
    returncode: int,
    elapsed_seconds: float,
    stderr_tail: str,
) -> dict[str, Any]:
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    diff = proof.get("diff", {})
    torch_ref = proof.get("torch_direct_reference", {})
    return {
        "candidate_id": candidate["candidate_id"],
        "conv2d_accumulation_mode": candidate["conv2d_accumulation_mode"],
        "conv2d_override_preset": candidate["conv2d_override_preset"],
        "conv2d_accumulation_overrides": proof.get(
            "conv2d_accumulation_overrides",
            candidate.get("conv2d_accumulation_overrides", {}),
        ),
        "conv2d_override_items": candidate.get("conv2d_override_items", []),
        "candidate_source": candidate.get("candidate_source"),
        "scope_search_kind": candidate.get("scope_search_kind"),
        "proof_path": _rel(proof_path),
        "proof_sha256": _sha256_file(proof_path),
        "returncode": int(returncode),
        "elapsed_seconds": float(elapsed_seconds),
        "byte_exact": diff.get("byte_exact") is True,
        "changed_byte_count": int(diff.get("changed_byte_count", 10**18)),
        "max_abs_uint8": int(diff.get("max_abs_uint8", 10**9)),
        "mean_abs_uint8": float(diff.get("mean_abs_uint8", 10**9)),
        "drift_localization_verdict": proof.get("drift_localization_verdict"),
        "torch_direct_reference_byte_exact": (
            torch_ref.get("byte_exact_with_public_inflate") is True
        ),
        "first_mismatch": diff.get("first_mismatch"),
        "per_frame_changed_byte_count": diff.get("per_frame_changed_byte_count"),
        "per_channel_changed_byte_count": diff.get("per_channel_changed_byte_count"),
        "stderr_tail": stderr_tail[-2000:],
        **FALSE_AUTHORITY,
    }


def _rank_key(row: dict[str, Any]) -> tuple[int, int, float, float, str]:
    return (
        0 if row.get("byte_exact") is True else 1,
        int(row.get("changed_byte_count", 10**18)),
        float(row.get("mean_abs_uint8", 10**9)),
        float(row.get("elapsed_seconds", 10**9)),
        str(row.get("candidate_id", "")),
    )


def _build_plan(args: argparse.Namespace, *, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    candidate_commands = []
    for candidate in candidates:
        proof_path = output_dir / f"{candidate['candidate_id']}.json"
        work_dir = output_dir / f"{candidate['candidate_id']}_work"
        candidate_commands.append(
            {
                **candidate,
                "proof_path": _rel(proof_path),
                "work_dir": _rel(work_dir),
                "python_command_args": _command_for_candidate(
                    candidate,
                    archive_zip=args.archive_zip.resolve(),
                    inflate_sh=args.inflate_sh.resolve(),
                    output_json=proof_path,
                    work_dir=work_dir,
                    mlx_device=args.mlx_device,
                    timeout_seconds=args.timeout_seconds,
                    max_output_bytes=args.max_output_bytes,
                    max_mismatch_samples=args.max_mismatch_samples,
                    allow_large_output=args.allow_large_output,
                    skip_torch_direct_reference=args.skip_torch_direct_reference,
                    allow_existing_output_dir=args.allow_existing_output_dir,
                ),
            }
        )
    return {
        "schema": PLAN_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "lane_id": "lane_pr95_hnerv_mlx_reproduction",
        "archive_zip": _rel(args.archive_zip.resolve()),
        "inflate_sh": _rel(args.inflate_sh.resolve()),
        "output_dir": _rel(output_dir),
        "expected_output": _rel(output_dir / "full_frame_parity_mode_sweep.json"),
        "mlx_device": args.mlx_device,
        "jobs": args.jobs,
        "candidate_count": len(candidates),
        "include_scope_search_candidates": args.include_scope_search_candidates,
        "scope_block_count": args.scope_block_count,
        "scope_include_individual_modules": args.scope_include_individual_modules,
        "scope_include_pair_blocks": args.scope_include_pair_blocks,
        "candidates": candidates,
        "candidate_commands": candidate_commands,
        "recommended_execution": {
            "tool": "tools/run_pr95_mlx_full_frame_parity_mode_sweep.py",
            "resource_kind": "local_mlx" if args.mlx_device == "gpu" else "local_cpu",
            "python_command_args": [
                ".venv/bin/python",
                "tools/run_pr95_mlx_full_frame_parity_mode_sweep.py",
                "--archive-zip",
                _rel(args.archive_zip.resolve()),
                "--output-dir",
                _rel(output_dir),
                "--mlx-device",
                args.mlx_device,
                "--jobs",
                str(args.jobs),
                "--allow-existing-output-dir",
            ],
            **FALSE_AUTHORITY,
        },
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": [
                "local_pr95_mlx_full_frame_parity_mode_sweep_is_not_score_authority",
                "requires_exact_cpu_cuda_auth_eval_before_score_claim",
            ],
        },
        **FALSE_AUTHORITY,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inflate-sh", type=Path, default=DEFAULT_INFLATE_SH)
    parser.add_argument("--candidate", action="append", help="<mode>:<preset>")
    parser.add_argument("--mlx-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-output-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-mismatch-samples", type=int, default=16)
    parser.add_argument("--allow-large-output", action="store_true")
    parser.add_argument("--skip-torch-direct-reference", action="store_true")
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--include-scope-search-candidates",
        action="store_true",
        help="Append canonical per-block/prefix Conv2d override candidates.",
    )
    parser.add_argument("--scope-block-count", type=int, default=6)
    parser.add_argument("--scope-no-presets", action="store_true")
    parser.add_argument("--scope-no-single-blocks", action="store_true")
    parser.add_argument("--scope-no-prefix-blocks", action="store_true")
    parser.add_argument("--scope-include-individual-modules", action="store_true")
    parser.add_argument("--scope-include-pair-blocks", action="store_true")
    parser.add_argument(
        "--scope-search-candidate-limit",
        type=int,
        default=0,
        help="Optional cap after canonical scope candidate generation; 0 means no cap.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Bounded concurrent child parity proofs. Use >1 only for small archives.",
    )
    return parser


def _run_candidate(
    candidate: dict[str, Any],
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    proof_path = output_dir / f"{candidate['candidate_id']}.json"
    work_dir = output_dir / f"{candidate['candidate_id']}_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    command = _command_for_candidate(
        candidate,
        archive_zip=args.archive_zip.resolve(),
        inflate_sh=args.inflate_sh.resolve(),
        output_json=proof_path,
        work_dir=work_dir,
        mlx_device=args.mlx_device,
        timeout_seconds=args.timeout_seconds,
        max_output_bytes=args.max_output_bytes,
        max_mismatch_samples=args.max_mismatch_samples,
        allow_large_output=args.allow_large_output,
        skip_torch_direct_reference=args.skip_torch_direct_reference,
        allow_existing_output_dir=args.allow_existing_output_dir,
    )
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=args.timeout_seconds + 30,
        )
    except subprocess.TimeoutExpired as exc:
        return _failure_row(
            candidate,
            proof_path=proof_path,
            returncode=None,
            elapsed_seconds=time.perf_counter() - started,
            verdict="child_timeout_expired",
            stderr_tail=f"{exc.stderr or ''}\n{exc.stdout or ''}",
        )

    elapsed = time.perf_counter() - started
    if not proof_path.exists():
        return _failure_row(
            candidate,
            proof_path=proof_path,
            returncode=result.returncode,
            elapsed_seconds=elapsed,
            verdict="proof_not_written",
            stderr_tail=result.stderr or result.stdout,
        )
    return _row_from_proof(
        candidate,
        proof_path=proof_path,
        returncode=result.returncode,
        elapsed_seconds=elapsed,
        stderr_tail=result.stderr or result.stdout,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_mismatch_samples < 0:
        raise SystemExit("--max-mismatch-samples must be >= 0")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    candidates = _candidate_specs(
        args.candidate,
        mlx_device=args.mlx_device,
        include_scope_search_candidates=args.include_scope_search_candidates,
        scope_block_count=args.scope_block_count,
        scope_include_presets=not args.scope_no_presets,
        scope_include_single_blocks=not args.scope_no_single_blocks,
        scope_include_prefix_blocks=not args.scope_no_prefix_blocks,
        scope_include_individual_modules=args.scope_include_individual_modules,
        scope_include_pair_blocks=args.scope_include_pair_blocks,
        scope_search_candidate_limit=args.scope_search_candidate_limit,
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.allow_existing_output_dir:
        raise SystemExit(
            f"output directory already exists: {_rel(output_dir)} "
            "(pass --allow-existing-output-dir to append/overwrite manifests)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = _build_plan(args, candidates=candidates)
    write_json_artifact(
        output_dir / "plan.json",
        plan,
        allow_overwrite=args.allow_existing_output_dir
        and (output_dir / "plan.json").exists(),
        expected_existing_sha256=_sha256_file(output_dir / "plan.json")
        if args.allow_existing_output_dir and (output_dir / "plan.json").exists()
        else None,
    )
    if args.plan_only:
        print(
            json.dumps(
                {
                    "ok": True,
                    "schema": "pr95_hnerv_mlx_full_frame_parity_mode_sweep_plan_summary.v1",
                    "plan": _rel(output_dir / "plan.json"),
                    "candidate_count": len(candidates),
                    **FALSE_AUTHORITY,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    rows: list[dict[str, Any]] = []
    max_workers = min(args.jobs, len(candidates))
    if max_workers == 1:
        rows = [
            _run_candidate(candidate, args=args, output_dir=output_dir)
            for candidate in candidates
        ]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_run_candidate, candidate, args=args, output_dir=output_dir)
                for candidate in candidates
            ]
            rows = [future.result() for future in concurrent.futures.as_completed(futures)]

    ranked = sorted(rows, key=_rank_key)
    non_exact = [row for row in ranked if row.get("byte_exact") is not True]
    summary = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "lane_id": "lane_pr95_hnerv_mlx_reproduction",
        "archive_zip": _rel(args.archive_zip.resolve()),
        "archive_sha256": _sha256_file(args.archive_zip.resolve()),
        "inflate_sh": _rel(args.inflate_sh.resolve()),
        "mlx_device": args.mlx_device,
        "jobs": args.jobs,
        "effective_jobs": max_workers,
        "candidate_count": len(rows),
        "best_candidate": ranked[0] if ranked else None,
        "best_non_exact_candidate": non_exact[0] if non_exact else None,
        "byte_exact_candidates": [row for row in ranked if row.get("byte_exact") is True],
        "rows": ranked,
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": [
                "local_pr95_mlx_full_frame_parity_mode_sweep_is_not_score_authority",
                "requires_exact_cpu_cuda_auth_eval_before_score_claim",
            ],
        },
        **FALSE_AUTHORITY,
    }
    summary_path = output_dir / "full_frame_parity_mode_sweep.json"
    write_json_artifact(
        summary_path,
        summary,
        allow_overwrite=args.allow_existing_output_dir and summary_path.exists(),
        expected_existing_sha256=_sha256_file(summary_path)
        if args.allow_existing_output_dir and summary_path.exists()
        else None,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "summary": _rel(summary_path),
                "best_candidate": summary["best_candidate"],
                **FALSE_AUTHORITY,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
