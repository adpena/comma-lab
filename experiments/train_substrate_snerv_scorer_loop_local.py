#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SNeRV local scorer-loop decoder/QAT trainer harness.

This wraps the existing receiver-priced SNeRV scorer-loop implementation with
contest-lab storage preflight, launch custody, and durable JSON/Markdown
artifacts. It is local false-authority: useful for pair-robust optimization and
byte-accounted archive replay, never for score/rank/promotion claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.operator_storage_waterfall import (
    operator_storage_policy_payload,
    operator_storage_tier_cli_specs,
)
from comma_lab.storage_tiers import (
    DEFAULT_RESERVE_FREE_GB,
    StorageTierError,
    parse_storage_tier_specs,
    plan_experiment_storage,
    require_selected_storage,
)
from tac.repo_io import sha256_file, write_json
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    FALSE_AUTHORITY,
)
from tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat import (
    SCHEMA as SNERV_QAT_RESULT_SCHEMA,
)
from tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat import (
    run_snerv_scorer_loop_decoder_qat_smoke,
)

TRAINER_SCHEMA = "snerv_scorer_loop_qat_local_trainer.v1"
TRAINER_AUTHORITY = "false_authority_macos_cpu_snerv_scorer_loop_no_contest_score_claim"
DEFAULT_WORKLOAD_SUBDIR = "snerv_scorer_loop_qat_local"
RESEARCH_DIR = REPO_ROOT / ".omx" / "research"


def _score_loop_main(args: argparse.Namespace) -> int:
    output_dir, storage_payload = _resolve_output_dir(args)
    launch_path = output_dir / "snerv_scorer_loop_qat_launch_preflight.json"
    launch = {
        "schema": TRAINER_SCHEMA,
        "authority": TRAINER_AUTHORITY,
        "family": "snerv",
        "output_dir": output_dir.as_posix(),
        "storage_preflight": storage_payload,
        "command": sys.argv,
        "score_loop_kwargs": _score_loop_kwargs_from_args(args),
        "blockers": [
            "local_snerv_scorer_loop_only_not_full600_authority",
            "paired_contest_cpu_cuda_pass_missing",
        ],
        **FALSE_AUTHORITY,
    }
    write_json(launch_path, launch)

    result = run_snerv_scorer_loop_decoder_qat_smoke(
        **_score_loop_kwargs_from_args(args)
    )
    report = _build_report(
        result=result,
        args=args,
        output_dir=output_dir,
        storage_payload=storage_payload,
        launch_path=launch_path,
    )
    result_path = output_dir / "snerv_scorer_loop_qat_result.json"
    report["result_path"] = result_path.as_posix()
    write_json(result_path, report)
    report["result_sha256"] = sha256_file(result_path)
    write_json(result_path, report)
    md_path = output_dir / "snerv_scorer_loop_qat_result.md"
    report["markdown_report_path"] = md_path.as_posix()
    md_path.write_text(render_snerv_scorer_loop_local_markdown(report), encoding="utf-8")
    write_json(result_path, report)

    research_json = _research_json_path(args)
    research_json.parent.mkdir(parents=True, exist_ok=True)
    report["research_json_path"] = research_json.as_posix()
    write_json(research_json, report)
    research_md = _research_md_path(args, research_json)
    if research_md is not None:
        report["research_markdown_path"] = research_md.as_posix()
        research_md.parent.mkdir(parents=True, exist_ok=True)
        research_md.write_text(
            render_snerv_scorer_loop_local_markdown(report),
            encoding="utf-8",
        )
        write_json(research_json, report)

    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--score-loop", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-local-output-dir", action="store_true")
    parser.add_argument("--storage-workload-subdir", default=DEFAULT_WORKLOAD_SUBDIR)
    parser.add_argument("--storage-expected-bytes", type=int, default=4 * 1024**3)
    parser.add_argument(
        "--storage-reserve-free-gb",
        type=float,
        default=DEFAULT_RESERVE_FREE_GB,
    )
    parser.add_argument("--research-json", type=Path, default=None)
    parser.add_argument("--research-md", type=Path, default=None)
    parser.add_argument("--n-pairs", type=int, default=1)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--wavelet", default="db2")
    parser.add_argument("--target-bits-per-coeff", type=float, default=5.0)
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--upstream-dir", default="upstream")
    parser.add_argument("--video-path", default="upstream/videos/0.mkv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--step-map-bins", type=int, default=16)
    parser.add_argument("--qat-bits", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=2)
    parser.add_argument(
        "--search-mode",
        choices=(
            "random_signed",
            "top_weight_coordinate",
            "learned_random_subspace",
            "nes_pair_robust",
        ),
        default="random_signed",
    )
    parser.add_argument("--perturb-scale", type=float, default=0.02)
    parser.add_argument("--byte-pressure-multiplier", type=float, default=1.0)
    parser.add_argument("--max-archive-byte-growth", type=int, default=None)
    parser.add_argument("--pose-slack", type=float, default=0.0)
    parser.add_argument("--seg-slack", type=float, default=0.0)
    parser.add_argument(
        "--pair-guard-min-score-improved-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--pair-guard-max-pose-worsened-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def _score_loop_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_pairs": int(args.n_pairs),
        "levels": int(args.levels),
        "wavelet": str(args.wavelet),
        "target_bits_per_coeff": float(args.target_bits_per_coeff),
        "pair_stride": int(args.pair_stride),
        "start_pair": int(args.start_pair),
        "upstream_dir": str(args.upstream_dir),
        "video_path": str(args.video_path),
        "device": str(args.device),
        "step_map_bins": int(args.step_map_bins),
        "qat_bits": int(args.qat_bits),
        "max_trials": int(args.max_trials),
        "search_mode": str(args.search_mode),
        "perturb_scale": float(args.perturb_scale),
        "byte_pressure_multiplier": float(args.byte_pressure_multiplier),
        "max_archive_byte_growth": (
            None
            if args.max_archive_byte_growth is None
            else int(args.max_archive_byte_growth)
        ),
        "pose_slack": float(args.pose_slack),
        "seg_slack": float(args.seg_slack),
        "pair_guard_min_score_improved_fraction": float(
            args.pair_guard_min_score_improved_fraction
        ),
        "pair_guard_max_pose_worsened_fraction": float(
            args.pair_guard_max_pose_worsened_fraction
        ),
        "seed": int(args.seed),
    }


def _build_report(
    *,
    result: Any,
    args: argparse.Namespace,
    output_dir: Path,
    storage_payload: dict[str, Any],
    launch_path: Path,
) -> dict[str, Any]:
    payload = result.as_jsonable()
    result_blockers = list(payload.get("blockers") or ())
    blockers = [
        *result_blockers,
        "full_600_pair_receiver_proof_missing",
        "paired_contest_cpu_cuda_pass_missing",
        "official_snerv_mfu_hfr_tub_parity_not_proven",
    ]
    return {
        "schema": TRAINER_SCHEMA,
        "authority": TRAINER_AUTHORITY,
        "axis_tag": payload.get("axis_tag", "[macOS-CPU advisory]"),
        "family": "snerv",
        "source_result_schema": payload.get("schema") or SNERV_QAT_RESULT_SCHEMA,
        "output_dir": output_dir.as_posix(),
        "launch_preflight_path": launch_path.as_posix(),
        "storage_preflight": storage_payload,
        "score_loop_kwargs": _score_loop_kwargs_from_args(args),
        "n_pairs": payload.get("n_pairs"),
        "levels": payload.get("levels"),
        "wavelet": payload.get("wavelet"),
        "qat_bits": payload.get("qat_bits"),
        "search_mode": payload.get("search_mode"),
        "scorer_loop_evaluations": payload.get("scorer_loop_evaluations"),
        "baseline_archive_bytes": _nested(payload, "baseline", "archive_bytes"),
        "best_archive_bytes": _nested(payload, "best", "archive_bytes"),
        "baseline_score_linf": _nested(payload, "baseline", "score_linf"),
        "best_score_linf": _nested(payload, "best", "score_linf"),
        "accepted_improvement": bool(payload.get("accepted_improvement")),
        "ready_for_pose_guard_gate": bool(payload.get("ready_for_pose_guard_gate")),
        "receiver_contract_satisfied": bool(payload.get("receiver_contract_satisfied")),
        "result": payload,
        "blockers": list(dict.fromkeys(blockers)),
        "forbidden_next_actions": [
            "claim_score_from_snerv_local_scorer_loop",
            "dispatch_exact_eval_from_snerv_local_scorer_loop",
            "promote_without_full600_receiver_proof",
        ],
        **FALSE_AUTHORITY,
    }


def render_snerv_scorer_loop_local_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SNeRV scorer-loop QAT local trainer",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        f"Axis: `{report.get('axis_tag')}`",
        f"Pairs: `{report.get('n_pairs')}`",
        f"Search mode: `{report.get('search_mode')}`",
        f"Evaluations: `{report.get('scorer_loop_evaluations')}`",
        f"Baseline score: `{report.get('baseline_score_linf')}`",
        f"Best score: `{report.get('best_score_linf')}`",
        f"Accepted improvement: `{report.get('accepted_improvement')}`",
        f"Receiver contract satisfied: `{report.get('receiver_contract_satisfied')}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in report.get("blockers") or ():
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def _resolve_output_dir(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.output_dir is not None:
        output = args.output_dir.expanduser()
        if not output.is_absolute():
            output = REPO_ROOT / output
        output = output.resolve(strict=False)
        if _looks_local(output) and not bool(args.allow_local_output_dir):
            raise StorageTierError(
                "snerv_scorer_loop_qat_output_storage_preflight_failed: "
                "local_disk_tier_disabled"
            )
        output.mkdir(parents=True, exist_ok=True)
        return output, {
            "schema": "snerv_scorer_loop_qat_explicit_output_preflight.v1",
            "selected_workload_root": output.as_posix(),
            "explicit_output_dir": True,
            "local_output_explicitly_allowed": bool(args.allow_local_output_dir),
            "operator_storage_policy": operator_storage_policy_payload(),
            "blockers": [],
            **FALSE_AUTHORITY,
        }

    tiers = parse_storage_tier_specs(
        operator_storage_tier_cli_specs(()),
        repo_root=REPO_ROOT,
        reserve_free_gb=float(args.storage_reserve_free_gb),
        allow_local_disk=False,
    )
    subdir = (
        f"{str(args.storage_workload_subdir).strip('/')}/"
        f"{int(args.n_pairs)}pairs_{args.search_mode!s}"
    )
    plan = plan_experiment_storage(
        tiers,
        workload_subdir=subdir,
        requested_bytes=int(args.storage_expected_bytes),
        min_free_bytes=0,
        create=True,
        probe_writable=True,
    )
    output = require_selected_storage(plan)
    payload = plan.to_dict()
    payload["operator_storage_policy"] = operator_storage_policy_payload()
    payload["selected_workload_root"] = output.as_posix()
    payload.update(FALSE_AUTHORITY)
    return output, payload


def _research_json_path(args: argparse.Namespace) -> Path:
    if args.research_json is not None:
        return _abs_repo_path(args.research_json)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return RESEARCH_DIR / f"snerv_scorer_loop_qat_local_trainer_{stamp}.json"


def _research_md_path(args: argparse.Namespace, research_json: Path) -> Path | None:
    if args.research_md is not None:
        return _abs_repo_path(args.research_md)
    return research_json.with_suffix(".md")


def _abs_repo_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)
    return (REPO_ROOT / expanded).resolve(strict=False)


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _looks_local(path: Path) -> bool:
    return not path.resolve(strict=False).as_posix().startswith("/Volumes/")


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": TRAINER_SCHEMA,
        "research_json_path": report.get("research_json_path"),
        "output_dir": report.get("output_dir"),
        "n_pairs": report.get("n_pairs"),
        "scorer_loop_evaluations": report.get("scorer_loop_evaluations"),
        "baseline_score_linf": report.get("baseline_score_linf"),
        "best_score_linf": report.get("best_score_linf"),
        "accepted_improvement": report.get("accepted_improvement"),
        "score_claim": report.get("score_claim"),
        "ready_for_exact_eval_dispatch": report.get("ready_for_exact_eval_dispatch"),
        "blockers": report.get("blockers"),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.score_loop:
        return _score_loop_main(args)
    raise AssertionError("argparse should require a mode")


__all__ = [
    "TRAINER_SCHEMA",
    "_build_parser",
    "_build_report",
    "_resolve_output_dir",
    "_score_loop_kwargs_from_args",
    "main",
    "render_snerv_scorer_loop_local_markdown",
]


if __name__ == "__main__":
    raise SystemExit(main())
