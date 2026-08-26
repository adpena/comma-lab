#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# AUTOCAST_FP16_WAIVED:macOS-CPU-local-scorer-loop/QAT-harness-(TRAINER_AUTHORITY-declares-macos_cpu,-no-CUDA-path)-torch-CUDA-autocast-inapplicable
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
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    SNERV_ARCHIVE_SCHEMA,
    SNERV_ARCHIVE_SCHEMA_V2,
    SnervArchiveError,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat import (
    BYTE_GROWTH_ADMISSION_MODES,
    COMPONENT_GUARD_MODES,
    DEFAULT_DYNAMIC_RANGE_REPAIR_GAINS,
    run_snerv_scorer_loop_decoder_qat_smoke,
)
from tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat import (
    SCHEMA as SNERV_QAT_RESULT_SCHEMA,
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

    native_mlx_controls = _native_mlx_decoder_training_controls(args)
    if native_mlx_controls["blockers"]:
        refusal_path = output_dir / "snerv_scorer_loop_qat_launch_refusal.json"
        refusal = {
            "schema": "snerv_scorer_loop_qat_launch_refusal.v1",
            "authority": TRAINER_AUTHORITY,
            "family": "snerv",
            "output_dir": output_dir.as_posix(),
            "launch_preflight_path": launch_path.as_posix(),
            "launch_refusal_reason": (
                "native_mlx_decoder_training_requested_on_cpu_scorer_loop_harness"
            ),
            "native_mlx_decoder_training_controls": native_mlx_controls,
            "redirect_to": (
                "tools/run_compact_renderer_mlx_spine_runner.py "
                "--execute-family snerv --snerv-score-aware-long-training-epochs N"
            ),
            "blockers": list(native_mlx_controls["blockers"]),
            **FALSE_AUTHORITY,
        }
        write_json(refusal_path, refusal)
        print(
            json.dumps(
                {
                    "schema": refusal["schema"],
                    "report_path": refusal_path.as_posix(),
                    "blockers": refusal["blockers"],
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            )
        )
        return 2

    result = run_snerv_scorer_loop_decoder_qat_smoke(
        **_score_loop_kwargs_from_args(args)
    )
    best_packet_materialization = _materialize_best_packet(result, output_dir)
    report = _build_report(
        result=result,
        args=args,
        output_dir=output_dir,
        storage_payload=storage_payload,
        launch_path=launch_path,
        best_packet_materialization=best_packet_materialization,
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
    parser.add_argument("--snerv-spectra-preserving-adapter", action="store_true")
    parser.add_argument("--snerv-fc-dim", type=int, default=9)
    parser.add_argument("--snerv-emb-size", type=int, default=0)
    parser.add_argument("--snerv-patch-radius", type=int, default=1)
    parser.add_argument("--snerv-mfu-scales", default="1,2,4")
    parser.add_argument("--snerv-hfr-gain", type=float, default=0.0)
    parser.add_argument("--snerv-temporal-context", type=int, default=0)
    parser.add_argument(
        "--snerv-temporal-mode",
        choices=("delta", "official_haar_dwt1d_lowpass"),
        default="delta",
    )
    parser.add_argument(
        "--snerv-scorer-loop-lf-payload-codec",
        default="portfolio_auto",
    )
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
    parser.add_argument(
        "--section-value-pressure-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier for train-time SNAR1 optional-section neutralization "
            "pressure, matching tools/run_snerv_scorer_loop_decoder_qat_smoke.py."
        ),
    )
    parser.add_argument("--max-archive-byte-growth", type=int, default=None)
    parser.add_argument(
        "--byte-growth-admission-mode",
        choices=BYTE_GROWTH_ADMISSION_MODES,
        default="hard_cap",
        help=(
            "hard_cap rejects archive growth above --max-archive-byte-growth; "
            "rate_paid admits extra bytes only when the byte-pressured local "
            "objective still improves. False-authority training guard only."
        ),
    )
    parser.add_argument("--pose-slack", type=float, default=0.0)
    parser.add_argument("--seg-slack", type=float, default=0.0)
    parser.add_argument(
        "--component-guard-mode",
        choices=COMPONENT_GUARD_MODES,
        default="score_primary",
    )
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
    parser.add_argument(
        "--dynamic-range-repair-gains",
        default="",
        help=(
            "Comma-separated HF-decoder gain candidates to receiver-replay before "
            "the perturbation search, or 'auto' for the bounded default set."
        ),
    )
    parser.add_argument(
        "--snerv-native-mlx-decoder-train-steps",
        type=int,
        default=0,
        help=(
            "Native-MLX HF decoder training step count. Values >0 fail closed "
            "in this local CPU scorer-loop harness; run the compact spine "
            "SNeRV native MLX export/training path for real decoder training."
        ),
    )
    parser.add_argument(
        "--snerv-native-mlx-decoder-train-lr",
        type=float,
        default=1.0e-5,
        help="Recorded learning rate for --snerv-native-mlx-decoder-train-steps.",
    )
    parser.add_argument(
        "--snerv-native-mlx-decoder-train-ridge",
        type=float,
        default=1.0e-6,
        help="Recorded ridge pressure for --snerv-native-mlx-decoder-train-steps.",
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
        "snerv_spectra_preserving_adapter": bool(
            args.snerv_spectra_preserving_adapter
        ),
        "snerv_fc_dim": int(args.snerv_fc_dim),
        "snerv_emb_size": int(args.snerv_emb_size),
        "snerv_patch_radius": int(args.snerv_patch_radius),
        "snerv_mfu_scales": _parse_positive_int_csv(args.snerv_mfu_scales),
        "snerv_hfr_gain": float(args.snerv_hfr_gain),
        "snerv_temporal_context": int(args.snerv_temporal_context),
        "snerv_temporal_mode": str(args.snerv_temporal_mode),
        "lf_payload_codec": str(args.snerv_scorer_loop_lf_payload_codec),
        "qat_bits": int(args.qat_bits),
        "max_trials": int(args.max_trials),
        "search_mode": str(args.search_mode),
        "perturb_scale": float(args.perturb_scale),
        "byte_pressure_multiplier": float(args.byte_pressure_multiplier),
        "section_value_pressure_multiplier": float(
            args.section_value_pressure_multiplier
        ),
        "max_archive_byte_growth": (
            None
            if args.max_archive_byte_growth is None
            else int(args.max_archive_byte_growth)
        ),
        "byte_growth_admission_mode": str(args.byte_growth_admission_mode),
        "pose_slack": float(args.pose_slack),
        "seg_slack": float(args.seg_slack),
        "component_guard_mode": str(args.component_guard_mode),
        "pair_guard_min_score_improved_fraction": float(
            args.pair_guard_min_score_improved_fraction
        ),
        "pair_guard_max_pose_worsened_fraction": float(
            args.pair_guard_max_pose_worsened_fraction
        ),
        "dynamic_range_repair_gains": _parse_dynamic_range_repair_gains(
            args.dynamic_range_repair_gains
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
    best_packet_materialization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = result.as_jsonable()
    distortion_contract = _distortion_contract_from_result_payload(payload)
    best_packet_materialization = best_packet_materialization or {
        "schema": "snerv_scorer_loop_best_packet_materialization.v1",
        "materialized": False,
        "blockers": ["snerv_native_scorer_loop_best_packet_not_materialized"],
        **FALSE_AUTHORITY,
    }
    if best_packet_materialization.get("materialized"):
        reported_bytes = payload.get("best_packet_bytes")
        materialized_bytes = best_packet_materialization.get("best_packet_bytes")
        if reported_bytes is not None and int(reported_bytes) != int(materialized_bytes):
            raise ValueError(
                "snerv_best_packet_materialization_mismatch: "
                f"reported={reported_bytes} materialized={materialized_bytes}"
            )
    result_blockers = list(payload.get("blockers") or ())
    native_mlx_decoder_training_controls = _native_mlx_decoder_training_controls(args)
    blockers = [
        *result_blockers,
        *distortion_contract["blockers"],
        *(best_packet_materialization.get("blockers") or ()),
        *native_mlx_decoder_training_controls["blockers"],
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
        "native_mlx_decoder_training_controls": native_mlx_decoder_training_controls,
        "distortion_contract": distortion_contract,
        "n_pairs": payload.get("n_pairs"),
        "levels": payload.get("levels"),
        "wavelet": payload.get("wavelet"),
        "snerv_model_size_adapter": payload.get("snerv_model_size_adapter"),
        "snerv_mfu_scales": payload.get("snerv_mfu_scales"),
        "snerv_hfr_gain": payload.get("snerv_hfr_gain"),
        "snerv_temporal_context": payload.get("snerv_temporal_context"),
        "snerv_temporal_mode": payload.get("snerv_temporal_mode"),
        "decoder_feature_count": payload.get("decoder_feature_count"),
        "lf_payload_codec": payload.get("lf_payload_codec"),
        "qat_bits": payload.get("qat_bits"),
        "search_mode": payload.get("search_mode"),
        "component_guard_mode": payload.get("component_guard_mode"),
        "scorer_loop_evaluations": payload.get("scorer_loop_evaluations"),
        "baseline_archive_bytes": _nested(payload, "baseline", "archive_bytes"),
        "best_archive_bytes": _nested(payload, "best", "archive_bytes"),
        "baseline_score_linf": _nested(payload, "baseline", "score_linf"),
        "best_score_linf": _nested(payload, "best", "score_linf"),
        "accepted_improvement": bool(payload.get("accepted_improvement")),
        "ready_for_pose_guard_gate": bool(payload.get("ready_for_pose_guard_gate")),
        "receiver_contract_satisfied": bool(payload.get("receiver_contract_satisfied")),
        "best_packet_materialized": bool(
            best_packet_materialization.get("materialized")
        ),
        "best_packet_path": best_packet_materialization.get("best_packet_path"),
        "best_packet_bytes": best_packet_materialization.get("best_packet_bytes"),
        "best_packet_sha256": best_packet_materialization.get("best_packet_sha256"),
        "best_packet_schema": best_packet_materialization.get("best_packet_schema"),
        "best_packet_wire_format": best_packet_materialization.get(
            "best_packet_wire_format"
        ),
        "best_packet_contest_submission_wire_format_ready": (
            best_packet_materialization.get("contest_submission_wire_format_ready")
            is True
        ),
        "best_packet_materialization": best_packet_materialization,
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
    distortion_contract = report.get("distortion_contract") or {}
    lines = [
        "# SNeRV scorer-loop QAT local trainer",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        f"Axis: `{report.get('axis_tag')}`",
        f"Pairs: `{report.get('n_pairs')}`",
        f"Search mode: `{report.get('search_mode')}`",
        f"Component guard: `{report.get('component_guard_mode')}`",
        f"Adapter: `{report.get('snerv_model_size_adapter')}`",
        f"MFU scales: `{report.get('snerv_mfu_scales')}`",
        f"HFR gain: `{report.get('snerv_hfr_gain')}`",
        f"Decoder features: `{report.get('decoder_feature_count')}`",
        f"LF payload codec: `{report.get('lf_payload_codec')}`",
        f"Evaluations: `{report.get('scorer_loop_evaluations')}`",
        f"Baseline score: `{report.get('baseline_score_linf')}`",
        f"Best score: `{report.get('best_score_linf')}`",
        f"Accepted improvement: `{report.get('accepted_improvement')}`",
        f"Receiver contract satisfied: `{report.get('receiver_contract_satisfied')}`",
        f"Best packet materialized: `{report.get('best_packet_materialized')}`",
        f"Best packet bytes: `{report.get('best_packet_bytes')}`",
        f"Best packet SHA-256: `{report.get('best_packet_sha256')}`",
        f"Best packet path: `{report.get('best_packet_path')}`",
        f"PoseNet YUV6 gradient proof present: `{distortion_contract.get('posenet_yuv6_gradient_proof_present')}`",
        f"PoseNet YUV6 gradient reachable: `{distortion_contract.get('posenet_yuv6_gradient_reachable')}`",
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


def _materialize_best_packet(result: Any, output_dir: Path) -> dict[str, Any]:
    raw_packet = getattr(result, "best_packet", None)
    if raw_packet is None:
        return _missing_best_packet_materialization("best_packet_attr_missing")
    packet = bytes(raw_packet)
    if not packet:
        return _missing_best_packet_materialization("best_packet_empty")
    try:
        decoded = unpack_snerv_archive(packet)
        wire_format = _packet_wire_format_for_schema(decoded.schema)
        packet_schema = decoded.schema
        decode_error = None
    except SnervArchiveError as exc:
        wire_format = None
        packet_schema = None
        decode_error = repr(exc)

    path = output_dir / "best_packet.snar"
    path.write_bytes(packet)
    blockers = []
    if wire_format != "snar2":
        blockers.append(
            "snerv_best_packet_materialized_as_snar1_debug_packet_repack_required"
            if wire_format == "snar1"
            else "snerv_best_packet_wire_format_unverified"
        )
    return {
        "schema": "snerv_scorer_loop_best_packet_materialization.v1",
        "materialized": True,
        "best_packet_path": path.as_posix(),
        "best_packet_bytes": path.stat().st_size,
        "best_packet_sha256": sha256_file(path),
        "best_packet_schema": packet_schema,
        "best_packet_wire_format": wire_format,
        "contest_submission_wire_format_ready": wire_format == "snar2",
        "decode_error": decode_error,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _missing_best_packet_materialization(reason: str) -> dict[str, Any]:
    return {
        "schema": "snerv_scorer_loop_best_packet_materialization.v1",
        "materialized": False,
        "best_packet_path": None,
        "best_packet_bytes": 0,
        "best_packet_sha256": None,
        "missing_reason": reason,
        "blockers": ["snerv_native_scorer_loop_best_packet_not_materialized"],
        **FALSE_AUTHORITY,
    }


def _packet_wire_format_for_schema(schema: str) -> str | None:
    if schema == SNERV_ARCHIVE_SCHEMA:
        return "snar1"
    if schema == SNERV_ARCHIVE_SCHEMA_V2:
        return "snar2"
    return None


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


def _native_mlx_decoder_training_controls(args: argparse.Namespace) -> dict[str, Any]:
    requested_steps = int(args.snerv_native_mlx_decoder_train_steps)
    blockers = (
        ["snerv_native_mlx_decoder_training_controls_unreachable_from_cpu_scorer_loop_harness"]
        if requested_steps > 0
        else []
    )
    return {
        "schema": "snerv_native_mlx_decoder_training_cli_control.v1",
        "requested_steps": requested_steps,
        "learning_rate": float(args.snerv_native_mlx_decoder_train_lr),
        "ridge": float(args.snerv_native_mlx_decoder_train_ridge),
        "consumed_by_cli": False,
        "consumed_by_archive_metadata": False,
        "native_mlx_training_executed": False,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _distortion_contract_from_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Persist the upstream evaluate.py distortion contract into terminal reports.

    SNeRV scorer-loop rows are often consumed by later planners as "pose-guard"
    evidence.  The upstream scorer makes that dangerous unless the PoseNet path
    is explicitly gradient-reachable: PoseNet scores both frames through YUV6,
    while the original ``rgb_to_yuv6`` is no-grad.  This report-level guard keeps
    the PR95/evaluate.py lesson attached to every row that tries to pass the
    pose continuation gate.
    """

    proof = _extract_posenet_yuv6_gradient_proof(payload)
    proof_present = isinstance(proof, dict)
    proof_schema = proof.get("schema") if proof_present else None
    gradient_reachable = (
        bool(proof.get("gradient_reachable")) if proof_present else False
    )
    blockers: list[str] = []
    if bool(payload.get("ready_for_pose_guard_gate")):
        if not proof_present:
            blockers.append(
                "snerv_pose_guard_gate_missing_posenet_yuv6_gradient_reachability_proof"
            )
        elif proof_schema != "posenet_yuv6_gradient_reachability_proof.v1":
            blockers.append(
                "snerv_pose_guard_gate_posenet_yuv6_gradient_proof_schema_mismatch"
            )
        elif not gradient_reachable:
            blockers.append(
                "snerv_pose_guard_gate_posenet_yuv6_gradient_not_reachable"
            )
    if proof_present:
        blockers.extend(str(value) for value in proof.get("blockers") or ())
    return {
        "schema": "snerv_scorer_loop_distortion_contract.v1",
        "upstream_evaluate_source": "upstream/evaluate.py",
        "upstream_modules_source": "upstream/modules.py",
        "upstream_frame_utils_source": "upstream/frame_utils.py",
        "segnet_domain": "last_frame_only_x[:, -1, ...]_at_384x512",
        "posenet_domain": "two_frame_pair_through_12ch_yuv6_at_384x512",
        "rate_term": "25 * archive_zip_bytes / uncompressed_total_bytes",
        "pose_guard_gate_requested": bool(payload.get("ready_for_pose_guard_gate")),
        "posenet_yuv6_gradient_proof_present": proof_present,
        "posenet_yuv6_gradient_reachable": gradient_reachable,
        "posenet_yuv6_gradient_reachability": dict(proof) if proof_present else None,
        "blockers": list(dict.fromkeys(blockers)),
        **FALSE_AUTHORITY,
    }


def _extract_posenet_yuv6_gradient_proof(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    for key in (
        "posenet_yuv6_gradient_reachability",
        "posenet_yuv6_gradient_reachability_proof",
    ):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    score_exact = payload.get("score_exact_saliency")
    if isinstance(score_exact, dict):
        value = score_exact.get("posenet_yuv6_gradient_reachability")
        if isinstance(value, dict):
            return value
    return None


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _looks_local(path: Path) -> bool:
    return not path.resolve(strict=False).as_posix().startswith("/Volumes/")


def _parse_positive_int_csv(raw: str) -> tuple[int, ...]:
    values = []
    for chunk in str(raw).split(","):
        text = chunk.strip()
        if not text:
            continue
        value = int(text)
        if value < 1:
            raise ValueError("positive integer list values must be >= 1")
        values.append(value)
    if not values:
        raise ValueError("at least one positive integer is required")
    return tuple(values)


def _parse_dynamic_range_repair_gains(raw: str) -> tuple[float, ...]:
    text = str(raw or "").strip()
    if not text:
        return ()
    if text.lower() == "auto":
        return DEFAULT_DYNAMIC_RANGE_REPAIR_GAINS
    values = []
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(float(token))
    return tuple(values)


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": TRAINER_SCHEMA,
        "research_json_path": report.get("research_json_path"),
        "output_dir": report.get("output_dir"),
        "n_pairs": report.get("n_pairs"),
        "snerv_model_size_adapter": report.get("snerv_model_size_adapter"),
        "snerv_mfu_scales": report.get("snerv_mfu_scales"),
        "snerv_hfr_gain": report.get("snerv_hfr_gain"),
        "decoder_feature_count": report.get("decoder_feature_count"),
        "scorer_loop_evaluations": report.get("scorer_loop_evaluations"),
        "baseline_score_linf": report.get("baseline_score_linf"),
        "best_score_linf": report.get("best_score_linf"),
        "accepted_improvement": report.get("accepted_improvement"),
        "best_packet_materialized": report.get("best_packet_materialized"),
        "best_packet_path": report.get("best_packet_path"),
        "best_packet_bytes": report.get("best_packet_bytes"),
        "best_packet_sha256": report.get("best_packet_sha256"),
        "score_claim": report.get("score_claim"),
        "ready_for_exact_eval_dispatch": report.get("ready_for_exact_eval_dispatch"),
        "blockers": report.get("blockers"),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    from tac.admission_guard import assert_governed_admission
    assert_governed_admission("train_substrate_snerv_scorer_loop_local")
    if args.score_loop:
        return _score_loop_main(args)
    raise AssertionError("argparse should require a mode")


__all__ = [
    "TRAINER_SCHEMA",
    "_build_parser",
    "_build_report",
    "_distortion_contract_from_result_payload",
    "_materialize_best_packet",
    "_parse_dynamic_range_repair_gains",
    "_resolve_output_dir",
    "_score_loop_kwargs_from_args",
    "main",
    "render_snerv_scorer_loop_local_markdown",
]


if __name__ == "__main__":
    raise SystemExit(main())
