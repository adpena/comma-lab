#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build planning-grade HiNeRV/SNeRV model-size budget artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_modelsize_budget import (  # noqa: E402
    DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID,
    SNERV_MODELSIZE_CONTROL_PROFILES,
    build_hinerv_modelsize_budget_report,
    build_snerv_modelsize_budget_report,
    snerv_modelsize_control_profile,
)
from tac.repo_io import write_json_artifact, write_text_artifact  # noqa: E402


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    if not str(value).strip():
        return ()
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-hinerv-json", required=True, type=Path)
    parser.add_argument("--output-snerv-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--hard-byte-ceiling", action="append", type=int)
    parser.add_argument("--num-pairs", type=int, default=600)
    parser.add_argument("--per-ceiling-limit", type=int, default=6)
    parser.add_argument(
        "--snerv-fc-dim",
        action="append",
        type=int,
        help=(
            "Manual SNeRV fc_dim candidate. Repeatable. Defaults to 9 when "
            "omitted."
        ),
    )
    parser.add_argument(
        "--snerv-emb-size",
        action="append",
        type=int,
        help=(
            "SNeRV embedding-size candidate. Repeatable. Defaults to 0 when "
            "omitted."
        ),
    )
    parser.add_argument(
        "--snerv-official-modelsize-mparams",
        action="append",
        type=float,
        help=(
            "Source-faithful SNeRV --modelsize value in millions of params. "
            "Repeatable. Solves fc_dim through the official equation and still "
            "requires archive-byte/receiver proof before promotion."
        ),
    )
    parser.add_argument(
        "--target-modelsize-mparams",
        action="append",
        type=float,
        help=(
            "Shared operator-facing model-size target, in millions of params. "
            "It expands into the real family controls: HiNeRV nearest local "
            "receiver-visible capacity search and SNeRV source-faithful "
            "--modelsize/fc_dim solve. Repeatable; still false-authority until "
            "archive bytes and receiver proof land."
        ),
    )
    parser.add_argument(
        "--hinerv-target-modelsize-mparams",
        action="append",
        type=float,
        help=(
            "Local HiNeRV inverse capacity target, in millions of params. "
            "Selects the nearest receiver-visible architecture row and remains "
            "false-authority until archive bytes and receiver proof land."
        ),
    )
    parser.add_argument(
        "--include-hinerv-partial-controls",
        action="store_true",
        help=(
            "Include local/partial HiNeRV control rows in the budget artifact. "
            "Default top-priority budget generation emits only candidates with "
            "both official hierarchical feature-grid and ConvNeXt controls."
        ),
    )
    parser.add_argument(
        "--snerv-modelsize-control-profile",
        choices=sorted(SNERV_MODELSIZE_CONTROL_PROFILES),
        default=DEFAULT_SNERV_MODELSIZE_CONTROL_PROFILE_ID,
        help=(
            "Named SNeRV modelsize stride/control profile. The default is the "
            "PACT receiver-closed contest profile, not the upstream parser default."
        ),
    )
    parser.add_argument(
        "--snerv-official-enc-strds",
        type=_parse_int_tuple,
        default=None,
        help=(
            "Comma-separated SNeRV encoder strides. Overrides the selected "
            "--snerv-modelsize-control-profile; use an empty string for []."
        ),
    )
    parser.add_argument(
        "--snerv-official-dec-strds",
        type=_parse_int_tuple,
        default=None,
        help=(
            "Comma-separated SNeRV decoder strides. Overrides the selected "
            "--snerv-modelsize-control-profile."
        ),
    )
    parser.add_argument(
        "--snerv-temporal-context",
        type=int,
        default=0,
        help=(
            "Receiver-visible SNeRV temporal context radius to include in "
            "budget candidates. Use with --snerv-temporal-mode for SNeRV_T "
            "Haar/DWT1D controls."
        ),
    )
    parser.add_argument(
        "--snerv-temporal-mode",
        action="append",
        choices=("delta", "official_haar_dwt1d_lowpass"),
        help=(
            "SNeRV temporal basis candidate. Repeatable. Defaults to delta "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help=(
            "Allow replacing existing output files, but only when the matching "
            "expected-output-* sha256 flag is supplied."
        ),
    )
    parser.add_argument("--expected-output-hinerv-json-sha256")
    parser.add_argument("--expected-output-snerv-json-sha256")
    parser.add_argument("--expected-output-md-sha256")
    args = parser.parse_args(argv)

    hard_byte_ceilings = tuple(
        int(value) for value in (args.hard_byte_ceiling or (216_000, 285_000, 360_000))
    )
    snerv_fc_dims = tuple(int(value) for value in (args.snerv_fc_dim or (9,)))
    snerv_emb_sizes = tuple(int(value) for value in (args.snerv_emb_size or (0,)))
    snerv_official_modelsize_mparams = tuple(
        float(value) for value in (args.snerv_official_modelsize_mparams or ())
    )
    target_modelsize_mparams = tuple(
        float(value) for value in (args.target_modelsize_mparams or ())
    )
    hinerv_target_modelsize_mparams = tuple(
        _dedupe_float_sequence(
            [
                *target_modelsize_mparams,
                *(args.hinerv_target_modelsize_mparams or ()),
            ]
        )
    )
    snerv_official_modelsize_mparams = tuple(
        _dedupe_float_sequence(
            [
                *target_modelsize_mparams,
                *snerv_official_modelsize_mparams,
            ]
        )
    )
    snerv_temporal_modes = tuple(args.snerv_temporal_mode or ("delta",))
    snerv_profile = snerv_modelsize_control_profile(
        str(args.snerv_modelsize_control_profile)
    )
    snerv_official_enc_strds = (
        tuple(int(v) for v in args.snerv_official_enc_strds)
        if args.snerv_official_enc_strds is not None
        else tuple(int(v) for v in snerv_profile["enc_strds"])
    )
    snerv_official_dec_strds = (
        tuple(int(v) for v in args.snerv_official_dec_strds)
        if args.snerv_official_dec_strds is not None
        else tuple(int(v) for v in snerv_profile["dec_strds"])
    )
    hinerv = build_hinerv_modelsize_budget_report(
        hard_byte_ceilings=hard_byte_ceilings,
        num_pairs=int(args.num_pairs),
        per_ceiling_limit=int(args.per_ceiling_limit),
        target_modelsize_mparams=hinerv_target_modelsize_mparams,
        official_controls_only=not bool(args.include_hinerv_partial_controls),
    )
    snerv = build_snerv_modelsize_budget_report(
        hard_byte_ceilings=hard_byte_ceilings,
        num_pairs=int(args.num_pairs),
        per_ceiling_limit=int(args.per_ceiling_limit),
        fc_dims=snerv_fc_dims,
        emb_sizes=snerv_emb_sizes,
        official_modelsize_mparams=snerv_official_modelsize_mparams,
        official_enc_strds=snerv_official_enc_strds,
        official_dec_strds=snerv_official_dec_strds,
        modelsize_control_profile_id=str(args.snerv_modelsize_control_profile),
        temporal_context=int(args.snerv_temporal_context),
        temporal_modes=snerv_temporal_modes,
    )
    hinerv_result = write_json_artifact(
        args.output_hinerv_json,
        hinerv,
        allow_overwrite=bool(args.allow_overwrite),
        expected_existing_sha256=args.expected_output_hinerv_json_sha256,
    )
    snerv_result = write_json_artifact(
        args.output_snerv_json,
        snerv,
        allow_overwrite=bool(args.allow_overwrite),
        expected_existing_sha256=args.expected_output_snerv_json_sha256,
    )
    md_result = None
    if args.output_md is not None:
        md_result = write_text_artifact(
            args.output_md,
            _render_markdown(hinerv, snerv),
            allow_overwrite=bool(args.allow_overwrite),
            expected_existing_sha256=args.expected_output_md_sha256,
        )

    summary = {
        "schema": "nerv_modelsize_budget_build.v1",
        "inputs": {
            "hard_byte_ceilings": list(hard_byte_ceilings),
            "num_pairs": int(args.num_pairs),
            "per_ceiling_limit": int(args.per_ceiling_limit),
            "target_modelsize_mparams": list(target_modelsize_mparams),
            "hinerv_target_modelsize_mparams": list(hinerv_target_modelsize_mparams),
            "hinerv_official_controls_only": not bool(
                args.include_hinerv_partial_controls
            ),
            "snerv_fc_dims": list(snerv_fc_dims),
            "snerv_emb_sizes": list(snerv_emb_sizes),
            "snerv_official_modelsize_mparams": list(
                snerv_official_modelsize_mparams
            ),
            "snerv_modelsize_control_profile_id": str(
                args.snerv_modelsize_control_profile
            ),
            "snerv_modelsize_control_profile": snerv_profile,
            "snerv_official_enc_strds": [int(v) for v in snerv_official_enc_strds],
            "snerv_official_dec_strds": [int(v) for v in snerv_official_dec_strds],
            "snerv_temporal_context": int(args.snerv_temporal_context),
            "snerv_temporal_modes": list(snerv_temporal_modes),
        },
        "hinerv_output_json": hinerv_result.path,
        "hinerv_output_sha256": hinerv_result.sha256,
        "snerv_output_json": snerv_result.path,
        "snerv_output_sha256": snerv_result.sha256,
        "output_md": None if md_result is None else md_result.path,
        "output_md_sha256": None if md_result is None else md_result.sha256,
        "hinerv_selected_candidate_count": int(hinerv["selected_candidate_count"]),
        "snerv_selected_candidate_count": int(snerv["selected_candidate_count"]),
        "snerv_invalid_candidate_count": int(
            snerv.get("invalid_candidate_count") or 0
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _render_markdown(hinerv: dict[str, Any], snerv: dict[str, Any]) -> str:
    lines = [
        "# NeRV Model-Size Budgets",
        "",
        "False-authority planning artifact. These rows price model-size controls",
        "before byte-closed archive export, receiver proof, local replay, or exact auth.",
        "",
        "## Summary",
        "",
        f"- HiNeRV selected candidates: `{hinerv['selected_candidate_count']}`",
        f"- SNeRV selected candidates: `{snerv['selected_candidate_count']}`",
        f"- SNeRV invalid official controls skipped: `{snerv.get('invalid_candidate_count', 0)}`",
        f"- Num pairs: `{hinerv['num_pairs']}`",
        f"- Score claim: `{hinerv['score_claim']}`",
        f"- Ready for exact eval: `{hinerv['ready_for_exact_eval_dispatch']}`",
        "",
        "## Top HiNeRV Candidates",
        "",
    ]
    lines.extend(_candidate_lines(hinerv.get("selected_candidates") or []))
    lines.extend(["", "## Top SNeRV Candidates", ""])
    lines.extend(_candidate_lines(snerv.get("selected_candidates") or []))
    invalid_snerv = snerv.get("invalid_candidates") or []
    if invalid_snerv:
        lines.extend(["", "## Skipped SNeRV Official Controls", ""])
        lines.extend(_invalid_candidate_lines(invalid_snerv))
    return "\n".join(lines).rstrip() + "\n"


def _dedupe_float_sequence(values: list[float]) -> tuple[float, ...]:
    out: list[float] = []
    seen: set[float] = set()
    for value in values:
        normalized = float(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return tuple(out)


def _candidate_lines(rows: list[dict[str, Any]]) -> list[str]:
    out = []
    for row in rows[:8]:
        payload_bytes = row.get("nominal_total_payload_bytes", row.get("total_payload_bytes"))
        out.append(
            "- "
            f"`{row.get('candidate_id')}` "
            f"payload=`{payload_bytes}` "
            f"nominal_under_ceiling=`{row.get('nominal_under_ceiling')}`"
        )
    return out or ["- none"]


def _invalid_candidate_lines(rows: list[dict[str, Any]]) -> list[str]:
    out = []
    for row in rows[:8]:
        out.append(
            "- "
            f"modelsize=`{row.get('official_modelsize_mparams')}` "
            f"emb_size=`{row.get('emb_size')}` "
            f"temporal_mode=`{row.get('temporal_mode')}` "
            f"error_type=`{row.get('error_type')}`"
        )
    return out or ["- none"]


if __name__ == "__main__":
    raise SystemExit(main())
