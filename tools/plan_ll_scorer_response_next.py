#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan the next LL scorer-response probes from a response dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.mlx_execution_plan import (  # noqa: E402
    MLXExecutionPlanError,
    build_mlx_scorer_response_execution_plan,
)
from tac.local_acceleration.mlx_production_contract import (  # noqa: E402
    build_mlx_scorer_production_contract_bundle_manifest,
)
from tac.optimization.scorer_response_dataset import (  # noqa: E402
    ScorerResponseDatasetError,
    build_next_probe_plan,
    render_next_probe_plan_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--null-byte-matrix",
        type=Path,
        help=(
            "Optional tools/probe_null_byte_master_gradient_matrix.py JSON output "
            "to weight the next LL surrogate training-data harvest."
        ),
    )
    parser.add_argument(
        "--null-byte-seed-budget-k",
        type=int,
        default=16,
        help="Seed budget key K to read from null-byte matrix predicted_delta_s_per_seed_budget.",
    )
    parser.add_argument(
        "--allow-legacy-null-byte-matrix-missing-authority",
        action="store_true",
        help=(
            "Accept historical null-byte matrices that predate explicit false "
            "authority fields. Future matrices should not use this flag."
        ),
    )
    parser.add_argument(
        "--magic-codec-seed-boundary-smoke",
        type=Path,
        help=(
            "Optional pair #4 procedural-seed orthogonality smoke JSON. "
            "When provided, the LL plan refuses magic-codec wrapping of raw "
            "procedural seed bytes if the boundary is validated."
        ),
    )
    parser.add_argument(
        "--decoder-q-response-surface",
        type=Path,
        help=(
            "Optional tools/plan_decoder_q_response_surface.py JSON output. "
            "When provided, the LL planner prioritizes preserving improving "
            "decoder-q windows while suppressing or inverting regressing ones."
        ),
    )
    parser.add_argument(
        "--decoder-q-surface-advisory-batch",
        type=Path,
        help=(
            "Optional tools/run_decoder_q_candidate_advisory_batch.py JSON output "
            "for candidates generated from the decoder-q response surface. When "
            "provided, exact-eval spend routing is blocked unless at least one "
            "surface-guided candidate improves advisory score."
        ),
    )
    parser.add_argument(
        "--mlx-torch-parity-sweep",
        type=Path,
        help=(
            "Optional tools/audit_mlx_scorer_torch_parity_sweep.py JSON output. "
            "MLX scorer-response rows are blocked from LL planning unless this "
            "gate is attached or an explicit research-only override is set."
        ),
    )
    parser.add_argument(
        "--mlx-profile-stability",
        type=Path,
        help=(
            "Optional MLX profile-stability manifest. When provided, attach a "
            "non-authoritative local scorer-response execution recommendation "
            "for harvesting the next LL rows."
        ),
    )
    parser.add_argument(
        "--mlx-score-calibration",
        type=Path,
        help=(
            "Optional MLX score-calibration manifest. When provided, the LL plan "
            "can distinguish local training use from calibrated exact-eval spend "
            "triage; absent or blocked calibration forbids MLX-based spend filtering."
        ),
    )
    parser.add_argument(
        "--mlx-production-contract",
        action="append",
        type=Path,
        help=(
            "Optional strict MLX production-contract manifest. May repeat to "
            "build a per-row contract bundle. Exact-eval spend triage from MLX "
            "rows is blocked unless every row is covered by a strict pass."
        ),
    )
    parser.add_argument(
        "--mlx-archive-size-bytes",
        type=int,
        help=(
            "Archive byte count for the MLX execution recommendation when the "
            "stability manifest predates profile_summary.archive_size_bytes."
        ),
    )
    parser.add_argument("--mlx-response-output", type=Path)
    parser.add_argument("--mlx-components-dir", type=Path)
    parser.add_argument("--mlx-progress-every", type=int, default=0)
    parser.add_argument(
        "--allow-mlx-gpu-research-signal",
        action="store_true",
        help=(
            "Permit a selected MLX GPU row as research signal in the attached "
            "execution plan. The plan remains non-authoritative."
        ),
    )
    parser.add_argument(
        "--allow-mlx-batch-shape-research-signal",
        action="store_true",
        help=(
            "Permit a selected non-singleton MLX batch row as explicit batch-shape "
            "research signal in the attached execution plan. The plan remains "
            "non-authoritative."
        ),
    )
    parser.add_argument(
        "--allow-mlx-parity-research-signal-override",
        action="store_true",
        help=(
            "Allow MLX rows into the LL planner when the attached parity sweep "
            "is not a strict pass. This remains non-promotional and cannot "
            "rank, claim, or dispatch exact eval."
        ),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        null_byte_matrix = (
            None
            if args.null_byte_matrix is None
            else json.loads(args.null_byte_matrix.read_text(encoding="utf-8"))
        )
        magic_codec_seed_boundary_smoke = (
            None
            if args.magic_codec_seed_boundary_smoke is None
            else json.loads(args.magic_codec_seed_boundary_smoke.read_text(encoding="utf-8"))
        )
        decoder_q_response_surface = (
            None
            if args.decoder_q_response_surface is None
            else json.loads(args.decoder_q_response_surface.read_text(encoding="utf-8"))
        )
        decoder_q_surface_advisory_batch = (
            None
            if args.decoder_q_surface_advisory_batch is None
            else json.loads(args.decoder_q_surface_advisory_batch.read_text(encoding="utf-8"))
        )
        mlx_torch_parity_sweep = (
            None
            if args.mlx_torch_parity_sweep is None
            else json.loads(args.mlx_torch_parity_sweep.read_text(encoding="utf-8"))
        )
        mlx_score_calibration = (
            None
            if args.mlx_score_calibration is None
            else json.loads(args.mlx_score_calibration.read_text(encoding="utf-8"))
        )
        mlx_production_contracts = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (args.mlx_production_contract or [])
        ]
        if not mlx_production_contracts:
            mlx_production_contract = None
        elif len(mlx_production_contracts) == 1:
            mlx_production_contract = mlx_production_contracts[0]
        else:
            mlx_production_contract = (
                build_mlx_scorer_production_contract_bundle_manifest(
                    mlx_production_contracts,
                    run_id="cli_bundle",
                    producer="tools.plan_ll_scorer_response_next",
                )
            )
        plan = build_next_probe_plan(
            dataset,
            null_byte_matrix=null_byte_matrix,
            null_byte_seed_budget_k=args.null_byte_seed_budget_k,
            allow_legacy_null_byte_matrix_missing_authority=(
                args.allow_legacy_null_byte_matrix_missing_authority
            ),
            magic_codec_seed_boundary_smoke=magic_codec_seed_boundary_smoke,
            decoder_q_response_surface=decoder_q_response_surface,
            mlx_torch_parity_sweep=mlx_torch_parity_sweep,
            allow_mlx_parity_research_signal_override=(
                args.allow_mlx_parity_research_signal_override
            ),
            mlx_score_calibration=mlx_score_calibration,
            mlx_production_contract=mlx_production_contract,
            decoder_q_surface_advisory_batch=decoder_q_surface_advisory_batch,
        )
        if args.mlx_profile_stability is not None:
            mlx_profile_stability = json.loads(args.mlx_profile_stability.read_text(encoding="utf-8"))
            if not isinstance(mlx_profile_stability, dict):
                raise MLXExecutionPlanError("MLX profile-stability payload must be an object")
            plan["mlx_scorer_response_execution_plan"] = (
                build_mlx_scorer_response_execution_plan(
                    mlx_profile_stability,
                    archive_size_bytes=args.mlx_archive_size_bytes,
                    repo_root=REPO_ROOT,
                    response_output=args.mlx_response_output,
                    components_dir=args.mlx_components_dir,
                    progress_every=args.mlx_progress_every,
                    allow_gpu_research_signal=args.allow_mlx_gpu_research_signal,
                    allow_batch_shape_research_signal=(
                        args.allow_mlx_batch_shape_research_signal
                    ),
                )
            )
    except (OSError, ValueError, ScorerResponseDatasetError, MLXExecutionPlanError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_next_probe_plan_markdown(plan), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "prohibitions": plan["prohibitions"],
                "probes": plan["probes"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
