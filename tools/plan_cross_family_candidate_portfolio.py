#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan a false-authority cross-family exact-eval candidate portfolio."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.cross_family_candidate_portfolio import (  # noqa: E402
    CrossFamilyCandidatePortfolioError,
    build_cross_family_candidate_portfolio,
    render_cross_family_candidate_portfolio_markdown,
    source_artifacts_from_paths,
    write_json,
)
from tac.optimization.mlx_dynamic_sweep_observations import (  # noqa: E402
    load_observation_rows,
)
from tac.repo_io import read_json  # noqa: E402

ACTION_SUMMARY_SCHEMA = "cross_family_candidate_portfolio_action_summary.v1"

FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "promotable",
    "dispatch_attempted",
    "gpu_launched",
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incumbent-score",
        type=float,
        required=True,
        help="Exact CUDA incumbent score used only as a planning baseline.",
    )
    parser.add_argument(
        "--mlx-selection",
        type=Path,
        action="append",
        default=[],
        help="mlx_effective_spend_triage_candidate_selection.v1 JSON. May repeat.",
    )
    parser.add_argument(
        "--pairset-acquisition",
        type=Path,
        action="append",
        default=[],
        help="decoder_q_pairset_acquisition.v1 JSON. May repeat.",
    )
    parser.add_argument(
        "--dqs1-drop-many-greedy-verdict",
        type=Path,
        action="append",
        default=[],
        help=(
            "dqs1_drop_many_build_1c_greedy_independent_heuristic_verdict.v1 "
            "JSON. May repeat; used as planning-only blocker signal for "
            "source-inherited independent drop-many rows."
        ),
    )
    parser.add_argument(
        "--hfv2-manifest",
        type=Path,
        action="append",
        default=[],
        help="hfv1_to_hfv2_sparse_sidecar_candidate_v1 JSON. May repeat.",
    )
    parser.add_argument(
        "--archive-contract-surface",
        type=Path,
        action="append",
        default=[],
        help=(
            "tac_archive_bound_candidate_contract_surface.v1 or "
            "tac_archive_bound_candidate_contract.v1 JSON; adapter packages "
            "with embedded contract surfaces are also accepted. May repeat."
        ),
    )
    parser.add_argument(
        "--posterior-ledger",
        type=Path,
        action="append",
        default=[],
        help=(
            "Planning posterior JSON/JSONL carrying negative outcomes, stack "
            "penalties, byte-credit blockers, or entropy-stage misses."
        ),
    )
    parser.add_argument(
        "--candidate-json",
        type=Path,
        action="append",
        default=[],
        help="Manual candidate JSON object/list, or object with candidates[].",
    )
    parser.add_argument(
        "--family-beliefs",
        type=Path,
        help="Optional family belief JSON object/list overriding weak defaults.",
    )
    parser.add_argument(
        "--observation-jsonl",
        type=Path,
        action="append",
        default=[],
        help=(
            "Append-only mlx_dynamic_sweep_observation.v1 JSONL. Exact-axis "
            "same-candidate observations demote repeat operator actions."
        ),
    )
    parser.add_argument(
        "--incumbent-score-by-axis",
        action="append",
        default=[],
        metavar="AXIS=SCORE",
        help=(
            "Optional exact-axis incumbent baseline, e.g. contest_cpu=0.19203. "
            "contest_cuda defaults to --incumbent-score."
        ),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        help=(
            "Optional compact operator action summary JSON. Planning-only; "
            "does not create dispatch authority."
        ),
    )
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--top-actions",
        type=_positive_int,
        default=8,
        help="Number of operator next-action rows to expose in stdout/summary.",
    )
    parser.add_argument(
        "--require-active-pairset-observation-model",
        action="store_true",
        help=(
            "Fail closed unless exact-axis pairset observations activate the "
            "pairset observation-response planning model."
        ),
    )
    parser.add_argument("--expected-improvement-weight", type=float, default=1.0)
    parser.add_argument("--information-gain-weight", type=float, default=0.01)
    return parser.parse_args(argv)


def _json_objects(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise CrossFamilyCandidatePortfolioError(f"{path}: expected JSON object")
        out.append(payload)
    return out


def _manual_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
            rows = payload["candidates"]
        elif isinstance(payload, dict):
            rows = [payload]
        else:
            raise CrossFamilyCandidatePortfolioError(
                f"{path}: expected candidate object/list"
            )
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise CrossFamilyCandidatePortfolioError(
                    f"{path}: candidate {index} must be object"
                )
            out.append(row)
    return out


def _json_or_jsonl_rows(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix == ".jsonl":
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise CrossFamilyCandidatePortfolioError(
                        f"{path}:{line_no}: expected JSON object"
                    )
                out.append(payload)
            continue
        payload = read_json(path)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            rows = payload["rows"]
        elif isinstance(payload, dict) and isinstance(payload.get("posterior_rows"), list):
            rows = payload["posterior_rows"]
        elif isinstance(payload, dict):
            rows = [payload]
        else:
            raise CrossFamilyCandidatePortfolioError(
                f"{path}: expected JSON object/list"
            )
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise CrossFamilyCandidatePortfolioError(
                    f"{path}: row {index} must be object"
                )
            out.append(row)
    return out


def _axis_scores(values: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise CrossFamilyCandidatePortfolioError(
                "--incumbent-score-by-axis must use AXIS=SCORE"
            )
        axis, raw_score = value.split("=", 1)
        axis = axis.strip()
        if not axis:
            raise CrossFamilyCandidatePortfolioError(
                "--incumbent-score-by-axis axis must be non-empty"
            )
        try:
            out[axis] = float(raw_score)
        except ValueError as exc:
            raise CrossFamilyCandidatePortfolioError(
                f"--incumbent-score-by-axis {axis} score must be numeric"
            ) from exc
    return out


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _false_authority_payload() -> dict[str, bool]:
    return dict.fromkeys(FALSE_AUTHORITY_FIELDS, False)


def _pairset_observation_response_model(
    portfolio: Mapping[str, Any],
) -> dict[str, Any]:
    feedback = _mapping(portfolio.get("observation_feedback"))
    model = feedback.get("pairset_observation_response_model")
    if isinstance(model, Mapping):
        return dict(model)
    return {
        "schema": "pairset_observation_response_model.v1",
        "active": False,
        "inactive_reason": "missing_from_portfolio",
        **_false_authority_payload(),
    }


def _pairset_component_marginal_model(
    portfolio: Mapping[str, Any],
) -> dict[str, Any]:
    feedback = _mapping(portfolio.get("observation_feedback"))
    model = feedback.get("pairset_component_marginal_model")
    if isinstance(model, Mapping):
        return dict(model)
    return {
        "schema": "pairset_component_marginal_model.v1",
        "active": False,
        "inactive_reason": "missing_from_portfolio",
        **_false_authority_payload(),
    }


def _drop_many_greedy_verdict_model(
    portfolio: Mapping[str, Any],
) -> dict[str, Any]:
    feedback = _mapping(portfolio.get("observation_feedback"))
    model = feedback.get("drop_many_greedy_verdict_model")
    if isinstance(model, Mapping):
        return dict(model)
    return {
        "schema": "dqs1_drop_many_greedy_verdict_feedback_model.v1",
        "active": False,
        "inactive_reason": "missing_from_portfolio",
        **_false_authority_payload(),
    }


def _top_operator_actions(
    portfolio: Mapping[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = portfolio.get("operator_action_rows")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        if not isinstance(row, Mapping):
            continue
        metadata = _mapping(row.get("source_metadata"))
        observation_feedback = _mapping(metadata.get("observation_feedback"))
        response_model = _mapping(metadata.get("observation_response_model"))
        blockers = [str(blocker) for blocker in row.get("dispatch_blockers") or []]
        out.append(
            {
                "operator_action_rank": row.get("operator_action_rank"),
                "bayesian_rank": row.get("rank"),
                "candidate_id": row.get("candidate_id"),
                "family_id": row.get("family_id"),
                "source_kind": row.get("source_kind"),
                "operator_next_action": row.get("operator_next_action"),
                "acquisition_value": row.get("acquisition_value"),
                "predicted_score_mean": row.get("predicted_score_mean"),
                "predicted_score_variance": row.get("predicted_score_variance"),
                "prediction_source": row.get("prediction_source"),
                "exact_archive_custody_ready": row.get("exact_archive_custody_ready"),
                "bayesian_ready_for_exact_eval_dispatch": row.get(
                    "bayesian_ready_for_exact_eval_dispatch"
                ),
                "observation_feedback_status": observation_feedback.get("status"),
                "observation_response_model_active": response_model.get("active"),
                "dispatch_blocker_count": len(blockers),
                "dispatch_blockers": blockers[:8],
                **_false_authority_payload(),
            }
        )
    return out


def _build_action_summary(
    portfolio: Mapping[str, Any],
    *,
    top_actions: int,
    json_out: Path,
    md_out: Path | None,
    summary_json_out: Path | None,
) -> dict[str, Any]:
    model = _pairset_observation_response_model(portfolio)
    component_model = _pairset_component_marginal_model(portfolio)
    drop_many_greedy_model = _drop_many_greedy_verdict_model(portfolio)
    return {
        "schema": ACTION_SUMMARY_SCHEMA,
        "producer": Path(__file__).name,
        "allowed_use": "operator_next_action_planning_only_no_score_or_dispatch_authority",
        "json_out": str(json_out),
        "md_out": None if md_out is None else str(md_out),
        "summary_json_out": (
            None if summary_json_out is None else str(summary_json_out)
        ),
        "portfolio_summary": dict(_mapping(portfolio.get("portfolio_summary"))),
        "pairset_observation_response_model": model,
        "pairset_component_marginal_model": component_model,
        "drop_many_greedy_verdict_model": drop_many_greedy_model,
        "top_action_limit": top_actions,
        "top_operator_actions": _top_operator_actions(
            portfolio,
            limit=top_actions,
        ),
        "dispatch_blockers": [
            str(blocker) for blocker in portfolio.get("dispatch_blockers") or []
        ],
        **_false_authority_payload(),
    }


def _require_active_pairset_observation_model(summary: Mapping[str, Any]) -> None:
    model = _mapping(summary.get("pairset_observation_response_model"))
    if model.get("active") is True:
        return
    reason = model.get("inactive_reason") or "unknown"
    raise CrossFamilyCandidatePortfolioError(
        "pairset observation response model inactive "
        f"({reason}); provide --observation-jsonl with at least two exact-axis "
        "pairset observations at distinct selected_pair_count values, or omit "
        "--require-active-pairset-observation-model for exploratory planning"
    )


def _render_action_summary_markdown(summary: Mapping[str, Any]) -> str:
    model = _mapping(summary.get("pairset_observation_response_model"))
    component_model = _mapping(summary.get("pairset_component_marginal_model"))
    drop_many_model = _mapping(summary.get("drop_many_greedy_verdict_model"))
    lines = [
        "## CLI Action Summary",
        "",
        f"- Summary schema: `{summary.get('schema')}`",
        f"- Allowed use: `{summary.get('allowed_use')}`",
        f"- Score claim: `{summary.get('score_claim')}`",
        "- Ready for exact eval dispatch: "
        f"`{summary.get('ready_for_exact_eval_dispatch')}`",
        "",
        "### Observation Response Model",
        "",
        f"- Active: `{model.get('active')}`",
        f"- Axis: `{model.get('axis')}`",
        f"- Training rows: `{model.get('training_row_count')}`",
        f"- Updated candidates: `{model.get('updated_candidate_count')}`",
        f"- Inactive reason: `{model.get('inactive_reason')}`",
        f"- Allowed use: `{model.get('allowed_use')}`",
        f"- Score claim: `{model.get('score_claim')}`",
        "- Ready for exact eval dispatch: "
        f"`{model.get('ready_for_exact_eval_dispatch')}`",
        "",
        "### Component Marginal Model",
        "",
        f"- Active: `{component_model.get('active')}`",
        f"- Axes: `{component_model.get('axes')}`",
        f"- Training rows: `{component_model.get('training_row_count')}`",
        "- Cross-axis diagnostics: "
        f"`{len(component_model.get('cross_axis_transfer_diagnostics') or [])}`",
        f"- Allowed use: `{component_model.get('allowed_use')}`",
        f"- Score claim: `{component_model.get('score_claim')}`",
        "- Ready for exact eval dispatch: "
        f"`{component_model.get('ready_for_exact_eval_dispatch')}`",
        "",
        "### Drop-Many Greedy Verdict",
        "",
        f"- Active: `{drop_many_model.get('active')}`",
        f"- Latest verdict: `{drop_many_model.get('latest_verdict')}`",
        "- Independent greedy status: "
        f"`{drop_many_model.get('independent_greedy_status')}`",
        f"- Verdict count: `{drop_many_model.get('verdict_count')}`",
        f"- Score claim: `{drop_many_model.get('score_claim')}`",
        "- Ready for exact eval dispatch: "
        f"`{drop_many_model.get('ready_for_exact_eval_dispatch')}`",
        "",
        "### Top Next Actions",
        "",
        "| action rank | bayes rank | candidate | source | action | acquisition | blockers |",
        "|---:|---:|---|---|---|---:|---:|",
    ]
    rows = summary.get("top_operator_actions")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| {action_rank} | {bayes_rank} | `{candidate}` | `{source}` | "
                "`{action}` | {acquisition:.12g} | {blockers} |".format(
                    action_rank=row.get("operator_action_rank"),
                    bayes_rank=row.get("bayesian_rank"),
                    candidate=row.get("candidate_id"),
                    source=row.get("source_kind"),
                    action=row.get("operator_next_action"),
                    acquisition=float(row.get("acquisition_value", 0.0)),
                    blockers=row.get("dispatch_blocker_count"),
                )
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source_artifacts = source_artifacts_from_paths(
            {
                "mlx_selections": args.mlx_selection,
                "pairset_acquisitions": args.pairset_acquisition,
                "dqs1_drop_many_greedy_verdicts": args.dqs1_drop_many_greedy_verdict,
                "hfv2_manifests": args.hfv2_manifest,
                "archive_contract_surfaces": args.archive_contract_surface,
                "posterior_ledgers": args.posterior_ledger,
                "manual_candidate_json": args.candidate_json,
                "family_beliefs": args.family_beliefs,
                "observation_jsonl": args.observation_jsonl,
            },
            repo_root=REPO_ROOT,
        )
        family_beliefs = read_json(args.family_beliefs) if args.family_beliefs else None
        observations = [
            row
            for path in args.observation_jsonl
            for row in load_observation_rows(path)
        ]
        portfolio = build_cross_family_candidate_portfolio(
            incumbent_score=args.incumbent_score,
            mlx_selections=_json_objects(args.mlx_selection),
            pairset_acquisitions=_json_objects(args.pairset_acquisition),
            drop_many_greedy_verdicts=_json_objects(
                args.dqs1_drop_many_greedy_verdict
            ),
            hfv2_manifests=_json_objects(args.hfv2_manifest),
            archive_contract_surfaces=_json_objects(args.archive_contract_surface),
            manual_candidates=_manual_candidates(args.candidate_json),
            observations=observations,
            posterior_ledger_rows=_json_or_jsonl_rows(args.posterior_ledger),
            incumbent_scores_by_axis=_axis_scores(args.incumbent_score_by_axis),
            family_beliefs=family_beliefs,
            source_artifacts=source_artifacts,
            source_artifact_paths={
                "mlx_selections": [path.as_posix() for path in args.mlx_selection],
                "pairset_acquisitions": [
                    path.as_posix() for path in args.pairset_acquisition
                ],
                "dqs1_drop_many_greedy_verdicts": [
                    path.as_posix() for path in args.dqs1_drop_many_greedy_verdict
                ],
                "hfv2_manifests": [path.as_posix() for path in args.hfv2_manifest],
                "archive_contract_surfaces": [
                    path.as_posix() for path in args.archive_contract_surface
                ],
                "posterior_ledgers": [
                    path.as_posix() for path in args.posterior_ledger
                ],
            },
            top_k=args.top_k,
            expected_improvement_weight=args.expected_improvement_weight,
            information_gain_weight=args.information_gain_weight,
        )
        action_summary = _build_action_summary(
            portfolio,
            top_actions=args.top_actions,
            json_out=args.json_out,
            md_out=args.md_out,
            summary_json_out=args.summary_json_out,
        )
        if args.require_active_pairset_observation_model:
            _require_active_pairset_observation_model(action_summary)
    except (
        OSError,
        json.JSONDecodeError,
        CrossFamilyCandidatePortfolioError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    write_json(args.json_out, portfolio)
    if args.summary_json_out:
        write_json(args.summary_json_out, action_summary)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(
            render_cross_family_candidate_portfolio_markdown(portfolio).rstrip()
            + "\n\n"
            + _render_action_summary_markdown(action_summary),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "summary_json_out": (
                    None
                    if args.summary_json_out is None
                    else str(args.summary_json_out)
                ),
                "ranked_candidate_count": portfolio["portfolio_summary"][
                    "ranked_candidate_count"
                ],
                "candidate_archive_custody_ready_count": portfolio[
                    "portfolio_summary"
                ]["candidate_archive_custody_ready_count"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
                "pairset_observation_response_model": action_summary[
                    "pairset_observation_response_model"
                ],
                "pairset_component_marginal_model": action_summary[
                    "pairset_component_marginal_model"
                ],
                "drop_many_greedy_verdict_model": action_summary[
                    "drop_many_greedy_verdict_model"
                ],
                "top_operator_actions": action_summary["top_operator_actions"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
