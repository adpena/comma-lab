#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Canonicalize pairset component-marginal observations into reusable signals.

This is the operator-facing helper for the DQS1 drop-pair loop:

1. ingest exact-axis observation JSONL and pairset-acquisition candidates;
2. emit the canonical portfolio/action summary with xray/equation refs;
3. verify the signal is wired to xray, canonical equations, and master-gradient
   consumers;
4. optionally append/register the canonical equation through the locked
   canonical-equations registry.

All emitted artifacts are planning-only and carry no score or dispatch
authority.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.canonical_equations import register_canonical_equation  # noqa: E402
from tac.canonical_equations.pairset_component_marginal import (  # noqa: E402
    build_pairset_component_marginal_score_decomposition_v1,
)
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
from tac.optimization.pairset_component_marginal import (  # noqa: E402
    FALSE_AUTHORITY,
    PAIRSET_COMPONENT_MARGINAL_SCORE_DECOMPOSITION_EQUATION_ID,
    PAIRSET_COMPONENT_MARGINAL_XRAY_PRIMITIVE_NAME,
)
from tac.repo_io import read_json, sha256_file  # noqa: E402

ACTION_SUMMARY_SCHEMA = "pairset_component_marginal_canonicalization_summary.v1"
TOOL = "tools/canonicalize_pairset_component_marginal_signal.py"

REQUIRED_MASTER_GRADIENT_CONSUMERS = frozenset(
    {
        "tac.master_gradient_consumers.per_pair_difficulty_atlas",
        "tac.master_gradient_consumers.per_pair_pareto_envelope",
        "tac.master_gradient_consumers.per_pair_lagrangian_lambda_bisection",
        "tac.master_gradient_consumers.per_pair_coding_budget_allocation",
    }
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
    parser.add_argument("--incumbent-score", type=float, required=True)
    parser.add_argument(
        "--incumbent-score-by-axis",
        action="append",
        default=[],
        metavar="AXIS=SCORE",
    )
    parser.add_argument(
        "--pairset-acquisition",
        type=Path,
        action="append",
        required=True,
        help="decoder_q_pairset_acquisition.v1 JSON. May repeat.",
    )
    parser.add_argument(
        "--observation-jsonl",
        type=Path,
        action="append",
        required=True,
        help="mlx_dynamic_sweep_observation.v1 JSONL with exact-axis rows.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=_positive_int, default=32)
    parser.add_argument("--top-actions", type=_positive_int, default=8)
    parser.add_argument(
        "--allow-inactive-component-model",
        action="store_true",
        help="Do not fail if the component-marginal model is inactive.",
    )
    parser.add_argument(
        "--register-equation",
        action="store_true",
        help="Append/register the canonical equation through the locked registry.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        help="Optional registry JSONL path, mainly for tests with --register-equation.",
    )
    parser.add_argument(
        "--registry-lock-path",
        type=Path,
        help="Optional registry lock path, mainly for tests with --register-equation.",
    )
    parser.add_argument("--agent", default="codex")
    parser.add_argument(
        "--subagent-id",
        default="pairset_component_marginal_canonicalization_helper",
    )
    parser.add_argument(
        "--equation-notes",
        default="canonicalize_pairset_component_marginal_signal",
    )
    return parser.parse_args(argv)


def _axis_scores(values: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise CrossFamilyCandidatePortfolioError(
                "--incumbent-score-by-axis must use AXIS=SCORE"
            )
        axis, raw = value.split("=", 1)
        axis = axis.strip()
        if not axis:
            raise CrossFamilyCandidatePortfolioError(
                "--incumbent-score-by-axis axis must be non-empty"
            )
        try:
            out[axis] = float(raw)
        except ValueError as exc:
            raise CrossFamilyCandidatePortfolioError(
                f"--incumbent-score-by-axis {axis} score must be numeric"
            ) from exc
    return out


def _json_objects(paths: Sequence[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise CrossFamilyCandidatePortfolioError(f"{path}: expected JSON object")
        out.append(payload)
    return out


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _component_model(portfolio: Mapping[str, Any]) -> dict[str, Any]:
    feedback = _mapping(portfolio.get("observation_feedback"))
    model = feedback.get("pairset_component_marginal_model")
    if isinstance(model, Mapping):
        return dict(model)
    return {
        "schema": "pairset_component_marginal_model.v1",
        "active": False,
        "inactive_reason": "missing_from_portfolio",
        **FALSE_AUTHORITY,
    }


def _assert_false_authority(payload: Any, *, path: str = "$") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in FALSE_AUTHORITY and value is not False:
                raise CrossFamilyCandidatePortfolioError(
                    f"{path}.{key} must be false, got {value!r}"
                )
            _assert_false_authority(value, path=f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _assert_false_authority(value, path=f"{path}[{index}]")


def _require_canonical_refs(model: Mapping[str, Any]) -> None:
    refs = _mapping(model.get("canonical_signal_refs"))
    xray = set(refs.get("xray_primitives") or [])
    equations = set(refs.get("canonical_equations") or [])
    consumers = set(refs.get("master_gradient_consumers") or [])
    missing: list[str] = []
    if PAIRSET_COMPONENT_MARGINAL_XRAY_PRIMITIVE_NAME not in xray:
        missing.append("xray_primitives.pairset_component_marginal")
    if PAIRSET_COMPONENT_MARGINAL_SCORE_DECOMPOSITION_EQUATION_ID not in equations:
        missing.append("canonical_equations.pairset_component_marginal_score_decomposition_v1")
    missing_consumers = sorted(REQUIRED_MASTER_GRADIENT_CONSUMERS - consumers)
    missing.extend(f"master_gradient_consumers.{value}" for value in missing_consumers)
    if missing:
        raise CrossFamilyCandidatePortfolioError(
            "pairset component marginal model missing canonical refs: "
            + ", ".join(missing)
        )


def _top_actions(portfolio: Mapping[str, Any], *, limit: int) -> list[dict[str, Any]]:
    rows = portfolio.get("operator_action_rows")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        if not isinstance(row, Mapping):
            continue
        blockers = [str(blocker) for blocker in row.get("dispatch_blockers") or []]
        out.append(
            {
                "operator_action_rank": row.get("operator_action_rank"),
                "bayesian_rank": row.get("rank"),
                "candidate_id": row.get("candidate_id"),
                "source_kind": row.get("source_kind"),
                "operator_next_action": row.get("operator_next_action"),
                "acquisition_value": row.get("acquisition_value"),
                "predicted_score_mean": row.get("predicted_score_mean"),
                "predicted_score_variance": row.get("predicted_score_variance"),
                "dispatch_blocker_count": len(blockers),
                "dispatch_blockers": blockers[:8],
                **FALSE_AUTHORITY,
            }
        )
    return out


def _register_equation_if_requested(args: argparse.Namespace) -> dict[str, Any]:
    if not args.register_equation:
        return {
            "registered": False,
            "reason": "register_equation_flag_not_set",
            **FALSE_AUTHORITY,
        }
    equation = build_pairset_component_marginal_score_decomposition_v1()
    register_canonical_equation(
        equation,
        path=args.registry_path,
        lock_path=args.registry_lock_path,
        agent=args.agent,
        subagent_id=args.subagent_id,
        notes=args.equation_notes,
    )
    return {
        "registered": True,
        "equation_id": equation.equation_id,
        "empirical_anchor_count": len(equation.empirical_anchors),
        "is_well_calibrated": equation.is_well_calibrated,
        **FALSE_AUTHORITY,
    }


def _render_summary_markdown(summary: Mapping[str, Any]) -> str:
    component = _mapping(summary.get("pairset_component_marginal_model"))
    lines = [
        "## Pairset Component Marginal Canonicalization",
        "",
        f"- Schema: `{summary.get('schema')}`",
        f"- Allowed use: `{summary.get('allowed_use')}`",
        f"- Component model active: `{component.get('active')}`",
        f"- Axes: `{component.get('axes')}`",
        f"- Training rows: `{component.get('training_row_count')}`",
        f"- Equation registration: `{_mapping(summary.get('equation_registration')).get('registered')}`",
        f"- Score claim: `{summary.get('score_claim')}`",
        f"- Ready for exact eval dispatch: `{summary.get('ready_for_exact_eval_dispatch')}`",
        "",
        "### Top Next Actions",
        "",
        "| action rank | bayes rank | candidate | action | acquisition | blockers |",
        "|---:|---:|---|---|---:|---:|",
    ]
    rows = summary.get("top_operator_actions")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| {action_rank} | {bayes_rank} | `{candidate}` | `{action}` | "
                "{acquisition:.12g} | {blockers} |".format(
                    action_rank=row.get("operator_action_rank"),
                    bayes_rank=row.get("bayesian_rank"),
                    candidate=row.get("candidate_id"),
                    action=row.get("operator_next_action"),
                    acquisition=float(row.get("acquisition_value", 0.0)),
                    blockers=row.get("dispatch_blocker_count"),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _build_summary(
    *,
    portfolio: Mapping[str, Any],
    component_model: Mapping[str, Any],
    output_dir: Path,
    top_actions: int,
    equation_registration: Mapping[str, Any],
) -> dict[str, Any]:
    portfolio_path = output_dir / "portfolio.json"
    action_summary_path = output_dir / "action_summary.json"
    return {
        "schema": ACTION_SUMMARY_SCHEMA,
        "producer": TOOL,
        "allowed_use": "canonical_signal_generation_only_no_score_or_dispatch_authority",
        "portfolio_json": str(portfolio_path),
        "portfolio_sha256": sha256_file(portfolio_path) if portfolio_path.is_file() else "",
        "portfolio_md": str(output_dir / "portfolio.md"),
        "action_summary_json": str(action_summary_path),
        "pairset_component_marginal_model": dict(component_model),
        "portfolio_summary": dict(_mapping(portfolio.get("portfolio_summary"))),
        "equation_registration": dict(equation_registration),
        "top_action_limit": top_actions,
        "top_operator_actions": _top_actions(portfolio, limit=top_actions),
        "dispatch_blockers": [
            str(blocker) for blocker in portfolio.get("dispatch_blockers") or []
        ],
        **FALSE_AUTHORITY,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        observations = [
            row
            for path in args.observation_jsonl
            for row in load_observation_rows(path)
        ]
        source_artifacts = source_artifacts_from_paths(
            {
                "pairset_acquisitions": args.pairset_acquisition,
                "observation_jsonl": args.observation_jsonl,
            },
            repo_root=REPO_ROOT,
        )
        portfolio = build_cross_family_candidate_portfolio(
            incumbent_score=args.incumbent_score,
            pairset_acquisitions=_json_objects(args.pairset_acquisition),
            observations=observations,
            incumbent_scores_by_axis=_axis_scores(args.incumbent_score_by_axis),
            source_artifacts=source_artifacts,
            source_artifact_paths={
                "pairset_acquisitions": [
                    path.as_posix() for path in args.pairset_acquisition
                ],
            },
            top_k=args.top_k,
        )
        component_model = _component_model(portfolio)
        if component_model.get("active") is not True and not args.allow_inactive_component_model:
            raise CrossFamilyCandidatePortfolioError(
                "pairset component marginal model inactive "
                f"({component_model.get('inactive_reason') or 'unknown'}); "
                "pass --allow-inactive-component-model only for exploratory runs"
            )
        _require_canonical_refs(component_model)
        _assert_false_authority(portfolio)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "portfolio.json", portfolio)
        (args.output_dir / "portfolio.md").write_text(
            render_cross_family_candidate_portfolio_markdown(portfolio),
            encoding="utf-8",
        )
        equation_registration = _register_equation_if_requested(args)
        summary = _build_summary(
            portfolio=portfolio,
            component_model=component_model,
            output_dir=args.output_dir,
            top_actions=args.top_actions,
            equation_registration=equation_registration,
        )
        _assert_false_authority(summary)
        write_json(args.output_dir / "action_summary.json", summary)
        (args.output_dir / "action_summary.md").write_text(
            _render_summary_markdown(summary),
            encoding="utf-8",
        )
    except (
        OSError,
        json.JSONDecodeError,
        CrossFamilyCandidatePortfolioError,
        ValueError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
