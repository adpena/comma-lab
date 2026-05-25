#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan local-only DQS1 decoder-q pair-set acquisition candidates."""

# ruff: noqa: E402

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

from tac.optimization.decoder_q_pairset_acquisition import (
    DecoderQPairsetAcquisitionError,
    build_decoder_q_pairset_acquisition_plan,
    load_json_object,
    write_json,
)
from tac.optimization.dqs1_materializer_feedback_bridge import (
    DQS1_OBSERVATION_SOURCE_SCHEMA,
    DQS1_OBSERVATION_SWEEP_CONFIG_ID,
)
from tac.optimization.mlx_dynamic_sweep_observations import (
    MLXDynamicSweepObservationError,
    load_observation_rows,
    observation_duplicate_key,
)


def _parse_csv_ints(text: str | None) -> list[int] | None:
    if text is None:
        return None
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    return values or None


def _false_authority_fields() -> tuple[str, ...]:
    return (
        "score_claim",
        "score_claim_valid",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
        "gpu_launched",
        "promotable",
    )


def _truthy_authority(value: object) -> bool:
    if value is True:
        return True
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _assert_no_authority(payload: object, *, label: str) -> None:
    fields = set(_false_authority_fields())

    def visit(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                next_path = f"{path}.{key}" if path else str(key)
                if key in fields and _truthy_authority(child):
                    raise DecoderQPairsetAcquisitionError(
                        f"{label} leaks authority at {next_path}"
                    )
                visit(child, next_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "")


def _load_eureka_planning(paths: list[Path]) -> dict[str, object] | None:
    if not paths:
        return None
    hint_rows: list[dict[str, object]] = []
    source_paths: list[str] = []
    signal_count = 0
    for path in paths:
        payload = load_json_object(path)
        _assert_no_authority(payload, label=str(path))
        planning = payload.get("local_cpu_eureka_planning")
        if not isinstance(planning, dict):
            planning = payload
        hints = planning.get("planner_hints")
        if isinstance(hints, list):
            for hint in hints:
                if isinstance(hint, dict):
                    hint_rows.append(hint)
        try:
            signal_count += int(planning.get("signal_count") or 0)
        except (TypeError, ValueError):
            pass
        source_paths.append(path.as_posix())
    if not hint_rows:
        return {
            "schema": "decoder_q_pairset_acquisition_cli_eureka_planning.v1",
            "active": False,
            "source_paths": source_paths,
            "signal_count": signal_count,
            "planner_hint_count": 0,
            "planner_hints": [],
            "inactive_reason": "no_planner_hints",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        }
    return {
        "schema": "decoder_q_pairset_acquisition_cli_eureka_planning.v1",
        "active": True,
        "source_paths": source_paths,
        "signal_count": signal_count,
        "planner_hint_count": len(hint_rows),
        "planner_hints": hint_rows,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-pareto", type=Path, required=True)
    parser.add_argument("--frame-policy", default="pair_all_frames")
    parser.add_argument("--prefix-ks", help="Comma-separated prefix sizes from the best selector.")
    parser.add_argument("--diversity-ks", help="Comma-separated diversity-spaced pair-set sizes.")
    parser.add_argument("--no-drop-one", action="store_true")
    parser.add_argument("--max-drop-two", type=int, default=128)
    parser.add_argument(
        "--drop-many-counts",
        help="Comma-separated pair-drop counts for bounded eureka drop-many probes.",
    )
    parser.add_argument(
        "--max-drop-many",
        type=int,
        default=None,
        help=(
            "Maximum bounded drop-many candidates. Defaults to 0 unless an "
            "eureka planning hint activates beyond-drop-two expansion."
        ),
    )
    parser.add_argument("--max-swap-in", type=int, default=32)
    parser.add_argument("--diversity-weight", type=float, default=0.15)
    parser.add_argument(
        "--eureka-planning-json",
        type=Path,
        action="append",
        default=[],
        help=(
            "frontier feedback refresh/cycle/eureka planning JSON. When it "
            "contains the beyond-drop-two hint, bounded drop-many probes become active."
        ),
    )
    parser.add_argument(
        "--dqs1-observation-jsonl",
        "--dqs1-observations",
        action="append",
        default=[],
        dest="dqs1_observation_jsonl",
        help=(
            "DQS1 local-first harvest observation JSONL used to suppress already "
            "observed acquisition candidates. May repeat."
        ),
    )
    parser.add_argument(
        "--include-observed-dqs1-candidate",
        action="store_true",
        help="keep observed DQS1 candidates in the acquisition plan for explicit replay/debug",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args(argv)


def _render_markdown(plan: dict[str, object]) -> str:
    summary = plan.get("summary", {})
    candidates = plan.get("candidates", [])
    lines = [
        "# Decoder-Q Pair-Set Acquisition Plan",
        "",
        f"- Candidate count: `{summary.get('candidate_count') if isinstance(summary, dict) else None}`",
        (
            "- Recommended acquisition: "
            f"`{summary.get('recommended_acquisition_id') if isinstance(summary, dict) else None}`"
        ),
        f"- Score claim: `{plan.get('score_claim')}`",
        f"- Promotion eligible: `{plan.get('promotion_eligible')}`",
        f"- Ready for exact eval dispatch: `{plan.get('ready_for_exact_eval_dispatch')}`",
        f"- Dispatch attempted: `{plan.get('dispatch_attempted')}`",
        "",
        "| rank | acquisition | kind | pairs | payload bytes | acquisition score | diversity | predicted mean |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    if isinstance(candidates, list):
        for row in candidates[:20]:
            if not isinstance(row, dict):
                continue
            predicted = row.get("predicted_score_mean")
            lines.append(
                "| "
                f"{row.get('acquisition_rank')} | `{row.get('acquisition_id')}` | "
                f"`{row.get('selector_kind')}` | {row.get('selected_pair_count')} | "
                f"{row.get('payload_bytes')} | `{row.get('acquisition_score')}` | "
                f"`{row.get('diversity_score')}` | `{predicted}` |"
            )
    lines.extend(
        [
            "",
            "Authority: planning-only local pair-set acquisition. It does not dispatch, "
            "materialize an archive, claim a score, promote a candidate, or rank/kill a lane.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_dqs1_observations(path_values: list[str]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, str | None], ...]] = set()
    for value in path_values:
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise DecoderQPairsetAcquisitionError(
                f"{path}: observation JSONL does not exist"
            )
        if path.suffix != ".jsonl":
            raise DecoderQPairsetAcquisitionError(
                f"{path}: DQS1 observations must be JSONL rows"
            )
        try:
            loaded = load_observation_rows(path)
        except OSError as exc:
            raise DecoderQPairsetAcquisitionError(
                f"{path}: cannot read observation JSONL"
            ) from exc
        except MLXDynamicSweepObservationError as exc:
            raise DecoderQPairsetAcquisitionError(
                f"{path}: invalid observation JSONL: {exc}"
            ) from exc
        if not loaded:
            raise DecoderQPairsetAcquisitionError(
                f"{path}: observation JSONL has no rows"
            )
        for row in loaded:
            if (
                row.get("source_schema") != DQS1_OBSERVATION_SOURCE_SCHEMA
                or row.get("sweep_config_id") != DQS1_OBSERVATION_SWEEP_CONFIG_ID
            ):
                raise DecoderQPairsetAcquisitionError(
                    f"{path}: non-local-first DQS1 observation row refused "
                    f"for candidate {row.get('candidate_id')!r}"
                )
            key = observation_duplicate_key(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return tuple(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        selector_pareto = load_json_object(args.selector_pareto)
        dqs1_observations = _load_dqs1_observations(args.dqs1_observation_jsonl)
        plan = build_decoder_q_pairset_acquisition_plan(
            selector_pareto,
            frame_policy=args.frame_policy,
            prefix_ks=_parse_csv_ints(args.prefix_ks),
            diversity_ks=_parse_csv_ints(args.diversity_ks),
            include_drop_one=not args.no_drop_one,
            max_drop_two=args.max_drop_two,
            drop_many_counts=_parse_csv_ints(args.drop_many_counts),
            max_drop_many=args.max_drop_many,
            max_swap_in=args.max_swap_in,
            diversity_weight=args.diversity_weight,
            dqs1_observations=dqs1_observations,
            include_observed_candidates=args.include_observed_dqs1_candidate,
            eureka_planning=_load_eureka_planning(args.eureka_planning_json),
        )
        write_json(args.json_out, plan)
        if args.md_out is not None:
            args.md_out.parent.mkdir(parents=True, exist_ok=True)
            args.md_out.write_text(_render_markdown(plan), encoding="utf-8")
    except (OSError, json.JSONDecodeError, DecoderQPairsetAcquisitionError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "candidate_count": plan["summary"]["candidate_count"],
                "unfiltered_candidate_count": plan["summary"]["unfiltered_candidate_count"],
                "suppressed_observed_candidate_count": plan["summary"][
                    "suppressed_observed_candidate_count"
                ],
                "recommended_acquisition_id": plan["summary"]["recommended_acquisition_id"],
                "drop_many_candidate_count": plan["summary"][
                    "drop_many_candidate_count"
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
