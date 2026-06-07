#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize measured inverse-evaluate ActionEffect candidates.

This tool is deliberately thin.  It consumes existing ``tac.action_effect.v1``
ledgers, re-emits only measured inverse-scorer atoms, and writes the exact
PR110 K=16 baseline reproduction blocker beside them.  It does not score an
archive, synthesize missing gradients, infer contest authority, or build a new
compiler stack.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.action_commutator import build_commutator_ledger  # noqa: E402
from tac.analysis.action_effect import ActionEffect, append_action_effect, read_action_effects  # noqa: E402
from tac.analysis.inverse_scorer_actions import (  # noqa: E402
    generate_inverse_scorer_candidates,
)
from tac.analysis.pr110_baseline_reproduction import (  # noqa: E402
    baseline_blockers_for_menu_ilp,
    build_pr110_k16_baseline_reproduction_from_action_effects,
    validate_pr110_k16_baseline_reproduction,
)
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS  # noqa: E402

DEFAULT_SSD_ROOT = Path("/Volumes/VertigoDataTier/pact/experiments/results")
OUTPUT_SCHEMA = "tac.inverse_evaluate_action_materialization.v1"


def _read_ledgers(paths: Sequence[Path]) -> list[ActionEffect]:
    effects: list[ActionEffect] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"ActionEffect ledger not found: {path}")
        effects.extend(read_action_effects(path))
    return effects


def _unique_effects(effects: Iterable[ActionEffect]) -> list[ActionEffect]:
    by_id: dict[str, ActionEffect] = {}
    order: list[str] = []
    for effect in effects:
        if effect.action_id not in by_id:
            order.append(effect.action_id)
        by_id[effect.action_id] = effect
    return [by_id[action_id] for action_id in order]


def _split_inverse_candidates(effects: Sequence[ActionEffect]) -> tuple[list[ActionEffect], list[ActionEffect]]:
    singles: list[ActionEffect] = []
    composites: list[ActionEffect] = []
    for effect in effects:
        if effect.frame_index == "both" or "composite" in effect.action_kind:
            composites.append(effect)
        else:
            singles.append(effect)
    return singles, composites


def _write_action_effect_ledger(effects: Sequence[ActionEffect], path: Path) -> int:
    path.unlink(missing_ok=True)
    for effect in effects:
        append_action_effect(effect, path)
    return len(effects)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(dict(row), sort_keys=True) + "\n")
            count += 1
    return count


def _default_output_dir() -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_SSD_ROOT / f"inverse_evaluate_actions_{stamp}"


def _first_blocker(summary: Mapping[str, Any]) -> str | None:
    for key in ("pr110_k16_blockers", "inverse_generation_blockers", "menu_ilp_blockers"):
        values = summary.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and values:
            return str(values[0])
    return None


def _render_blocker_note(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Inverse Evaluate Action Materialization",
        "",
        f"- schema: `{summary['schema']}`",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- action_effect_rows: `{summary['action_effect_row_count']}`",
        f"- inverse_candidate_rows: `{summary['inverse_candidate_count']}`",
        f"- pr110_replay_rows: `{summary['pr110_replay_row_count']}`",
        f"- menu_ilp_allowed: `{summary['menu_ilp_allowed']}`",
        "",
        "## First Blocker",
        "",
        f"`{_first_blocker(summary) or 'none'}`",
        "",
        "## PR110 K16 Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in summary["pr110_k16_blockers"])
    lines.extend(
        [
            "",
            "## Inverse Generation Blockers",
            "",
        ]
    )
    lines.extend(f"- `{blocker}`" for blocker in summary["inverse_generation_blockers"])
    if not summary["inverse_generation_blockers"]:
        lines.append("- none")
    lines.extend(
        [
            "",
            "All rows remain false-authority planning/advisory rows. No score, rank,",
            "promotion, dispatch, or menu-ILP authority is minted here.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-action-effects",
        action="append",
        type=Path,
        required=True,
        help="Measured HiNeRV/SNeRV ActionEffect JSONL ledger; repeatable.",
    )
    parser.add_argument(
        "--pr110-action-effects",
        action="append",
        type=Path,
        default=[],
        help="Measured PR110 selector/replay ActionEffect JSONL ledger; repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to the SSD tier "
            "/Volumes/VertigoDataTier/pact/experiments/results/inverse_evaluate_actions_<utc>."
        ),
    )
    parser.add_argument(
        "--exclude-rejected",
        action="store_true",
        help="Drop measured source rows whose exact_score_decision is reject.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_dir = args.output_dir.resolve(strict=False) if args.output_dir is not None else _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        seed_effects = _read_ledgers(args.seed_action_effects)
        pr110_effects = _read_ledgers(args.pr110_action_effects) if args.pr110_action_effects else []
        inverse_generation = generate_inverse_scorer_candidates(
            seed_effects,
            include_rejected=not args.exclude_rejected,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"FATAL: could not materialize inverse ActionEffect rows: {exc}", file=sys.stderr)
        return 2

    inverse_effects = list(inverse_generation["action_effects"])
    pr110_proof = build_pr110_k16_baseline_reproduction_from_action_effects(
        pr110_effects,
        source=",".join(path.as_posix() for path in args.pr110_action_effects),
    )
    pr110_validation = validate_pr110_k16_baseline_reproduction(
        pr110_proof,
        source="generated_from_action_effects",
    )
    menu_ilp_blockers = baseline_blockers_for_menu_ilp(pr110_validation)

    singles, composites = _split_inverse_candidates(inverse_effects)
    commutator = build_commutator_ledger(
        singles,
        composites,
        first_measurement_command=(
            "uv run python tools/generate_inverse_evaluate_actions.py "
            + " ".join(f"--seed-action-effects {path.as_posix()}" for path in args.seed_action_effects)
            + " "
            + " ".join(f"--pr110-action-effects {path.as_posix()}" for path in args.pr110_action_effects)
            + f" --output-dir {out_dir.as_posix()}"
        ),
    )

    action_effect_path = out_dir / "action_effect_rows.jsonl"
    queue_path = out_dir / "inverse_candidate_queue.jsonl"
    commutator_path = out_dir / "commutator_summary.json"
    pr110_proof_path = out_dir / "pr110_k16_baseline_reproduction.json"
    pr110_validation_path = out_dir / "pr110_k16_baseline_validation.json"
    summary_path = out_dir / "summary.json"
    blocker_note_path = out_dir / "next_blocker.md"

    output_effects = _unique_effects([*pr110_effects, *inverse_effects])
    action_effect_count = _write_action_effect_ledger(output_effects, action_effect_path)
    queue_count = _write_jsonl(queue_path, inverse_generation["candidate_queue"])
    _write_json(commutator_path, commutator)
    _write_json(pr110_proof_path, pr110_proof)
    _write_json(pr110_validation_path, pr110_validation)

    summary = {
        "schema": OUTPUT_SCHEMA,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "output_dir": out_dir.as_posix(),
        "seed_action_effect_paths": [path.as_posix() for path in args.seed_action_effects],
        "pr110_action_effect_paths": [path.as_posix() for path in args.pr110_action_effects],
        "action_effect_rows_path": action_effect_path.as_posix(),
        "inverse_candidate_queue_path": queue_path.as_posix(),
        "commutator_summary_path": commutator_path.as_posix(),
        "pr110_k16_baseline_reproduction_path": pr110_proof_path.as_posix(),
        "pr110_k16_baseline_validation_path": pr110_validation_path.as_posix(),
        "next_blocker_path": blocker_note_path.as_posix(),
        "seed_action_effect_count": len(seed_effects),
        "pr110_replay_row_count": len(pr110_effects),
        "inverse_candidate_count": len(inverse_effects),
        "action_effect_row_count": action_effect_count,
        "queue_row_count": queue_count,
        "measured_commutator_count": commutator["measured_commutator_count"],
        "needs_measurement_count": commutator["needs_measurement_count"],
        "inverse_generation_blockers": list(inverse_generation["blockers"]),
        "pr110_k16_blockers": list(pr110_validation["blockers"]),
        "menu_ilp_allowed": not menu_ilp_blockers,
        "menu_ilp_blockers": menu_ilp_blockers,
        "policy": {
            "measured_effects_only_no_synthetic_scorer_motion": True,
            "pr110_k16_baseline_required_before_menu_ilp": True,
            "commutator_values_measured_never_invented": True,
            "default_artifact_tier": "ssd",
        },
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }
    _write_json(summary_path, summary)
    blocker_note_path.write_text(_render_blocker_note(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True) + "\n", end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
