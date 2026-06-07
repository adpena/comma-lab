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
    build_score_program_word,
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
BLOCKER_REVERSE_ORDER_COMPOSITE_PRODUCER_MISSING = (
    "inverse_scorer_reverse_order_composite_producer_missing"
)
BLOCKER_COMPOSITE_BASE_IDENTITY_PRODUCER_MISSING = (
    "inverse_scorer_composite_base_identity_producer_missing"
)


def _read_ledgers(paths: Sequence[Path]) -> list[ActionEffect]:
    effects: list[ActionEffect] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"ActionEffect ledger not found: {path}")
        effects.extend(read_action_effects(path))
    return effects


def _read_training_artifacts(paths: Sequence[Path]) -> list[ActionEffect]:
    effects: list[ActionEffect] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"training artifact not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"training artifact JSON malformed at {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"training artifact must be a JSON object: {path}")
        seen_payload_ids: set[int] = set()
        for source in _find_hinerv_four_arm_sources(payload):
            if id(source) in seen_payload_ids:
                continue
            seen_payload_ids.add(id(source))
            effects.extend(
                ActionEffect.from_hinerv_four_arm_ablation(
                    source,
                    consumer="inverse_evaluate_candidate_queue",
                )
            )
    return effects


def _find_hinerv_four_arm_sources(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if (
                isinstance(value.get("four_arm_ablation"), Mapping)
                or str(value.get("schema") or "") == "hi_nerv_target_region_birth_four_arm_ablation.v1"
            ):
                out.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                walk(child)

    walk(payload)
    return out


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
    return DEFAULT_SSD_ROOT / f"actioneffect_inverse_scorer_{stamp}"


def _first_blocker(summary: Mapping[str, Any]) -> str | None:
    for key in (
        "pr110_k16_blockers",
        "inverse_generation_blockers",
        "menu_ilp_blockers",
        "commutator_measurement_blockers",
    ):
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
            "## Commutator Measurement Blockers",
            "",
        ]
    )
    lines.extend(f"- `{blocker}`" for blocker in summary["commutator_measurement_blockers"])
    if not summary["commutator_measurement_blockers"]:
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


def _apply_menu_gate_to_queue(rows: Sequence[Mapping[str, Any]], blockers: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    allowed = not blockers
    for row in rows:
        updated = dict(row)
        updated["menu_ilp_allowed"] = allowed
        updated["menu_ilp_blockers"] = list(blockers)
        out.append(updated)
    return out


def _render_test_log(summary: Mapping[str, Any]) -> str:
    lines = [
        "inverse-evaluate ActionEffect materializer smoke",
        f"schema={summary['schema']}",
        f"action_effect_rows={summary['action_effect_row_count']}",
        f"inverse_candidate_rows={summary['inverse_candidate_count']}",
        f"pr110_replay_rows={summary['pr110_replay_row_count']}",
        f"measured_commutators={summary['measured_commutator_count']}",
        f"needs_measurement={summary['needs_measurement_count']}",
        f"menu_ilp_allowed={summary['menu_ilp_allowed']}",
        "assertions:",
        f"- has_action_effect_rows={summary['action_effect_row_count'] > 0}",
        f"- has_inverse_candidates={summary['inverse_candidate_count'] >= 3}",
        f"- has_commutator_summary={summary['measured_commutator_count'] >= 1}",
        f"- pr110_k16_gate_passed={not summary['pr110_k16_blockers']}",
        f"- commutator_measurement_blockers={len(summary['commutator_measurement_blockers'])}",
        "",
    ]
    return "\n".join(lines)


def _commutator_measurement_blockers(commutator: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    queue = commutator.get("measurement_queue")
    if not isinstance(queue, Sequence) or isinstance(queue, (str, bytes)):
        return blockers
    for row in queue:
        if not isinstance(row, Mapping):
            continue
        reason = row.get("reason")
        if isinstance(reason, str) and reason:
            blockers.append(reason)
        command_blockers = row.get("measurement_command_blockers")
        if isinstance(command_blockers, Sequence) and not isinstance(command_blockers, (str, bytes)):
            blockers.extend(str(item) for item in command_blockers if str(item).strip())
    return _dedupe(blockers)


def _measurement_command_blockers_for_inverse_commutator(
    singles: Sequence[ActionEffect],
) -> list[str]:
    blockers = [BLOCKER_REVERSE_ORDER_COMPOSITE_PRODUCER_MISSING]
    if not _all_single_effects_have_base_identity(singles):
        blockers.append(BLOCKER_COMPOSITE_BASE_IDENTITY_PRODUCER_MISSING)
    return blockers


def _all_single_effects_have_base_identity(effects: Sequence[ActionEffect]) -> bool:
    if not effects:
        return False
    for effect in effects:
        if effect.base_state_sha256:
            continue
        if effect.archive_sha256 and effect.payload_sha256:
            continue
        return False
    return True


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-action-effects",
        action="append",
        type=Path,
        default=[],
        help="Measured HiNeRV/SNeRV ActionEffect JSONL ledger; repeatable.",
    )
    parser.add_argument(
        "--seed-training-artifact",
        action="append",
        type=Path,
        default=[],
        help=(
            "Measured training artifact containing HiNeRV four_arm_ablation payloads; "
            "repeatable. Used to rebuild ActionEffect rows with producer-side identity."
        ),
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
    if not args.seed_action_effects and not args.seed_training_artifact:
        print("FATAL: provide --seed-action-effects or --seed-training-artifact", file=sys.stderr)
        return 2
    out_dir = args.output_dir.resolve(strict=False) if args.output_dir is not None else _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        seed_effects = [
            *_read_ledgers(args.seed_action_effects),
            *_read_training_artifacts(args.seed_training_artifact),
        ]
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
        first_measurement_command=None,
        measurement_command_blockers=_measurement_command_blockers_for_inverse_commutator(singles),
        use_default_measurement_command=False,
    )
    commutator_measurement_blockers = _commutator_measurement_blockers(commutator)

    candidate_queue_rows = _apply_menu_gate_to_queue(inverse_generation["candidate_queue"], menu_ilp_blockers)

    action_effect_path = out_dir / "action_effect_rows.jsonl"
    queue_path = out_dir / "inverse_candidate_queue.jsonl"
    score_program_word_path = out_dir / "score_program_word.json"
    commutator_path = out_dir / "commutator_summary.json"
    pr110_proof_path = out_dir / "pr110_k16_baseline_reproduction.json"
    pr110_validation_path = out_dir / "pr110_k16_baseline_validation.json"
    summary_path = out_dir / "summary.json"
    blocker_note_path = out_dir / "next_blocker.md"
    test_log_path = out_dir / "test_log.txt"

    output_effects = _unique_effects([*pr110_effects, *inverse_effects])
    action_effect_count = _write_action_effect_ledger(output_effects, action_effect_path)
    queue_count = _write_jsonl(queue_path, candidate_queue_rows)
    score_program_word = build_score_program_word(candidate_queue_rows)
    _write_json(score_program_word_path, score_program_word)
    _write_json(commutator_path, commutator)
    _write_json(pr110_proof_path, pr110_proof)
    _write_json(pr110_validation_path, pr110_validation)

    summary = {
        "schema": OUTPUT_SCHEMA,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "output_dir": out_dir.as_posix(),
        "seed_action_effect_paths": [path.as_posix() for path in args.seed_action_effects],
        "seed_training_artifact_paths": [path.as_posix() for path in args.seed_training_artifact],
        "pr110_action_effect_paths": [path.as_posix() for path in args.pr110_action_effects],
        "action_effect_rows_path": action_effect_path.as_posix(),
        "inverse_candidate_queue_path": queue_path.as_posix(),
        "score_program_word_path": score_program_word_path.as_posix(),
        "commutator_summary_path": commutator_path.as_posix(),
        "pr110_k16_baseline_reproduction_path": pr110_proof_path.as_posix(),
        "pr110_k16_baseline_validation_path": pr110_validation_path.as_posix(),
        "next_blocker_path": blocker_note_path.as_posix(),
        "test_log_path": test_log_path.as_posix(),
        "seed_action_effect_count": len(seed_effects),
        "pr110_replay_row_count": len(pr110_effects),
        "inverse_candidate_count": len(inverse_effects),
        "action_effect_row_count": action_effect_count,
        "queue_row_count": queue_count,
        "score_program_operation_count": score_program_word["operation_count"],
        "score_program_blockers": list(score_program_word["blockers"]),
        "score_program_promotion_blockers": list(score_program_word["promotion_blockers"]),
        "measured_commutator_count": commutator["measured_commutator_count"],
        "needs_measurement_count": commutator["needs_measurement_count"],
        "commutator_measurement_blockers": commutator_measurement_blockers,
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
    test_log_path.write_text(_render_test_log(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True) + "\n", end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
