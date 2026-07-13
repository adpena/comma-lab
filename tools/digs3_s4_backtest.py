#!/usr/bin/env python3
"""Deterministic $0 receipts for DIG-S3 and DIG-S4.

This tool is deliberately read-only with respect to run directories and controller state.
It reconstructs the 72-row S3 duty bank at the crosswalk source epoch, audits historical
activation outcomes under a strict custody/outcome/cost join, and inventories S4 traces for
common-checkpoint stay/advance counterfactuals.  It never fabricates rewards, runs a trainer,
or actuates a controller.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
SOURCE_EPOCH = "31bb1e324fe7b4a649442b98c1f0ce4da06c8827"
DSL_PATH = "src/tac/witness_dsl/curriculum_dsl.py"
ACTIVATION_LEDGER = REPO / ".omx/state/lever_activation_ledger.jsonl"
SIGNIFICANCE_LEDGER = REPO / ".omx/state/lever_relative_significance.jsonl"
SCORE_DENOMINATOR_BYTES = 37_545_489
TRACE_DIRS = (
    "experiments/results/levelset_n600_crucible_v6_run1_20260708T095730Z",
    "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z",
    "experiments/results/levelset_n600_witness_20260709T105312Z",
    "experiments/results/levelset_v752_baseline_20260710T185913Z",
    "experiments/results/v9_cgauge_432_coherent_arm_20260711",
)

_FLAG_RE = re.compile(r"^--[a-z0-9][a-z0-9-]*$")
_FUNC_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _git_show(revision: str, path: str) -> str:
    proc = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def _constructs(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(part, ast.Call)
        and isinstance(part.func, ast.Name)
        and part.func.id == name
        for part in ast.walk(node)
    )


def _returns_lever(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.returns is None:
        return False
    try:
        return "Lever" in ast.unparse(node.returns)
    except Exception:
        return False


def _called_names(node: ast.AST) -> set[str]:
    return {
        part.func.id
        for part in ast.walk(node)
        if isinstance(part, ast.Call) and isinstance(part.func, ast.Name)
    }


def _literal_defaults(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[dict[str, Any], list[str]]:
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults: dict[str, Any] = {}
    if node.args.defaults:
        for arg, default in zip(
            positional[-len(node.args.defaults) :], node.args.defaults, strict=True
        ):
            try:
                defaults[arg.arg] = ast.literal_eval(default)
            except (ValueError, TypeError):
                pass
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
        if default is None:
            continue
        try:
            defaults[arg.arg] = ast.literal_eval(default)
        except (ValueError, TypeError):
            pass
    required = [arg.arg for arg in positional if arg.arg not in defaults]
    required.extend(
        arg.arg
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
        if default is None
    )
    return defaults, required


def _static_number(expr: ast.AST, defaults: dict[str, Any]) -> int | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
        return int(expr.value)
    if isinstance(expr, ast.Name) and isinstance(defaults.get(expr.id), (int, float)):
        return int(defaults[expr.id])
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "int"
        and len(expr.args) == 1
    ):
        return _static_number(expr.args[0], defaults)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
        value = _static_number(expr.operand, defaults)
        if value is None:
            return None
        return value if isinstance(expr.op, ast.UAdd) else -value
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub)):
        left = _static_number(expr.left, defaults)
        right = _static_number(expr.right, defaults)
        if left is None or right is None:
            return None
        return left + right if isinstance(expr.op, ast.Add) else left - right
    return None


def _factory_surface(source: str) -> dict[str, dict[str, Any]]:
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, _FUNC_DEFS)]
    factories: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if _constructs(node, "WitnessProgram"):
            continue
        if not (_constructs(node, "Lever") or _returns_lever(node)):
            continue
        defaults, required = _literal_defaults(node)
        flags = sorted(
            {
                part.value
                for part in ast.walk(node)
                if isinstance(part, ast.Constant)
                and isinstance(part.value, str)
                and _FLAG_RE.match(part.value)
                and not part.value.startswith("--no-")
            }
        )
        lever_calls = [
            part
            for part in ast.walk(node)
            if isinstance(part, ast.Call)
            and isinstance(part.func, ast.Name)
            and part.func.id == "Lever"
        ]
        direct_costs: list[int | None] = []
        for call in lever_calls:
            kw = next((item for item in call.keywords if item.arg == "epochs_delta"), None)
            direct_costs.append(0 if kw is None else _static_number(kw.value, defaults))
        cost: int | None
        if required:
            cost = None
        elif direct_costs and all(item is not None for item in direct_costs):
            cost = sum(int(item) for item in direct_costs if item is not None)
        elif direct_costs:
            cost = None
        else:
            cost = None
        factories[node.name] = {
            "node": node,
            "own_flags": flags,
            "called_names": _called_names(node),
            "required_args": required,
            "measurement_cost_epochs": cost,
        }

    # Match the canonical registry's transitive flag closure.  For pure delegating
    # composites, also derive cost only when every delegated default cost is known.
    for name, row in factories.items():
        seen: set[str] = set()
        stack = [name]
        flags: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            flags.update(factories[current]["own_flags"])
            stack.extend(
                called
                for called in factories[current]["called_names"]
                if called in factories and called not in seen
            )
        row["flags"] = sorted(flags)
        if not _constructs(row["node"], "Lever") and not row["required_args"]:
            children = [
                factories[called]["measurement_cost_epochs"]
                for called in row["called_names"]
                if called in factories
            ]
            if children and all(value is not None for value in children):
                row["measurement_cost_epochs"] = sum(int(value) for value in children)
    return factories


def _stratum(name: str, flags: list[str]) -> str:
    text = " ".join([name, *flags]).lower()
    strata = (
        ("optimizer_compute", ("muon", "adam", "grad", "micro-batch", "compile", "fused")),
        ("rate_payload", ("entropy", "code", "byte", "uniward", "texture-trunk")),
        ("pose_temporal", ("pose", "temporal", "screw", "warp-real-luma")),
        ("topology_birth", ("island", "birth", "persistence", "nucleus", "ground-frame")),
        ("boundary_margin", ("margin", "lane", "boundary", "eikonal", "length", "logit")),
        ("render_receiver", ("render", "aa-coverage", "chroma", "out-tex", "palette")),
        ("curriculum_tau", ("curriculum", "tau", "tail", "anneal", "stage", "taper")),
    )
    for stratum, tokens in strata:
        if any(token in text for token in tokens):
            return stratum
    return "other"


def _latest_significance() -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _jsonl(SIGNIFICANCE_LEDGER):
        if row.get("lever"):
            latest[str(row["lever"])] = row
    return latest


def _activation_state(name: str, events: list[dict[str, Any]]) -> str:
    rows = [row for row in events if row.get("lever") == name]
    if any(row.get("event") == "retired" for row in rows):
        return "retired"
    if any(row.get("event") == "measured" for row in rows):
        return "measured"
    if any(row.get("event") == "fired" for row in rows):
        return "fired-unmeasured"
    return "never-fired"


def _activation_bits(name: str, events: list[dict[str, Any]]) -> tuple[bool, bool, bool]:
    rows = [row for row in events if row.get("lever") == name]
    return (
        any(row.get("event") == "fired" for row in rows),
        any(row.get("event") == "measured" for row in rows),
        any(row.get("event") == "retired" for row in rows),
    )


def _resolve_ref(ref: Any) -> Path | None:
    if not isinstance(ref, str) or not ref.strip():
        return None
    token = ref.strip().split(" ", 1)[0]
    path = Path(token)
    return path if path.is_absolute() else REPO / path


def _audit_measured_outcomes(
    events: list[dict[str, Any]], factories: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    complete: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "measured":
            continue
        lever = str(event.get("lever", ""))
        reasons: list[str] = []
        path = _resolve_ref(event.get("verdict_ref"))
        artifact: Any = None
        if lever not in factories:
            reasons.append("no_exact_source_epoch_factory_descriptor")
        if path is None or not path.is_file():
            reasons.append("verdict_ref_not_a_file")
        elif path.suffix.lower() != ".json":
            reasons.append("verdict_ref_not_structured_json")
        else:
            try:
                artifact = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                reasons.append("verdict_json_unreadable")

        outcome_delta_s: float | None = None
        cost_seconds: float | None = None
        cohort: int | None = None
        derivation: str | None = None
        if isinstance(artifact, dict):
            for key in ("n_pairs", "n_samples"):
                value = artifact.get(key)
                if isinstance(value, (int, float)):
                    cohort = int(value)
                    break
            if cohort != 600:
                reasons.append("not_structured_n600")
            if isinstance(artifact.get("delta_s"), (int, float)):
                outcome_delta_s = float(artifact["delta_s"])
                derivation = "artifact.delta_s"
            elif all(isinstance(artifact.get(k), (int, float)) for k in ("score_before", "score_after")):
                outcome_delta_s = float(artifact["score_before"]) - float(artifact["score_after"])
                derivation = "score_before-score_after"
            elif isinstance(artifact.get("measured_byte_delta_best_vs_lbnd2"), (int, float)):
                # This artifact explicitly certifies a pure-rate, bit-identical statistic mutation.
                byte_delta = float(artifact["measured_byte_delta_best_vs_lbnd2"])
                outcome_delta_s = -25.0 * byte_delta / SCORE_DENOMINATOR_BYTES
                derivation = "-25*measured_byte_delta_best_vs_lbnd2/37545489"
            else:
                reasons.append("no_exact_comparable_delta_s")
            for key in ("wall_s", "wall_seconds", "elapsed_seconds"):
                value = artifact.get(key)
                if isinstance(value, (int, float)) and float(value) >= 0:
                    cost_seconds = float(value)
                    break
            if cost_seconds is None:
                reasons.append("no_exact_comparable_cost")

        accepted = not reasons and outcome_delta_s is not None and cost_seconds is not None
        row = {
            "accepted": accepted,
            "activation_ts": event.get("ts"),
            "lever": lever,
            "verdict_ref": event.get("verdict_ref"),
            "artifact_sha256": _sha256(path) if path is not None and path.is_file() else None,
            "n_pairs": cohort,
            "outcome_delta_s_positive_is_improvement": outcome_delta_s,
            "outcome_derivation": derivation,
            "cost_seconds": cost_seconds,
            "rejection_reasons": reasons,
        }
        audit.append(row)
        if accepted:
            descriptor = factories[lever]
            complete.append(
                {
                    **row,
                    "descriptor": {
                        "flags": descriptor["flags"],
                        "flag_count": len(descriptor["flags"]),
                        "stratum": _stratum(lever, descriptor["flags"]),
                        "measurement_cost_epochs": descriptor["measurement_cost_epochs"],
                    },
                }
            )
    complete.sort(key=lambda row: (str(row["activation_ts"]), row["lever"]))
    audit.sort(key=lambda row: (str(row["activation_ts"]), row["lever"]))
    return complete, audit


def _p8_rows(owed: list[dict[str, Any]], significance: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    # Use the live canonical P8 helper, then freeze only rows in the charter bank.
    # This preserves its floor-awareness and canonical pointer reader instead of
    # reimplementing a subtly different display rank.
    try:
        from tac.witness_dsl.activation_ledger import duty_to_measure_ranked

        live = {row["lever"]: row for row in duty_to_measure_ranked()}
    except Exception:
        live = {}
    rows: list[dict[str, Any]] = []
    for item in owed:
        sig = significance.get(item["lever"], {})
        canonical = live.get(item["lever"], {})
        estimate = canonical.get("est_delta_s", sig.get("est_delta_s"))
        rel = canonical.get("rel_sig")
        rows.append(
            {
                "lever": item["lever"],
                "est_delta_s": estimate if isinstance(estimate, (int, float)) else None,
                "delta_s_label": canonical.get("delta_s_label", sig.get("delta_s_label")),
                "axis": canonical.get("axis", sig.get("axis")),
                "relative_significance": rel if isinstance(rel, (int, float)) else None,
                "floor_status": canonical.get("floor_status"),
                "s_current": canonical.get("s_current"),
                "s_target": canonical.get("s_target"),
            }
        )
    rows.sort(
        key=lambda row: (
            row["relative_significance"] is None,
            -(row["relative_significance"] or 0.0),
            row["lever"],
        )
    )
    return rows


def _s3_receipt() -> dict[str, Any]:
    source = _git_show(SOURCE_EPOCH, DSL_PATH)
    factories = _factory_surface(source)
    events = _jsonl(ACTIVATION_LEDGER)
    significance = _latest_significance()
    owed: list[dict[str, Any]] = []
    for name in sorted(factories):
        state = _activation_state(name, events)
        ever_fired, ever_measured, retired = _activation_bits(name, events)
        # Canonical duty_to_measure semantics: a row remains owed until it has BOTH
        # fired and been measured.  A mechanism-only measurement without a matching
        # fired training arm (for example HeadOffsetSolver) does not discharge duty.
        if retired or (ever_fired and ever_measured):
            continue
        flags = factories[name]["flags"]
        sig = significance.get(name, {})
        owed.append(
            {
                "lever": name,
                "activation_state": state,
                "ever_fired": ever_fired,
                "ever_measured": ever_measured,
                "flags": flags,
                "flag_count": len(flags),
                "stratum": _stratum(name, flags),
                "measurement_cost_epochs": factories[name]["measurement_cost_epochs"],
                "est_delta_s": sig.get("est_delta_s") if isinstance(sig.get("est_delta_s"), (int, float)) else None,
                "delta_s_label": sig.get("delta_s_label"),
                "axis": sig.get("axis"),
                "descriptor_provenance": f"git:{SOURCE_EPOCH}:{DSL_PATH}+activation_ledger_exact_name_join",
            }
        )
    if len(owed) != 72:
        raise RuntimeError(f"source-epoch duty bank drift: expected 72, derived {len(owed)}")

    complete, audit = _audit_measured_outcomes(events, factories)
    chronological_folds = max(0, len(complete) - 1)
    policy_names = (
        "vime_style_information_gain",
        "posterior_sampling",
        "pseudo_count_anti_starvation",
        "double_q_debiased_selection",
        "current_p8",
        "cheapest_first",
        "family_round_robin",
        "random_seed_0",
    )
    policies = {
        name: {
            "status": "NOT_IDENTIFIED",
            "chronological_folds": chronological_folds,
            "simple_regret": None,
            "top_k_hit_rate": None,
            "measurements_to_first_confirmed_improvement": None,
            "calibration": None,
            "reason": "fewer_than_two_custody_complete_outcome_cost_rows",
        }
        for name in policy_names
    }
    p8 = _p8_rows(owed, significance)
    cost_known = [row for row in owed if row["measurement_cost_epochs"] is not None]
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cost_known:
        by_stratum[row["stratum"]].append(row)
    for rows in by_stratum.values():
        rows.sort(key=lambda row: (int(row["measurement_cost_epochs"]), row["lever"]))
    fallback: list[dict[str, Any]] = []
    strata = sorted(by_stratum)
    while any(by_stratum.values()):
        for stratum in strata:
            if by_stratum[stratum]:
                fallback.append(by_stratum[stratum].pop(0))
    return {
        "schema_version": "digs3-finite-bank-backtest-v1",
        "containment": "$0 local; descriptors/backtest only; no controller actuation",
        "source_epoch": SOURCE_EPOCH,
        "source_epoch_factory_count": len(factories),
        "owed_bank_count": len(owed),
        "owed_bank": owed,
        "historical_measured_event_count": sum(row.get("event") == "measured" for row in events),
        "custody_complete_outcome_cost_row_count": len(complete),
        "custody_complete_outcome_cost_rows": complete,
        "outcome_custody_audit": audit,
        "chronological_test_fold_count": chronological_folds,
        "policy_backtest": policies,
        "current_p8_reconstruction": p8,
        "current_p8_top_12": list(p8[:12]),
        "fallback_preview_first_12": [row["lever"] for row in fallback[:12]],
        "double_q_guard": {
            "required_future_protocol": (
                "at every chronological fold, select with a model fit only on one past-data split "
                "and evaluate the selected arm with a disjoint past-data split; never use the same "
                "posterior noise for selection and evaluation"
            ),
            "active_in_this_receipt": False,
            "reason": "zero chronological test folds; debiasing cannot create support or outcomes",
        },
        "verdict": {
            "winner_vs_p8": "NONE_IDENTIFIED; P8_NOT_BEATEN",
            "backtested_regret": None,
            "backtested_regret_label": "NOT_IDENTIFIED",
            "primary_falsifier": "FIRED_UNCALIBRATED_UNCERTAINTY",
            "operational_fallback": "STRATIFIED_CHEAPEST_FIRST",
            "rnd_icm_status": "REJECTED_NO_THEATER",
        },
    }


def _parse_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                row = dict(row)
                row["_line"] = line_no
                rows.append(row)
    return rows


def _first_loss_plateau(
    rows: list[dict[str, Any]], *, min_stage_epochs: int, windows: int = 4, rel_eps: float = 1e-4
) -> dict[str, Any]:
    """Exact replay of the trainer's `_stage_converged` slope test on logged loss_terms rows."""
    losses = [
        (int(row["ep"]), float(row["total"]))
        for row in rows
        if row.get("stage") == "loss_terms"
        and isinstance(row.get("ep"), (int, float))
        and isinstance(row.get("total"), (int, float))
    ]
    w = max(2, int(windows))
    for index in range(len(losses)):
        prefix = losses[: index + 1]
        if len(prefix) < max(1, int(min_stage_epochs)) or len(prefix) < w:
            continue
        ys = [pair[1] for pair in prefix[-w:]]
        xm = (w - 1.0) / 2.0
        ym = sum(ys) / w
        denom = sum((float(i) - xm) ** 2 for i in range(w))
        slope = sum((float(i) - xm) * (value - ym) for i, value in enumerate(ys)) / denom
        relative = None if abs(ym) <= 0.0 else abs(slope / ym)
        plateau = slope == 0.0 if relative is None else relative <= float(rel_eps)
        if plateau:
            return {
                "fired": True,
                "epoch": prefix[-1][0],
                "relative_slope_abs": relative,
                "window": w,
                "min_stage_epochs": int(min_stage_epochs),
                "rel_eps": float(rel_eps),
            }
    return {
        "fired": False,
        "epoch": None,
        "relative_slope_abs": None,
        "window": w,
        "min_stage_epochs": int(min_stage_epochs),
        "rel_eps": float(rel_eps),
        "loss_rows": len(losses),
    }


def _dseg_direction_after(epoch: int | None, verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    if epoch is None:
        return {"status": "NO_TRIGGER"}
    before = [row for row in verdicts if int(row["epoch"]) <= int(epoch)]
    after = [row for row in verdicts if int(row["epoch"]) > int(epoch)]
    if not before or not after:
        return {"status": "NO_BRACKETING_NEXT_VERDICT"}
    left, right = before[-1], after[0]
    delta = float(right["d_seg"]) - float(left["d_seg"])
    return {
        "status": "DESCENDING" if delta < 0 else "RISING_OR_FLAT",
        "before_epoch": int(left["epoch"]),
        "before_d_seg": float(left["d_seg"]),
        "after_epoch": int(right["epoch"]),
        "after_d_seg": float(right["d_seg"]),
        "delta_d_seg": delta,
    }


def _s4_receipt() -> dict[str, Any]:
    from tac.witness_control.event_wirings import annulus_plateau_event
    from tac.witness_control.powerlaw_exit import powerlaw_meat_exit

    traces: list[dict[str, Any]] = []
    global_branch_groups: dict[str, set[str]] = defaultdict(set)
    for rel in TRACE_DIRS:
        run_dir = REPO / rel
        log_path = run_dir / "run.log"
        if not log_path.is_file():
            continue
        rows = _parse_trace(log_path)
        verdicts = [row for row in rows if row.get("stage") == "verdict"]
        checkpoints = [row for row in rows if row.get("stage") == "checkpoint"]
        transitions = [
            row
            for row in rows
            if row.get("stage") in {
                "curriculum_transition",
                "tau_advance",
                "tau_advance_armed",
                "muon_finisher_switch",
                "stage_transition_reset_moments",
            }
        ]
        topology = [
            row
            for row in rows
            if row.get("stage") in {"lane_band_would_fire", "annulus_convergence"}
        ]
        lane_rows = [row for row in rows if row.get("stage") == "lane_band_would_fire"]
        annulus_series = []
        for row in rows:
            if row.get("stage") != "annulus_convergence" or not isinstance(row.get("epoch"), (int, float)):
                continue
            threshold = row.get("threshold")
            if isinstance(threshold, dict) and isinstance(threshold.get("annulus_flip_frac"), (int, float)):
                annulus_series.append((float(row["epoch"]), float(threshold["annulus_flip_frac"])))
        annulus_replay = annulus_plateau_event(annulus_series)
        total_dseg = [(float(row["epoch"]), float(row["d_seg"])) for row in verdicts]
        if len(total_dseg) >= 8:
            powerlaw_replay = powerlaw_meat_exit(total_dseg, min_points=8, seed=0)
            # The complete nested fits are unnecessary for this trace-level sensor receipt.
            powerlaw_replay.pop("per_class", None)
        else:
            powerlaw_replay = {
                "exhausted": False,
                "status": f"INSUFFICIENT_POINTS_{len(total_dseg)}_LT_8",
            }
        default_loss_plateau = _first_loss_plateau(rows, min_stage_epochs=150)
        live_loss_plateau = _first_loss_plateau(rows, min_stage_epochs=250)
        default_loss_plateau["next_dseg_direction"] = _dseg_direction_after(
            default_loss_plateau.get("epoch"), verdicts
        )
        live_loss_plateau["next_dseg_direction"] = _dseg_direction_after(
            live_loss_plateau.get("epoch"), verdicts
        )
        # A valid pair must explicitly name one common checkpoint and distinct stay/advance
        # policies.  Similar epoch numbers or independent runs are not counterfactual custody.
        branch_rows = [
            row
            for row in rows
            if row.get("common_checkpoint_sha256")
            and row.get("branch_policy") in {"stay", "advance"}
            and row.get("common_horizon") is not None
        ]
        for row in branch_rows:
            key = f"{row['common_checkpoint_sha256']}|{row['common_horizon']}"
            global_branch_groups[key].add(str(row["branch_policy"]))
        traces.append(
            {
                "run_dir": rel,
                "run_log_sha256": _sha256(log_path),
                "run_log_bytes": log_path.stat().st_size,
                "verdict_rows": len(verdicts),
                "checkpoint_rows": len(checkpoints),
                "transition_rows": len(transitions),
                "topology_sensor_rows": len(topology),
                "lane_nucleus_would_fire_rows": sum(bool(row.get("would_fire")) for row in lane_rows),
                "annulus_plateau_replay": annulus_replay,
                "powerlaw_meat_replay": powerlaw_replay,
                "loss_slope_plateau_default_min150_replay": default_loss_plateau,
                "loss_slope_plateau_live_min250_replay": live_loss_plateau,
                "explicit_common_horizon_branch_rows": len(branch_rows),
                "epoch_range": (
                    [min(int(row["epoch"]) for row in verdicts), max(int(row["epoch"]) for row in verdicts)]
                    if verdicts
                    else None
                ),
                "best_d_seg": min((float(row["d_seg"]) for row in verdicts), default=None),
                "last_d_seg": float(verdicts[-1]["d_seg"]) if verdicts else None,
                "seg_forms": sorted({str(row.get("seg_form")) for row in verdicts if row.get("seg_form")}),
            }
        )
    common_pairs = sum(policies == {"stay", "advance"} for policies in global_branch_groups.values())
    loss150_fires = [
        row["loss_slope_plateau_default_min150_replay"]
        for row in traces
        if row["loss_slope_plateau_default_min150_replay"]["fired"]
    ]
    loss250_fires = [
        row["loss_slope_plateau_live_min250_replay"]
        for row in traces
        if row["loss_slope_plateau_live_min250_replay"]["fired"]
    ]
    annulus_fires = sum(bool(row["annulus_plateau_replay"].get("fired")) for row in traces)
    powerlaw_fires = sum(bool(row["powerlaw_meat_replay"].get("exhausted")) for row in traces)
    sensor_tests = {
        "existing_topology_full_facet_gates": {
            "trace_observations": sum(row["topology_sensor_rows"] for row in traces),
            "lane_nucleus_would_fire_rows": sum(row["lane_nucleus_would_fire_rows"] for row in traces),
            "annulus_plateau_final_replay_fires": annulus_fires,
            "causal_common_horizon_pairs": common_pairs,
            "beats_existing_gates": False,
            "status": "REFERENCE_GUARDS_PRESERVED",
        },
        "powerlaw_meat": {
            "trace_observations": sum(row["verdict_rows"] for row in traces),
            "final_replay_fires": powerlaw_fires,
            "causal_common_horizon_pairs": common_pairs,
            "beats_existing_gates": False,
            "status": "SENSOR_ONLY_NO_COUNTERFACTUAL_ADVANTAGE_RECEIPT",
        },
        "rolling_slope_plateau": {
            "trace_observations": sum(row["verdict_rows"] for row in traces),
            "default_min150_fires": len(loss150_fires),
            "live_min250_fires": len(loss250_fires),
            "default_min150_next_dseg_directions": [row["next_dseg_direction"] for row in loss150_fires],
            "live_min250_next_dseg_directions": [row["next_dseg_direction"] for row in loss250_fires],
            "causal_common_horizon_pairs": common_pairs,
            "beats_existing_gates": False,
            "status": "SENSOR_ONLY_NO_COUNTERFACTUAL_ADVANTAGE_RECEIPT",
        },
        "ncde_forecast": {
            "trace_observations": 0,
            "causal_common_horizon_pairs": common_pairs,
            "beats_existing_gates": False,
            "status": "NO_FIT_STABILITY_RECEIPT_IN_SELECTED_TRACES",
        },
        "transactional_upper_vs_lower_confidence": {
            "trace_observations": 0,
            "causal_common_horizon_pairs": common_pairs,
            "beats_existing_gates": False,
            "status": "UNAVAILABLE_WITHOUT_STAY_ADVANCE_LOSS_PAIRS",
        },
    }
    return {
        "schema_version": "digs4-option-trace-backtest-v1",
        "containment": "$0 local read-only trace audit; no controller actuation",
        "trace_count": len(traces),
        "traces": traces,
        "common_checkpoint_common_horizon_stay_advance_pair_count": common_pairs,
        "sensor_tests": sensor_tests,
        "any_sensor_beats_existing_gates_on_traces": False,
        "primary_falsifier": "FIRED_NO_AFFORDABLE_COMMON_HORIZON_COUNTERFACTUAL",
        "schedule_verdict": "PRESERVE_FIXED_SCHEDULE",
        "option_contract": {
            "state": "checkpoint hash + stage/tau rung + optimizer/controller/resume state",
            "eligibility": "existing #315 topology/nucleus and full-facet safety guards",
            "switching_cost": "stage reset/rewarmup plus equal-horizon score loss and runtime",
            "minimum_dwell": "typed launch value --curriculum-min-stage-epochs=250",
            "hysteresis": "advance only after dwell and confidence-separated common-horizon loss",
            "rollback": "pre-boundary complete EMA/resume checkpoint; preserve each stage checkpoint",
            "decision": (
                "advance only if eligible and UCB(L_advance + switching_cost) < LCB(L_stay); "
                "otherwise stay, or insufficient-information"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "experiments/results/digs3_s4_20260713",
    )
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    s3 = _s3_receipt()
    s4 = _s4_receipt()
    _atomic_json(out_dir / "s3_finite_bank_backtest.json", s3)
    _atomic_json(out_dir / "s4_option_trace_backtest.json", s4)
    manifest = {
        "schema_version": "digs3-s4-receipt-manifest-v1",
        "tool": "tools/digs3_s4_backtest.py",
        "source_epoch": SOURCE_EPOCH,
        "seed": 0,
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE",
        "actuation": "NONE",
        "pointer_moved": False,
        "outputs": {},
    }
    for name in ("s3_finite_bank_backtest.json", "s4_option_trace_backtest.json"):
        path = out_dir / name
        manifest["outputs"][name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    _atomic_json(out_dir / "receipt_manifest.json", manifest)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir.relative_to(REPO)),
                "s3_owed": s3["owed_bank_count"],
                "s3_complete_rows": s3["custody_complete_outcome_cost_row_count"],
                "s3_folds": s3["chronological_test_fold_count"],
                "s3_verdict": s3["verdict"],
                "s4_traces": s4["trace_count"],
                "s4_common_horizon_pairs": s4[
                    "common_checkpoint_common_horizon_stay_advance_pair_count"
                ],
                "s4_any_sensor_beats": s4["any_sensor_beats_existing_gates_on_traces"],
                "s4_verdict": s4["schedule_verdict"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(main())
