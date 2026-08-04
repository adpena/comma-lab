#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Replay TJ1 trajectory-derived stopping from recorded receipts only.

No SegNet/PoseNet forward pass is imported or executed here.  The tool reads
solver receipts already written by SQ1/CW1/NG1, fits the canonical stopping law,
and writes a positive-control receipt plus an SQ2 validation target.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.optimization.trajectory_stopping import (  # noqa: E402
    TRAJECTORY_STOPPING_LAW_REF,
    TrajectoryPoint,
    TrajectoryStopConfig,
    byte_score_units,
    evaluate_trajectory_stop,
    projection_interval,
    seg_flip_score_units,
)

SCHEMA = "ddm_tj1_trajectory_replay.v1"
RECEIPT_DIR = REPO / ".omx" / "research" / "ddm_tj1_20260805"
DEFAULT_STAGE25 = Path("/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32.json")
DEFAULT_AGG25 = Path("/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_aggregate_n32.json")
DEFAULT_STAGE50 = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap50_cw1.json"
)
DEFAULT_AGG50 = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_aggregate_n32_uncap50_cw1.json"
)
DEFAULT_SQ2_PARTIAL = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap100_sq2.json"
)
DEFAULT_NG1_SWEEP = REPO / ".omx" / "research" / "ddm_ng1_20260805" / "cap_artifact_sweep.jsonl"
DEFAULT_HOT_STATE = REPO / ".omx" / "state" / "main_hot_state.md"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"could not read {path}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError(f"{path} root must be a JSON object")
    return obj


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise RuntimeError(f"{path}:{lineno} root must be a JSON object")
        rows.append(obj)
    return rows


def _aggregate_curve(stage_receipt: dict[str, Any], *, max_step: int | None = None) -> list[dict[str, float]]:
    rows = stage_receipt.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("stage receipt has no rows")
    accum: dict[int, float] = {}
    counts: dict[int, int] = {}
    for row in rows:
        curve = row.get("solved_convergence_curve")
        if not isinstance(curve, list) or not curve:
            raise RuntimeError("stage row lacks solved_convergence_curve")
        for point in curve:
            step = int(point["step"])
            if max_step is not None and step > max_step:
                continue
            accum[step] = accum.get(step, 0.0) + float(point["proxy_flips"])
            counts[step] = counts.get(step, 0) + 1
    n_rows = len(rows)
    full_steps = [step for step in sorted(accum) if counts[step] == n_rows]
    if len(full_steps) < 2:
        raise RuntimeError("stage receipt has fewer than two full-population curve steps")
    return [{"step": float(step), "objective": float(accum[step])} for step in full_steps]


def _points(curve: list[dict[str, float]]) -> tuple[TrajectoryPoint, ...]:
    return tuple(
        TrajectoryPoint(compute=float(point["step"]), objective=float(point["objective"]))
        for point in curve
    )


def _realized_summary(stage_receipt: dict[str, Any], aggregate_receipt: dict[str, Any]) -> dict[str, Any]:
    rows = stage_receipt["rows"]
    total_before = int(sum(int(row["flips_before"]) for row in rows))
    total_after = int(sum(int(row["S4_solvedpaint_flips_after"]) for row in rows))
    total_described = int(sum(int(row["described_in_band"]) for row in rows))
    eta_from_rows = (total_before - total_after) / float(total_described)
    eta_from_aggregate = float(
        aggregate_receipt["S4_residual"]["S4_solvedpaint"]["eta_net_pooled"]
    )
    return {
        "n_pairs": len(rows),
        "flips_before": total_before,
        "flips_after_realized": total_after,
        "described_in_band": total_described,
        "eta_from_rows": eta_from_rows,
        "eta_from_aggregate": eta_from_aggregate,
        "aggregate_eta_matches_rows": abs(eta_from_rows - eta_from_aggregate) <= 5.0e-4,
    }


def _eta_interval(
    *,
    interval_low_objective: float,
    interval_high_objective: float,
    flips_before: int,
    described_in_band: int,
) -> dict[str, float]:
    eta_low = (flips_before - interval_high_objective) / float(described_in_band)
    eta_high = (flips_before - interval_low_objective) / float(described_in_band)
    return {
        "eta_low": float(eta_low),
        "eta_high": float(eta_high),
    }


def _control(
    *,
    name: str,
    curve_stage_receipt: dict[str, Any],
    realized_stage_receipt: dict[str, Any],
    aggregate_receipt: dict[str, Any],
    max_step: int,
    cfg: TrajectoryStopConfig,
) -> dict[str, Any]:
    curve = _aggregate_curve(curve_stage_receipt, max_step=max_step)
    decision = evaluate_trajectory_stop(
        _points(curve),
        cfg,
        safety_bound_compute=float(max_step),
    )
    realized = _realized_summary(realized_stage_receipt, aggregate_receipt)
    return {
        "name": name,
        "max_step": max_step,
        "curve": curve,
        "decision": decision.to_payload(),
        "realized": realized,
        "pass": (
            decision.stop_reason == "safety_bound_REPORTED"
            and decision.bound_reported is True
            and decision.marginal_score_gain_per_compute
            > decision.threshold_score_gain_per_compute
        ),
    }


def _ng1_class_map(ng1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    recipient_by_id = {
        "sq1_25_step_solved_paint": "sq1 solved-paint loop",
        "sq1_50_step_uncap_cw1": "sq1 solved-paint loop",
        "gn_pose_solve_850": "terminal_pose_gn marginal floor",
        "q31_50_step_q3_constrained": "q31 Q3 constrained paint",
        "et1_budget_ladder_eta_floor": "ET1 budget ladder",
        "na3_lr2_solved_paint_ladder": "NA3 solved-paint ladder",
    }
    for row in ng1_rows:
        if row.get("record_type") != "cap_artifact_row":
            continue
        row_id = str(row.get("id", ""))
        regrade = str(row.get("regrade", ""))
        if regrade == "FLOOR-NOT-CONVERGED":
            family = "cap_bound_floor_not_converged"
            cure = "trajectory-derived stop law plus adaptive depth allocation"
            fire_order = "extend or waterfill before any convergence/promotion wording"
        elif regrade == "GENUINE":
            family = "genuine_stop_or_stale_off_chain"
            cure = "consume current cured surface, do not rerun stale cap premise"
            fire_order = str(row.get("current_live_disposition", "fold if stale/off-chain"))
        else:
            family = "unclassified_cap_result"
            cure = "manual review"
            fire_order = "HOLD"
        out.append(
            {
                "id": row_id,
                "cap_value": row.get("cap_value"),
                "regrade": regrade,
                "family": family,
                "recipient": recipient_by_id.get(row_id, "manual recipient required"),
                "existing_cure_surface": cure,
                "fire_order": fire_order,
                "source_path": row.get("source_path"),
                "last_delta_evidence": row.get("last_delta_evidence"),
            }
        )
    return out


def build_replay_payload(
    *,
    stage25_path: Path = DEFAULT_STAGE25,
    agg25_path: Path = DEFAULT_AGG25,
    stage50_path: Path = DEFAULT_STAGE50,
    agg50_path: Path = DEFAULT_AGG50,
    sq2_partial_path: Path = DEFAULT_SQ2_PARTIAL,
    ng1_sweep_path: Path = DEFAULT_NG1_SWEEP,
    hot_state_path: Path = DEFAULT_HOT_STATE,
) -> dict[str, Any]:
    stage25 = _load_json(stage25_path)
    agg25 = _load_json(agg25_path)
    stage50 = _load_json(stage50_path)
    agg50 = _load_json(agg50_path)
    cfg = TrajectoryStopConfig(
        score_units_per_objective=seg_flip_score_units(),
        marginal_score_gain_per_compute=byte_score_units(),
    )

    # The 50-step CW1 receipt contains the full 0..50 trajectory.  Its prefix is
    # the scorer-free positive control for the 25-step cap artifact.
    control25 = _control(
        name="sq1_prefix_25_from_cw1_50_step_receipt",
        curve_stage_receipt=stage50,
        realized_stage_receipt=stage25,
        aggregate_receipt=agg25,
        max_step=25,
        cfg=cfg,
    )
    control50 = _control(
        name="sq1_full_50_cw1_receipt",
        curve_stage_receipt=stage50,
        realized_stage_receipt=stage50,
        aggregate_receipt=agg50,
        max_step=50,
        cfg=cfg,
    )

    interval25_to_50 = projection_interval(_points(control25["curve"]), cfg, target_compute=50.0)
    realized50 = control50["realized"]
    eta25_to_50 = _eta_interval(
        interval_low_objective=interval25_to_50.objective_low,
        interval_high_objective=interval25_to_50.objective_high,
        flips_before=int(realized50["flips_before"]),
        described_in_band=int(realized50["described_in_band"]),
    )
    step50_after = float(realized50["flips_after_realized"])

    interval50_to_100 = projection_interval(_points(control50["curve"]), cfg, target_compute=100.0)
    eta50_to_100 = _eta_interval(
        interval_low_objective=interval50_to_100.objective_low,
        interval_high_objective=interval50_to_100.objective_high,
        flips_before=int(realized50["flips_before"]),
        described_in_band=int(realized50["described_in_band"]),
    )

    sq2_partial: dict[str, Any] = {"path": str(sq2_partial_path), "present": sq2_partial_path.exists()}
    if sq2_partial_path.exists():
        sq2_obj = _load_json(sq2_partial_path)
        sq2_rows = sq2_obj.get("rows", [])
        sq2_partial.update(
            {
                "rows_present": len(sq2_rows),
                "rows_expected": len(stage50["rows"]),
                "complete": len(sq2_rows) == len(stage50["rows"]),
                "solver": sq2_obj.get("solver"),
            }
        )
        if sq2_rows:
            partial_curve = _aggregate_curve(sq2_obj)
            partial_decision = evaluate_trajectory_stop(
                _points(partial_curve),
                cfg,
                safety_bound_compute=float(sq2_obj.get("solver", {}).get("steps", 100)),
            )
            sq2_partial["partial_curve_last"] = partial_curve[-1]
            sq2_partial["partial_stop_decision"] = partial_decision.to_payload()

    controls_pass = all(item["pass"] for item in (control25, control50)) and (
        eta25_to_50["eta_low"]
        <= float(realized50["eta_from_aggregate"])
        <= eta25_to_50["eta_high"]
    )
    hot_state = hot_state_path.read_text(encoding="utf-8", errors="replace") if hot_state_path.exists() else ""
    return {
        "schema": SCHEMA,
        "law_ref": TRAJECTORY_STOPPING_LAW_REF,
        "score_claim": False,
        "promotion_eligible": False,
        "scorer_forwards_executed": 0,
        "source_receipts": {
            "stage25": str(stage25_path),
            "aggregate25": str(agg25_path),
            "stage50": str(stage50_path),
            "aggregate50": str(agg50_path),
            "sq2_partial": str(sq2_partial_path),
            "ng1_cap_sweep": str(ng1_sweep_path),
        },
        "controls": [control25, control50],
        "prefix25_projection_to_step50": {
            **interval25_to_50.to_payload(),
            **eta25_to_50,
            "measured_step50_realized_flips_after": step50_after,
            "measured_step50_eta": float(realized50["eta_from_aggregate"]),
            "measured_step50_inside_interval": (
                interval25_to_50.objective_low <= step50_after <= interval25_to_50.objective_high
            ),
        },
        "sq2_validation_target": {
            **interval50_to_100.to_payload(),
            **eta50_to_100,
            "status": "PENDING_COMPLETE_RECEIPT",
            "validation_rule": "when complete n32 SQ2 lands, realized flips_after and eta must be compared to this interval before any convergence wording",
            "partial": sq2_partial,
        },
        "ng1_class_map": _ng1_class_map(_load_jsonl(ng1_sweep_path)),
        "positive_controls_pass": controls_pass,
        "frontier_status": {
            "own_vehicle": "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]",
            "contest_pointer": "S = 0.1910828242 [contest-CPU] borrowed/unmoved",
            "main_hot_state_excerpt_found": "0.7539807296911207" in hot_state,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ddm_tj1 trajectory-derived stopping replay",
        "",
        f"Schema: `{payload['schema']}`. Law: `{payload['law_ref']}`.",
        "Scorer forwards executed by this replay: `0`.",
        "",
        "## Positive controls",
        "",
    ]
    for control in payload["controls"]:
        decision = control["decision"]
        realized = control["realized"]
        lines.extend(
            [
                f"- `{control['name']}`: step `{control['max_step']}`, stop_reason "
                f"`{decision['stop_reason']}`, eta `{realized['eta_from_aggregate']:.12f}`, "
                f"marginal S/step `{decision['marginal_score_gain_per_compute']:.12g}`.",
            ]
        )
    proj = payload["prefix25_projection_to_step50"]
    lines.extend(
        [
            "",
            "## Prefix projection",
            "",
            f"Step-25 projection to step 50 predicts objective "
            f"`[{proj['objective_low']:.6f}, {proj['objective_high']:.6f}]` and eta "
            f"`[{proj['eta_low']:.12f}, {proj['eta_high']:.12f}]`.",
            f"Measured step-50 eta `{proj['measured_step50_eta']:.12f}`; inside interval: "
            f"`{proj['measured_step50_inside_interval']}`.",
            "",
            "## SQ2 target",
            "",
        ]
    )
    sq2 = payload["sq2_validation_target"]
    lines.extend(
        [
            f"Target compute `{sq2['target_compute']}` predicts objective "
            f"`[{sq2['objective_low']:.6f}, {sq2['objective_high']:.6f}]` and eta "
            f"`[{sq2['eta_low']:.12f}, {sq2['eta_high']:.12f}]`.",
            f"Status: `{sq2['status']}`. Partial rows present: "
            f"`{sq2['partial'].get('rows_present', 0)}/{sq2['partial'].get('rows_expected', 'unknown')}`.",
            "",
            "## NG1 Class Map",
            "",
        ]
    )
    for row in payload["ng1_class_map"]:
        lines.append(
            f"- `{row['id']}` -> `{row['family']}`; recipient `{row['recipient']}`; "
            f"fire-order `{row['fire_order']}`."
        )
    lines.extend(
        [
            "",
            "## Frontier",
            "",
            f"Own vehicle: `{payload['frontier_status']['own_vehicle']}`.",
            f"Contest pointer: `{payload['frontier_status']['contest_pointer']}`.",
            "Pointer moved by TJ1: `False`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RECEIPT_DIR)
    parser.add_argument("--stage25", type=Path, default=DEFAULT_STAGE25)
    parser.add_argument("--agg25", type=Path, default=DEFAULT_AGG25)
    parser.add_argument("--stage50", type=Path, default=DEFAULT_STAGE50)
    parser.add_argument("--agg50", type=Path, default=DEFAULT_AGG50)
    parser.add_argument("--sq2-partial", type=Path, default=DEFAULT_SQ2_PARTIAL)
    parser.add_argument("--ng1-sweep", type=Path, default=DEFAULT_NG1_SWEEP)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args(argv)

    payload = build_replay_payload(
        stage25_path=args.stage25,
        agg25_path=args.agg25,
        stage50_path=args.stage50,
        agg50_path=args.agg50,
        sq2_partial_path=args.sq2_partial,
        ng1_sweep_path=args.ng1_sweep,
    )
    out_json = args.out_dir / "trajectory_replay.json"
    out_md = args.out_dir / "tj1_summary.md"
    _write_json(out_json, payload)
    _write_markdown(out_md, payload)

    if args.register:
        from tac.canonical_equations.trajectory_derived_stopping_20260805 import (
            populate_trajectory_derived_stopping_law_v1,
        )

        equation = populate_trajectory_derived_stopping_law_v1(
            source_receipt=out_json,
            agent="codex",
            subagent_id="tj1",
        )
        registration = {
            "schema": "ddm_tj1_canonical_equation_registration.v1",
            "equation_id": equation.equation_id,
            "source_receipt": str(out_json),
            "registered": True,
        }
        _write_json(args.out_dir / "canonical_equation_registration.json", registration)

    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    if not payload["positive_controls_pass"]:
        print("positive controls failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
