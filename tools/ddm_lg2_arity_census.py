#!/usr/bin/env python
"""ddm_lg2 — three measurements of ONE defect genus: ARITY MISMATCH.

An instrument suffers an arity mismatch when the quantity it reports has a
different support than the thing it claims to constrain.  Three landed
instances, one subcommand each; every subcommand REPORTS ITS DENOMINATOR and
emits ``VACUOUS`` (never ``PASS``) on an empty scope, because an empty scope
that emits the same symbol as a clean full scope is the day's recurring bug.

  smoke-scope        (#821)  ``check_lane_scripts_have_e2e_smoke_proof`` iterates
                             over 204 lane scripts, but the evidence function
                             ``canonical_local_auth_eval_smoke.smoke_archive``
                             does not read its ``lane_name`` argument in any
                             stage.  So N per-lane assertions are ONE assertion
                             fanned out N times.  This measures the fan-out
                             factor directly from the stored proofs.

  ladder-authority   (#822)  ``tools/supervise_ddm_b4s_burn4.py`` raises
                             ``--lane-guard-lambda-init`` by ``LG1_LAMBDA_STEP``
                             on a LANE_EROSION rollback.  The rung equals one
                             ``lambda_step_cap``, which is exactly what one
                             slack gate removes, so the raise is an INITIAL
                             CONDITION on a contracting map, not a floor.  This
                             replays the shipped ``dual_ascent`` over a real
                             ``g`` series and reports whether the ladder climbs.

  bracket-direction  (#871)  ``experiments/ddm_v4d_resolve.py`` brackets with
                             ``for sign in (1.0, -1.0): ... break``, so the
                             ``-1.0`` direction is never evaluated whenever
                             ``+1.0`` improves at all.  This reads the shipped
                             pw1 probe receipt and reports the occupancy of that
                             binary — the pw1 discriminator applied to a
                             direction choice rather than a menu.

Authority: ``[macOS-CPU advisory]``; ``research_only=True``; ``score_claim=False``.
Reads only; runs no training, no scorer job, and mutates nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Emitted instead of PASS/FAIL when the scope is empty.  An instrument that
#: reports the same symbol for "nothing to check" and "everything checked and
#: clean" cannot be trusted in either direction.
VACUOUS = "VACUOUS"


# --------------------------------------------------------------------------
# #821 — smoke-proof scope
# --------------------------------------------------------------------------
def smoke_proof_scope_census(
    repo_root: Path | None = None,
    proofs_rel: str = ".omx/state/lane_e2e_smoke_proofs.json",
) -> dict[str, Any]:
    """Measure the fan-out factor of the lane E2E smoke-proof gate.

    The gate's scope variable is the lane script; its evidence is a proof row.
    If every stored proof carries the SAME ``archive_sha256`` /
    ``fixture_archive`` / ``stages_passed``, then the evidence does not vary
    with the scope variable and the honest population is the number of DISTINCT
    evidence tuples, not the number of lane scripts.

    ``distinct_evidence_tuples`` is that honest population.  ``fan_out`` is how
    many times each distinct fact is asserted.
    """
    root = repo_root or REPO_ROOT
    scripts = sorted((root / "scripts").glob("remote_lane_*.sh"))
    proofs_path = root / proofs_rel
    proofs: dict[str, Any] = {}
    if proofs_path.exists():
        try:
            loaded = json.loads(proofs_path.read_text())
            if isinstance(loaded, dict):
                proofs = loaded
        except (json.JSONDecodeError, OSError):
            proofs = {}

    waived = [
        s.stem for s in scripts if "# E2E_SMOKE_OPT_OUT:" in s.read_text(errors="ignore")
    ]
    tuples = {
        (
            str(v.get("fixture_archive")),
            str(v.get("archive_sha256")),
            tuple(v.get("stages_passed") or ()),
            str(v.get("submission_dir")),
        )
        for v in proofs.values()
        if isinstance(v, dict)
    }
    n_scripts = len(scripts)
    n_proofs = len(proofs)
    n_tuples = len(tuples)
    elapsed = [
        float(v["elapsed_seconds"])
        for v in proofs.values()
        if isinstance(v, dict) and isinstance(v.get("elapsed_seconds"), (int, float))
    ]
    stamps = sorted({str(v.get("timestamp_utc")) for v in proofs.values() if isinstance(v, dict)})

    # R2 self-review: n_tuples == 0 means every stored row was unusable, which is
    # an EMPTY scope wearing a full-looking denominator — it must not fall through
    # to ARITY_MISMATCH (which would read as a finding drawn from nothing).
    if n_scripts == 0 or n_proofs == 0 or n_tuples == 0:
        verdict = VACUOUS
    elif n_proofs < 2:
        # R1 self-review: a single proof cannot exhibit fan-out either way.
        # Reporting it as "scope varies" would be a clean verdict drawn from a
        # population that cannot support one — the vacuity bug at n=1.
        verdict = "SINGLE_PROOF_UNDETERMINED"
    elif n_tuples <= 1:
        verdict = "ARITY_MISMATCH"
    else:
        verdict = "SCOPE_VARIES_WITH_EVIDENCE"

    return {
        "measurement": "smoke_proof_scope_census",
        "denominator_lane_scripts": n_scripts,
        "denominator_stored_proofs": n_proofs,
        "waived_opt_out": len(waived),
        "distinct_evidence_tuples": n_tuples,
        "fan_out": (n_proofs / n_tuples) if n_tuples else None,
        "total_elapsed_seconds_all_proofs": round(sum(elapsed), 3) if elapsed else None,
        "distinct_timestamps": len(stamps),
        "timestamp_span": [stamps[0], stamps[-1]] if stamps else None,
        "verdict": verdict,
        "reading": (
            "evidence does not vary with the scope variable: the honest population is "
            f"{n_tuples}, not {n_scripts}"
            if verdict == "ARITY_MISMATCH"
            else "empty scope — nothing measured"
            if verdict == VACUOUS
            else "one proof — population too small to decide"
            if verdict == "SINGLE_PROOF_UNDETERMINED"
            else "evidence varies across scope items; per-item iteration is warranted"
        ),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
    }


# --------------------------------------------------------------------------
# #822 — escalation-ladder authority
# --------------------------------------------------------------------------
def ladder_authority(
    rung: float,
    eta_lambda: float,
    lambda_step_cap: float,
    lambda_max: float,
    g_series: list[float],
    escalate_at: float,
) -> dict[str, Any]:
    """Does the supervisor's rollback-and-raise ladder actually climb?

    The dual is a clipped rectified integrator::

        step   = clip(eta * g, -cap, +cap)
        lambda = clip(lambda + step, 0, lambda_max)

    A rollback relaunch sets ``lambda = rung`` and the dual then runs.  When
    ``g < 0`` and ``|eta*g| >= cap`` the step is the full ``-cap``, so a rung of
    height ``cap`` is erased by ONE gate.  ``rung`` is therefore an initial
    condition on a contracting map, not a lower bound — the distinction the
    telemetry never made.

    Reports the measured saturation fraction, the replayed trajectory, and how
    many consecutive ``g > 0`` gates the operator-escalation threshold needs.
    Returns ``VACUOUS`` on an empty series rather than a clean verdict.
    """
    n = len(g_series)
    if n == 0 or not (lambda_step_cap > 0.0):
        return {
            "measurement": "ladder_authority",
            "denominator_gates": n,
            "verdict": VACUOUS,
            "reading": "empty g series or non-positive step cap — nothing measured",
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
        }

    saturating = sum(1 for g in g_series if abs(eta_lambda * g) >= lambda_step_cap)
    n_positive = sum(1 for g in g_series if g > 0.0)
    lam = float(rung)
    traj: list[float] = []
    for g in g_series:
        step = max(-lambda_step_cap, min(lambda_step_cap, eta_lambda * g))
        lam = max(0.0, min(lambda_max, lam + step))
        traj.append(lam)
    gates_alive = sum(1 for x in traj if x > 0.0)
    # Gates of sustained violation needed to reach the operator-escalation rung
    # from zero, given the per-gate ceiling.
    gates_to_escalate = escalate_at / lambda_step_cap

    if n_positive == 0 and gates_alive == 0:
        verdict = "LADDER_INERT"
    elif gates_alive == 0:
        verdict = "LADDER_ERASED"
    else:
        verdict = "LADDER_HOLDS"

    return {
        "measurement": "ladder_authority",
        "denominator_gates": n,
        "rung": float(rung),
        "lambda_step_cap": float(lambda_step_cap),
        "rung_over_cap": float(rung) / float(lambda_step_cap),
        "eta_lambda": float(eta_lambda),
        "lambda_max": float(lambda_max),
        "gates_saturating_cap": saturating,
        "gates_saturating_cap_pct": round(100.0 * saturating / n, 2),
        "gates_with_positive_g": n_positive,
        "lambda_after_first_gate": traj[0],
        "gates_with_lambda_gt_zero": gates_alive,
        "escalate_at": float(escalate_at),
        "consecutive_violation_gates_needed_to_escalate": gates_to_escalate,
        "verdict": verdict,
        "reading": (
            "the raise is erased at the first gate; the rung is an initial condition on a "
            "contracting map, not a floor"
            if verdict in ("LADDER_INERT", "LADDER_ERASED")
            else "the raised multiplier survives at least one gate"
        ),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
    }


def _load_g_series(telemetry_paths: list[Path]) -> tuple[list[float], dict[str, Any]]:
    """Collect ``g_s_units`` plus the dual hyperparameters from lane_guard rows.

    R1 self-review: taking the FIRST row's hyperparameters silently hides a
    mid-run change, so every distinct value seen is recorded under
    ``*_distinct`` and the caller can see drift instead of inheriting it.
    """
    gs: list[float] = []
    params: dict[str, Any] = {}
    seen: dict[str, set[float]] = {k: set() for k in ("eta_lambda", "lambda_step_cap", "lambda_max")}
    for p in telemetry_paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "lane_guard":
                continue
            if "g_s_units" in row:
                gs.append(float(row["g_s_units"]))
            for k in ("eta_lambda", "lambda_step_cap", "lambda_max"):
                if k in row:
                    seen[k].add(float(row[k]))
                    if k not in params:
                        params[k] = float(row[k])
    for k, vals in seen.items():
        params[f"{k}_distinct"] = sorted(vals)
    return gs, params


# --------------------------------------------------------------------------
# #871 — bracket-direction occupancy
# --------------------------------------------------------------------------
def bracket_direction_occupancy(arms_jsonl: Path) -> dict[str, Any]:
    """Occupancy of the ``for sign in (1.0, -1.0): ... break`` binary.

    The pw1 discriminator (occupancy piled at a bound ⇒ the bound binds) applied
    to a DIRECTION choice.  A pair whose ``*_probes`` list carries exactly one
    ``phase == "probe"`` entry committed to the first direction without ever
    evaluating the second: an untested binary commitment.

    Also reports, among pairs where BOTH directions were evaluated, how often
    the second direction was the one that improved — the asymmetry the probe
    ORDER hides.
    """
    if not arms_jsonl.exists():
        return {
            "measurement": "bracket_direction_occupancy",
            "denominator_pairs": 0,
            "verdict": VACUOUS,
            "reading": f"receipt absent: {arms_jsonl}",
            "axis": "[macOS-CPU frozen-PoseNet advisory]",
            "score_claim": False,
        }
    # R2 self-review: fail LOUD on a malformed receipt.  Silently skipping bad
    # lines would shrink the denominator without saying so — the same vacuity
    # bug this tool exists to measure.
    rows = []
    for ln, line in enumerate(arms_jsonl.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{arms_jsonl}:{ln} is not valid JSON ({exc}); refusing to report a "
                "denominator that silently dropped rows"
            ) from exc
    if not rows:
        return {
            "measurement": "bracket_direction_occupancy",
            "denominator_pairs": 0,
            "verdict": VACUOUS,
            "reading": "receipt present but empty",
            "axis": "[macOS-CPU frozen-PoseNet advisory]",
            "score_claim": False,
        }

    total_mass = sum(float(r.get("arm_ab_d", 0.0)) for r in rows)
    out: dict[str, Any] = {
        "measurement": "bracket_direction_occupancy",
        "denominator_pairs": len(rows),
        "receipt": str(arms_jsonl),
        "arms": {},
        "axis": "[macOS-CPU frozen-PoseNet advisory]",
        "score_claim": False,
    }
    union_short: set[int] = set()
    for key in ("arm_a_probes", "arm_b_probes"):
        short = 0
        both = 0
        second_won = 0
        neither_won = 0
        first_won_in_both = 0
        mass = 0.0
        for i, r in enumerate(rows):
            probes = [x for x in (r.get(key) or []) if x.get("phase") == "probe"]
            if not probes:
                continue
            if len(probes) == 1:
                short += 1
                mass += float(r.get("arm_ab_d", 0.0))
                union_short.add(i)
                continue
            both += 1
            # `d_ctrl` is a PROXY for the bracket's entry `best_d` (the true entry
            # value is not recorded).  R1 self-review: the proxy is validated by
            # `consistency_first_improved_in_both_bucket` below — the `break`
            # guarantees that a first probe which truly improved leaves ONE probe,
            # so any first-probe "improvement" seen in the both-evaluated bucket
            # means the threshold is wrong.  A non-zero count invalidates the two
            # columns that use it; the untested-commitment count does not depend
            # on any threshold and stays exact.
            base = float(r.get("d_ctrl", float("inf")))
            improved = [j for j, x in enumerate(probes[:2]) if float(x["d"]) < base]
            if improved == [1]:
                second_won += 1
            elif not improved:
                neither_won += 1
            if 0 in improved:
                first_won_in_both += 1
        out["arms"][key] = {
            "pairs_first_direction_committed_untested": short,
            "pct_untested": round(100.0 * short / len(rows), 2),
            "pct_of_dpose_mass_on_untested": (
                round(100.0 * mass / total_mass, 2) if total_mass else None
            ),
            "pairs_both_directions_evaluated": both,
            "of_those_only_second_direction_improved": second_won,
            "of_those_neither_improved": neither_won,
            # MUST be 0: see the comment above.  This is the threshold's self-check.
            "consistency_first_improved_in_both_bucket": first_won_in_both,
            "threshold_proxy_valid": first_won_in_both == 0,
        }
    union_mass = sum(float(rows[i].get("arm_ab_d", 0.0)) for i in union_short)
    out["union_untested_pairs"] = len(union_short)
    out["union_pct_of_dpose_mass"] = (
        round(100.0 * union_mass / total_mass, 2) if total_mass else None
    )
    out["verdict"] = "UNTESTED_BINARY_COMMITMENT" if union_short else "BOTH_DIRECTIONS_ALWAYS_TESTED"
    out["reading"] = (
        "probe ORDER, not menu width: the second direction is evaluated only when the "
        "first fails, so the two points of this binary are not symmetrically sampled"
        if union_short
        else "every pair evaluated both directions"
    )
    return out


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke-scope", help="#821 lane E2E smoke-proof scope census")

    lad = sub.add_parser("ladder-authority", help="#822 lane-guard escalation ladder")
    lad.add_argument(
        "--telemetry", type=Path, nargs="*", default=[],
        help="lane_guard telemetry.jsonl path(s); empty => VACUOUS, never PASS",
    )
    lad.add_argument("--rung", type=float, default=0.1,
                     help="supervisor LG1_LAMBDA_STEP (default 0.1)")
    lad.add_argument("--escalate-at", type=float, default=1.0,
                     help="supervisor LG1_LAMBDA_ESCALATE (default 1.0)")

    br = sub.add_parser("bracket-direction", help="#871 bracket-direction occupancy")
    br.add_argument("--arms-jsonl", type=Path, required=True,
                    help="pw1_arms.jsonl probe receipt")

    args = ap.parse_args(argv)

    if args.cmd == "smoke-scope":
        result = smoke_proof_scope_census()
    elif args.cmd == "ladder-authority":
        gs, params = _load_g_series(list(args.telemetry))
        result = ladder_authority(
            rung=args.rung,
            eta_lambda=params.get("eta_lambda", 0.0),
            lambda_step_cap=params.get("lambda_step_cap", 0.0),
            lambda_max=params.get("lambda_max", 5.0),
            g_series=gs,
            escalate_at=args.escalate_at,
        )
        result["telemetry_files_seen"] = sum(1 for p in args.telemetry if p.exists())
        result["telemetry_files_requested"] = len(args.telemetry)
        result["hyperparameter_drift"] = {
            k: v for k, v in params.items() if k.endswith("_distinct")
        }
    else:
        result = bracket_direction_occupancy(args.arms_jsonl)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
