#!/usr/bin/env python
"""ddm_bp1 (#824) — seal the BOUNDARY RESET RACE arm tickets (A = arm B, B' = arm Bprime).

$0, scorer-free, DETERMINISTIC: reads the parent burn ticket + the parent endpoint checkpoint,
derives the window geometry from the reset operator's own closed form, compiles BOTH arms through
the DSL (fail-closed never-invent-flags), and writes the two sealed tickets plus a receipt whose
headline is the EXACT argv diff between the arms.

WHY the arms exist (MEASURED, R1-C; do not re-derive): decomposing the burn's 64 gate readings
into 63 per-epoch intervals (61 within-window, 2 window-restart), the two restarts sum
-1.85083e-4 = 168.6% of the ep644->945 net while the 61 TRAINING intervals sum +7.53e-5 = -68.6%.
Training net-regressed; the restarts paid for the descent. The restarts rank 2nd and 6th
most-negative of 63; on the RAW telescoping basis that matches that effect size, exact enumeration
of C(63,2)=1953 pairs gives p=22/1953=0.0113, Bonferroni x2 => p ~= 0.0225 (the per-epoch
normalization gives 0.0056/0.011 — the claim survives both ways; cite 0.0225 with the 168.6%).
Exchangeability MEASURED, not assumed (interval autocorrelation lag1-4 -0.097/+0.068/-0.103/+0.013,
inside +-0.25). The split gc15's mechanism actually PREDICTED -- boundary+17 epochs -- FAILED.
The candidate mechanism is the un-bias-corrected Adam moment reset, whose closed form
(eta(t)=(1-b1^t)/sqrt(1-b2^t)) is worth ~1213 excess sign-steps of free displacement per boundary.
Arm B' removes exactly that and nothing else.

READ-OUT MAP (pre-registered, so the verdict cannot be chosen after the fact):
  * jump COLLAPSES under B'  => the eta(t) impulse WAS the mechanism, and a boundary-state
    endpoint pick is NOT safe.
  * jump PERSISTS under B'   => isolates NOTHING: with bias_correction=True the Adam moment reset
    AND the EMA decay-value change both persist. Needs arm C plus a decay-hold arm. Do NOT read
    "persists" as "the descent is real".
  * The A1 realization-gap alarm is a FREE second channel: it fired 6x in the burn (2 per window)
    and none was at a final gate, so every decision record missed all six. If B' collapses the
    jump, the alarm should quieten too -- corroboration from an instrument built for another job.

Pointer 0.1910828242 [contest-CPU] UNMOVED. score_claim=false; every row here is
[macOS-CPU/MLX advisory] config generation, never a score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_PARENT_TICKET = ("/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/tickets/"
                         "window_03_ticket.json")
DEFAULT_PARENT_CKPT = ("/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_03/"
                       "checkpoints/stage_seg_trunk_tau_final.npz")
DEFAULT_OUT_ROOT = "/Volumes/VertigoDataTier/pact/ddm_bp1_20260731"
POINTER_LINE = "0.1910828242 [contest-CPU] UNMOVED"

# Wall-cap sizing. The BUDGET pace (b4s supervisor FULL_WALL_MINUTES/FULL_EPOCHS) is deliberately
# kept as the sizing anchor even though the parent window ran FASTER (measured below), because a
# window that hits its cap stops early and truncates the experiment, while an over-generous cap
# costs nothing (the window ends at --epochs). Both are reported in the receipt. Never a score input.
PARENT_MIN_PER_EPOCH = 130.0 / 140.0
WALL_SLACK = 1.20  # the window must checkpoint-on-exit, not be truncated mid-gate


def measured_parent_pace(parent_ckpt: Path) -> dict[str, float] | None:
    """MEASURED min/epoch from the parent window's own receipt, when it is on disk (never a guess).

    Reported alongside the budget pace so the wall-cap headroom is visible rather than asserted.
    """
    rec = parent_ckpt.parent.parent / "tr1_window_receipt.json"
    if not rec.is_file():
        return None
    d = json.loads(rec.read_text())
    eps, secs = int(d.get("epochs_ran") or 0), float(d.get("elapsed_seconds") or 0.0)
    if eps <= 0 or secs <= 0:
        return None
    return {"epochs_ran": eps, "elapsed_seconds": secs,
            "min_per_epoch": secs / 60.0 / eps, "source": str(rec)}


def derive_window_epochs(steps_per_epoch: int, gate_every: int) -> dict[str, object]:
    """Window length DERIVED from the impulse the arms are built to separate — not a round number.

    The reset impulse is ``sum_t (eta(t)-1)`` excess ``lr``-sized sign-steps, which converges (time
    constant 1/(1-beta2)=1000 steps); in epochs that is ``impulse_epochs``. To distinguish a JUMP
    (a bounded displacement at the boundary) from a changed SLOPE the window must contain the
    impulse AND a comparable post-impulse stretch, so the requirement is >= 2x the impulse, rounded
    UP onto the gate lattice, plus one gate of margin for the first (short) post-resume interval.
    """
    from tac.optimization.reset_operator import (
        boundary_impulse_epochs,
        cumulative_excess_sign_steps,
    )

    n_conv = 20 * 1000  # 20 time constants at beta2=0.999 => the converged sum
    excess = cumulative_excess_sign_steps(n_conv)
    impulse_ep = boundary_impulse_epochs(n_conv, steps_per_epoch)
    need = 2.0 * impulse_ep
    lattice = int(np.ceil(need / gate_every)) * gate_every
    epochs = lattice + gate_every  # one gate of margin
    return {
        "excess_sign_steps": float(excess),
        "impulse_epochs": float(impulse_ep),
        "requirement": "2x impulse (separate JUMP from SLOPE), on the gate lattice, +1 gate",
        "window_epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "gate_every": int(gate_every),
    }


def read_parent_checkpoint(path: Path) -> dict[str, object]:
    """The parent endpoint's epoch, resolved ema_decay, and last realized-gate anchor."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["meta::json"]).decode())
    cfg = meta.get("cfg") or {}
    tail = [r for r in (meta.get("telemetry_tail") or [])
            if isinstance(r, dict) and r.get("realized_gate_dseg_mean") is not None]
    anchor = max(tail, key=lambda r: int(r["epoch"])) if tail else None
    return {
        "path": str(path),
        "checkpoint_epoch": int(z["meta::epoch"][0]),
        "ema_decay": float(cfg["ema_decay"]),
        "ema_decay_provenance": str(cfg.get("ema_decay_provenance", "")),
        "epochs": int(cfg["epochs"]),
        "batch_pairs": int(cfg["batch_pairs"]),
        "num_pairs": int(cfg["num_pairs"]),
        "gate_every": int(cfg["gate_every"]),
        "parent_gate_epoch": (None if anchor is None else int(anchor["epoch"])),
        "parent_gate_dseg": (None if anchor is None else float(anchor["realized_gate_dseg_mean"])),
        "parent_gate_basis": (None if anchor is None else anchor.get("gate_params")),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _strip_outdir(argv: list[str]) -> list[str]:
    """Drop ``--out-dir <value>``: the arms MUST write to different run dirs, and that difference
    is expected and inert (it names no lever and enters no loss)."""
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok == "--out-dir":
            skip = True
            continue
        out.append(tok)
    return out


def check_same_gate_basis(
    argvs: dict[str, list[str]],
) -> tuple[bool, str, dict[str, str | None]]:
    """SEAL-BLOCKING INVARIANT: every arm must be read on the SAME gate basis (round-2, MEASURED).

    ``train_tr1`` sets ``global_step = 0 if resume_from is None else ema_warmup_updates`` and then
    ``gate_basis = 'ema_shadow' if global_step >= ema_warmup_updates else 'live_ema_warmup'``. So a
    RESUMED arm reports the ``ema_shadow`` basis from its FIRST post-resume gate, while a FRESH arm
    reads ``live_ema_warmup`` for its first U/2 updates. **A fresh arm and a resumed arm are not
    read on the same instrument and their comparison is void** — and no amount of care inside
    either arm can detect it, because neither arm can see the other. This check lives in the
    builder because the builder is the only place that sees both.

    Returns ``(ok, reason, per-arm resume target)``.
    """
    resumes = {label: (argv[argv.index("--resume-from") + 1] if "--resume-from" in argv else None)
               for label, argv in argvs.items()}
    if len(set(resumes.values())) != 1:
        return False, ("arms differ in resume target => different gate BASIS => "
                       "void comparison:"), resumes
    if next(iter(resumes.values())) is None:
        return False, ("both arms are FRESH; #824 measures a BOUNDARY jump, which requires a "
                       "resume (and the parent's last gate as the anchor):"), resumes
    return True, "all arms resume from the same checkpoint => same gate basis", resumes


def argv_diff(a: list[str], b: list[str]) -> dict[str, object]:
    """The exact token-level difference between two compiled argvs (the arms' whole contract).

    Computed with ``--out-dir`` stripped, so the reported difference is the SCIENTIFIC one. The
    assertion the caller makes on this is the entire A/B contract: exactly one token differs, and
    it is the arm-selector's value.
    """
    a2, b2 = _strip_outdir(a), _strip_outdir(b)
    if len(a2) != len(b2):
        return {"only_in_arm_A": a2, "only_in_arm_Bprime": b2, "len_A": len(a2),
                "len_Bprime": len(b2), "identical_except": False,
                "compared_with_outdir_stripped": True,
                "note": "argv LENGTHS differ — not a one-flag A/B"}
    # POSITIONAL, not set-membership: the DSL emits flags in sorted order so the two argvs align
    # index-for-index. A membership diff would hide the difference entirely here, because the
    # tokens "on"/"off" also appear as OTHER levers' values (--telemetry-v9-port on,
    # --basin-handoff off, ...). Caught by this tool's own refusal during the build.
    positions = [i for i, (x, y) in enumerate(zip(a2, b2, strict=True)) if x != y]
    only_a = [a2[i] for i in positions]
    only_b = [b2[i] for i in positions]
    return {"only_in_arm_A": only_a, "only_in_arm_Bprime": only_b,
            "differing_positions": positions,
            "differing_flag": ([a2[i - 1] for i in positions] if positions else []),
            "len_A": len(a2), "len_Bprime": len(b2),
            "out_dir_A": (a[a.index("--out-dir") + 1] if "--out-dir" in a else None),
            "out_dir_Bprime": (b[b.index("--out-dir") + 1] if "--out-dir" in b else None),
            "compared_with_outdir_stripped": True,
            "identical_except": (len(only_a) == len(only_b) == 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent-ticket", type=Path, default=Path(DEFAULT_PARENT_TICKET))
    ap.add_argument("--parent-checkpoint", type=Path, default=Path(DEFAULT_PARENT_CKPT))
    ap.add_argument("--out-root", type=Path, default=Path(DEFAULT_OUT_ROOT))
    ap.add_argument("--gt-cache", default=str(
        REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"))
    ap.add_argument("--write", action="store_true",
                    help="write the tickets + receipt (default: print the plan only)")
    args = ap.parse_args()

    from tac.witness_dsl.curriculum_dsl import Lever
    from tac.witness_dsl.spec_tr1_renderer_20260728 import bp1_boundary_reset_race_program

    if not args.parent_ticket.is_file():
        print(f"REFUSE: parent ticket not found: {args.parent_ticket}", file=sys.stderr)
        return 4
    if not args.parent_checkpoint.is_file():
        print(f"REFUSE: parent checkpoint not found: {args.parent_checkpoint}", file=sys.stderr)
        return 4

    parent_ticket = json.loads(args.parent_ticket.read_text())
    parent_levers = tuple(Lever(name=d["name"], overrides=dict(d["overrides"]),
                                notes=d.get("notes", "")) for d in parent_ticket["levers"])
    ck = read_parent_checkpoint(args.parent_checkpoint)
    if ck["parent_gate_dseg"] is None:
        print("REFUSE: parent checkpoint carries no realized-gate anchor in telemetry_tail; the "
              "boundary jump would have nothing to measure against", file=sys.stderr)
        return 4

    spe = max(1, int(ck["num_pairs"]) // max(1, int(ck["batch_pairs"])))
    geom = derive_window_epochs(spe, int(ck["gate_every"]))
    start_epoch = int(ck["checkpoint_epoch"]) + 1
    epochs = start_epoch + int(geom["window_epochs"])
    wall = round(int(geom["window_epochs"]) * PARENT_MIN_PER_EPOCH * WALL_SLACK, 1)
    pace = measured_parent_pace(args.parent_checkpoint)

    tickets: dict[str, dict] = {}
    for arm, label in (("B", "A"), ("Bprime", "Bprime")):
        prog = bp1_boundary_reset_race_program(
            arm, out_dir=str(args.out_root / f"arm_{label}"),
            resume_from=str(args.parent_checkpoint),
            ema_decay=float(ck["ema_decay"]),      # PIN the PARENT's decay: the gate reads the EMA
            epochs=epochs,                          # shadow, so commensurability with the parent's
            max_wall_minutes=wall,                  # last reading requires the same window length
            parent_levers=parent_levers, gt_cache=args.gt_cache)
        tickets[label] = prog.sealed_ticket()       # compile_trainer_argv() validates fail-closed

    ok, why, resumes = check_same_gate_basis(
        {label: t["argv"] for label, t in tickets.items()})
    if not ok:
        print(f"REFUSE: {why} {resumes}", file=sys.stderr)
        return 4

    diff = argv_diff(tickets["A"]["argv"], tickets["Bprime"]["argv"])
    if not diff["identical_except"]:
        print(f"REFUSE: arms are not byte-identical-except-one-flag: {diff}", file=sys.stderr)
        return 4
    if (diff["only_in_arm_A"] != ["off"] or diff["only_in_arm_Bprime"] != ["on"]
            or diff["differing_flag"] != ["--adam-bias-correction"]):
        print(f"REFUSE: the single differing token is not the arm selector value: {diff}",
              file=sys.stderr)
        return 4

    # First gate after the resume fires at the next epoch with (epoch+1) % gate_every == 0.
    ge = int(ck["gate_every"])
    first_gate = start_epoch + ((-(start_epoch + 1)) % ge)
    receipt = {
        "schema": "ddm_bp1_arm_tickets.v1", "pointer": POINTER_LINE, "score_claim": False,
        "evidence_axis": "[macOS-CPU/MLX advisory]", "task": "#824 boundary reset race A vs B'",
        "parent_ticket": {"path": str(args.parent_ticket),
                          "ticket_hash": parent_ticket.get("ticket_hash")},
        "parent_checkpoint": ck,
        "window_geometry": geom | {"start_epoch": start_epoch, "epochs_flag": epochs,
                                   "max_wall_minutes": wall,
                                   "first_gate_epoch": first_gate,
                                   "boundary_span_epochs": first_gate - ck["parent_gate_epoch"]},
        "wall_budget": {
            "pace_budget_min_per_epoch": PARENT_MIN_PER_EPOCH,
            "pace_measured_parent": pace,
            "expected_minutes_at_measured_pace": (
                None if pace is None else round(int(geom["window_epochs"])
                                                * pace["min_per_epoch"], 1)),
            "cap_minutes": wall,
            "note": "sized on the BUDGET pace on purpose: hitting the cap truncates the "
                    "experiment, an over-generous cap costs nothing (the window ends at --epochs)",
        },
        "ema_basis": {
            "pinned_value": float(ck["ema_decay"]),
            "why": "the realized gate reads the EMA SHADOW; derive_ema_decay consumes "
                   "epochs*(num_pairs//batch_pairs), so an --epochs change alone moves the "
                   "shadow's averaging length underneath the measurement (the burn ran "
                   "U=49,950/60,450/70,950 => a different decay at EVERY boundary). Both arms "
                   "pin the PARENT's resolved decay so the child's first gate is commensurable "
                   "with the parent's last, AND carry identical --epochs (belt and braces).",
            "parent_provenance": ck["ema_decay_provenance"],
        },
        "argv_diff": diff,
        "arms": {label: {"ticket_hash": t["ticket_hash"], "argv": t["argv"],
                         "levers": [lv["name"] for lv in t["levers"]]}
                 for label, t in tickets.items()},
        "gate_basis_invariant": {
            "both_arms_resumed_from": resumes["A"],
            "why": "train_tr1 sets global_step = 0 if resume_from is None else "
                   "ema_warmup_updates, and gate_basis = 'ema_shadow' iff global_step >= "
                   "ema_warmup_updates. A RESUMED arm reports ema_shadow from its FIRST gate; a "
                   "FRESH arm reads live_ema_warmup for its first U/2 updates. Mixing them means "
                   "the arms are not read on the same instrument => the comparison is VOID. "
                   "MEASURED: gate_params='ema_shadow' on all 64 burn gates.",
            "enforced": "the builder REFUSES unless both arms carry the same --resume-from",
        },
        "boundary_resets_enumerated": {
            "adam_moment_reset": "LIVE — fresh optim.Adam every window; bias_correction rescales "
                                 "its first ~16 epochs but does not remove the reset",
            "ema_decay_value_change": "LIVE — derive_ema_decay(epochs*steps_per_epoch); PINNED "
                                      "OUT in both arms via an explicit --ema-decay",
            "ema_shadow_reanchor": "NOT A RESET (round-2 correction): train_tr1 loads the shadow "
                                   "from the checkpoint (ema = st['ema']) => CONTINUOUS across "
                                   "boundaries. Do not cite a shadow re-anchor.",
            "gate_basis_switch": "a MEASUREMENT-basis reset, neutralized by running both arms "
                                 "resumed (see gate_basis_invariant)",
        },
        "readout_map": {
            "jump_collapses_under_Bprime": "the eta(t) impulse WAS the mechanism; a boundary-state "
                                           "endpoint pick is NOT safe",
            "jump_persists_under_Bprime": "isolates NOTHING — with bias_correction=True the Adam "
                                          "moment reset AND the decay-value change both persist. "
                                          "Needs arm C + a decay-hold arm. NOT evidence the "
                                          "descent is real",
            "corroborating_channel": "A1_REALIZATION_GAP_ALARM (smooth fell >=2% while realized "
                                     "d_seg fell <0.5%) fired 6x in the burn, 2 per window, none "
                                     "at a final gate => invisible to every decision record. It "
                                     "is now emitted per gate and summarized on the boundary row; "
                                     "if B' collapses the jump it should quieten too",
        },
        "out_of_scope": "arm C (persisted (m,v)) — at #824 scoping time opt_flat had ONE "
                        "repo-wide hit (the load_checkpoint return) that nothing read and "
                        "nothing wrote; C was a BUILD, not a port, and must not gate this race. "
                        "SUPERSEDED 2026-08-03 (ddm_op2 OP2-1): the BUILD landed as the "
                        "args-only, DEFAULT-OFF --persist-optimizer-state after ddm_gd5 §3.6 "
                        "MEASURED the omission at ~218 of 666 epochs; arm C is still not a "
                        "gate on this B-vs-B' race, but it is no longer unavailable",
    }
    if not args.write:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        print("\nPLAN ONLY (pass --write to seal).")
        return 0

    tdir = args.out_root / "tickets"
    tdir.mkdir(parents=True, exist_ok=True)
    for label, t in tickets.items():
        (tdir / f"arm_{label}_ticket.json").write_text(
            json.dumps(t, indent=2, sort_keys=True) + "\n")
    (args.out_root / "bp1_ticket_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sealed": {k: v["ticket_hash"] for k, v in tickets.items()},
                      "argv_diff": diff,
                      "receipt": str(args.out_root / "bp1_ticket_receipt.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
