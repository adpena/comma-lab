"""ddm_tc1 — seal the ONE next-burn TR1 ticket: the QA83 semantic-primary head.

WHY THIS BURN (and no other). The tc1 lifecycle composition measured TR1's binding
term from primary receipts:

  * TR1 BEST byte-closed n600 (ddm_gr1 cell_drop50, archive a6398e441f…, 359,221 B):
    d_seg 0.004310 -> seg term 0.43104, rate term 0.23919, seg+rate 0.67023.
  * hv1 frontier [contest-CUDA T4 n600]: S 0.15959729295498598 @ 182,759 B
    (seg 0.029611 + pose 0.0082945765 + rate 0.1216917).
  * => TR1's SEG TERM ALONE (0.43104) is 2.85x the ENTIRE seg+rate budget a
    sub-frontier score allows (0.15130). A FREE archive does not rescue TR1.

So the binding term is TR1's realized d_seg floor, NOT its rate. Every rate lever on
this trainer is therefore DOMINATED, and the only lever aimed at the binding term that
is already BUILT is the QA83 factorized output head: hg1 §6 measured that hv1's token
field reproduces the GT SegNet argmax to 8.48e-06 (it codes the PARTITION) while TR1
codes an RGB-ish LATENT that a 3,284 B renderer must expand. `--renderer-head-mode
class_field` makes TR1's output space semantic (k=1 class scalar + fixed comma10k luma
lift, rule-118-FREE decoder code) instead of RGB. That is Amendment-1 item 2
("semantic-primary never got a full-vehicle race") on the live vehicle, at zero build
cost.

THE CONTROL IS FREE. ddm_hg1 arm_a_control_ce completed ep399 rc=0 with EXACTLY these
levers minus the head. This ticket is a ONE-AXIS delta against a control already in
hand, matched on seed, geometry, schedule, window and init.

NAMED PRE-REGISTERED CONFOUND (found by reading source, not assumed).
`--token-init-mode solve_project` (line ~4497 of the trainer) projects GT frame_1 *RGB*
into the token lattice and is HEAD-MODE-BLIND: it writes an RGB-derived init whatever
the head mode is. The treatment therefore starts from an init designed for the rgb
head. The confound is ASYMMETRIC and can only UNDERSTATE class_field:
  * class_field WINS  -> admissible (it won despite a mismatched init).
  * class_field LOSES -> INADMISSIBLE as a family verdict; routes to the head-matched
    solve-init build order (project the GT argmax through the luma lift) and a re-test.
This bound is recorded in the ticket so no reader can quote a loss as a family kill.

NOT LAUNCHED. launch_now=false. MAIN fires at the ddm_hg1 arm_b_hinge endpoint
boundary (arm_b fired 18:52Z; the launcher's G4 gate admits ONE n600 job at a time).

axis: [macOS-CPU advisory] — score_claim=False, promotion_eligible=False. Nothing here
is a contest score; the pointer is UNMOVED.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tac.witness_dsl.curriculum_dsl import Lever
from tac.witness_dsl.spec_tr1_renderer_20260728 import (
    TR1RendererProgramV1,
    lever_a1_gate,
    lever_basin_handoff,
    lever_lotto,
    lever_renderer_capacity,
    lever_renderer_head,
    lever_desc_level_roundtrip,
    lever_token_grid,
    lever_token_init,
    lever_token_temporal,
    lever_variant,
    lever_window,
)

REPO = Path(__file__).resolve().parents[1]
CONTROL_TICKET = REPO / ".omx/research/configs/ddm_hg1_tr1_ticket_arm_a_control_ce_20260816.json"
OUT_TICKET = REPO / ".omx/research/configs/ddm_tc1_tr1_ticket_semantic_head_class_field_20260817.json"
OUT_DIR = "/Volumes/APDataStore/pact/ddm_tc1_semantic_head_20260817/arm_c_class_field"

# --- MEASURED anchors this seal is priced against (every one carries its receipt) ---
ANCHORS = {
    "hv1_frontier_S": 0.15959729295498598,
    "hv1_frontier_bytes": 182759,
    "hv1_seg_term": 0.029611,
    "hv1_pose_term": 0.0082945765,
    "hv1_rate_term": 0.1216917,
    "hv1_receipt": ".omx/state/canonical_frontier_pointer.json (effective_frontier; "
                   "archive sha 80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e)",
    "tr1_best_byteclosed_bytes": 359221,
    "tr1_best_byteclosed_dseg": 0.004310,
    "tr1_best_byteclosed_seg_plus_rate": 0.67023,
    "tr1_best_receipt": ".omx/research/ddm_gr1_granularity_rerace_20260730.md "
                        "(n600 confirm table; archive a6398e441f4bc818…, 359,221 B)",
    "tr1_control_endpoint_dseg": 0.016963393599898728,
    "tr1_control_endpoint_bytes": 523315,
    "tr1_control_receipt": "/Volumes/APDataStore/pact/ddm_hg1_ring0_hinge_ab_20260816/"
                           "arm_a_control_ce/telemetry.jsonl ep399 a1_gate (n=36, "
                           "[macOS-CPU frozen-scorer advisory])",
}


def build_program(out_dir: str = OUT_DIR) -> TR1RendererProgramV1:
    """arm_a's EXACT lever set + the single QA83 head lever."""
    levers: list[Lever] = [
        lever_variant("lotto"),
        lever_token_grid(16, 4),
        lever_renderer_capacity(24),
        lever_desc_level_roundtrip(16, "round"),
        lever_token_temporal("shared_base"),
        lever_a1_gate(10),
        lever_window(400, 480.0, batch_pairs=8, lr=2e-3),
        lever_lotto(118, 0.5),
        lever_token_init("solve_project"),
        lever_basin_handoff("on"),
        # seg trunk weights + form, lifted out of the bundle exactly as hg1 split them
        # so the ONLY delta vs the control is the head lever.
        Lever(name="hg1_seg_trunk_weights",
              overrides={"--class-weight-lane": "1.0", "--w-seg": "100.0"},
              notes="MATCHED to ddm_hg1 arm_a_control_ce: the seg trunk weights."),
        Lever(name="hg1_seg_form_ce_control",
              overrides={"--margin-target": "1.0", "--seg-form-start": "ce"},
              notes="MATCHED to ddm_hg1 arm_a_control_ce: the CE seg form."),
        # ---- THE ONLY DELTA ----
        lever_renderer_head("class_field"),
    ]
    return TR1RendererProgramV1(levers=tuple(levers), num_pairs=600, out_dir=out_dir,
                                seed=0, full_confirm=True)


def control_delta(ticket_argv: list[str]) -> dict[str, list[str]]:
    """Prove the one-axis property against the COMPLETED control's sealed argv."""
    ctrl = json.loads(CONTROL_TICKET.read_text())["argv"]

    def as_map(argv: list[str]) -> dict[str, str]:
        out, i = {}, 1
        while i < len(argv):
            a = argv[i]
            if a.startswith("--"):
                nxt = argv[i + 1] if i + 1 < len(argv) else None
                if nxt is not None and not nxt.startswith("--"):
                    out[a] = nxt
                    i += 2
                    continue
                out[a] = "<flag>"
            i += 1
        return out

    c, t = as_map(ctrl), as_map(ticket_argv)
    diff = sorted(set(c) | set(t))
    return {k: [c.get(k, "<absent>"), t.get(k, "<absent>")]
            for k in diff if c.get(k) != t.get(k)}


def main() -> int:
    prog = build_program()
    payload = prog.sealed_ticket()          # validates: never-invent-flags, fail-closed
    delta = control_delta(payload["argv"])

    payload.update({
        "tc1_arm": "arm_c_class_field",
        "tc1_out_dir": OUT_DIR,
        "tc1_launch_now": False,
        "tc1_fire_priority": "CONTINGENT — NOT RECOMMENDED FIRST. ddm_tc1 §6 F4 dominates: TR1 "
                             "is a RETIRED vehicle (the live trainer is "
                             "src/tac/pr130_lift/train_semantic_quantized_resumable.py = hv1), "
                             "and that successor ALREADY IS the semantic-primary architecture "
                             "this burn would test. Fire ONLY if the Metal slot is otherwise "
                             "idle and no live-vehicle burn (wd3 / b2e / hb1) is queued. Per "
                             "[[m18]], retired-vehicle levers are LESSON-ONLY.",
        "tc1_fires_when": "ddm_hg1 arm_b_hinge reaches its endpoint (fired 18:52Z, ~4.5-8 h); "
                          "the launcher G4 gate admits ONE n600 job at a time. MAIN fires; "
                          "this arm never launches.",
        "tc1_control": {
            "ticket": str(CONTROL_TICKET.relative_to(REPO)),
            "status": "COMPLETED ep399 rc=0",
            "run_dir": "/Volumes/APDataStore/pact/ddm_hg1_ring0_hinge_ab_20260816/arm_a_control_ce",
            "endpoint_dseg": ANCHORS["tr1_control_endpoint_dseg"],
            "endpoint_total_counted_bytes": ANCHORS["tr1_control_endpoint_bytes"],
            "note": "matched control ALREADY IN HAND — this burn costs one arm, not two.",
        },
        "tc1_argv_delta_vs_control": delta,
        "tc1_measured_anchors": ANCHORS,
        "tc1_falsifiers": {
            "primary": "class_field endpoint realized d_seg >= the control's d_seg at MATCHED "
                       "total_counted_bytes => the QA83 factorized-output family closes at "
                       "INSTANCE scope on TR1 and the semantic-primary route moves to a "
                       "representation change (PR130-class partition tokens), not a head swap. "
                       "This falsifier is the DSL lever's own pre-registered one.",
            "confound_bound": "A LOSS is INADMISSIBLE as a family verdict because "
                              "--token-init-mode solve_project is head-mode-blind (it projects "
                              "GT frame_1 RGB into the lattice regardless of head mode), so the "
                              "treatment starts from an rgb-matched init. A loss routes to the "
                              "head-matched solve-init build order + re-test. A WIN is "
                              "admissible (it won DESPITE the mismatched init).",
            "inert": "renderer_bytes identical to the control's 3,284 B at every gate => the "
                     "head lever did not change the head DOF => wiring is inert, run confounded.",
            "scope_ceiling": "Even a FULL win does not reach the frontier by itself: TR1's "
                             "BEST-EVER seg term (0.389011, from d_seg 0.00389011, n600 real "
                             "evaluate.py) is 2.571x the whole seg+rate budget (0.151303), so a "
                             "FREE archive still scores 2.49x the frontier. This burn changes "
                             "the ROUTING, not the pointer. Any report that quotes it as a "
                             "frontier move is a NO-FAKE violation.",
            "ema_warmup_contamination": "MEASURED DEFECT on this vehicle: --ema-decay derives to "
                                        "0.997 per-update at 1 update/epoch => warmup 667 "
                                        "updates ~ ep1318, so a 400-ep run sits ENTIRELY inside "
                                        "declared warmup (shadow ~64% seed at ep800). Both arms "
                                        "share it, so the A/B DIFFERENCE survives; neither arm's "
                                        "ABSOLUTE d_seg may be quoted against any other lineage.",
        },
        "tc1_owed_before_any_claim": [
            "byte-closed archive + [contest-CUDA] n600 exact eval; advisory MLX/CPU rows are "
            "never a score.",
            "a MEASURED pose leg — raw TR1 ships NO pose section and its objective is pose-blind "
            "(jd1_w_pose=0); the control's d_pose 144.6 is an advisory trend channel, and rn1 "
            "measured the advisory instrument 18.2x optimistic on pose.",
        ],
        "tc1_resumability_p0": {
            "resume": "--resume-from <out_dir>/checkpoints/<stage>.npz (launcher passes it "
                      "through); per-stage EMA-shadow checkpoints written by the trainer.",
            "checkpoint_head_mode_lock": "trainer line ~2838: a checkpoint may only resume into "
                                         "a model built with the SAME renderer_head_mode, so "
                                         "this arm is FRESH-START by construction (matched to "
                                         "the control, which was also fresh-start).",
            "window": "--epochs 400 --max-wall-minutes 480.0 --basin-handoff on",
            "done_receipt_watcher": "REQUIRED at fire time (the hg1 arm_a silent-death lesson is "
                                    "one day old: it died at ep279 with watchers:[] and went "
                                    "unnoticed ~17 h). Arm the done-receipt + a liveness watcher "
                                    "in the same command that launches.",
        },
        "score_claim": False,
        "promotion_eligible": False,
        "axis": "[macOS-CPU advisory]",
        "sealed_by": "ddm_tc1",
        "sealed_utc": "2026-08-17",
    })

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["sealed_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    OUT_TICKET.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"WROTE {OUT_TICKET.relative_to(REPO)}")
    print(f"  ticket_hash   {payload['ticket_hash']}")
    print(f"  sealed_sha256 {payload['sealed_sha256']}")
    print(f"  levers        {len(payload['levers'])}")
    print(f"  launch_now    {payload['tc1_launch_now']}")
    print("\nARGV DELTA vs the COMPLETED control (must be head + out-dir ONLY):")
    for k, (a, b) in delta.items():
        print(f"  {k}: control={a!r} -> treatment={b!r}")
    only_expected = set(delta) <= {"--renderer-head-mode", "--out-dir"}
    print(f"\nONE-AXIS CHECK: {'PASS' if only_expected else 'FAIL'} "
          f"(delta keys = {sorted(delta)})")
    return 0 if only_expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
