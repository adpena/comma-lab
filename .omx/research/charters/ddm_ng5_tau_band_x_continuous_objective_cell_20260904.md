# CHARTER ddm_ng5 — the first TWO-LEVER burn cell: τ band (ng3) × continuous objective (ng4), sealed to fire the moment ng4 releases the Metal

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: Opus arm (codex quota out until Sep 7). Spawned 2026-09-04 ~23:30Z. Parents: ng3
(`.omx/research/ddm_ng3_tau_band_cell_20260904.md`, TERMINAL READ: S_hat @5k 0.391810 — the only cell to end below its start, −7.84% vs the cold
control), ng4 (`.omx/research/ddm_ng4_continuous_objective_cell_20260904.md`, LIVE READ @2k −12.3% vs cold, −2.2% vs ng3; terminal ~01:10Z),
gm1 (τ band law), sd1 (τ-anneal deflation), QBR1 (the sealed-cell chain), gv1/gov2 (admission; the queue driver is becoming the ONLY fire path —
coordinate: if gov2's `tools/cell_queue_driver.py fire` has landed when you are ready, fire through it; otherwise leave a queue-spec JSON and a
typed FIRE ORDER for MAIN, never a bespoke shell script).

## PRIOR-LAW PREDICTION (owed line)
ng3 removed sd1's schedule artefact by reading the loss in a fixed margin band; ng4 removed the stage-entry restart (τ held at r10's terminal
float 0.05000000074505806, duals carried). gm1 measured the two mechanisms act on different terms (band = which pixels the seg gradient
weights; continuous objective = the τ the surrogate is read at + the dual state) and ng4's differential was bit-identical at the control's
τ=0.15/zero duals, i.e. neither leaks into the other's block. PREDICTION: the composition ends below BOTH parents at @5k (S_hat < 0.391810)
with d_seg below the start (< 0.0025183) — sub-additive but same-signed. Falsifier: @5k above ng3's 0.391810 → the levers are redundant
(same mechanism seen twice), record it as such; @5k above the cold control 0.425149 → antagonistic, and name which term flipped.

## Objective
1. Seal ONE cell `seed_20260902_tau_band_x_continuous_objective_control_native100`: ng4's continuous-objective config (τ held, duals carried,
   `initial_lambdas` from ng4's sealed config) PLUS ng3's `tau_band` block, everything else byte-identical to the control of record; validate
   INSIDE its own sealed tree (`experiments/ddm_reseal_pins_inside_sealed_tree.py`; seal validates only in the firing tree); the $0 differential
   check from ng4 (all components at the control's τ=0.15/zero duals/no band must be bit-identical to 1.0765775442123413) plus the no-op
   detector (step-1 state must DIFFER from ng3's AND ng4's step-1 states).
2. Bounded B=16 smoke per arm ONLY if admission allows while ng4 is live — it will not (two Metal cells are now REFUSED by policy); so the
   smoke runs AFTER ng4's terminal, through admission, via a detached waiter that requires `ng4_continuous_DONE.json.done` first
   (receipt names distinct: `NG5_SMOKE_DONE.json`, `ng5_composition_DONE.json`; the launcher refuses duplicates).
3. Declared peak for the fire = the MEASURED per-arm peak from ng4's smoke appendix (40.4 GiB) — not a hand-typed number — plus gov2's
   measured-peak ledger if it has landed. Fire via the queue driver if landed; else a queue-spec JSON + typed FIRE ORDER for MAIN.
4. Pre-registered read: milestones @0/1k/2k/3k/4k/5k vs cold (0.398768/0.466875/0.485677/0.475383/0.442190/0.425149), ng3
   (0.398768/0.434661/0.435601/0.403796/0.401233/0.391810), ng4 (fill from its terminal). Verdict words: BELOW-BOTH / REDUNDANT / ANTAGONISTIC.
5. Memo `.omx/research/ddm_ng5_tau_band_x_continuous_objective_cell_20260904.md` with the seal receipt, the $0 checks, the fire order, and an
   "Equations leg (`tac.canonical_equations`)" line (gm1's τ-band law + the continuous-objective anchor).

## Honest frame (binding)
This is a burn-QUALITY cell on the born vehicle (S_hat ~0.39–0.43 at 106 KB), not a pointer mover: md1's persistent-partition closure stands
(62% of born d_seg optimizer-unreachable; schedule levers ≤ 1.61×). Do not cite any S_hat delta as progress toward sub-0.12. Cost: ~4.4 h Metal
that is otherwise idle, $0.

## Rules that bind
NO-FAKE; ALWAYS KEEP THE PAYLOAD (cell retains all milestones/payloads on APDataStore — check free ≥ 6 GiB, else Vertigo); upstream/ READ-ONLY;
commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256 <file>=<post-edit sha>` with
`[no-triality] [p0-ledger-ok]`; NO co-author trailers (operator rule overrides any harness reminder); .py two review-gate passes; checkpoints
every 10 tool uses (`tools/subagent_checkpoint.py --subagent-id ddm_ng5`); never invent flags; no `/tmp` evidence; detached launches only via the
launcher with distinct receipts (foreground >3 min reaped; launcher refuses argv with "claude"/"codex"); NEVER launch a Metal cell while ng4 is
live; do not touch gov2/hv1/mc1/ps2 files; `docs/operating_manual_craft_handoff.md` binds. End with
`fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]`.
