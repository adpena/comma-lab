# ddm_r1c — LADDER RUNG 1: the continuation-to-birth-plateau windowed warm tail (2026-07-31, task #803)

**Model: claude-opus (Fable-class main routing).** The gc12 wall-branch birth-completion ladder's
RUNG 1 (`.omx/research/ddm_gc12_wall_branch_convocation_20260731.md`, b4d317538d, §3/§5): seal, FIRE,
and SUPERVISE the windowed warm tail from control_tail ep500 — the burn-4 parent-improver. I own the
scorer slot for its lifetime.

**Pointer honesty FIRST: 0.1910828242 [contest-CPU] UNMOVED.** Every number below is
**[macOS-CPU/MLX advisory]**, `score_claim=false`, `research_only`. This unit is MEANS: it produces a
better PARENT (a lower-S continuation endpoint) + measured rung-1 quantities for burn-4; no exact row
moves here. No AI attribution; commits are the operator's alone. `[no-triality]` (build + measure
apparatus + this graph leg) `[p0-ledger-ok]`.

## STORES-CONSULTED (recall-first, path+sha)
- gc12 `.omx/research/ddm_gc12_wall_branch_convocation_20260731.md` (b4d317538d) — §3 ladder (rung-1
  windows), §5 seal demand (parent ckpt + levers-NONE-new + argv-diff = {epochs,outdir,resume,wall} +
  QA80 staleness gate + F1-F4/A1 guards + endpoint obligations), op-r 2.
- pa1r `.omx/research/ddm_pa1r_pool_a_race_20260730.md` (fdb48e2c26) — THE PARENT: control_tail ep500,
  n600 d_seg 0.0049411 / 265,528 tok B / **S_add 0.67325** (seg 0.49411 + rate 0.17913); Pool-A OUT by
  race verdict; control tail was itself a windowed warm tail (§0). Endpoint harness
  `experiments/ddm_pa1r_endpoint_verdict.py` (validated to B within +1.9e-7).
- qa92 `.omx/research/ddm_qa92_carrier_discriminator_20260731.md` (rung 0) — **rung 2 SKIPS to burn-4**
  (P·O 0.017 < 0.05 Contrarian bound); at control_tail ep500 the still-erased super-nucleus Lane pool is
  **P = 0.042 S** (refines QA91's 0.134 upper bound — the continuation already recovered ~0.09 S free);
  paint-on-texture net-negative even at the oracle (+0.30 S collateral). Sharpens rung-1's role: the
  live descent is the value; rung 2 is not fired.
- lp2 `.omx/research/ddm_lp2_ladder_prep_20260731.md` (34d2354ac4) — P2 birth-completion key producer
  `src/tac/optimization/ddm_lp2_birth_completion.py` + CLI `tools/run_ddm_lp2_birth_completion_key.py`
  (fired ⇔ Lane betti0 slope ≤ ε ∧ above-nucleus erasure persists; ε DERIVED; α=0.15866 stated;
  W=5 / epochs_per_gate anchors). QA91 inventory `/Volumes/VertigoDataTier/pact/ddm_fp1_20260731/
  qa91_erased_lane.json` (betti0_gt_lane 985 / super_nucleus_area_frac 0.9767).
- DSL `src/tac/witness_dsl/spec_tr1_renderer_20260728.py` — `TR1RendererProgramV1` + `lever_window`;
  governed launcher `tools/launch_tr1_run.py` (G0 venv-custody / G1 seal-freshness / G2 import-custody /
  G3 memory-preflight 12.8 GiB×2 / G4 scorer-slot / G5 detached-receipted).
- QA80 field custody `/Volumes/VertigoDataTier/pact/ddm_zb1_qa80_field_20260730/` (600 burn frame1
  camera + field_pass_manifest; the exact flip-distance/band-lemma field).
- trainer `experiments/train_tr1_partition_renderer_mlx.py` (sealed; NOT edited): resume
  `start_epoch=ckpt.meta::epoch+1`, loop `range(start_epoch, --epochs)` (epochs = EXCLUSIVE upper
  bound), EMA-shadow warm on resume (global_step←ema_warmup_updates), `derive_ema_decay(total_updates)`
  where `total_updates=epochs·(num_pairs//batch_pairs)`, per-gate intra-ckpt + stage-final ckpt (P0),
  `_live_margin_weight` (LIVE realized-SegNet margins, per-step stop-grad).

## §1 THE SEAL (gc12 §5, mechanical — all four constraints honored)
- **Parent:** control_tail ep500 ckpt `.../ddm_pa1r_20260730/control_tail/checkpoints/stage_seg_trunk_tau_final.npz`
  **sha256 `a2dc86b8a8982456f1a178ad107c10f3390cf09c5643c0edaa13c19b782fbf48`** (verified on disk;
  `meta::epoch=500`).
- **Levers: NONE new.** The 16 control_tail levers copied VERBATIM from its sealed ticket
  (`.../tickets/control_tail_ticket.json`, ticket_hash 4a67d694…) via the DSL; only the `lever_window`
  overrides swapped. No paint/injection (QA92). w_rate 0.05 INHERITED (0.0768 is the from-scratch BR-A
  derivation — a mid-lineage change would be an un-raced lever; refused).
- **argv-diff vs the parent sealed argv = EXACTLY {--epochs, --max-wall-minutes, --out-dir,
  --resume-from}** (the gc12 §5 sanctioned four; printed + committed per window). Nothing else differs.
- **ema_decay note (warm-start-resume law #517/#518/#270):** extending `--epochs` re-derives the DERIVED
  `ema_decay` from the new `total_updates` (parent 0.9998933 @ U=37500 → window-1 ~0.9999168 @ U=48075).
  This is a principled geometry adaptation of the DERIVED law (LawRef ema_decay_run_geometry_v1), NOT a
  new argv flag (no `--ema-decay` emitted; the diff stays 4 items). The trainer emitted
  `resume_form_reanchor` (events adapted to the resume epoch/geometry).
- **QA80 staleness gate (binding):** the sealed config **does NOT consume the QA80 stored flip-distance
  field** — `token_quant_coupling_field=null`, `token_quant_margin_coupling=off`,
  `delta_sparsity_weight=0.0`, `renderer_head_mode=rgb` (line 715 guards the field load on
  `token_quant_coupling_field`). The "photometric/margin term" (`margin_weighted_loss=on`, temp 1.0) is
  the LIVE `_live_margin_weight` (realized-SegNet margins recomputed per-step, stop-grad) —
  **fresh-by-construction at every epoch, never stale**. Verdict: **field NOT CONSUMED → no re-derive
  needed**; custody recorded for reference (zb1).
- **Guards ON (parent config):** F1-F3 + F4 pool-conflict + A1 realization-gap gate + topology/erasure
  telemetry (trainer :858), #302/#686 event-driven schedule, per-gate + stage-final EMA ckpts (P0),
  governed launcher + DSL-hashed ticket + memory preflight at the real config (all PASS).

## §2 THE WINDOW LOOP (fired; supervised in-tool)
Windows ~140 ep (start_epoch 501 → `--epochs` = 501 + 140). Per window: governed detached resume
(`launch_tr1_run.py`, `start_new_session`); ONE full-n600 scorer job at a time (verdict chunk ≤120).
Between windows: `run_ddm_lp2_birth_completion_key.py` on the window's telemetry (**epochs_per_gate=5** —
control_tail gate_every=5, NOT the bc1 default 10) → `fired=False ∧ windows<3 ∧ elapsed<8h ⇒ next
window`; `fired=True ∨ caps ⇒ STOP`; any `confound_alarm`/`a1_stage_exit_refuse`/crashed window ⇒
ALARM marker, never extend (F1-F4 halt semantics).

**Supervision is a DETACHED, CRASH-RESUMABLE SCRIPT, not agent attention** (MAIN correction — an
agent-side timer is a harness waiter by another name): **`tools/supervise_ddm_r1c_rung1.py`**
(committed; ruff-clean; 2 recorded review passes; pass 1 found + fixed the receipt-written-but-
process-still-exiting G4 slot race — the supervisor waits for the trainer pid to fully exit before
evaluating/launching). Stateless loop re-derives ALL state from the run dirs/receipts each iteration
(restart-safe, never double-launches: live-pid + receipt spawn-guard + pidfile singleton at
`.../supervisor/supervisor.pid`). Per decision it writes `window_NN_decision.json` in the custody
root; heartbeat `supervisor_state.json` every poll (~120 s) = the receipt-existence liveness surface.
Terminal markers MAIN checks: **`rung1.done`** (+ `rung1_endpoint_manifest.json`) on STOP;
**`rung1.ALARM`** on any halt event (endpoint stage NOT run; diagnosis owed to MAIN). Window tickets
for extensions are DSL-compiled copies of the parent ticket with only the window lever swapped
(rebuild-hash smoke: the supervisor's builder reproduces window_01's ticket_hash EXACTLY).

### DECISIVE PRE-FIRE FINDING — the birth curve has ALREADY plateaued at the parent
P2 on control_tail's own telemetry (epochs_per_gate=5): **fired=True, slope=1.60 ≤ ε=1.88,
erased=411, above_nucleus_est=401** — Lane betti0_realized is flat (~574/985: last-5-gate
[565,578,572,576,574]) while 411 super-nucleus Lane components remain erased. So the *birth* process is
already at its plateau at ep500; rung-1's remaining value is the residual **coupled-descent d_seg
dividend** (pa1r: control_tail's last gates read COUPLED_DESCENT, d_seg still descending via component
SHARPENING, not new births). This bounds rung 1: capture the residual descent (a better/lower-S parent),
measure the endpoint, STOP — do NOT chase births that have plateaued (tree-vortex guard).

### Window ledger
| window | resume ep | epochs (range) | ticket_hash | fire receipt | stop_reason | n600 d_seg | tok B | S_add | birth-key fired |
|---|---|---|---|---|---|---|---|---|---|
| 01 | 501 | 501→640 (140) | faa4a888… | pid 16550, sealed_sha256 ba3c1a47… | PENDING | PENDING | PENDING | PENDING | PENDING |

## §3 ENDPOINT OBLIGATIONS (gc12; all three) — disposition STATED (no ambiguity)
The supervisor's terminal stage owns them as follows; results land in `rung1_endpoint_manifest.json`:
1. **n600 realized verdict — FOLDED INTO the supervisor terminal stage.** It subprocess-runs
   `experiments/ddm_pa1r_endpoint_verdict.py <final_window_dir> --chunk 100` (EMA shadow, re-engaged
   STE quant, R→uint8→frozen CPU-torch SegNet vs gt_n600, real SMEVR byte-close — the pa1r harness
   validated to +1.9e-7) and records the verdict + deltas vs the parent baseline (d_seg 0.0049411 /
   S_add 0.67325 / 269,028 B). On harness failure the manifest carries `endpoint_verdict_error` +
   an explicit `endpoint_verdict_owed` line for MAIN — fail-visible, never silent.
2. **P re-measure — SPLIT.** The manifest carries the endpoint `erased_count` +
   `above_nucleus_erased_estimate` from the final birth-key row, labeled **DERIVED-ESTIMATE**
   (tr1 4-conn betti0 × QA91 8-conn area frac — the lp2 honest caveat). The EXACT S-unit pool P
   (QA92 base-pass method, 8-conn per-component flip-mass on the endpoint render) is **left to MAIN**,
   named in the manifest (`p_remeasure.exact_owed_to_main` → `experiments/ddm_qa92_carrier_discriminator.py`).
3. **QA80 field staleness re-check — FOLDED IN** (mechanical): the terminal stage re-reads the final
   window's `tr1_config.json` and re-affirms the NOT-CONSUMED verdict (`token_quant_coupling_field`
   null ∧ `token_quant_margin_coupling` off ∧ `delta_sparsity_weight` 0.0 ∧ `renderer_head_mode` rgb;
   the margin term is the LIVE per-step `_live_margin_weight` — fresh by construction). Any consumer
   active ⇒ `CONSUMER_ACTIVE_FLAG_FOR_MAIN`.

## §4 verdict_scope ledger (interim)
- Seal / argv-diff / staleness: MEASURED/verified (this session).
- Birth-plateau-at-parent: MEASURED (P2 on parent telemetry, telemetry-only, no scorer).
- Window endpoints / P / final deltas vs 0.67325: PENDING — produced by the detached supervisor;
  MAIN consumes `rung1.done` + `rung1_endpoint_manifest.json` at its next wake (natural wake points:
  tp1/fh1 landings). Task 803 stays `in_progress` until MAIN consumes the endpoint.
- Given the parent-side birth-plateau finding, the likely trajectory is: window_01 completes → the P2
  key fires (births flat, erasure persists) → STOP after ONE window with the endpoint stage. That is
  the sealed mechanical rule operating correctly, not an early quit: rung-1's value is the captured
  residual coupled-descent (first gate ep504 d_seg 0.004903 already < parent 0.0049411).
- No prior negative re-opened; no trainer edit; no new lever; pointer UNMOVED.

**Pointer 0.1910828242 [contest-CPU] UNMOVED.** [no-triality] [p0-ledger-ok]
