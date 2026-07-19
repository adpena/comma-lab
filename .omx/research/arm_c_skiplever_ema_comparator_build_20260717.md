# ARM-C build memo — Lane stride-2 skip lever (#524) · EMA decay LawRef (p0_ema_calibration) · shadow-vs-live byte-close comparator (2026-07-17)

Branch `p0_build_skiplever_ema_20260717` · SSoT: SPEC_v10 §13.1 row 4, §13.3, §13.5
(`claude/p0_521_spec_v10_capstone_20260717`). All levers DEFAULT-OFF, byte-identical when unset.
Pointer UNMOVED (everything here is MEANS; no score claim anywhere in this memo).

## 1. Deliverable 1 — Lane stride-2 skip-band lever (task #524)

**DERIVED chain (from MEASURED facts in `segnet_recursive_fractal_factorization_20260715.md` §5):**
1. MEASURED — final SegNet decoder block is skipless (conv1_in=32): all sub-stride-4 boundary
   localization flows through the ONE 16-ch stride-2 skip at (192,256).
2. MEASURED — destroying the skip's sub-stride-4 detail (down-up 2× ablation) induces 8,072 flips,
   77% Road-Lane: Lane is THE skip-limited pair.
3. DERIVED — the render-side structure that survives the skip path is exactly the DETAIL BAND that
   ablation destroys: `SB(x) = D2(x) − U2(D2(D2(x)))` (D2 = 2×2 avg-pool, U2 = nearest ×2) on
   BT.601 luma/255, at the skip's own (192,256) grid — the [stride-4, stride-2) Nyquist band.
   ASSUMED (stated) — the learned 3×3 stride-2 stem is approximated as a band-limiter; SB is the
   band-limited render-side sufficient statistic UP TO that local linear filter.
4. DERIVED lever — supervise witness SB toward GT SB on the dilated GT-Lane band (Lane class 1,
   comma10k CANONICAL order): masked MSE, mean over band. Lane markings are thin all-boundary
   double-edges, so the dilated class support IS the boundary band.

**Built:** `tac.boundary_math.lane_skipband` (numpy-fp32 reference authority: luma/pool/detail/
mask/term + CLOSED-FORM adjoint gradient, finite-difference-verified) + levelset-trainer wire-in
(`--lane-skipband-weight/-start-epoch/-dilate`, default 0.0 ⇒ byte-identical; term rides the SHARED
realized through-R `_f1` — no 2nd render/SegNet forward; engagement gate + spike-guard re-treat;
resume-divergence persistence `__cfg_lane_skipband_*`; `lane_skipband` in LOSS_TERM_KEYS both
surfaces; fail-closed `--micro-batch-pairs > 1` guard) + DSL `LaneSkipBand` Lever factory
(registry-discovered; never-invent-flags test green). Provider precompute ~236 MB @n600
(f32 target+mask at (192,256); NOTED for the launcher memory-preflight, same class as the
phase-advect providers).

**$0 bindingness probe (MEASURED, `tools/probe_lane_skipband_bindingness.py`; n=24 pairs of the
real mod32cap EMA render via the canonical numpy-fp32 oracle on the byte-closed dequantized blob;
artifact `experiments/results/shadow_vs_live_mod32cap_ep1000_smoke_20260717/lane_skipband_bindingness_n24.json`):**
- `binds_when_enabled = true` on all 24 scored pairs: term_on mean **1.549e-3** (> 0; OFF path is
  structurally 0), closed-form grad L2 mean **1.05e-3** (> 0 ⇒ enabling changes dL/d(render) on the
  real inputs).
- DIAGNOSTIC: witness lane-band skip-band energy **1.68e-4** vs GT **1.70e-3** — the witness
  carries only **~10%** of GT's Lane skip-band detail. The fractal memo's prediction (witness
  under-carries the band that owns 77% of Lane skip flips) is CONFIRMED as a live deficit on the
  real cached render. Bindingness proof, NOT a d_seg claim; 24-pair subset stated; d_seg effect is
  RUN-GATED (duty-to-measure A/B). `# FORMALIZATION_PENDING: the deficit ratio becomes a canonical
  equation anchor with the first measured lever A/B (registering a law from a 24-pair diagnostic
  alone would be premature).`

**Composition/antagonism vs existing levers:** rides the same `_nonwa` shared forward as
lane-edge/margin-saliency/chroma-boundary (bit-identical sharing; +0 SegNet forwards). It is a
RENDER-side band-matching term — orthogonal to the margin-hinge levers (which push logits) and to
chroma-boundary (chroma; SB is luma-band). Possible overlap with `LaneRenderBand` (the analytic
lane band composite): under `lane_offloaded` regimes the analytic band carries Lane — the skip-band
lever then supervises a surface the analytic band paints, so compose with care (measure with
lane_render_band OFF first). Antagonism risk: pulling SB toward GT adds texture pressure inside
the dilated band that the satisfice/taper levers deliberately relax elsewhere — band-limited to
the Lane band, so bounded.

## 2. Deliverable 2 — EMA decay LawRef + finisher duty registration

**Canonical equation `ema_decay_run_geometry_v1`** (module
`src/tac/canonical_equations/ema_decay_run_geometry_20260717.py`, evaluator LawRef-executable in
`evaluators.py`): exact identities `eps = d^U`, `W = 2/(1−d)`, `phi = W/U`, inverted:
`d = eps^(1/U)` (pin terminal seed fraction) / `d = 1 − 2/(phi·U)` (pin warmup completion).
MEASURED basis (SPEC §13.3 + live c2 run): incumbent 0.997 @ 1 update/epoch ⇒ warmup 667 updates
(executable cross-check `ema_warmup_updates(0.997)=667`) ≈ ep1318/1400; warm-start seed retains
0.997^149 = **0.6391** at ep800 (SPEC-recorded ~64% — residual 0.0009) and 0.997^749 = **0.1054**
at terminal; EMA−live −0.00095 @ep775 (SPEC-recorded). The Quantizr per-step-minibatch provenance
does NOT transfer to full-batch (noise-averaging rationale vanishes) — calibration, not a bug.

**DSL:** `EmaDecayCalibrated(updates_per_run, target_seed_fraction | warmup_fraction)` — resolves
`--ema-decay` through the LawRef compiler (`derived_at_config` rung; lawrefs + constant_manifest
custody; mode passed as numeric code 1..4 because LawRef literals are numeric-only). NOT composing
it leaves the trainer default 0.997 (byte-identical). `EmaDecayFinisher` factory now HOLDS the
built-never-fired `--ema-decay-finisher` (SWA-style wider finisher).

**Duty/activation registration (EXECUTED against the ledger the costate reads):**
`tools/register_ema_finisher_duty.py --ledger-root /Users/adpena/Projects/pact` appended an honest
UNMEASURED significance row for `EmaDecayFinisher` (no guessed ΔS) to
`.omx/state/lever_relative_significance.jsonl`; verified `duty_to_measure_ranked()` on main now
surfaces it (state `not-registered` → flips to duty-to-MEASURE `never-fired` once this branch's
factory merges). Idempotent CLI; re-run on main post-merge is a no-op or upgrade.

## 3. Deliverable 3 — shadow-vs-live byte-close comparator (SPEC §13.5 gate)

**Built:** `tools/compare_shadow_vs_live_byte_close.py` — given a per-stage RESUME npz
(`emaP__*`/`liveP__*` weight sets; key layout MEASURED from
`levelset_n600_witness_20260717T113932Z/levelset_resume_stageMuonStart_ep726.npz`, 187 keys) plus
the paired deploy stage ckpt (cfg custody template), it materializes deploy-format ema+live npz
(SAME cfg bytes, only weights differ), enforces an ANTI-ALIASING gate (key-set equality; max-abs
ema-live delta > 0; template==EMA-shadow check; distinct npz sha256 — refuses a fake two-arm
comparison), then byte-closes + realized-scores BOTH arms through the real decode harness
(`select_best_weights_arm(arms=["ema","live"])` = shipped inflate + numpy oracle + frozen
CPU-torch parity) and emits paired provenance-stamped rows (`score_claim=false`,
`promotable=false`, `[macOS advisory] NON-PROMOTABLE`).

**SMOKE (MEASURED; mod32cap `levelset_resume_stageTau_muon_ep1000.npz`, gt_n96):**
| arm | pairs | d_seg realized | d_pose realized | archive B | S_advisory |
|---|---|---|---|---|---|
| ema | 96 | **0.003976** | 141.88 | 83,442 | 38.120 |
| live | 96 | 0.004884 | 138.64 | 83,419 | 37.779 |
(n8 smoke consistent: ema 0.003535 / live 0.004058.) Anti-aliasing verified: max-abs ema-live
delta > 0; template==EMA exact; distinct sha256; BOTH arms decode (no aliasing).
**HONEST caveat:** mod32cap is pose-BLIND (w_pose=0) — the composite-S "winner=live" is driven by
the garbage pose term (the tool's own POSE-BLIND banner fires); the MEANINGFUL paired axis here is
d_seg, where the EMA shadow beats live by **18.6%** — consistent with the EMA non-negotiable.
96-pair subset = SMOKE scope (stated in the report's verdict_scope), not n600 evidence; the §13.5
gate at the c2 terminal/per-stage ckpts is the real decider. Artifacts:
`experiments/results/shadow_vs_live_mod32cap_ep1000_{smoke,n96}_20260717/` (mirrored to the main
repo; worktree copies are transient).

## 4. Value-provenance labels (summary)

- MEASURED: fractal §5 skip numbers (upstream memo); bindingness term/grad/energy rows (n24);
  comparator paired rows (n8/n96); warmup 667; seed fractions 0.6391/0.1054 (exact closed forms of
  the incumbent config); ledger row append + costate queue surfacing.
- DERIVED: SB operator form; lane-band construction; d = eps^(1/U) and d = 1 − 2/(phi·U); the
  closed-form term gradient (FD-verified).
- INFERRED: EMA-doctrine consistency of the d_seg gap (ema < live) on a 96-pair pose-blind ckpt.
- ASSUMED (stated): stem-as-band-limiter approximation; avg-pool/nearest operator choice for the
  ablation complement; 0.05 starting weight and 0.999 finisher decay (RUN-GATED, not optima).

## 5. ROUND-1 ADVERSARIAL SELF-REVIEW (attack surfaces; every lever BINDS when enabled)

1. **SB proxy vs the actual 16-ch skip features.** The lever supervises the render-side BAND, not
   the encoder's skip activations. The strong form (render → learned stem → 16-ch target) costs an
   encoder forward in-loop. The probe shows the band form already has a live 10× deficit target;
   feature-space upgrade is a named follow-up if the A/B under-delivers.
2. **Operator-form ambiguity.** The memo's ablation "down-up 2×" interpolation kernel isn't
   specified; I fixed avg-pool + nearest-up canonically in the reference module (tested). A
   bilinear variant would shift band edges slightly — implementation-level, not paradigm.
3. **Micro-batch twin NOT wired** for `lane_skipband` — fail-closed guard refuses
   `--micro-batch-pairs > 1` (counted-but-inert is a NO-FAKE violation; refusing beats
   half-wiring). Live config family runs B=1, so the guard does not bite the incumbent path.
   Twin wire-in = round-2 item.
4. **No end-to-end trainer step executed** with the lever ON (needs the full MLX+GT stack; the MLX
   twin expression is parity-tested in isolation and the branch is source-scan-tested). A 1-epoch
   n6 trainer smoke with `--lane-skipband-weight 0.05` is the round-2 verification.
5. **Provider RAM (~236 MB @n600) not added to `witness_memory_preflight` projection** — noted in
   the stage row like the phase-advect providers; preflight-projection addition owed (gated
   default-off, so no current-launch exposure).
6. **Comparator winner-by-S is misleading on pose-blind ckpts** (measured here). The report keeps
   the full per-arm rows and the caveat; a pose-blind-aware ranking mode (rank by d_seg when
   `pose_blind`) is a follow-up.
7. **EmaDecayCalibrated resolves at factory-call time** — a stale updates_per_run (config edited
   after compose) would carry a stale decay; the LawRef manifest records inputs so the #332
   bijection/self-recompile surfaces drift. Mode-as-numeric-code is a workaround for numeric-only
   LawRef literals; documented in the evaluator.
8. **Ledger registration pre-merge** — the factory exists only on this branch until merge; the
   significance row (main ledger) surfaces as duty-to-BUILD meanwhile. Idempotent CLI re-run
   post-merge upgrades it to duty-to-MEASURE. No fake "registered+held" claim before merge.
9. **Skip-band vs LaneRenderBand composition unmeasured** — first A/B should run with
   lane_render_band OFF to avoid supervising an analytically-painted surface.
10. **`upstream` symlink in the worktree** (needed for the smoke: worktrees lack the gitignored
    pinned snapshot) is untracked and NOT committed; no upstream bytes touched (read-only).

## 6. Triality legs

- **equations:** `ema_decay_run_geometry_v1` registered (builder + LawRef evaluator + tests);
  skip-band law FORMALIZATION_PENDING (waiver + rationale in §1).
- **DSL:** `LaneSkipBand`, `EmaDecayCalibrated`, `EmaDecayFinisher` Lever factories
  (registry-discovered; never-invent-flags green).
- **DAG:** this memo is the durable artifact; FEED row owed at merge/A/B time (the arm's branch
  does not edit the shared DAG file to avoid merge conflicts with sibling arms — flagged for the
  merge conductor).

## STORES CONSULTED
SPEC_v10 §13 (spec branch) · segnet_recursive_fractal_factorization_20260715.md ·
frozen_scorer_exact_factorization_20260715.md (via memo refs) · CLAUDE.md EMA non-negotiable ·
activation_ledger/lever_registry sources · levelset trainer lever patterns (lane-edge/chroma/
satisfice) · tools/levelset_byte_close_and_eval.py (arm selection, ckpt loader, oracle) ·
live-run launch.sh + resume npz key layout (READ-ONLY) · confound_observability.ema_warmup_updates.
