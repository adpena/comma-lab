# Fresh-Eyes Confound Hunt H4 — CONFIG↔LIVE DRIFT + RESUME/CHECKPOINT INTEGRITY + LOSS-SCALE

**Surface owner:** H4 (blind-parallel; siblings H1=liveness/guards/schedule, H2=measurement/verdict-authority, H3=lever-efficacy).
**Target:** live C0 baseline run `experiments/results/levelset_n600_witness_20260715T095030Z/` (pid 72377, `v9_cgauge_ideal_mod19`).
**Mode:** $0, report-only, no training/dispatch/score claim. git `cacff6c1a2`.
**Run age at hunt:** ~12 min (ep0 setup; last row `mem_probe before_v0_verdict`; RSS ~26 GiB).

## L3 verdict-clearance precondition (the apparatus-validity assertion this hunt clears)

For C0's eventual multi-day d_seg verdict to be trustworthy, THREE apparatus preconditions must hold on THIS run:
1. **the live config == the sealed DSL spec** (no hand-drift — the run must BE `v9_cgauge_ideal_mod19`, not a hand-mutated cousin);
2. **checkpoints are per-stage-resumable** (a crash/OOM in a 7.5-day run must not lose all work / must continue bit-faithfully);
3. **no loss-scale pathology silently corrupts descent** (no term silently dominating/vanishing without an alarm).

**VERDICT: all three preconditions HOLD. No config/resume/loss-scale confound found on this run.** Details + cites below.

---

## SURFACE 1 — CONFIG↔LIVE DRIFT: **CLEAN (proven byte-for-byte)**

Method (recall-guarded): re-compiled the config fresh via `tac.witness_dsl.spec_v9_cgauge.compile_v9_cgauge_ideal_mod19_launch_config()` (the EXACT compiler the run's family tag names) and diffed its 224 trainer flags key-by-key against the live `launch.sh`. This heeds FEED-v752cfg (2026-07-10): a prior "config drift" finding was FALSE because it diffed against the WRONG compiler function — so I diffed against the named one.

- **Flag-set:** live 224 == fresh 224. `keys-only-live = []`, `keys-only-fresh = []`. **No hand-appended flag; no dropped flag.**
- **Values:** CLEAN value mismatches (excl. `--out-dir`) = **0**. The only difference is `--out-dir` (per-run locator: fresh default `.../v9_cgauge_ideal_mod19_20260713`, live `.../levelset_n600_witness_20260715T095030Z`) — expected, not drift.
- **Derived constants are LawRef-resolved, NOT stale literals.** `constants_manifest.json` (run dir) matches a fresh recompile value-for-value on all 8: `softmax_temp_end 0.31`, `hosc_beta_end 3.177`, `lr_anneal_epochs 1000`, `lr_hold_frac 1.0`, `polyak_finisher_start_epoch 2546`, `seg_phase_advect_start_epoch 726`, `eikonal_retention_tau_rung {base 0.01, end 0.05}`, `margin_saliency_reachability False`. Each carries `ladder_class` ∈ {measured_anchor, derived_at_config} + equation_id; launch.sh carries the RESOLVED value (e.g. `--hosc-beta-end 3.177`, the `derived_at_config` output — NOT the inherited `10.0` the manifest records as `inherited_manifest_value_replaced`).
- The two `# LAUNCH_READINESS_DEFER:` header comments (HorizonWeightedMargin, StepNativeActivation) are expected: this run IS the C0 control those phase-2 treatments A/B against (per P0 campaign queue).

**Cited fact:** `launch.sh` 224 flags ≡ `compile_v9_cgauge_ideal_mod19_launch_config().to_trainer_flags()` (0 key diffs, 0 value diffs modulo per-run out-dir); `constants_manifest.json` 8/8 constants == fresh recompile.

**Low-confidence adjacency for H3 (NOT a config confound):** `hosc_beta_end`'s manifest provenance says RE-DERIVE on any change to `muon-start-epoch`/`--anneal-epochs`/`--hosc-beta-anneal`; the derivation's β-slope-reproduction reasoning (endpoint pinned to reproduce the control's β(ep) on [1,726] over the shared 3000-ep denominator) is a lever-EFFICACY question, out of my surface. Config↔live parity is clean regardless.

## SURFACE 2 — RESUME / CHECKPOINT INTEGRITY: **CLEAN (mechanism compliant; no ckpt yet is expected)**

- **No checkpoint file exists yet** — the run is at ep0 setup (~12 min in); `--ckpt-every 25` → first rolling ckpt at ep25 (~90 min at 3.60 min/ep). This is EXPECTED, not a durability confound.
- **Atomic writes:** `_do_checkpoint` writes every npz via `_atomic_savez` (tmp + `os.replace`, refuses `/tmp`) — no partial/corrupt npz on mid-write death (`experiments/train_levelset_witness_realized_through_R_mlx.py:666`, `:9014-9017`).
- **EMA-shadow, not live:** the deploy/byte-close npz `levelset_witness_ema_mlx.npz` ships the EMA SHADOW (`:647`, `_build_ema_checkpoint_arrays(deploy_shadow_np,...)`); live weights + optimizer + RNG go to the SEPARATE `levelset_resume_state.npz` sidecar (`:9017`). Satisfies the EMA-shadow-at-inference non-negotiable.
- **Per-stage PRESERVED, distinct-filename, not overwritten:** at every curriculum transition `is_transition` fires `_do_checkpoint(stage_tag=...)` → `levelset_ckpt_{stage_tag}_ep{epoch}.npz` + `levelset_resume_{stage_tag}_ep{epoch}.npz` (`:12303-12318`, `:9032-9040` "PRESERVED stage-encoded ckpt (NOT overwritten -> per-stage A/B)"). Muon-phase tagged distinctly.
- **Intra-stage periodic:** `do_periodic = ckpt_every>0 and ep%ckpt_every==0` → rolling latest (`:12308`).
- **NOT loop-end-only:** both transition + periodic saves fire INSIDE the epoch loop; the FORBIDDEN loop-end-only pattern is absent.
- **Resume is fail-closed & bit-faithful:** `--resume-from` restores decoder + EMA shadow + optimizer + RNG streams + spike-guard window via the canonical resume registry; ~20 `_resume_*_divergences` guards refuse a resume that silently drops/changes a lever (`:1144`, `:1628-1637`, `:3616-3617`). Poisoned-resume (deadlocked state) is guarded (`:12568`).

**Cited fact:** `_do_checkpoint` (`:8971`) writes atomic EMA-shadow deploy npz + separate live resume sidecar + per-stage PRESERVED distinct-filename ckpts, fired at both transitions (`:12303`) and every `--ckpt-every 25` (`:12308`) inside the loop.

## SURFACE 3 — LOSS-SCALE: **CLEAN (three composed stabilizers + instrumented alarms)**

- **grad-clip / per-group / normalize compose sanely, in a documented order** (`:11753-11930`): (1) global `optim.clip_grad_norm(mean_grads, 0.5)` → `gnorm` reference (spike-guard/telemetry, left untouched); (2) `--per-group-grad-clip` re-clips EACH top-level param group to 0.5 independently — the **C4 confound FIX** preventing a volatile regularizer gradient from throttling seg/pose grads on other groups via the shared 1/gnorm scale; (3) `--grad-normalize per-param` normalizes each leaf to unit norm on the ALREADY-clipped tree (deep-unroll stabilizer, gnorm telemetry preserved). These are complementary defensive stabilizers, not a pathological stack. Per-param normalization further makes the step scale-invariant to absolute loss magnitude (weights set mixing DIRECTION, not step size), so large absolute weights cannot silently blow up the step.
- **The "silent" leg is killed by two L1 alarms:** `gnorm_hijack` (`_GNORM_HIJACK_MULT = 100.0`, `:11815`) fires when global grad-norm >> clip budget (seg-starvation risk); `term_domination` (`_TERMDOM_FRAC = 0.40`, `:10342`, `:11842`) fires when any single reg term > 40% of the post-weight total loss. Plus per-term loss telemetry (`:11828`, #304 item 4) logs the full term breakdown every chunk, BEFORE the skip branch (covers spike-skipped chunks too).
- **`--w-seg 100 --w-pose 1.0`:** pose is BANKED/blind until ep726 (`pose_finish_armed start_backstop 726`; `--pose-finish-start-epoch 726`) — the run.log confirms "pose-blind until d_seg converges, then terminal joint pose-descent". So w-pose 1.0 vs w-seg 100 is not a live imbalance during the seg phase; it is by design.
- **`--weight-entropy-penalty-lambda 15.0`:** a rate regularizer on weights; its post-weight contribution is watched by `term_domination` (>40% ⇒ loud). Under per-param normalization it shapes direction, not step magnitude. Whether it distorts d_seg descent is a lever-efficacy question (H3); the SCALE surface is instrumented, so it cannot silently dominate/vanish.

**Cited fact:** grad path = global-clip(0.5) → per-group-clip(0.5) → per-param-normalize (`:11753/:11762/:11922`), with `gnorm_hijack` (100×, `:11815`) + `term_domination` (40%, `:11842`) L1 alarms + per-term telemetry (`:11828`) instrumenting the domination/vanish modes.

## SURFACE 4 — WALL-CLOCK / THROUGHPUT REALISM: **CLEAN (no silent truncation before convergence)**

- **Projection vs budget vs hard cap:** nominal 3000 ep × 3.60 min/ep = **7.50 d**; `cfg.wall_clock_budget_days = 8.314 d` (DERIVED: `scorer_throughput_gate.derive_wall_clock_budget_days(RUN1_MEASURED_MIN_PER_EP × epochs × WALL_CLOCK_SLACK_FACTOR)`, ~11% slack over nominal); **safe_run hard `--timeout 1209600.0 s = 14.0 d`** (from `run.log` durable-daemon launch line).
- Ordering: **7.50 d (nominal) < 8.314 d (budget) < 14.0 d (hard cap)**. The 8.314 d budget is an ESTIMATE, not the truncation point; the only hard truncation is safe_run's 14 d. Headroom 14/7.5 = **1.87×** — even a 50% per-ep slowdown (→11.2 d) finishes before the cap. No silent pre-convergence truncation.
- **Fast-path is genuinely engaged** (so 3.60 min/ep is real, not optimistic): `run.log` shows `custom_grouped_backward active:true` ("~17x backward") + `--fused-r-kernel` in argv. `--verdict-batch 32` is set (the #205 OOM fix); `mem_probe before_v0_verdict rss 23.68 GiB / mlx_active 11.6 GiB` << safe_run `--rss-mb 90000` (90 GiB) ⇒ no OOM-truncation risk at the verdict-batch spike.

**Cited fact:** nominal 7.50 d < budget 8.314 d < hard safe_run timeout 14.0 d (`run.log` `--timeout 1209600.0`); fast-path (`custom_grouped_backward active:true` + `--fused-r-kernel`) confirms the min/ep basis.

---

## Ranked findings (all CLEAN; no confound manufactured)

| # | Surface | Signature | Cited fact | Poison-scope | L1/L2/L3 status |
|---|---------|-----------|-----------|--------------|-----------------|
| 1 | Config↔live drift | none | launch.sh 224 flags + 8 constants ≡ fresh `compile_v9_cgauge_ideal_mod19_launch_config()` (0 diffs modulo per-run out-dir) | — | CLEAN. L2 already covered by #332 provenance-bijection gate; run IS the sealed spec |
| 2 | Resume/checkpoint | none | `_do_checkpoint` atomic EMA-shadow + per-stage PRESERVED + intra-stage@25, resume fail-closed (`:8971`,`:12303`,`:1144`) | — | CLEAN. Mechanism compliant; no ckpt yet = ep0 (expected). L1 `frozen_epoch`/liveness = H1 surface |
| 3 | Loss-scale | none | grad-clip(0.5)→per-group→per-param-normalize + `gnorm_hijack`(100×) + `term_domination`(40%) alarms (`:11753`,`:11815`,`:11842`) | — | CLEAN. Silent-leg killed by L1 alarms; composition documented |
| 4 | Wall-clock | none | 7.50 d nominal < 8.314 d budget < 14.0 d safe_run cap; fast-path active | — | CLEAN. 1.87× headroom; no silent truncation |

## Honest limit

This clears the config/resume/loss-scale APPARATUS-VALIDITY preconditions on this run AS OF ep0 setup. It does NOT clear: (a) whether descent actually converges by 7.5 d (that is the run's outcome, not an apparatus fact); (b) lever efficacy / inert-vs-binding (H3); (c) liveness/guard-firing at ep>0 (H1); (d) verdict-authority (H2). A per-ep-time regression that pushes wall-clock past 14 d, or a resume executed with a divergent lever set, would each re-open surface 2/4 — both are guarded (fail-closed resume; 1.87× cap headroom) but should be re-checked if the run is ever resumed or slows.
