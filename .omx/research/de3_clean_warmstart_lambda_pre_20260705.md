---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "the headline is a NEGATIVE dressed as rigor: we could not MEASURE the gold sharpness, so we cannot say it is flatter OR sharper. Do not let the built mechanism (the flag) launder the unmeasured precondition into a GO — warm-start-from-gold stays a GATED secondary, not the primary, exactly because the probe walled."
council_assumption_adversary_verdict:
  - assumption: "lambda_pre at the gold basin is measurable with the same FD-HVP probe that gave 3.66e6 at ep100"
    classification: CARGO-CULTED
    rationale: "MEASURED-falsified: the power iteration CONVERGES at ep100 (rel 0.025-0.134) but does NOT converge at the gold 0.025 basin in EITHER moment config (rel 1.05 restored / 13.8 eps-floor; oscillating sign + growing residual across iters=12 AND iters=30). A low-loss near-converged basin has a small gradient => the FD-HVP signal falls below the fp32/GPU noise floor (amplified through eps-floor/small-v preconditioner coords). The probe is a clean instrument at a mid-training wall, NOT at a converged basin."
  - assumption: "a clean warm-start-from-weights needs a NEW trainer flag"
    classification: CARGO-CULTED
    rationale: "MEASURED-falsified by source trace: --resume-from a DEPLOY ema/BEST npz ALREADY loads the weights as LIVE with has_opt=False (fresh AdamW) + no __recent_losses (fresh spike guard) + epoch=__epoch (=> start 126). The deploy-npz path IS the clean warm-start. The new --warm-start-weights-only flag's marginal value is ONLY the poisoned-sidecar case (force weights-only from a full resume_state)."
council_decisions_recorded:
  - "op-routable #1 (analysis, no launch): DE#3 flatter=>safer precondition is UNCONFIRMED (gold lambda_pre unmeasurable); warm-start-from-gold is NOT licensed on safety grounds"
  - "op-routable #2 (built, default-off): --warm-start-weights-only + --warm-start-epoch flags + the deploy-npz clean-warm-start path; byte-identity of the default resume path proven"
  - "op-routable #3: DE#3 config = PRIMARY clean-start-from-ep100 + adaptive-eps (#320); warm-start-from-gold a GATED secondary only"
related_deliberation_ids: [witness_config_differential_equations_derivation_20260705, eik_stab_build_20260705, stepping_instability_diagnostic_20260705]
---

# DE #3 — CLEAN WARM-START-FROM-WEIGHTS + the λ_pre-at-gold binding pre-condition

**Axis discipline: every λ_pre number is `[n24 advisory — mechanism probe, NOT n600 evidence]`.
NOTHING trained (the probe EXITS before any step); NOTHING launched at n600. Pointer contest-CPU
0.19110 UNMOVED — this whole unit is means/apparatus.** Build commit provenance: git `4cb47237a`
(pre-commit); artifacts `experiments/results/de3_lambda_pre_at_gold_20260705/`.

## TL;DR (the measurement decided, not the hope)

1. **λ_pre at the 0.025 GOLD basin is UNMEASURABLE with the FD-HVP probe** — it CONVERGES cleanly at
   ep100 (rel 0.025–0.134, reproducing the eik-stab sibling's 3.663e6 EXACTLY) but does NOT converge
   at the gold basin in EITHER moment config (restored-moments rel 1.05; eps-floor rel 13.8;
   sign-oscillating with growing residual across iters=12 AND iters=30). ⇒ the DE#3 "λ_pre(gold) <
   λ_pre(ep100) ⇒ warm-start" precondition **cannot be established** ⇒ the flatter/safer claim is
   **UNCONFIRMED**. Per the DE's own rule (derivation §DE#3 lines 208-209: "else the basin is NOT
   safer"), warm-start-from-gold is **NOT licensed on safety grounds**.
2. **The clean warm-start-from-WEIGHTS path ALREADY EXISTS** for the gold via the deploy-npz route:
   `--resume-from …/levelset_witness_ema_BEST.npz` loads the gold weights as LIVE with **fresh AdamW
   moments** (`has_opt=False`), **fresh spike guard** (no `__recent_losses`), and **epoch 125→126** —
   the exact clean-warm-start the DE#3 wanted, no new code. The NEW `--warm-start-weights-only` flag's
   marginal value is closing the **poisoned-resume FOOTGUN**: forcing weights-only + fresh moments +
   cleared guard + an epoch override EVEN from a full `levelset_resume_state.npz` (whose moments are
   the ep150 deadlock and whose `__recent_losses` is the runaway window).
3. **DE#3 config recommendation:** PRIMARY = the sibling's **clean-start-from-ep100 + adaptive-ε**
   (#320 visco_03 GO, the MEASURED-stable arm at the killer lr). Warm-start-from-gold = **GATED
   SECONDARY** only (creep/spike gates armed), since the sharpness precondition did not clear it.

## 1. The measurement (deliverable 1)

**Method — reuse, not reimplement.** The eik-stab build's `--lambda-pre-probe-iters` mode (Adam-
preconditioned power iteration, forward-diff HVPs over the full n24 batch gradient, fp64 accumulation,
central-difference consistency row) + the stepping-probe `slice_snapshot`/`_base_argv` helpers are
used verbatim. The ONLY new step is **snapshot-doctoring**: copy the ep100 resume snapshot and swap
its shadow (`emaP__`) + live (`liveP__`) model-param arrays for the GOLD `ema_BEST` weights, keeping
the ep100 optimizer moments + step + cfg identical. Resume seeds live←shadow, so the probe evaluates
H at the GOLD 0.025 basin under a **bit-identical preconditioner** — the ONLY A/B variable is the
weights (pure basin geometry). Runner: `experiments/probe_lambda_pre_at_gold.py`.

**Four probes** (n24, GPU; iters 12 for ep100, 12+30 for gold):

| probe | weights | moments (preconditioner) | λ_pre | central | rel | converged? |
|---|---|---|---|---:|---:|---|
| **E_restored** | ep100 | ep100 (v_norm 6.4e-3, step 6837) | **3.663e6** | 3.23e6 | 0.134 | **YES** (reproduces sibling) |
| **E_fresh** | ep100 | zero (eps-floor, uniform) | −9.985e11 | −1.024e12 | 0.025 | **YES** |
| **G_restored** | **GOLD** | ep100 (identical to E_restored) | 5.9e7 | 2.9e7 | **1.05** | **NO** |
| **G_fresh** | **GOLD** | zero (eps-floor, identical to E_fresh) | 2.06e13 | −1.6e12 | **13.8** | **NO** (oscillating) |

**Reading:**
- **E_restored reproduces 3.663e6 exactly** ⇒ the doctoring pipeline is validated; G_restored's only
  difference from E_restored is the weights (same v_norm 6.4e-3, same opt_step 6837 — confirmed in the
  `lambda_pre_probe_start` rows).
- **The gold basin does not converge in EITHER config.** E and G differ ONLY in weights (E_fresh vs
  G_fresh share the identical eps-floor preconditioner), so the non-convergence is a property of the
  **gold basin's Hessian spectrum**, not the preconditioner. The G_fresh iter trace oscillates in
  sign (2.8e7 → −3.8e11 → 2.8e12 → 7.4e12 → −4.6e12 → …) with a GROWING residual — a classic
  non-convergent / noise-dominated power iteration. More iterations (30) did not fix it (G_restored
  rel 1.05).
- **Why (hypothesis, NON-binding):** a low-loss near-converged basin (d_seg 0.025) has a small
  gradient, so the FD-HVP signal `g(θ+hu)−g(θ)` falls near the fp32/GPU noise floor — especially
  amplified through the eps-floor / small-v preconditioner coordinates (the 2506.04805 decoupling
  suspect the eik-stab memo already named). This is AMBIGUOUS between "flat near-degenerate top
  spectrum" and "sharp ill-conditioned/non-normal operator"; the unconverged magnitudes (5.9e7–2e13)
  are NOISE, not curvature. **Not a measurement — do not read a direction into it.**

**Verdict (deliverable-1 honest answer):** the 0.025 gold basin is **NOT flatter-confirmed nor
sharper-confirmed — it is UNMEASURABLE** with this probe. The DE#3 precondition is unconfirmed.

## 2. The clean warm-start-from-WEIGHTS path (deliverable 2)

**Source trace (the honest minimal finding).** `_load_resume_state` already has a deploy-npz fallback:
a plain ema/BEST npz (unprefixed param keys + `__epoch`) loads as `live` weights with `ema={}`,
`opt={}`, `has_opt=False`, `epoch=__epoch`. Downstream the resume block seeds the EMA shadow from
`rs["ema"] if rs["ema"] else rs["live"]` (→ gold weights), sets `start_epoch=epoch+1` (=126), and the
optimizer-state restore is gated on `if rs["has_opt"]` (=False → **fresh AdamW**). The spike-guard
restore is gated on `"__recent_losses" in resume_cfg` (absent in a deploy npz → **fresh guard**). So:

```
--resume-from experiments/results/v5_dseg0026_preserved_20260705/levelset_witness_ema_BEST.npz \
    --resume-allow-lever-drift
```

**already IS the clean weights-only warm-start** the DE#3 asked for — gold 0.025 weights, fresh
moments, fresh guard, epoch 126 — with ZERO new code. This is distinct from the **poisoned** path
`--resume-from …/levelset_resume_state.npz` (the file also sitting in the gold dir): that sidecar has
`__resume_epoch=150` (the deadlock), restored ep150 moments, and the runaway `__recent_losses`
(20→114) — resuming it re-enters the v2/v3 stale-resume deadlock.

**The new flag (footgun closure).** Because the moments/guard behavior silently depends on WHICH npz
`--resume-from` targets, an operator who grabs the `resume_state.npz` (the obvious "resume" file) gets
the poison. `--warm-start-weights-only` (+ `--warm-start-epoch`) makes the weights-only intent
EXPLICIT and safe from a full sidecar: it discards `rs["opt"]` + sets `has_opt=False` (fresh AdamW),
clears the spike-guard window, auto-allows lever drift (a warm-start is an intentional re-treatment),
and lets `--warm-start-epoch 126` reset the start epoch off the deadlock epoch. Implemented as the
pure helper `_resolve_weights_only_warm_start` (unit-tested) wired at the resume site.

**Byte-identity of the DEFAULT resume path (proven):**
1. **Writer untouched** — `_build_resume_state_arrays` (the sidecar/deploy npz WRITER) has ZERO diff
   hunks (`git diff` confirmed). Saved resume/deploy bytes are unchanged.
2. **Load/warm-start path is fully flag-gated** — `--warm-start-weights-only` defaults False;
   `_resolve_weights_only_warm_start(flag=False)` returns the IDENTITY decision (opt preserved,
   `has_opt` preserved, `start_epoch = ckpt_epoch+1`, `allow_lever_drift=False`,
   `clear_spike_guard=False`) — unit-test `test_flag_off_is_noop_moments_preserved`. Each callsite
   reduces to its exact prior expression when the helper returns identity values (`… or False` ≡ the
   original; `start_epoch = ckpt_epoch+1`). F821 clean.
3. Tests: `src/tac/tests/test_levelset_warm_start_weights_only.py` — **16 passing** (flag off/on
   moments; weights untouched; epoch default/override/zero/negative; guard+lever gates; ckpt_had_opt;
   deploy-npz-fallback vs full-sidecar distinction; argparse contract + helper-wired-at-callsite).

## 3. DE#3 config recommendation (deliverable 3) — MEASURED, not hoped

The DE#3 gate was: warm-start from 0.025 ONLY IF the gold basin is not-worse than ep100. **The
measurement could not establish that** (gold λ_pre unmeasurable; weak unconverged signal ambiguous).
Per the DE's own rule ("else the basin is NOT safer"), the recommendation is:

- **PRIMARY (trusted):** `clean-start-from-ep100 snapshot + adaptive-ε` (sibling #320's `visco_03`
  GO — the arm MEASURED-stable at the killer lr 1e-3, n24 40-ep, descending 75.7→18.4, best d_seg
  0.031). This is the arm with an actual stability measurement behind it.
- **SECONDARY (optional, GATED):** warm-start-from-gold-WEIGHTS
  `--resume-from …/levelset_witness_ema_BEST.npz --warm-start-weights-only --warm-start-epoch 126`
  + adaptive-ε, run ONLY with the pre-registered creep/spike gates armed (SC1' every-epoch skip
  alarm + eikonal trough-ratio ≤ 2). Upside: starts at d_seg **0.025** (vs ep100's ~0.07),
  potentially saving ~25 epochs IF it does not re-deadlock. Its safety is **probe-UNPROVEN**, so it
  is an experiment, not a default. Do NOT promote it to primary on the (unconfirmed) flatness hope.

## 4. Honest risks / limits

1. **The probe walled at a converged basin** — the FD-HVP instrument is calibrated for a mid-training
   sharpness wall (ep100), not a low-gradient converged basin. A cleaner gold measurement would need
   central-diff HVPs throughout + CPU bit-exact + a smaller/auto-scaled fd + many more iters (or an
   analytic/Lanczos HVP). Not built here (time-boxed; the NEGATIVE is already decision-sufficient:
   unmeasurable ⇒ precondition unconfirmed ⇒ gated-secondary).
2. **n24 slice** — all λ_pre are the 24-pair full-batch H, the same slice the 3.66e6 used (apples).
3. **Deterministic-repro** — the deploy-npz warm-start advances epoch 125→126; the poisoned
   resume_state path is the trap to AVOID. `--warm-start-weights-only` persists none of the deadlock
   state by construction.

## Artifacts
- Runner + verdict: `experiments/probe_lambda_pre_at_gold.py` +
  `experiments/results/de3_lambda_pre_at_gold_20260705/de3_lambda_pre_at_gold_report.json`
  (+ per-probe `*/trainer_out/lambda_pre_probe.json`, `*/probe.log`, `run_all.log`,
  `run_gold_iters30.log`).
- Trainer flags + helper: `experiments/train_levelset_witness_realized_through_R_mlx.py`
  (`_resolve_weights_only_warm_start`, `--warm-start-weights-only`, `--warm-start-epoch`).
- Tests: `src/tac/tests/test_levelset_warm_start_weights_only.py` (16).
