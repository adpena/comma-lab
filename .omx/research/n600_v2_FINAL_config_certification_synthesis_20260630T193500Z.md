# FINAL n600 v2 config certification — SYNTHESIS (3-axis cert + θ* fire-order + recursive pass)

**UTC** 2026-06-30T19:35Z · **tag** `[macOS-MLX advisory · config certification · NON-PROMOTABLE]` · **pointer 0.19110 UNMOVED.**

Synthesizes the 3 parallel deep-math certifications into ONE certified n600 config + the θ* fire-order + the recursive adversarial pass:
- arch/basis/optimizer — commit `2833d25cc`, `.omx/research/n600_final_config_cert_arch_basis_optimizer_20260630T191137Z.md`
- curriculum/regularizer/seed — commit `b8c990941`, `.omx/research/n600_final_config_cert_curriculum_regularizer_seed_20260630T191327Z.md`
- θ* per-lever A/B campaign — commit `356385f4a`, `.omx/research/thetastar_per_lever_AB_campaign_ready_20260630T191903Z.md`

**means≠ends:** this CERTIFIES the config (a MEANS). Only a byte-closed n600 exact row < 0.19110 from `upstream/evaluate.py` (CPU/CUDA, never MPS) moves the pointer. NO GPU touched; the burn + θ* arms await operator GO.

## VERDICT — the n600 config is CERTIFIED
It is the **PROVEN n200 Muon arm** (realized-through-R d_seg **0.0036976 @ ep1000, still descending**; `levelset_thetastar_muon_arm/levelset_best.json`) reproduced at n600, with ONE design delta (`--mod-dim 32→26`) + the 4 review revisions.
- **34 of 39 knobs** (17 arch/basis/optimizer + 17 curriculum/regularizer/seed) OPTIMUM-CONFIRMED **by reproduction** (byte-identical to the existence proof — the strongest footing possible).
- **MANDATORY: `--muon-lr 0.002`** (7.8× the frozen base-LR; omit → default 1e-4 = 20× too low → the descent does NOT reproduce). Independently double-confirmed on both axes.
- **4 review revisions baked:** `mod-26` (capacity-safe), `hidden-96` (RD-confirmed: 120 is +0.01–0.014 S), `muon-lr-0.002`, `verdict-pairs-96`.
- **Flag validation: 78/78 + per-axis dogfoods PASS** against the real 116-flag argparse — no invented flag.

## THE RECURSIVE ADVERSARIAL PASS — 3 NO-FAKE catches, cross-checked, no contradictions (clean)
1. **(arch)** proven `mod-dim` was **32**, NOT 21 as the launch doc claimed → `mod-26` is an honest delta off 32; `mod-21` is the *predicted* RD-optimum (Whitney floor of the measured ~9-D nonlinear manifold), **unproven**.
2. **(curriculum)** `--muon-lr 0.002` MANDATORY (LR + softmax-temp both freeze at the Muon boundary, code L1905) — omitting silently 20×-undershoots the finisher.
3. **(θ\*)** warm-start MUST resume from `levelset_resume_state.npz` (has epoch+opt), NOT the EMA `BEST.npz` (no epoch → `start_epoch=1` → the temp-anneal/curriculum RE-SOFTENS the converged partition). Further: `--lane-prior-phi1` (clobbered by resume `model.update`) and `--max-bank-freq`/`--self-orient`/`--mod-dim`/`--freq-across` (change `in_feat` → resume shape-break) are **NOT valid warm-start levers** → from-scratch only.

**Cross-agent consistency check (the consolidating pass):** the arch-θ* levers agent-1 flagged (`mod-dim`, `freq-across`) are EXACTLY the ones agent-3 proved cannot warm-start (they change `in_feat`). → consistent split: **arch refinements are from-scratch; surgical-loss refinements are warm-start.** No contradiction. The β-anneal (`hosc-β`) is correctly a warm-start arm on both (activation param, no shape change). PASS — 1 clean consolidating pass over 3 independently-reviewed axes.

## THE FINAL CONFIG (ready to fire)
The `witness_autoconfig --emit-command` output (commit `daaf0d811`), which already carries `--muon-lr 0.002 --mod-dim 26 --hidden-dim 96 --verdict-pairs 96`, from-scratch, `--epochs 1000`, curriculum tau@300/l7@600/muon@726, the directional Fourier bank, the openpilot seed (`--structured-init --lane-prior-phi1`), the variational regularizers (`--eikonal-weight 0.01 --length-weight 0.001`), the stage-transition re-warmup, `--ckpt-every 25 --stage-checkpoints`, `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`. Surgical levers + DM1 OFF (attribution-clean).

## THE θ* FIRE-ORDER — TWO TIERS (because arch ≠ warm-startable)
**TIER-W (warm-start, cheap, ~13 GPU-h total, ranks surgical LOSS levers — resume from `resume_state.npz`):**
1. **A2 lane-thin** −0.0004…−0.0010 (the measured dominant residual: thin Lane dashes, 93% of <5px missed; PRIMED) — TOP EV
2. **A1 margin-saliency all-class** −0.0003…−0.0008 (defends 100% of the flip band)
3. **A3 hosc-β step-native** −0.0001…−0.0005 (Gibbs/ringing suppression; birth-death-supported)
4. **A4 hardness-realized** −0.0001…−0.0004
5. **A7 UNIWARD on A1 winner** −0.0001…−0.0003 (wave-2)
6. **A5 DM1 conditioning** −0.0000…−0.0003 (byte-free; primary value = n600 amortization)
7. **A6 lane-edge** −0.0001…−0.0004 (ablation: confirms all-class > class-1-only)

**TIER-S (from-scratch small-n, the arch/schedule refinements that can't warm-start):**
- **`mod-dim {19, 21, 26}` rate-curve** — THE top arch refinement (`mod-21` = predicted RD-optimum, ~0.003–0.005 S cheaper than 26; the manifold intrinsic-dim ~9 is pairs-independent → a small-n (n96) from-scratch sweep transfers).
- **`l7-start {480, 600}`** — τ saturates ~ep450; 600 over-runs the knee ~150 ep.
- **`freq-across` post-R Nyquist** — lower priority.

## ROOT-TRACKING SCHEDULER — SKIP-BUILD-BEFORE-BURN
Burn on the proven static schedule now; build the scheduler in parallel as a follow-on (CPU/no-GPU), gated by a $0 anneal-shape A/B. Rationale: its 2 highest-EV pieces (moment-reset + transition-rewarmup) are already shipped as flags; its root-cure half is measured-likely-a-non-problem (nonlinear-ID ~9 ≪ linear-PR 26); it's mostly wall-clock not floor; the static schedule has an existence proof (monotone descent, no quench).

## THE SEQUENCE (wall-clock-optimal; each GPU step awaits operator GO)
1. **TIER-W θ* campaign** (~13 GPU-h, one GPU sequential) → rank surgical levers → keep winners.
2. **TIER-S `mod-dim {19,21,26}` small-n from-scratch sweep** (cheap) → pick the optimal mod.
3. **Bake winners (surgical levers + optimal mod) → n600 burn ONCE at optimal form** (~1000 ep).
4. **byte-close → dual CPU/CUDA exact eval → pointer.**
Root-tracking scheduler builds in parallel (CPU) as a follow-on.

## DECISION FOR OPERATOR
- **(A, recommended — your stated instinct, wall-clock-optimal):** GO the θ* campaign (TIER-W ~13 GPU-h + TIER-S mod-sweep) → ranks levers + resolves mod-dim BEFORE the n600 burn → the burn fires ONCE at optimal form.
- **(B):** fire the n600 NOW at the certified `mod-26` (skip θ*, get the floor row fast, refine later) — valid if you want a measured n600 exact row on the board immediately.

Recommend **A**. Either way the config is certified and ready; the only thing standing between us and a measured n600 row is your GO on the GPU.
