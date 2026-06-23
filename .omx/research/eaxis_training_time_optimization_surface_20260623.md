# E-AXIS (training-time) response surface — d_seg(epochs, config), convergence models, fast-convergence recipe

**Date:** 2026-06-23 · **Subagent:** eaxis-trainingtime-20260623 · **Authority:** `[advisory]` / `[contest-CPU advisory]` **NON-PROMOTABLE** (mined measurements from trajectory logs + contest-math recompute; NOT new exact evals; $0, CPU-only, no MPS contention with the live bc36 run pid 19241). Frontier pointer UNMOVED 0.19110.

**Mission framing:** the operator named *training-time* as an under-exploited optimization axis. This memo delivers (1) the mined `d_seg(epochs, config)` dataset, (2) fitted convergence models, (3) the **fast-convergence recipe** (reach the d_seg basin in dramatically fewer epochs — both a race/velocity edge AND what makes the whole config search affordable), (4) the training-time→score Pareto + the min-training-time-to-sub-0.19/0.15 estimate (labelled extrapolation). Feeds the math-optimal solver (sister agent).

---

## 0. Cross-check discipline applied (existence-proof, per `feedback_terminal_conclusion_needs_existence_proof_crosscheck`)

Every "floor/plateau/asymptote" below is cross-checked against the **PR95 existence proof: d_seg = 5.6e-4 reached in 29,650 epochs**. **No fitted asymptote in our data is at or below 5.6e-4** → therefore **every asymptote here is a recipe/capacity artifact, NOT a physical d_seg floor.** Short-run final d_seg is never treated as a floor; the analysis is convergence-RATE-based with explicit extrapolation tags. The d_seg axis is NOT exhausted — our best recipe asymptotes at 2.06e-3, which is **3.7× above** the known-achievable PR95 basin.

---

## 1. The dataset (mined, $0)

- Tool: `tools/build_convergence_dataset.py` (read-only miner, no training, no GPU/MPS).
- Output: `.omx/research/eaxis_convergence_dataset_20260623.json` — **60 runs, 1238 d_seg eval points.**
- Two trajectory schemas unified: `torch_vehicle_trajectory.jsonl` (eval rows: `d_seg`,`d_pose`,`score`,`global_epoch`,`wall_clock_s`,`stage_name`,`rate`,`archive_bytes`,`muon_lr`/`adamw_lr`) + capstone `trajectory.jsonl` (`exact_d_seg`,`mean_d_pose`,`global_epoch`,`elapsed_s`,`base_channels`).
- Per-run config pulled from `torch_vehicle_summary.json` (run_meta) + `PROVENANCE.json` (levers, devices, command, n_pairs, base_channels). Score column recomputed via `tac.contest_score.compute_contest_score` for internal consistency.

**Longest / highest-value curves:**

| run | bc | n_pairs | epoch span | #pts | min d_seg | notes |
|---|---|---|---|---|---|---|
| yousfi_r3_taper_marginhinge_e5 | 20 | 600 | 25→15650 | 218 | 2.061e-3 | leading n=600 recipe (margin-hinge + taper, 8-stage) |
| distortion_arm_l235 | 20 | 600 | 450→5525 | 203 | 2.466e-3 | levers 2/3/5 (soft_cosine+FiLM+margin) |
| yousfi_r2_arm_a_marginhinge | 20 | 600 | 25→4300 | 84 | 2.202e-3 | margin-hinge, no taper |
| bindall_arm_b_canonical50k(_mh) | 20 | 600 | 50→5709 | 53–62 | 2.26e-3 | canonical 50k curriculum |
| torch_vehicle_full_mps_basin | 20 | 600 | 10→2325 | 32 | 2.561e-3 | plain-CE basin (control) |
| oomph_scaled_n96/scaled_sw1.5 | 20 | 96 | 10→210 | 21 | 2.706e-3 | n=96 oomph |
| capstone capacity 2×2 (bc20/bc24) | 20/24 | (p48/p192) | 10→120 | 12 | 2.83e-3 (bc24) | capacity micro-ablation |

---

## 2. Convergence models

- Tool: `tools/fit_convergence_models.py`. Output: `.omx/research/eaxis_convergence_models_20260623.json` (60 fits, 13 config groups).
- Two laws fit per run: **power-law-to-floor** `d_seg(E) = floor + c·E^(−p)` and **exponential-to-floor** `d_seg(E) = floor + amp·exp(−E/τ)`; best-R² selected.

**Headline fit — the leading n=600 recipe (yousfi_r3_taper, 218 pts):**
```
d_seg(E) = 2.06e-3 + c·E^(−0.83)      [power-law, R²=0.981]
```
Power-law wins on every long n=600 run (p ≈ 0.34–1.0). **Power-law = "slow but never flat"** — the curve keeps descending, just with diminishing returns; it is NOT an exponential hitting a hard wall. This is consistent with the prior "power-law-slow not a wall" finding and the PR95 existence proof (PR95 simply paid the 29,650-epoch power-law tail).

**Stage decomposition (yousfi_r3_taper) — where the d_seg drop actually happens:**

| stage entered | @epoch | d_seg | Δ from prev |
|---|---|---|---|
| stage1 CE | 25 | 1.67e-2 | (from 0.5) |
| stage2 softplus | 3025 | 2.33e-3 | **−86% (the basin is made here)** |
| stage3 smooth | 8700 | 2.13e-3 | −9% |
| stage4 QAT | 10175 | 2.09e-3 | −2% |
| stage5 C1a | 10700 | 2.08e-3 | −0.5% |
| final | 15650 | 2.08e-3 | ~0 |

**→ Stage1 CE + early stage2 reach the basin; stages 3–8 are diminishing-returns polish (≈10% over 12,000 epochs).** This is the single most actionable structural fact for the E-axis.

**MUONJUMP stage8 (muon-only finetune, 18 pts ep24675→25700): d_seg FLAT** (2.084e-3 → 2.111e-3; d_pose even rises). **Muon finetune is a pose/quantization-stability tool, NOT a d_seg lever.** Negative result — do not spend epochs on muon-finetune expecting d_seg gains.

---

## 3. THE FAST-CONVERGENCE RECIPE (the innovative result)

**Goal: reach the d_seg basin (≤0.005, then ≤0.003) in the FEWEST epochs.** Measured epochs-to-basin across configs:

| config | ep→d_seg≤0.005 | ep→d_seg≤0.003 | mechanism |
|---|---|---|---|
| **warm-start / transfer-init** (g0_600_transfer, longtrack_a) | **~0** (starts at 2.3e-3) | **~0** | resumes from a prior basin checkpoint at stage8; basin is the *initialization*, not earned |
| n=96 oomph (margin/scaled) | **10** | **80–130** | fewer pairs → far fewer steps-to-basin (memorization regime) |
| bc24 (capstone p48) | 70 | 110 | **more capacity converges faster AND lower** (bc24 2.83e-3 < bc20 3.73e-3 @ep120) |
| yousfi_r2 margin-hinge n=600 | 100 | 500 | margin-hinge accelerates vs plain CE |
| yousfi_r3 taper n=600 | 175 | 775 | taper (best asymptote, slightly slower entry) |
| plain-CE basin n=600 (control) | 240 | 1040 | baseline from-scratch |

### The recipe (ranked levers, each measured):

1. **Warm-start from an existing basin checkpoint** — the dominant lever. A from-scratch n=600 run needs **~240 epochs to ≤0.005 and ~1040 to ≤0.003**; a transfer-init run **starts already at 2.3e-3**. For config search (sweeping levers/taper/capacity), fork from a saved basin and run only the differential stages → **~10–100× fewer epochs per candidate.** This is what makes the whole config search affordable.
2. **margin-hinge seg loss > plain CE** for basin-entry speed: ~100 ep vs ~240 ep to ≤0.005 at n=600 (2.4× faster), and a deeper asymptote (2.06e-3 vs 2.56e-3).
3. **More capacity converges faster, not slower** (bc24 < bc20 in both speed and depth at matched epoch) — refutes the "capacity slows convergence" intuition; consistent with the corrected MEMORY power-law refutation. (bc28/bc36: live run pid 19241 + g1_capacity_rd/bc28 have **no evaluated d_seg yet** — queued.)
4. **Don't spend epochs on stage3–8 or muon-finetune for d_seg** — the basin is made in stage1-CE→stage2; later stages are pose/quant polish with ~flat d_seg. For a d_seg-only velocity target, a **2-stage CE→softplus schedule reaches ≈95% of the asymptote** at a fraction of the 29,650-epoch budget.
5. **n=96 for fast lever-screening only** — reaches basin in 10 ep, ideal for $0 lever A/B; but it is a memorization proxy (must re-confirm winning levers at n=600 before any score claim).

**Fast-convergence config-search loop (the affordability recipe):** save one basin checkpoint → fork-init every candidate from it → run the differential stage(s) only → screen at n=96 (10-ep basin) → confirm top-k at n=600 from warm-start (10–100 differential epochs). Turns a ~30,000-epoch-per-candidate cost into ~10–800 epochs-per-candidate.

---

## 4. Training-time → score Pareto + min-training-time to sub-0.19 / sub-0.15

Score isolated on the E-axis using the canonical **bc20 small-basis rate+pose floor S_floor = 0.1178** (rate 0.0594 + pose 0.0585, MEMORY anchor; byte-closed) so `S(E) ≈ 100·d_seg(E) + 0.1178`. (The trajectory logs' `archive_bytes` are latent-only/partial, rate≈0.002 — NOT byte-closed; the 0.1178 floor is the proper rate+pose anchor.)

Projecting the leading recipe's power-law `d_seg(E)=2.06e-3 + c·E^(−0.83)`:

| epochs | d_seg | S ≈ 100·d_seg + 0.1178 | tag |
|---|---|---|---|
| 100 | 6.6e-3 | 0.783 | in-range |
| 500 | 3.3e-3 | 0.444 | in-range |
| 1,000 | 2.74e-3 | 0.392 | in-range |
| 3,000 | 2.33e-3 | 0.351 | in-range |
| 10,000 | 2.16e-3 | 0.334 | in-range |
| 29,650 | 2.10e-3 | 0.328 | EXTRAPOLATED |
| 120,000 | 2.07e-3 | 0.325 | EXTRAPOLATED |

**The Pareto knee is at ~1,000–3,000 epochs** (S ≈ 0.39→0.35). Beyond ~3,000 epochs the curve is essentially flat at S ≈ 0.33 for *this recipe* — i.e. **more training time on this config does NOT reach sub-0.19.**

**Min-training-time to thresholds (this recipe, EXTRAPOLATED):**
- **S < 0.19** needs d_seg ≤ **7.2e-4** — **BELOW this recipe's asymptote (2.06e-3) → UNREACHABLE by epochs alone on this config.**
- **S < 0.15** needs d_seg ≤ **3.2e-4** — **also below asymptote → UNREACHABLE by epochs alone on this config.**

**→ The decisive E-axis verdict:** *epochs alone, on the current bc20 margin-hinge recipe, cannot reach sub-0.19/0.15.* The blocker is the **recipe asymptote (2.06e-3), which the cross-check proves is NOT the physical floor** (PR95 = 5.6e-4). Reaching sub-0.19/0.15 therefore requires changing the asymptote-setting levers (capacity bc≥24/28/36, KD-from-frontier-teacher init, taper realloc, FiLM-pose decoupling) — NOT more epochs on the current config. Training-time's role is **velocity/affordability** (fast-convergence recipe §3), which makes the *asymptote-lowering config search* cheap enough to run.

**What WOULD reach it (for the sister solver):** if a config lowers the asymptote to PR95's 5.6e-4 (capacity + warm-start + taper), the *same power-law shape* implies the basin is reached in ~stage1-CE→stage2 epochs from a warm start, then the deep tail is the diminishing-returns polish PR95 paid 29,650 epochs for. The min training-time is then dominated by the from-scratch basin-entry (~1–3k epochs once) + cheap per-candidate differential epochs.

---

## 5. Reactivation criteria

1. **bc28/bc36 capacity curves**: the live bc36 run (pid 19241) + g1_capacity_rd/bc28 have NO evaluated d_seg yet. Re-run `tools/build_convergence_dataset.py` once they produce eval rows → re-fit → confirm whether higher capacity lowers the asymptote toward 5.6e-4 (the existence proof). This is the single highest-value missing curve.
2. **KD-from-frontier-teacher init**: no run in the dataset warm-starts from the 0.191 frontier (PR101-class) decoder as a d_seg teacher. A KD-warm-start probe (queued — needs a gradient device; do NOT contend for MPS) would test whether teacher-init reaches a deeper asymptote faster.
3. **2-stage truncated curriculum probe**: CE→softplus only, n=600, warm-started — measure whether it reaches ≈95% of the full-curriculum asymptote (predicted yes from §2 stage decomposition) at a fraction of the budget. $0-ish if warm-started on CPU; else queue for GPU slot.
4. Refresh both JSONs + this memo whenever a new long n=600 or capacity run lands.

---

## 6. Six-hook wire-in (per Catalog #125)

1. **Sensitivity-map**: ACTIVE — `d_seg/d_epoch` per stage is the training-time sensitivity (stage1-CE→stage2 carries ~95% of the drop; stages 3-8 + muon ≈ 0). Feeds the planner's stop/continue rule.
2. **Pareto constraint**: ACTIVE — training-time→score Pareto (§4); knee ≈ 1–3k epochs; the recipe-asymptote constraint (epochs cannot cross sub-0.19 on bc20-marginhinge) is a binding Pareto fact for the solver.
3. **Bit-allocator hook**: N/A — E-axis is training-time, not byte allocation (sister rate-axis agent owns bytes). Declared N/A.
4. **Cathedral autopilot dispatch**: ACTIVE-as-prior — the fast-convergence recipe (warm-start + n=96 screen + n=600 confirm) is a dispatch-cost model the ranker can consume to price candidates by epochs-to-basin.
5. **Continual-learning posterior**: ACTIVE — the convergence fits (asymptote, p/τ, epochs-to-basin per config) are reusable priors; refreshed by re-running the two tools as runs land.
6. **Probe-disambiguator**: ACTIVE — reactivation probes §5 (bc28/36 capacity, KD-teacher-init, 2-stage truncation) are the disambiguators between "epochs-bound" vs "asymptote/config-bound" interpretations. The data already disambiguates: **asymptote-bound, not epoch-bound** (cross-check).

---

## 7. Tooling delivered

- `tools/build_convergence_dataset.py` — mines trajectory logs → unified `d_seg(epochs,config)` JSON. Reviewed, $0, read-only.
- `tools/fit_convergence_models.py` — fits power-law/exp convergence models, extracts asymptote / convergence-rate / epochs-to-target (measured + extrapolated, separated) / config-group lever effects. Reviewed.
- `.omx/research/eaxis_convergence_dataset_20260623.json` (60 runs, 1238 pts).
- `.omx/research/eaxis_convergence_models_20260623.json` (60 fits, 13 config groups).

All advisory / NON-PROMOTABLE; no frontier claim; pointer UNMOVED 0.19110.
