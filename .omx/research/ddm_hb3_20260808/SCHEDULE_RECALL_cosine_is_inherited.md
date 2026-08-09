# Cosine is INHERITED, not derived — and PR130 dropped the warmup its own ancestors carried

Operator steer 2026-08-09: *"cosine is pretty much never optimal, we have better than that"* +
*"pretty much every problem you run into we have a solution for or there's a solution in PR one
thirty or other PRs."* Both confirmed by recall, with one measured exception worth naming.
`score_claim=false` throughout.

## 1. PR130 does NOT have this solution (MEASURED — the exception)

| surface | finding |
|---|---|
| schedulers | `CosineAnnealingLR` in **8 of 8** trainers (hpac_self_compress, hpac_integer, semantic_quantized, semantic_full, pose_carrier_full ×2, refine_pose_coeff, learned_pose_carrier_oracle, semantic_renderer_oracle) |
| warmup | **NONE** anywhere except `--float-warmup-steps` in `train_semantic_quantized.py`, which is a QAT-phase switch, not an LR warmup |
| optimizer-state checkpointing | **ZERO** — `grep "optimizer.state_dict"` across the whole repro repo returns nothing |

So PR130's trainers are **fresh-start-only by construction**, and cosine-once is their uniform
inherited default. The one place they hand-rolled optimizer internals is
`train_pose_carrier_full.py`'s custom sparse Adam (explicit `betas`, `bias_correction2`).

## 2. The OTHER PRs DO have it (MEASURED)

`lr_scheduler.SequentialLR` + `warmup_sch` + `warmup_epochs` across the PR95-family intakes
(pr100 / pr101 / pr103 / pr105 / pr106 / pr81 / pr82): **296 occurrences** in the full intake,
6–12 per individual PR. That is `SequentialLR(warmup → cosine)`.

**PR130 dropped the warmup half that its own ancestors carried.** That is a real regression in
their lineage, and it is exactly the half that matters on a fresh/reset optimizer.

## 3. OURS is better than either, and it was already built

- **`ResumeLRWarmup`** (`witness_dsl/curriculum_dsl.py:5853`, task #518) — the warmup LENGTH is
  DERIVED, not guessed, through the registered law `adam_v_variance_warmup_length_v1`:
  `warmup_epochs = ceil(c/(1−β₂)/steps_per_epoch)` (RAdam variance-rectification, arXiv 1908.03265).
  Rationale in its own docstring: with fresh/reset AdamW moments the early steps are
  **quasi-isotropic (v≈0)**, so the ramp must SPAN the second-moment memory `1/(1−β₂)`.
- **`LrAnnealPin`** (`:2599`) — LR gets its OWN cosine denominator + optional hold, and states the
  law that matters here: *"a shallow shared-den cosine CANNOT reproduce a DEEPER LR descent by
  endpoint choice — the CURVATURE differs."*
- **`TailCycles`** (`:2619`) — cyclic/warm-restart tail (cycles_max, dwell_min, tau_halving).
- Plus `MuonLRGauge`, `TypedAnneal`, `WitnessNativeMorseContinuationSchedule`,
  `BirthPlateauKneeConjunct`, `PoseFinishBetaAnnealCoupling`.

## 4. What this diagnosed (MAIN's own two errors, both from not recalling first)

Resuming hb3's dead ep32 state into `train_hpac_self_compress.py`:

| arm | LR at resume | ep0 | ep1 | ep2 | later |
|---|---|---:|---:|---:|---|
| v1 (naive `--epochs 28`) | 0.003 (full restart, 2.2×) | 135,828 | 136,287 | 137,210 | killed — ASCENT |
| v2 (cosine-value matched 0.001376) | matched | 135,828 | 135,591 | **135,289** | 135,682 / 136,050 / 135,942 / 135,978 — WANDERS |

- v1 = the #518 class re-committed: full-LR restart from a converged state ascends.
- v2 = my "fix" matched the cosine's VALUE but not its CURVATURE (`LrAnnealPin`'s stated law), and
  more importantly **neither arm addresses the moment reset** — `--init` loads only `state_dict`,
  so AdamW's `exp_avg`/`exp_avg_sq` are zeroed. With v≈0 the update is `m/(√v+ε)`: sign-like and
  huge *regardless of nominal LR*. Matching the LR bought 2 descending epochs; then it wandered,
  because there is **no warmup in this trainer at all**.
- The derived warmup this needs: β₂=0.999, 600/8=75 steps/epoch → **c=1 → 14 epochs, c=2 → 27**.
  On a 28-epoch tail that is the whole run. **The moment reset, not the LR value, is the
  structural problem** — and the trainer is a READ-ONLY public-PR intake clone, so we cannot add
  the `SequentialLR` warmup its ancestors have.

**Therefore: don't resume this trainer. Run it clean.** Fired `clean60` (pid 34461): 60 epochs
from `hpac_p64_exact_from_archive.pt`, original `T_max=60` cosine, `qat_fraction 0.5`, ~44 min
local Metal. No reset, no curvature mismatch, no warmup debt.

## 5. Banked from v2 anyway (its `best` state_dict is saved)

`best_epoch 2`, bpp 0.0077987, top1_err 0.0019308:

| section | PR130 | ours | Δ |
|---|---:|---:|---:|
| **tokens** (61.26% of their archive) | 116,980 | **114,997** | **−1,983** |
| hpac model | 20,179 | 20,292 | +113 |
| joint | 137,159 | **135,289** | **−1,870** → ΔS **−0.0012452** |

**Our AR prior beats theirs on the token axis itself** — the largest single section of the archive.
Safe to state: `ADDENDUM2` measured rate to be provenance-insensitive to 0.047% (~65 B here),
so −1,983 B is ~30× the uncertainty.

## 6. NOT checked

- Whether a schedule better than cosine *would* help here — **untested**, because the flag surface
  offers no scheduler choice and the trainer is READ-ONLY. The available flags are `--lr`,
  `--lr-bits`, `--lr-exponent`, `--epochs`, `--qat-fraction`. Recording cosine as INHERITED is a
  provenance finding, not a measured inferiority claim on this vehicle.
- Why the original hb3 run died at ep32. Still undiagnosed.
- The joint figures are the trainer's own estimates (`bpp × 117,964,800 / 8` + packed model), not a
  packed archive. No score claim.
