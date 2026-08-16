---
arm: ddm_rg1b
title: "the band objective is BUILT and the rg1 probe is SEALED: the stock loss spends its gradient on the 1-px label band in exact proportion to AREA (2.16%, ratio 1.0016) while 99.22% of the seg debt lives there -- a 45.9x misallocation and an 83.3 degree direction error; the debt-density table is derived from the retained payloads with the pre-registered instrument control passing BIT-EXACT (2,551,464 band px); av3's F7 strengthens from 'short runs' to STRUCTURAL (the warmup ramp dominates at EVERY N because the crossover is 1.954*N by construction); four instrument fixes landed with tests"
utc: 2026-08-16
charter: "ddm_av3 NEXT_IF_RESUMED items #1 and #3 + ddm_rc2 section 2.1 + section 2.7 build order"
axis: "[macOS-CPU advisory] derivation from retained payloads -- NEVER a score. No training ran."
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "per-finding INSTANCE on the artifacts named in each row; the one FAMILY-scoped claim (EMA warmup dominance) is flagged as DERIVED with its closed form"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_rg1b — the band objective, built and sealed

**STORES CONSULTED (read at source, never from a summary):**
`.omx/research/ddm_rc2_regime_charter_and_lr_probe_20260816.md` §1, §2.1, §2.7 (commit `d7efa7128f`) ·
`.omx/research/ddm_av3_fresh_eyes_review_20260816.md` §F1b, F2, F3, F7, routing (commits `b345ff7562`,
`1c16945bf7`) · `.omx/research/ddm_rt1_seg_roundtrip_decomposition_20260816.md` §6.1–§6.4 ·
`.omx/research/ddm_ns1_negative_signal_audit_and_missing_patterns_20260816.md` §A ·
`src/tac/pr130_lift/train_semantic_quantized_resumable.py` · `lifted/semantic_renderer_oracle.py` ·
`src/tac/pr130_lift/editability_levers.py` · `tools/safe_run.py` · `experiments/ddm_rt1_*.py` ·
retained payloads `/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/` and
`/Volumes/APDataStore/pact/ddm_lr1/{C0,A1,A2,A3}/` (READ-ONLY; nothing written there).

---

## ANSWER FIRST

**The direction mismatch av3 inferred is now MEASURED, and it is bigger than the diagnosis implied.**
On real transmitted-label fields, the stock `curriculum_loss` puts **2.161%** of its gradient mass on
the 1-px label band. The band is **2.157%** of pixels. The ratio is **1.0016** — the stock objective
allocates its gradient to the band in **exact proportion to AREA**. But rt1 measured **99.22%** of the
seg debt sitting there. That is a **45.9× misallocation**, and the angle between the stock gradient and
the debt-weighted gradient is **83.3°** — nearly orthogonal, in all three curriculum phases. No learning
rate rotates a gradient 83°. av3's plateau law now has its mechanism.

**The band objective is built, and its pre-registered instrument control PASSES bit-exact.** The band
recomputed from the transmitted labels equals the retained `free_band_mask.npy` frame-for-frame across
all 600 frames, at **2,551,464** px — `RT1_GEOMETRY.json::ring_population[0]` exactly. The independently
recomputed round trip is **33,743** flips with **99.2176%** on-band, reproducing rt1's headline, and my
recomputed `pred→GT` confusion reproduces all 18 entries of `RT1_EDGESHAPE.json`. The instrument is rt1's.

**The weight table is debt DENSITY, and density ranks nothing like flip share.** Road↔Lane carries the
most flips (43.99% on the label basis) but is only the **6th densest** edge, because its boundary is the
longest (1,143,639 band px, 44.8% of the band). Movable↔MyCar is **10× denser** at 0.1316 flips/px. rc2
refused to substitute flip-share for exactly this reason; the measurement confirms the refusal was right.

**I found and fixed a units mismatch in my own first pass.** My first table keyed the numerator on the
CONFUSION pair `(pred, label)` and the denominator on the GEOMETRIC pair — two different objects in one
ratio — and double-counted the 264 off-band flips into on-band numerators. Both now use the same
incidence mask, so `W_e` is literally the flip rate on edge `e`, and `on_band + off_band == total` is
asserted at derivation time. The confusion table is retained as the cross-check that validates the
instrument against rt1, never as a weight numerator.

**av3's F7 is stronger than av3 recorded.** F7 reads "inert for any run shorter than ~1,167 steps." It is
STRUCTURAL. The LawRef derives `decay = f^(1/N)` from the run geometry, so the warmup crossover is
`t* = (10·decay − 1)/(1 − decay) ≈ 9N/ln(1/f) ≈ **1.954·N**` — a ratio independent of N. Verified at
N = 100 … 1,000,000: `t*/N` = 1.900 → 1.9543, dominance TRUE at every one. **At the canonical
`target_seed_fraction = 0.01` the resolved decay is never applied on any step of any run.** It would
govern only for `f < e⁻⁹ ≈ 1.234e-4`. The warmup ramp is the entire policy; the `derived_at_config`
value is decorative.

**Four instrument fixes landed with tests, and one blocker found for the F3 compose.** F3
(`--film-row-dropout`) as built **cannot honour the ns1 protection list** — see §5. That is named, not
worked around.

**Pointer UNMOVED.** No training ran, no archive was built, no score was produced. This unit is MEANS.

---

## §1 — The four instrument fixes (av3 F2/F3/F7, the owed 2-landings)

Each is fix + gate. Tests in `src/tac/pr130_lift/tests/test_av3_instrument_fixes.py` (24 tests).

| # | defect | fix | gate |
|---|---|---|---|
| **F2a** | `--save`/`--out` at an existing DIRECTORY crashes at the FINAL `os.replace`, after the full run | `parse_args` refuses in milliseconds | `test_directory_at_an_output_path_refuses_at_parse_time` (both flags); an existing FILE still allowed |
| **F2b** | `result.json` written AFTER the checkpoint, so the checkpoint's failure took the result with it | order inverted: cheap+irreplaceable JSON first | `test_result_json_is_written_before_the_final_checkpoint` (source-order gate, refuses re-inversion) |
| **F3** | headline is the argmin INCLUDING step 0 → a degrade-only run reports its INIT with `verdict: PASS` | `best_step`, `init_quantized_exact_seg`, `improved_over_init` added; `best_step`/`init_seg` threaded through checkpoint + resume | payload args kept **REQUIRED** (a default would silently reintroduce it); legacy-tolerant lazy restore |
| **F7** | LawRef decay recorded as governing a run it never governs | `warmup_dominates_target`, `governing_policy`, crossover, realised retention | dominance asserted STRUCTURAL at 5 run lengths + the closed-form threshold |
| **safe_run** | `status="ok"` on a crashed child (A2's receipt is literally `status ok, exit 1`) | additive `child_exit_nonzero` + `receipt_status_disagrees_with_exit` | two live subprocess tests (exit 1 and exit 0) |

**The exit-code passthrough contract is UNCHANGED** — `safe_run` still returns the child's code, and
`status` still carries its existing values, so every current consumer keeps working. The two new fields
are additive so a consumer that keys on `status` alone can see the disagreement.

**F2b, verified against the incident:** A2's `history` was NOT fully lost. It survives inside
`checkpoints.stage-expected_flip.step000600.full_state.pt` and I read the full 7-point trajectory from
it. What was lost is only the post-loop finalize product — `final_seg`, `ema_deployed_argmax_parity`,
`packed_parameter_bytes`, `verdict`. That narrows av3's F2 slightly and does not change its verdict.

**Resume safety.** `ema_policy` gained derived fields, and the resume path compares `ema_policy` by
equality. Those fields are excluded via `EMA_POLICY_DERIVED_OBSERVABILITY_KEYS`. This cannot weaken the
guard: every excluded key is a pure function of `decay` (still compared, inside `ema_policy`) and
`updates_per_run` (still compared, as `steps`, in the producing config). Asserted both ways —
old-checkpoint-into-new-reader reconciles, and a real decay change still refuses.

---

## §2 — The band objective (`src/tac/pr130_lift/band_objective.py`)

### §2.1 The instrument control, run before anything else

rc2 pre-registered it: the n600 band must reproduce `ring_population[0] = 2,551,464` exactly. **It does,
and better than "exactly" —** the recomputed boundary is **bit-equal to the retained
`free_band_mask.npy` frame-for-frame across all 600 frames.** The check is wired into the derivation
itself (`derive_table_from_payloads` refuses on the first mismatching frame), so the table cannot be
built from the wrong field.

**One measured discrepancy, reported rather than smoothed.** rt1's band is on the hv1 ep0634 transmitted
labels; the trainer's `target` comes from `gt_cache_600_official_ada.pt`. The two fields differ by
**72 px of 117,964,800 (6.10e-7)**, giving band populations 2,551,464 vs **2,551,446** (−18 px, 7e-6
relative). The table is derived on rt1's field, which is the one every rt1 number was measured on. The
in-loop band is recomputed per batch from whichever `target` the trainer is given, so the term follows
the trainer's own field by construction.

### §2.2 The rule, and what it is not

```
W_e   = flips_e / band_px_e     per unordered class pair, on the band
W_off = flips_off / px_off      the SAME density rule, off the band
```

One rule, applied uniformly: **weight is proportional to measured flip density**. Nothing is picked. The
off-band weight is not a floor invented to protect the interior — it is the same measurement.

Not a port of `_live_margin_weight`: that weights by margin MAGNITUDE on the MLX witness. This weights by
measured per-edge debt on a geometric band. Margin magnitude is a separate variable and does not ride
this term.

### §2.3 The measured table (`band_weight_table_rt1_n600.json`, sha `72658f02…`)

Machine-generated, never hand-typed, committed so a run needs no external volume:

| pair | flips (incident) | band_px | `W_e` | ×off-band | flip-share rank | density rank |
|---|---:|---:|---:|---:|---:|---:|
| Lane/Movable | 315 | 2,198 | 0.1433121 | 62,652 | 7 | **1** |
| Movable/MyCar | 35 | 266 | 0.1315789 | 57,523 | 9 | 2 |
| Lane/Undrivable | 314 | 2,981 | 0.1053338 | 46,049 | 8 | 3 |
| Lane/MyCar | 434 | 9,566 | 0.0453690 | 19,834 | 6 | 4 |
| Undrivable/Movable | 6,108 | 150,332 | 0.0406301 | 17,762 | 3 | 5 |
| Road/Movable | 4,617 | 142,296 | 0.0324464 | 14,185 | 4 | 6 |
| Road/Undrivable | 7,096 | 511,947 | 0.0138608 | 6,060 | 2 | 7 |
| **Road/Lane** | **14,917** | **1,143,639** | **0.0130435** | 5,702 | **1** | **8** |
| Road/MyCar | 1,965 | 606,287 | 0.0032410 | 1,417 | 5 | 9 |
| Undrivable/MyCar | 0 | **0** | 0.0000000 | — | 10 | — |
| OFF-BAND | 264 | 115,413,336 | 0.0000023 | 1 | — | — |

Cross-checks, all reproduced independently: `total_flips` **33,743** (rt1's round trip) ·
`on_band_flips` **33,479**, share **99.2176%** (rt1's 99.22% headline) · `on + off == total` asserted ·
`pred→GT` confusion reproduces all 18 `RT1_EDGESHAPE.json` entries, Road↔Lane **43.44%** on the GT basis
(rc2's cited figure) and **43.99%** on the label basis (the trainer's own metric).

**The rank inversion is the finding.** Road↔Lane is #1 by flips and #8 by density. Weighting by share
would have spent the budget on the longest boundary; weighting by density spends it where each pixel
actually flips.

Undrivable↔MyCar has **zero band pixels** — the two classes are never adjacent in n600. Its weight is
0.0 and it is unselectable at runtime; a test asserts that any zero weight has zero band pixels, so a
zero can never silently suppress a real edge.

### §2.4 Two DOFs, declared with measured blast radius

**Junction assignment.** A pixel on more than one edge takes the **max** of its incident densities.
Measured blast radius: junctions are **17,990 of 2,551,464 band px = 0.705%**, so max vs sum vs mean
moves under 1% of the band. Max is chosen because a pixel can only flip once — its risk is set by its
riskiest incident edge — and because it is bounded and tie-break free.

**Mixing, not switching.** The flag is a fraction:

```
w = (1 − α)·1  +  α·W/mean(W)
```

`mean(w) == 1` holds **algebraically for every α and every field**, because both terms already have mean
1. This matters more than it looks: ddm_lr1 just spent four arms measuring what learning rate does here,
and a term that rescaled the loss would rescale the effective lr and confound exactly that. Scale
neutrality is enforced twice — by the algebra above, and by reducing with `(w·l).sum()/w.sum()`, so no
weight field, normalised or not, can change the loss scale. α = 0 is an exact no-op; α = 1 is pure
measured density. Verified `mean(w) = 1.0000` on real fields at α ∈ {0, ¼, ½, ¾, 1}.

### §2.5 The controls that make the term admissible

1. **α = 0 / `weight=None` DELEGATES** to the lifted `curriculum_loss`. Bit-identity is structural, not
   numerical luck. Asserted with `torch.equal` in all three phases.
2. **Uniform-weight control**, exercising the reimplemented per-pixel path against the oracle. The
   residual is **MEASURED and printed**, not assumed: `ce` **6.045e-08** relative, `softplus_margin`
   **exactly 0**, `expected_flip` **exactly 0**. The CE residual is float32 reduction-order noise between
   `reduction="mean"` and `reduction="none"` + weighted mean.
3. `lifted/semantic_renderer_oracle.py` is **untouched** (sha `ffdf0988…`), so the trainer's phase-parity
   assertion at `:1063` still guards against curriculum drift.

38 tests in `test_band_objective.py`; 106 pass across the whole `pr130_lift` suite.

### §2.6 Where the weight mass actually goes (EXACT — real label geometry, no synthesis)

| α | band weight mass | band mean w | off-band mean w | max w |
|---:|---:|---:|---:|---:|
| 0.00 | 2.16% | 1.000 | 1.000 | 1.0 |
| 0.25 | 26.43% | 12.25 | 0.752 | 120.3 |
| 0.50 | 50.71% | 23.50 | 0.504 | 239.6 |
| 0.75 | 74.98% | 34.76 | 0.256 | 358.8 |
| **1.00** | **99.25%** | **46.01** | **0.00763** | 478.1 |

At α = 1 the weight mass on the band is **99.25%** and rt1's measured debt on the band is **99.22%**.
That correspondence is not a coincidence — weight ∝ flip density implies weight mass ∝ flip mass — but
it is a clean confirmation that the rule reproduces the measured debt distribution.

### §2.7 The direction change (logits SYNTHETIC — see the caveat)

Gradient w.r.t. the SegNet logits, on 8 real transmitted-label fields, synthetic logits calibrated to
rt1's measured margin scale (98.3% of flips need < 0.3 logits):

| phase | α | cos(stock, band) | angle | stock grad mass on band | band grad mass on band |
|---|---:|---:|---:|---:|---:|
| ce (step 0) | 1.0 | 0.1165 | **83.31°** | 2.157% | 99.25% |
| softplus_margin (400) | 1.0 | 0.1166 | **83.30°** | 2.161% | 99.25% |
| expected_flip (560) | 1.0 | 0.1168 | **83.29°** | 2.166% | 99.26% |
| softplus_margin (400) | 0.5 | 0.2287 | 76.78° | 2.161% | 50.76% |

**The headline number:** stock grad mass on band **2.161%** ÷ band area fraction **2.157%** = **1.0016**.
The stock objective is *exactly* area-proportional. Against a debt that is 99.2176% on-band, that is a
**45.9× misallocation**. This is the same statement as av3's "the trainer's direction is not the metric's
direction," now with a number and a mechanism.

⚠ **Scope.** The logits are synthetic, so 83.3° bounds the direction change at a representative
operating point — it is not the trained model's own gradient. The weight-mass and area figures use only
real label geometry and are exact. *verdict_scope: INSTANCE (8 real n600 label fields, synthetic logits
at rt1's margin scale).*

---

## §3 — The re-derived plateau law and the pre-registered bar

I **re-derived** av3's F1b from the retained checkpoints rather than quoting its table. Receipt:
`.omx/research/ddm_rg1b_lr1_refit_and_bar_20260816.json`.

| arm | lr | ‖Δw‖@100 | ‖Δw‖@600 | peak Δpx | end Δpx | best_step | improved? |
|---|---:|---:|---:|---:|---:|---:|---|
| C0 | 2e-7 | 0.0014435 | 0.0014451 | 5,879 | 2,164 | **0** | **no** |
| A1 | 2e-6 | 0.0095209 | 0.0087717 | 14,960 | 3,828 | **0** | **no** |
| A2 | 2e-5 | 0.0473997 | 0.0504533 | 27,170 | 8,049 | **0** | **no** |
| A3 | 2e-4 | 0.3614357 | 0.5349090 | 76,594 | 59,357 | **0** | **no** |

Reproduced exactly: exponent **0.457640** (av3: 0.458), **R² = 0.996948** (av3: 0.9969),
`‖Δw‖₁₀₀ ∝ lr^0.7893` R² 0.9982, `end ∝ ‖Δw‖₆₀₀^0.5573` R² 0.9460. **Which horizon the law uses was
ambiguous in the memo and is now pinned: the PEAK fit uses ‖Δw‖ at step 100, the END fit uses step 600.**
All four arms confirm `best_step = 0` and `improved_over_init = false` — F3's finding, on real data, and
the validation that the new fields report it correctly.

**Fit:** `peak_flips = 118,563.2 · ‖Δw‖₁₀₀^0.457640`, σ_log = 0.072827, dof = 2.

**BAR — BREAK THE EXPONENT (derived from the fit's own residual scatter; nothing picked).** At the arm's
own measured `d = ‖Δw‖₁₀₀`, the fit predicts `P(d)`. The arm BREAKS the law iff

> `peak_dpx_measured  <  P(d) / K(d)`,  `K(d) = exp(t₀.₉₉,₂ · SE_pred(d))`,
> `SE_pred(d) = σ·√(1 + 1/n + (ln d − x̄)²/Sxx)`

At the matched operating point (A2's `d = 0.047400`, since the rg1 arm is fired at the same lr):
`P = 29,372`, `SE_pred = 0.08257`, `K₉₉ = 1.7772` → **bar = 16,527 peak Δpx**. A2 measured 27,170, so
this asks for a **1.64× reduction in peak flips at the same weight displacement**. The 95% band
(`K₉₅ = 1.2727` → 23,079) is recorded as SUGGESTIVE so a partial result is neither promoted nor binned.

⚠ **Instrument capacity, stated plainly (na2's law).** n = 4, dof = 2, so t₀.₉₉ = 6.96 and the interval
is WIDE. **This bar detects a large direction change only.** A 20% improvement sits inside the noise of a
four-point fit and would correctly read as NOT BROKEN. A negative on this bar falsifies "large effect at
600 steps," never the family.

**BAR — DESCEND.** `min(history.quantized_exact_seg) < 0.00028616163465711804` (the init, identical in
all four arms), read via the new `best_step` / `improved_over_init` fields. Metric granularity is
1/117,964,800, so one pixel is resolvable.

---

## §4 — THE SEALED rg1 PROBE TICKET (MAIN fires; this arm does not)

**ONE bounded window. ONE variable: `--band-objective-weight`.** Everything else is byte-identical to
ddm_lr1/A2, which is therefore the matched control — same init, cache, seed, steps, lr, curriculum,
`--weight-qat-q3q4`, device.

```bash
.venv/bin/python tools/safe_run.py \
  --rss-mb 12288 --timeout 4200 \
  --label ddm_rg1_band_a1 \
  --status-receipt /Volumes/APDataStore/pact/ddm_rg1/band_a1/safe_run_status.json \
  --child-pidfile  /Volumes/APDataStore/pact/ddm_rg1/band_a1/child.pid \
  -- \
  .venv/bin/python -m tac.pr130_lift.train_semantic_quantized_resumable \
  --challenge-root upstream \
  --cache /Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt \
  --init  /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/\
checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt \
  --bits 4 \
  --weight-qat-q3q4 \
  --steps 600 --lr 2e-5 --float-warmup-steps 0 \
  --ce-fraction 0.50 --softplus-fraction 0.85 \
  --eval-every 100 --checkpoint-every 100 \
  --device mps --seed 20260715 \
  --band-objective-weight 1.0 \
  --out  /Volumes/APDataStore/pact/ddm_rg1/band_a1/result.json \
  --save /Volumes/APDataStore/pact/ddm_rg1/band_a1/ckpt
```

`--save …/ckpt` (a file basename, per av3's correction). The new argparse guard now refuses the A2 shape
outright, so this cannot repeat even if mistyped.

**Why α = 1.0, derived.** Because `mean(w) == 1` is algebraic, α does **not** trade off against
measurement cleanliness — the effective lr is neutral at every α. So the only question is mechanism
risk: at α = 1 the interior carries 0.0076× weight and could in principle drift. Two measured reasons it
cannot drift far in this window: (a) A2 moved ‖Δw‖ by 0.0505 against ‖init‖ = 74.4 — **0.068%** of the
weight norm in 600 steps; (b) SegNet reads REGIONS, so interior drift large enough to matter would raise
`quantized_exact_seg`, which is the measured quantity. The metric is its own detector. α = 0.5 is the
pre-declared fallback if the receipt shows interior damage.

**Pins — re-hash at fire time, do not trust these strings as current:** `--init` sha256
`3948ccfc…` · `--cache` sha256 `382d7dfe…` · `upstream/modules.py` `065961ba…` · band table
`72658f02012c640c75da669318eae095b1b3a4b36072b5beae6784b3651bbf8f` (recorded automatically into
`result.json.config.band_weight_table_sha256`, and into `band_objective.table_sha256`) ·
`train_semantic_quantized_resumable.py` `d104139d…` · `band_objective.py` `81e187f6…`. **The trainer
file changed in THIS unit — re-hash it, never cite a pre-08-16 pin.**

**The two pre-registered bars.**

1. **BREAK THE EXPONENT** — measured `peak_dpx < 16,527` at the matched displacement, or more precisely
   `< P(d)/1.7772` evaluated at the arm's own `‖Δw‖₁₀₀` (recompute `SE_pred` if `d` lands far from
   A2's). SUGGESTIVE band: `< 23,079`.
2. **DESCEND** — `improved_over_init == true`, i.e. history-min `< 0.00028616163465711804`.

**Reading the receipt (av3 F3/F2 now make this safe).** `result.json` carries `best_step`,
`improved_over_init`, `init_quantized_exact_seg` and the full `history`, and is written BEFORE the
checkpoint. `band_objective.activations` must be **600**; the run raises if the term was requested but
never fired, so a held-but-inert term cannot report as active.

**Budget.** ~600 × 3.33 s ≈ 33 min, plus the per-batch band weight (two shifted comparisons and a
lookup on a 384×512 field — negligible against a SegNet forward+backward). One Metal fire, governed.
**Resumability:** unchanged, 6 stage checkpoints; `best_step`/`init_seg` now round-trip through resume.
**Payload:** `/Volumes/APDataStore/pact/ddm_rg1/band_a1/` — every checkpoint, `result.json` with the
full history, the safe_run receipt, the launcher tree. The trajectory IS the measurement.

**What this probe CANNOT conclude:** nothing about the contest score (it builds no archive); nothing
about editability (F3 is not in this window, by design); and a NO bounds "large effect at 600 steps from
this init," never the family — see the instrument-capacity caveat in §3.

---

## §5 — The F3 compose note, and a BLOCKER

rc2 §2 fires the band objective and F3 (`--film-row-dropout`) together in the full rg1 burn. Two things.

**The probe in §4 is band-ONLY, deliberately.** Attributing an exponent break to DIRECTION requires one
variable. Firing F3 in the same window would confound it. F3 rides the full burn, after the probe reads.

**BLOCKER — F3 as built cannot honour the ns1 protection list.** ns1 §A located the pose-critical
subspace as `blocks_1` FiLM rows and its constructive product is "exclude blocks_1 FiLM from ANY future
perturbation." But F3 applies to `FILM_ROW_FAMILY = {blocks.1, blocks.2, blocks.3}.film.weight`
(`editability_levers.py:101`) and its only protection knob, `--film-row-dropout-protect-top N`, protects
the top-N highest-norm rows **within each tensor** (`_row_dropout:425`). All three tensors are
**(192, 8)** — measured from the init checkpoint. So protecting all of `blocks.1` needs `protect-top 192`,
which also fully protects `blocks.2` and `blocks.3` and makes F3 **entirely inert**. There is no setting
that both drops rows and honours the list.

**The cure is one narrow change, and it is OWED, not done here** (it changes lever semantics, so it needs
its own charter and review): a tensor-name exclusion — either a `--film-row-dropout-exclude-tensors`
option defaulting to the ns1 list, or restricting F3's family to `blocks.{2,3}` — so F3 operates on the
protected list's COMPLEMENT as the binding requires. Until that lands, **F3 must not fire**, because the
composition would perturb the exact rows ns1 measured at ~94× sensitivity.

---

## OPTIMAL FORM

**Reference form.** The family is per-pixel/spatially-weighted segmentation loss (boundary-weighted CE,
distance-transform weighting, hard-example mining). Reference: weight the loss by a measured per-region
error statistic and reduce with a weighted mean. This build is at reference form for the mechanism: real
per-edge densities from n600 receipts, exact per-pixel reduction, both controls green.

**Deltas, each declared.**

| delta | class | note |
|---|---|---|
| 600 steps, one α, one init | **SCOPE** | legal; the matched control is A2 at the identical scope |
| n600 real fields, real receipts, all 5 classes, all 10 pairs | none | full population; no prefix, no subset ([[m88]]/[[m96]]) |
| junction = max (not sum/mean) | **DOF, declared** | blast radius MEASURED at 0.705% of band px |
| logits synthetic in the §2.7 direction measurement only | **SCOPE, labelled** | weight-mass and area figures are exact; the 83.3° is bounded, not the trained gradient |
| `pred_vs_label` numerator basis | none | matches the trainer's own metric; `pred_vs_gt` retained as cross-check, tables agree in rank order and within 16% worst-case |
| F3 excluded from the probe | **SCOPE** | one variable; F3 additionally BLOCKED per §5 |

**No MECHANISM reduction. No TOY-BRACKET declared, and none needed** — every input is the real n600
payload, the band operator is rt1's verbatim, and the instrument control is bit-exact.

**Provenance pins.** rc2 `d7efa7128f` · av3 `b345ff7562`, `1c16945bf7` · `RT1_EDGESHAPE.json`
`3def0d22…` · `free_band_mask.npy` `649dd26f…` · transmitted labels `9ba2e52b…` · `argmax_base.npy`
`2aeb1e6b…` · trainer (post-edit) `d104139d…` · `band_objective.py` `81e187f6…` · table `72658f02…` ·
`safe_run.py` `59fb3d4f…` · oracle UNTOUCHED `ffdf0988…`.

---

## What this unit did NOT establish

- **No training ran.** Not one gradient step of the real model. Every number is derivation from retained
  payloads or a unit-test-scale forward/backward.
- **No score.** No archive, no `evaluate.py`, no pointer movement. The own-vehicle frontier is UNMOVED at
  hv1 ep0634 S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]`.
- **No evidence the band objective descends.** The direction change is measured; whether rotating the
  direction 83° *lowers* the metric is exactly what the sealed probe asks, and it is unanswered.
- **The 83.3° uses synthetic logits.** Bounded at a representative operating point, not the trained
  model's own gradient. The trained-gradient version needs a real forward pass and was not run.
- **The junction max-rule is not ablated.** Its blast radius is measured (0.705%); the alternative was
  not run.
- **No sweep of `safe_run` status-only consumers.** av3 checked one and could not verify the rest; I did
  not extend that sweep. The new fields are additive, so nothing breaks either way.
- **F7's structural generalisation is DERIVED, not measured on runs.** The closed form and the N-sweep
  are arithmetic on the LawRef; no run was fired at a long horizon to confirm the realised retention.
- **The 72-px label-field difference is unexplained.** Measured and reported; I did not trace its cause.

---

## NEXT_IF_RESUMED

1. **MAIN fires §4** (one Metal fire, governed). Read `improved_over_init` and the `(peak_dpx, ‖Δw‖₁₀₀)`
   point against the §3 bar. Everything needed is sealed; nothing else is owed first.
2. **If BROKEN but not DESCENDED:** the direction is right and the budget is wrong — the natural next
   variable is α (0.5) or steps, not lr (the ladder's R² 0.997 already bounds what a fifth lr shows).
3. **If NEITHER:** report it plainly against the instrument-capacity caveat — "no large effect at 600
   steps from this init," scoped INSTANCE — and route to av3's W1/R1/N0 discriminators, which are still
   unrun and cost one flag each.
4. **F3 cure (owed, §5):** a tensor-name exclusion so F3 operates on the ns1 protected list's
   complement. Blocks the rg1 full burn's editability half; does not block §4.
5. **av3's remaining unrun items:** W1 (`--float-warmup-steps > 0`), R1 (`--resume-from` restores AdamW
   moments), N0 (`--lr 0`). Each ~380 s. An arm landing OFF the R² 0.997 curve is the informative one.
6. **Consider raising F7 to the LawRef itself.** If the target decay can never govern at
   `f = 0.01`, the honest options are to change `EMA_TARGET_SEED_FRACTION` below `e⁻⁹`, to change the
   warmup ramp, or to record the ramp as the policy. Currently the manifest just tells the truth about
   being decorative. Not this arm's call.
