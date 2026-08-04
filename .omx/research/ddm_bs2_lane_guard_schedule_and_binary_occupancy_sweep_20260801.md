---
schema: ddm_bs2_lane_guard_schedule_and_binary_sweep.v1
date_utc: 2026-08-01
arm: ddm_bs2 (task #871 — lane-guard budget SCHEDULE P0 + discrete/binary occupancy sweep)
lane_id: "lane_ddm_bs2_20260801"
research_only: true
score_claim: false
promotion_eligible: false
axis: "[macOS-CPU advisory — telemetry re-read, byte-exact coder race, module build + tests. NO training, NO scorer job, NO paid dispatch, NO pointer mutation]"
consumes:
  - /Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_0{1,2,3}/telemetry.jsonl  (64 lane_guard + 64 a1_gate rows — the primary data)
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_pw1_archive.zip     (the live composed archive)
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/pw1/final_pw1.jsonl              (600 per-pair rows)
  - /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/d1_warp_solve.partial.jsonl  (600 s_t_idx rows)
  - src/tac/optimization/lane_guard.py (ddm_lg1 #808), .omx/research/ddm_b4s_guard_audit_20260801.md
consumers: [MAIN, any burn-5 guard decision, ddm_tw1 (#869), the #822 owner]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_bs2 — the Lane budget SCHEDULE, and what the occupancy discriminator says about the rest of the binaries

## §0 POINTER HONESTY

**The exact frontier did NOT move.** This unit ran ZERO training and ZERO scorer jobs. Every number is
`[macOS-CPU advisory]`, `score_claim=false`.

**BASELINE REPRICED (a ΔS without its baseline is unanchored, and baselines move).** My charter priced
against `S_current = 0.9639878`, gap 0.7918468, 1% = 11,892 B. The live own-vehicle best moved while this
arm ran: **S = 0.7910689 @ 353,805 B `[macOS-CPU advisory]`, gap-to-bar 0.6189279, 1% of gap = 0.0061893 S
= 9,296 B.** Every percentage below is against the CURRENT gap. Note the byte measurements in §5.2 were
taken on the **v4d/pw1 composed archive** (360,238 B total, 346,478 B tokens) — the current best is a
different, smaller archive whose token section I did not re-measure, so the ΔB figures are exact for the
stream measured and indicative, not exact, for the current best.

## §1 TWO SEED FACTS WERE STALE — corrected at source before anything was built

| seed claim | status |
|---|---|
| "#822's own title records that lambda=0 was CORRECT KKT / budget never tightens" | **WRONG.** #822's real title is *"lane-guard sign disagreement: `realized_lane_s_units` vs `net_betti0_realized_lane_delta` are different quantities; TEST whether the budget metric is self-defeating"*. A different defect entirely. Tested in §2. |
| "Task row #871 (your charter)" | **#871 HAS NO ROW** in `.omx/state/canonical_task_status.jsonl` (checked all rows; 808/822/824 present, 869/871 absent). The specification-without-registration class ddm_rg5 closed at #825, recurring. Registered by this arm. |
| charter sweep seeds: self-orient · margin_weighted_loss · v19/v19b greedy accept | **3 of 4 are NOT on the live chain.** `self_orient` appears only in `experiments/measure_*`/`probe_*`/`rl_annulus_unlock_gate.py`/`tac.boundary_math.lever_b_generator`; `margin_weighted*` only in trainers + `spec_tr1_renderer`; `v19`/`v19b` only in `tac.ddm_campaign_*`/`ddm_ms2r_*`/`ddm_ws4_*`. None is imported transitively by any of the four live-chain files. Only `token_ste` is live. |

## §2 #822 TESTED AND REFUTED — the budget metric is sound (so the ratchet is safe to build on)

Before tightening a constraint I had to check the constraint measures the right thing. Over the 64 paired
`lane_guard` + `a1_gate` rows:

| correlation | value | expected sign | verdict |
|---|---|---|---|
| `realized_lane_s` vs `gt_components_erased[Lane]` | **+0.9697** | positive (both COSTS) | ✅ |
| `realized_lane_s` vs `betti0_realized[Lane]` | −0.7796 | negative (cost vs benefit) | ✅ |
| `betti0_realized` vs `gt_components_erased` | −0.8857 | negative | ✅ |

**#822's hypothesis is REFUTED at r = +0.97 (verdict_scope: INSTANCE — this burn, 64 gates, one run).**
The pixel-recall budget metric and the GT-referenced topological Lane cost move together almost perfectly.
The dual's metric is not self-defeating.

The real defect the sign question was pointing at is one level over: `betti0_realized` ≠ `betti0_gt −
erased` (mean gap **43.28 components = 8.2%**), i.e. `betti0_realized` counts ~43 Lane components with no
surviving GT parent. The **topology ALARM** consumes that contaminated quantity; the **dual** consumes the
clean GT-referenced one. That is the actual asymmetry, and it favours the dual.

## §3 THE DEFECT, RE-DERIVED AT SOURCE (not confirmed — recomputed)

64 `lane_guard` rows, burn-4 windows 01–03, ep644→945:

```
lambda_lane      == 0.0        on 64/64 gates      100% of mass at the KKT lower bound
g = realized-bud <  0          on 64/64 gates      max g = -0.003452, min = -0.053665
budget_s_units   == 0.12589    on 64/64 gates      ONE value, the whole run
realized_lane_s  0.122438 -> 0.072225              the run WON 0.050213 S of Lane
```

Applying the pw1 discriminator: **100% of λ mass at a bound.** But unlike pw1's menus, this bound is
KKT-*correct* — a slack constraint has multiplier zero. The clipping is in the **budget menu**, which had
occupancy 1 (one value, 64 gates) while the signal it constrains moved 39% of its range entirely inside
the slack region.

**Quantified hole:** a budget pinned at the *starting* level licenses the primal to give back **all
0.050213 S-units** of won Lane before the guard can respond. That is **8.11% of the current gap-to-bar**
(0.6189279). It is the size of the hole, **not** a ΔS anyone can bank — a guard pays only if erosion
actually occurs.

## §4 THE DERIVED SCHEDULE

Shape is **derived, not picked**. Lane is 985 GT components on a 384×512 plane with 44–55% already erased;
a thin component that loses its last supporting pixels has no gradient support left to recover through, so
Lane loss is effectively **irreversible**. The correct constraint for an irreversible loss is *do not give
back what has been won* — a monotone non-increasing budget.

```
L_hat(t) = mean of the last m realized values          m = the dual's own integration time (matched
sigma    = derive_noise_floor(history)                     bandwidth => the level loop and the dual
k        = calibrate_deadband_k(sigma, eta, cap, W)        loop cannot resonate)
budget(t) = min( budget(t-1),  L_hat(t) + k*sigma )
```

* **sigma — MEASURED online, never a constant.** Trend-agnostic first-difference estimator
  `sd(diff)/sqrt(2)`; first-differencing annihilates any constant slope, so a descending run does not
  inflate it. Burn-4: **0.00142148 S**; outlier-robust MAD twin **0.00146636** (agree to 3.2% ⇒ no
  outlier contamination); an OLS-detrend estimator gives **0.00356611**, 2.5× inflated by within-window
  curvature — that is the estimator *not* to use.
* **k — DERIVED, then CALIBRATED.** Condition: *noise alone must not move the dual by more than one
  `lambda_step_cap` over the horizon.* Analytic closed form
  `phi(k) − k*Phi(−k) = cap/(eta*sigma*W)` gives **k = 1.7395** at W=64.
* **Feasible by construction:** the ratchet target is a level the run has already held for `m` gates,
  never an extrapolation.
* **Bonus defect removed:** the constant budget compared a 36-of-600-pair gate estimate against a
  600-pair-measured constant, carrying gd1's MEASURED **+3.34% Lane design error**. The gate subset is
  fixed (`gate_ids_n == 36`, `gate_basis == "ema_shadow"` on all 64 rows), so a ratchet — comparing the
  estimator only to itself — cancels that offset by construction.

### §4.1 MY FIRST IMPLEMENTATION FAILED ITS OWN NEGATIVE CONTROL

Shipping the analytic k gave, on 200 stationary null trials × 64 gates at the measured sigma:

> **200/200 trials engaged; 36.20% of all gates engaged.** A thrash generator — precisely the failure the
> deadband exists to prevent.

Cause: the analytic derivation prices the residual against a **fixed** reference, but the shipped budget is
a **running minimum**, and the minimum of W correlated trailing means is biased low by roughly
`(sigma/sqrt(m))*sqrt(2 ln W)`. **Min-selection bias eats the deadband.**

Fix keeps the condition and replaces only its *evaluation*: `calibrate_deadband_k` bisects the SAME
condition against the true running-minimum statistic by simulating the shipped arithmetic under the null.
The analytic k stays as the bisection's lower bracket, so the shipped k is always ≥ it and the gap **is**
the selection bias, reported as `selection_bias_k`. The null statistic is scale-free in one dimensionless
group — `gain = eta*sigma/cap` — which is what makes the memo key small and the simulation cheap.

| | analytic k | calibrated k | selection bias |
|---|---|---|---|
| burn-4 sigma, W=64 | 1.7395 | **2.2699** | **+0.5304** |

### §4.2 CONTROLS (P4 — no meter without a canary)

**NEGATIVE** (200 trials × 64 gates, stationary + N(0,sigma), no erosion):

| | analytic k (rejected) | calibrated k (shipped) |
|---|---|---|
| gates engaged | **36.20%** | **6.35%** |
| trials with any engagement | 200/200 | 179/200 |
| **E[max λ]** | — | **0.0657 ≤ cap 0.1 → PASS** |
| worst-trial max λ | — | 0.243 (ceiling λ_max = 5.0) |
| adversarial −4σ lucky gate | 18/64 engaged | **0/64 engaged** |

The designed guarantee bounds **magnitude**, not incidence: 89.5% of trials show *some* transient λ, but
E[max λ] = 0.0657 is 4.6× smaller than a single `--class-weight-lane 1.3` step and is transient.

**POSITIVE** (descent 0.120→0.075 then genuine erosion over 24 gates):

| erosion | RATCHET | LEGACY constant |
|---|---|---|
| +0.002 S | 0/64 — below deadband, correctly ignored | 0/64 |
| +0.005 S | 6/64, max λ 0.046 | **0/64** |
| +0.010 S | 12/64, max λ 1.062 | **0/64** |
| +0.025 S | 18/64, max λ 1.743 | **0/64** |
| +0.050 S | 20/64, max λ 1.934 | 1/64, max λ 0.001 |

**Measured detection floor sits between 0.002 and 0.005 S — and the derived deadband is
k·sigma = 0.00323 S.** Derivation and behaviour are the same object. The legacy constant budget is blind
to erosion up to +0.050 S.

**On the REAL burn-4 series the ratchet also stays at λ=0 — and that is the correct answer**: that series
improved monotonically, so there was nothing to catch and any engagement would have been a false positive.
What changed is that the budget tracked **0.125890 → 0.078965** (42 distinct values), so the guard is now
armed at the achieved level. Erosion coverage went from **0** to **0.046925 S-units**.

## §5 THE OCCUPANCY SWEEP — 84 discrete choice points inventoried, 5 measured

Inventory denominators (read 4 live-chain files + 6 transitively imported modules = **10 files**):
**12** boolean flags changing numerics, **30** discrete menus, **26** all-or-nothing accept/reject rules,
**16** mode strings = **84 rows**; **22** excluded as pure I/O; **0** in-scope rows belonged to ddm_tw1
(waterfill) or #824 (bias_correction) — neither symbol appears anywhere in the 10 files.

**INTERIOR rows are reported because a sweep that reports only positives commits the selection defect it
is auditing.**

| row | admissible set | MEASURED occupancy | class |
|---|---|---|---|
| **`ST_GRID` per-pair translation scale** (`pfs1_warp_receiver.py:18`) | 11 values, 0.0…0.24 | idx 6/7/8/9 = 22 / **364** / 156 / 58; **0 at idx 0–5 and 0 at idx 10** | **INTERIOR — HONESTLY CLOSED.** Independently reproduces pw1's s_t result at source. *Sub-finding: 60.67% pile on one value (0.08); the defect here would be resolution starvation at the mode, not clipping. 11→16 points is **byte-free** (the index is one byte / 4 bits either way), so refining inside 0.06–0.16 is a zero-rate lever.* |
| **`AUTO_CODECS`** (`ddm_r7_token_coder.py:55`) — `auto` races 2 of 9 `CODEC_IDS` | 9 codecs | **raced all 9 byte-exact on the live 346,478 B stream**: smevr **346,478** (live, winner) · brotli11 +14.42% · lzma1 +14.88% · kt_o8_prev5_backoff +15.02% · kt_prev1 +17.22% · cae_inspired +21.76% · rans_o0 +37.12% · huffman_nibble +50.00% · rans_o0_adj +54.83%. All 9 verified lossless. | **CLOSED — the narrow menu costs nothing.** argmin over 9 == argmin over the 2 searched; best excluded codec is +51,546 B worse. |
| **`selector` ∈ {0 single-plane, 1 two-plane}** (`inflate_runner_v4d.py:163`) | 2 | 376 (62.67%) / **224 (37.33%)**, n=600 | **INTERIOR** — both values carry real mass. *Note: no partial/blended compose is representable; a scaled move does not exist in the receiver.* |
| **token lattice: `token_quant_levels = 16` at bound `2 ≤ levels ≤ 16`, AND the `±1.0` range literal** (`ddm_r7_token_coder.py:1454`; `train_tr1_partition_renderer_mlx.py:624,711`) | 16 levels over a hardcoded ±1 range | interior decay 5→14 **monotone**, then level 15 **jumps 5.65×** (10,222→57,708); per-channel ch3 **8.30×**, ch2 7.60×, ch1 2.28×, ch0 1.26×. **33.30% of all token mass pinned at the two bounds** (lvl0 30.165% + lvl15 3.131%). | **CLIPPING — CONFIRMED AT SOURCE (§5.1).** The clipped knob is the **range literal**, which has no flag, no menu and no DSL lever at all. |
| **`beta_mag` / `dim0`** | — | ddm_pw1 (5ea9cd3f0a) measured CLIPPING and fixed both | **already closed by pw1 — not re-run** |
| **`token_ste` ∈ {round, dither}** (`ddm_tr1_runtime.py:344`) | 2 | live = `round`; `dither` **never swept anywhere in the chain** | **UNMEASURED** |
| **GN line-search `scale ∈ (1.0, 0.5)`** — 5 independent sites | 2 | **no code path records which scale won** | **UNMEASURED — and unmeasurable without adding telemetry** |
| **first-improving bracket direction** (`ddm_v4d_resolve.py:216-218`, `:334-336`) | ±1 | `break` after the `+1.0` probe ⇒ `−1.0` is **never evaluated** when `+1.0` improves at all | **UNMEASURED (premature-closure class, not menu width)** |

### §5.1 ADJUDICATED — CLIPPING CONFIRMED, and the clipped knob is not the one I first suspected

The statistical signature (monotone interior decay 5→14, then a 5.65× jump at the terminal bin; 8.30× on
ch3) is ambiguous on its own: a saturating nonlinearity predicts exactly the same shape. **Eyeballing the
histogram cannot decide it. Reading the source can.**

| candidate generator | source check | verdict |
|---|---|---|
| sigmoid saturation | `sigmoid(x)*255` exists at `train_tr1_partition_renderer_mlx.py:471,478` but is the **RGB/photo head**, not the token path | **REFUTED — there is no sigmoid on the token path** |
| explicit hard clip | `raw_tokens` = `tokens_base + tokens_delta[idx]` (`:612-621`) is a **free unbounded learned parameter**; `quantized_tokens` applies `mx.clip(raw_tokens, -1.0, 1.0)` at `:624`, and the export path applies the identical `np.clip(tokens, -1.0, 1.0)` at `:711` | **CONFIRMED** |

**VERDICT: CLIPPING, MEASURED AND SOURCE-CONFIRMED.** The token latent is an unbounded free parameter
hard-clamped to ±1 by an explicit literal. **33.30% of all token mass sits pinned at the two clip bounds**
(level 0: 30.165%, level 15: 3.131%) — the optimizer is driving these parameters past the clamp and being
cut off. (Train and export clip **identically**, so there is no train/export mismatch — I checked for one
and did not find it.)

**But the clipped knob is the RANGE, not `token_quant_levels`.** These are two separate knobs that the
at-bound signature conflates:

* **`token_quant_levels = 16`** at its bound `2 ≤ levels ≤ 16` is a **RESOLUTION** ladder over a fixed
  range. The 16 ceiling is the 4-bit nibble-packing boundary (`pack_nibbles`), a representation choice.
* **The range `±1.0`** is a **hardcoded literal with no flag, no menu, and no DSL lever at all** — it is
  not even a discrete choice, so no occupancy sweep would ever have surfaced it. It is the knob the mass
  is actually piling against.

That distinction is the finding. Widening the range trades clamping against coarser resolution at fixed
levels; adding levels trades bytes against resolution at fixed range. Which one pays cannot be settled at
$0 — it needs a retrain, because the clamped parameters' *desired* values do not exist in any artifact.

Cheapest probe of the same mechanism, already in the tree and never swept: **`token_ste = dither`**
(`:637`), which changes exactly how near-edge values quantise. `token_ste` and the range literal are
coupled.

### §5.2 What the rate curve prices (MEASURED byte-exact; the distortion side is NOT measured)

`tokens.dr7t` is **346,478 B = 96.2% of the 360,238 B archive** — the rate axis. Re-lattice + re-encode
(smevr, lossless round-trip verified):

| L | bytes | saves vs L=16 | ΔS_rate | % of gap | **pays iff Δd_seg <** | (rel. to live d_seg 0.00431179) |
|---|---|---|---|---|---|---|
| 15 | 330,748 | 15,730 | −0.010474 | 1.69% | 1.047e-04 | 2.43% |
| 14 | 322,823 | 23,655 | −0.015751 | 2.54% | 1.575e-04 | 3.65% |
| 12 | 297,789 | 48,689 | −0.032420 | 5.24% | 3.242e-04 | 7.52% |
| 10 | 267,889 | 78,589 | −0.052329 | 8.46% | 5.233e-04 | 12.14% |
| 8 | 240,824 | 105,654 | −0.070351 | 11.37% | 7.035e-04 | 16.32% |

Widening (L=17) costs ≈ **+15,730 B = +0.010474 S**, so it must buy **> 1.047e-04 d_seg** to pay.

**Pre-registered break-even, both directions.** This is the rate side only; post-hoc re-quantisation is a
LOWER bound on quality (training at L would do better than rounding to L). Exact next measurement:
re-quantise to L, render through `inflate_runner_v4d.py`, score — one scorer job per L. **Not fired here**
(MAIN has a scorer job running).

## §6 WHAT LANDED

* `src/tac/optimization/lane_guard.py` — `derive_noise_floor`, `derive_deadband_k`,
  `calibrate_deadband_k`, `_null_expected_max_lambda`, `derive_ratchet_budget`, memoised
  `_cached_calibrated_k`; `LaneGuardConfig.{budget_ratchet, ratchet_mean_gates, ratchet_horizon_gates}`;
  `LaneGuardState.{realized_history, budget_s_current, inert_slack_gates}`; `dual_ascent(budget_s=…)`
  override (back-compatible); `gate_update` ratchets before the dual and emits the ratchet provenance.
* **Inertness self-report** — `inert_slack_gates` + `inertness_alarm` in every gate row. Burn-4 ran 64/64
  gates in the inert state and **no telemetry field said so**. "Off" is now a tracked, surfaced state.
* `experiments/train_tr1_partition_renderer_mlx.py` — `--lane-guard-ratchet`,
  `--lane-guard-ratchet-horizon`, threaded through `TR1Config` → `LaneGuardConfig`, echoed in
  `lane_guard_init`.
* `src/tac/witness_dsl/spec_tr1_renderer_20260728.py` — `lever_lane_guard_ratchet` (triality: a lever is
  not built until it is a `Lever` factory), with the measured defect, both controls, and its falsifier.
* `src/tac/tests/test_lane_guard.py` — **19 new tests, 71 total green**, including both controls as
  executable regression guards and a regression guard reproducing the measured 64/64 inertness.

**DEFAULT-OFF, deliberately:** `--lane-guard-ratchet` defaults False so sealed tickets recompile
bit-identical and landed runs stay reproducible. This is not an orphan — the guard now reports its own
inertness every gate, per CLAUDE.md *"'Off' is a tracked queue, never a forgotten default"*.

## §7 WHAT I DID NOT DO

* No training, no scorer job, no paid dispatch, no `upstream/` edit, pointer untouched.
* The **distortion** side of every §5.2 row is unmeasured — only the rate side is byte-exact.
* `token_ste = dither`, the GN line-search scale, and the first-improving bracket direction are
  **UNMEASURED**, not cleared. Two of them are unmeasurable without first adding telemetry.
* The ratchet has **never run inside a real trainer** — only against replayed and synthetic series. Its
  first live use is an A/B against `--lane-guard` without the flag.

## §8 FALSIFIERS

1. **Ratchet:** a run where the ratcheted guard engages while every GT-referenced Lane cost is falling ⇒
   it is tracking optimizer noise, and the deadband *calibration* (not the ratchet form) is refuted.
2. **#822 refutation:** a run where `realized_lane_s` and `gt_components_erased[Lane]` decorrelate
   (|r| < 0.5) ⇒ the +0.97 is burn-4-specific and the metric question reopens.
3. **Levels:** re-quantising to L=14 and scoring shows Δd_seg > 1.575e-04 ⇒ the L=16 operating point is
   already rate-optimal downward and only the widening direction remains.
