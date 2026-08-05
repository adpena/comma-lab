---
council_tier: T3
council_attendees: [Schmidhuber, Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Quantizr, Hotz, MacKay, Ballé, Boyd, Tishby-memorial, Time-Traveler, PR95Author]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: true
council_override_rationale: "What would schmidhuber and the pantheon and the pantheon of pantheons and all minds they would invite or want to consult suggest and consider and wonder and pursue and audit and backcast and hybridize and sandwich and exhaust and research and derive variant or original of from here"
model_provenance: "Opus 5 (claude-opus-5). Prior convocations gc11/gc12/gc13 ran on Fable-5; Fable is at its usage limit. Model-identity honesty per the ddm_b4s death record in burn4_cap_decision_ddm_b4r.json."
related_deliberation_ids: [ddm_gc13_optimal_control_shape_20260731, ddm_fl1_perclass_flicker_floors_20260731, ddm_rv1_conditional_validity_regrade_20260728]
---

# ddm_gc14 (#814) — the first descent, audited: it is a BOUNDARY STEP, not a rate

**16th operator-convened pantheon convocation. Scorer-FREE. $0. Pointer `0.1910828242` [contest-CPU] UNMOVED — this convocation is MEANS, not a score mover.**

---

## §0 HEADLINE (answer first)

Burn-4 window_02 is a **real, n600-confirmed improvement of −0.021003 S on the seg axis** (net −0.018303 S after rate). It is NOT what MAIN's seed took it to be. Three corrections, each measured from the primary telemetry, each decision-relevant:

1. **The descent is BOUNDARY-LOCALIZED, not a per-epoch rate.** Over window_02's 29 gates the realized d_seg OLS slope is **−1.46e-7/gate, t = −0.09** — statistically ZERO — while the training loss fell 13.4% (0.54275 → 0.46984). 139 epochs of genuine loss descent bought **no** realized d_seg. The level moved in a **step at the window boundary**. Live window_03 reproduces this exactly: a **−1.118e-4 step** at ep805→809, then flat (in-window slope t=+1.20 full / −0.26 excluding the resume transient). The boundary step is **larger than the whole window's delta so far** (|step| / |Δwindow| = 1.36).

2. **The mechanism is a confirmed apparatus event, not renderer learning.** `experiments/train_tr1_partition_renderer_mlx.py` passes `opt_state_flat={}` at **every** `save_checkpoint` call (lines 1536/1826/1857/1977/2059/2103) and constructs `optimizer = optim.Adam(learning_rate=cfg.lr)` **fresh, unconditionally** (line 1543). **Every window boundary is a full Adam warm restart with zeroed moments.** The signature is present at all three boundaries: ep_loss jumps **+9.6%** (w01→w02) and **+22.8%** (w02→w03) at the first epoch after resume, then re-descends over ~6 epochs, with gnorm ratios 0.94–0.99 (gradients normal ⇒ the spike is optimizer step-scaling, not data). Simultaneously `ema_decay` is **re-derived from cumulative `--epochs` at every window** (0.99991992 → 0.99993383 → 0.99994362; shadow time-constant **166 → 202 → 236 epochs**). The gate is read on `gate_params: "ema_shadow"`. Restart-excursion + lengthening weight-average = an unintended **SWA/warm-restart** regime.

3. **The extrapolation fails by 1–2 orders of magnitude and the arithmetic says so.** Pairing burn-4 with gc13's ep499→641 receipt gives a **MEASURED per-window decay ratio r = 0.310** (−0.06769 S → −0.02101 S over two comparable ~140-epoch windows). Geometric remaining seg from ep805 = **+0.00946 S**. Closing the 0.36640 S seg debt would require **r ≥ 0.9458** (≈ zero deceleration for 17 restarts). Burn-4 window_02 closes **2.3%** of the own-vehicle gap to the bar.

**Verbatim correction to the seed:** "R6 `--class-weight-lane 1.3` + the lg1 λ_Lane guard MEASURED WORKING" is **NOT established**. `lambda_lane` stayed **exactly 0.0 at every gate of all three windows** — the guard never actuated, so it cannot have worked. And the Lane `+42 / slope +11.66 / t=+5.54` reading is a **5-gate-window phase artifact**: over window_02's full 29 gates Lane's OLS slope is **−0.497/gate, t = −1.50**, and over all 38 gates of the burn it is **−0.158/gate, t = −0.69** (flat). The 5-point fit window happens to begin at Lane's trough (485 at ep784).

**What survives and is genuinely good news:** `gt_components_erased` fell **567 → 508 (−59)** across the burn, **Lane −53**, in-loop and unaided. That is the cleanest live receipt the ERF-collateral law has ever had.

---

## §1 PROVENANCE AND AUTHORITY

| item | value | source |
|---|---|---|
| venv | `/Users/adpena/Projects/pact/src/tac/__init__.py` | hijack check clean |
| git HEAD | `c72d1e7b75ee7513bb36e14becd874f060981d64` | — |
| burn custody | `/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/` | READ-ONLY; no writes by this convocation |
| scorer jobs run | **0** | window_03 owns the single n600 slot |
| evidence axis | `[macOS-CPU/MLX advisory]` throughout | `score_claim=false`, `promotable=false` |
| pointer | `0.1910828242` [contest-CPU] **UNMOVED** | `burn4_t0.json` |

**Authority rule applied.** `realized_gate_dseg_mean` is computed on `gate_ids_n = 36` pairs — a **subset**, not evidence, per the allergic-to-non-n600 non-negotiable. Only `full_confirm_dseg` is n600. Every S-arithmetic claim below uses the n600 pair; the 36-pair series is used **only** for shape/trend/mechanism, always labelled.

**Measured gate bias (new finding, staleness-at-consumption):** the 36-pair gate's bias against n600 **MOVED** across the burn — w01 `−1.10e-6`, w02 `+1.52e-5`. Consequence: **the gate OVERSTATES the window_02 descent by 7.2%** (gate −0.022634 S vs n600 −0.021003 S). MAIN's relayed figure `−0.019934 S` net is the **gate-basis** number; the n600-authority number is **−0.018303 S**. Both are recorded; the n600 one binds.

---

## §2 BASELINE DECLARATION (seed item 3 — ONE baseline, propagated)

**DECLARED BASELINE: `window_01` endpoint** — n600 d_seg `0.004277157253689236`, `272,023 B`.

Rationale (apples-to-apples discipline): window_01 is the only baseline measured on the **same apparatus, same coder (`smevr`), same config family, same gate geometry** as window_02. The rung-1 parent was produced under `ddm_r1c_20260731` with a different `--epochs` and therefore a different derived `ema_decay` — a different measuring instrument. Choosing it would confound the descent with the apparatus change, which is itself measured at **+0.00131 S** (ep641 → ep665 re-smoke made seg slightly *worse*).

| baseline | seg ΔS | rate ΔS | net ΔS |
|---|---:|---:|---:|
| **window_01 endpoint (DECLARED)** | **−0.021003** | **+0.002700** (+4,055 B) | **−0.018303** |
| rung-1 parent, n600 `0.004264077` @ 273,004 B | −0.019695 | +0.002047 (+3,074 B) | −0.017648 |
| gate-basis (arm's supersession record) | −0.022634 | +0.002700 | −0.019934 |

The spread across baselines is **0.0023 S** — small relative to the descent but **12.6% of it**. All downstream arithmetic in this memo uses the declared row.

---

## §3 THE AUDIT — MAIN's seed, row by row

**Verdict key:** DERIVED-with-receipt / PLAUSIBLE-with-named-$0-measurement / HAND-WAVED-discard / **CORRECTED**.

| # | seed claim | verdict | receipt |
|---|---|---|---|
| S1 | n600 d_seg 0.0042778 → 0.0040519, Δ −0.000226 ≈ −0.0226 S | **CORRECTED** | Mixes bases. n600 full_confirm pair is 0.004277157 → 0.004067128 = **−0.021003 S**. The `0.0040519` cited is the **36-pair gate**, not n600. |
| S2 | NEW BEST realized d_seg, beats rung-1 endpoint 0.0042641 | **DERIVED** | n600 0.004067128 < 0.004264077. Holds on the n600 pair. |
| S3 | Lane 498→540 net +42, slope +11.66/gate, t=+5.54 ⇒ R6 + λ_Lane guard MEASURED WORKING | **CORRECTED — two independent errors** | (a) `lambda_lane = 0.0` at every gate of all 3 windows (`realized_lane_s_units` 0.1167 < budget 0.12589) ⇒ **the guard never engaged**. (b) 5-gate fit is phase-conditioned: full-window Lane slope **−0.497/gate t=−1.50**; all-38-gate slope **−0.158/gate t=−0.69**. Lane oscillates ±25 comps with no trend. |
| S4 | Undrivable eroding (slope −0.799 vs derived ε 0.633, net −3 comps) | **CORRECTED — predicate lacks a reference term** | Undrivable GT betti0 = **38**. Realized went **42 → 41** (start above GT). |realized − GT| went **4 → 3 = IMPROVED**. `gt_components_erased` went **2 → 2 = UNCHANGED**. The model removed **spurious fragments**; the alarm read convergence-to-GT as erosion. |
| S5 | total_counted_bytes 276,078 (w01 272,023; rung-1 parent 273,004) | **DERIVED** | Confirmed. Note all variation is in `tokens_bytes_smevr`; `renderer_bytes=3284` and `selector_ledger_bytes=216` are **constant at every gate**. |
| S6 | Birth key NOT fired (slope ≫ ε) | **DERIVED but non-informative** | Correct arithmetic; but the key is fit on the same aliased 5-gate window as S3, so "not fired" carries no trend information. |
| S7 | "A measured dS/dt now exists for the first time ⇒ real economics are computable" | **CORRECTED — this is the load-bearing error** | There is no dS/dt. Within-window d_seg slope is **t = −0.09** (w02) and **t = +1.20 / −0.26** (w03). What exists is a **per-boundary step**. Economics keyed to epochs/hours are mis-specified; economics keyed to **restarts** are computable. |
| S8 | 11:1 favorable rate:seg exchange TODAY | **CORRECTED** | The ratio is arithmetically right (0.021003/0.002700 = 7.8:1 on n600) but it is **not an exchange**. Gate-to-gate correlation of Δbytes with Δd_seg is **r = +0.212, t = +1.30** — bytes and d_seg drift *together* (both worse), not traded. No evidence that spending bytes buys d_seg. |
| S9 | Rate grows with complexity ⇒ computable crossover | **PLAUSIBLE, refined** | Counted bytes are **non-monotone**: 272,883 → **271,508** (ep689 min) → **276,224** (ep799 max) → 275,872 (ep834). But raw token entropy IS monotone: `tokens_bytes_zlib` **439,843 → 454,511 (+3.3%)** while smevr rose only +1.1%; the coder ratio drifted 0.6125 → 0.5993. **The coder is absorbing entropy growth; the rollover is coder-side, not entropy-side.** |
| S10 | 0.3664 / 0.0226 ≈ 16 windows ≈ 33h — is multi-day continuation the right shape? | **CORRECTED — answer is NO, and by a wider margin than the seed feared** | With the declared baseline it is 42 windows / 86 h **ignoring all deceleration**. With the measured r = 0.310 the total remaining is **0.00946 S** — 2.6% of the debt. See §5. |

---

## §4 THE DISCRIMINATOR — pre-registered, three branches, non-overlapping

MAIN's seed offered two branches. **They are not mutually exclusive**, because fl1 §0.2/§3 explicitly scopes its floors as FORMULATION-level ("binds ONLY witnesses temporally SMOOTHER than GT... NEVER a paradigm floor"; pierced by PR130 2.966e-4, FEED-ma 0.00086, our own ep641 0.004264). A binary would force a false choice. And the telemetry surfaces a **third** branch the seed did not consider, which is currently the front-runner.

### 4.1 The branches

- **(a) REACHABLE-HEADROOM DRAIN.** The descent consumes the above-floor pool: Undrivable +0.01635 S, Movable +0.00945 S, total **0.0258 S** (fl1 §3). Predicts flattening on contact with the bound. Window_02's −0.021003 already consumed **81%** of that entire pool in one window ⇒ **≤1.23 windows of runway from ep665**.
- **(b) FLOOR-PIERCING.** The renderer descends *through* the smooth-label floors via structural (not temporal) improvement. **Road is the decisive class**: ep641 level 0.18845 vs floor 0.18894 ⇒ **zero above-floor headroom**, so ANY Road descent is floor-piercing by construction. Supporting leading indicator already in hand: Road `betti0_realized` **47 → 66** toward GT 77 (**t = +10.99**), and Road `gt_components_erased` **10 → 8**.
- **(c) BOUNDARY-STEP / APPARATUS ARTIFACT — NEW, and currently leading.** The level moves at window boundaries (Adam warm restart + EMA time-constant lengthening), not within windows. Predicts: step at each boundary, flat inside, magnitude set by restart count not epoch count, and partial give-back as the live weights re-diffuse.

### 4.2 Live evidence as of ep834 (IN-WINDOW telemetry, 36-pair gate — NOT the discriminator's answer)

MAIN's caveat is binding and is honored: early-window rates run hot and this is not an n600 endpoint verdict.

```
w02 last gate  ep805 : 0.0040519
w03 gates      ep809 : 0.0039402   <-- boundary step -1.118e-4
               ep814 : 0.0039725
               ep819 : 0.0039728
               ep824 : 0.0040241
               ep829 : 0.0039982
               ep834 : 0.0039695
w03 in-window OLS  : +7.87e-6/gate, t = +1.20  (full, n=6)
w03 ex-transient   : -3.56e-6/gate, t = -0.26  (n=4, gates >= ep819)
|boundary step| / |window delta so far| = 1.118e-4 / 8.24e-5 = 1.36
```

**Reading (honest):** MAIN is right that the LEVEL has not flattened. But it did not *descend* either — it **stepped, then held**, and has given back 29% of the step. This is branch (c)'s exact signature, and it is *inconsistent* with a sustained within-window rate under (a) or (b).

### 4.3 Pre-registered thresholds — stated so the current trajectory lands cleanly in ONE branch

Consumed from `experiments/ddm_b4r_endpoint_extras.py` output at the ~18:40Z endpoint. Let
`Δ_tot = 100 × (full_confirm_w03 − 0.004067128)` (n600, seg S units), and let `ΔS(c)` be the per-class n600 seg S deltas in xp1's exact convention.

**TEST 1 — LOCALITY (fires first; determines whether 2 and 3 are even meaningful).**
Using window_03's `a1_gate` series, `step = |d_seg(ep809) − 0.00405191|`, `slope_t` = OLS t of gates with `epoch ≥ 819`:

| condition | verdict |
|---|---|
| `step ≥ 0.5 × |Δ_window_gate|` **AND** `slope_t > −2.0` | **(c) BOUNDARY-LOCALIZED** — descent is an apparatus event |
| `slope_t ≤ −2.0` **AND** `step < 0.5 × |Δ_window_gate|` | **DISTRIBUTED** — genuine within-window learning; proceed to Tests 2/3 |
| otherwise | **MIXED** — report both legs, do not collapse |

*Current standing at ep834: `step`=1.118e-4, `|Δ_window_gate|`=8.24e-5 ⇒ ratio **1.36 ≥ 0.5**; `slope_t` = **−0.26 > −2.0** ⇒ **(c) BOUNDARY-LOCALIZED**, unambiguously, with no overlap into the other cells.*

**TEST 2 — MAGNITUDE (the drain/pierce envelope on n600).**

| `Δ_tot` band | verdict |
|---|---|
| `Δ_tot > −0.0030` | **DRAIN COMPLETE** — pool exhausted, stop |
| `−0.0100 ≤ Δ_tot ≤ −0.0030` | **(a) DRAIN CONFIRMED** — consistent with r = 0.310 (point prediction −0.00652) |
| `−0.0140 < Δ_tot < −0.0100` | **AMBIGUOUS** — Test 3 adjudicates |
| `Δ_tot ≤ −0.0140` | **(b) SUSTAINED / FLOOR-PIERCING** — r = 0.310 FALSIFIED; deceleration model discarded |

**TEST 3 — DECOMPOSITION (mechanistic; the only test that separates drain from pierce).**
Let `R = |ΔS(Undrivable) + ΔS(Movable)|` (the two above-floor classes) and `P = |ΔS(Road) + ΔS(Lane) + ΔS(MyCar)|` (all at-or-below their fl1 floors, so any negative here is piercing by construction).

| condition | verdict |
|---|---|
| `R ≥ 0.60 × |Δ_tot|` | **(a) DRAIN-DOMINANT** — runway bounded by the residual pool |
| `P ≥ 0.60 × |Δ_tot|` | **(b) PIERCE-DOMINANT** — fl1's pool does NOT bound the runway; only the rate crossover stops it |
| neither | **MIXED** — report the split; runway = residual pool + an unbounded pierce term |

**Auxiliary pre-registration (falsifiable, cheap):** residual above-floor pool at the w03 endpoint,
`pool = max(0, ΔS_level(Undriv) − 0.03939) + max(0, ΔS_level(Movable) − 0.02847)`.
Branch (a) predicts `pool ≤ 0.010`. Branch (b) tolerates any `pool`.

### 4.4 Point prediction (pre-registered before the endpoint exists)

From the measured r = 0.310: **w03 seg ΔS = −0.00652, endpoint n600 full_confirm d_seg = 0.0040019, net ΔS = −0.00382.**

The live ep834 gate (0.0039695, bias-corrected to n600 ≈ 0.003985) already sits **below** this endpoint prediction with 112 epochs remaining. **I am pre-registering that this prediction will be beaten** — which, per Test 2, would put `Δ_tot` in the DRAIN-CONFIRMED or AMBIGUOUS band, not the sustained band, *unless* the endpoint reaches ≤ 0.0039271 (`Δ_tot ≤ −0.0140`). **My stated expectation: `Δ_tot ∈ [−0.0110, −0.0075]`, Test 1 = BOUNDARY-LOCALIZED, Test 3 = MIXED.** If Test 1 returns DISTRIBUTED I am wrong about the mechanism and §6's burn-5 shape must be discarded.

---

## §5 THE CROSSOVER / STOPPING DERIVATION (seed item 2)

### 5.1 The series is long enough — but only if you key it to the right variable

MAIN asked for a derivation from the per-window series and asked me to say honestly if the series is too short. **It is not too short, because gc13 §2.0 supplies a third n600 point.** The seg-axis trajectory (all n600, all exact):

| epoch | seg S | source | window Δ |
|---|---:|---|---:|
| ep499 | 0.49410 | qa92 per-class sum (gc13 §2.0) | — |
| ep641 | 0.42641 | xp1 endpoint | **−0.06769** (142 ep) |
| ep665 | 0.42772 | burn-4 w01 full_confirm | +0.00131 (24 ep — apparatus change cost) |
| ep805 | 0.40671 | burn-4 w02 full_confirm | **−0.02101** (140 ep) |

**Measured per-window decay ratio r = 0.02101 / 0.06769 = 0.3104** (n = 2 comparable ~140-epoch windows; **verdict_scope: INSTANCE** — one vehicle, one lineage, two windows, no SE available from n=2).

### 5.2 The crossover

Net ΔS per window = seg gain + rate cost. Rate cost is **+0.002700 S/window** (+4,055 B measured). Upper bound if the coder stops absorbing (`tokens_bytes_zlib` growth of +84 B/epoch × 0.599 ratio × 139 ep ≈ +6,978 B): **+0.004646 S/window**.

Net flips positive when `|seg gain| < rate cost`, i.e. when the descent has decelerated by:

- **7.78×** from the current −0.021003 (observed, coder-absorbing), or
- **4.52×** (coder-saturating upper bound).

At r = 0.310, `ln(7.78)/ln(1/0.310)` = **1.75 net-negative windows remaining from ep805**.

| window | predicted seg ΔS | rate ΔS | net | verdict |
|---|---:|---:|---:|---|
| w03 (LIVE) | −0.00652 | +0.00270 | **−0.00382** | net-negative |
| w04 | −0.00202 | +0.00270 | **+0.00068** | **CROSSOVER — STOP** |
| w05 | −0.00063 | +0.00270 | +0.00207 | net-positive |

**Two independent estimates converge.** The deceleration model says **1.75 windows** of runway from ep805; fl1's reachable-headroom pool says **1.23 windows** from ep665 (0.0258 S / 0.021003). These agree within their (unquantified) uncertainty and were derived from completely different evidence — a topological/flicker accounting and an S-trajectory fit. **This convergence is the strongest single result in this memo.**

### 5.3 Geometric total, and why the extrapolation fails

Remaining reachable seg from ep805 under geometric decay: `0.021003 × r/(1−r)` =

| r | remaining seg | net-negative windows left |
|---:|---:|---:|
| 0.310 (**MEASURED**) | **0.00946 S** | 1.75 |
| 0.50 | 0.02100 S | 3.0 |
| 0.70 | 0.04901 S | 5.8 |
| 0.85 | 0.11902 S | 12.6 |
| 0.95 | 0.39906 S | 40.0 |

**To close the 0.36640 S seg debt requires r ≥ 0.9458** — essentially zero deceleration for ~17 consecutive restarts. The measured r is 0.310. **The debt is not closable by continuation under any decay model the data supports.**

### 5.4 The derived stopping rule (supersedes the judgment-based E2 handoff for this axis)

gc13's E2 graduation rule ("when a class's corrected-gap closes within its target band, it graduates") is a **per-class, level-triggered** rule. It is correct and retained. What it lacks is a **campaign-level, rate-triggered** stop. This is that rule, and it is arithmetic:

> **STOP CONTINUATION when the projected next-window seg gain falls below the measured next-window rate cost.**
> `STOP ⟺ |ΔS_seg(k)| × r̂ < ΔS_rate(k)`
> with `r̂` re-estimated from the last two n600 window endpoints and `ΔS_rate` from the last two `total_counted_bytes` endpoints. Both inputs already exist in every window decision record. Fires at **w04** on current values.

**Schmidhuber's contribution to the deliberation makes this exact rule the natural one.** In the compression-progress formulation, the drive is not compression but *the first derivative of compression progress*; when that derivative decays the intrinsic reward vanishes and the agent must move to a domain where progress is still available. Our r = 0.310 is a measured collapse of that derivative. The stopping rule above is compression-progress applied to a burn: **stop when the derivative of the derivative says the domain is mined out.** Schmidhuber's own dissent is recorded in §9.

---

## §6 THE BURN-5 DECISION FUNCTION (pre-registered, typed branches → named configs)

Modelled on gc11's burn-3 gate. Consumes exactly the `ddm_b4r_endpoint_extras.py` fields. **Every branch is scorer-budget-aware; none auto-fires; heavy/paid remains operator-GO (CONTAINMENT).**

```
INPUTS (all from the ~18:40Z endpoint bundle + window_03 telemetry):
  D_tot        = 100*(full_confirm_w03 - 0.004067128)            # n600 seg S delta
  R            = |dS(Undrivable) + dS(Movable)|                   # above-floor classes
  P            = |dS(Road) + dS(Lane) + dS(MyCar)|                # at/below-floor classes
  step         = |gate_dseg(ep809) - 0.00405191|
  slope_t      = OLS t of window_03 a1_gate dseg, epoch >= 819
  pool         = max(0, level(Undriv)-0.03939) + max(0, level(Movable)-0.02847)
  bytes_w03    = total_counted_bytes at endpoint
  r_hat        = |D_tot| / 0.021003                               # realized decay ratio
```

**BRANCH B5-A — `RESTART-CADENCE A/B` (fires if Test 1 = BOUNDARY-LOCALIZED; currently indicated).**
Config: identical seal, identical total epochs, **window length halved** (~70 ep ⇒ 2× the restarts per hour) vs an unbroken control of the same total epochs. Two arms, same seed, same starting checkpoint.
Falsifier: if the half-window arm's n600 endpoint does **not** beat the unbroken control by ≥ 1.5× the measured gate noise (2.3e-5 within-window sd from w01), the restart mechanism is FALSIFIED and B5-C fires instead.
Cost: 2 arms × ~2h local, $0. Consumer: burn-5 ticket + a candidate DSL lever (`--restart-cadence`).
**Why this is the top branch:** it converts an accident into a controlled lever, and it is the *only* config that separates mechanism (c) from (a)/(b) causally rather than correlationally.

**BRANCH B5-B — `POOL-DRAIN FINISH` (fires if Test 3 = DRAIN-DOMINANT AND `pool > 0.004`).**
Config: one further window with the {Lane, Undrivable} guarded set engaged per gc13 §8 item 2 — Undrivable budget = its measured endpoint level, η from its measured erosion — **only if** §7's alarm re-calibration has landed first (else the guard will actuate on convergence-to-GT and *fight* an improvement).
Falsifier: `pool` at the next endpoint ≥ its current value ⇒ the drain framing is wrong.
Stop: unconditional at the §5.4 rule.

**BRANCH B5-C — `HAND OFF THE SEG AXIS` (fires if `r_hat ≤ 0.45` OR §5.4 STOP fires OR B5-A falsifies).**
No further continuation windows. The seg axis is handed to the terminal/solve family and **the slot goes to the rate axis** (§8 ranks these). This is the default branch on current arithmetic and I expect it to fire.

**BRANCH B5-D — `SUSTAINED-DESCENT CONTINUATION` (fires ONLY if Test 2 returns `D_tot ≤ −0.0140` AND Test 1 returns DISTRIBUTED).**
Config: continue at current window length; re-estimate `r̂` each window; §5.4 STOP still binds.
This is the only branch that vindicates the seed's framing. **Pre-registered: I predict it does not fire.**

**Precedence:** Test 1 gates everything. If BOUNDARY-LOCALIZED ⇒ B5-A fires first and B5-B/D are held pending its result, because under mechanism (c) the per-class decomposition is measuring the *restart's* effect, not a learning rate, and would misattribute.

---

## §7 PRIOR-LAW PREDICTION LINES (mandatory anti-re-anchor discipline)

Stated before composing; diffed after.

| law | what it ALREADY predicted | diff vs gc14 |
|---|---|---|
| **non-additive pools** | seg and rate are not independent budgets; same-pool levers compete, never sum | **CONFIRMED in an unexpected direction.** They are not even *trading*: Δbytes vs Δd_seg gate-to-gate `r = +0.212, t = +1.30`. The "11:1 exchange" presumes an exchange the data does not exhibit. New corollary: **an exchange rate is only meaningful between quantities shown to be causally coupled.** |
| **caps-law / #312 (loss weights at stage boundaries only)** | nothing changes per-step; changes belong at boundaries | **HONORED for loss weights, VIOLATED in spirit elsewhere.** Adam state and `ema_decay` both change at every boundary. The law's unstated corollary — **a boundary change is a MEASUREMENT EVENT and must be recorded as one** — is the gap gc14 closes. |
| **verdict-scope ladder** | one window = INSTANCE | **NOT HONORED by the seed.** "The campaign's FIRST genuine coupled descent" is a campaign-level reading of an INSTANCE-scope observation. Correctly scoped: INSTANCE. |
| **ERF-collateral law** | in-loop births pay no post-hoc tax; recovery routes through in-loop mechanisms only | **CONFIRMED — strongest live receipt yet.** `gt_components_erased` 567→508 (−59), Lane −53, in-loop, zero injection, zero collateral. |
| **constants-are-poison** | thresholds must be DERIVED, never hand-set | **PARTIALLY HONORED, two new instances found.** ε is correctly derived, but (i) `UNDRIV_EROSION` has **no GT-reference term**, so convergence-to-GT reads as erosion; (ii) `n_points = 5` is a **hard-coded estimator window** that aliases a ~30-gate oscillation. **New law: constants-are-poison applies to an estimator's WINDOW LENGTH and REFERENCE FRAME, not only to its threshold.** |
| **staleness-is-a-named-confound (freshness at consumption)** | consume fresh, input-hash lineage, fail closed | **NEW INSTANCE.** The 36-pair gate is a biased estimator of the n600 quantity **whose bias moved** (−1.1e-6 → +1.52e-5), overstating the descent by 7.2%. Freshness is not the only staleness axis; **estimator-bias drift is one too.** |
| **alarm-predicates-are-per-vehicle-calibration-objects** (today's law, from window_01's `term_domination`) | first fire = calibration event | **CONFIRMED, and gc14 supplies calibration events #2 and #3** (UNDRIV reference-term; 5-gate aliasing). The law is generalizing exactly as written. |
| **fl1 FORMULATION-scoping of the floors** | floors bind only smooth-label witnesses; piercing is expected and already evidenced | **LOAD-BEARING.** This is *why* MAIN's (a)/(b) cannot be a binary and why §4 is a decomposition. fl1 predicted this shape; the seed's framing did not consume it. |
| **gc13 Pontryagin TPBVP form** | forward primal + backward dual meet at gates, **settle at window boundaries** | **AMENDED — a genuine defect found.** The TPBVP treats the boundary as a *settlement point*. Measured: the boundary is also a **state discontinuity** (Adam moments → 0, `ema_decay` re-derived). **A two-point boundary-value problem whose forward dynamics jump at its own settlement points is ill-posed unless the jump is in the model.** gc13 §8 needs a jump/reset term. See §10. |
| **rv1 conditional-validity re-grade** | changed preconditions re-open descent-conditioned verdicts | **APPLIES, but weakly.** Because the descent is boundary-localized (Test 1), the "trained trunk sits at its flat-basis optimum" family (recall A1–A6) is **NOT** re-opened by burn-4 — the trunk's *within-window* behaviour is still flat. See §12. |

---

## §8 PURSUE — ranked op-routables (falsifier + consumer + fire-timing each)

| # | routable | falsifier | consumer | fires |
|---|---|---|---|---|
| **R1** | **Restart-cadence A/B (B5-A)** — half-window vs unbroken control, same total epochs, same seed | half-window arm fails to beat control by ≥1.5× gate-noise (3.5e-5) | burn-5 ticket; candidate `--restart-cadence` DSL lever | **at the w03 endpoint**, if Test 1 = BOUNDARY-LOCALIZED |
| **R2** | **Alarm re-calibration ×2** (add GT-reference term to `UNDRIV_EROSION`; derive the estimator window length from the series' own autocorrelation instead of `n_points=5`) | a re-calibrated predicate still fires on the w01–w03 series where the GT-distance improved | supervisor predicates; cg1 #809 sensor leg | **before B5-B** — a guard on a mis-signed predicate would fight an improvement |
| **R3** | **Adopt the §5.4 rate-triggered STOP** into the window decision record | `r̂` estimated over 3+ windows proves non-geometric (e.g. plateau-then-drop) | burn supervisor; gc13 §8 amendment | immediately, $0 — inputs already in every decision JSON |
| **R4** | **Hand the slot to the RATE axis** — QA24 `cell_drop50` (359,221 B, n600 −0.098 seg+rate, byte-closed `a6398e44`) and wr1 #766 Knee-A/B (274,333 B / 174,578 B) | composed re-measure fails to reproduce the banked byte-closed deltas on the current parent | su2 ordering (rate parent FIRST, then re-solve pose) | on B5-C |
| **R5** | **Fix or formalize the optimizer-state discontinuity** — either persist Adam moments (restoring bit-faithful resume per the P0 non-negotiable) *or* declare the reset a **named, scheduled lever** with recorded provenance | neither: the current state (saved-as-`{}`, silently reset, undocumented at the decision surface) is indefensible under both readings | TR1 trainer; resume-registry; deterministic-repro spine | with R1 (they are the same experiment) |
| **R6** | **`camera_fl = 910` + yuv6 2×2 polyphase → terminal pose solve** (us1 F2/F4) | the 6-eq GN's realized Pose MSE does not improve when the true focal length replaces whatever it currently assumes | #366/su2/QA43 chain | post-burn slot (unchanged) |
| **R7** | Full row-by-row re-grade of the #390 negative-findings register + `ddm_deferral_queue_ledger` under the *descent* precondition | Test 1 = BOUNDARY-LOCALIZED ⇒ precondition did not actually change ⇒ **do not run** | rv1 §2b | **HELD** pending Test 1 |

---

## §9 SANDWICH (descent rate above / floors below, per class)

Per-class, with fl1 floors as the lower leg and the measured trajectory as the upper leg. Levels are ep641 (xp1); floors are fl1 §2 (/598 interior).

| class | ep641 level S | fl1 floor S | above-floor pool | ep499→641 rate | burn-4 topology signal | sandwich reading |
|---|---:|---:|---:|---:|---|---|
| Road | 0.18845 | 0.18894 | **0.0000** (at floor) | −0.03360 | betti0 47→66 (t=+10.99) toward GT 77; erased 10→8 | **The pierce test.** No pool left; any descent is structural piercing. Fastest historical mover. |
| Lane | 0.12589 | 0.23162 | 0 (already **46% below**) | **+0.00151** | betti0 flat (t=−0.69 over 38 gates); erased **539→486 (−53)** | Already pierced. Erasure recovery is real and large; component *count* is not the right observable here. |
| Undrivable | 0.05574 | 0.03939 | **+0.01635** | **+0.00204** | betti0 42→41; **|realized−GT| 4→3**; erased 2→2 | Largest reachable pool. The alarm mis-signed it (§7). |
| Movable | 0.03792 | 0.02847 | **+0.00945** | −0.01778 | betti0 110→114 toward GT 134; erased 16→12 | Second reachable pool; already draining. |
| MyCar | 0.01840 | 0.04343 | 0 (below floor) | −0.01987 | betti0 **36 = GT 36 exactly, constant at every gate** | **Graduated.** Zero variance across 38 gates. gc13's first graduation candidate — confirmed exact. |
| **total** | **0.42641** | 0.53184 | **+0.0258** | **−0.06769** | erased **567→508 (−59)** | Aggregate already pierces the smooth-label floor by 0.104 S. |

**Sandwich verdict:** the *reachable* leg (0.0258 S) is 81% consumed by window_02 alone. The *pierce* leg is unbounded by fl1 but is gated on Road, which is the class the endpoint decomposition must price. **MyCar is done** (betti0 exactly GT, zero variance) — its dual should retire now, not later.

---

## §10 HYBRIDIZE — which compose, which conflict

| pair | verdict |
|---|---|
| descent × **rate waterfill (#766 / QA24 gr1)** | **COMPOSE, and this is the high-value pairing.** All burn-4 byte variation is in `tokens_bytes_smevr` (renderer 3,284 B and selector 216 B are frozen). The waterfill operates on exactly that lattice. But composition must be **re-measured, never summed** (gr1's own discipline; the 8.8%-additivity trap). |
| descent × **cg1 guard ledger** | **CONFLICT until R2 lands.** A guard built on the current `UNDRIV_EROSION` predicate would actuate against convergence-to-GT — it would *defend* spurious fragments. Sequence: R2 → then guards. |
| descent × **terminal pose solve (#366/su2/QA43)** | **COMPOSE, unchanged.** #383 conditioning gate is untouched by this convocation; pose fires after the seg trunk is conditioned. But note the known integrity defect (`ADVISORY_vehicle_line_synthesis`: #383 mode absent from the compiled launch; epoch-726 bypass) — that is a pre-existing owed item, not a gc14 finding. |
| descent × **phase-faithfulness axis (#425 / #535 / W1-COH)** | **HELD — see §11.** These are the natural consumers of a *floor-piercing* verdict and are dominated under a *drain* verdict. |
| descent × **gc13 Pontryagin TPBVP** | **AMEND.** The forward dynamics have a jump at every settlement point. Either (i) model the reset as an impulse in the forward system, or (ii) remove the discontinuity (R5) so the TPBVP's own assumption holds. **(ii) is cleaner and is the recommendation.** |
| descent × **Adam-restart mechanism** | **This IS the composition** — and it was unintentional. R1 makes it deliberate or kills it. |

---

## §11 THE PHASE-FAITHFULNESS ADJUDICATION (mandated routing ii)

**Standing:** #425 (phase carrier), #535 (FiLM flicker sidecar), W1-COH (flicker-phase coherence) are all **designed-never-fired**, all held under `QF02` / `QA10` in `ddm_deferral_queue_ledger_20260729.md`, none registered in the canonical task ledger. Lane is the #1 phase-faithfulness debt (0.2316 S = **43.6% of floor mass**, **13.1× its corner-C allocation**).

**Honest cost, from the memos (no invented numbers):**

| item | measured/predicted bytes | predicted S benefit | blocker |
|---|---|---|---|
| #425 raster leg | **10,682 B** (rate +0.007113) | **NONE BANKED** — `recovered_d_seg = OWED_through_R_n600_AB` | through-R A/B needs trainer render plumbing against the live-run dir |
| #425 STORE leg (dash δ(s)) | **37,158 B** / 29,958 B excl-ξ | **NONE BANKED** — through-R d_seg OWED | 16.6× over the 0.9–1.8 KB anchor budget; **post-hoc pose-value storage without compact code-to-photometry inverse is FORMULATION-scoped dead on the witness vehicle** (`post_hoc_stored_corrections_dead_joint_descent_required_law_20260718` re-scoped by `ddm_ub1`/VS1). This is not a seg/TR1 law and cannot kill a joint-trained store leg. |
| #535 FiLM flicker sidecar | **2,400 B** int8 (rate +0.00160) | **NONE BANKED** — "No 1–5 KB success claim exists before the exact packer row" | hard `[REQUIRED BEFORE BUILD]`: Fisher-weighted Jacobian spectrum never measured |
| W1-COH | 12–26 KB all-in (static map 9.7–13.5 KB) | d_seg reach ≤ **0.00167** (all flicker) / 0.00037 (deep tail); B/err 0.075–0.141 vs water 1.273 | **uint8 actuation cost kills the band** (gc8); Bayes-floor preflight owed to Yousfi |

**ADJUDICATION.** This axis fires on a **PIERCE-DOMINANT** verdict and is **dominated** on a DRAIN verdict, for a reason that is arithmetic rather than aesthetic: under DRAIN the runway is the 0.0258 S pool, which continuation is already draining at ~$0; paying 10–37 KB (0.007–0.025 S of rate) to chase a d_seg reach that **no memo has ever banked** is negative-expectation against a free alternative. Under PIERCE, the pool no longer bounds the runway, the floors become the live object, and Lane's 13.1× corner-C debt becomes the largest single named target.

**Named fire criteria (pre-registered):**
- **W1-COH actuation** fires iff Test 3 = PIERCE-DOMINANT **AND** the Yousfi flicker-Bayes-floor preflight (owed, $0, from existing W1-COH receipts) returns a floor **below** 0.00167. Otherwise the channel stays HELD — its uint8 actuation cost is already known to kill the break-even band.
- **#535** fires iff PIERCE-DOMINANT **AND** its `[REQUIRED BEFORE BUILD]` Fisher-spectrum measurement lands first. That measurement is $0 and can run in any quiet slot; **it should be queued regardless of branch**, because it is the cheapest way to convert a never-fired design into a priced one.
- **#425** stays HELD in both branches. Its post-hoc pose-value store-apply ancestor is FORMULATION-scoped dead on the witness vehicle, but no standing seg/TR1 law is banked; the train-side leg is not a burn-5-shaped object. Note the recorded caveat that the killing probe was partly a strawman ("the EFFICACY of conditioning + the real coherent codec is UNTESTED") — that is a genuine re-open hook, but it must be tested as joint training / through-R benefit rather than cited as a killed carrier family.

**PR130's mechanism is LESSONS-ONLY** per the no-old-lineage ban. Its 2.966e-4 is cited here **only** as an existence proof that the smooth-label floor is pierceable — never as a vehicle, carrier, or calibration source.

---

## §12 AUDIT — conditional-validity re-grade of no-descent verdicts

Recall pass 1 surfaced a large descent-conditioned kill cluster (flat-amplitude exhaustion A1–A6; capacity-bound plateau B1–B10; DEFERs C1–C5). **The re-grade is NARROWER than it first appears, and that narrowing is itself the finding.**

The A-family's precondition is verbatim "**the trained trunk sits at its flat-basis optimum**" — a statement about the trunk's *within-window* behaviour. Burn-4 measures that directly: **within-window d_seg slope is t = −0.09 (w02) and t = −0.26 (w03 ex-transient)**. The trunk is *still* flat within windows. **Burn-4 therefore does NOT re-open the flat-amplitude family.** The level moved at boundaries, which is an apparatus event, not evidence that the trunk has left its optimum.

This is the honest, disappointing answer to "what does a working descent make possible that was closed before?" — and it is the *right* answer, because the alternative (re-opening ~20 verdicts on a misread) is exactly the signal-loss failure the rv1 methodology exists to prevent.

| family | re-grade |
|---|---|
| A1–A6 flat-amplitude exhaustion | **NOT re-opened.** Precondition (trunk at within-window optimum) still holds. |
| B1–B10 capacity-bound / "epochs alone cannot reach" | **STRENGTHENED, not weakened.** r = 0.310 is a fresh, independent, same-vehicle receipt for exactly this claim. B2's precedent (a capacity verdict that flipped once on a descent receipt) is noted and does not apply: that flip had a *within-window* receipt; this one does not. |
| C1–C2 structured-decomp DEFER | unchanged; reactivation conditions untouched. |
| C4 dw1 "endpoint is NOT converged — plain continuation still descends" | **AMENDED.** Still true at the level; but gc14 shows continuation's gain is boundary-keyed. "More/rewarmed optimization" should read **"more RE-WARMS"** — which is precisely R1. |
| C5 "Road AT its formulation floor (capacity exhausted under current render family)" | **PROVISIONAL-PENDING-VERIFICATION.** Road's betti0 rose 47→66 (t=+10.99) with erased 10→8 — structural improvement at a class with zero above-floor pool. The w03 per-class endpoint adjudicates. |
| D-family (ERF law; gc12 SKIP rule; gc13 pool census row C) | **unchanged / strengthened**, as recall predicted. |

**R7 (full register re-grade) is therefore HELD**, contingent on Test 1. If Test 1 returns DISTRIBUTED, R7 fires immediately.

---

## §13 EXHAUST — pool census with Contrarian P·O bounds

`P·O` = probability × outcome, in S. Contrarian SKIP rule (gc12 §3.3, adopted): **P·O < 0.05 S ⇒ SKIP.**

| pool | P·O bound | status |
|---|---:|---|
| **T·continuation (all classes)** | geometric remaining **0.00946 S** at measured r=0.310 | **DRAINED — SKIP by the Contrarian bound (0.0095 ≪ 0.05).** This is the census's headline. |
| **T·restart-cadence (R1)** | unbounded above by the same pool; if restarts are the true lever, `2×` restarts ⇒ at most `2×0.00946 = 0.019 S` under the same geometric envelope | **SKIP-ADJACENT** on outcome, **PURSUE** on information: it is the only causal test of the mechanism, at $0. Justified by information gain, not by S. |
| **Undrivable above-floor** | +0.01635 S at ep641, ~81% consumed | drained; guard needed only to *hold*, and only after R2 |
| **Movable above-floor** | +0.00945 S, already descending | drained |
| **Road structural pierce** | **UNBOUNDED by fl1** (floor is FORMULATION-scoped) | **the one seg pool that survives the census** — priced by the w03 per-class endpoint |
| **Lane phase-faithfulness** | 0.2316 S floor mass, 13.1× corner-C | HELD; no banked S benefit on any carrier (§11) |
| **RATE — QA24 cell_drop50** | **−0.098 S** (n600, byte-closed `a6398e44`) | **LIVE, largest measured single pool in the census** |
| **RATE — wr1 #766 Knee-A / Knee-B** | −0.197 S / −0.263 S rate, pose damage shown to be stale-params (ck1 recovery parity 0.98×) | **LIVE** |
| **Pose terminal solve** | banked fallback 0.12689 S consumes 74% of the bar ⇒ terminal solve MANDATORY | unchanged |
| **C (post-hoc injection, all classes)** | qa92 P·O = 0.0171; ERF collateral net-worse | **DEAD** — gc14 adds a receipt *for* the law, not against it |

**Census verdict:** after burn-4, **the seg-continuation pool is exhausted by the Contrarian bound.** The two surviving pools of material size are **Road structural pierce** (unpriced, adjudicated at ~18:40Z) and **RATE** (measured, byte-closed, ~gap-sized). This is the census that routes B5-C.

---

## §14 BACKCAST from the bar

Bar = `min(0.15, official 0.172141)`. Own-vehicle exact-protocol line = **0.9639878** (measured 07-31, v4d refine stack).

| step | S | note |
|---|---:|---|
| own-vehicle line | 0.963988 | |
| + burn-4 w02 net (if it composes — **NOT byte-close-verified**) | **0.945685** | −0.018303 |
| official bar | 0.172141 | gap **0.773544** |
| **burn-4 share of the gap closed** | **2.3%** | |
| windows to close at held rate, **ignoring all deceleration** | **42** (≈86 h) | |
| `r` required to close the 0.36640 S seg debt | **0.9458** | measured `r` = **0.310** |

**Per-window boundary requirement (what must be true at each window boundary to stay on a bar-reaching trajectory):** each ~140-epoch window must deliver ≥ 0.0184 S net. Window_02 delivered exactly that (−0.018303). **Window_03 must deliver the same to stay on trajectory** — and the pre-registered forecast is −0.00382, a 4.8× shortfall. The backcast therefore *predicts its own failure at the very next boundary*, which is the honest way to state it.

**Conclusion of the backcast:** continuation is a **2.3%-per-window contributor with a measured 3.2× per-window decay**. The bar is not reachable on this axis. The gap is RATE (as MEMORY already records: "gap=RATE"), and the census (§13) agrees.

---

## §15 SUGGEST / CONSIDER / WONDER — beyond the seed

**What every prior convocation assumed because nothing descended:** that the *cause* of any future descent would be **learning**, and therefore that the *control variable* would be **epochs/hours**. Every economics frame in gc11–gc13 — windows-to-target, S/window, hours-to-corner — inherits that assumption. Burn-4 is the first data that can test it, and it fails: the control variable is **restarts**, not epochs. **This is the assumption the Assumption-Adversary should have surfaced three convocations ago, and it could not have been surfaced without a descent to audit.** That is the honest answer to "what does a working descent make possible" — not new levers, but the first opportunity to *falsify the campaign's economic model*, which it did.

**Wonder (unresolved, worth naming):** if a boundary is worth −1.1e-4 in realized d_seg and costs ~4 seconds of process restart, then the burn's *chunking policy* — chosen for supervisor governance, not for optimization — has been a silent, unpriced hyperparameter for the entire campaign. How many other governance knobs are secretly optimizers? The `|net betti0| ≤ 10` pre-authorization bound is explicitly labelled **class-4 governance** (owner MAIN, re-derivation trigger = λ_undriv via cg1 #809) — correctly labelled. The *window length* was never labelled at all.

**Derive-original (from the frozen scorer's structure, not a transplant):** the restart mechanism's benefit should be predictable from the scorer's own geometry. The head is exact rank-4 with flip distance `d = |m|/‖Δw_e‖`; realized d_seg is a *count of sign flips of a margin field*. A weight-space average over restart excursions reduces the **variance** of the margin field near zero without moving its mean — which lowers the expected flip count by exactly the mass the margin distribution has within ±(excursion scale) of zero. **This predicts the restart gain is proportional to the margin density at zero**, which is measurable at $0 from the banked QA80 margin atlas. That is a derived, vehicle-native prediction, and it is the cheapest possible falsifier for R1 — it can be computed *before* the A/B runs. **Queued as R1's preflight.**

---

## §16 cg1 (#809) AMENDMENTS (mandated routing iv)

`TaskUpdate` is **not available in this environment** (no such tool in my tool set) and #809 has **no row in `.omx/state/canonical_task_status.jsonl`** — its registration lives only in `.omx/state/current_focus.md` L107-111 and its build ticket in gc13 §9 R2. Amendments are therefore recorded here, per the deliverable's fallback instruction.

1. **SENSOR LEG IS NOW P0, AHEAD OF THE DUAL LEG.** cg1's calibration table was gated on the burn-4 endpoint. The endpoint will supply the *numbers*, but gc14 shows two of the *predicates* are mis-specified. **A guard built on `UNDRIV_EROSION` as currently written would actuate against convergence-to-GT** (Undrivable realized 42→41 with GT 38; `gt_components_erased` unchanged at 2). Land R2 before any λ_undriv.
2. **ADD A GT-REFERENCE TERM to every erosion predicate.** Erosion must be `d/dt |realized − GT|` (or `d/dt gt_components_erased`), never `d/dt realized`. `gt_components_erased` is already in the telemetry at every gate and is the correct, sign-unambiguous observable.
3. **DERIVE THE ESTIMATOR WINDOW.** `n_points = 5` is a hard-coded constant that aliases a ~30-gate oscillation (Lane: 5-gate `t=+5.15` vs full-window `t=−1.50` vs 38-gate `t=−0.69`). Derive the fit window from the series' own autocorrelation length. **Constants-are-poison applies to the estimator window, not only to the threshold** (new law, §7).
4. **RECORD BOUNDARY EVENTS AS FIRST-CLASS LEDGER ROWS.** Every window boundary changes Adam state and `ema_decay`. The ledger must carry `boundary_event {restart: true, ema_decay_before/after, opt_state_restored: false}` so no future reader attributes a boundary step to a lever.
5. **MyCar DUAL RETIRES NOW.** `betti0_realized = 36 = GT 36` at every one of 38 gates, zero variance. gc13 named MyCar the first graduation candidate; gc14 confirms it exactly. Do not spend a dual on it.
6. **λ_Lane's engagement threshold is untested, not validated.** `lambda_lane = 0.0` throughout; `realized_lane_s_units` 0.1167 vs budget 0.12589. The guard has never actuated. cg1 must not inherit "lg1 works" as a premise — it inherits "lg1 has never been exercised."
7. **The `|net betti0| ≤ 10` pre-authorization bound is correctly typed class-4 governance** and should be *kept* as such, with its re-derivation trigger unchanged. gc14 endorses this typing explicitly: it is the model of how a governance knob should be labelled, and the contrast case (window length, unlabelled) is the defect.

---

## §17 WHAT I COULD NOT DO / OWED

- **No scorer jobs** (window_03 owns the slot) ⇒ no per-class n600 for burn-4. Every per-class claim above is from `topology_per_class` (component counts), which is a *proxy* for d_seg, not d_seg. The endpoint bundle closes this at ~18:40Z.
- **r = 0.310 is n = 2**, INSTANCE scope, no SE. It is the best available and it is stated as such. Window_03 makes it n = 3.
- **The SWA/warm-restart interpretation is an INFERENCE**, not a measurement. Confirmed: opt state never saved/restored; loss spikes at every boundary; `ema_decay` changes. Not confirmed: that these *cause* the d_seg step. **R1 is the experiment that would establish causation**, and until it runs the mechanism is DERIVED-from-code + CORRELATED-with-telemetry, not proven.
- **My own falsified hypothesis, recorded:** I first read line 1159 as `load_checkpoint` collapsing live weights onto the EMA shadow (an SWA step). **Wrong** — line 1159 is inside `ema_snapshot_swap`; `load_checkpoint` restores `param::` into the live model correctly. Recorded because a fix built on that misreading would have been worse than the defect.
- **Burn-4's composition onto the own-vehicle 0.9639878 line is NOT byte-close-verified.** §14's 0.945685 is arithmetic on the assumption of composition, labelled as such, and is not a score claim.

---

## §18 TYPED VERDICTS (per-landing pantheon review, §22 discipline)

| # | claim | scope | evidence grade | verdict |
|---|---|---|---|---|
| V1 | window_02 delivered n600 seg −0.021003 S / net −0.018303 S | INSTANCE | n600 `full_confirm` ×2, `[macOS-CPU/MLX advisory]` | **MEASURED** |
| V2 | The descent is boundary-localized, not a within-window rate | INSTANCE→FORMULATION | 38-gate OLS: w02 t=−0.09, w03 t=−0.26 ex-transient; step/window ratio 1.36 | **MEASURED** (36-pair basis, labelled) |
| V3 | Every window boundary is an Adam warm restart with zeroed moments | STRUCTURAL | source: `opt_state_flat={}` ×6 call sites; `optim.Adam` fresh at L1543; loss spikes +9.6%/+22.8% | **MEASURED (code + telemetry)** |
| V4 | `ema_decay` is re-derived per window (τ 166→202→236 ep) | STRUCTURAL | three `start` cfg records | **MEASURED** |
| V5 | The restart *causes* the d_seg step | — | — | **INFERRED — R1 owed** |
| V6 | "λ_Lane guard MEASURED WORKING" | INSTANCE | `lambda_lane=0.0` at all 38 gates | **FALSIFIED** (never engaged) |
| V7 | "Lane strongly birthing (+42, t=+5.54)" | INSTANCE | full-window t=−1.50; 38-gate t=−0.69 | **FALSIFIED as a trend** (5-gate phase artifact) |
| V8 | `UNDRIV_EROSION` fire is a real priced trade | INSTANCE | \|realized−GT\| 4→3; `gt_components_erased` 2→2 | **RE-CLASSIFIED — predicate lacks a GT-reference term; the reading is convergence, not erosion.** Distinct from window_01's SPURIOUS_PORTED_PREDICATE: that predicate was mis-ported; this one is mis-*referenced*. |
| V9 | Measured per-window decay r = 0.310 | INSTANCE (n=2) | qa92/xp1/burn-4 n600 trajectory | **MEASURED, low-n** |
| V10 | Continuation cannot close the seg debt | FORMULATION | requires r≥0.9458 vs measured 0.310 | **DERIVED** |
| V11 | Crossover at window 4; STOP rule §5.4 | FORMULATION | 1.75 (decel) / 1.23 (fl1 pool) windows, two independent estimates | **DERIVED, convergent** |
| V12 | 36-pair gate overstates the descent by 7.2% | INSTANCE | bias −1.1e-6 → +1.52e-5 | **MEASURED** |
| V13 | `gt_components_erased` 567→508 (Lane −53) in-loop | INSTANCE | 38-gate series | **MEASURED — strongest ERF-law receipt to date** |
| V14 | MyCar is graduated | INSTANCE | betti0 = GT = 36, zero variance, 38 gates | **MEASURED** |
| V15 | gc13's TPBVP has an unmodelled jump at its settlement points | STRUCTURAL | V3+V4 | **DERIVED — gc13 §8 amendment owed** |
| V16 | Burn-4 does NOT re-open the flat-amplitude family | FORMULATION | within-window slope still flat (V2) | **DERIVED** |
| V17 | Pointer moved | — | — | **NO. `0.1910828242` [contest-CPU] UNMOVED.** |

**Relative significance note (pace≠direction):** V1's −0.018303 S is **2.3%** of the own-vehicle gap and **0.0%** of the pointer. The largest *measured* number in this memo is not the descent — it is QA24 `cell_drop50`'s banked **−0.098 S**, five times larger, on the axis the census says survives. [magnitude-ok on all dismissals below 0.001 S.]

---

## §19 DISSENT (verbatim, preserved per maximum-signal)

- **Schmidhuber (LEAD):** "I am given a first derivative and asked for economics. I return the second derivative, because that is where the drive lives — and it has collapsed by 3.2× in one step. But I dissent from the SKIP: r=0.310 is n=2 and both points come from *different apparatus*. The ep499→641 window ran on r1c; burn-4 runs on b4s, and this memo itself measures the apparatus change at +0.00131 S. A decay ratio computed across an instrument change is not a decay ratio. I want window_03 as the third point on *one* instrument before §5.4 binds."
- **Contrarian:** "The convergence of 1.75 and 1.23 windows is being sold as independent corroboration. It is not fully independent — both are normalized by the same −0.021003. If that number is wrong (and §1 shows the gate version of it was wrong by 7.2%), both estimates move together. The agreement is weaker than it reads."
- **Assumption-Adversary:** "This memo congratulates itself for finding an unexamined assumption while resting on one: that a geometric decay model is the right functional form. Nothing in the data selects geometric over, say, plateau-then-cliff, or a sum of two exponentials with a slow component. With n=2 you cannot distinguish them, and the three models give wildly different answers at window 6. §5.3's table is a sensitivity analysis dressed as a forecast."
- **Yousfi:** "The `gt_components_erased` result — Lane −53 recovered in-loop — is being filed as a corroborating receipt for an existing law. It is bigger than that. 53 recovered GT components on the class holding 43.6% of the floor mass is the most interesting number here, and this memo does not price it in S because it could not run a scorer. That is a real gap, not a footnote."
- **Rudin (CO-LEAD):** "I endorse the decision function's form — every branch is a readable rule chain over named fields with a stated falsifier. But B5-A's falsifier uses '1.5× the measured gate noise' where the noise estimate (2.3e-5) comes from a 6-gate window. Six points is not a variance estimate."
- **Time-Traveler:** "You already had this. The `A1_REALIZATION_GAP_ALARM` classification fires in the telemetry at ep649, 659, 674, 714, 814 — the instrument was *naming* the loss/realized decoupling the whole time, and three convocations read past it. We do not need more information. We need to read what the apparatus is already saying."

---

## §20 ASSUMPTION-ADVERSARY VERDICTS (with `empirical_verification_status`)

| assumption | classification | `empirical_verification_status` | rationale |
|---|---|---|---|
| "A measured dS/dt exists ⇒ epoch-keyed economics are computable" | **CARGO-CULTED** | `VERIFIED_VIA_EMPIRICAL_ANCHOR` (38-gate OLS + changepoint) | Falsified this convocation. Within-window slope t=−0.09; the level is boundary-keyed. |
| "The 36-pair a1 gate is a usable stand-in for n600 d_seg" | **CARGO-CULTED** | `VERIFIED_VIA_EMPIRICAL_ANCHOR` | Bias moved −1.1e-6 → +1.52e-5; overstates the descent 7.2%. |
| "A window boundary is a settlement point (gc13 TPBVP)" | **CARGO-CULTED** | `VERIFIED_VIA_SOURCE_INSPECTION` (trainer L1536–1543, L1548–1577) | It is also a state discontinuity. |
| "The derived-ε erosion machinery is correct, so its fires are real" | **HARD-EARNED (ε) / CARGO-CULTED (the observable)** | `VERIFIED_VIA_SOURCE_INSPECTION` + telemetry | ε derivation is sound; the *observable* it is applied to lacks a GT reference. |
| "Geometric decay is the right functional form for r" | **CARGO-CULTED** | `ASSUMED_AWAITING_VERIFICATION` | n=2 cannot select functional form. Verdict downgraded accordingly (§21). |
| "fl1's floors bound the runway" | **HARD-EARNED, correctly scoped by fl1 itself** | `VERIFIED_VIA_EMPIRICAL_ANCHOR` | fl1 states the FORMULATION scope explicitly; §4 honors it via decomposition rather than a binary. |
| "MyCar has graduated" | **HARD-EARNED** | `VERIFIED_VIA_EMPIRICAL_ANCHOR` | betti0 = GT = 36, zero variance, 38/38 gates. |

---

## §21 VERDICT STATUS

**`PROCEED_WITH_REVISIONS`**, and per the recursive-self-reflection protocol two claims are downgraded:

- **§5.3's per-r table and the "1.75 windows" figure are `PROVISIONAL-PENDING-VERIFICATION`**, on the Assumption-Adversary's functional-form objection and Schmidhuber's cross-instrument objection. Verification: window_03's n600 endpoint as the third point, on one instrument.
- **§5.4's STOP rule is NOT downgraded.** It does not depend on the functional form — it compares a *projected next-window* gain against a *measured* rate cost using whatever `r̂` the last two endpoints give. It is safe to adopt now (R3).
- **V5 (restart causes the step) stands as INFERRED** and gates nothing except R1's design.

---

## §22 THE ONE-PARAGRAPH ANSWER

Burn-4 window_02 is real and it is small: **−0.018303 S net (n600), 2.3% of the own-vehicle gap to the bar**, delivered not as a training rate but as a **step at a window boundary caused by an unintended Adam warm restart** that the code makes at every resume and that no decision record has ever priced. The seed's two most load-bearing readings — a computable epoch-keyed dS/dt, and a working Lane guard — are both falsified from the primary telemetry: the within-window d_seg slope is statistically zero while the training loss fell 13.4% (a realization gap the a1 classifier was already naming), and `lambda_lane` was 0.0 at every gate of all three windows. Pairing burn-4 with gc13's ep499→641 receipt gives a measured per-window decay of **r = 0.310**, which two independent routes (deceleration, and fl1's 0.0258 S reachable pool) agree leaves **~1.2–1.8 net-negative windows** before the rate term wins; closing the seg debt would need r ≥ 0.9458. **Continuation is exhausted by the Contrarian bound; the surviving pools are Road structural pierce (unpriced, adjudicated at the ~18:40Z endpoint) and RATE (measured, byte-closed, gap-sized).** The pointer is **`0.1910828242` [contest-CPU], UNMOVED** — this convocation moved no score and is MEANS.
