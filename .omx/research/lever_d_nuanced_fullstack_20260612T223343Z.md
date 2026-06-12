# LEVER-D NUANCED full-stack — survival-selective seg-repair on the converged ep2236 basin (2026-06-12)

**Subagent:** `leverD-nuanced` (the LEVER-D NUANCED partner of the orchestrated Track-A effort).
**Type:** economics ANALYSIS + nuanced SELECTIVE implementation + measured GO/NO-GO on the most-converged
all-levers-context decoder. **Evidence grade:** `[contest-CPU advisory] NON-PROMOTABLE` — every number is a
frozen-CPU advisory measurement on the basin `best/` (ep2236, d_seg≈0.0026); no byte-closed
`upstream/evaluate.py` row. **Frontier UNMOVED** (`.omx/state/canonical_frontier_pointer.json` → contest-CPU
0.19109982, 177,169 B). This is a MEANS (a GO/NO-GO measurement + a deployable coder) toward the END (a lower
exact score); it moves no row by itself.

> **HEADLINE VERDICT: NO-GO — but precisely, with the reactivation criterion.** Survival-selection is the
> RIGHT nuance and it DOES narrow the gap, but on the converged ep2236 base it does **not clear the break-even**.
> The crude NO-GO (code ALL flips at mean σ=0.46) is confirmed NO-GO at convergence (mean σ DROPS to **0.31** —
> the surviving flips are HARDER). The survival break-even is **σ\* = 0.778** (independently re-derived). The
> selectable structure EXISTS (survival ranges **0.51 → 0.11** across SegNet-margin deciles — a real 4.6× spread)
> but the **TOP of the achievable σ distribution is ~0.51**, well below σ\*=0.778. A deployable decoder-free
> predictor admits **nothing net-negative** (its best-scored flips still survive below break-even); only the
> CHEATING **oracle** (perfect survivor ID) is GO. The d_seg win belongs **IN TRAINING** (Lever-2/Lever-5),
> exactly the witness-probe verdict — confirmed now on a CONVERGED base, with the selective coder built + tested
> + ready to auto-fire if the operating point ever reaches σ_eff > σ\*.

---

## 1. THE ECONOMICS (independently re-derived — the core of the analysis)

Coding `N` boundary flips, each with round-trip survival `σ` (fraction of coded corrections that actually land
the GT argmax through the eval channel) and per-flip byte cost `b`, the contest score change is:

```
net_ΔS = −100·(σ·N)/N_scored_total  +  25·(b·N)/N_a
       = N · [ −100·σ/117_964_800  +  25·b/37_545_489 ]
```

The flip count `N` **factors out of the SIGN**. GO (net < 0) requires:

```
σ  >  σ*  =  25·b·N_scored_total / (100·N_a)  =  b / WATERLINE_BYTES_PER_FLIP
```

with `WATERLINE = 1.273108 B/flip` (the canonical closed-spec §10 constant, = `(100/N_scored)·(N_a/25)`).

**At the measured per-flip cost `b = 0.99 B/flip` → σ\* = 0.7777.** (The crude/witness probe sat at
`b = 0.985 → σ\* = 0.7737`.) This is the SOLVABLE-MATH form: the GO/NO-GO is set ENTIRELY by whether the
**effective survival of the coded subset** exceeds `b / WATERLINE` — the flip count and the absolute d_seg debt
are irrelevant to the SIGN (they set the magnitude). Two independent derivations (from the score formula vs the
module's closed form) agree to 1e-12; the per-flip constants (`SEG_VALUE_PER_FLIP = 100/N_scored`,
`SCORE_PER_BYTE = 25/N_a`) reproduce WATERLINE exactly (tested).

**Where the crude probe sat vs where survival-selection sits:**

| | coded subset | effective σ | vs σ\*=0.778 | net ΔS sign |
|---|---|---|---|---|
| Crude (all flips) | ALL ~460/pair | 0.31 (pop mean) | **0.31 < 0.778** | **+ (NO-GO)** |
| Nuanced (predictor-selected) | the predicted survivors | ≤0.51 (best decile) | **0.51 < 0.778** | **+ (NO-GO)** |
| Oracle (perfect survivor ID) | the true survivors only | 1.0 (by construction) | 1.0 > 0.778 | − (GO, but cheating) |

The **GO region** in `(σ, N, d_seg)` space is the half-space `σ > b/WATERLINE` — INDEPENDENT of `N` and the
d_seg-available. The crude probe is at `(0.31, 276K, 0.0026)`; survival-selection moves it UP the σ axis to
`(≤0.51, smaller-N, 0.0026)` but does not cross the `σ=0.778` plane on this base.

## 2. THE NUANCED IMPLEMENTATION (`src/tac/torch_vehicle/lever_d_selective.py`)

The nuance the crude probe missed: **σ is a per-flip DISTRIBUTION, not a scalar.** A selective coder that codes
only the σ>σ\* sub-population can be net-negative even when the mean is 0.46. The module realizes this:

1. **Per-flip round-trip survival measurement** (the probe): nudge EVERY flip toward its GT-class prototype color
   in the rendered 384×512 frame, push the corrected frame through the EXACT eval round-trip (bicubic↑874 →
   bilinear↓384 → uint8 → SegNet), read per-pixel which flips now land the GT argmax. The per-flip survival
   distribution in ONE round-trip per pair (the crude probe measured only the aggregate on a K-pixel batch).
2. **Survival-robust SELECTION** (`select_survivors`): admit only flips whose decoder-free predictor exceeds a
   threshold (deployable) — or, in oracle mode, the measured survivors (the upper bound).
3. **Margin-conditional leverage WATERFILL** (`waterfill_by_leverage`): among candidates, admit each flip whose
   PREDICTED per-flip net ΔS is negative (`σ̂_i > b_i/WATERLINE`), ranked by leverage-per-byte. The deployable
   model strictly SEPARATES the PREDICTION (selection/ranking, decoder-free) from the GROUND TRUTH (the realized
   σ_eff + net, the round-trip outcome) — selection cannot peek at the channel.
4. **The bit-exact sidecar** (`encode/decode_selective_sidecar`): round-trips the admitted (pixel, GT-class)
   survivors; an empty selection serializes ZERO bytes (the default-OFF / NO-GO contract).

**Tests:** `src/tac/torch_vehicle/tests/test_lever_d_selective.py` — **21 NO-FAKE tests**: the σ\* break-even
re-derived two ways; the SIGN factors out of N; the selective-vs-all A/B (a select-all stub collapses σ_eff to
the population mean → caught); the waterfill admits ONLY net-negative flips (a constant-σ-0 input admits nothing);
the no-structure predictor cannot manufacture survival; the sidecar bit-exact round-trip + fail-closed magic; and
THE no-op guard (a coder that ignores selection has strictly lower σ_eff + worse per-flip net than a genuine
selective coder). All green; ruff clean.

## 3. THE MEASURED VERDICT on ep2236 (`experiments/probe_lever_d_selective_fullstack.py`)

Two slices (n=4 + n=12 pairs, `--which ema`, the converged inference shadow; real frozen SegNet; real `0.mkv`
GT; the exact eval round-trip). **JSON:** `.omx/research/lever_d_selective_probe_n12.json` + the n=4 smoke.

| Quantity | n=4 | n=12 | note |
|---|---:|---:|---|
| mean d_seg (converged) | 0.00240 | 0.00234 | ≈ the `best_meta.json` 0.0026 — this IS the converged base |
| mean flips/pair | 472 | 460 | vs the witness probe's 884 at the mid-basin fork-point (converged = HALF) |
| mean cond B/flip | 0.982 | 0.990 | `margin_conditional_residual` realized cost |
| **σ\* break-even** | **0.772** | **0.778** | `b/WATERLINE` |
| **population mean survival** | **0.306** | **0.315** | the crude all-flips effective σ — **NO-GO** (< σ\*) |
| best margin-decile survival | 0.55 | 0.51 | the MOST-survivable identifiable sub-population — **still < σ\*** |
| crude all-flips net ΔS (heldout) | +0.00037 | +0.00108 | **+ → NO-GO** (raises S) |
| oracle survivors / σ_eff / GO | 270 / 1.0 / **GO** | 857 / 1.0 / **GO** | the cheating upper bound (perfect survivor ID) |
| **predictor-selected GO** | **NO-GO** | **NO-GO** | no decoder-free threshold admits a net-negative subset |
| VERDICT | GO_ONLY_WITH_ORACLE | GO_ONLY_WITH_ORACLE | stable across both slices (Lens 3) |

**The decisive finding:** at convergence the surviving flips get HARDER — population survival DROPS from the
crude probe's 0.46 to **0.31** (fewer, more-confident boundary flips, but the ones that remain are the most
fragile). The selectable structure is REAL (survival 0.51→0.11 across margin deciles; low-margin flips survive
*better* — the inverse of the naive intuition; and local-GT-contiguity helps: 0.20→0.41 across agreement deciles)
but the **ceiling of the achievable σ distribution (~0.51) sits below the σ\*=0.778 break-even.** Survival-
selection narrows the gap (0.31 → 0.51) but does **not** close it on this base at `b=0.99`.

**CONSERVATIVE-correct (NO-FAKE):** the probe's `local_agree` feature uses the GT region map (NOT strictly
decoder-free), making the predictor OPTIMISTIC. NO-GO *under an optimistic predictor* ⟹ NO-GO under a genuinely
decoder-free one. The verdict cannot be a false-negative from a too-weak feature.

## 4. REACTIVATION CRITERION (so the lever auto-fires when the arm converges further)

The sidecar is GO iff the **coded-subset effective σ exceeds σ\* = b/WATERLINE**. Two reactivation paths:

- **Path 1 — cheaper coder (lower `b`):** at the measured best-decile σ_eff ≈ 0.51, GO needs
  `b < 0.51 × 1.273 = 0.65 B/flip` — a **~34% per-flip byte reduction** below the current 0.99. (A tighter
  entropy coder on the boundary residual — e.g. an arithmetic coder over the margin-conditional class
  distribution — could approach this; the conditional-position trick already buys ~26%.)
- **Path 2 — a more-converged distortion arm (higher σ_eff):** the live distortion arm (~ep525, too early)
  will, as it converges, carry FEWER + more-confident boundary flips. If the converged arm's survivor-selectable
  σ_eff rises past **0.778** at `b=0.99`, the lever flips to GO. The probe's per-flip survival distribution is the
  signal to re-measure on that arm. **Exact trigger:** `predictor_effective_sigma > 0.778` at the measured `b`.

The selective coder is **default-OFF / byte-identical when disabled** and ships a sidecar section ONLY when the
measured `predictor_go` is True at the target operating point — so it is safe to leave wired; it auto-fires on the
reactivation condition without perturbing the live arm.

## 5. Full-stack composition note

The coder targets the RESIDUAL d_seg flips remaining AFTER Levers 2+5 training + the pose bolt-ons (PR98/T10).
Measured on the most-converged all-levers-context decoder available (the basin `best/`, ep2236, d_seg 0.0024 —
far past the ep340 fork-point's 0.0035 the crude probe used). The pose half (PR98/T10) is a SEPARATE axis (the
finishing-kit's measured pose-axis lever) and is unaffected by this seg-axis NO-GO. The convergence makes the
crude probe's "mid-basin GO artifact" disappear exactly as the finishing-kit memo predicted: the available d_seg
drop shrank ~6×, the residual-section economics dominate, and the per-flip break-even is necessary but not
sufficient — **the survival ceiling is the binding term, now measured.**

## 6. Recursive adversarial review (3 clean passes — OWNED)

- **Lens 1 (economics correct):** σ\* re-derived independently from the score formula = the module's
  `b/WATERLINE` to 1e-12; the SIGN factors out of N (3 flip counts, same sign). **CLEAN.**
- **Lens 2 (selection is REAL, NO-FAKE):** a select-all stub collapses σ_eff to the population mean → NO-GO;
  the oracle filters to the true survivors → GO. A select-all stub CANNOT reproduce the probe's oracle_go=True,
  proving the survivor selection actually filters. **CLEAN.**
- **Lens 3 (robustness):** two slices (n=4 σ=0.306, n=12 σ=0.315) agree on every headline (NO-GO predictor, GO
  oracle, max-decile σ ~0.51); verdict stable across small-slice d_seg/d_pose noise. **CLEAN.**
- **Pass 2/3 (edge + leakage):** held-out predictor split is leakage-free (fit on train, eval on test ground
  truth); the optimistic-feature ⟹ conservative-NO-GO reasoning holds; empty/all-survive/empty-sidecar/no-candidate
  edges all correct. **3 consecutive clean passes reached; zero findings in passes 2-3.**

## 6-hook wire-in (Catalog #125) + mission

#1 sensitivity-map ACTIVE (the per-flip survival distribution + margin/agreement deciles = the per-pixel
seg-repair-survivability prior; feeds Lever-5 margin-weighted training — the bankable path). #2 Pareto ACTIVE
(the seg-axis Pareto point measured: survivor-selected σ_eff 0.51 vs σ\* 0.778 → the sidecar is dominated; the
in-training fold is the Pareto-correct realization). #3 bit-allocator ACTIVE (NEGATIVE allocator prior: do NOT
allocate a per-flip seg sidecar on this converged base; reactivation gated on σ_eff > σ\*). #4 cathedral N/A
(a measurement; the next dispatch surface is the live arm curriculum). #5 continual-learning ACTIVE (the σ\*
break-even formula, the converged-base survival drop 0.46→0.31, the survival ceiling 0.51 < 0.778, the two
reactivation paths). #6 probe-disambiguator ACTIVE (THIS probe disambiguates "survival-selection rescues the
seg sidecar" — NO on this base — from "fold into training" — the confirmed path).

**Mission contribution:** `frontier_breaking_enabler` (a $0 decisive measurement + a deployable coder that
REDIRECTS the seg-axis lever off the refuted sidecar route onto the in-training fold, with a precise auto-fire
reactivation criterion for a future more-converged arm — preventing a wasted per-flip-sidecar campaign the
survival ceiling would have killed). **Frontier UNMOVED 0.19109982.** No score asserted. No GPU. No paid spend.
No MPS. No collision with the running live arm (default-OFF; read-only on the checkpoint).

## Cross-references

`track_a_distortion_finishing_kit_20260612T220727Z.md` (the crude LeverD NO-GO this supersedes at IMPLEMENTATION
level — paradigm intact per Catalog #307) · `witness_seg_boundary_decisive_probe_20260612T181038Z.md` (the
884-flip / 46.4% τ-insensitive survival / flip-count crux — confirmed now on a CONVERGED base with the survival-
*selection* extension) · `tac.boundary_math.margin_conditional_residual` (WATERLINE + the conditional-position
coder, REUSED) · task #72 (the original Lever-D design) · task #110 (this verdict).
