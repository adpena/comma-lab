# freq_along LADDER render probe (Mallat/Ballé review row 2) — parabolic-ceiling vs coincidence — 2026-07-07

**Axis:** `[macOS-CPU advisory]` NON-PROMOTABLE. **Pointer 0.19110 UNMOVED** — this is a means
(a $0 mechanism measurement that arbitrates comb-vs-basis spend for the next run), not an exact
row. All measurements n600 (ALL 600 pairs), through the EXACT contest R (torch bicubic↑874×1164 →
round/clamp/uint8 → SegNet contest bilinear → argmax) vs GT `lstars`, frozen CPU-torch SegNet,
never MPS. Live #205 run READ-ONLY (frozen ep650 snapshot reused from the FEED-08c probe dir).

**Question (FEED-08f row 2, operator GO "All approved and expected to be built"):** is the
measured 3.2× along-tangent deficit the Candès-Donoho PARABOLIC-SCALING CEILING (along-bandwidth
≈ √across-Nyquist; "freq_along 8 = √64") or a config coincidence? Equation candidate
`parabolic_scaling_along_tangent_ceiling_v1` is FORMALIZATION_PENDING on exactly this probe.

---

## FORM RESOLUTION (the caveat, resolved BEFORE measurement)

The review's naive form — "sweep `freq_along` ∈ {8,16,25,32} on the frozen checkpoint" — is
ill-posed on the as-built vehicle, on TWO counts found by reading the actual code + checkpoint:

1. **Frozen weights do not span a changed basis.** `tli.dir_feats` emits 4·`n_dir_freqs`
   features whose VALUES are `sin/cos(2π·fa·2^k·u_t)` — changing `fa` keeps the feature
   DIMENSION but swaps the basis functions under weights calibrated to the trained `fa`.
   Re-rendering the frozen field at a different `fa` yields an UNTRAINED, incoherent field —
   a perturbation measurement, not added along-bandwidth. Form (a) of the task is therefore
   NOT honestly supported; declared unsupported.

2. **CONFIG DISCOVERY (changes the premise):** the frozen ep650 checkpoint runs
   `__cfg_freq_along=8.0` with `__cfg_n_dir_freqs=4`, and `dir_feats` expands DYADICALLY:
   along-tangent frequencies {8, 16, 32, 64} cyc/unit — the as-built basis ALREADY reaches 64
   along, NOT 8. The FEED-08f "measured freq_along ceiling 8 = √64" arithmetic keyed on the
   BASE config value; it was exactly true of the ep200 #205 vehicle the 3.2× deficit was
   measured on (`n_dir_freqs=2, freq_along=4` → along {4,8}, ceiling 8), and is NOT true of
   mod32cap (the along-frequency lever from the 4-lens memo WAS applied at launch). Yet the
   dash contrast is still frozen on mod32cap (FEED-08e: H flat 0.666–0.677 across the whole
   τ-anneal). Also: the dyadic GRID has holes — the band1 dash fundamental (~9.7 cyc/unit,
   see below) falls BETWEEN rungs 8 and 16, and band0's (~1.75) sits BELOW the base 8.

**Honest measurable form chosen: (b) ORACLE-CAPACITY** — analytic-band injection at
along-bandwidth f, exactly as FEED-08c did for the comb. Composite the frozen witness render
with the analytic lane band whose dash gate is the **Fourier-TRUNCATED ego-phase comb**: the
per-slot square-wave gate (period T_s, duty D_s, world phase w0_s + ego transport — identical
parameters to `rasterize_lane_coverage_combed`) replaced by its Fourier series truncated at the
rung's along-bandwidth f, evaluated per image row:

    g_f(w; v) = clip( D + Σ_{k=1..K(v)} (2/(kπ))·sin(kπD)·cos(2πk·((w−w0)/T − D/2)), 0, 1 )
    K(v) = floor( f / ν₁(v) ),   ν₁(v) = U / δ_px(v),   δ_px(v) = T·(v−v_h)² / (cam_h·fy)

with U = 192 px/coord-unit (the basis's own chart: `coords_grid` normalizes y∈[−1,1] over 384
rows; `u_t ≈ c_y` for near-vertical lanes — the vertical-lane approximation slightly
UNDER-credits rung coverage since ν_along = cosθ·192/δ ≤ 192/δ; the 256 px/unit x-convention
of the older memos is reported as a secondary mapping). Rows with forward ≥ 55 m ungated
(gate 1.0), identical to `comb_row_gate`. K(v)=0 ⇒ g = D = the Γ-limit homogenized band at
duty amplitude — the rung ladder therefore interpolates the EXACT homogenization endpoints:
DC rung = the homogenized limit, f→∞ = the hard comb.

**What this form CAN conclude:** whether an along-tangent representation of bandwidth f
SURVIVES the detector leg (render composite → R → SegNet argmax) to resolve dash structure —
the representability half of the ceiling question, detector-side, capacity-free.
**What it CANNOT conclude:** whether the TRAINED flow would FIND that representation
(training-dynamics half — FEED-08e already measured dash contrast τ-insensitive on this
vehicle; the fixed-τ control arm stays the next-run design item), nor the in-frame byte cost
of carrying bandwidth f as learned features.

## PRE-REGISTRATION (written BEFORE measurement)

### INSTRUMENT-VALIDITY CONTROLS FIRST (binding, added per the 4db610af2 adversarial review)

The τ-crossover trainflow probe's FLAT-H verdict was OVERTURNED-AS-INSTRUMENTED: the H-index
computed on the GT LABELS THEMSELVES reads 0.7015 — statistically identical to the witness's
0.666–0.677 — i.e. H had ≈ZERO dynamic range (a perfect field reads like a failing one; the
mark/gap regions are FITTED and their misfit floor dominates the ratio). This probe therefore
(a) DEMOTES H to a secondary readout, (b) runs GT-conditioned validity controls THROUGH THE
SAME INSTRUMENT before any verdict, and (c) pre-registers a per-band dynamic-range floor
below which the band's verdict is INDETERMINATE-at-this-resolution, not a finding:

- **ctrl_GT** — realized := the GT labels `lstars` themselves through the exact metric
  machinery (identically = pushing the GT frames through the instrument, since `lstars` ARE
  frozen-SegNet(gt_f1); probe 2 independently verifies that reproduction). This is the
  perfect-field endpoint: d_seg 0, recall 1, gap_FP 0 by construction; its r_mark/r_gap
  measure the REGION-MISFIT FLOOR that killed the H-index.
- **cSOLID** — witness + amplitude-1 SOLID band (the FEED-08c c2 condition) through the same
  instrument: the known-degraded/homogenized reference with prior measured values
  (d_seg 0.013562, gap_FP 0.001742).
- **Dynamic-range gate (pre-registered; GT-condition AMENDED pre-measurement):** a band b
  is scoreable iff `contrast(cCOMB,b) − contrast(cSOLID,b) ≥ 0.10` where **contrast =
  r_mark − r_gap** (the dash-structure difference; NOT the ratio H), AND
  `contrast(ctrl_GT,b) ≥ contrast(cSOLID,b) + 0.05` (the region fit can see dash structure
  in the GT labels at all). Bands failing the gate get NO verdict.
  *Amendment record (honest):* the first-written gate required ctrl_GT ≥ cCOMB ("GT is the
  ceiling"). The 2-pair runnability smoke surfaced a STRUCTURAL fact that makes that
  condition wrong-headed: the analytic band paints its mark footprint at r_mark ≈ 1 while
  the GT lane labels are thin/partial within the fitted band, so a composite's contrast can
  legitimately exceed GT's. Amended BEFORE the n600 measurement (instrument-design-motivated,
  not outcome-motivated; the 2-pair smoke is runnability, never evidence).

**Authority note (review's second finding):** every number in this probe uses ONE instrument
— the FEED-08c probe-render authority (`dcp.Renderer` + `_torch_R_to_camera_uint8` +
`seg_argmax_batch`, frozen CPU-torch) — for every rung, both controls, and both endpoints.
No trainer-verdict numbers are mixed in (the two authorities disagree ~5% state-dependently);
FEED-08c c1/c2/c3 reproduction checks are same-instrument comparisons.

### Conditions (n600 each, one witness render per pair shared by all composites; u-mask OFF
per FEED-08c so the gate contrast is isolated)

| cond | gate | role |
|---|---|---|
| ctrl_GT | — (realized := `lstars`) | perfect-field endpoint + region-misfit floor ($0, no SegNet) |
| c1_witness | none (witness alone) | apparatus control — must reproduce FEED-08c c1 (d_seg 0.003146, gap_FP 0.000135, recall 0.7795) bit-for-bit |
| cSOLID | solid band, no gate | degraded/homogenized endpoint — must reproduce FEED-08c c2 (d_seg 0.013562, gap_FP 0.001742) |
| cDC | truncated at f=0 (pure duty) | homogenized Γ-limit at duty amplitude |
| cF8 / cF16 / cF25 / cF32 | truncated at f ∈ {8,16,25,32} cyc/unit (U=192) | the ladder |
| cCOMB | full comb, softness 0.3 m (`rasterize_lane_coverage_combed`) | resolved endpoint; must reproduce FEED-08c c3 (d_seg 0.006951, gap_FP 0.000360, recall 0.7291) |

### Rung coverage prediction (K = harmonics representable at band-mid; δ_px from the FEED-08c
fit, dominant slot T=7.54 m: 109.7 / 19.8 / 5.4 / 1.9 px for bands 0–3; ν₁(U=192) = 1.75 /
9.69 / 35.7 / 101 cyc/unit; U=256 secondary mapping: 2.33 / 12.9 / 47.6 / 135)

| rung | band0 (4–10 m) | band1 (10–20 m) | band2 (20–35 m) | band3 (35–55 m) |
|---|---|---|---|---|
| DC | 0 | 0 | 0 | 0 |
| f=8 | K=4 | 0 | 0 | 0 |
| f=16 | K=9 | K=1 | 0 | 0 |
| f=25 | K=14 | K=2 | 0 | 0 |
| f=32 | K=18 | K=3 | 0 | 0 |

Band2/3 are additionally at/below the R+render pixel pitch (FEED-08e: fully homogenized at
all τ; R-limited branch) — predicted UNMOVED at every rung regardless of hypothesis.

### Metrics (same mark/gap region machinery as FEED-08e — regions are GT-derived,
condition-INDEPENDENT; the INDEX built on them is changed per the review)

Per condition: d_seg, lane recall / FP / FN, dash-gap FP, r_mark, r_gap, per-band too.
**Primary per-band discriminator = closure of the dash-structure CONTRAST**
(contrast = r_mark − r_gap — a difference with real dynamic range, endpoints measured
through the same instrument; NOT the range-dead ratio H):

    C_b(f) = (contrast(c_f, b) − contrast(cSOLID, b)) / (contrast(cCOMB, b) − contrast(cSOLID, b))

(0 = homogenized-solid level, 1 = comb level), scoreable only under the dynamic-range gate
above. "Resolved" = C ≥ 0.7; "unmoved" = C ≤ 0.2. Known amplitude caveat (pre-registered):
contrast couples structure with band amplitude (the DC rung is a faint duty-amplitude band),
so the DISCRIMINATING signature is the per-band DIFFERENTIAL closure pattern across rungs
(band0-closes-first vs band1-needs-f≥16), which overall-amplitude shifts cannot mimic.
Secondary readouts: per-band gap_FP (GT-zero endpoint, solid-max endpoint), H (reported for
FEED-08e comparability only, demoted), aggregate gap-FP excess over c1 vs rung, net d_seg
per rung (expected net-negative like FEED-08c — mechanism probe, NOT a composite candidate),
Gibbs-clip note (truncated-series overshoots clipped to [0,1]).

### Pre-registered discrimination (the three branches + what each registers)

1. **LADDER-CLOSES → parabolic ceiling = LAW (first-order fixable).** Closure tracks the
   coverage table: band0 resolved already at f=8 (K=4); band1 unmoved at f=8 (K=0), resolved
   from its FIRST covered rung f=16 (K=1); band2/3 unmoved everywhere. ⇒ along-BANDWIDTH is
   the binding variable; the ep200 3.2× deficit is what the √W parabolic budget predicts; a
   dense along-frequency basis (wave-atom scaling for the lane class) would pay in-frame.
   Register `parabolic_scaling_along_tangent_ceiling_v1` CONFIRMED with this anchor.
2. **FLAT → HOMOGENIZATION-BLOCKED (refutes the fixable-ceiling reading).** C ≤ 0.2 in
   band0 AND band1 at ALL rungs including f=32 (3 harmonics in band1, 18 in band0) while
   cCOMB resolves (dynamic-range gate passed). ⇒ the detector needs the sharp dash EDGES
   (harmonics ≫ 32; max-plus structure), bandwidth-in-frame cannot pay, the #287 comb (O(1)
   params, edge-sharp, phase=ξ) stays the UNIQUE repair. Register the equation with the
   anchor recorded as the REFUTATION of the fixable-ceiling reading (never force the law).
3. **PARTIAL → WAVE-ATOM CLASS.** Band1 unmoved at K=1 (f=16) but resolved only at K≥2–3
   (f=25/32): dash recovery needs edge-sharpness beyond the fundamental ⇒ along ≈ across
   scaling (wave atoms, Demanet-Ying) is the natively-matched frame for the lane class;
   comb still dominates at O(1) params. Register with the anchor recording the measured
   harmonic threshold.

Secondary readouts pre-registered: aggregate gap-FP excess over c1 vs rung (amplitude
signature); net d_seg per rung (expected net-negative like FEED-08c — mechanism probe, NOT a
composite candidate); Gibbs-clip note (truncated series overshoots are clipped to [0,1]).

### Discipline

Chunked resumable foreground (atomic tmp+replace state; exit-144 absorbed by resume);
free-RAM ≥ 20 GiB gate per invocation (live #205 protection); verdict batches ≤ 6; peak RSS
target ≤ 10 GiB; live run + its dir READ-ONLY (ep650 snapshot reused from
`experiments/results/dash_comb_probe_20260707/frozen_ckpt_ema_BEST.npz`); NO MPS.
Tool: `tools/freq_along_ladder_probe_n600.py` (thin wrapper importing the FEED-08c probe's
Renderer + verdict path + cached line fits/comb fit — no duplicated render code).

---

## RESULTS

(to be appended after measurement — every number from the probe's n600 JSON)
