# Training-flow τ-crossover probe — the owed anchor of `dash_erasure_homogenization_v1` — 2026-07-07

**Axis:** `[macOS-CPU advisory]` NON-PROMOTABLE. **Pointer 0.19110 UNMOVED** — this is a means
(mechanism measurement on the law's training-flow leg), not an exact row. All measurements n600
(ALL 600 pairs), through the EXACT contest R (torch bicubic↑874×1164 → round/clamp/uint8 → SegNet
contest bilinear → argmax) vs GT `lstars`, frozen CPU-torch SegNet, never MPS.

**Context.** The render-side τ/smoothing-crossover probe (dash-comb probe 2, 2026-07-07) was
**REFUTED-AS-IMPLEMENTED** — amplitude confound: blurring a thin (softness 1 px) rendered band
collapses peak α toward no-op, so the sweep measured corrector-amplitude decay, not homogenization
(`.omx/research/dash_comb_probe_verdict_20260707.md` probe 2). This probe measures the REAL trained
field along the live #205 run's OWN τ-anneal trajectory instead — operator directive verbatim:
*"Do the training-flow version of the probe or as much as you can and update the tracking."*

---

## PRE-REGISTRATION (written BEFORE measurement)

### Hypothesis (from `dash_erasure_homogenization_v1`)

Dash-gap FP — GT dash-GAP pixels rendered as lane class-1 through the exact R + frozen SegNet =
the homogenized-solid-band signature — stays HIGH while the training-time smoothing scale (softmax
temperature τ) is above the dash-period crossover, and falls as τ(ep) anneals below it — **UNLESS**
the R-Nyquist scale binds at range, in which case gap-FP stays high at ALL τ, which CONFIRMS the
law's R-limited branch (the #287 comb is then the unique repair). Either outcome refines the law;
neither is forced.

### Honest deconfound limit (stated up front, not hidden)

Along a single trajectory, **epoch count and τ co-vary** — this probe CANNOT fully separate
more-training from lower-τ. The discriminating signature is SHAPE + CONTRAST:

- Compare the gap-FP trajectory against the total-d_seg and lane-RECALL trajectories at the same
  checkpoints. If gap-FP improvement is localized to a τ window while total d_seg improves
  smoothly → supports τ-crossover. If gap-FP is flat (or rises with recall) while recall
  improves → supports R-Nyquist-bound / pinned-homogenized rendering.
- **Partial internal control (anticipated, pre-registered):** between ep726 (τ=0.21682) and the
  live EMA snapshot (ep925, τ=0.21569), τ is nearly FROZEN (Δτ ≈ −0.5%) while ~200 epochs of Muon
  training elapse. Any gap-FP/recall movement across that segment is ≈ pure-training-at-fixed-τ,
  whereas ep299→ep650→ep726 mixes anneal (0.806→0.310→0.217) with training. This within-trajectory
  contrast partially mitigates — but does NOT eliminate — the confound.
- A definitive deconfound needs a **fixed-τ control arm** (two runs, identical seed/config, one
  with τ frozen at 0.8, one annealed) — a NEXT-run design item for the council (named in the
  council-draft addendum; NOT run here).

### Amplitude-normalized signature (added to kill the prior probe's confound class)

Raw gap-FP is amplitude-confounded by overall lane-rendering strength (a witness that renders
almost no lane anywhere has near-zero gap-FP trivially). Pre-registered primary contrast is the
**homogenization index** H, per pair and per forward-range band:

    mark region = solid-band footprint ∧ comb-gate ON   (GT dash MARKS)
    gap  region = solid-band footprint ∧ comb-gate OFF  (GT dash GAPS)
    r_mark = P(realized == lane | mark)     r_gap = P(realized == lane | gap)
    H = r_gap / r_mark

H → 1 = homogenized (field paints marks and gaps alike = solid band); H → 0 = dash structure
resolved. H is invariant to overall lane amplitude/recall, so it is the shape discriminator the
render-side probe lacked. Regions come from the SAME cached per-pair line fits + global ego-phase
comb the dash-comb probe used (`experiments/results/dash_comb_probe_20260707/lines_and_comb_fit.json`,
deterministic, GT-derived, checkpoint-independent).

**Band-resolved prediction:** δ_along (dominant-slot dash period in image px) per forward band:
110 / 19.8 / 5.4 / 1.9 / 0.39 px (bands 4–10 / 10–20 / 20–35 / 35–55 / 55+ m). The far bands
(δ_along ≲ 2 px) are at/below the render+R Nyquist pitch → the law predicts NO τ-resolution there
at any τ (R-limited branch). If a τ-crossover exists it must appear FIRST in the near bands
(δ_along ≫ px pitch). Band4 (>55 m) is expected homogenized at all checkpoints regardless.

### Measurement points (τ from the run's OWN record — checkpoint cfg cross-checked vs run.log telemetry)

| checkpoint (read-only; snapshotted to out-dir first) | epoch | τ (ckpt `__cfg_softmax_temp`) | stage |
|---|---|---|---|
| `levelset_ckpt_stageCE_ep299.npz` | 299 | 0.80625 | CE end |
| `levelset_witness_ema_BEST.npz` | 650 | 0.30982 | tau-best — **REUSED from dash-comb c1 arm** (same GT cache + machinery; not re-measured) |
| `levelset_ckpt_stageMuonStart_ep726.npz` | 726 | 0.21682 | Muon start |
| `levelset_witness_ema_mlx.npz` (snapshot) | 925 | 0.21569 | live EMA at snapshot time |
| final/ep1000 EMA if the run finishes mid-probe | — | — | added as 5th point if available |

Telemetry cross-check: run.log `loss_terms` rows carry `softmax_temp`; at ep925–929 the log reads
0.2157, matching the ema_mlx ckpt cfg 0.21569 — the ckpt cfg IS the run's own annealed record
(never hand-derived from assumed anneal shapes).

Note on the ep650 reuse row: the dash-comb verdict table's "+0.00161" was the solid band's ADDED
gap-FP (c2 0.00174 − c1 0.00013); the witness-alone (c1) row this probe reuses is
gap-FP **0.000135**, d_seg **0.003146**, recall **0.7795**, lane-FP 0.000595, lane-FN 0.001243.
c1 gap/mark rates for the H index are re-derivable only per-pair; the ep650 H row is therefore
measured by this probe ONLY if budget allows a re-render — otherwise the ep650 row carries the
reused scalar metrics and H is reported for the other checkpoints (whichever happened is stated
in results). Plan default: re-render ep650 too (4 measured points share one machinery pass and
the c1 scalars double as the internal positive control).

### Metrics per checkpoint (n600, witness-alone render, NO band compositing)

d_seg (total), lane recall, lane FP, lane FN, dash-gap FP (same normalization as the dash-comb
probe: gap-region realized-lane FP / total px), r_mark, r_gap, H, all of the latter three also
per forward band. Apparatus: thin wrapper `tools/tau_crossover_trainflow_probe_n600.py` importing
the dash-comb probe's Renderer + verdict path (op-for-op the canonical torch inflate primitives;
no duplicated render code). Chunked resumable foreground (atomic tmp+replace state), verdict
batches 6 ≤ 12, free-RAM ≥ 20 GiB gate before each chunk, LIVE RUN READ-ONLY.

### Verdict vocabulary (pre-registered)

SUPPORTS-τ-crossover / SUPPORTS-R-Nyquist-bound / INDETERMINATE-at-this-resolution — each with
the concrete numbers that drove it. n=4–5 points, coarse: NO curve fitting; raw table + the
pre-registered shape/contrast reading only.

---

## RESULTS (measured after pre-registration; every number below is from the probe's n600 JSON)

### Raw table (n600 each; [macOS-CPU advisory] NON-PROMOTABLE; pointer 0.19110 UNMOVED)

| epoch | τ | gap_FP | lane_recall | d_seg_total | lane_FP | r_mark | r_gap | H = r_gap/r_mark |
|---|---|---|---|---|---|---|---|---|
| 299 (CE end) | 0.80625 | 0.000191 | 0.6331 | 0.004594 | 0.000665 | 0.4113 | 0.2739 | **0.6661** |
| 650 (tau-best EMA) | 0.30982 | 0.000135 | 0.7795 | 0.003146 | 0.000595 | 0.4704 | 0.3167 | **0.6733** |
| 726 (Muon start) | 0.21682 | 0.000124 | 0.7847 | 0.003033 | 0.000568 | 0.4715 | 0.3167 | **0.6716** |
| 925 (live EMA snapshot) | 0.21569 | 0.000190 | 0.7494 | 0.003867 | 0.000776 | 0.4654 | 0.3152 | **0.6774** |

Internal positive control REGISTERED: the ep650 row reproduces the dash-comb c1 arm bit-for-bit
(d_seg 0.003146 / recall 0.7795 / gap_FP 0.000135) — same GT cache, same machinery, independent
re-render. (ep925 live-EMA is WORSE than ep650/726 on d_seg/recall — the mid-Muon EMA state at
snapshot time, amplitude-level only; the run was still training, telemetry ~ep929.)

Per-band H (bands 4–10 / 10–20 / 20–35 / 35–55 m; δ_along 110 / 19.8 / 5.4 / 1.9 px; band4 >55 m
has NO gap pixels by design — comb gate-off beyond `dash_forward_max_m=55` — so H is undefined there):

| epoch | τ | H band0 (110 px) | H band1 (19.8 px) | H band2 (5.4 px) | H band3 (1.9 px) |
|---|---|---|---|---|---|
| 299 | 0.80625 | 0.668 | 0.559 | 0.975 | 0.830 |
| 650 | 0.30982 | 0.664 | 0.555 | 1.011 | 0.923 |
| 726 | 0.21682 | 0.663 | 0.550 | 1.015 | 0.904 |
| 925 | 0.21569 | 0.671 | 0.558 | 1.018 | 0.905 |

### Pre-registered shape/contrast reading

1. **H is FLAT — in aggregate AND in every band — across the entire reached τ range.** Aggregate
   H sits at 0.666–0.677 from τ=0.806 down to τ=0.216 (a 3.7× anneal); per-band variation across
   checkpoints is < 0.02 in bands 0–1 and < 0.10 in bands 2–3 with NO monotone-in-τ trend. The
   amplitude-normalized dash contrast NEVER improves as τ anneals.
2. **What DOES move is amplitude, not structure.** r_mark grows 0.411→0.471 (ep299→726) with
   recall 0.633→0.785, and r_gap grows in LOCKSTEP (0.274→0.317) — the field paints more lane
   everywhere, marks and gaps alike. Raw gap_FP consequently tracks overall lane amplitude/error
   (0.000191→0.000124→back to 0.000190 at the regressed ep925 EMA), exactly the amplitude
   confound the H index was pre-registered to remove.
3. **Mid/far bands are FULLY homogenized at all τ.** Band2 (δ_along 5.4 px) H ≈ 0.98–1.02 and
   band3 (1.9 px) H ≈ 0.83–0.92 at every checkpoint: at/below the render+R pixel pitch the field
   paints gaps at (essentially) the mark rate — the pure R-Nyquist bound, τ-independent, as the
   law's R-limited branch predicts.
4. **Near bands hold PARTIAL dash contrast that is frozen from CE-stage onward.** Band0/band1
   (δ_along 110/19.8 px ≫ pixel pitch, so R-Nyquist does NOT bind there) sit at H ≈ 0.66/0.55
   already at ep299 and DO NOT move through the anneal — partial dash structure exists but its
   contrast is pinned; 626 further epochs and a 3.7× τ drop release nothing. This is the pinning
   /zero-effective-mobility face of the law rather than a τ-crossover.

### VERDICT

**SUPPORTS-R-Nyquist-bound (mid/far bands) + pinned-interface (near bands); NO τ-crossover
observed anywhere in the reached range τ ∈ [0.216, 0.806].** Driving numbers: aggregate H flat
0.666→0.677 across the full anneal; band2/3 H ≈ 1.0/0.9 at all τ (full homogenization at/below
pixel pitch); band0/1 H frozen at 0.66/0.55 despite δ_along ≫ pitch; gap_FP moves only with
amplitude (0.000191↔0.000124↔0.000190, tracking recall/d_seg, not τ). The τ-crossover branch of
`dash_erasure_homogenization_v1` is NOT registered by this trajectory: within everything this run
ever reached, dash contrast is τ-INSENSITIVE, and the operational consequence is the strong form
of the law's repair claim — **no reachable τ-anneal buys dash resolution; the corrector-class
lever (#287 comb, IN-TRAINING form `n287_dash_comb`) is the only live repair path**, consistent
with (and strengthening) the dash-comb probe's corrector verdict. The τ_end coupling rule keeps
its homogenization meaning, with the refinement that within [0.216, 0.806] no crossover exists
to couple against — a crossover below τ=0.216 (or above 0.806) is NOT excluded by this data.

### What was NOT verified

- **The deconfound limit as pre-registered:** epoch and τ co-vary on this single trajectory. For
  a FLAT result the confound is less corrosive (there is no improvement to mis-attribute), but a
  τ-crossover hiding OUTSIDE the sampled range, or one masked by simultaneous capacity effects,
  is not excluded. The definitive **fixed-τ control arm** (identical seed/config; τ frozen at
  0.8 vs annealed) remains the next-run design item (council draft §19).
- **No final/ep1000 row:** at completion the live run (pid 97677) was still training (~ep929+),
  no final checkpoint existed. The probe state is resumable: re-invoke with `--add-final` once a
  final EMA lands to add the 5th point.
- τ→px mapping is ORDINAL (τ in logit units; px smoothing scale depends on |∇φ|). The reading
  uses band ordering + flatness, not an absolute τ_c.
- The ep726→925 "quasi-fixed-τ" contrast conflates fixed-τ with the AdamW→Muon optimizer change;
  with H flat everywhere this contrast carried no load in the verdict.
- Band2 H marginally >1 (1.01–1.02): R-blur spreads lane paint from thin marks into adjacent
  gaps at that scale; treated as saturation (H ≈ 1), not signal.

**Artifacts:** JSON table
`experiments/results/tau_crossover_trainflow_20260707/tau_crossover_trainflow_n600_20260707.json`
(gitignored, rebuildable via the committed tool `tools/tau_crossover_trainflow_probe_n600.py`);
resumable probe state beside it. Peak RSS 5.7 GiB (far under the governor band); free RAM ≥ 20 GiB
gate honored per invocation; live run untouched (read-only snapshots; pid 97677 alive throughout;
12 chunked foreground invocations, one harness kill (exit 144) absorbed by the resume spine).

means ≠ ends: pointer 0.19110 moves only via `upstream/evaluate.py` on exact archive bytes.
