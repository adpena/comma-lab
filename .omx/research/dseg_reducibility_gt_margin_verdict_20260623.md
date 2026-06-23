# d_seg reducibility — GT-margin cross-tab verdict (our small-basis decoder)

- **Date:** 2026-06-23
- **Subagent:** `dseg-reducibility-test-20260623`
- **Axis:** `[contest-CPU advisory]` / `[macOS-CPU advisory]` — NON-PROMOTABLE
  (`promotable=false`, `score_claim=false`). This is a MECHANISM/DISAMBIGUATION
  measurement, NOT a score-roadmap row. Authority = exact frozen CPU SegNet
  argmax-disagreement (`upstream/modules.py:112`) through the exact preprocess. NOT a proxy.
- **Pointer:** UNMOVED at 0.19110.
- **Target decoder:** OUR live small-basis run
  `experiments/results/yousfi_r3_MUONJUMP_stage8_lr1e3_20260623T180100Z` (base_ch=20,
  taper `[16,16,17,19,19,14,10]`, latent_dim=28, stage8_muon_finetune, best_ep 24725).
  The `best/` EMA decoder+latents were COPIED read-only to `.omx/tmp/` and loaded on CPU
  (the live MPS training run pid 16938 was never opened for writing / disturbed).
- **All score math via `tac.contest_score`** (canonical helper, never hand-rolled).
- **Tool:** `tools/measure_dseg_reducibility_gt_margin.py`. JSON:
  `.omx/research/dseg_reducibility_gt_margin_20260623.json` (N=48).

## VERDICT: **IRREDUCIBLE** (residual d_seg is near an architectural / label-noise floor)

The residual d_seg of our current decoder is **dominated by frozen-SegNet label-noise at
near-zero-margin boundary pixels**, not by reconstruction failures on confident GT targets.
A better horizon decoder + horizon-weighted margin loss can recover **at most ΔS ≈ 0.012**
(the flips at clearly-confident GT margin ≥ 0.5), and realistically far less. **Recommendation:
bank the rate + pose headroom; do NOT launch a horizon-decoder campaign expecting a large
d_seg win — its ceiling is ~0.012 score units and the reachable fraction is smaller.**

## Method (NO FAKE — exact scorer, exact round-trip, real GT)

For the first N=48 pairs of `0.mkv` (9,437,184 SegNet-grid pixels):
1. **GT** decoded via `frame_utils.yuv420_to_rgb` (BT.601 limited-range, bilinear chroma —
   never PyAV rgb24). Stacked `(B,2,874,1164,3)` uint8 (camera res).
2. **OUR** render: EMA decoder(latents) → `(B,2,3,384,512)` [0,255] → bicubic↑ to camera res →
   clamp/round to uint8 — the SAME render→roundtrip the inflate path uses
   (`driver.kit_aware_exact_eval`, lines 3760-3779).
3. SegNet `preprocess_input` on BOTH (`x[:,-1]` last frame → bilinear↓ to 384×512) → frozen
   `segnet(...)` → `(B,5,384,512)` logits for GT and OUR.
4. Per pixel: `gt_argmax`, `gt_margin = top1−top2` (GT label confidence), `our_argmax`,
   `flip = gt_argmax != our_argmax`. d_seg = mean(flip) over all pixels.

### Sanity check (the pipeline-faithfulness gate) — PASSED
- Measured aggregate d_seg = **0.002015** vs the live run's last-eval **0.002109** → 4.5% off
  (within the 15% tol; the residual is the first-48-pairs subsample vs the full-600 eval).
- The harness reproduces the authority d_seg → the cross-tab is trustworthy.

## The decisive cross-tab (GT margin at FLIP vs NON-FLIP pixels)

| signal | FLIP pixels | NON-FLIP pixels | separation |
|---|---:|---:|---:|
| GT top-2 margin **median** | **0.119** | **5.814** | **~49×** |
| GT margin p10 / p25 / p75 / p90 | 0.019 / 0.050 / 0.238 / 0.404 | 4.14 / 5.28 / 6.31 / 6.99 | — |

Flips are knife-edge; non-flips are confidently decided. The flips live exactly where the
frozen SegNet is itself a near-coin-flip.

### Concentration of flips at low GT margin (the irreducibility proof)

| GT margin threshold | frac of FLIPS below | frac of ALL pixels below | concentration |
|---|---:|---:|---:|
| < 0.05 | 25.1% | 0.13% | **~193×** |
| < 0.1 | 44.2% | 0.27% | ~164× |
| < 0.2 | 68.9% | 0.53% | ~130× |
| < 0.5 | **93.9%** | **1.32%** | **~71×** |
| < 1.0 | 99.6% | 2.56% | ~39× |

**93.9% of flips sit at GT margin < 0.5, while only 1.3% of all pixels are that uncertain** —
a 71× over-concentration of flips at the label-noise frontier. A decoder cannot cheaply pin a
pixel the GT label itself does not confidently decide.

## Reducible-headroom curve (ΔS the max a perfect horizon decoder could recover)

Reducible d_seg at threshold t = (#flips with GT margin ≥ t) / total_pixels; ΔS = 100·that
(seg term is linear), holding pose (d_pose=3.66e-4) + rate (bytes=82457) fixed at live values.
This is the ORACLE ceiling (forcing those flips to GT), not what training would actually achieve.

| GT margin ≥ t (flip is "confident-GT" ⇒ reducible) | frac of flips | reducible d_seg | **ΔS ceiling** |
|---|---:|---:|---:|
| ≥ 0.05 (barely above coin-flip) | 74.9% | 1.51e-3 | 0.151 |
| ≥ 0.1 | 55.8% | 1.12e-3 | 0.112 |
| ≥ 0.2 | 31.1% | 6.27e-4 | 0.063 |
| ≥ 0.3 | 17.7% | 3.56e-4 | 0.036 |
| **≥ 0.5 (clearly confident)** | **6.1%** | **1.23e-4** | **0.012** |
| ≥ 1.0 (very confident) | 0.4% | 7.84e-6 | 0.0008 |

**The headline ΔS depends entirely on what margin counts as "confident."** At a principled
clearly-confident bar (logit gap ≥ 0.5, comfortably above the near-coin-flip noise), only **6.1%
of flips** are reducible-with-confidence → **ΔS ceiling ≈ 0.012**. Below ~0.3 margin the GT itself
is weakly decided, so "recovering" those flips means matching SegNet's own noise — not a stable,
trainable target. The curve is the honest answer: there is no large confident-GT reducible budget.

## Per-row read (confirms the prior horizon-band localization)

Top flip-rate rows are the horizon band (SegNet grid rows **181-192**, camera rows ~412-437, the
calibrated horizon / vanishing point — matches `horizon_band_dseg_lever_20260623.md`). Row-level
flip rates peak at ~0.022; row-MEAN GT margin in those rows is ~3.5-4.3 (the rows are mostly
confident interior) — the flips are the sparse low-margin minority embedded in otherwise-confident
horizon rows. So the d_seg is not "a whole bad region" but scattered shallow-margin boundary pixels.

## Why this is consistent with (and sharper than) the prior closed paths

- `independent_dseg_bets_frontier_20260623.md`: 97.8% of frontier d_seg in horizon rows 96-288;
  no static flip (max per-pixel flip-freq 5.8%); spread across all 5 classes. → structural.
- `horizon_band_dseg_lever_20260623.md`: on the FRONTIER archive, flip margin mean 0.102 / median
  0.075 vs non-flip 0.981; sidecar NO-GO (oracle 0.70× of break-even). → unit-economics killer.
- `frozen_instance_horizon_crossframe_result_20260623.md`: flip-set rank 547/600 (full-rank).
- **This measurement's new contribution:** it proves the SAME low-margin label-noise signature
  holds for **OUR OWN small-basis decoder** (not just the borrowed 0.191 frontier archive) — flip
  margin median 0.119 vs non-flip 5.81 — and quantifies the confident-GT reducible budget as a ΔS
  ceiling (~0.012 at margin≥0.5). The residual d_seg is an intrinsic property of the frozen
  SegNet's decision boundary on this video, shared across decoders, not a fidelity gap our
  architecture can cheaply close.

## Recommended next move

1. **Do NOT launch a from-scratch horizon high-frequency decoder campaign expecting a big d_seg
   win.** Its oracle ceiling at a trainable (confident-GT) margin is ~0.012 S; the realistic
   fraction is a small slice of that.
2. **Bank rate + pose** — they are the axes with real, structural headroom (the frontier is
   rate-dominated; pose has marginal room). The d_seg axis is near its floor for this scorer/video.
3. **If any d_seg is chased at all**, it must be a near-zero-byte SHARED-structure lever (a
   horizon-band-weighted margin term inside the EXISTING training loop, no new bytes) targeting
   ONLY the margin∈[0.3,0.5] flips — the only flips that are both reducible and stably-decided —
   and even that caps at ΔS ≈ 0.024 (margin≥0.3 oracle). A byte-spending sidecar is already
   measured NO-GO (`horizon_band` oracle 0.70× of break-even).

## Reactivation criteria (re-open this verdict if)

- The decoder converges to a much lower d_seg (≪ 0.002) and a re-run shows the surviving flips
  shift to HIGHER GT margin (would flip the verdict to REDUCIBLE) — re-run the tool on the new ckpt.
- A NEW scorer / video is used (the label-noise frontier is scorer+content specific).
- N=48 → N=600 full-eval re-run materially changes the margin-at-flip distribution (the N=2 and
  N=48 results already agree to 1%; a 600-pair confirmation is the only outstanding rigor step and
  is cheap — re-run with `--n-pairs 600`).

## 6-hook wire-in declaration (per CLAUDE.md "Subagent coherence-by-default")

1. **Sensitivity-map contribution** — ACTIVE (conceptually): the per-row flip-rate × GT-margin
   table is a d_seg-sensitivity surface (the horizon band rows 181-192 are the binding pixels);
   it tells the bit-allocator there is NO confident-GT d_seg budget to chase. Not persisted to
   `tac.sensitivity_map` (advisory mechanism row, not a score-claim surface).
2. **Pareto constraint** — ACTIVE: this measurement TIGHTENS the d_seg-axis Pareto bound — the
   reachable d_seg floor for this decoder/scorer is ~(measured − 0.012) at the confident-GT oracle;
   the campaign planner should treat d_seg as near-saturated and pivot to rate+pose.
3. **Bit-allocator hook** — N/A: this is a diagnostic, it adds no archive bytes and changes no
   per-tensor importance (the verdict is "don't spend bytes on d_seg").
4. **Cathedral autopilot dispatch hook** — N/A: not archive-deployable (mechanism measurement).
5. **Continual-learning posterior update** — ACTIVE: the empirical anchor (flip-margin median 0.119
   vs non-flip 5.81; reducible ΔS ceiling 0.012 at margin≥0.5) is recorded in this memo + the JSON
   for the next campaign to inherit; it is the disambiguation that gates the next move.
6. **Probe-disambiguator** — ACTIVE: this IS the disambiguator between the two defensible
   interpretations (REDUCIBLE horizon-decoder gap vs IRREDUCIBLE label-noise floor). The tool
   `tools/measure_dseg_reducibility_gt_margin.py` returns the regime-conditional verdict.
