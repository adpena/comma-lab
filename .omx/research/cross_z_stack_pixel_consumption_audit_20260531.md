# Cross-Z-stack pixel-consumption audit (WAVE-1 Subagent D)

**Date:** 2026-05-31 · **Lane:** `lane_cross_z_stack_pixel_consumption_audit_20260531`
**Axis:** `[macOS-CPU advisory]` — structural-consumption diagnostic, NEVER a contest
score (Catalog #127/#192/#323/#341). $0 MLX-LOCAL, READ-ONLY on archives + substrate
source (sister C owns DreamerV3 source; I only imported + ran it).

## The question this audit answers

Sister B (commit `182b88406`) discovered Z8's contest archive is a classical
Mallat-wavelet + Wyner-Ziv **codec**: the `wavelet_blob` drives pixels but the
TRAINED categorical-posterior HNeRV renderer's `decoder_state_dict` packs as
PARSE_GUARD_ONLY — inflate reconstructs purely from the wavelet inverse and
**ignores the trained decoder weights**. So a "Z8 ratification" would score the
CODEC, not the hierarchical-PC training that solved pose 99.94%.

**Open question:** do the OTHER Z-stack substrates have the same trap? If they pack
placeholder / parse-guard decoder weights, ratifying them buys anchors that
validate codecs, not the predictive-coding paradigm — a phantom-provenance-adjacent
waste of the ≤$20 funded-ratification budget (Catalog #321/#322 spirit).

## Methodology (NO FAKE IMPLEMENTATIONS)

The probe (`tools/probe_substrate_archive_pixel_consumption.py`) is a generalized
per-substrate Catalog #272 byte-mutation distinguishing-feature probe with a
per-substrate adapter registry. For each renderer-family substrate it:

1. Parses the REAL built archive into the substrate's decoder `state_dict`.
2. Renders the first K pairs at the substrate's NATIVE scorer resolution via the
   substrate's REAL inflate reconstruct path (`model(idx)` / `reconstruct_pair`).
3. **Sweeps a perturbation across EVERY decoder tensor** (1.0× std) and re-renders.
4. Classifies on the ACTUAL pixel delta + the base reconstruction's spatial
   variance.

**Why perturb the decoded WEIGHTS, not the raw bytes:** trained decoder blobs are
brotli/zlib-compressed. A raw byte flip in a compressed blob ALWAYS corrupts the
stream → parse error (structural consumption per Catalog #105) and CANNOT
distinguish "weights drive pixels" from "weights are placeholder zeros silently
dropped by `load_state_dict(strict=False)`". Perturbing the *decoded* weights and
re-rendering is the faithful test. (B's Z8 probe is re-used verbatim for Z8.)

**Why sweep ALL tensors:** perturbing only the largest-NORM tensor MISSES real
consumption when that tensor has low pixel-influence — this is exactly what bit Z7
(largest-norm `initial_proj.weight` → 5.96e-08, but `blocks.12.weight` → 1.88e-04).
The sweep + max is the faithful "are ANY of these weights consumed?" test. A
dedicated NO-FAKE test (`test_sweep_finds_the_consumed_tensor_not_the_largest_norm`)
locks this in.

**Verdict vocabulary:**
- `PIXEL_CONSUMED` — perturbing decoder weights moves pixels AND base recon has
  spatial structure → trained weights drive contest pixels → ratification scores
  the TRAINING. **Ratification-ready.**
- `PIXEL_CONSUMED_BUT_NEAR_UNTRAINED` — weights wired (pixels move) BUT base recon
  is near-constant flat-gray (mock-teacher / underconverged) → ratifying scores a
  degenerate reconstruction, NOT validated training. **Needs a converged
  real-teacher re-train first.**
- `CODEC_DRIVEN` — pixels come from a classical codec blob; no trained-renderer
  pixel slot → ratifying buys a CODEC anchor, not a PC-training anchor.
- `PLACEHOLDER_OR_PARSE_GUARD` — decoder section exists but produces zero pixel
  delta (the Z8 `decoder_blob` trap surfaced via the renderer path).
- `NOT_BUILT` / `ADAPTER_NOT_IMPLEMENTED` — reported honestly, never fabricated.

## Per-substrate audit table

| substrate | archive on disk? | magic | trained-weight section | base_var | max Δpixel | PIXEL_CONSUMED? | a ratification would validate | archive-grammar revision needed? |
|---|---|---|---|---|---|---|---|---|
| **z6_v2_cargo_cult_unwind** (canonical 29650ep MLX) | YES (569 KB) | `Z6V2` | FiLM Rao-Ballard decoder (76 tensors) | **0.116** | **0.464** | **YES — PIXEL_CONSUMED** | **PC/renderer TRAINING** | NO |
| **time_traveler_l5_z5** (Rao-Ballard hinton-distilled long) | YES | `Z5RB` | 2-level Rao-Ballard decoder (18 tensors) | **0.127** | **0.621** | **YES — PIXEL_CONSUMED** | **PC TRAINING (predictive coding)** | NO |
| **dreamer_v3_rssm** (driver_validate; READ-ONLY) | YES (501 KB zip) | `RSSC` | categorical decoder (30 tensors) | **7.80** ([0,255]) | **245.5** | **YES — PIXEL_CONSUMED** | **PC TRAINING (categorical RSSM)** | NO |
| **z4_cooperative_receiver_loss** (v2 smoke 2026-05-15) | YES | `Z4CR` | cooperative-receiver decoder (12 tensors) | 0.0225 | 0.040 | **YES — PIXEL_CONSUMED** (weakly trained smoke) | PC TRAINING (Atick-Redlich coop-receiver) — but archive is a stale smoke | NO (rebuild a converged archive recommended) |
| **time_traveler_l5_z7_mamba2** (wave_n11 stabilizer) | YES (1.28 MB) | `Z7M2` | Z6-compat decoder (12 tensors) | **2.84e-05** | 1.88e-04 | **NO — NEAR_UNTRAINED** (mock-teacher; flat-gray) | the DECODER WIRING, NOT a converged result | NO — needs a **converged real-teacher re-train** before ratification is meaningful |
| **z8_hierarchical_predictive_coding** (M10 smoke) | YES (92 KB) | `Z8HPC1` | trained HNeRV decoder = `decoder_blob` | n/a (codec) | wavelet=1.0 | **NO — CODEC_DRIVEN** | the CLASSICAL Mallat WAVELET codec, NOT the trained renderer | **YES — wire trained decoder weights into an inflate-side consumer** (currently a research-substrate trap if naively packed) |
| **nscs06_v8_chroma_lut** (overnight v phase2) | YES (187 B) | `CH08` | none (grayscale + chroma-LUT) | n/a (codec) | n/a | **NO — CODEC_DRIVEN** | the CLASSICAL chroma-LUT codec | N/A (codec by design) |
| **nscs06_v8_path_b_wavelet** (Modal T4) | YES (Modal) | wavelet | none (wavelet + Wyner-Ziv) | n/a (codec) | n/a | **NO — CODEC_DRIVEN** | the CLASSICAL wavelet + Wyner-Ziv codec | N/A (codec by design) |

Per-substrate proof JSONs:
`experiments/results/cross_z_stack_pixel_consumption_audit_20260531/*.json`.

### Empirical detail confirming the methodology is faithful (not a constant)

- **base_var correlates with consumption strength** across the renderer family:
  Z5 (0.127, Δ0.62) ≈ Z6 (0.116, Δ0.46) > Z4 smoke (0.0225, Δ0.04) >> Z7 mock
  (2.8e-05, Δ1.9e-4). Dreamer is [0,255]-range so its 7.80/245 are on a different
  scale but clearly structured.
- **Z7 is the decisive disambiguation:** the wave_n11 stabilizer archive's decoder
  IS wired (perturbing `blocks.12.weight` moves pixels) but the base recon is
  flat-gray (mid-0.5) — the archive used a **mock-scorer-teacher** (per the
  Z7 wave_n11 landing memo: "used mock-scorer-teacher so pose=0; stability surface
  validated"). Ratifying it scores a gray reconstruction, not the Mamba-2 PC.
- **Z8 re-confirmed:** delegating to B's canonical Z8 probe → `decoder_blob`
  PARSE_GUARD_ONLY, `wavelet_blob` PIXEL_CONSUMED. Codec-driven.

## Ratification-readiness gate (the budget allocation)

**Ratification-ready arms (trained weights pixel-consumed, base recon structured):**
1. `z6_v2_cargo_cult_unwind` — canonical 29650ep MLX-LOCAL run (the strongest
   anchor: known-trained renderer, base_var 0.116, Δpixel 0.46).
2. `time_traveler_l5_z5` — Rao-Ballard hinton-distilled long (base_var 0.127,
   Δpixel 0.62 — the strongest pixel-consumption of all; the canonical 2-level
   predictive-coding arm).
3. `dreamer_v3_rssm` — driver_validate archive (READ-ONLY; categorical RSSM
   decoder strongly consumed). NOTE: sister C is actively iterating DreamerV3 —
   ratify the archive C blesses as final, not this driver_validate snapshot.

**Need archive-grammar revision / re-train BEFORE ratification is meaningful:**
- `z8_hierarchical_predictive_coding` — the contest archive is a CLASSICAL wavelet
  codec; the trained categorical-posterior HNeRV renderer weights are NOT
  inflate-consumed. Ratifying scores the wavelet codec. To validate the
  hierarchical-PC TRAINING, an inflate-side decoder-weight consumer must be wired
  (B's `export_z8_hier_pc_mlx_to_pytorch_state_dict.py` is the bridge; the
  inflate-side consumer is the open Catalog #220 work). **Do NOT spend the funded
  budget ratifying Z8 as a PC-training anchor — it would buy a codec anchor.**
- `time_traveler_l5_z7_mamba2` — needs a CONVERGED real-teacher re-train (the
  wave_n11 archive is a mock-teacher stability surface). Z7 real-Hinton runs exist
  (`z7_real_hinton_smoke_probe`) but are smokes (also flat-gray, base_var ~0). A
  proper 600pair+ real-teacher converged Z7 archive must land before ratification.
- `z4_cooperative_receiver_loss` — pixel-consumed but the only archive on disk is a
  2026-05-15 v2 smoke (weakly trained, base_var 0.0225). The Z4 Atick-Redlich L1
  scaffold landed TODAY is `research_only` with no converged archive. A converged
  Z4 archive is the prerequisite.

**Codec substrates (ratify validates the codec, by design):** nscs06_v8_chroma_lut
+ nscs06_v8_path_b_wavelet are deliberately classical codecs — they are legitimate
contest archives but a ratification scores the codec engineering, NOT a
predictive-coding paradigm. They are valid frontier arms on their own terms; they
are simply not "PC-training validation" spend.

## Recommended ≤$20 funded-ratification spend allocation

Per Catalog #246 (paired CPU+CUDA), each contest-faithful ratification arm =
~$0.30–$1.00 (paired Modal T4 smoke-then-full). The READY arms validate the
predictive-coding paradigm. Recommended allocation (~$3–6, well within $20):

| arm | spend | what it buys |
|---|---|---|
| **z6_v2 canonical 29650ep** | ~$1 paired | the strongest known-trained renderer anchor; validates the FiLM Rao-Ballard binding |
| **z5 Rao-Ballard hinton-distilled** | ~$1 paired | the canonical 2-level predictive-coding anchor (strongest pixel-consumption) |
| **dreamer_v3_rssm** (C's final archive) | ~$1 paired | categorical RSSM PC anchor — **coordinate with sister C** to ratify the archive she blesses, not the stale driver_validate snapshot |

Reserve the remaining budget (~$14–17) for the NON-ready arms ONLY AFTER their
prerequisites land: a converged real-teacher Z7-Mamba-2 archive, a converged Z4
Atick-Redlich archive, and (the highest-value structural work) a Z8 inflate-side
decoder-weight consumer that makes the hierarchical-PC training contest-eligible.

**Do NOT** spend the funded budget ratifying Z7-stabilizer, Z4-smoke, or Z8-as-PC
— those buy degenerate-gray / weakly-trained / codec anchors respectively
(phantom-provenance-adjacent waste per Catalog #321/#322).

## 6-hook wire-in declaration (Catalog #125)

- **hook #1 sensitivity-map** = N/A (defensive structural-consumption diagnostic;
  emits no per-axis score contribution).
- **hook #2 Pareto constraint** = N/A.
- **hook #3 bit-allocator** = N/A.
- **hook #4 cathedral autopilot dispatch** = **ACTIVE** — the per-substrate
  `ratification_ready_for_pc_training` verdict is the canonical gate that prevents
  the funded-ratification ranker from spending paid GPU ratifying codec-driven or
  near-untrained archives as if they were PC-training anchors. The audit JSONs are
  machine-readable for the autopilot ranker.
- **hook #5 continual-learning posterior** = N/A (advisory diagnostic; no posterior
  write — but the verdicts inform the next dispatch-ranking cycle).
- **hook #6 probe-disambiguator** = **ACTIVE** — this probe IS the canonical
  disambiguator between "trained weights drive pixels (PC-training)" vs
  "classical codec drives pixels (codec)" vs "weights wired but untrained
  (degenerate)" at the per-substrate archive surface.

## Highest-EV next step

Ratify **z5 Rao-Ballard hinton-distilled** (the strongest pixel-consumption of all
the PC arms, Δpixel 0.62, base_var 0.127) paired CPU+CUDA per Catalog #246 — it is
the cleanest predictive-coding-paradigm validation anchor and is ratification-ready
right now. Pair it with z6_v2 canonical as the known-trained-renderer baseline.
