# UNIWARD cost-map → bit_allocator.per_byte → NSCS06 v8 GRAYSCALE STREAM — LANDED (WAVE-5B retarget of orphan-loop #1570)

- **Lane:** `lane_uniward_grayscale_stream_bit_allocation_20260531`
- **Status:** LANDED — the #1570 wire is retargeted from the 240-byte chroma LUT
  to the DOMINANT grayscale byte surface. Verdict is an HONEST NEGATIVE
  (IMPLEMENTATION-LEVEL falsified per Catalog #307); the apparatus is closed.
- **Date:** 2026-05-31
- **Mission contribution (Catalog #300):** `frontier_breaking` (tested the
  highest-EV reactivation path the #1570 landing memo itself recommended — apply
  UNIWARD bit-allocation to the dominant grayscale stream rather than the tiny LUT
  — and produced a decisive, mechanistically-explained falsification that prunes
  a whole class of "spatial UNIWARD allocation on entropy-coded rasters" lanes).
- **Horizon class (Catalog #309):** `plateau_adjacent` (NSCS06 v8 chroma path).
- **Paid spend:** $0 (numpy/torch-CPU on real `upstream/videos/0.mkv` frames; zero GPU).
- **Evidence grade:** `[macOS-CPU advisory]` / `[macOS-MLX research-signal]` —
  non-promotable per Catalog #192/#317/#341/#127/#323.

## What this is (the #1570 memo's own next step, verbatim)

The sister `bit_allocation_per_lut_byte.py` closed orphan-loop #1570 on the
240-byte chroma LUT but the HONEST VERDICT was that the LUT is too small
(max |ΔS_rate| ≈ 2.8e-5, two orders of magnitude BELOW the contest noise floor).
The #1570 landing memo's reactivation path (b) is verbatim:

> "apply the UNIWARD bit-allocation to a LARGER entropy-coded surface than the
>  240-byte chroma LUT (e.g. the grayscale stream which dominates the v8
>  archive bytes)."

This lane does exactly that. **Only the target surface changes**; the canonical
`tac.bit_allocator.per_byte.allocate_per_byte` allocator + `tac.uniward_delta`
cost map + the no-op proof are REUSED.

## The grayscale stream IS the dominant surface

The CH08 `GRAYSCALE_STREAM` = `num_pairs * grayscale_h * grayscale_w` raw uint8
bytes (one byte per low-res luma cell). At realistic v8 shapes
(num_pairs=600, gh=96, gw=128) = **7,372,800 bytes** vs the 240-byte LUT — a
**30,720×** larger surface. In the smoke (24 real frames) the grayscale stream is
294,912 raw bytes vs 240 LUT bytes = **1,229× larger** even at tiny N.

## The spatial-allocation design (distinct from the LUT)

The grayscale stream is SPATIAL (byte `(p, gy, gx)` = luma of a low-res cell), as
is the UNIWARD cost map `(N, H, W)`. The wire aggregates the per-pixel UNIWARD cost
into a per-`(gy, gx)` cell sensitivity (block-averaged, then averaged over pairs),
allocates a bit budget across the `gh*gw` CELLS via the canonical allocator, and
broadcasts each cell's bit-depth across all `num_pairs` bytes at that spatial
position. This runs the dict-based allocator over `gh*gw` (~12K) cells rather than
`num_pairs*gh*gw` (~7M) bytes (which would be intractable), is spatially coherent,
and is applied to the REAL stream bytes the v8 inflate consumes.

New module:
`src/tac/substrates/uniward_per_pixel_distortion/nscs06_v8_chroma_lut_integration/bit_allocation_per_grayscale_cell.py`
(placed in the EXISTING integration package — the canonical v8↔UNIWARD home —
to avoid a duplicate namespace per Catalog #302/#340).

## Proof the cost-map actually changes the allocation (NON-FAKE, Catalog #105/#139/#220)

`allocation_diff_from_uniform_cells(uniward_bits, uniform_bits)` returned **12,288
(every cell) at every sub-full budget** — the cost-map is genuinely consumed; a
no-op wire would produce a zero diff. The CORE NO-FAKE test
(`test_uniward_and_uniform_allocate_differently_NONFAKE_PROOF`) + the end-to-end
behavioral test (`test_end_to_end_coarsens_textured_cells_keeps_smooth`) assert
this and would fail if the helper ignored the cost-map. 23 NO-FAKE tests verify
BEHAVIOR (not constants); 59 total pass in the integration package (23 new + 36
sister), 163 pass across the consumed bit_allocator + v8 substrate surfaces.

## Empirical smoke ($0, real `upstream/videos/0.mkv`, 24 frames @ 96×128, min_bits floor=2)

base grayscale stream: 294,912 raw → brotli q=11 = 114,898 bytes;
per-cell cost dyn-range 2.2× (12,288/12,288 cells nonzero); UNIWARD cost
range [0.0, 9.97].

**Matched-BUDGET sweep** (UNIWARD vs uniform at the SAME total bits):

| avg bits/cell | alloc-diff | UW brotli | UF brotli | Δ bytes | UW render-MSE | UF render-MSE | Δ MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 (full) | 0 | 114,898 | 114,898 | +0 | 0.00 | 0.00 | +0.00 |
| 6 | 12,288 | 111,508 | **47,119** | **+64,343** | 92.16 | **7.40** | +84.76 |
| 5 | 12,288 | 100,606 | 27,884 | +72,722 | 136.86 | 22.02 | +114.84 |
| 4 | 12,288 | 85,118 | 14,538 | +70,580 | 181.78 | 55.18 | +126.60 |
| 3 | 12,288 | 67,748 | 8,436 | +59,312 | 236.33 | 92.30 | +144.03 |

**UNIWARD loses on BOTH axes at every budget.** Uniform is smaller AND has better
rendered fidelity.

**Matched-FIDELITY apples-to-apples** (bytes each method needs to reach a target
rendered recon-MSE — the canonical Pareto-dominance question):

| target render-MSE | UNIWARD bytes | uniform bytes | Δ bytes | rate-axis ΔS (UW−UF) | winner |
|---:|---:|---:|---:|---:|:---|
| 20 | 115,452 | 30,545 | +84,906 | **+0.0565** | uniform |
| 40 | 116,005 | 20,647 | +95,358 | **+0.0635** | uniform |
| 60 | 114,648 | 13,745 | +100,903 | **+0.0672** | uniform |
| 80 | 112,695 | 10,457 | +102,238 | **+0.0681** | uniform |

To reach a target rendered-MSE, **uniform needs ~3.8× fewer bytes than UNIWARD**.

**Research-exhaustion variant (Forbidden premature KILL):** I also tested the
UNIWARD-ALIGNED direction (coarsen the BLIND high-cost cells MORE than uniform,
keep SENSITIVE cells) rather than TOP_K-keeps-precision. Every variant still LOST
to uniform at matched fidelity (best ΔS = +0.0016, still above zero). Both
allocation directions are dominated.

Manifest: `experiments/results/uniward_grayscale_stream_bit_allocation_smoke_20260531/smoke_output.json`.

## HONEST VERDICT (Catalog #307 — IMPLEMENTATION-LEVEL, NOT a paradigm kill)

**UNIWARD per-cell bit-allocation is the WRONG allocation shape for the
entropy-coded spatial grayscale raster, in BOTH directions, at EVERY operating
point.** The mechanism (genuine system intelligence):

1. **The surface is large enough** — unlike the 240-byte LUT (|ΔS| ~1e-5 below
   noise floor), the grayscale stream's |ΔS| ~0.05–0.07 is WELL ABOVE the contest
   noise floor. The byte surface is NOT too small. UNIWARD simply allocates in the
   WRONG DIRECTION.
2. **brotli compressibility of a spatial luma raster is dominated by GLOBAL
   bit-depth reduction, NOT spatial concentration.** Uniform low-bit-everywhere
   produces a globally low-entropy stream brotli crushes (1.5 bits/cell → 1,093
   bytes). Any non-uniform per-cell allocation leaves a high-entropy minority that
   dominates the compressed output — UNIWARD TOP_K at 1.5 avg bits is still 37,767
   bytes (34× larger than uniform's 1,093).
3. **The v8 substrate already does the right thing** — its grayscale_levels=16
   (4-bit) quantization IS uniform global bit-depth reduction. The grayscale
   stream wants MORE uniform coarsening, not spatial UNIWARD targeting.

This PRUNES a whole class of "spatial UNIWARD bit-allocation on entropy-coded
rasters" lanes: the inverse-steganalysis cost map is a per-pixel SAFE-TO-PERTURB
signal, but on a stream whose rate cost comes from global entropy (not from
per-element precision that a downstream renderer reads), spatial concentration is
counterproductive. UNIWARD is the right lever for a DIRECT-payload surface (where
each byte's precision is consumed independently, e.g. a per-pair residual or a
per-pixel translation sidechannel), NOT for a globally-entropy-coded spatial
luma raster fed through a bilinear-upsample + LUT-lookup renderer.

Per CLAUDE.md "Forbidden premature KILL" this is DEFER-pending-different-surface,
not a kill. Reactivation (none recommended on THIS surface; the negative is
robust): (a) apply UNIWARD bit-allocation to a DIRECT-payload entropy surface
where per-element precision IS the cost (per-pair residual sidecar / per-pixel
translation sidechannel) rather than a globally-entropy-coded raster; (b) if a
future v8 variant entropy-codes the grayscale stream with a CONTEXT model that
rewards spatial structure, re-test (current brotli does not).

## 6-hook wire-in declaration (Catalog #125)

- **hook #1 sensitivity-map = ACTIVE** — the per-cell inverse-UNIWARD cost IS the
  canonical spatial sensitivity surface feeding the allocator.
- **hook #2 Pareto constraint = ACTIVE** — `total_budget_bits` is the Pareto bound;
  the matched-fidelity sweep measures the rate/distortion Pareto frontier the
  verdict adjudicates (uniform Pareto-dominates UNIWARD).
- **hook #3 bit-allocator = PRIMARY** — this work IS the bit-allocator hook;
  `tac.bit_allocator.per_byte.allocate_per_byte` is the canonical allocator
  consumed (over `gh*gw` cells).
- **hook #4 cathedral autopilot dispatch = N/A** — research-signal only; no
  archive promotion (honest-negative on the score axis; no Pareto-dominant point).
- **hook #5 continual-learning posterior = ACTIVE** — the probe-outcomes ledger
  PROCEED-advisory row + this memo become the posterior anchor for the surface
  (the "spatial UNIWARD allocation is wrong for entropy-coded rasters" lesson).
- **hook #6 probe-disambiguator = ACTIVE** — this memo + the matched-fidelity sweep
  IS the disambiguator between "the #1570 LUT negative was because the surface was
  too small" (FALSIFIED — the grayscale surface IS large enough) vs "spatial
  UNIWARD allocation is the wrong lever for entropy-coded rasters regardless of
  surface size" (CONFIRMED — uniform dominates on a 30,720× larger surface too).

## Canonical equation status (Catalog #344; FORMALIZATION_PENDING — NOT registered)

`uniward_grayscale_stream_bit_allocation_savings_v1` is **proposed but NOT
registered** — registering an equation for a falsified-negative surface would be
a fake equation per CLAUDE.md "NO FAKE IMPLEMENTATIONS". The empirical finding is
the negative result; there is no positive savings law to formalize. The honest
posterior anchor is the probe-outcomes PROCEED-advisory row + this memo, NOT a
canonical equation. If a future DIRECT-payload surface (reactivation path a)
yields a Pareto-dominant point at $0, THAT surface's equation can be registered.

## Paired-CUDA op-routable (DO NOT FIRE — operator-funded, separate step)

A ~$0.06 paired-CUDA RATIFICATION (Catalog #246 paired CPU+CUDA) of a v8 archive
built with the UNIWARD-bit-allocated grayscale stream vs the uniform stream would
empirically anchor the contest score delta. **It is STRONGLY NOT recommended** —
the advisory finding is a robust honest-negative on BOTH allocation directions
across the full sweep (uniform Pareto-dominates by ~3.8× bytes at matched
fidelity); a paid anchor would only confirm UNIWARD is worse on contest-CUDA too.
Re-route the $0.06 to the DIRECT-payload reactivation path (a) BEFORE any paired
dispatch. The op-routable is surfaced for completeness; the recommendation is
DEFER the paid anchor permanently on this surface.

## NO FAKE IMPLEMENTATIONS verification (Slot EEE 5 classes)

- Class 1 (markers-without-work): AVOIDED — the allocator genuinely consumes the
  cost-map; `allocation_diff_from_uniform_cells` proves it (12,288-cell diff, not 0).
- Class 2 (tests-verify-constants): AVOIDED — 23 NO-FAKE tests verify BEHAVIOR
  (cost-map preserves textured/smooth split, sensitivity inverse-of-cost,
  block-average not subsample, alloc-diff non-empty, quantization coarsens,
  bit-depth broadcasts across pairs, end-to-end concentrates on smooth cells).
  Every test fails if the body is replaced by a no-op.
- Class 3 (synthetic-fixture): AVOIDED — the smoke runs on REAL
  `upstream/videos/0.mkv` frames (24 decoded), not toy fixtures.
- Class 4 (placeholder-string): AVOIDED — the manifest carries real measured
  numbers + canonical non-promotable Provenance markers, no `TBD`/`pending`.
- Class 5 (enum-padding): N/A — the two allocation methods (TOP_K vs UNIFORM) are
  the canonical `PerByteAllocationMethod` enum, structurally distinct.

## Bonus artifact committed alongside (separate from the UNIWARD finding)

`tools/nscs06_v8_chroma_lut_hinton_distill_600pair_long_mlx.py` — a real,
canonical NSCS06 v8 chroma_lut 600-pair MLX-LOCAL probe driven by a REAL
Hinton-distilled SegNet teacher (argmax labels drive the deterministic-LUT
derivation policy + the SegNet noise-floor probe). It is the "executable Slot 1
of the cross-paradigm pivot wave" (sister of the V3 600-pair pattern, commit
`92a39dc62`), adapted for v8's deterministic-LUT-codec paradigm: decode 600 real
`upstream/videos/0.mkv` GT pairs (Catalog #213/#114), iterate the 5 canonical
cargo-cult-unwind chroma-LUT policy arms, emit per-axis decomposition
(seg/pose/recon/archive_bytes) per checkpoint, pack both v1 (inline LUT) and v2
(procedural seed) archives + verify inflate roundtrip, and anchor canonical
equation #26 per Catalog #344. It is `$0` MLX-LOCAL, `research_only`, and
NON-PROMOTABLE `[macOS-MLX research-signal]` per Catalog #323/#192/#341. It is
DISTINCT from the UNIWARD bit-allocation finding above (different surface,
different paradigm); it is committed here only because it was a bonus artifact
the same recovery session produced. It is not executed/anchored by this landing
(it is the dispatch-ready probe driver, not a run record).

## Working-tree scope (Catalog #340 sister-disjoint)

ONLY files under
`src/tac/substrates/uniward_per_pixel_distortion/nscs06_v8_chroma_lut_integration/`
(new grayscale module + tests) + this memo + the smoke manifest + lane registry +
probe outcome. NSCS06 v8 + bit_allocator + uniward_delta are READ-ONLY
consumer-imports (unmodified). Disjoint from the in-flight z8 seg-lever subagent
(owns `z8_*`) and the z5 harvest poll (owns z5 + Modal ledger). Did NOT touch z8,
z5, or fire any cloud dispatch.

## Single highest-EV next step

Apply the EXISTING UNIWARD→allocate_per_byte wire (now proven structurally
feasible + NON-FAKE) to a **DIRECT-payload entropy surface** where per-element
precision IS the rate cost — e.g. a per-pair pose-delta residual sidecar or a
per-pixel translation sidechannel — rather than a globally-entropy-coded spatial
raster. That is the surface class where the inverse-steganalysis cost map's
"spend bits where the scorer is sensitive" intuition is dimensionally correct.
Both the 240-byte LUT (too small) and the 7.4 MB grayscale stream (wrong
allocation shape) have now empirically ruled out the chroma/luma-raster surfaces;
the residual/sidechannel surface is the untested high-EV direction.
