# UNIWARD cost-map → bit_allocator.per_byte → NSCS06 v8 chroma-LUT wire-in — LANDED (orphan-loop #1570 CLOSED)

- **Lane:** `lane_uniward_cost_map_to_bit_allocator_to_nscs06_v8_wire_in_20260530`
- **Status:** LANDED — the long-orphaned inverse-steganalysis loop is now wired END-TO-END.
- **Date:** 2026-05-31
- **Mission contribution (Catalog #300):** `frontier_breaking_enabler` (closes the
  orphan-signal loop per CLAUDE.md "Results must become system intelligence"; the
  UNIWARD cost-map now feeds the canonical `tac.bit_allocator.per_byte` allocator
  which now allocates the NSCS06 v8 chroma-LUT byte budget — three previously-disjoint
  canonical surfaces are now a single executed pipeline).
- **Horizon class (Catalog #309):** `plateau_adjacent` (NSCS06 v8 chroma path).
- **Paid spend:** $0 (numpy/torch-CPU on real `upstream/videos/0.mkv` frames; zero GPU dispatch).
- **Evidence grade:** `[macOS-CPU advisory]` / `[macOS-MLX research-signal]` — non-promotable
  per Catalog #192/#317/#341/#127/#323.

## Predecessor reconciliation (Catalog #313 + #229 + #307)

A predecessor (`uniward_resume_2_wire_in_20260531`, commit `2c9d8f3e`) reached a
HALT-CLEAN DEFER 30-day verdict on this exact lane, with TWO blocker axes:

- **Axis A — "the bit-allocator does not exist at HEAD".** This was TRUE at the
  predecessor's HEAD (before `tac.bit_allocator/per_byte.py`,
  `tac.uniward_delta.py`, and `tac.composition/` landed). It is **VOID at this
  HEAD** (`5992179b9`): all three are REAL and import cleanly. The predecessor's
  HALT was a *stale-HEAD* verdict, vindicating the operator's re-grep-everything
  directive — I re-verified every cited API via pathlib (rtk-grep is unreliable
  in this environment) and found them all present.
- **Axis B — "the chroma section is 15 fixed bytes, no variable budget".** The
  predecessor read the **WRONG substrate** —
  `nscs06_carmack_hotz_strip_everything/archive.py` (the v6 strip-everything, with
  a fixed 15-byte 5-anchor chroma). The prompt's actual target is the SEPARATE,
  newer **`src/tac/substrates/nscs06_v8_chroma_lut/`** package whose `chroma_lut`
  is a dense `(grayscale_levels=16, num_segnet_classes=5, 3) = 240`-byte uint8 LUT
  — a GENUINE variable-precision surface. The v6→v8 substrate distinction is the
  load-bearing fact that reverses Axis B.

Per CLAUDE.md "Forbidden premature KILL": the predecessor's DEFER was an
implementation-infeasible-at-stale-HEAD blocker (paradigm intact). Its
reactivation criteria #1 (allocator lands) + #2 (variable-bit grammar) + #5
(sister UNIWARD primitives) are all satisfied at this HEAD. This landing is the
canonical reactivation.

## What landed (the connective tissue)

New module
`src/tac/substrates/uniward_per_pixel_distortion/nscs06_v8_chroma_lut_integration/bit_allocation_per_lut_byte.py`
(placed in the EXISTING integration package — the canonical v8↔UNIWARD home —
to avoid a duplicate namespace per Catalog #302/#340). The sister
`lut_derivation_uniward_weighted.py` weights *which RGB value* each (level, class)
bin picks (UNIWARD-weighted median); this module is ORTHOGONAL and DISTINCT: it
allocates a *bit budget* across the 240 LUT bytes by UNIWARD sensitivity.

The END-TO-END WIRE (#1570 closure):

    real frames (upstream/videos/0.mkv)
      → tac.uniward_delta.compute_uniward_cost_map  (B,3,H,W)→(B,H,W)
         [HIGH cost = textured = scorer-BLIND = SAFE = LOW scorer sensitivity]
      → aggregate_per_pixel_uniward_weights_into_lut_bins  (16,5) per-bin weight
      → per_lut_byte_sensitivity = 1/(eps + per_bin_weight)  (240,)  [INVERSE]
      → tac.bit_allocator.per_byte.allocate_per_byte(budget, sensitivity)
         [TOP_K_BY_SENSITIVITY vs UNIFORM_BASELINE]
      → quantize_lut_by_allocation(lut, bits_per_byte, min_bits_floor=2)
      → NSCS06 v8 CH08 archive with the quantized LUT
         [inflate-time lookup_rgb_via_chroma_lut consumes EVERY LUT byte ⇒ not a no-op]

## Proof the cost-map actually changes the allocation (NON-FAKE, Catalog #105/#139/#220)

`allocation_diff_from_uniform(uniward_bits, uniform_bits)` returns the LUT byte
offsets whose bit-depth differs. On the real-frame smoke the diff was **240
(every byte) at every sub-full budget** — the cost-map is genuinely consumed; a
no-op wire would produce a zero diff. The CORE NO-FAKE test
(`test_uniward_and_uniform_allocate_differently_NONFAKE_PROOF`) asserts this and
would fail if the helper ignored the cost-map. The `test_coarsened_lut_changes_v8_render_NONFAKE`
test further proves the coarsened LUT changes the inflate-time RGB render (every
LUT byte is consumed at lookup).

## Empirical smoke ($0, real `upstream/videos/0.mkv`, 16 frames @ 96×128, min_bits floor=2)

base LUT 240B → brotli q=11 = 134B; 20/80 bins nonempty (luma-class proxy);
per-bin dynamic range 25405× (strong UNIWARD signal).

| budget (bits) | top_k | alloc-diff | UW brotli | UF brotli | Δ bytes | UW recon-MSE | UF recon-MSE | Δ MSE | rate-axis ΔS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1920 (full) | 240 | 0 | 134 | 134 | +0 | 0.00 | 0.00 | +0.00 | +0.0 |
| 1440 (75%) | 160 | 240 | 87 | 129 | **−42** | 79.94 | 0.79 | **+79.14** | −0.0000280 |
| 960 (50%) | 80 | 240 | 89 | 91 | −2 | 79.94 | 31.71 | +48.23 | −0.0000013 |
| 720 (37%) | 40 | 240 | 84 | 58 | **+26** | 79.94 | 96.83 | **−16.89** | +0.0000173 |

Manifest: `experiments/results/uniward_bit_allocator_nscs06_v8_wire_in_smoke_20260531T145851Z/smoke_output.json`.

## HONEST VERDICT (Catalog #307 — IMPLEMENTATION-LEVEL, NOT a paradigm kill)

**The wire CLOSES the orphan loop structurally but does NOT help the contest
score on this surface.** The UNIWARD sensitivity ranking and the uniform baseline
trade rate for distortion in OPPOSITE directions: at 75% budget UNIWARD saves 42
bytes but costs +79 recon-MSE; at 37% budget UNIWARD has better recon (−16.89) but
worse bytes (+26). **There is no budget where UNIWARD Pareto-dominates uniform at
matched fidelity.** Two structural reasons:

1. **The LUT is tiny.** 240 bytes → brotli-compressed to ~134B. Even the maximal
   byte delta (42B) is ΔS ≈ −0.000028 rate-axis — two orders of magnitude below
   the contest noise floor. The byte-savings surface is too small to matter.
2. **The bins are sparse.** With the SegNet-free luma-class proxy only 20/80 bins
   are nonempty, so the per-bin sensitivity ranking has little to concentrate.

This is an honest negative on the SCORE axis with a real structural win on the
APPARATUS axis (the orphan loop is closed; the three canonical surfaces are wired).
Per CLAUDE.md "Forbidden premature KILL" this is DEFER-pending-larger-surface, not
a kill. Reactivation: (a) real SegNet class labels (more nonempty bins → richer
ranking); (b) apply the UNIWARD bit-allocation to a LARGER entropy-coded surface
than the 240-byte chroma LUT (e.g. the grayscale stream which dominates the v8
archive bytes); (c) a Pareto-aware budget selection per Catalog #356 that picks the
operating point where ΔS_rate + ΔS_distortion is jointly minimized rather than
sweeping a fixed grid.

## 6-hook wire-in declaration (Catalog #125)

- **hook #1 sensitivity-map = ACTIVE** — the UNIWARD per-LUT-byte sensitivity IS
  the canonical sensitivity surface feeding the allocator.
- **hook #2 Pareto constraint = ACTIVE** — `total_budget_bits` is the Pareto bound;
  the rate/distortion trade is the Pareto frontier the verdict measures.
- **hook #3 bit-allocator = PRIMARY** — this work IS the bit-allocator hook;
  `tac.bit_allocator.per_byte.allocate_per_byte` is the canonical allocator
  consumed.
- **hook #4 cathedral autopilot dispatch = N/A** — research-signal only; no
  archive promotion (the score finding is honest-negative on this surface).
- **hook #5 continual-learning posterior = ACTIVE** — the canonical equation
  anchor below + the probe-outcomes ledger PROCEED-advisory row become the
  posterior anchor for this surface.
- **hook #6 probe-disambiguator = ACTIVE** — this memo + the
  `allocation_diff_from_uniform` smoke IS the disambiguator between "UNIWARD
  bit-allocation helps the v8 LUT" (FALSIFIED on this surface) vs "the wire is
  structurally feasible and the byte-savings surface is just too small"
  (CONFIRMED).

## Canonical equation proposal (Catalog #344; FORMALIZATION_PENDING)

`uniward_cost_map_bit_allocation_per_lut_byte_savings_v1`
[prediction; FORMALIZATION_PENDING until a paired-CUDA empirical anchor lands per
CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"]. The closed-form rate-axis
prediction `ΔS = 25 * (brotli(uniward_lut) - brotli(uniform_lut)) / 37,545,489`
matched the smoke (e.g. −42B → −0.000028) but the DISTORTION axis (recon-MSE
trade) is not yet formalized; the equation is FORMALIZATION_PENDING on the joint
rate-distortion surface, not on the rate axis alone.

## Paired-CUDA op-routable (DO NOT FIRE — operator-funded, separate step)

A ~$0.06 paired-CUDA RATIFICATION (Catalog #246 paired CPU+CUDA) of a v8 archive
built with the UNIWARD-bit-allocated LUT vs the uniform-allocated LUT would
empirically anchor the score delta. **It is NOT recommended given the
honest-negative advisory finding** — the ΔS ≈ −0.000028 rate savings is below the
contest noise floor and the recon-MSE cost is large. Re-route the $0.06 to the
reactivation paths above (larger surface / real SegNet bins) BEFORE any paired
dispatch. The op-routable is surfaced for completeness; the recommendation is
DEFER the paid anchor until a Pareto-dominant operating point exists at $0.

## NO FAKE IMPLEMENTATIONS verification (Slot EEE 5 classes)

- Class 1 (markers-without-work): AVOIDED — the allocator genuinely consumes the
  cost-map; `allocation_diff_from_uniform` proves it (240-byte diff, not 0).
- Class 2 (tests-verify-constants): AVOIDED — 15 NO-FAKE tests verify BEHAVIOR
  (cost-map nonuniform, sensitivity inverse-of-weight, alloc-diff non-empty,
  quantization coarsens, floor prevents destruction, coarsened LUT changes
  render). Every test fails if the body is replaced by a no-op.
- Class 3 (synthetic-fixture): AVOIDED — the smoke runs on REAL
  `upstream/videos/0.mkv` frames, not toy fixtures.
- Class 4 (placeholder-string): AVOIDED — the manifest carries real measured
  numbers and canonical non-promotable Provenance markers, no `TBD`/`pending`.
- Class 5 (enum-padding): N/A — the two allocation methods (TOP_K vs UNIFORM) are
  the canonical `PerByteAllocationMethod` enum, structurally distinct.

## Working-tree scope (Catalog #340 sister-disjoint)

ONLY files under
`src/tac/substrates/uniward_per_pixel_distortion/nscs06_v8_chroma_lut_integration/`
+ this memo + the smoke manifest + lane registry + probe outcome. NSCS06 v8 +
bit_allocator + uniward_delta are READ-ONLY consumer-imports (unmodified). Disjoint
from the in-flight z8 seg-lever subagent (owns `z8_*`) and the z5 harvest poll.
