# Our own representation: the bitmask+magnitude+ops WITNESS FORMAT, interpreted by inflate.py (design, 2026-06-23)

**Source:** operator 2026-06-23 — *"We can create our own representation or file format … bit packing and
shifting and firmware and extreme edge compute exploits … Like bitmap or bitmask but combined with magnitudes
and behaviors or ops … extremely compressible and with as much packed into inflate.py as possible … inflate.py
as an interpreter or to do … clever and outrageous and visionary and inverse things using cpu and/or gpu."*
This is the **Evaluator-Equivalent Witness Compiler Paradigm** (CLAUDE.md NON-NEGOTIABLE) made concrete.
Authority `[design]` over MEASURED anchors; score math via `tac.contest_score`; pointer UNMOVED 0.19110.
Builds on `manifold_topology_dseg_deep_synthesis_20260623` + the L13 witness + #99 firmware bit-packing.

---

## 0. The thesis (the orthogonal factorization)
`S = 100·d_seg + √(10·d_pose) + 25·bytes/N` factors into TWO orthogonal levers the operator's vision unifies:
- **d_seg / d_pose (distortion) = a TRAINED GENERATOR** (the convergence: the d_seg-binding islands are
  full-rank content-noise; only a learned generative program touches them).
- **bytes (rate) = the CUSTOM WITNESS FORMAT** (this memo): package the generator's task-space output —
  bitmask (WHERE) + magnitudes (HOW-MUCH) + ops/behaviors (HOW-TO-EVOLVE/SOLVE) — interpreted by inflate.py,
  extreme-bit-packed, with the structure in the (cheap) CODE and minimal DATA.
**The format does NOT lower d_seg; it lowers the RATE at which a given d_seg/d_pose is carried.** The two
compose: trained generator × custom format = sub-0.15.

## 1. Existence proof — L13 already validated the FORMAT half (honest, measured)
`witness_L13_optimal_pose_carrier_result_20260621` + `CAPSTONE_witness_taskspace_roundtrip_byte_floor`:
- **Pose axis CLOSED by the format:** a rendered pose-carrier, ~22.5 KB (brotli-q11 real byte-close),
  d_pose 12.66 → **0.006**, round-trip-threaded (uint8 STE → exact frozen PoseNet), clean float64 parity
  (1 LSB). **−59% vs the frontier's RGB rate.** PROOF the rendered-witness format (a) compresses and (b)
  SURVIVES the eval round-trip (vs the flat-sidecar which did NOT — see §3).
- **d_seg axis NOT closed by the format alone:** L13's d_seg = **0.0068** (term 0.68) — the SAME full-rank
  island wall. The witness's mask must be PRODUCED by a generator that hits d_seg ~6e-4; the format only
  packages it cheaply. **Corrects the prior "72KB lossless-parity sub-0.15" over-claim: L13 is S≈0.79, pose
  closed / d_seg open.**
→ The format works (pose, −59%, survives). The open half is the d_seg generator + its mask-grammar packaging.

## 2. The format (bitmask + magnitudes + ops — a domain-specific bytecode)
A 4-layer witness program, decoded by inflate.py:
1. **WHERE — bitmask / contour layer.** The codim-1 boundary band (the only d_seg-relevant pixels; interiors
   are free). NOT a per-pixel bitmap (543 KB, §3) — a **run-length / chain-code contour** of the COARSE
   boundary (topology rank 4 → cheap) + a sparse island index. Homography-static classes (road/horizon/hood,
   known comma camera geometry) are WHERE-priors at near-zero bytes.
2. **HOW-MUCH — magnitude layer.** The SegNet margin values in the band (how far each boundary pixel is from
   flipping), quantized — ONLY in the band, not the frame. Fisher-√ bit allocation (more bits where the
   margin is shallow = where flips happen). For pose: the luma-subspace carrier coefficients.
3. **HOW-TO-EVOLVE — ops / behavior layer.** Per-frame opcodes that evolve the boundary across the 600-frame
   clip: `WARP(motion_model)` + `CORRECT(delta)` + `BIRTH/DIE(island_ops)`. Temporal amortization: store the
   boundary ONCE + cheap per-frame ops. This is where the clip's continuity is exploited.
4. **HOW-TO-SOLVE — the inflate.py interpreter (the "inverse" / "outrageous" layer).** inflate.py is NOT a
   table lookup — it RUNS the program: warp the boundary, apply ops, and INVERSE-SOLVE for a round-trip-
   surviving RGB witness whose post-round-trip SegNet-argmax = the coded mask and PoseNet-6dim = the coded
   pose. CPU or GPU (GPU-witness contest-legal, README L114, 30-min budget). The solver (level-set projection
   / feasibility / NCA step — #55/#73/#143 lineage) is CODE (cheap, in the ≤200-LOC inflate budget); the
   DATA is the compact spec. **Kolmogorov: short program + short data → complex witness.**

## 3. Why this beats the two SOLID failures (the honest blockers it must clear)
- **Flat boundary-sidecar was NO-GO** (`witness_seg_boundary_decisive`: 543 KB flip-bitmap + **46%**
  round-trip survival). The format beats BOTH: (a) 543 KB → ops/temporal-MC/contour/homography crush the
  COARSE boundary; (b) 46% survival → RENDER a real witness frame (L13 proved render-survives), not a flat
  sidecar that the bicubic↑→uint8→bilinear↓→YUV6 round-trip destroys.
- **BUT the topology verdict bounds it honestly:** the COARSE boundary (rank 4) compresses beautifully under
  ops/contour; the **class-1 ISLANDS (rank 53, content-noise, R²=0.23 ego)** do NOT compress via flat
  ops/warp. The islands are the residual that the FORMAT cannot crush — they need the **inverse-solve
  generator** (layer 4) running a LEARNED rule, OR they fold into the d_seg the trained generator carries.
  This is the precise seam between "format" (handles coarse, low-rank, smooth) and "generator" (handles
  islands, full-rank, content). **The format is necessary (−59% rate) but not sufficient (islands need the
  generator).**

## 4. Firmware bit-packing + edge-compute exploits (#99 lineage — the constant-factor multiplier)
The format's bytes are the brotli of a packed byte-stream. #99 measured a **2.85× ARM-vs-real gap** —
firmware-grade encoding (bit-shifting, sub-byte packing, op-fusion, fixed-point, no-padding structs) recovers
it. Applied here: pack the contour chain-codes (2-3 bits/step), the band magnitudes (Fisher-√ variable bits),
and the op-stream (variable-length opcodes) at the bit level, then brotli. The inflate.py interpreter unpacks
with shifts/masks (the "edge-compute exploit" — the same bit-tricks firmware uses). This is a constant-factor
(~2-3×) on the format's bytes, ON TOP of the structural −59%.

## 5. The honest unification (what moves S, and how the pieces compose)
| piece | lowers | status | evidence |
|---|---|---|---|
| trained generator (from-scratch sweep / spectral arch) | d_seg | in flight / queued | the convergence; FINER/WIRE a0e28b5 |
| custom witness format (this memo) | bytes (rate) | pose half PROVEN (−59%) | L13 22.5 KB / d_pose 0.006 |
| inflate.py inverse-solve interpreter (layer 4) | bytes (islands via generation not storage) | DESIGN | #55/#73/#143 lineage |
| firmware bit-packing (#99) | bytes (constant 2-3×) | measured gap | B1-PACK 2.85× |
**Composed projection (directional, NOT measured):** generator at d_seg ~6e-4 (term 0.067) + pose 0.006
(term ~0.024, improvable) + witness format rate ~50-72 KB (term 0.033-0.048) → **S ≈ 0.12-0.14, sub-0.15** —
IF the generator hits d_seg ~6e-4 AND the islands fold into it. The format is the proven rate half; the
generator is the open d_seg half. Same verdict as the 4-line convergence, now with the format quantified.

## 6. Decisive next measurements (the operator's "imagination + inverse" made empirical, $0)
1. **Representation-level intrinsic-dimension search (spawned, $0/CPU):** measure the class-1 islands'
   intrinsic dim across {pixel-linear, spectral/DCT, Fourier-descriptor contour, nonlinear-motion-comp,
   small-AE}. The level where `m` collapses = the basis to build layers 1-2 in. Decides whether the islands
   are format-compressible (some basis) or generator-only (no basis).
2. **inflate.py inverse-solve byte floor (queued, $0):** for the REAL 600-frame mask, measure the compressed
   size of the contour+ops+inverse-solve format vs the 543 KB flat sidecar, at d_seg-faithful + survival.
   The make-or-break for the format's d_seg half.
3. **The generator (the END):** the from-scratch sweep outputs frames; this format packages them. Fire it
   (with FINER/WIRE arch verdict + latent-dim arm) → byte-close in THIS format → exact eval.

## 7. Honest ledger / NO-FAKE
- MEASURED: L13 pose carrier 22.5 KB / d_pose 0.006 / −59% / survives (real byte-close); flat sidecar 543 KB
  / 46% survival (NO-GO); islands rank 53 / ego R²=0.23; #99 2.85× ARM gap.
- DESIGN (this memo): the 4-layer format + inflate.py inverse-solve interpreter + the composed S projection.
  NOT a built codec; NOT a score. The format is the RATE half of the sub-0.15 vehicle; the generator (open)
  is the d_seg half. The END is a byte-closed exact row in this format from the trained generator.
- The witness MUST land in the frozen evaluator cells on the EXACT contest video and survive the round-trip
  (the L13 discipline) — no flat sidecar, no phantom parity.

## 6-hook wire-in
#1 sensitivity-map: the band/island routing prior. #2 Pareto: format-rate vs d_seg/d_pose frontier. #3
bit-allocator: Fisher-√ magnitude bits + #99 firmware packing. #4 cathedral: N/A (design). #5
continual-learning: this memo + the two §6 measurements. #6 probe-disambiguator: §6.1 (which basis
compresses islands) + §6.2 (inverse-solve vs flat) ARE the format-viability disambiguators.

---

## Observability surface

*(OBSERVABILITY-ADDENDUM 2026-08-25 — APPEND-ONLY per Catalog #110/#113. This
section is an INDEX into this memo's own content per Catalog #305's 6 facets;
it adds no new claim. Facets with no counterpart in this memo say so plainly.)*

1. **Per-layer inspection** — §2 "The format (bitmask + magnitudes + ops — a domain-specific bytecode)" specifies each stored layer separately (bitmask / magnitudes / op stream), so each is inspectable in isolation at decode.
2. **Per-signal decomposition** — §5 "The honest unification (what moves S, and how the pieces compose)" decomposes which piece moves which score term.
3. **Run-to-run diff** — the format is a fixed bytecode, so two builds diff at the section level; §1 "Existence proof — L13 already validated the FORMAT half" is the matched prior build to diff against.
4. **Post-hoc query** — the interpreter is `inflate.py` (FREE, unsized); the counted payload is the format's own sections. §6 "Decisive next measurements ($0)" names the queries that resolve the design.
5. **Cite-chain** — §7 "Honest ledger / NO-FAKE" plus the "6-hook wire-in" section carry the provenance and consumer chain.
6. **Counterfactual hooks** — §3 "Why this beats the two SOLID failures" is an explicit against-two-baselines counterfactual; §4 "Firmware bit-packing + edge-compute exploits" is the constant-factor ablation axis.
