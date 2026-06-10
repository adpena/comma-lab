# Boundary-math seg-core — build + exact-scorer verdict (task #52, 2026-06-10)

**Authority:** `[local CPU-torch advisory]` (exact contest SegNet on CPU, GT-decoded via
`upstream/frame_utils.yuv420_to_rgb`; NOT the 600-sample contest harness → non-promotable per
the GOAL authority ladder). `$0` spend, no GPU, no paid dispatch, no MPS.

**Spec executed:** `.omx/research/closed_spec_boundary_math_system_of_equations_20260610.md`
(§3 data structures, §4 the polytope system of inequalities, §10 the waterfilling water level
λ\* = 1.27 B/flip). This is the **seg core of offensive lever A** (evaluator-equivalence quotient
compiler) — the right data structures + a real combinatorial SOLVE, replacing lever G's killed
rule-family search (NO-FAKE class 6).

## What was built (real solve over real data structures, NOT a search)
`src/tac/boundary_math/` (6 modules + 22 behavior tests, all green; ruff clean):
- `partition.py` — region adjacency graph + 4-connectivity connected components of `L*`
  (scipy.ndimage.label). The O(boundary) representation of the SegNet argmax partition.
- `contour_codec.py` — reversible (bit-exact) partition codec; `partition_description_bytes`
  is the real compressed length. Interior = constant runs (free); compressed size ≈ boundary
  entropy. Tests prove it COMPRESSES (a fake identity codec fails) and grows with boundary
  complexity (checkerboard > single-split).
- `bitmask_dseg.py` — exact `d_seg = popcount(XOR)/N`; cross-checked bit-for-bit vs the
  authority argmax-compare on 20 random partitions.
- `margin_polytope.py` — per-pixel free-budget `b(p)=m(p)/‖g_p‖` (§4 system of linear
  inequalities); jacobian path verified to divide by ‖g_p‖ (fake passthrough fails).
- `region_merge.py` — the MDL region-merge SOLVE at water level `WATER_LEVEL_BYTES_PER_FLIP =
  1.2731` (DERIVED from `(100/(600·384·512))/(25/37_545_489)`, matches §10's 1.27). A real
  greedy graph contraction over the RAG (merge small→large iff `marginal_bytes <
  flips_fixed·1.27`), NOT a parameter sweep, NOT a candidate search.
- `seg_core.py` — orchestrator: load real SegNet, decode GT frame1 via yuv420_to_rgb, extract
  `L*`, encode, solve, measure on the exact scorer.

## PRE-REGISTERED PREDICTION (written before the measurement)
1. The exact contour-coded partition lands at **d_seg ≈ 0** (it IS the SegNet argmax) at byte
   cost = the partition's boundary entropy.
2. The region-merge solve trades tiny regions for bytes at exactly the 1.27 B/flip threshold
   (keep iff contour bytes < flips-avoided · 1.27).

## PRE-REGISTERED KILL CRITERION (written before the measurement)
If the boundary entropy of `L*` (the partition carrier's byte cost) exceeds the current
carrier's seg-relevant byte budget — so the partition carrier cannot beat it — record the
finding + which contour-coder efficiency (STC/UNIWARD lever D, below 1.27 B/flip) would be
needed to cross the water level.

## THE TYPED ROW (exact local-CPU-torch, 16 GT frames; JSON:
`experiments/results/boundary_math_seg_core_20260610/lstar_measurement.json`)

| field | value |
|---|---|
| authority | `local-CPU-torch-advisory` |
| frames measured | 16 (of 600) |
| **d_seg of stored L\*** | **0.0 (ALL 16 frames, exact)** |
| roundtrip bit-exact | **True (ALL 16 frames)** |
| mean partition_bytes/frame | **895.7 B** |
| mean regions/frame | 36.2 connected components |
| **extrapolated 600-frame seg-partition** | **524.8 KB** |
| d_pose check (kept partition) | **0.0** (real DistortionNet, GT pair vs GT pair — frame1 unperturbed) |
| water level | 1.2731 B/flip |
| current frontier archive (total seg+pose) | 177,169 B = 173 KB |

Region-merge solve, real `L*` + 20 spurious-flip candidate: bytes 981→889 (−92), flips 20→11,
d_seg 1.02e-4→5.6e-5 — the solve correctly contracts small wrong regions into large GT-matching
neighbours, **reducing both bytes AND d_seg** (strict improvement). The merge is not a no-op
(tested) and respects the water-level threshold (tested at high/low λ).

## VERDICT — prediction CONFIRMED; KILL criterion TRIGGERED at the baseline coder
- **Prediction 1 CONFIRMED, exactly:** storing the contour-coded `L*` lands at **d_seg = 0.0**
  (bit-exact, all 16 frames) — the stored object IS the SegNet argmax. The d_seg=0 carrier
  perturbs frame1 by NOTHING, so **d_pose = 0** (verified on the real DistortionNet). The
  seg-axis is fully solvable in closed form: there is no search, no training, no smooth
  surrogate — the partition is stored and reproduced exactly.
- **Prediction 2 CONFIRMED:** the region-merge SOLVE is a real combinatorial contraction that
  trades regions for bytes at the 1.27 threshold (demonstrated; threshold-sensitive).
- **KILL criterion TRIGGERED for the LZMA baseline coder:** the partition carrier costs
  **524.8 KB for seg alone** — **2.96× the entire current 173 KB archive** (which already
  carries BOTH seg and pose). The naive partition carrier does **NOT** beat the carrier's seg
  byte budget. **The boundary entropy of `L*` under the reversible LZMA-over-labels baseline
  (≈896 B/frame) is too high to be a competitive standalone seg carrier.**

**The mechanism (the crux this localizes):** `d_seg = 0` is FREE to *describe* combinatorially,
but EXPENSIVE to *store losslessly* with a generic coder. The SegNet argmax partition has ~36
connected regions/frame with thin, jagged 1D boundaries; LZMA-over-labels pays ~896 B/frame for
that boundary entropy. This is exactly the §10 prediction that **lever D (STC/UNIWARD contour
coding below 1.27 B/flip) is the GATE, not optional**: coding-theoretic efficiency is the only
way the partition carrier crosses the water level.

**The needed efficiency (quantified handoff to lever D / lever F):**
- Current baseline: 895.7 B/frame for ~36 regions with d_seg=0.
- The 173 KB total archive ⟹ a competitive seg slice must be **≲ 100–150 KB for 600 frames**
  ≈ **170–250 B/frame** — a **3.6–5.3× reduction** vs the LZMA baseline.
- That is the target for the contour coder: a context-arithmetic / chain-code / STC boundary
  coder operating at the per-pixel margin-polytope free-budget (the boundary is a thin 1D set;
  91% of *boundary* pixels are low-margin, but only ~1.4% of ALL pixels are margin<0.5 — the
  partition is mostly free interior, so the true coded object is a sparse 1D contour, not an
  area). A boundary coder at ≲0.5 B/boundary-pixel would cross the water level.

**This is a DEFER (not a KILL) per Catalog #307 + Forbidden-premature-KILL:** the *paradigm*
(store the argmax partition directly, d_seg=0 by construction) is PROVEN exactly — only the
*baseline LZMA coder implementation* is too fat. Reactivation = swap the contour coder for a
margin-aware boundary entropy coder (lever D) and re-measure against the 170–250 B/frame target.

## ANTI-FAKE self-checks (all pass)
- The solve is a real RAG graph contraction (merge small→large at the closed-form 1.27
  threshold) — NOT a sweep/grid/rule-search. No parameter gridding anywhere.
- Tests verify BEHAVIOR not constants: codec roundtrip is bit-exact AND compresses (identity
  codec fails); popcount d_seg == argmax-compare on random partitions; the merge actually drops
  the predicted spurious regions and rewrites the label map (no-op merge fails); the polytope
  budget actually divides by ‖g_p‖ (passthrough fails); the merge respects the water level
  (high/low λ flip the decision).
- d_seg measured on the EXACT contest SegNet (`modules.SegNet` + `segnet.safetensors`),
  GT-decoded via `yuv420_to_rgb` ONLY (no PyAV rgb24 phantom). d_pose on the real DistortionNet.
  NEVER MPS.

## HANDOFF (the per-region/per-boundary marginals → the waterfilling allocator, task #54)
This build produces the CLOSED-FORM marginals the waterfilling allocator (§10) needs:
- **per-region**: `(region_id, pixels, label, contour_marginal_bytes, flips_fixed)` from the
  region-merge solve — the exact bytes-vs-flips trade per region, ranked steepest-first at λ\*.
- **per-boundary-pixel**: the margin-polytope free-budget `b(p)=m(p)/‖g_p‖` — which pixels are
  free (interior) vs paid (tight boundary), the seg constraint set the allocator projects onto.
- **the water level itself**: `WATER_LEVEL_BYTES_PER_FLIP = 1.2731` (the KKT dual variable λ\*).
The allocator equalizes these marginals at λ\* (a SOLVE, not a sweep). The seg-core's job — give
the allocator EXACT marginals — is done; the next dependency is lever D's boundary coder to bring
the partition bytes below the water level so the seg carrier is fundable.

## Scoreboard (re-verified, not trusted)
- UPPER: S = 0.19109982 [contest-CPU] (recoded-R3 defensive hold) — ABOVE T_1 → GOAL UNSATISFIED.
- LOWER (seg floor contribution): this build gives the **seg description-length** at d_seg=0:
  ≈896 B/frame under the LZMA baseline; the lever-F seg floor = boundary-contour entropy of L*
  under the OPTIMAL coder (≲ this, target 170–250 B/frame). The seg axis is d_seg=0-achievable;
  the floor question is purely the contour coding rate.
- Burning question: **what is the true entropy of the SegNet argmax boundary contour?** A
  context-model / STC chain-coder estimate sets the lever-F seg floor and decides whether the
  partition carrier can ever cross the 1.27 B/flip water level standalone — or whether it must
  COMPOSE with the existing carrier (store only the partition DELTA the renderer gets wrong).
