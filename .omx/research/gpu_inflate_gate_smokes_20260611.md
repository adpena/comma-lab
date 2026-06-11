# GPU-INFLATE GATE SMOKES — the $0 gate before any paid GPU prototype (SMOKE A + SMOKE B)

UTC 2026-06-11 · claude (Opus 4.8) · **RESEARCH SMOKES, $0 macOS-CPU, NO GPU, NO MPS, NO dispatch.**
`[macOS-CPU advisory]` · `promotable=false` · `score_claim=false` · `ready_for_exact_eval_dispatch=false`.
torch-CPU is authority for every byte/d_seg/d_pose number here. The two smokes the GPU-inflate
evaluator-equivalence design (`gpu_inflate_evaluator_equivalence_exploit_design_20260611.md`, commit
`bfe309a9e`) specified as the gate before any paid GPU work. NO FAKE: real pyav decode of real
`upstream/videos/0.mkv`, real SegNet/PoseNet weights, real argmax-flip `d_seg` and pose-MSE `d_pose`.

    S = 100·d_seg + √(10·d_pose) + 25·B/D     D = 37,545,489
    Frontier (pointer): contest-CPU 0.19109982 @ 177,169 B · contest-CUDA 0.20533 @ 186,876 B

---

## VERDICT: **DEFER** the GPU-inflate G1 (cell-solve) candidate — the baked cell spec blows the byte budget

The design's whole ballgame (its §2.2/§6/§7) rested on ONE load-bearing `[predicted]` number: that the
baked evaluator-cell spec (boundary SegNet label band + margin + pose tube + seed) costs **~27–60 KB**, well
under the 177 KB frontier. **SMOKE A measured it. It does not.** Even at the tightest, most-favorable drift
threshold, the boundary label band alone is **~330–465 KB at 600 frames** — ~2–2.6× *over* the frontier and
~6–17× over the design's G1 prediction. The cell-spec idea is **rate-dominated**, exactly the failure mode
§7 said would trigger a DEFER. This is the honest gate result.

---

## SMOKE A — cell-spec sizing (the decisive measurement)

**Method.** Decoded the SegNet-scored last-frame of N=24 sampled pairs via the EXACT upstream path
(`av` + `frame_utils.yuv420_to_rgb`, BT.601 limited-range — matches `AVVideoDataset`). Ran the REAL SegNet
(`tu-efficientnet_b2`, 5-class, `x[:,-1]`, bilinear→384×512) → per-pixel 5-class argmax + top1−top2 margin.
Built the baked boundary label band = keep the argmax label ONLY where `margin < drift` (the fragile band the
GPU solve must pin); release the certified-free interior. Measured **actual compressed bytes** three ways and
took the best: (a) sentinel-band (robust→sentinel 7, brotli-q11/lzma-9e), (b) temporal-delta change-mask, and
(SMOKE A.2) (c) sparse `(Δposition varint, 3-bit label)` + the information-theoretic floor
`Σ_f log2 C(P, k_f) + H(class)·Σk_f`. Sample→600-frame extrapolation is linear in frame count.

**Sample is representative (NO-FAKE cross-check):** my 24-frame fragile-pixel fractions match the MEASURED
600-frame margin field (`segnet_margin_field_20260609.json`) within ~2–4% relative at every threshold
(e.g. drift<0.1: mine 0.270% vs measured 0.282%; drift<2.0: 4.699% vs 4.833%).

**Result — the boundary label band at 600 frames (the dominant cost):**

| drift (logit) | kept px/frame | %px | sentinel-band @600 | sparse @600 | entropy floor @600 |
|---|---:|---:|---:|---:|---:|
| **0.1** (the L7 cross-hw drift, tightest) | ~530 | 0.27% | **438 KB** | **331 KB** | **465 KB** |
| 0.25 | ~1325 | 0.67% | 620 KB | 524 KB | 1033 KB |
| 0.5 | ~2633 | 1.34% | 725 KB | 681 KB | 1860 KB |
| 1.0 | ~5097 | 2.59% | 793 KB | 790 KB | 3229 KB |
| 2.0 | ~9238 | 4.70% | 869 KB | 879 KB | 5205 KB |

Pose tube (600×6 fp16+compress, smooth-trajectory conservative upper estimate): **~6.2 KB** (at floor, fine).
Design's seed add-on: `[predicted]` 8–15 KB.

**Total baked cell (best case, drift<0.1, sparse codec):** label **331 KB** + pose 6 KB + seed [8–15] KB =
**~345–352 KB** — vs the 177 KB frontier and the design's 27–60 KB G1 target.

**Why it's codec-independent (the decisive point).** The sparse encoding and the information-theoretic floor
*agree*: at drift<0.1, ~530 fragile pixels are scattered along all class/object contours of each of 600
*distinct* frames. Just storing WHERE those contours are costs `log2 C(196608, 530) ≈ 4500 bits/frame ≈
562 B/frame` → ~337 KB for positions alone — a hard lower bound the GPU solve cannot avoid, because it must
know where to pin each argmax. The design's optimistic "~12–25 KB boundary band" under-counted the
*positional* information of a 600-frame contour set by ~15–25×. RLE/temporal-delta cannot rescue it: the
contours move frame-to-frame (it's a moving dashcam), so the change-mask is itself ~full-frame entropy.

Artifacts: `experiments/results/gpu_inflate_smoke_a_20260611T120037Z/smoke_a_cell_spec_size.json` +
`experiments/results/gpu_inflate_smoke_a2_20260611T120202Z/smoke_a2_sparse_entropy.json`.

---

## SMOKE B — solve-reachability (secondary; partial, directional)

**Method.** On real pairs, optimized a witness pixel field (Adam, deterministic seed) from a 16×-downsampled
GT seed (OUT-of-cell: the cell-tolerance curve says 4× is in-cell, so 16× is a real lift) against the cell
target (CE on the baked SegNet argmax + 10× pose-MSE), then re-scored through the EXACT SegNet/PoseNet.
In-cell thresholds from the MEASURED cell-tolerance curve: d_seg<0.005, d_pose<1e-3.

**Result (pair 0 of 3; stopped after pair 0 — ~7.5 min/pair full-res CPU PoseNet fwd/bwd, and SMOKE A
already gates the verdict):**

| | seed | after 150 steps | in-cell? |
|---|---:|---:|---|
| **d_seg** | 0.0304 | **0.0007** | ✅ YES (< 0.005) |
| **d_pose** | 3.45 | **87.5** (worse) | ❌ NO |

- **Seg axis IS reachable.** A deterministic gradient solve pinned the SegNet argmax band to the baked target
  (0.0304 → 0.0007, well in-cell). The seg half of the §2.3 solve premise holds.
- **Pose axis NOT reached with the naive joint objective.** Freely optimizing the full pixel field broke the
  pose geometry (the CE-seg gradient dominated; 10× pose weight insufficient; pose is extremely sensitive).
  This is a known seg↔pose gradient-conflict difficulty, not proof of impossibility — but it shows the solve
  is non-trivial and would need a much more careful (pose-tube-projected, warp-consistency) objective. The
  GPU buys *speed*, not a fix for this conflict.

Artifact: `experiments/results/gpu_inflate_smoke_b_partial_20260611T123200Z/smoke_b_reachability_partial.json`.

---

## THE GATE DECISION (per design §6/§7, honest)

| question | answer | source |
|---|---|---|
| Is the baked cell spec small (~27–60 KB)? | **NO — ~330–465 KB even at the tightest drift; codec-independent** | SMOKE A + A.2 + entropy floor |
| Does it clear the 177 KB / 0.191 frontier on rate? | **NO — ~2–2.6× over, before distortion/seed** | SMOKE A |
| Can a deterministic solve land in-cell? | **Seg: yes. Pose: not naively (gradient conflict).** | SMOKE B (partial) |

**→ DEFER the GPU-inflate G1 (cell-solve) candidate. Do NOT spend any paid GPU on it.** Both independent
sub-questions point away: the cell spec is rate-dominated (the binding failure), AND the pose half of the
solve is non-trivial. The design's §6 claim "the tax is not the binding constraint; the in-cell distortion
is" was wrong about the binding constraint — **the cell-spec RATE is the binding constraint**, and it fails
the gate by itself. The 80.67% resize-null + the tolerant cell make the *interior* free, but the *contour
positions* across 600 moving frames are irreducibly expensive to store, and that is precisely what the cell
spec must carry.

**What this does NOT kill (DEFER, not KILL, per Forbidden-premature-KILL):** the GPU-inflate *paradigm* is
not falsified — only the G1 "store the boundary argmax cell" instantiation. Reactivation paths if revisited:
(1) a cell representation that does NOT store explicit per-frame contour positions — e.g. a *generative-rule*
that the GPU runs to RE-DERIVE the contours from a tiny seed (the G5 CA grower, which faces the same
positional-information wall but with a possibly-cheaper program); (2) carry only the ~2.16% true class-
boundary band AND prove a temporal-warp model collapses the per-frame position cost (untested — SMOKE A's
temporal-delta already did poorly, so the bar is high); (3) pursue the GPU's *distortion* lever (design DOOR
1 / G2) on a CPU-feasible-seed substrate where the rate is paid CPU-side and the GPU only refines — but that
is dominated by CPU-routing unless a measured CPU-infeasibility wall exists. None of these is near-term; all
are below the active CPU-side rate/entropy work on the existing 177 KB frontier.

---

## WIRE-IN (Catalog #125)

1. **sensitivity-map** — ACTIVE. New MEASURED prior: *"the baked evaluator-cell spec (boundary SegNet argmax
   contour band over 600 frames) costs ~330–465 KB at the tightest drift — its positional information is
   irreducible (~562 B/frame floor); the GPU-inflate cell-solve is RATE-dominated, not distortion-dominated."*
   Re-ranks GPU-inflate G1 back BELOW its prior DEFER, now with a measured byte number, not a prediction.
2. **Pareto constraint** — ACTIVE. A GPU-inflate cell-solve atom is admitted iff its baked cell < ~128 KB
   (the tax break-even); SMOKE A measures it at ~345 KB → the atom is REJECTED at the rate constraint.
3. **bit-allocator** — ACTIVE. Confirms the allocator's existing prior (`segnet_margin_field`): the boundary
   band is where seg bytes go, BUT storing the band as an explicit per-frame cell is ~3-17× costlier than the
   resize-null/cone-waterfill approaches already on the frontier — do NOT allocate to an explicit cell spec.
4. **cathedral autopilot dispatch** — N/A (this is the $0 gate that PREVENTS a dispatch; the verdict is no-go).
5. **continual-learning posterior** — the SMOKE-A `[empirical]` cell-byte-cost rows (`[macOS-CPU advisory]`,
   non-promotable) are the reseed; they pin the cell-spec rate so no future memo re-predicts it optimistically.
6. **probe-disambiguator** — ACTIVE. SMOKE A + B ARE the disambiguator for *"is the baked cell small AND
   solve-reachable?"* — answered: small NO (decisive), reachable seg-YES/pose-NO. Question resolved.

## CROSS-REFERENCES

`gpu_inflate_evaluator_equivalence_exploit_design_20260611.md` (the design these smokes gate — its §7 SMOKE A
predicted this exact DEFER path if the cell ran ~150 KB; the measured ~345 KB is even further off) ·
`segnet_margin_field_20260609.json` (the MEASURED margin field my fragile fractions cross-check against) ·
`evaluator_cell_tolerance_20260609.json` (the in-cell thresholds SMOKE B uses) ·
`evaluator_invisibility_basis_landed_20260610.md` (the 80.67% free interior — real, but it is the interior, not
the expensive contour positions) · `canonical_frontier_pointer.json` (contest-CPU 0.19109982 @ 177,169 B).
