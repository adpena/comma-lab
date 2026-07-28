# DDM-IV2 — independent built-asset plug-in sweep (what we already built that plugs into the live composed archive)

**Date:** 2026-07-28 · **Arm:** `ddm_iv2` (independent inventory; READ-ONLY) · **Evidence axis:** `[inventory — docstrings + landed receipts; every number labeled MEASURED/DERIVED/UNMEASURED]`
**score_claim=false · promotion_eligible=false · pointer UNMOVED at 0.19108 [contest-CPU].**

Independent of any sister arm. I did NOT read `ddm_iv1_*` (a sibling in-flight sweep surfaced in grep) —
worked from my own enumeration only.

---

## The anchor the whole inventory hangs on (MEASURED, the live row)

`r6cal` byte-close eval — real `upstream/evaluate.py` n600 CPU on the exact `291,205,400 B` archive
(sha `e3d0581f…`, git `7b7f803a`):

| term | value | S contribution | share |
|---|---|---:|---:|
| d_seg | **0.00115997** | 0.116 | 0.06% — **BOX MET** (≤0.00116) |
| d_pose | **0.01663316** | √(10·d_pose) = **0.408** | 0.21% |
| rate | 291.2 MB / 37.5 MB | 25·B/D = **193.9** | **99.73%** |
| **S** | | **194.43** | |

**The entire game is RATE.** Seg is solved to box; pose is loose but tiny in S; the archive is 1455×
too big (target box ≤~200 KB ⇒ rate ≈0.13). The 291 MB decomposes (MEASURED, ddm_oc1):
**residual 210 MB (72.1%) + frame-0 bootstrap 81 MB (27.9%)** — both are *dead-node* symptoms: the
shipped codec's PREDICT node is dead (`descriptor_len=0`, mode uniformly `SPATIAL_SMOOTH_121`) and it
pays a DENSE 210 MB residual (89% nonzero) to hit the box. **The compounding lever (MEASURED,
ddm_oc1): copy-PREDICT already leaves only 0.864% of 117,964,800 sites flipped — a residual RESTRICTED
to that flip support + SegNet RF dilation is 60–100× sparser than dense.** Every asset below is graded
by whether it attacks that: sparse support, real coders, pose crush, frame-0 crush.

---

## RANKED PLUGS-IN TABLE

Integration cost: S=drop-in (wire an existing receiver/coder), M=needs a measurement/adapter, L=needs new glue.

| # | Stage | Asset (path) | The MEASURED receipt that makes it "super optimal" | Cost |
|---|---|---|---|---|
| 1 | 4 POSE | **R1 `dxi` banked pose** (`--pose-carrier-xi-from-ckpt`; `boundary_math/xi_pose_coder.py`) | **SHIPPABLE, byte-closed: 7.2 KB counted ξ → d_pose 0.001610 n600, contrib 0.127** (`r1_dxi_shippability_byteclose_20260708.md`). r6cal ships d_pose 0.0166/contrib 0.408 → drop-in **−0.28 S** and −74 KB vs any dense pose. | S |
| 2 | 4 POSE | **xi_pose_coder** (`boundary_math/xi_pose_coder.py`) | store-nothing pose = **7,200 B ξ (fp16), H DERIVED FREE at decode** (`H=K(R−t nᵀ/d)K⁻¹`); kills the 43,200 B (82.9%) redundant fp64 H the old serializer shipped (`finding1_store_nothing…`, MEASURED to the byte). Temporal-delta+arith-codes the smooth trajectory. The optimal home for asset #1's payload. | S |
| 3 | 2 SUPPORT | **argmax-flip support extractor** (ddm_oc1 tool `experiments/ddm_oc1_xi_temporal_measure.py` + `boundary_math/bitmask_dseg.py`) | MEASURED n600 through frozen SegNet: copy-PREDICT leaves **1,019,467/117,964,800 = 0.864%** flipped; the escape residual is 60–100× sparser than dense. This IS stage-2. | M |
| 4 | 3/4 CODING | **context_partition_codec** (`boundary_math/context_partition_codec.py`) | SOTA context-arith (JBIG/LOCO-I/CABAC) for the 5-class argmax partition L*; achievable len = Σ N_ctx·H(p_ctx) via `constriction` RangeEncoder. Explicit TOP-AIML replacement for LZMA-over-labels (prototype 669–873 B/frame ⇒ rate 0.27–0.35, DEAD). Codes SUPPORT geometry. | M |
| 5 | 3 RESIDUAL | **uint8_lattice + tie_aware_preimage** (`optimization/uint8_lattice_feasibility.py`, `tie_aware_preimage.py`) | Places the camera integer lattice INSIDE the inverse solve; factor-2 blocks = small bounded Diophantine (35–931 exact preimages/block MEASURED). Determines RESIDUAL VALUES that survive R+uint8 exactly. **Nuance: canonical support-fill is a certified NO-OP on the factor-2 spine (MEASURED 0/117.96M, `factor2_canonical_preimage_fp32_exact_v1`)** — so it's the feasibility certificate, not a free-byte win, at factor-2. | M |
| 6 | 5 EXPORT | **ddm_runtime_exporter + ddm_runtime_receiver** (`optimization/ddm_runtime_*.py`) | The WORKING byte-close path: compiles sealed DDM state → runtime packet; receiver copied byte-for-byte to `inflate.py`, **stdlib+Torch+Brotli only**, all instance bytes manifest-covered. This is what r6cal's 291 MB row already flowed through — the stage-5 spine is BUILT and exercised. | S |
| 7 | 4 CODING (frame0) | **keyframe_codec** (`boundary_math/keyframe_codec.py`) | Attacks the **81 MB (27.9%) frame-0 bootstrap** directly: pure numpy/cv2/PIL/scipy keyframe primitives; MEASURED rate ~0.03–0.07 for 13 keyframes (the #202 crux). frame_0 is seg-free (SegNet reads last frame only) so it is a pure pose+rate object. | M |
| 8 | 1 PREDICT | **warp_real_luma_frame0 + stratified_depth_warp** (`boundary_math/`) | SE(3)-screw pose carrier warping a real keyframe by stored ξ; **bit-parity guaranteed** (all-zero flow ⇒ OP-for-OP == numpy fp64 warp). Lifts the W8 dual-use conflict (frame0 seg-free ⇒ warp it purely for d_pose, ZERO d_seg cost). Feeds the frame-0 bootstrap crush + pose. | M |
| 9 | 1/3 | **predict_project_receiver** (`optimization/predict_project_receiver.py`, 167 KB) | Deterministic NumPy PREDICT→PROJECT receiver: predicts cells from a seed, extracts only VIOLATED constraints, projects finite linear cell/tube systems, composes factor-2 interval/support-fill/full-kernel. The composed stage-1+stage-3 receiver primitive, scorer-agnostic. | M |
| 10 | 4 CODING | **movable_site_coder** (`boundary_math/movable_site_coder.py`) | Movable (1.56% area) as a SPARSE SITE LIST (per-object cx,cy,w,h + Hungarian temporal tracking → small per-track delta) instead of a whole-scene bitmap. The last of 5 whole-scene edges made geometric. Composes with the sparse-support residual. | M |

**Runner-ups (built, plug-in-relevant, lower leverage or needs A/B):**
`ego_xi_trajectory` (P-frame predictive-coding of lane coeffs by ξ — no inverse-warp, bit-identical by construction; the ξ producer that feeds #1/#2/#8); `cross_pair_waterfilled_corrector` (E3: d_pose pools BEFORE sqrt ⇒ cross-pair pose trades 1:1 — the correct global pose amortization); `movable_deshare` (dedup audit — stop coding a pixel in two carrier ledgers); `lane_ground_factorization`+`inverse_depth_compander` (IPM ground-frame + SPD-cone anisotropic lane coding, −27% precedent on pose section); `arith_selfcomp_rate_coders` / `xi_temporal_delta_coder` (framed range coders for local rate measurement); `resize_null_preimage`/`evaluator_invisibility_basis` (certified free-byte null-space postprocessor — but a NO-OP on the factor-2 spine per #5's nuance).

---

## PER-AXIS ENUMERATION COUNTS (honest coverage)

| Axis | Scanned | Plugs-in | Superseded/limited | Irrelevant-to-live-pipeline | Coverage honesty |
|---|---|---|---|---|---|
| A. Research memos (`.omx/research`) | 9,432 total; ~40 by targeted filename grep; 7 read in full | ~12 receipt memos | — | ~9,390 not opened | **~1% by filename patterns.** Anchored on the freshest (ddm_oc1, r6cal, pc1, r1_dxi, finding1) — the live line — but the long tail is unscanned. |
| B. Memory corpus | MEMORY.md fully in context + established-findings references | ~8 headline findings (pose solved, seg solved, null-subspace, factorization) | tie-aware NO-OP, past-solves-rate-naive | most topic files | **Good on headlines, partial on topic files.** |
| C. Code (4 dirs) | ALL modules listed (~250); **~30 docstrings read** mapped to the 5 stages | 20 (table + runner-ups) | ~5 (invisibility/tie-aware limited to non-factor-2) | ~200 not docstring-read | **Complete listing; ~12% docstring-read, biased to stage-mapped names.** The `taskspace_g*`/`g111`/`g120` family (dozens of modules) is the exact-frontier receiver plumbing — scanned by name, NOT read; likely holds more stage-5 export assets. |
| D. Git (200 commits) | 200 log lines read | ddm_oc1, r6cal, pi1/pr130 intake, hb1 HOPE, pc1 | G111/G120/G121 exact-frontier churn (in-flight, not a clean plug-in) | — | **Full 200; the G1xx line dominates recent history and is exact-frontier receiver work, not yet a drop-in.** |

---

## TOP-10 with one-line integration instructions

1. **R1 dxi banked pose** — wire `--pose-carrier-xi-from-ckpt` into the composed archive's pose member; drop-in **−0.28 S** vs r6cal's 0.408 pose, at 7.2 KB. Highest measured leverage per byte in the repo.
2. **xi_pose_coder** — make it the serializer for #1's ξ (H derived free); guarantees the 43 KB redundant-H is never re-shipped.
3. **argmax-flip support extractor** — run `ddm_oc1_xi_temporal_measure.py` support mode to emit the 0.864% flip mask + RF dilation; this mask is the input to stages 3–4.
4. **context_partition_codec** — point it at the support-geometry / partition L* stream; replaces LZMA-over-labels; measure Σ N_ctx·H floor on the sparse support.
5. **uint8_lattice + tie_aware_preimage** — use as the RESIDUAL-VALUE feasibility certificate on the flip support (values that survive R+uint8+argmax exactly); do not expect free bytes on factor-2 (NO-OP proven).
6. **ddm_runtime_exporter/receiver** — the existing stage-5 spine; compose the new sparse members into its manifest rather than building a new inflate path.
7. **keyframe_codec** — crush the 81 MB frame-0 bootstrap to ~0.03–0.07 rate with 13 keyframes; frame0 is seg-free so this is pure pose+rate.
8. **warp_real_luma_frame0 / stratified_depth_warp** — synthesize frame0 from a keyframe + ξ for the pose axis (bit-parity to numpy fp64); pairs with #7 to shrink the bootstrap.
9. **predict_project_receiver** — the composed PREDICT→PROJECT stage-1+3 receiver; use its violated-constraint extraction to build the sparse residual on #3's support.
10. **movable_site_coder** — code Movable as sparse sites; compose with #4 so the whole-scene bitmap is never paid.

## The one honest caveat on "super optimal"
ddm_oc1 MEASURED that 2D-homography PREDICT is **NEGATIVE** on both axes (L2 +4.6%, task +116% vs copy).
So the PREDICT-stage assets (#8, #9, `ego_xi_trajectory`) are "super optimal" for the POSE/frame-0
sub-problem, NOT as a residual-shrinking predictor — the compounding lever is stage-3 SPARSE support
(#3→#4→#5), not stage-1. The pose plug-ins (#1, #2) are the only two assets that are *byte-closed,
measured, and immediately S-moving as drop-ins* on the current archive.
