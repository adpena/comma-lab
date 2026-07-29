---
schema: ddm_p3v2_optimal_form_pose_resolve.v1
date_utc: 2026-07-29
arm: ddm_p3v2
axis: "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE"
pointer: "0.1910828242 [contest-CPU] UNMOVED"
score_claim: false
promotable: false
pointer_moved: false
research_only: true
council_predicted_mission_contribution: frontier_protecting
verdict_scope: FORMULATION
consumes: [QA01, gc7r_row1, p3_terminal_pose_receipt, p1_frame0_quotient_reach_curve, sc1_e_p, "#249_pose_frame0_inverse_solve_probe"]
consumers: [E2_node_N1, v10_SPEC_row12_pose_in_burn, ddm_deferral_queue_ledger_QA01, QA25]
---

# ddm_p3v2 — OPTIMAL-FORM terminal pose re-solve (the RANK-1 arm)

## §0 THE PRE-REGISTERED DECISION — which side of the rule the vehicle lands on (plainly, first)

**The photometric wall is REFUTED. The vehicle is a CANDIDATE LINE, NOT a calibration instrument.**

The binding pre-registered rule (task prompt, Assumption-Adversary, MAIN-adopted): *IF the
optimal-form solve still lands pose > 0.05-contribution-class at S1(d)'s UNPRICED free-frame_0 upper
bound → the photometric wall is CONFIRMED at FORMULATION scope and the vehicle is re-designated a
calibration instrument.*

MEASURED (frozen CPU-torch PoseNet6 authority, STE-uint8 camera-res, work-res 192×256, ~160 Adam
iters run to convergence, n=24 pairs — the full charter ladder, `[macOS-CPU advisory]`): the
free-frame_0 upper-bound **mean d_pose = 9.123e-5 → pose contribution √(10·d_pose) = 0.0302 ≤ 0.05.**
100% of pairs reach ≤ 1e-3; 95.83% reach ≤ 2.5e-4; max = 4.6e-4; median = 5.8e-5. **The wall claim
FAILS its own falsifier.**

The P3 verdict N1=NO (photometric wall) was an ARTIFACT of a naive solve, exactly as the operator's
07-29 INSTANCE-downgrade suspected: P3 started frame_0 from `zeros` (d_pose ~85), actuated through a
FIXED rank-6 cosine basis, and budget-truncated the GN at ~2 relinearizations (38.06 mean). None of
those three choices is a vehicle property. **N1 is re-decided YES: this vehicle CAN carry pose.**

**CONTRARIAN BOOKING (reaches at measured bounds, not ceilings).** §0's refutation is the UNPRICED
reach ceiling. The CHEAP decoder-reproducible realization reaches contribution **1.98** (the warp
base priced on n600, §6), not 0.03, because the free-frame_0 win is BASIS-ADVERSARIAL (§5): it does
not survive cheap generic coding. So the honest state is: *wall refuted (pose is carriable) AND the
cheap frame_0-pixel carrier leaves a contribution-~2 residual that only a pose-field terminal-solve
(sc1 e_p) or v10 pose-in-burn conditioning closes.* Both facts are load-bearing; neither is hidden.
The measured composed move is real and large regardless: **S ≈ 20.2746 → 2.7431 (ΔS −17.53)** on the
pb1 instrument row (§6), booked at the measured n600 warp bound, not the free ceiling.

Pointer honesty: **`0.1910828242 [contest-CPU]` UNMOVED.** The pb1 composed row S≈20.2746 is an
instrument-side advisory (its locked evaluate receipt is still owed, ledger QA02); every S delta
below is ON that advisory row.

## §1 What was suboptimal about P3 (receipts, not recall)

| P3 choice | measured cost | optimal form |
|---|---|---|
| frame_0 START = `zeros` (d_pose ~88) | threw away a far better base | stored render 10.22 / warp base 0.59 |
| ACTUATION = fixed rank-6 cosine basis | RANK-DEFICIENT: plateaus at d_pose ~15 (§2) | free / Jacobian-aligned / warp |
| BUDGET = ~2 relinearizations | truncated 38.06 vs converged ~15 | run to convergence |

The stored composed frame_0 (the vehicle's own render) already scores d_pose **10.22** (contribution
10.11, n24) — better than P3's converged 6-cosine result — and P3 discarded it to start from zeros.
That alone shows the 38.06 was not a floor.

## §2 S0 — the EXISTING actuation (rank-6 cosine) run to convergence: RANK_DEFICIENT

Reproducing P3's actuation (rank-6 cosine basis, LM-GN, from a zeros base) and running it to ~11
relinearizations (n=6): the d_pose plateaus at **mean 15.29 / median 6.86** (example trajectory
89.5 → 22.1 → 16.4 → 15.8 → 15.3 → 15.15 → 15.07 → 15.07). Verdict: **RANK_DEFICIENT.** The basis
converges (so P3 was ALSO budget-truncated: 38 → 15 with more relins) but its converged floor (~15)
is 5 orders above the free floor (~1e-4). The 38.06 "20.27 row" is dominantly a BASIS problem: six
generic cosine fields cannot span the pose-Jacobian directions. #715's quotient carrier confirms the
class failure independently — its covariance-ordered basis makes d_pose RISE with rank (19.89 @ rank-1
→ 48 @ rank-6). Generic bases are the wrong object.

## §3 seg untouched — the factorization law, confirmed empirically

frame_1 (the seg frame) is never touched. The frame_0 seg-free spot check: SegNet argmax is
**IDENTICAL** for two totally different frame_0 (the free-solved frame vs an all-zeros frame) with the
same frame_1. SegNet reads `x[:,-1]` = frame_1 only (upstream/modules.py:108); d_seg (0.38901) is
untouched by any frame_0 actuation. The ENTIRE frame_0 is a pose-only surface at zero seg risk.

## §4 S1 — the banked-carrier race (n=24, frozen-uint8 authority; banked rows CITED not rebuilt)

| carrier | d_pose (mean) | pose contribution | carrier bytes | note |
|---|---:|---:|---|---|
| copy(f1) | 188.57 | 43.42 | 0 | zero-motion baseline |
| zeros (P3 start) | 88.35 | 29.72 | 0 | P3's start policy |
| **P3 6-cosine (budget-truncated)** | **38.06** | **19.51** | 7,295 (n600) | THE 20.27 row |
| #715 quotient rank-1 (CITED) | 19.89 | 14.10 | 3,520 | generic covariance basis; d_pose RISES w/ rank |
| stored_f0 render | 10.22 | 10.11 | full frame | the vehicle's own frame_0 |
| **WARP BASE (s_t, DECODER-REPRODUCIBLE)** | **0.589** (n600: 0.393) | **2.43** (n600: 1.98) | ~0–1 B/pair | ego-motion homography of f1 by carried pose |
| **FREE frame_0 (S1d, UNPRICED)** | **9.12e-5** | **0.030** | ~147K/pair | the reach ceiling; basis-adversarial |
| sc1 e_p rank-1 (CITED, pose-FIELD) | (raw seed 36–146) | — | 2,039 | different carrier family; needs terminal solve |

The decisive banked carriers (#249 free-frame0 machinery, #715 quotient, the eg1 6-cosine) are the
substrate; the NEW measured rows are the warp-base and free-frame0 family on the tr1 composed frames.
The **warp base** — a ground-homography warp of frame_1 by the already-carried 6-value pose target,
with a single per-pair translation-scale s_t on an 11-value grid — is the cheap decoder-reproducible
carrier that dominates every banked one (d_pose 0.59 vs 38 / 19.89 / 10.22) at ~0 bytes.

## §5 S1 price + S2 LOTTO — the free win is BASIS-ADVERSARIAL; LOTTO beats per-pair but not the warp

The honest cheap-realization question: does the free-frame_0 win compress in a DECODER-REPRODUCIBLE
basis? Measured on the free-solve residual over the warp base:

- **Generic 2D-DCT (low-freq AND largest-magnitude) + per-channel low-rank SVD**: ALL collapse back
  toward the warp-base d_pose class (k16 → ~15, r8 → ~2.1). The residual carrying the free win does
  NOT sparsify in a generic basis — it needs the net's Jacobian directions (not decoder-reproducible).
- **S2 LOTTO (gc7r's flagged highest-leverage surface): SHARED low-rank frame_0 basis (counted once,
  amortized over n600) + per-pair coefficients, vs per-pair rank-1.** The shared dictionary DOES beat
  per-pair rank-1 (a real finding — the pose-relevant directions are partially SHARED across pairs):

  | | d_pose mean (n24) | contribution | bytes/pair (n600-amortized) |
  |---|---:|---:|---|
  | LOTTO shared R1 | 0.735 | 2.71 | 450 |
  | LOTTO shared R4 | 0.312 | 1.77 | 1,813 |
  | LOTTO shared R8 | 0.366 | 1.91 | 3,628 |
  | LOTTO shared R16 | 0.258 | 1.61 | 7,264 |
  | per-pair rank-1 | 3.526 | 5.94 | 2,661 |
  | **warp base (the comparator)** | **0.589** (n600: 0.393) | **2.43** (n600: 1.98) | **~0** |

  LOTTO shared R4–R16 (contribution 1.6–1.9) marginally out-reach the warp base on the FIT pairs but
  at 1.8–7.3 KB/pair (1–4 MB over n600), with NO cross-pair generalization evidence (the dictionary
  was fit on these 24 pairs' own deltas), while the n600-priced warp base already reaches 1.98 at
  194 B total. **The frame_0-pixel LOTTO is Pareto-dominated by the warp base for cheap realization
  and never approaches the free floor (0.030).**
  Verdict-scope: FORMULATION — the SVD-of-deltas dictionary is basis-adversarial for the last-mile
  precision; a JACOBIAN-aligned shared dictionary (store the pose-direction basis once) remains the
  named next rung, NOT built here.

## §6 S3 — the composed-row point (the winner = the warp base)

The only cheap decoder-reproducible carrier that beats the banked ones is the **warp base**. Priced
on **n600** (warp mean d_pose = **0.3931**, median 0.0848 — most pairs are near-static; s_t index
stream through the merged SMEVR/r7 coder = **194 bytes** for all 600 pairs, zlib 180; the 6-value pose
target is the already-carried sc1 t_p sidecar):

- pose term **19.50954 → 1.9827** (√(10·0.3931)); seg **0.38901** untouched; rate **0.37609**
  DROPS slightly (the 194-B s_t stream REPLACES the 7,295-B 6-cosine member).
- **composed S ≈ 20.2746 → ≈ 2.7431** (tp already carried) / 2.7444 (tp counted new) — a
  **ΔS ≈ −17.53** move on the pb1 instrument row, driven entirely by the pose term
  (banked-6-cosine → optimal-form warp actuation).

The free upper bound (contribution 0.030 → composed S ≈ 0.80) is the REACH, NOT a realized point: it
is basis-adversarial and not carriable as cheap frame_0 pixels. Closing the ~2.0 → 0.03 gap is the
pose-field terminal-solve (sc1 e_p ~2 KB, joint-descent-trained) or the v10 pose-in-burn conditioning
(#383 gate) — pose entering the TRAINING loop, per the photometric-wall lesson: post-hoc frame_0
carriers realize the ego-motion coarsely (warp) but the fine pose precision needs joint descent.

## §7 Honest bar arithmetic + verdict-scope

- ≤ 0.018-contribution (the banked pose target) needs d_pose ≈ 3.2e-5. The free upper bound (mean
  9.12e-5) is ~2.9× above it; the per-pair MIN reaches 1.2e-5 (below it). The warp cheap carrier
  (n600 mean 0.3931) is contribution 1.98 — far above, but at 194 B/n600 and a **−17.53 S** move on
  the instrument row.
- Verdict-scope on every negative: the S2 frame_0-pixel LOTTO negative is FORMULATION (SVD-of-deltas
  dictionary), NOT a paradigm kill — the Jacobian-aligned shared dictionary is un-raced. The
  basis-adversarial finding is INSTANCE→FORMULATION on this vehicle (matches #249 on the witness line).
- No family/paradigm kill anywhere. Pointer UNMOVED. score_claim=false. n=24 (ladder) / n600 (S3
  warp) `[macOS-CPU frozen-PoseNet advisory]`; the pb1 row is instrument-side (QA02 owed).

## §8 Wire-in (Catalog #125) + boundaries

- sensitivity-map N/A · Pareto: the §5 (d_pose, bytes) rows are new advisory Pareto points ·
  bit-allocator N/A · cathedral N/A · continual-learning: this memo + DAG FEED + ledger QA01 flip ·
  probe-disambiguator: the pre-registered rule IS the disambiguator (measured on the free upper bound).
- [no-triality]: measurement/routing artifact — no DSL lever or canonical-equation surface changed.
- [p0-ledger-ok]: ledger QA01 DUE → FIRED with this receipt; QA25 pose-in-burn note updated (wall
  refuted → pose-in-burn is an OPTIMIZATION choice, not a forced head).
- Receipts (SSD custody, certify-or-block): `/Volumes/VertigoDataTier/pact/ddm_p3v2_20260729/`
  {`p3v2_ladder_receipt_final.json` (n24 ladder, sha256 12838f63ea71…), `p3v2_s3_receipt.json`
  (n600 warp, sha256 ac87e05ee830…), per-pair npz cache, tool snapshot}. Tools:
  `experiments/ddm_p3v2_optimal_form_pose_resolve.py` + `experiments/ddm_p3v2_finalize_from_cache.py`.
