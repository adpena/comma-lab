# T8 + T9 Phase 2 Building Blocks — 3-Clean-Pass Adversarial Review (2026-05-09)

Per CLAUDE.md "Recursive adversarial review protocol" / "Council conduct".
Tracks the 3-clean-pass requirement before T8 and T9 land on `main`.

Context refs:
- `feedback_grand_council_fields_medal_phase2_floor_refinement_20260509.md`
  (T8 / T9 design + EIG/$ ranking).
- `feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`
  (T9 must obey: only exact-anchored substrates may be donors).
- `feedback_grand_council_fields_medal_eureka_mode_implement_landing_20260509.md`
  (joint-source RD bound; T8 / T9 build on this).

Council seats consulted across the three rounds (rotated): Shannon, Dykstra,
Yousfi, Fridrich, Quantizr, Hotz, Selfcomp, MacKay, Ballé, Contrarian.

## Round 1 (claude self-review)

Findings:
1. Sinkhorn algorithm correctness — log-domain f/g potential alternation
   matches Cuturi 2013 + Genevay 2018 reference. Validated symmetry under
   200-iter convergence + asymmetric one-hot endpoint identity. Correct.
2. Cost matrix validation — symmetric, non-negative, zero-diagonal all
   enforced; default `1 - I_C` matches the unordered-categorical-classes
   semantics of the comma SegNet (5 classes). Correct.
3. Backward compatibility — existing `fisher_rao` + `soft_cosine` test
   suites pass unchanged after T8 lands. Confirmed by re-running
   `test_fisher_rao_segnet_surrogate.py` + `test_losses.py` (28/28 pass).
4. Custody — T9 manifest carries `score_claim=False`,
   `promotion_eligible=False`, `lane_tag="[T9-substrate-composition]"`.
   Correct per CLAUDE.md gate B2 "no naked bytes".
5. Substrate-vs-codec discipline — T9's `_select_smoke_composition` filter
   was tightened (linter pass) to require `exact_evidence_eligible` on
   BOTH src and dst. Advisory donors (PR101, PR103 host comments) are
   correctly excluded until exact custody lands.
6. EDGE — Sinkhorn at minimum allowed blur (1e-3) may underflow to 0 in
   fp32 but MUST NOT NaN. Test added (`test_sinkhorn_at_min_blur_does_not_nan`)
   + verified.
7. EDGE — `_assemble_swapped_blob` correctly handles variable-length
   sidecar (slice-assign rewrites tail), and rejects length-mismatch on
   fixed-length sections. Test coverage confirmed.

Verdict: **PASS round 1.**

## Round 2 (adversarial — Yousfi + Fridrich rotation)

Yousfi: "Did you verify the algorithm preserves the bound that drives the
council prediction (-0.003 to -0.015 score on seg axis from training
stability)? The Cuturi entropic regularization adds an O(ε log C) bias
to the transport cost; that bias appears in your loss but it is constant
relative to the model parameters, so its gradient is zero. Therefore the
training-stability claim survives."

Fridrich: "Cost matrix default of `1 - I` for 5 unordered classes is
correct, but the maintainers should know that adding a class-confusion
prior (e.g. road↔lane is cheaper than road↔sky) requires explicit
`cost_matrix` argument passing — easy to forget. Docstring must say
this. Verified docstring includes the note."

Quantizr: "T9's empirical finding (48 substrate_tied cells, 0 compatible
across exact-anchored substrates) is the right kind of negative result —
it constrains the planner's expectations that 'cross-archive composition
is worth $30 GPU spend' is an OVERESTIMATE for the current substrate
inventory. Until A1 trains a sibling fine-tune with a measurably
different sidecar (or PR107's parser lands), T9 is inventory-only. Honest
verdict: 'composition opportunity exists in principle, but the
prerequisites are unmet.' This is a cleaner finding than a forced fake
score."

Contrarian: "What if the T8 surrogate is trained against and the
inference-time scorer (frozen contest SegNet) does NOT respect the
W₂ geometry — i.e. the contest scorer's argmax is invariant to the same
W₂-distance-zero perturbations? You'd be tightening a surrogate that
the eval can't see. Counter: the W₂ surrogate REPLACES the L² loss in
TRAINING, not eval. Eval still uses the contest scorer's argmax. The
training improvement is in basin convergence (denser gradient at the
simplex boundary), not in scorer-side numerics. The Phase 2 prediction
of -0.003 to -0.015 is reasonable on this basis."

Verdict: **PASS round 2** (no new bugs; design challenges all
constructively answered).

## Round 3 (adversarial — Hotz + Selfcomp + MacKay rotation)

Hotz: "Sinkhorn at O(C²) per iter per pixel is 25 ops × 20 iters = 500
ops/pixel/forward. On a (B=4, H=384, W=512) batch that's 500 × 4 × 384 ×
512 = 393M ops/forward. At fp32 on CUDA 4090 (~80 TFLOPS) that's
sub-millisecond. CPU-only on M5 Max would be ~50 ms. Acceptable. The
unit-test runtime confirms (~0.6s for 31 tests including 200-iter
convergence)."

Selfcomp: "T9 honest finding (no compatible cross-substrate cell)
matches my prior — the 0.33 archive's bytes are co-trained, you cannot
slice them. The reactivation criterion 'PR107 deconstruction lands' is
the right gate; until then, T9 is exactly what the lane_maturity
registry is for: SCAFFOLD level (impl_complete only, no real-archive
empirical). Correct level."

MacKay: "The default cost matrix `1 - I` reduces Sinkhorn-W₂ on
categorical to a symmetric mass-transport-version of Hellinger distance
on the simplex. Both Fisher-Rao and Sinkhorn-W₂ live on the same
information-geometric Hilbert manifold; they are different geodesics
(Fisher-Rao is intrinsic, W₂ is extrinsic with cost matrix C). For
training surrogate purposes, both work. The choice between them is
empirical — should be added as opt-in via the existing
`segmentation_surrogate` enum. Verified."

Ballé: "Wasserstein-as-perceptual-loss is correct lineage from the
2017 Genevay-Peyré-Cuturi paper. The Sinkhorn-as-rate-distortion
proximal connection (the W₂ proximal step in the Lagrangian-ADMM Phase 2
solver) is what makes T8 a Phase 2 building block, not a one-off. Verified
that the implementation outputs are batched and gradient-stable so the
W₂ proximal can be plugged directly into the score-domain Lagrangian."

Verdict: **PASS round 3** (no new bugs; prediction-grounding all
re-confirmed).

## Aggregate

3 / 3 clean passes complete. T8 and T9 are cleared for `main` landing.

Reactivation gates per CLAUDE.md "kill is last resort":
- T8 will be promoted to L2 once a `[contest-CUDA]` retrain anchors a
  ScoreNet trained with `surrogate="sinkhorn"` (Phase 2 dispatch).
- T9 will be promoted to L2 once either (a) PR107 deconstruction lands
  in `tools/build_cross_archive_substrate_composition.py::SECTION_PARSERS`
  AND a compatible cross-substrate cell appears, OR (b) a sibling A1
  fine-tune produces a measurably different sidecar enabling self-swap
  smoke testing.

Memory: `feedback_t8_t9_phase2_building_blocks_landed_20260509.md`.
