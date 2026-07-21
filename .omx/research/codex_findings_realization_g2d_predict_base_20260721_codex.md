# Codex findings — Task #578 G2d contextual predict-base realization

`lane_id=lane_realization_g2d_predict_base_578_20260721` ·
`verdict=MEASURED_CONTEXTUAL_PREDICT_BASE_NOT_ADMISSIBLE` ·
`[macOS-CPU advisory]` · `score_claim=false` · `promotion_eligible=false` ·
`pointer=0.1910828242 [contest-CPU] UNMOVED` · `MAIN_REVIEW_REQUIRED=true`

## Outcome first

The specified sequential contextual instance is not admissible.  At n600 the
unchanged `predict_project_realization_admissibility_v1` gate passed n600,
factor-2 uint8 exactness, double-decode identity, and receiver-derived RGB, but
failed whole-description semantic exactness, Pose-tube containment, and zero
added seed bytes.  This is a formulation-scoped negative, not a family negative.

The most useful positive result is a second exact margin-survival replication:
all 1,580 positive-margin declared writes survived and none of 1,608
nonpositive-margin writes survived.  That sharp separator held at n16, n64, and
n600.  Declared-write survival therefore must not be conflated with the much
stronger whole-description condition, which remained 0/600.

## D1 — semantic realization

- Whole represented-cell-field equality after frozen SegNet: `0/600` pairs.
- Declared writes: `1,580/3,188` survived; only `69/600` pairs preserved every
  declared write.
- Margin contingency: positive `1,580/1,580`; nonpositive `0/1,608`.
- Class survival: Road `870/1,379`; Lane `709/1,807`; class 2 `1/1`; class 4
  `0/1`.
- Stratum survival: boundary codim-1 `1,578/3,183`; critical event `0/3`;
  movable track `2/2`.
- Mean d_seg: realized versus frozen target `0.2870645395914714`; description
  versus target `0.3434977213541667`; realized argmax versus description
  `0.5626695081922743`.
- Factor-2 uint8 exactness and double-decode identity were both `600/600`.

Interpretation: the bounded lattice did exactly what the prior law predicts.
Moving a declared site into a positive target-class cell was sufficient for that
write, but projecting only initially violated declared sites cannot realize the
entire represented semantic field.  The data falsify any inference from sparse
write survival to whole-description exactness.

## D2 — real motion and Pose

The two decoded frames were not identical by construction, closing the prior
zero-motion confound, yet Pose-tube containment remained `0/600`.  Mean realized
d_pose versus the frozen target was `153.98256580439468`; mean squared debt
outside the declared tube was `148.79828058518876`.

The negative is narrow.  Within-pair motion used the exact banked G1 target;
cross-pair motion used the nearest-target-pair `gt_poses[t]` proxy because no
exact banked cross-pair target exists.  This result does not close exact
cross-pair motion, learned residual motion, or a pose-factorized xi child.

## D3 — counted bytes

The exact scorer-plane frame-zero bootstrap compressed to `368,201` Brotli-11
bytes, already `151,979` bytes above the entire `216,222`-byte target box.
Adding the settled seed baseline (`78,969` bytes) and 1,967 sparse exception
records (`19,151` bytes) produced `466,321` bytes total, or `250,099` bytes over
the box.  The bootstrap is explicitly video-derived and counted; treating it as
free would be false authority.

## D4 — sequential replay

The source-hash-closed decoder replayed all 1,200 frames in
`66.62605445901863` seconds on `[macOS-CPU advisory]`, invoked no scorer, and
produced frame hash tree
`5f73e388c960f13c3b7fa37431fcabb76f1efdfcbe7782f5ac7ac3a861e3c5d5`.
All 600 pair stages and exception sidecars are preserved on SSD; all 600 pairs
double-decoded identically.  Wall time is engineering telemetry, not an
admission or score axis.

## D5 — unmodified admission

`predict_project_realization_admissibility_v1` returned false with exactly:

- failed: `semantic_cells_to_rgb_exact`, `pose_within_declared_tube`,
  `zero_added_seed_bytes`;
- passed: `n600`, `factor2_uint8_exact`, `double_decode_identical`,
  `receiver_derived_rgb`;
- counted additional seed bytes: `387,352` (bootstrap plus exceptions).

No equation, threshold, score pointer, or promotion state was changed.

## Reusable system intelligence and bounded next route

Keep the scorer-free sequential decoder, atomic per-pair resume format, strict
exception parse/re-encode checks, source closure, and the positive/nonpositive
margin contingency as regression surfaces.  Do not reuse the exact bootstrap
as an admissible payload, the cross-pair motion proxy as exact motion, or the
one-round prototype projection as a family verdict.

The measured blocker ordering is concrete:

1. eliminate or generatively derive the frame-zero bootstrap before any
   micro-optimization—the bootstrap alone cannot fit the box;
2. require an exact or explicitly learned cross-pair xi surface before treating
   the Pose negative as formulation-complete;
3. replace naive one-round prototype movement with the registered corrected
   inner-Jacobian/secant/QP projection and retain only positive-margin writes;
4. if residuals remain, rank only necessary edge/saddle obligations by the
   registered Fisher/margin EV field and rate break-even, using a genuine
   curvelet/shearlet carrier rather than a blanket RGB exception stream.

Those are reactivation criteria, not a GO or a score claim.

## Custody

Full SSD receipt:
`/Volumes/VertigoDataTier/pact/evidence/realization_g2d_20260721/canonical_v1/receipt.json`
SHA-256 `f7452ae660d6529f49c1de1e44826860b52a840c9459eea4af40d6737bcf6394`.
The compact repository receipt records dependency, scorer, seed, cache,
bootstrap, stage-tree, exception-tree, and prefix-checkpoint hashes.

## STORES CONSULTED

Delegated authority SHA `8bcc0bbc9534197d14b6b514fc447d3b334ce4f8a2a1eef6bb100eb9c4d8c1fc`;
`CLAUDE.md`; `AGENTS.md`; craft handoff; v7.5 section 8; Task #597
PREDICT-to-PROJECT interfaces; G1/G3 custody receipt; G2/G2b/G2c code,
receipts, equations, tests, DAGs, and reuse manifests; predictor round-1 and
round-2 code/findings; r1b6 negative; frozen seed/cache/scorer; lane/progress
state; n16/n64/n600 SSD artifacts; both inboxes through
`2026-07-19T19:48:01Z`.

## Triality and landing boundary

- DSL/code: `tools/measure_realization_g2_lattice.py` composes the settled
  receiver, G1 xi transport, lattice, #557 coder, and unchanged admission law.
- DAG: `realization_g2d_predict_base_DAG_FEED_20260721.md` routes the measured
  blockers and regression surfaces.
- Equations: `predict_project_realization_admissibility_v1` was invoked
  unmodified; the measured Fisher/margin separator is reported, not promoted
  into a new equation by this branch.

MAIN must review the branch diff and merge boundary before any result is treated
as landed repository truth.
