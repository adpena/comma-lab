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

## Late operator addendum — prediction prior is reuse

The exact-I-frame rate blocker is not a contextual-family blocker.  After the
operator routed the already-built openpilot per-class charts, frozen-scorer
constants, and rank-4 head/VJP surfaces, a separate pair-0 race measured three
base priors through the same exact factor-2 receiver and native scorer:

| frame-0 prior + one G1 refinement | counted base bytes | headroom vs 216,222 | d_seg vs target | description d_seg | realized-vs-description d_seg | Pose d_pose | declared writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact source I-frame control | 447,170 | -230,948 | 0.00747681 | 0 | 0.00747681 | 185.1728 | 5/8 |
| keyframe-class description | 78,990 | +137,232 | 0.47434489 | 0.34750875 | 0.63923645 | 183.0988 | 1/8 |
| openpilot per-class geometric solve | 121,128 | +95,094 | 0.28089905 | 0.01121012 | 0.28408813 | 179.2494 | 0/8 |

The openpilot base is a real improvement over the keyframe-class RGB mapping:
it reuses the counted 835-byte zlib static chart, 41,303-byte Brotli lane
polynomial chart, 21-byte frozen-scorer palette, G1 geometry, static hood,
movable tracks, and five protected class sites.  All compressed packets parsed
back exactly; all three candidates were factor-2 exact and double-decode
identical with zero decoder scorer calls.  The openpilot and keyframe bases fit
the box before correction, so “eliminate the bootstrap” is now closed at the
base-prior level.  Neither is semantically or Pose admissible.

The required rank-4 correction was deliberately not faked.  Current VJP
custody is `COMPLETE_N600` (`600/600`, no refusal), superseding the earlier
24-pair snapshot.  But those pullbacks bind source/native winner-rival
arrangements; the three generated candidates disagreed with the source winner
chart at 1,470, 93,260, and 55,227 pixels respectively.  The rank-4 closed form
is exact in the 144-dimensional penultimate feature quotient, while the
candidate-arrangement realized-backbone secants and receiver-closed QP remain
absent under blocker
`R1B2_RANK4_FIRST_ORDER_REALIZED_SECANT_CUSTODY_ABSENT`.  Therefore no
rank-4 RGB exception stream, rounding-ball magnitude, or admission claim was
emitted.  The 32/44/44-byte #557 rows in the race receipt are only syntactic
full-prototype upper bounds, not semantic corrections.

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

In the original n600 formulation, the exact scorer-plane frame-zero bootstrap compressed to `368,201` Brotli-11
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

1. preserve the measured openpilot base prior (`121,128` bytes total before
   correction) rather than returning to the exact I-frame;
2. compute candidate-arrangement rank-4 first-order plus realized-backbone
   secants and close the receiver QP; do not reuse source-arrangement normals
   across the measured 55,227-pixel winner mismatch;
3. require an exact or explicitly learned cross-pair xi surface before treating
   the Pose negative as formulation-complete;
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

Late frame-0 race receipt:
`/Volumes/VertigoDataTier/pact/evidence/realization_g2d_20260721/prior_race_v2/receipt.json`
SHA-256 `0aa7e9cc86b1eb11ef3816b2bcf95182ca504d7e514107112f22a52bfee7412e`;
projection blocker SHA-256
`f4a7e36872d14174e082610d8317372ba5782a9de25b8de3130200b3c7d1bf0b`.

## STORES CONSULTED

Delegated authority SHA `8bcc0bbc9534197d14b6b514fc447d3b334ce4f8a2a1eef6bb100eb9c4d8c1fc`;
`CLAUDE.md`; `AGENTS.md`; craft handoff; v7.5 section 8; Task #597
PREDICT-to-PROJECT interfaces; G1/G3 custody receipt; G2/G2b/G2c code,
receipts, equations, tests, DAGs, and reuse manifests; predictor round-1 and
round-2 code/findings; r1b6 negative; frozen seed/cache/scorer; lane/progress
state; n16/n64/n600 SSD artifacts; both inboxes through
`2026-07-21T12:27:12Z`; frozen rank-4 prototype receipt; terminal n600 VJP
campaign; M1 band/inner-Jacobian blocker; Task 595 lane chart and Task 578
static-chart artifacts; pair-0 prior-race packets and receipts.

## Triality and landing boundary

- DSL/code: `tools/measure_realization_g2_lattice.py` composes the settled
  receiver, G1 xi transport, lattice, #557 coder, openpilot chart predictor,
  counted frozen-scorer constants, and unchanged admission law.
- DAG: `realization_g2d_predict_base_DAG_FEED_20260721.md` routes the measured
  blockers and regression surfaces.
- Equations: `predict_project_realization_admissibility_v1` was invoked
  unmodified; the measured Fisher/margin separator is reported, not promoted
  into a new equation by this branch.

MAIN must review the branch diff and merge boundary before any result is treated
as landed repository truth.
