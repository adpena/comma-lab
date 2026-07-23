---
title: Codex findings - DDM PT1 continuous-paint ceiling n600 execution
date_utc: 2026-07-23T20:50:28Z
lane_id: lane_ddm_pt1_continuous_paint_ceiling_20260723
research_only: true
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: PRIMARY_MECHANISM_BAR_PASS_BOX_BAR_FAIL
verdict_scope: "Exact typed PT1 arms on SHA-bound n600 inputs; formulation-only advisory evidence"
pointer_moved: false
main_landing_review_required: true
---

# Outcome

The hard camera-placement formulation **passes its preregistered mechanism
bar** but **fails the secondary 0.0142 box bar**. The appearance arms show that
the dominant measured correction is global amplitude statistics, not the
tested spectrum residual. This is not a geometry-family negative and not a
contest score.

| Arm | FIRST-RUNG | Errors | d_seg | Counted delta bytes | Disposition |
|---|---:|---:|---:|---:|---|
| PT1 native-grid flat-palette control | yes | 2,648,079 | 0.022448043823 | 0 | honest within-PT1 control |
| hard camera placement | yes | 2,592,874 | 0.021980065240 | 0 reused / 68,464 fresh | mechanism PASS; box FAIL |
| analytic coverage blend | yes | 2,470,714 | 0.020944502089 | 0 reused / 68,464 fresh | separate control |
| global amplitude-statistics match | yes | 1,016,725 | 0.008618884616 | 30 | strongest measured row |
| stratum-spectrum match | yes | 1,034,847 | 0.008772506714 | 186 | diagnostic only |

All five rows use pinned camera-side bicubic/uint8 R followed by the real
`SegNet.preprocess_input` bilinear downsample and frozen SegNet.

# Falsifiers

## Primary mechanism bar: PASS

- Independent boundary survival wall:
  `1,387,404 / 5,152,536 = 0.26926624093456114`.
- Required recovery fraction: `1 - wall = 0.7307337590654388`.
- Placement-attributable errors: **136,673**.
- Placement-recovered errors: **117,055**.
- Observed recovery fraction: **0.8564603103758607**.
- Observed residual wall: **0.14353968962413932**.

Because `0.8564603103758607 >= 0.7307337590654388`, the hard-placement
formulation passes its SHA-bound mechanism bar. The old 16% SINE-family prior
was not used.

## Secondary box bar: FAIL

- Threshold: `d_seg <= 0.0142`.
- Hard-placement result: `0.021980065240`.
- Signed residual: **`+0.007780065240`**.

The failure is scoped to this hard camera-placement formulation. It does not
kill analytic geometry or the broader geometry family.

# Four-way operational decomposition

| Mechanism bucket | Corrected sites | Verdict scope |
|---|---:|---|
| sub-cell placement | 117,055 | hard placement inside the target boundary band |
| BN/SE amplitude statistics | 1,578,514 | 30-byte global mean/variance arm after its flat-camera control; operational, not a one-layer causal claim |
| texture prior or region ERF | 7,233 | primary residual corrected only by the spectrum diagnostic |
| class interaction | 110,376 | hard-placement correction outside the target boundary band |

The buckets are disjoint operational attribution, not Shapley values. The
honest fork does **not** route this result as texture-dominated:

- texture-diagnostic sites: **7,233**;
- combined placement-correction sites: **227,431**;
- route: `MIXED_OR_PLACEMENT_FOLLOWUP`;
- geometry-family negative allowed: **false**.

# Curve provenance and rate ownership

The owner masks cover exactly the independent wall's 5,152,536 boundary
sites. Already-described sites remain zero-byte; freshly fitted sites carry the
complete exact-parseback SDWL1 object.

| Owner | Sites | Delta bytes | Hard corrected / introduced | Analytic corrected / introduced | 30B stats corrected / introduced | 186B spectrum corrected / introduced |
|---|---:|---:|---:|---:|---:|---:|
| already described | 1,043,712 | 0 | 20,598 / 9,908 | 29,211 / 8,232 | 182,124 / 31,326 | 178,361 / 30,547 |
| freshly fitted | 4,108,824 | 68,464 | 96,457 / 80,628 | 116,551 / 74,743 | 478,988 / 86,026 | 479,060 / 86,138 |

The fitted object is 68,464 bytes, exact-parseback true, payload SHA
`2b67caa997f353d1aee25b66737fcae1c0067deb92a0850401ce56f2f2537cab`,
and semantic SHA
`e7dee11d0fd162470bb206acca3c4667c79100cc41259d5d1ecb293e31e225f3`.

# Scorer-native profiles and trajectory

The receipt contains **2,660** detailed layer rows:
38 batches × 5 variants × 14 SegNet layers. Every row records channel group,
spatial band, static relative norm, Fisher-weighted delta, within-batch
trajectory delta, and the previous-batch endpoint delta where defined.

- All five variants are present in all 38 batches.
- All 37 cross-batch boundaries are present.
- Static and within-batch first divergence is the SegNet stem for every
  variant in every batch.
- Cross-batch first divergence is the stem for all 37 defined boundaries;
  only each variant's first batch is correctly marked unavailable.

This supports a stem-entry mismatch result. It does not prove that a
downstream-only relay correction is impossible under a different input
representation.

# Receipt defect found after execution

The immutable measurement receipt's nested
`rate_doctrine.streams[*].audit_triple` statuses still say
`PENDING_N600_EXECUTION`, although the top-level rows and all 38 batch ledgers
are measured. The cause was a static prepared-state helper. The receipt was
not mutated. A follow-up code fix makes future execute receipts emit
`MEASURED_N600_ADVISORY` and adds a regression test. Commit
`38a62781b6ff3c89f243657719a2071b42ed90eb` preserves the exact executed
source named by the immutable receipt.

# Pose directive disposition

MAIN's `2026-07-23T19:52:35Z` correction is consumed as a constraint:
PoseNet's useful photometric band must be measured independently and is likely
lower-frequency than SegNet's boundary band because PoseNet uses BN/LayerScale
and large-kernel reparameterized convolutions, not SegNet-like SE behavior.

Per-frequency Pose response was **deferred, not inferred**. It was not cheap
inside this arm because `pose_secondary_enabled=false` and there is no
independently custodied decoder-side xi-to-camera warp or structured two-frame
carrier. Reusing the SegNet period-4/6/8 result as a Pose claim would be fake.

# Doctrine directive-consumption table

| Directive | Disposition | Durable application / next owner |
|---|---|---|
| 1 closure axiom | CONSUMED | wall classified as representation mismatch; no missing-data claim |
| 2 scorer-native coordinates | PARTIAL | SegNet 14-layer static/trajectory census landed; PoseNet deferred to the enabled pose leg |
| 3 non-linguistic axes | PARTIAL | amplitude and spectrum arms measured; phase/contrast/pose carrier remain owed |
| 4 derive before measure | PARTIAL | pinned R and prepared passband reused; full BN/kernel analytic atlas remains `at1` |
| 5 factored total influence | DEFERRED | pair/site costate and exact total lambda remain `at1` |
| 6 measure-solve-represent-realize | PARTIAL | measure complete; non-additive composition and relay solve route to `menu1` / `rs1` |
| 7 instrument modernization | DEFERRED | atlas/costate unification remains `at1` |
| 8 philosophy checklist | CONSUMED | exact n600, resumability, immutable stages, FIRST-RUNG labels, no score claim |
| Pose low-frequency correction | DEFERRED-EXPLICIT | enabled PoseNet frequency sweep only after custodied xi/structured-paint surface exists |

# Verification and custody

- Authority prompt SHA:
  `f30c7daa8e2024a5e12f51757904e832e1c39ba72d708968546fe850f974ff76`.
- Prepared receipt SHA:
  `3977b5ab3f6fef7c2d7739be578a7024be54ce17903b68fa305f576c44df9a60`.
- Wall receipt SHA:
  `b056030ed38ac36d2643b60929053e029cf6931d63b0649cef59451f57d9dee3`.
- Measurement receipt SHA:
  `83e06ef47027a4997e598c824c8bb36b2185d4225ca8fcfaf8a8568fbff4b4b9`.
- Aggregate re-derivation from 38 batch JSON ledgers: PASS.
- Stage-transition and curve-owner conservation: PASS.
- Focused plus component suites after the metadata regression fix:
  **138 passed, 4 skipped**.
- No paid dispatch, remote execution, GPU, training, PoseNet evaluation,
  contest CPU/CUDA evaluation, score claim, or pointer mutation occurred.

# STORES CONSULTED

- `CLAUDE.md`
- `AGENTS.md`
- `docs/operating_manual_craft_handoff.md`
- `.omx/research/ddm_scorer_native_doctrine_and_synthesis_20260723.md`
- current canonical lane, task, frontier, and subagent ledgers
- latest Codex findings/session and Claude council/design memos
- per-arm inbox through `2026-07-23T19:52:35Z`
- fleet broadcast through `2026-07-21T13:15:53Z`

# Triality and pointer delta

- **DSL:** explicit execution config binds target, scorer, independent wall,
  payload sizes, and `research_only=true` / `score_claim=false`.
- **DAG:** independent wall → hard/analytic/stats/spectrum arms → pinned R →
  frozen scorer → per-layer/trajectory ledgers → mechanism/box falsifiers.
- **Equations:** recovery law, exact transition conservation, signed box
  residual, and counted SDWL1/amplitude/spectrum rates all close.
- **Pointer:** unchanged.

# MAIN landing review required

Before merging, MAIN must independently review:

1. the cached-label and E2-control premise corrections;
2. the wall definition and mechanism recovery arithmetic;
3. the disjoint four-way attribution and non-texture honest fork;
4. the 0-byte reused versus 68,464-byte fresh curve ownership;
5. the immutable receipt's stale nested rate-doctrine statuses and the
   follow-up prevention patch;
6. the explicit Pose-frequency deferral;
7. preservation of the ignored local endpoint NPZ checkpoints until MAIN has
   completed landing review; and
8. that no advisory row is promoted into a contest score or pointer claim.
