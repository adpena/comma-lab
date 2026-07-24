# Codex findings — DDM TS1 typed-stream schema and G1 reprice

Date: 2026-07-24
Axis: local CPU structural verification; Torch/OpenMP threads pinned to 4
Authority: delegated `ddm_ts1_typed_stream_schema_g1_reprice` arm
Score claim: `false`
Pointer: `0.1910828242 [contest-CPU]` — **UNCHANGED**
Promotion authority: none; **MAIN landing review required**

## Verdict

`PASS_TO_MAIN_REVIEW_WITH_PREMISE_CORRECTION`.

The five-type schema is executable and its named consumers fail closed. The
requested G1 reprice does **not** reveal reclaimable bytes in the selected n600
knees: the measured mispricing is **0 bytes**. Source and receipt inspection
falsified the premise that G1 had counted generic vocabulary/operator code.
The existing harness already counted only video-derived production payloads,
production framing, and the envelope header. Per-clip parameters remain
counted even when their generic receiver operator is free.

This is an INSTANCE verdict on the three preserved G1 selected knees. It is not
a negative verdict on a future ξ-predictable CONNECTION formulation or a
receiver-null GAUGE formulation; neither appears as a measured positive-byte
payload in the G1 receipt.

## Durable changes

1. `tac.optimization.ddm_min_description_contract` is the single source of
   truth for:
   - `StreamType = {SKELETON, CONNECTION, FIBER, GAUGE, RESIDUAL}`;
   - the five legal `layer_home` values;
   - `TypedStreamTag(type, layer_home,
     evaluate_py_recursion_level_cited, counted_bytes,
     free_receiver_code)`;
   - exact JSON round-trip and GAUGE = 0-byte invariants.
2. `build_minimum_description_headline` withholds headline authority when type
   custody is missing. The first landing returns the exact
   `TYPED_STREAM_TAG_CUSTODY_MISSING_WARN_ONLY` diagnostic instead of raising.
   `strict_typed_stream_tags=True` is the later flip. A waiver is recorded but
   is explicitly non-authorizing.
3. The #636 runtime exporter tags every counted section. The standalone
   receiver validates the sealed tag vocabulary, recursion citation, section
   byte equality, exact boolean, and GAUGE = 0 rule at consumption time.
4. RD1 dual-cube component and bucket rows now expose `stream_type: null` with
   an exact byte-home-custody blocker. No type is inferred from scorer
   visibility alone.
5. DR2b exact-R-null rungs emit GAUGE/0B and are not priced. Scorer-visible
   measured rungs emit FIBER and are the only rungs admitted to a rate price.
6. The G2 correspondence is a callable canonical equation:
   `ddm_g2_five_type_correspondence_v1`.
   Its EmpiricalAnchor cites the preserved G2 n600 aggregate ledger. The law
   maps:
   - scorer-invisible → GAUGE;
   - ξ-predictable → CONNECTION;
   - chart-expressible → SKELETON + FIBER;
   - irreducible → RESIDUAL.
   The receipt remains blocked on receiver delta-dseg, so the law grants no
   carrier admission.
7. The G1 typed arithmetic and per-production table are preserved in
   `.omx/research/ddm_ts1_g1_typed_stream_reprice_20260724.json`.

## G1 answer

| Stratum | Selected n600 candidate | Before counted bytes | After counted bytes | Mispriced |
|---|---|---:|---:|---:|
| Movable | `movable_track_shape_abs_eps1p0` | 29,810 | 29,810 | 0 |
| Lane | `lane_slots_delta_dash_tolx2p0` | 27,692 | 27,692 | 0 |
| Boundary | `boundary_arc_eps2p0` | 219,288 | 219,288 | 0 |
| **Total** | three independently selected knees | **276,790** | **276,790** | **0** |

Real coder candidates were Brotli quality 11, raw LZMA1 preset 1 / 1 MiB,
and zlib level 9. Generic receiver operator code is 0 counted bytes.
EVENT/ARC_EVENT and framing are SKELETON. Centroids, shapes, lane
center/width/dash/range, and arc vertices are video-derived FIBER parameters.
No selected G1 production was proven exact-R-null, and none was relabeled
GAUGE.

## Epistemic ledger

- **MEASURED**: preserved G1 real-coder production byte rows and selected-knee
  totals; G2 operator byte waterfall and range/kernel energy split.
- **DERIVED**: five-type crosswalk; arithmetic before/after totals; exact
  section tag reconciliation; GAUGE = 0-byte and visible-band FIBER admission.
- **FALSIFIED**: “G1 counted generic vocabulary/receiver operator code.”
- **UNMEASURED**: receiver-visible efficacy of the five-type partition,
  positive-byte ξ CONNECTION payload savings, G1 GAUGE savings, contest-CPU or
  contest-CUDA score movement.

## Validation status at memo emission

- Focused suite: `56 passed`.
- New typed-contract coverage includes positive, malformed, missing-tag
  warning, strict-flip, non-authorizing waiver, byte mismatch, RD1 NULL
  custody, DR2b GAUGE/FIBER, G1 arithmetic, G2 callable, and registry-anchor
  cases.
- Final three clean review passes are a landing requirement and are recorded
  in the delegated checkpoint / final handoff after completion.

## Directive consumption

| Directive/source | Disposition | Application |
|---|---|---|
| Delegated wrapped authority | ADOPTED | Exact scope, local-$0 boundary, outputs, tests, checkpoints, and MAIN review gate. |
| Recursive-scorer representation typology memory | ADOPTED | Five types and G2 correspondence; no coefficient type inferred without byte home. |
| Metric-first charter memory | ADOPTED | Recursion citations and visible-band admission point to Fisher/margin and Pose geometry, not Euclidean parameter distance. |
| Operating manual craft handoff | ADOPTED | Durable code, tests, ledger artifact, pointer-delta honesty, no chat-only conclusion. |
| Operator EV reverse-waterfill broadcast 2026-07-19T19:42:07Z | ADOPTED-AS-GUARD | DR2b prices only scorer-visible FIBER; no new repricing or admission beyond measured custody. |
| Operator Fisher/inner-Jacobian/curvelet broadcast 2026-07-19T19:48:01Z | ADOPTED-AS-GUARD | No Fourier/Euclidean efficacy inference; missing receiver delta and inner Jacobian remain explicit G2 blockers. |
| Paid dispatch / exact eval / training | NOT AUTHORIZED | None performed. |

## STORES CONSULTED

- Delegated wrapped authority file and its verified SHA-256.
- `docs/operating_manual_craft_handoff.md`.
- `.omx/research/ddm_scorer_native_doctrine_and_synthesis_20260723.md`.
- Recursive-scorer typology and metric-first memory notes named by authority.
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`, and
  canonical equation registry.
- G1 compact receipt plus the SHA-custodied SSD primary receipt.
- G2 aggregate ledger and its prior Codex findings.
- #636 exporter/receiver, RD1, DR2b, minimum-description contract, relevant
  tests, per-arm inbox, and broadcast inbox.

## Remaining exact blockers

1. MAIN must review whether the existing runtime archive schema identifier
   should be version-bumped now that section tags are mandatory; missing tags
   are intentionally refused.
2. The warn-only counter must reach zero before flipping headline construction
   to strict-by-default.
3. RD1 may not populate its type column until candidate-delta × dimension
   byte-home custody exists.
4. A ξ CONNECTION measurement must separate generic operator code (free) from
   video-derived per-clip parameters (counted).
5. G2 carriers remain blocked until receiver-closed coefficient perturbations
   produce realized delta-S per counted byte.
