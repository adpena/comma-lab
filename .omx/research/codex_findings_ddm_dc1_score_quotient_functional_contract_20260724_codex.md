---
title: "Codex findings: DDM DC1 score-quotient functional contract"
date_utc: "2026-07-24T13:18:00Z"
lane_id: "lane_ddm_dc1_score_quotient_functional_contract_20260724"
research_only: true
execution_allowed: false
score_claim: false
promotion_eligible: false
authority_axis: "[macOS-CPU frozen-scorer advisory]"
verdict_scope: "family-(d) typed packet, receiver, rate-in-objective, capacity, and future-fit interface"
verdict: "BINDING_INCOMPLETE_FIT_OWED"
missing_stream: "FIT_RESULT_RECEIVER_CLOSED_V14_OR_BETTER"
pointer_before: "0.1910828242 [contest-CPU]"
pointer_after: "0.1910828242 [contest-CPU]"
pointer_delta: 0
main_review_required: true
---

# Outcome

DC1 binds family (d) as an executable description/receiver contract without
pretending that the family has already fit the video:

`BINDING_INCOMPLETE_FIT_OWED`.

The landing is additive:

- `tac.optimization.ddm_score_quotient_functional_contract` defines the
  deterministic bytes, real coders, typed stream homes, strict parser,
  scorer-plane receiver, exact contest functional, capacity derivation,
  v14 falsifier, and future-fit request;
- `tac.canonical_equations.ddm_score_quotient_functional_20260724` defines the
  callable EQUATIONS leg and locked-registry population helper;
- 9 focused tests prove the structural contract, including a bounded
  hard-tail-first `n=24` constant-plane fixture through the real `R` operator;
- no training, descent, paid dispatch, exact evaluator, candidate archive,
  score row, launch argv, or frontier mutation occurred.

The contract does **not** claim that its toy separable rank-one basis is the
optimal family or that a constant-plane fixture is a video result. Its purpose
is to make the objective and custody boundary executable so a later fit can
arbitrate the representation rather than hiding rate, receiver, or scorer
gaps in prose.

# Bound functional

For a typed candidate \(q\),

\[
S(q)=100d_{\rm seg}(\mathcal R D(q))
  +\sqrt{10d_{\rm pose}(\mathcal R D(q))}
  +25B_{\rm counted}(q)/37{,}545{,}489.
\]

`score_quotient_functional_objective` delegates the scalar exactly to
`tac.contest_score.compute_contest_score`. It never hand-rolls the scorer
formula. Its byte input is `receipt.total_counted_bytes`, measured from the
actual deterministic compiler output:

\[
B_{\rm counted}=
B_{\rm named\ base}+B_{\rm outer\ framing}
+B_\theta+B_z+B_{25}+B_\epsilon .
\]

The active delta packet references a SHA-256-bound named base; deployment must
count both the base bytes and packet bytes. If all four streams are inactive,
the compiler returns the named base bytes themselves, byte-identically, with
no wrapper. This preserves the RG1 additive-grammar invariant.

For internal DC1-owned streams, the compiler computes all three deterministic
choices—pass-through, zlib level 9, and raw LZMA1 with a pinned 1 MiB
dictionary—and selects the shortest with a stable coder-ID tie break. The
chosen bytes, not entropy estimates, enter the objective. DM1 placement values
arrive already real-coded and are deliberately pass-through at DC1’s outer
layer so DC1 cannot duplicate or silently alter their prices.

The reverse-waterfill stop threshold is the exact rate derivative:

`25 / 37,545,489 = 6.658584...e-7 score units per byte`.

The future fitter must stop spending on a coordinate when measured marginal
score benefit per added emitted byte drops below that threshold. This module
records the threshold; it does not fabricate the missing empirical marginal.

# Typed payload and ownership

| Section | Sealed tag | Counted content | Decoder/price owner | Empty behavior |
|---|---|---|---|---|
| parameters | `SKELETON × L1_program` | two-frame/channel DC plus separable row/column bases | DC1 real coder | omitted |
| temporal latents | `CONNECTION × L2_chart` | canonical pair index, six plane coefficients, six `xi` pose targets | DC1 real coder | omitted |
| demand placements | `FIBER × L3_raster` | exactly 25 canonical `(pair,bucket,slot)` records | DM1 external decoded-value schema and real coder price | omitted |
| exceptions | `RESIDUAL × L4_scorer_feature` | canonical `(pair,frame,y,x,channel,value)` records | DC1 real coder | omitted |

The placement tag is a transport home, not a re-adjudication of DM1’s
per-row homing. Each placement retains an external `coder_id`, decoded-value
SHA-256, and exact coded record. Nonempty placement sets must contain slots
`0..24` exactly, must be sorted by `(pair,bucket,slot)`, and must have unique
`(pair,bucket)` addresses. A receiver refuses placement-bearing packets unless
the external DM1 decoder/applier is supplied.

Exceptions accept only the literal scope
`AT_RISK_FLIP_ANNULUS`. This enforces the Fisher/margin directive and prevents
the sparse stream from becoming an unbounded camera-RGB repaint path.

Every section carries a raw length, emitted length, raw CRC-32, coder ID,
SHA-256 receipt, and sealed `TypedStreamTag`. The packet prefix and base
reference framing are charged to the first active typed section, so the sum of
section-tag counted bytes equals the exact emitted packet length. The outer
body carries its own length and CRC-32. The parser requires sorted unique
section kinds, validates the named base length/SHA, decodes the actual coder,
reconstructs typed objects, recompiles them, and requires exact packet-byte
equality.

# Receiver contract and proof scope

The deterministic receiver expands each pair to:

- two uint8 RGB planes shaped `(384,512,3)`, exactly the scorer-plane object;
- six `xi_q12/4096` pose-target statistics;
- externally applied DM1 placement content, if present; and
- at-risk scorer-plane exceptions, if present.

For every requested frame, the caller must provide an explicit preimage
realizer that returns uint8 camera bytes shaped `(874,1164,3)`. The receiver
runs those bytes through
`tac.through_r.resolution_chain.contest_faithful_R_numpy`, rounds/clamps back to
uint8, and requires exact equality with the expected scorer plane. A cache
deduplicates only byte-identical camera inputs; it does not infer parity across
different inputs.

The proof API refuses:

- fewer than 24 pairs;
- repeated or out-of-range pair IDs;
- a subset whose first 24 entries do not equal the supplied hard-tail order;
- missing temporal latents;
- missing DM1 decoder/applier when placement records exist;
- DM1 decoded placement bytes whose SHA-256 differs from the external record;
- wrong camera shape/dtype;
- any through-R pixel difference.

The focused proof used 24 canonical pair IDs and constant scorer planes with a
constant camera preimage. That is a real-R/uint8/parse-back **contract proof**,
not a real-video quality measurement. The previous yhat-native n24 work
established that arbitrary two-plane preimages are a separate runtime/custody
problem; DC1 does not relabel the constant fixture as closure of that debt.

The six stored pose statistics are target coordinates. A promotion-grade
candidate still owes the actual two-frame YUV6 frozen-PoseNet forward on
receiver bytes. DC1 does not equate a stored target with a measured
`d_pose`.

# Capacity derivation

No capacity dimension is handpicked:

| Component | Exact dimension | Epistemic status | Derivation |
|---|---:|---|---|
| Seg argmax head | 4 | `DERIVED` plus existing measured rank anchor | centered differences of `K=5` logits have ceiling `K-1`; frozen head anchor is rank 4 |
| Lane homography orbit | `NULL` | `NULL_DERIVATION_OWED` | `~8` is retained only as an approximate research-sidecar hint until a realized-through-R Jacobian rank certificate is supplied |
| Pose `xi` | 6 | `DERIVED` from source | frozen PoseNet verdict consumes the first six output coordinates |
| Demand records | 25 | `DERIVED` from IS1/DM1 boundary | exact delegated set size; values and coder prices remain DM1-owned |
| Exact total | `NULL` | `NULL_DERIVATION_OWED` | the lane-orbit term is not silently replaced by 8 |

If a later caller supplies a typed lane-orbit rank certificate with exact
rank, durable artifact, artifact SHA-256, measurement axis, and
`realized_through_r=true`, the helper returns
`4 + certified_rank + 6 + 25`; it refuses a bare integer. Without that
certificate, the total remains `NULL`. This is a capacity interface, not an
admission claim.

# v14 binding falsifier

The authority requires family (d) to express the v14 receiver baseline at no
more than its byte cost:

- v14 `d_seg = 0.027470296224`;
- v14 archive bytes `= 133,247`.

No DC1 fit was authorized or run. Therefore the current mandatory result is:

`INCOMPLETE: FIT_RESULT_RECEIVER_CLOSED_V14_OR_BETTER`.

`v14_baseline_falsifier` returns an expressibility pass only when all three are
true:

1. finite candidate `d_seg <= 0.027470296224`;
2. integer candidate bytes `<= 133247`;
3. `receiver_closed=true`.

Otherwise it requires a named missing stream. The code has no permissive
default. A future v14 pass remains advisory until actual frozen scorers,
archive custody, and the authority axis are supplied.

# `DDMEventContinuationV1` fit readiness

`build_ddm_event_continuation_v1_fit_request` emits a typed request with:

- exact packet SHA and total counted bytes;
- objective and receiver callable paths;
- `n>=24`, hard-tail-first, exact parse-back gates;
- real-coder-in-loss requirement;
- exact reverse-waterfill rate threshold;
- `execution_allowed=false`;
- `score_claim=false`;
- unchanged frontier pointer.

Its status is `INTERFACE_ONLY_NOT_EXECUTABLE`. It names four support gaps:

1. no executable `DDMEventContinuationV1` schedule/engine;
2. no supplied DM1 decoded-value schema and 25 real coder-price rows;
3. no receiver-closed frozen PoseNet/SegNet n600 fit;
4. no exact lane-orbit rank certificate.

This deliberately composes with SCHED1’s finding that
`DDMEventContinuationV1` is absent. The adapter is a future fit request, not a
new launcher, invented CLI flag, or surrogate engine.

# Triality and system integration

- **DSL / typed contract:** the packet dataclasses, sealed tags, parser,
  receiver, falsifier, and fit-request dataclass are the callable contract.
  There is no argv surface and no executor.
- **DAG:** `FEED-DDM-DC1-SCORE-QUOTIENT-20260724` routes DM1 prices plus an
  eventual fitter into compiler → receiver → frozen scorers → exact
  functional → v14 gate → MAIN review.
- **EQUATIONS:** `ddm_score_quotient_functional_v1` has a build helper and
  fcntl-locked append-only population helper. It is design-only, has zero
  empirical anchors, declares the current `INCOMPLETE` state, and cannot
  validate a score claim.

Six-hook wire-in:

| Required hook | DC1 disposition |
|---|---|
| sensitivity map | future candidate ordering consumes registered Fisher/margin, corrected inner-Jacobian, resize, necessity, and pose-`xi` laws; DC1 adds no duplicate metric |
| Pareto constraint | exact scalar `S` is the binding Seg/Pose/rate Pareto action |
| bit allocator | per-section real coded byte receipts plus exact `25/N` stop derivative are callable |
| cathedral/autopilot dispatch | fail closed: request is non-executable and requires MAIN review; no dispatch hook is armed |
| continual-learning posterior | no score empirical anchor exists; structural law is registered design-only and findings/DAG preserve the blocker |
| probe disambiguator | internal coder ambiguity is resolved by compiling all fixed real coders and taking exact shortest bytes; representation-family ambiguity is intentionally deferred to the exact-S fit |

# Directive-consumption table

| Directive | Consumed action | Durable surface |
|---|---|---|
| delegated authority file, SHA `21d54c...c485a` | bounded DC1 to build-only, $0, no fit/dispatch/eval/pointer mutation | this memo; packet/receiver tests |
| Directive 7 family (d) | represented scorer-plane function, six pose targets, real coder in exact S; excluded human fidelity | contract module and equation |
| DM1 delegated authority boundary | accepted opaque externally priced records; did not materialize values, run #669c homing, or derive prices | placement dataclass and mandatory external applier |
| IS1 findings | preserved four-family comparison and `NULL` minimum until receiver/coder/fit closure | v14 falsifier and support gaps |
| SCHED1 findings | exposed `DDMEventContinuationV1` interface only; no invented executable engine | fit request |
| operator reverse-waterfill broadcast, 2026-07-19 19:42Z | pinned stop threshold `25/37545489` and exact emitted bytes | objective and fit request |
| operator Fisher/basis/factorization broadcast, 2026-07-19 19:48Z | exceptions restricted to Fisher/argmax annulus; pose represented by `xi`; no Fourier residual path | exception schema and findings |
| yhat-native generator specification | reused scorer-plane/preimage separation and n24 proof pattern; did not import HNeRV vehicle/lineage | receiver contract |
| RG1 receiver grammar | preserved inactive byte identity, CRC, canonical addresses, and re-encode equality | compiler/parser |

Per-arm inbox was empty at the implementation checkpoint. The broadcast inbox
contained many stale unrelated fleet messages; only the two operator
directives above materially apply to DC1.

# Verification

Environment threads were pinned:

`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4`.

Focused command:

`/Users/adpena/Projects/pact/.venv/bin/pytest -q src/tac/optimization/tests/test_ddm_score_quotient_functional_contract.py src/tac/canonical_equations/tests/test_ddm_score_quotient_functional_20260724.py`

Latest focused result after Round-0 fixes: `9 passed in 1.04s`.

The tests cover inactive byte identity, all four typed sections, external
25-row custody, decoded-value SHA validation, real-coder/CRC failure, exact
canonical parse-back, typed-byte conservation, n24 hard-tail-first real-R/uint8
equality, missing external applier refusal, Torch thread pinning,
contest-score parity, capacity NULL discipline, v14 falsifier, non-executable
fit request, and locked registry population into a temporary ledger.

Round 2 additionally ran 50 seeded random packet/parser/coder/corruption
property trials; all were clean. The related RG1 and upstream contest-score
parity suite result was `55 passed in 46.23s`; the final blind re-derivation
rerun was `55 passed in 40.19s`.

# Independent self-review disposition

Round 0 found four real custody defects:

1. packet/base-reference framing bytes were counted in the objective but not
   assigned to a typed section;
2. the external DM1 decoded-value SHA was stored but not checked;
3. `cpu_threads=4` was reported without actually pinning Torch;
4. the capacity helper accepted a bare lane-orbit integer, permitting a
   handpicked `8` to masquerade as a certificate.

All four were fixed before the clean-pass counter began. The review tracker
records three subsequent reviewed marks for every entity in all four new
Python files:

1. `round1_schema_rate_custody_clean`;
2. `round2_seeded_parseback_edge_cases_clean`;
3. `round3_blind_derivation_upstream_parity_clean`.

No `.omx/state/review_policy.json` exists in this isolated worktree, so the
tracker could not produce a policy-compliance verdict; it did record the
three clean marks. This does not replace the required independent MAIN review.

The assumption-challenge axis asked whether the separable rank-one primitive
had silently become “the optimum representation.” It has not: it is only the
smallest executable receiver fixture. A richer function family may be the
actual optimum and must be admitted by the future exact-S fit rather than
suppressed by this grammar. The second shared assumption is that an active
functional delta should reference a named base. A base-free standalone
function may beat it; the contract permits a different, smaller named base but
does not claim the v14 base is optimal. These are open substrate choices, not
reasons to weaken exact bytes or receiver gates.

The append-only equation ledger contains two historical registrations for the
same equation ID. Round 0 corrected the displayed LaTeX from a four-term
shorthand that omitted base/framing to the exact
`B_counted(q)` receipt. Latest-row-wins semantics select the corrected row;
the earlier row is retained rather than erased.

Final reviewed source SHA-256 values:

| File | SHA-256 |
|---|---|
| functional contract | `3da681103def103d2a49c3e1fce8da3404e83b59978e44037ae262c79ce41786` |
| functional tests | `db1ad878e53f6552a9032d568fd256390a22c0bac803d263cbdc058f9e922aa6` |
| canonical equation | `1c71f4676a30248bbbcf46c9b40cddf291e19682ed12befe722f752ec3dad0c7` |
| equation tests | `e00114f4ef3470764d5b591fc732c454983cf93de8c0eb42dd72a6ea2329461f` |

# Source custody

| Input | SHA-256 |
|---|---|
| delegated authority | `21d54c6664c67c94a8742bfedeaef18d0c14a0bdcc1300eef7deb20ee15c485a` |
| `CLAUDE.md` | `6618865f8210365342185768bf38079f6b68bc7eb22049e7958bbab92d8083ba` |
| `AGENTS.md` | `79f951d596b18e9f1e4d500e601d22ebca14acaff0295c224fd072f54abf8493` |
| `PROGRAM.md` | `a6d5f79f3241ca1ae17b2587afd9940e1a4ea598804fd9efa152f2330e15db82` |
| operating manual | `40d157a039d4dd242bfb189d53e6b82abcc5d037adceb0a52c9bb2956903f212` |
| IS1 findings | `ccb302c2770ba1baa67f5c5000d72a5e34e0a875796943f1d097268f0b05adff` |
| SCHED1 findings | `f383e3a30d8f51ded655aa1fd1b4f8e397f825d6ed861d3293f2490bbb384369` |
| Directive 7 | `7a9cc45f0d4f60487fb12afcc8c59c46436d908907cf28a8422999bec614ca0e` |
| yhat-native specification | `1991cfcd8b82594bc9c778687b392fde86bb49508f01f67281f29663146f791f` |
| sealed minimum-description tags | `4c8c632917abf346d8f4e2ab2ae056befb4f215d03f73c62a3df07b1adfc0c95` |
| RG1 grammar | `6532b12751c415ed56cbd87f2831b54df69bd6d3afc744864f20b5a248e29069` |
| canonical contest score helper | `372c407e6e2f62532f12a0a6aaab99fa675f2ada2a11348e0187bd5f278e196f` |
| real-R NumPy chain | `6d476b13d2130a7c9d366ceef5cd679bc356f34ccda508729dfad72b0f2934d0` |
| Seg rank-4 equation | `bce50545bba699f1968742c3a4aff70a6eadbad16768c9d92c047952524c4af9` |
| upstream `modules.py` read-only snapshot | `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa` |

# STORES CONSULTED

- delegated authority and DM1 authority files
- full `CLAUDE.md`, full `AGENTS.md`, `PROGRAM.md`, and craft handoff manual
- current frontier, lane registry/maturity, subagent ownership/checkpoints,
  canonical equation registry, per-arm inbox, and broadcast inbox
- IS1, SCHED1, Directive 7, yhat-native, minimum-description, RG1, scorer,
  real-R, frozen-head rank, and upstream frozen-scorer source surfaces
- top ten operator Claude-memory entries and current Codex memory registry

MAIN must independently review the packet framing, external DM1 boundary,
real-R proof scope, capacity NULL, v14 falsifier, equation row, and serializer
commit before merge. This isolated branch is not authority for promotion or
execution.
