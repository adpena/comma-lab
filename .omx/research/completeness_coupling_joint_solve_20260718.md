# Shared-resize Seg/Pose coupling and joint-solve term

Date: 2026-07-18
Task / lane: `#538` / `completeness_coupling_20260718`
Authority: advisory BUILD + `$0` local measurement; **NO launch**
Verdict scope: `INSTANCE x BASE-INR n600-TRAINED EMA x DETERMINISTIC n8/600 x macOS-CPU FROZEN-SCORER FORMULATION`
Pointer: `0.19108` **UNMOVED**
Sacred c2: **READ-ONLY; NOT MUTATED**

## Verdict

**DERIVED — the constrained inverse formulation is joint, not two independent
Seg/Pose solves.** Both losses consume the same camera render through the same
bilinear resize `A`, while `Q_uint8`, payload capacity, and archive rate are
shared constraints. The correct local score costate is therefore

```text
q_joint = 100*g_seg + [5/sqrt(10*d_pose)]*g_pose,
g_seg  = d(ell_seg_surrogate)/d(render),
g_pose = d(d_pose)/d(render),
A_seg == A_pose == A.
```

There is no tunable coupling coefficient in that expression. At this measured
baseline, `d_pose=127.82099628448486`, so the score-derived pose marginal is
`lambda_pose=0.13985207202716476` and `lambda_seg=100`.

**DERIVED (smooth local surrogate) — this checkpoint/subset has a nonzero B1
shared-frame pullback-Gram cross term.** The local winner-rival/Pose pullback
Gram has shared-frame off-diagonal
`G_sp=1.3205035467867e-09` and normalized overlap
`+0.004992744642171348`. This is not an exact `d_seg` measurement.

**MEASURED (finite uint8 response) — this checkpoint/subset has
direction-dependent interference after realization.** The frozen B32 one-LSB
secants show that the Pose direction harms Seg at both measured support sizes,
while the joint direction improves both measured distortions at both sizes.
The Seg direction helps both at the smaller support but ceases to improve its
own Seg target at the larger support. Thus “solve Seg, then solve Pose” is not a
safe composition rule for this instance even though the derived smooth cosine
is small.

**INFERRED — build the live solve as one trust-region/waterfill consumer, but do
not adopt from this receipt.** Full-n600 native batching, a carrier-aware
receiver candidate, counted bytes, resume/byte-close evidence, and contest CPU
and CUDA axes remain owed. This lane is argv-inert and cannot authorize launch,
promotion, or pointer movement.

## Measurement custody

Durable receipt:
`.omx/research/completeness_coupling_joint_solve_measurement_20260718.json`

- Receipt SHA-256: `05cf34068053a4e2f744dfb35cde729579353686298da3ee1ceaf925f5a71f5f`
  (`14,366` bytes).
- Measurement source commit: `6d2d905cecd5c4d8dd08752769a813869b33d2c7`.
- Sample: deterministic cyclic stride, seed `538`, pair IDs
  `[50,125,200,275,350,425,500,575]` — honestly labeled `n8/600`, not n600
  measurement coverage.
- Candidate: real n600-trained banked V9-c2 EMA checkpoint, SHA-256
  `b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef`.
  Its 59-key manifest was scanned before decode; `carrier_absent=true` and
  `base_inr_only=true`. A carrier-bearing checkpoint refuses rather than being
  silently misdecoded.
- GT cache SHA-256:
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
  Targets used in the measurement were rederived from `gt_f0`/`gt_f1` under
  the exact B32 duplicate-last scorer call, not trusted from cache metadata.
  Cached Seg labels matched all `1,572,864` sampled pixels. Cached Pose values
  differed in 44/48 float elements under this batch geometry, with maximum
  absolute difference `1.33514404296875e-05` and MSE
  `7.051369037280875e-12`; the rederived B32 values carried authority.
- Frozen model hashes: SegNet
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`;
  PoseNet
  `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576`.
- Shared forward checks: camera `(874,1164)` to scorer `(384,512)`, bilinear
  `align_corners=False`; Seg preprocess tensor equality `true`; differentiable
  YUV6 clone maximum absolute discrepancy `0.0`.
- Axis: `[macOS-CPU advisory]`. Smooth VJP geometry is
  `B1_LOCAL_DERIVED`; finite response geometry is
  `B32_DUPLICATE_LAST_SUBSET_ADVISORY`, explicitly
  `native_or_full_n600_comparable=false`.
- Runtime: `326.94222033396363` seconds, CPU torch, one thread,
  deterministic algorithms enabled. No paid dispatch, trainer activation,
  archive score, candidate archive, or source-input mutation occurred.

The path-bound equation loader reopened the JSON and recomputed the cache,
checkpoint, model, and frozen-source hashes before accepting the receipt as a
subset anchor. An arbitrary in-memory mapping cannot create an empirical
anchor.

## Smooth shared-frame coupling

Exact `d_seg` is discontinuous. The smooth Seg row below is therefore a
**DERIVED surrogate**: a winner-versus-rival zero-margin hinge VJP. The scalar
“off-diagonal” is not an off-diagonal entry of a two-row Jacobian; it is the
cross entry of its render-space pullback Gram, `G=J J^T`.

Primary shared-frame-1 surface:

```text
G_shared = [[1.8304522611187202e-11, 1.3205035467867000e-09],
            [1.3205035467867000e-09, 3.8215718450373970e-03]]
cos_shared(g_seg,g_pose) = +0.004992744642171348

Lambda G_shared Lambda =
           [[1.8304522611187202e-07, 1.8467515713734010e-08],
            [1.8467515713734010e-08, 7.4744602923683870e-05]]
||q_joint||^2 = 7.496458318122321e-05
```

The full-pair context, which adds Pose-only frame-0 energy, has
`G_pp=0.01071188662899658`, overlap `+0.002982135677261322`, and priced joint
norm squared `0.00020972950804308413`. Frame 0 holds
`0.6432439266262499` of the local Pose-gradient squared norm. It is reported
separately and is not misnamed as the shared-frame coupling magnitude.

The smooth input-gradient cosine and the finite response-column cosine are
**noncommensurate diagnostics**: they live in different spaces, use different
batch geometries, and the finite columns mix raw `d_seg`/`d_pose` units. No
smooth-versus-finite calibration residual is claimed. The empirical anchor's
zero residual applies only to the exact shared-forward/YUV6 parity check.

## Measured one-LSB help/harm matrix

Baseline on the selected B32-padded subset was
`d_seg=0.003166834532748908`, `d_pose=127.82099628448486`; the derived
no-rate objective value was `36.06874581119837`. “Score delta” below excludes
archive rate and is therefore **DERIVED FROM MEASURED COMPONENTS**, not a
contest score.

| support over pair render | realized frame-1 RGB changes | Seg direction: `delta d_seg` / cross `delta d_pose` | Pose direction: `delta d_pose` / cross `delta d_seg` | joint direction: `delta d_seg`, `delta d_pose` | joint no-rate delta |
|---:|---:|---|---|---|---:|
| `0.0001` | `4,888` | `-8.45591421239078e-05` / `-0.016260147094726562` (`HELP` / `HELP`) | `-1.2642812728881836` / `+1.2715172488242388e-06` (`HELP` / `HARM`) | `-1.2716336641460657e-06`, `-1.2631988525390625` | `-0.1772267761307944` |
| `0.001` | `48,840` | `+0.0004730224027298391` / `-0.024808883666992188` (`UNINFORMATIVE TARGET`; cross not credited) | `-10.01187801361084` / `+1.1444120900705457e-05` (`HELP` / `HARM`) | `-1.33514404296875e-05`, `-10.001636505126953` | `-1.4285726984284253` |

At `0.0001`, the Seg-only no-rate delta was `-0.010730001799196032`, the
Pose-only delta was `-0.17712459368841849`, and the joint delta was
`-0.1772267761307944`. At `0.001`, Seg-only worsened the no-rate objective by
`+0.043832498118014485`, Pose-only improved it by
`-1.4275850262473426`, and joint improved it by
`-1.4285726984284253`. These are instance/subset effects; no claim is made
that this sparse sign family is globally optimal.

## Typed joint-solve lever

The new DSL surface is
`shared_resize_joint_coupling_policy.v2`, with measurement schema
`shared_resize_joint_coupling_measurement.v2`. It seals:

1. one shared `A` and its exact input/output geometry;
2. the score-derived coefficients `(100, 5/sqrt(10*d_pose))`;
3. both `G_shared(frame1)` and `G_full(pair)` with the former primary;
4. B1/B32 authority labels and a deterministic n-of-600 sample contract;
5. carrier absence, GT rederivation, response-matrix recomputation, realized
   LSB conservation, and path/hash-bound empirical anchors;
6. `live_trainer_argv=()` and fail-closed escalation flags.

The canonical candidate resolver ID is
`shared_resize_joint_coupling_through_a_v1`. The task brief's capital-`A`
spelling is retained only as `NON_RESOLVING_DISPLAY_ALIAS`; no unsupported
alias lookup is implied.

A future live consumer should solve the shared trust-region step once, then
measure it after `Q_uint8 -> A -> {SegNet,PoseNet}` and admit only a
receiver-closed score-unit-per-byte improvement. It must preserve the separate
frame-0 Pose leg and must not infer full inverse separability from a small local
Gram overlap.

## Completeness and triality

- DSL: the argv-inert policy and strict eleven-leaf completeness-manifest
  compiler in `src/tac/witness_dsl/shared_resize_joint_coupling_policy.py`.
- DAG: `.omx/research/completeness_coupling_joint_solve_DAG_FEED_20260718.md`.
- Equation: `shared_resize_joint_coupling_through_a_v1`, with structural
  shared-forward parity plus complementary, explicitly noncommensurate
  coupling diagnostics.
- Factor matrix:
  `.omx/research/inverse_solve_completeness_matrix_20260718.md` records all ten
  factors as eleven leaves because factor 3 splits into camera preimage (3a)
  and shared-objective coupling (3b). Overall verdict remains
  `NOT_COMPLETE_BY_CONSTRUCTION`.
- Fourier: **CARGO-CULTED** for this term. The natural form is a shared
  pullback Gram plus score costate/KKT waterfill, not a Fourier substitute.

## Round-1 adversarial self-review

1. **Carrier omission attack:** the base decoder could silently omit a Pose
   carrier. **Closed:** carrier/config keys now refuse; this checkpoint is
   explicitly carrier-free and base-INR-only.
2. **GT-cache authority attack:** a hashed but batch-mismatched target cache
   could green the result. **Closed:** sampled GT targets are rederived B32 and
   cached mismatches are recorded; rederived targets are used.
3. **Wrong Gram attack:** Pose frame-0 energy could be mislabeled shared-A
   coupling. **Closed:** shared-frame-1 is primary and full-pair context is
   separate.
4. **Batch equivalence attack:** B1 VJPs and duplicate-last B32 secants could be
   presented as native n600 evidence. **Closed:** the two exact labels and
   `native_or_full_n600_comparable=false` are validator-enforced.
5. **Fabricated-anchor attack:** a structurally plausible mapping could become
   `VERIFIED`. **Closed:** empirical anchors require a receipt path, receipt
   digest, deterministic IDs, and rehashed named artifacts.
6. **Canonical-ID attack:** the display spelling could silently fail registry
   resolution. **Closed:** lowercase is the sole resolver ID; uppercase is
   explicitly non-resolving.
7. **Self-reported-help attack:** a zero-response receipt could call itself
   `MEASURED_HELP`. **Closed:** the validator independently recomputes matrices,
   deltas, ratios, direction classifications, and requires realized changes.
8. **Incommensurate-residual attack:** B1 input-gradient cosine could be
   subtracted from B32 output-column cosine. **Closed:** both remain separate
   diagnostics; only exact shared-forward parity has a zero residual.

Independent round-3 review was `CLEAN`; the focused suite passed `34` tests
before measurement. MAIN must still review the branch diff and the real receipt
before landing.

## Remaining literal blockers

- No native/full-n600 finite-response measurement; only deterministic n8/600.
- No receiver-bearing/carrier-aware candidate measurement; this candidate is
  base-INR-only.
- No counted archive bytes or rate curve, hence no factor-10 waterfill/adoption
  solution.
- No resume schema, byte-close archive, contest-CPU replay, or contest-CUDA
  replay.
- No live trainer consumer; the DSL emits no argv.
- No family/paradigm conclusion from this single checkpoint and sparse
  perturbation family.

## Stores consulted

- `docs/operating_manual_craft_handoff.md`, `CLAUDE.md`, `AGENTS.md`, and
  `PROGRAM.md`.
- V10 SPEC §14.7–§14.11 and the campaign constrained-MDL completeness check.
- `.omx/research/frozen_scorer_exact_factorization_20260715.md`.
- `.omx/research/sol_ultra_v10_true_final_form_review_20260717.md`.
- Current lane registry, subagent ownership/checkpoints, pointer surfaces,
  latest Codex/Claude memos, and recent directive files.
- The exact checkpoint, cache, scorer weights, and frozen scorer sources named
  and hashed in the receipt.

## MAIN landing review

Reopen the receipt and rehash its named inputs; verify the n8/B1/B32 labels,
shared-versus-full Gram split, finite help/harm table, parity-only equation
residual, argv-inert policy, exact ten-factor/eleven-leaf matrix, and unchanged
pointer. Do not reinterpret this branch as launch, score, promotion, or
full-n600 authority.
