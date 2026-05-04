# PR85 Adaptive Masking Joint Frame Model Intake - Codex

Timestamp: 2026-05-03T23:42Z

Evidence grade: external public claim plus local static archive profile. No PR85
score claim is promoted here until exact CUDA T4 replay lands through the
canonical replay job artifacts.

## Objective

Preserve the current PR85 public claim, local archive anatomy, exact replay
custody, comparison to the PR84 exact frontier, the current PR84+PR82 negative,
and the contest-faithful next-action branch. This ledger does not dispatch or
claim any new remote work.

## PR85 Public Claim

- Public PR: `https://github.com/commaai/comma_video_compression_challenge/pull/85`.
- Submission name: `adaptive_masking_joint_frame_model`.
- Public-reported exact score: `0.25806622496743437`.
- Public report components over `600` samples:
  - PoseNet: `0.00018940`.
  - SegNet: `0.00057185`.
  - Archive bytes: `236328`.
  - Display-rounded final score: `0.26`.
- Public claim says GPU is required for evaluation/inflation.
- Public claim says no compression script is included or requested for merge.
- Public branch at intake: `ottokunkel:adaptive_masking_joint_frame_model`.
- Public PR state at intake: open and ready for review.

Recomputing the public components with the contest formula gives:

```text
100 * 0.00057185
+ sqrt(10 * 0.00018940)
+ 25 * 236328 / 37,545,489
= 0.25806622496743437
```

This is a public/external claim until our exact T4 replay produces
`contest_auth_eval.adjudicated.json` for the exact archive bytes and replay
runtime.

## Local Intake Artifacts

- Local public-intake directory:
  `experiments/results/public_pr85_intake_20260503_codex/`.
- Downloaded archive:
  `experiments/results/public_pr85_intake_20260503_codex/archive.zip`.
- Static profile:
  `experiments/results/public_pr85_intake_20260503_codex/profile_pr85_bundle.json`.
- Public patch snapshot:
  `experiments/results/public_pr85_intake_20260503_codex/pr85.patch`.
- Replay runtime mirror:
  `experiments/results/public_pr85_intake_20260503_codex/replay_submission/`.

Archive ZIP anatomy:

- Archive bytes: `236328`.
- Archive SHA-256:
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`.
- ZIP members: exactly one stored member, `x`.
- Member `x` file size: `236228`.
- Member `x` compressed size: `236228`.
- Member `x` SHA-256:
  `53bc78effa78cc7850d08a9ddc5488665b93136e9843549d917c17df729a1c50`.
- ZIP overhead: `100` bytes.

Profiled PR85 internal bundle format:
`pr85_v5_micro_24bit_lengths_fixed_bias_region`, with `24` header bytes.

Internal bundle segments:

| segment | bytes | decoded bytes | signal |
|---|---:|---:|---|
| `mask` | `159011` | n/a | QMA9X semantic mask stream |
| `model` | `57074` | `61590` | Brotli-compressed QH0 model payload |
| `pose` | `1487` | `1806` | Brotli-compressed P1D1 pose stream |
| `post` | `1400` | `2400` | Brotli-compressed correction stream |
| `shift` | `226` | `603` | Brotli-compressed SD4 shift stream |
| `frac` | `106` | `179` | Brotli-compressed FV1 stream |
| `frac2` | `149` | `603` | Brotli-compressed FH2 stream |
| `frac3` | `154` | `603` | Brotli-compressed FD3 stream |
| `bias` | `223` | `603` | Brotli-compressed BD1 stream |
| `region` | `273` | `603` | Brotli-compressed RH1 stream |
| `randmulti` | `16101` | `27105` | Brotli-compressed random/multi-action stream |

The mask segment SHA-256 is
`4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`,
matching the PR84/PR81 QMA9 mask stream already profiled locally. PR85's
claimed improvement therefore appears to come from model/pose/correction
streams and runtime behavior, not from a smaller mask entropy stream.

## Exact Replay Jobs And Claims

No new dispatch was launched for this ledger. Current state was read from the
existing claim table and batch state files.

Primary T4 replay:

- Lane claim: `public_pr85_adaptive_masking_joint_frame_model_t4_replay`.
- Claim row time: `2026-05-03T23:23:52Z`.
- Job: `exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z`.
- State file:
  `.omx/state/public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z_batch_jobs.json`.
- Manifest:
  `.omx/state/public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z_manifest.json`.
- Expected archive bytes: `236328`.
- Expected archive SHA-256:
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`.
- Last local observed status at `2026-05-03T23:31:22Z`: `Running`.
- Score claim: false until exact JSON and adjudication land.

Duplicate T4 hedge:

- Lane claim: `public_pr85_adaptive_masking_joint_frame_model_t4_replay`.
- Claim row time: `2026-05-03T23:25:55Z`.
- Job:
  `exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_hedge_g4dn2_20260503T2335Z`.
- State file:
  `.omx/state/public_pr85_adaptive_masking_joint_frame_model_t4_hedge_g4dn2_20260503T2335Z_batch_jobs.json`.
- Manifest:
  `.omx/state/public_pr85_adaptive_masking_joint_frame_model_t4_hedge_g4dn2_20260503T2335Z_manifest.json`.
- Expected archive bytes/SHA: identical to the primary T4 replay.
- Last local observed status at `2026-05-03T23:31:22Z`: `Running`.
- Intended custody: wall-clock hedge only. Stop/close the duplicate after the
  first valid T4 JSON is harvested.

L40S diagnostic replay:

- Lane claim:
  `public_pr85_adaptive_masking_joint_frame_model_l40sdiag`.
- Claim row time: `2026-05-03T23:28:42Z`.
- Job:
  `exact_eval_public_pr85_adaptive_masking_joint_frame_model_l40sdiag_20260503T2338Z`.
- State file:
  `.omx/state/public_pr85_adaptive_masking_joint_frame_model_l40sdiag_20260503T2338Z_batch_jobs.json`.
- Manifest:
  `.omx/state/public_pr85_adaptive_masking_joint_frame_model_l40sdiag_20260503T2338Z_manifest.json`.
- Expected archive bytes/SHA: identical to the T4 replay.
- Last local observed status at `2026-05-03T23:31:22Z`: `Pending`.
- Evidence role: diagnostic component signal only. L40S cannot promote or
  replace T4 contest-faithful evidence.

All three replay records contain provider/UI paths in private state JSON. Those
paths are custody/debug metadata only and must not be copied into public-facing
reports without redaction.

## PR84 Exact Frontier

Current internal exact frontier before PR85 replay:

- Candidate: PR84 adaptive range mask, no-router public archive.
- Artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr84_adaptive_range_mask_no_router_t4_20260503T214008Z/contest_auth_eval.adjudicated.json`.
- Evidence grade: A++ exact CUDA T4 replay.
- Score recomputed from components: `0.2751401491321396`.
- Archive bytes: `215735`.
- Archive SHA-256:
  `a607a6c3ae9b610e6edfb546c3206004ae40fc348ecaef2446b7134a19b8e07f`.
- PoseNet: `0.00049341`.
- SegNet: `0.00061248`.
- Samples: `600`.
- Device: CUDA.

The PR85 public claim would beat PR84 by:

```text
0.25806622496743437 - 0.2751401491321396
= -0.017073924164705212
```

That delta is not promotable until the exact T4 replay confirms the same
component distances on the exact PR85 archive/runtime pair.

## PR84 Plus PR82 Negative

The only current exact PR84+PR82 stack result with score JSON is negative:

- Candidate:
  `exact_eval_pr84_pr82_qrm1_all072_randmulti_packedp_fix1_g4dn_t4_20260503T2253Z`.
- Artifact:
  `experiments/results/lightning_batch/exact_eval_pr84_pr82_qrm1_all072_randmulti_packedp_fix1_g4dn_t4_20260503T2253Z/contest_auth_eval.adjudicated.json`.
- Evidence grade: A++ exact CUDA T4 replay, A-negative method signal.
- Score recomputed from components: `0.3965876514599474`.
- Archive bytes: `232355`.
- Archive SHA-256:
  `d82bc774de93c073b31c1acc9dd24c57b28d14b554cee53c7b7be3404c8f2e2c`.
- PoseNet: `0.00332949`.
- SegNet: `0.00059403`.
- Samples: `600`.
- Delta versus PR84: `+0.12144750232780782`.

Interpretation: the PR82 all-72 randmulti sidecar improves/holds SegNet
slightly relative to PR84 but damages PoseNet and adds bytes, so this measured
stack is not a frontier candidate. Scope the negative to this exact PR84+PR82
randmulti packed-payload configuration.

Related invalid/non-method evidence:

- The earlier expanded PR84+PR82 wave failed with empty remote artifact
  directories: no score JSON, no method conclusion.
- The first packed PR84+PR82 wave mirrored exact archives but failed before
  scoring on a no-router QPOST runtime `NameError`; invalid harness evidence.
- The fixed-source controls-all600 and controls+QRM1 siblings reported
  Lightning infrastructure/no-local-artifact failures in the claim table; no
  score JSON and no method conclusion.
- Later top-k PR84+PR82 exact-eval claims remain unresolved in the current
  claim table and must not be folded into this negative until JSON lands.

## Contest-Faithful Next-Action Branch

1. Harvest whichever PR85 T4 replay first produces
   `contest_auth_eval.json` and `contest_auth_eval.adjudicated.json`.
2. Verify exact archive identity: bytes `236328`, SHA-256
   `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`,
   `600` samples, CUDA device, and adjudication provenance.
3. Recompute the score from JSON components. Do not parse or promote from human
   logs.
4. If the T4 replay matches the public claim and passes component gates,
   promote PR85 over PR84 as the exact frontier, close the primary claim with a
   terminal score row, and close/stop the duplicate hedge without treating it
   as independent evidence.
5. If L40S lands first, record it as diagnostic only and wait for T4 before any
   promotion/rank statement.
6. If T4 replay regresses, mismatches archive identity, misses JSON, or fails
   before scoring, classify the failure as method, runtime, archive-custody, or
   provider infrastructure based on artifacts. Preserve the exact failure
   record and keep PR84 as frontier unless a valid T4 sibling says otherwise.
7. Only after a clean T4 replay should PR85 be considered for robust-current
   native intake, stack interaction tests, or public report promotion. Any such
   follow-up needs a fresh lane claim and must keep the PR85 public runtime
   replay separate from our own runtime-custody experiments.

## Hygiene Notes

- This ledger intentionally avoids raw Lightning UI links, SSH endpoints,
  provider account identifiers, and local absolute paths.
- Relative `.omx/state` and `experiments/results` paths are included because
  this is an internal research ledger.
- No sanitized report fragment was written: `reports/latest.md` and
  `reports/writeup_working.md` already contain unrelated dirty edits, and no
  obvious PR85-specific public report target exists.

## Codex Update 2026-05-03T23:52Z

PR85 is now the exact internal frontier.

- Primary T4 replay artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.adjudicated.json`.
- Hedge T4 replay artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_hedge_g4dn2_20260503T2335Z/contest_auth_eval.adjudicated.json`.
- Exact score, recomputed from components: `0.25806611029397786`.
- Archive bytes: `236328`.
- Archive SHA-256:
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`.
- PoseNet: `0.0001894`.
- SegNet: `0.00057185`.
- Samples: `600`.
- Device: CUDA Tesla T4.
- Evidence grade: A++ contest T4.
- Promotion eligible: true.

The L40S diagnostic replay also completed and was harvested:

- Artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_l40sdiag_20260503T2338Z/contest_auth_eval.adjudicated.json`.
- Score: `0.2593036547804398`.
- Evidence role: diagnostic only; not promotion-grade because hardware is L40S.

The first exact PR85 replay improved over PR84 by:

```text
0.25806611029397786 - 0.2751401491321396
= -0.01707403883816172
```

### Safe Side-Channel Ablation Wave

The PR85 v5 runtime fixes `bias` and `region` compressed lengths at `223` and
`273` bytes. Earlier candidate dirs that shortened those fixed slices were
stale/unsafe and were removed from the generated results tree. The guarded
builder now keeps only parser-safe variable-length ablations:

- Builder:
  `experiments/build_pr85_sidechannel_ablation_candidates.py`.
- Focused tests:
  `src/tac/tests/test_build_pr85_sidechannel_ablation_candidates.py`.
- Verification: `7 passed`.
- Candidate summary:
  `experiments/results/public_pr85_sidechannel_ablations_20260503_codex/candidate_summary.json`.
- Staging manifest:
  `.omx/state/pr85_safe_sidechannel_ablation_exact_20260503T2350Z_manifest.json`.

Safe byte-screen candidates:

| policy | bytes | SHA-256 | bytes vs PR85 |
| --- | ---: | --- | ---: |
| `minus_randmulti` | `220239` | `f8ddecc5a96d013453acfdd5bbfc3265048f93ca65ccf97755cf45e3553ae995` | `-16089` |
| `minus_all_safe_corrections` | `218271` | `ff23264c58666d041820eda73266f09a7ba2318d8233aedeb2790acd0ab6da41` | `-18057` |
| `minus_post_motion` | `234360` | `be449e36b6d8ffeceb7a9984daad82240c4cce9f8af6b1f6b5dd4acd22c90262` | `-1968` |
| `minus_post` | `234941` | `b370e0c406c4cc1080162a24f81b02f488a8eebac36a6aa037c1e0f658ba587d` | `-1387` |
| `minus_motion_stack` | `235747` | `e0e68f6d58a0be1098bec59d3595c2f3b9ad371a58a351f2cdad2f0d7337a319` | `-581` |

Submitted exact-eval wave:

- T4 promotion-grade:
  `exact_eval_pr85_minus_randmulti_t4_20260503T2350Z`,
  state `.omx/state/pr85_minus_randmulti_t4_20260503T2350Z_batch_jobs.json`.
- T4 promotion-grade:
  `exact_eval_pr85_minus_post_motion_t4_20260503T2350Z`,
  state `.omx/state/pr85_minus_post_motion_t4_20260503T2350Z_batch_jobs.json`.
- L40S diagnostic:
  `exact_eval_pr85_minus_all_safe_corrections_l40sdiag_20260503T2350Z`,
  state `.omx/state/pr85_minus_all_safe_corrections_l40sdiag_20260503T2350Z_batch_jobs.json`.
- L40S diagnostic:
  `exact_eval_pr85_minus_post_l40sdiag_20260503T2350Z`,
  state `.omx/state/pr85_minus_post_l40sdiag_20260503T2350Z_batch_jobs.json`.
- L40S diagnostic:
  `exact_eval_pr85_minus_motion_stack_l40sdiag_20260503T2350Z`,
  state `.omx/state/pr85_minus_motion_stack_l40sdiag_20260503T2350Z_batch_jobs.json`.

These are attribution and pruning tests. A T4 score below `0.25806611029397786`
promotes as a new exact frontier. L40S scores remain diagnostic and must be T4
confirmed before any rank claim.

### 12-Hour Council Decision

The read-only council session `019df03b-e066-7330-a4a0-b48d6aa148ea` returned
the same next branch: PR85 side-channel lossless/near-lossless recoding and
randmulti water-fill are the highest-confidence below-PR85 paths. In concrete
terms:

1. Harvest this ablation wave.
2. If any removal is neutral or positive, immediately T4-promote the smallest
   winning archive.
3. Implement decoded-parity-preserving recoders for `randmulti`, `post`, and
   the small motion streams, with fixed-slice parser guards for `bias` and
   `region`.
4. Use the exact component trace to water-fill randmulti groups and hard-pair
   micro-actions.
5. Stop broad architecture/training work from occupying the critical path until
   this PR85 grammar surface is exhausted or exact negatives prove it spent.

## Codex Update 2026-05-04T00:20Z

The safe side-channel ablation wave has returned exact-negative T4 evidence.
PR85's explicit correction bytes are not dead weight; they buy substantial
PoseNet/SegNet stability, so the next path is recoding/water-filling inside the
correction grammar rather than deleting whole streams.

Promotion-grade T4 exact results harvested:

| candidate | bytes | score | delta vs PR85 | PoseNet | SegNet | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `minus_randmulti` | `220239` | `0.2826740754796167` | `+0.024607965185638847` | `0.0005759` | `0.00060138` | exact negative |
| `minus_post` | `234941` | `0.3093560838497741` | `+0.05128997355579623` | `0.00087577` | `0.00059336` | exact negative |
| `minus_motion_stack` | `235747` | `0.363610914426998` | `+0.10554480413302014` | `0.00223358` | `0.00057185` | exact negative |
| `minus_post_motion` | `234360` | `0.38432756803992785` | `+0.12626145774595` | `0.0028541` | `0.00059336` | exact negative |

Primary artifacts:

- `experiments/results/lightning_batch/exact_eval_pr85_minus_randmulti_t4_20260504T0002Z/contest_auth_eval.adjudicated.json`
- `experiments/results/lightning_batch/exact_eval_pr85_minus_post_t4_20260504T0002Z/contest_auth_eval.adjudicated.json`
- `experiments/results/lightning_batch/exact_eval_pr85_minus_motion_stack_t4_20260504T0002Z/contest_auth_eval.adjudicated.json`
- `experiments/results/lightning_batch/exact_eval_pr85_minus_post_motion_t4_20260504T0002Z/contest_auth_eval.adjudicated.json`

Diagnostic L40S results support the same branch:

- `minus_all_safe_corrections`: score `0.2800192866079282`, bytes `218271`.
- `minus_motion_stack`: score `0.3641375270069792`, bytes `235747`.

Two duplicate T4 jobs from the earlier wave were superseded by the completed
`20260504T0002Z` siblings. The Lightning stop call timed out, but status
reconciled back to Pending and terminal claim rows were appended as
`stopped_superseded_duplicate`.

### Magic Codec / Rust / HPAC Notes

- `runtime-rs/crates/qma-codec` now exists as a Rust-first QMA9 feasibility
  crate with fail-closed header parsing and deterministic storage-order decode
  tests. It is not wired into inflate yet; public PR85 replay still preserves
  `range_mask_codec.cpp` until full archive parity and runtime custody are
  proven.
- `experiments/build_problem_space_manifest.py` now uses canonical exact score
  as frontier truth instead of silently trusting rounded visible components,
  and missing `promotion_eligible` remains `null` rather than becoming false.
  The manifest also records the durable Rust-over-C++ preference for
  Apogee-owned native codecs.
- A local PR86 HPAC-to-PR85 mask parity probe failed before parity with:
  `AssertionError: Tried to decode from compressed data that is invalid for the
  employed entropy model.` Artifact:
  `experiments/results/public_pr86_intake_20260504_codex/pr86_hpac_pr85_qma9_parity_probe.json`.
  This is a decode-contract/dependency negative, not a method kill. PR86 HPAC
  remains an architecture prior until the exact PR86 entropy runtime is
  reproduced and decoded-token parity is proven.
- The local recode candidate builder now records runtime dependency closure
  over `inflate.sh`, `inflate.py`, and `range_mask_codec.cpp`, but no tested
  recode produced a negative byte delta under strict semantic parity. Artifact:
  `experiments/results/magic_codec_pr85_qma9_prefix_20260504_codex/pr85_sidechannel_recodes/candidate_summary.json`.

Next branch:

1. Let the still-running `minus_all_safe_corrections` T4 finish only as a
   cliff-map confirmation; it is not expected to beat PR85 after the L40S
   diagnostic and individual T4 negatives.
2. Move the main PR85 path to group-level `randmulti` and `post/motion`
   water-fill: keep correction atoms with proven component value, recode or
   prune only low-marginal atoms, and dispatch only candidates with strict byte
   decrease plus semantic/runtime closure.
3. Reproduce PR86 HPAC end-to-end in an isolated local submission runtime
   before attempting any PR85 HPAC transplant.
4. Use Rust for new native Apogee-owned codec/parsing work unless public replay
   custody requires preserving an existing C++ source file.

## Codex Update 2026-05-04T02:05Z

The PR85 fixed-runtime bridge blocker was reduced to a single missing runtime
decoder and then closed locally.  The bridge now expands PR85's one-member
public `x` payload into standard robust_current members:

- `masks.qma9`
- `renderer.bin` with decoded `QH0` JointFrameGenerator bytes
- `optimized_poses.bin` with P1D1 materialized to raw fp16
- `qpost.bin` with PR85 side channels and headerless `randmulti` transcoded to
  runtime-supported `QRM1`

New runtime support:

- `src/tac/qh0_renderer_codec.py` decodes `QH0`/`QM0` without importing the
  public replay runtime or using pickle.
- `submissions/robust_current/inflate_renderer.py` now loads `QH0`/`QM0`
  through the canonical JointFrameGenerator shim.

Local bridge candidate:

- Archive:
  `experiments/results/pr85_fixed_runtime_bridge_candidates_20260504_codex/expanded_qpost_qrm1_posefp16/archive.zip`
- Bytes: `239966`
- SHA-256:
  `3c4e1b2d6a2743be17495b533f35d78d15723482e9fcebd6c49c263e19492d97`
- Dispatch gate:
  `eligible_for_exact_eval_after_lane_claim`

Readiness:

- `experiments/results/public_pr85_intake_20260503_codex/pr85_fixed_runtime_readiness_preflight.json`
  reports `ready_for_fixed_runtime_exact_eval=true` and no blockers.
- Focused local verification passed: `17 passed` across QH0 codec, PR85 bundle,
  fixed-runtime bridge builder, and fixed-runtime readiness tests.
- `py_compile` passed for the touched runtime/bridge files.

Remote diagnostic exact eval:

- Claim:
  `public_pr85_fixed_runtime_bridge_l40sdiag`.
- Job:
  `exact_eval_public_pr85_fixed_runtime_bridge_l40sdiag_20260504T0208Z`.
- Machine:
  `g6e.4xlarge` / L40S diagnostic.
- State:
  `.omx/state/public_pr85_fixed_runtime_bridge_l40sdiag_20260504T0208Z_batch_jobs.json`.
- Source manifest:
  `.omx/state/public_pr85_fixed_runtime_bridge_l40sdiag_20260504T0208Z_manifest.json`.
- Initial status:
  `Pending`.

This bridge is not expected to beat PR85 by itself because the fixed-runtime
archive is `3638` bytes larger than the public single-member archive.  Its
purpose is to create a standard, auditable runtime substrate for PR79-style
action transfer, qpost/randmulti water-fill, mask/pose/member swaps, and exact
component tracing without depending on public replay source as the inflate
runtime.

### Correction-Stream Lagrangian Read From Exact Ablations

The exact PR85 ablations imply that the visible correction streams have very
different marginal value densities.  Using the PR85 T4 component trace as the
reference (`score=0.258066478975606`, bytes `236328`), stream deletion has the
following measured effects:

| removed stream(s) | bytes saved | rate saved | distortion penalty | net score delta |
| --- | ---: | ---: | ---: | ---: |
| `randmulti` | `16089` | `0.01071300469678262` | `0.03532117327588133` | `+0.02460816857909871` |
| `post` | `1387` | `0.00092354636798044` | `0.052213712847052785` | `+0.051290166479072344` |
| `motion_stack` | `581` | `0.000386864051764` | `0.10593110473499335` | `+0.10554424068322935` |
| `post_motion` | `1968` | `0.0013104104197444` | `0.12757183085951247` | `+0.1262614204397681` |

Interpretation:

- `motion_stack` and `post` are high-value protected atoms.  Deleting them is
  a byte win but a severe PoseNet loss; they should be recoded only under
  decoded-parity or exact group-level trust-region evidence.
- `randmulti` is the largest byte surface and still valuable, but its value per
  byte is much lower than the smaller streams.  It remains the best first
  target for group-level water-fill, PR79-style sparse-action replacement, and
  QRM1 policy optimization.
- The non-additive `post_motion` result is worse than summing isolated rate
  savings and confirms that stream interactions must be exact-evaluated as
  stacked archives, not composed from scalar deltas.

Dispatch implication:

Do not submit more whole-stream deletion variants.  The next PR85 variants
should preserve post/motion semantics, vary `randmulti` or sparse action
groups, and record group/pair policies in the manifest before any exact eval
claim.

### 2026-05-04T02:17Z QMA9 Archive Whitelist Closure

The first fixed-runtime bridge diagnostic did not reach inflate or scorer
evidence.  It failed inside the strict archive validator:

- Job:
  `exact_eval_public_pr85_fixed_runtime_bridge_l40sdiag_20260504T0208Z`.
- Archive bytes `239966`, SHA
  `3c4e1b2d6a2743be17495b533f35d78d15723482e9fcebd6c49c263e19492d97`.
- Failure class:
  `runtime_or_harness_failure_before_score_json`.
- Root cause:
  `experiments/contest_auth_eval.py` did not yet whitelist charged
  `masks.qma9` members.

This is a harness allowlist bug, not PR85 bridge method evidence.  The terminal
dispatch claim was closed as `failed_archive_whitelist_harness_bug`.

Permanent guard:

- `_KNOWN_ARCHIVE_SUFFIXES` now includes `.qma9`.
- `test_archive_member_validator_accepts_charged_pr85_qma9_members` protects
  PR85 bridge archives.
- Focused verification after the fix:
  `20 passed` across QH0 codec, PR85 bundle, fixed-runtime bridge readiness,
  and PR85/PR86 charged-member archive-validator tests.

Relaunch:

- Claim:
  `public_pr85_fixed_runtime_bridge_l40sdiag_retry1`.
- Job:
  `exact_eval_public_pr85_fixed_runtime_bridge_l40sdiag_retry1_20260504T0216Z`.
- Source manifest:
  `.omx/state/public_pr85_fixed_runtime_bridge_l40sdiag_retry1_20260504T0216Z_manifest.json`.
- Initial status:
  `Pending`, cost `$0.0`.

Do not dispatch PR85 bridge sparse-action variants until this bridge substrate
produces exact CUDA score JSON or a new runtime failure is harvested and fixed.

Additional wall-clock hedges after queue delay:

- `exact_eval_public_pr85_fixed_runtime_bridge_rtxprodiag_20260504T0220Z`
  on `g7e.4xlarge` / RTX PRO diagnostic, same archive SHA/bytes and same
  source manifest.
- `exact_eval_public_pr85_fixed_runtime_bridge_t4_retry1_20260504T0223Z`
  on `g4dn.2xlarge` / T4 with the required cu124 Torch pin, same archive
  SHA/bytes and same source manifest.

These are duplicate runtime-substrate checks, not distinct scientific lanes.
Harvest the first terminal score JSON, then close or stop redundant siblings
so duplicate spend does not persist.

### 2026-05-04T02:27Z Sparse-Action Dispatch Guard

The bridge sparse-action builder now converts the exact PR85 deletion negatives
into a hard dispatch preflight:

- Whole `randmulti`, whole `post`, whole `motion`, and combined `post_motion`
  deletion are blocked.
- All PR85 post/motion qpost groups are protected until an explicit exact-
  evidence override covers every blocker id.
- Blocked candidates receive
  `ready_for_exact_eval_dispatch_claim=false` and
  `dispatch_gate=planning_only/preflight_blocked`.

Regenerated status:

- `dispatchable_candidate_count=0`.
- `dispatch_preflight_blocked_candidate_count=4`.
- Current candidate priority if future exact evidence justifies an override:
  first `bridge_rm_top001_post123_motion`, second
  `bridge_rm_top004_post23_motion`; keep `bridge_rm_top008_motion_only` and
  `bridge_no_randmulti_post123_motion` blocked by default.

Verification:

- `8 passed` for
  `src/tac/tests/test_build_pr85_bridge_sparse_action_candidates.py`.
- Scoped `git diff --check` is clean.

This permanently fixes the bug class where a byte-screened bridge derivative
could be queued despite exact evidence that it deletes protected correction
families.

### 2026-05-04T02:44Z PR85 Single-Atom T4 Dispatch

The next exact PR85 atom is byte-closed and submitted:

- Candidate:
  `preserve_post_all_shift_frac2_frac3`.
- Archive:
  `experiments/results/pr85_post_motion_group_policy_candidates_20260504_codex/preserve_post_all_shift_frac2_frac3/archive.zip`.
- Archive bytes `236231`, SHA
  `6a780cdf32389878fd8c07ff19750fa69bef00f880a58032d7575dae20b8a1ae`.
- Source manifest:
  `.omx/state/pr85_preserve_post_all_shift_frac2_frac3_t4_20260504T0242Z_manifest.json`.
- Job:
  `exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4_20260504T0242Z`.
- Dispatch claim:
  `pr85_preserve_post_all_shift_frac2_frac3_t4`.
- Initial Lightning status:
  `Pending`, cost `$0.0`.

This candidate preserves all PR85 post stages plus `motion_shift`,
`motion_frac2`, and `motion_frac3`, and neutralizes only `motion_frac`.
Formula-only rate arithmetic predicts `-0.0000645883` score if components do
not drift.  That is not a score claim; the job is queued solely to get exact
T4 CUDA truth on the exact archive bytes.

Submit hardening/fixes encountered before successful queue:

- `--studio this_studio` was a filesystem placeholder, not the Lightning SDK
  Studio identity.  Successful submission used
  `--studio lossy-compression-challenge --teamspace comma-lab --user adpena`.
- The replay runtime required `replay_submission/README.md` in the staged
  manifest; missing runtime metadata now blocks before dispatch.
- T4/g4dn exact eval requires explicit inflate-side CUDA wheel pins:
  `INFLATE_TORCH_SPEC=torch==2.5.1+cu124`,
  `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`, and
  `UV_INDEX_STRATEGY=unsafe-best-match`.

Next action:

Harvest via state-derived Lightning path when terminal, adjudicate against
PR85 score `0.25806611029397786`, then close the dispatch claim as either a
new frontier, exact negative, or precise pre-score failure class.

Wall-clock hedge:

- Claim:
  `pr85_preserve_post_all_shift_frac2_frac3_t4_hedge_g4dn2`.
- Job:
  `exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4hedge_g4dn2_20260504T0246Z`.
- Machine:
  `g4dn.2xlarge` / T4.
- Same archive bytes and SHA as the primary job.
- Initial status:
  `Pending`, cost `$0.0`.

This is duplicate custody for wall-clock only.  Harvest whichever job finishes
first, then close or stop the redundant sibling so the same archive is not
counted twice.

### 2026-05-04T02:55Z PR85 Bit-Budget Profile

The PR85 archive now has a deterministic bit-budget profile for stack
selection:

- Tool:
  `experiments/profile_pr85_archive_bit_budget.py`.
- JSON:
  `experiments/results/pr85_archive_bit_budget_20260504_codex/profile_pr85_archive_bit_budget.json`.
- Markdown:
  `experiments/results/pr85_archive_bit_budget_20260504_codex/profile_pr85_archive_bit_budget.md`.
- Verification:
  `2 passed` for `src/tac/tests/test_profile_pr85_archive_bit_budget.py`.

Top byte surfaces:

- `mask`: `159011` bytes, `67.284029%` of archive,
  formula-only rate contribution `0.105878897995`.
- `model`: `57074` bytes, `24.1503334%` of archive,
  formula-only rate contribution `0.03800323389`.
- `randmulti`: `16101` bytes, `6.8121398%` of archive, protected by exact
  deletion negatives; target recode/water-fill, not deletion.
- `pose`: `1487` bytes, `0.6292103%` of archive.
- `post`: `1400` bytes, `0.592397%` of archive, protected by exact deletion
  negatives.

Decision:

The large remaining contest-faithful levers are mask entropy/model
self-compression and protected correction-stream recoding.  Blind deletion of
post/motion/randmulti stays blocked by exact negative evidence.

### 2026-05-04T02:58Z PR85 QMA Mode Sweep

Full-stream PR85 C++ range-mask mode sweep was run against the extracted
QMA9 token source:

- Raw token SHA-256:
  `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`.
- Source QMA9 bytes:
  `159011`, SHA-256
  `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`.
- Summary:
  `experiments/results/pr85_qma9_mode_sweep_20260504_codex/pr85_qma9_cpp_mode_sweep_summary.json`.
- Evidence grade:
  local planning / byte and decode-parity screen only.

Result:

- `adaptive9bin` is the submitted source mode and remains best at `159011`
  bytes with token parity.
- Closest alternate by bytes, `adaptive9up2left2`, is larger by `2023` bytes
  and does not decode to the PR85 token SHA through the submitted runtime's
  QMA9 decode path.
- Best nonbaseline runtime-compatible alternate, `adaptive8prpdup2`, is larger
  by `4477` bytes.
- No exposed C++ mode is accepted for archive replacement.

Decision:

Do not spend exact eval on exposed QMA mode toggles.  The mask-byte path must
move to a PR85-owned fitted entropy model or a new runtime-backed grammar with
decoded-token SHA parity and archive/runtime closure.

### 2026-05-04T03:05Z PR85 Preserve-Post/Motion-Fraction Atom Exact Negative

The `preserve_post_all_shift_frac2_frac3` exact T4 eval completed and was
harvested through the state-derived Lightning path.

- Primary job:
  `exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4_20260504T0242Z`.
- Hedge job:
  `exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4hedge_g4dn2_20260504T0246Z`.
- Archive bytes:
  `236231`.
- Archive SHA-256:
  `6a780cdf32389878fd8c07ff19750fa69bef00f880a58032d7575dae20b8a1ae`.
- Primary artifact:
  `experiments/results/lightning_batch/exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4_20260504T0242Z/contest_auth_eval.adjudicated.json`.
- Hedge artifact:
  `experiments/results/lightning_batch/exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4hedge_g4dn2_20260504T0246Z/contest_auth_eval.adjudicated.json`.
- Hardware:
  Tesla T4, CUDA, `600` samples.
- Exact score:
  `0.27100583104425036`.
- Components:
  SegNet `0.00057185`, PoseNet `0.0003195`.
- Delta versus PR85 exact T4 baseline:
  `+0.012939720750272499`.
- Evidence grade:
  `A++ contest T4`.

Interpretation:

The atom saved `97` bytes, but neutralizing only `motion_frac` perturbed
PoseNet enough to dominate the rate win.  This is a scoped exact negative for
that measured implementation/config, not a broad kill of post/motion
sidechannel compression.  The duplicate hedge completed before the stop
request took effect; it reproduced the same score and was closed as a duplicate
negative with no distinct scientific claim.

Decision:

Do not exact-eval more blind protected sidechannel deletions.  Future
post/motion/randmulti work must either prove decoded-output parity, remain
inside a measured trust region, or carry a component-response reason that the
PoseNet increase is paid for by a larger byte or SegNet win.

### 2026-05-04T03:08Z PR85 QMA9 Escape Screens

Bounded-prefix local escape screens were run against the PR85 QMA9 planning
wrapper:

- Wrapper:
  `experiments/results/pr85_qma9_escape_screens_20260504_codex/pr85_qma9_planning_fixedslice_wrapper.zip`.
- Wrapper bytes:
  `159172`.
- Wrapper SHA-256:
  `23337100285209b989e960134b3fdbb9a0ac76032867f82b5bc32eb6d0e60694`.
- Source QMA9 stream bytes:
  `159011`.
- Source QMA9 stream SHA-256:
  `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`.
- Evidence grade:
  `empirical/planning_only`.
- Score claim:
  `false`.
- Remote dispatch:
  none.

Screens:

- `QMC1` context-conditioned copy/run escape, prefix `16`, min-run `64`:
  subset delta `+429` bytes, projected full-mask delta `+21777` bytes.
- `QMC1` min-run `32`:
  subset delta `+487` bytes, projected `+23951` bytes.
- `QMC1` min-run `16`:
  subset delta `+513` bytes, projected `+24927` bytes.
- `QMH1` horizontal run-tail escape, min-run `128`:
  subset delta `+24` bytes, projected `+6589` bytes.
- `QMH1` min-run `64`:
  subset delta `+38` bytes, projected `+7114` bytes.
- `QMH1` min-run `32`:
  subset delta `+54` bytes, projected `+7714` bytes.
- `QMH1` min-run `64`, allowing up-match:
  subset delta `+12676` bytes, projected `+481039` bytes.
- `QMB1` vertical block escape, block width `32`:
  subset delta `+1612` bytes, projected `+66139` bytes.
- `QMB1` block width `16`:
  subset delta `+2319` bytes, projected `+92651` bytes.
- `QMB1` block width `8`:
  subset delta `+3115` bytes, projected `+122501` bytes.

Decision:

The current local escape grammars are not byte-competitive with PR85's QMA9
model on the tested prefix.  The closest miss is QMH1 min-run `128`, but even
that loses bytes before runtime overhead.  Keep the artifacts as negative
training signal for the mask-entropy profiler; do not promote any QMC1/QMH1/
QMB1 screen to exact eval without a full-stream byte win and runtime parity.

Generic raw-token compression sanity check:

- Raw PR85 token tensor:
  `117964800` bytes, SHA-256
  `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`.
- `zlib` level 9:
  `1062758` bytes, `+903747` versus QMA9.
- Python `lzma` preset 6:
  `756712` bytes, `+597701` versus QMA9.
- Python `lzma` preset 9 extreme:
  `618968` bytes, `+459957` versus QMA9.

This rules out a generic raw-layout container swap as a serious PR85 mask
candidate.  Any mask win must be a better fitted context/range model, a
changed representation with scorer-safe geometry, or a runtime grammar that
beats QMA9 while preserving decoded tokens.

### 2026-05-04T04:20Z QH1 Lossless Model-Repack Greenup And Exact Negative

Implemented a production-style local greenup loop for a PR85 `QH1` model
repack path.  QH1 is a lossless wrapper around decoded PR85 `QH0` renderer
bytes: it stores a compressed base template plus compressed record patches and
reconstructs the exact original `QH0` byte stream before the reviewed tensor
decoder runs.

Changed/added code:

- `src/tac/qh0_renderer_codec.py`: `reconstruct_qh1_payload()` plus decode
  delegation through the existing QH0/QM0 loader.
- `submissions/robust_current/inflate_renderer.py`: runtime recognition of
  `QH1` renderer payloads.
- `src/tac/preflight.py`: renderer magic and unsafe-loader guard recognize
  `QH1`.
- `src/tac/pr85_bundle.py`: PR85 fixed-runtime expansion accepts decoded
  `QH1` renderer members for local build/preflight work.
- `experiments/build_pr85_qh1_model_candidate.py`: deterministic local
  byte-screen builder for PR85 single-member `x` candidates.
- `src/tac/tests/test_qh0_renderer_codec.py` and
  `src/tac/tests/test_build_pr85_qh1_model_candidate.py`: parity and
  fail-closed tests.

Verification:

- `py_compile` passed for the touched Python/runtime files.
- Focused tests passed:
  `src/tac/tests/test_qh0_renderer_codec.py`
  `src/tac/tests/test_build_pr85_qh1_model_candidate.py`
  -> `9 passed`.
- Real PR85 `QH1` reconstruction verified byte-identical to source `QH0`.
- QH1 malformed/trailing/overlapping patch cases fail closed.

Byte-screen result:

- Artifact:
  `experiments/results/pr85_qh1_model_candidates_20260504_codex/candidate_summary.json`.
- Source PR85 archive:
  `236328` bytes, SHA-256
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`.
- Source model segment:
  `57074` bytes.
- Best QH1 candidate:
  `qh1_top1_minsaving16`.
- Best candidate archive:
  `237204` bytes, SHA-256
  `31bfb8e35c6cf2b6b77a8a5ad9a0c37d48c02e277c97c26712071cab8a6cbeb2`.
- Delta versus PR85:
  `+876` archive bytes, formula-only rate penalty
  `+0.0005832924429350221`.
- Selected record:
  `pose_mlp.2.weight`, `4609` bytes, record-local Brotli patch `3878`
  bytes.
- Dispatch:
  none.
- Evidence grade:
  `empirical/planning_only exact-lossless-byte-screen`.

Decision:

Do not exact-eval QH1 record-wrapper candidates.  The design is lossless and
safe in principle, but the base/header/patch overhead beats the record-local
compression gain.  This retires the measured `QH1` wrapper implementation, not
the broader QH0/model self-compression family.  Any next model-byte attempt
must be a true deterministic QH0 serializer change or learned/quantized model
rewrite with runtime parity, not a patch wrapper around the existing byte
stream.

Council synthesis from the recovered workers:

- PR85 exact T4 replay remains the measured A++ anchor:
  score `0.25806611029397786`, `236328` bytes, PoseNet `0.0001894`,
  SegNet `0.00057185`.
- PR86 remains public design evidence until HPAC decode/reencode parity is
  locally reproduced.
- PR85's mask segment is `159011` bytes (`67.28%` of archive); simple context
  entropy replacement is byte-negative, so mask work must be QMA9-native or a
  structurally different representation.
- PR85's model segment is `57074` bytes (`24.15%`); generic wrapper/recode and
  QH1 are byte-negative, so model work must target the serializer or the model
  itself.
- `randmulti`/post/motion streams are protected by exact negatives.  Future
  savings there need group-level water-fill, decoded-output parity, or
  component-response support.

Mathematical operating point:

- At PR85, `dscore/dbyte = 25 / 37545489 = 6.658589531e-7`.
- `dscore/dseg = 100`.
- `dscore/dpose = 5 / sqrt(10 * 0.0001894) ~= 114.89`.

This explains the current leaderboard trajectory: after the QMA9 semantic-mask
pivot, the winning moves are not generic compression but charged
sufficient-statistic programs.  Tiny PoseNet drift can erase thousands of
bytes, so every candidate must be priced as an atom with rate benefit,
component-risk, trust-region evidence, and exact CUDA confirmation.

### 2026-05-04T04:35Z PR85 QMA9 Macro-Prior And Randmulti Waterfill Closure

Additional PR85 mask and sidechannel probes were reconciled before dispatching
new GPU work.

QMA9 macro-prior screen:

- Artifact directory:
  `experiments/results/pr85_qma9_macro_prior_screen_20260504_codex`.
- Baseline public PR85 `range_mask_codec.cpp` rebuild exactly reproduced the
  charged QMA9 mask stream size:
  `159011` bytes (`158991` bitstream bytes plus `20` model bytes).
- Tested compile-time prior variants:
  - `u2_l4_p3_f3_i60000`: `159044` bytes.
  - `u3_l3_p3_f3_i60000`: `159019` bytes.
  - `u3_l4_p2_f3_i60000`: `159015` bytes.
  - `u3_l4_p3_f2_i60000`: `159022` bytes.
  - `u3_l4_p3_f3_i30000`: `159012` bytes.
  - `u3_l4_p3_f3_i65000`: `159011` bytes.
  - `u3_l4_p3_f4_i60000`: `159012` bytes.
  - `u3_l4_p4_f3_i60000`: `159012` bytes.
  - `u3_l5_p3_f3_i60000`: `159011` bytes.
  - `u4_l4_p3_f3_i60000`: `159018` bytes.
- Result:
  no byte-improving macro-prior variant.  The best variants only tie the
  baseline, so this screen is a local byte-negative/tie result and does not
  justify exact eval.

Randmulti waterfill exact wave:

- Candidate family:
  `experiments/results/pr85_randmulti_group_policy_candidates_20260504_codex`.
- Exact T4/L40S results:

| candidate | hardware | archive bytes | score | PoseNet | SegNet | decision |
|---|---:|---:|---:|---:|---:|---|
| `top001` | T4 | `221736` | `0.2807384658678841` | `0.00053225` | `0.00060138` | A++ exact negative |
| `top001` | L40S | `221736` | `0.28132865593655954` | `0.00054122` | `0.00060116` | diagnostic negative |
| `top004` | T4 | `222964` | `0.3024608857881686` | `0.00088594` | `0.00059874` | A++ exact negative |
| `top004_fb_allowfb` | L40S | `223344` | `0.30525181986353395` | `0.00093154` | `0.0006002` | diagnostic negative |
| `top008` | T4 | `224292` | `0.30536973018825064` | `0.00094704` | `0.00058707` | A++ exact negative |
| `top016` | L40S | `226766` | `0.3262003862199225` | `0.00137101` | `0.00058116` | diagnostic negative |

Interpretation:

- Removing or coarsening randmulti groups gives real rate savings, but the
  saved bits are buying PoseNet-critical correction semantics.  At PR85's
  operating point, `top001` saves `14592` bytes but loses roughly `0.00034285`
  PoseNet distance versus PR85, which costs far more score than the rate win.
- This retires the measured randmulti group-deletion/waterfill implementation
  family, not the broader sidechannel compression target.
- Future randmulti/post/motion work must be either decoded-output preserving
  recode, learned/scorer-gradient selected sparse replacement, or a stack whose
  exact CUDA component trace proves the PoseNet loss has been neutralized.

PR86 intake integration:

- Worker ledger:
  `.omx/research/pr86_hpac_parity_worker_20260504.md`.
- Guard tool:
  `experiments/diagnose_pr86_hpac_parity.py`.
- Verification:
  `src/tac/tests/test_diagnose_pr86_hpac_parity.py` -> `6 passed`.
- Current PR86 status:
  reproducible archive custody exists (`207579` bytes, SHA-256
  `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`),
  but local replay is blocked by
  `hpac_entropy_decode_contract_mismatch` at frame `0`, HPAC group `10`,
  symbol `191`.
- Dispatch policy:
PR86 remains external design evidence only until a byte-exact HPAC
decode/reencode gate reproduces `tokens.bin` SHA-256
`14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225`.

### 2026-05-04T04:50Z Scorer-Gradient Profiler Baseline Re-anchoring

The scorer-gradient atom profiler landed as a planning-only tool, but its
worker ledger was generated from the `preserve_post_all_shift_frac2_frac3`
exact negative rather than the true PR85 frontier.  The tool is valid; the
default artifact choice was the confound.  It was rerun against the true PR85
T4 anchor:

```bash
.venv/bin/python experiments/plan_pr85_scorer_gradient_atoms.py \
  --exact-eval-json experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.adjudicated.json \
  --output-json /tmp/pr85_scorer_gradient_atoms_plan_baseline.json \
  --ledger-md /tmp/pr85_scorer_gradient_atoms_plan_baseline.md \
  --max-atoms 64
```

Baseline formula derivatives:

- `dscore/dseg = 100.0`
- `dscore/dpose = 114.88941471483075`
- `dscore/dbyte = 6.658589531221714e-07`
- recomputed score from rounded public components:
  `0.25806622496743437`
- canonical exact JSON score:
  `0.25806611029397786`

Top true-PR85 pair opportunities:

| pair atom | first-order score opportunity | break-even charged bytes |
|---|---:|---:|
| `pair_0192` | `0.00034404413500734173` | `516.6922114573065` |
| `pair_0060` | `0.00032246634072639115` | `484.2862579445188` |
| `pair_0164` | `0.0003097759391894709` | `465.22756469211794` |
| `pair_0197` | `0.0003015551815877018` | `452.8814701277624` |
| `pair_0070` | `0.0002987457568988698` | `448.6622211777276` |
| `pair_0496` | `0.00029818525587378686` | `447.820449774858` |
| `pair_0106` | `0.00029151605494359765` | `437.8045133683296` |
| `pair_0522` | `0.00028768490941409006` | `432.0508240749086` |

Decision:

- Treat the worker-generated `pair_0139` ranking as a ranking for the exact
  negative basin, not the true PR85 frontier.
- True PR85 pair atoms are polish-scale unless the charged implementation is
  extremely cheap or stacks with a larger parity-preserving recode.
- Next pair-atom work must build a closed archive candidate and prove non-noop
  payload/raw-output change; scorer-gradient rank alone remains
  `blocked_planning_only`.

### 2026-05-04T05:00Z Full-Stack Matrix Exact-Negative Guard Fix

The PR85 full-stack opportunity matrix was refreshed after the randmulti exact
wave and exposed a stale-priority bug: empirical randmulti byte-screen
candidates still ranked above other stack plans even though exact T4/L40S
evidence had already shown the measured group-deletion/waterfill configs
regress.

Permanent fix:

- `experiments/plan_pr85_full_stack_opportunity_matrix.py` now detects scoped
  full-sample exact regressions for candidate families, starting with
  `pr85_randmulti`.
- `src/tac/tests/test_plan_pr85_full_stack_opportunity_matrix.py` now asserts
  that a scoped randmulti exact negative marks
  `protected_randmulti_group_waterfill` as `already_refuted=true`,
  `blocked=true`, and removes it from `top_stack_plans`.

Verification:

```bash
.venv/bin/python -m py_compile \
  experiments/plan_pr85_full_stack_opportunity_matrix.py \
  src/tac/tests/test_plan_pr85_full_stack_opportunity_matrix.py
.venv/bin/python -m pytest src/tac/tests/test_plan_pr85_full_stack_opportunity_matrix.py -q
```

Result: `3 passed`.

Refreshed matrix:

- JSON:
  `experiments/results/pr85_full_stack_opportunity_matrix_20260504_codex_orchestrator_fix1/pr85_full_stack_opportunity_matrix.json`
- Markdown:
  `experiments/results/pr85_full_stack_opportunity_matrix_20260504_codex_orchestrator_fix1/pr85_full_stack_opportunity_matrix.md`
- Stable digest:
  `1ecffc9d88ac151d39a1a35428bcc0e2a4cf656ded3ae6a03218ce773f05983f`

Current machine-readable top stack plans:

1. `qh0_record_level_model_repack`
2. `qma9_native_run_grammar_or_table_reduction`
3. `pr86_hpac_pr85_mask_contract_port` (blocked on byte-exact HPAC parity)
4. `protected_post_motion_group_policy`
5. `pr89_final_bias_stack_on_pr85` (blocked unless exact component benefit
   exceeds charged bytes)

Decision:

No new randmulti group-deletion/waterfill dispatch should occur from stale
byte-screen priority.  Randmulti remains a valuable sidechannel surface only
for decoded-output-preserving recode or new component-response microatoms.

### 2026-05-04T03:54Z Worker Closure And Matrix Re-Grounding

The PR85/PR86 swarm returned four local-only implementation screens and all
were folded into the orchestration matrix as durable guardrails:

- PR86 HPAC replay shim:
  `src/tac/pr86_hpac_codec.py`,
  `experiments/replay_pr86_hpac_tokens.py`,
  `src/tac/tests/test_pr86_hpac_codec.py`.  The archive custody and HPAC state
  loader pass, but submitted `tokens.bin` decode still fails closed at frame
  `0`, group `10`, symbol `191`.  `dispatch_unlocked=false`.
- PR85 pair-atom builder:
  `experiments/build_pr85_pair_atom_candidates.py`.  The true PR85 scorer
  gradient ranks `pair_0192` first with break-even `516.6922114573065` bytes,
  but no legal pair-action spec or runtime contract exists yet.
  `dispatch_unlocked=false`.
- PR85 QH0/QM0 serializer:
  `experiments/build_pr85_qh0_serializer_candidates.py`.  Best result is
  source passthrough at `0` bytes; best non-source recode is byte-negative.
  `dispatch_unlocked=false`.
- PR85 correction-stream decoded-parity recoder:
  `experiments/build_pr85_correction_recode_candidates.py`.  All
  runtime-supported recodes for `post`, `shift`, `frac`, `frac2`, `frac3`,
  `bias`, `region`, and `randmulti` selected source bytes.  Best archive delta
  is `0`; no archive emitted.
- PR85 QMA9 runtime-supported native grammar screen:
  `experiments/build_pr85_qma9_native_grammar_candidates.py`.  The source QMA9
  segment has no bytes after the declared bitstream and no trailing zero bytes
  inside the declared stream.  No runtime-supported trim candidate exists.

Permanent matrix fixes:

- `experiments/plan_pr85_full_stack_opportunity_matrix.py` now discovers and
  consumes durable worker outputs for:
  - QH0 serializer screens;
  - PR85 pair-atom readiness;
  - decoded-parity correction recodes;
  - QMA9 runtime-supported grammar screens.
- The matrix also now blocks `protected_post_motion_group_policy` when the
  current best policy falls into the exact-negative post/motion deletion basin.
  This prevents re-dispatching `preserve_motion_only`/whole-post deletion after
  exact T4 evidence already showed PoseNet/SegNet regression.

Verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_build_pr85_qma9_native_grammar_candidates.py \
  src/tac/tests/test_plan_pr85_full_stack_opportunity_matrix.py \
  src/tac/tests/test_build_pr85_pair_atom_candidates.py \
  src/tac/tests/test_qh0_record_serializer.py \
  src/tac/tests/test_build_pr85_correction_recode_candidates.py \
  src/tac/tests/test_pr86_hpac_codec.py \
  src/tac/tests/test_diagnose_pr86_hpac_parity.py \
  src/tac/tests/test_plan_pr85_correction_atom_waterfill.py \
  src/tac/tests/test_plan_pr85_scorer_gradient_atoms.py -q
```

Result: `37 passed`, with only expected duplicate-ZIP warnings in fail-closed
archive tests.

Current matrix:

- JSON:
  `experiments/results/pr85_full_stack_opportunity_matrix_20260504_orchestrator_final2/pr85_full_stack_opportunity_matrix.json`
- Markdown:
  `experiments/results/pr85_full_stack_opportunity_matrix_20260504_orchestrator_final2/pr85_full_stack_opportunity_matrix.md`
- Stable digest:
  `3b86419321a7b04b3a16c1382fc0896b654bb30750c894a9bb873c8c10b90b86`

Current top stack priorities:

1. `qma9_native_run_grammar_or_table_reduction`: only unblocked high-upside
   family, but it requires a full-stream deterministic QMA9-compatible encoder
   or an alternate runtime grammar with explicit custody.
2. `pr86_hpac_pr85_mask_contract_port`: blocked on full PR86
   decode/reencode byte parity.
3. `scorer_gradient_pair_atom_policy`: blocked on explicit pair-action spec
   and pair-atom runtime contract.
4. `pr89_final_bias_stack_on_pr85`: blocked until exact component benefit
   exceeds the charged final-bias bytes.

Decision:

Do not spend T4 on stale PR85 wrapper recodes, whole sidechannel deletion, or
runtime-supported QMA9 trims.  The next aggressive implementation pressure is
true mask grammar/entropy replacement and PR86 HPAC parity, with pair-gradient
and final-bias work kept as secondary until they produce legal archive bytes.

### 2026-05-04T04:11Z QMA9 Alternate Grammar Guard Fold-In

Euclid's local QMA9 alternate-grammar worker completed a full-stream byte
screen over the PR85 decoded mask-token source.  It did not dispatch GPU work
and did not claim score.

Artifact:

- `experiments/results/pr85_qma9_alt_grammar_candidates_20260504/candidate_summary.json`

Key custody:

- Source PR85 archive SHA-256:
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- Source QMA9 segment bytes:
  `159011`
- Source decoded token SHA-256:
  `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`
- Best alternate mode:
  `adaptive9up2left2`
- Best alternate payload bytes:
  `161034`
- Delta vs source QMA9:
  `+2023` bytes

Decision:

The screened neighbor/table alternate grammars are now a measured local
negative and must not be redispatched as a GPU lane.  The remaining high-EV
QMA9 work is a structurally different run grammar or HPAC-style model with
decoded-token SHA parity, explicit charged runtime mode, deterministic archive
closure, and exact CUDA auth eval after a Level-2 lane claim.

Permanent matrix update:

- `experiments/plan_pr85_full_stack_opportunity_matrix.py` now discovers
  `pr85_qma9_alt_grammar_candidates_*` summaries.
- The matrix emits
  `qma9_alternate_neighbor_table_grammar_screen` as
  `empirical_alt_grammar_full_stream_no_byte_win`,
  `already_refuted=true`, and `blocked=true`.
- Refreshed matrix:
  `experiments/results/pr85_full_stack_opportunity_matrix_20260504_orchestrator_final3/pr85_full_stack_opportunity_matrix.json`
- Stable digest:
  `e7a285dab1b57c1dbd704e7cc1354000325cc7b3830826a825a06d0a4ccd97cb`

Verification:

```bash
.venv/bin/python -m py_compile \
  experiments/plan_pr85_full_stack_opportunity_matrix.py \
  src/tac/tests/test_plan_pr85_full_stack_opportunity_matrix.py
.venv/bin/python -m pytest \
  src/tac/tests/test_plan_pr85_full_stack_opportunity_matrix.py -q
```

Result: `3 passed`.
