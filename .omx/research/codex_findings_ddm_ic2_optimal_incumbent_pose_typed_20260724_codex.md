---
schema: codex_findings.ddm_ic2_optimal_incumbent_pose_typed.v1
date_utc: 2026-07-24
lane_id: lane_ddm_ic2_optimal_incumbent_20260724
axis: "[macOS-CPU frozen-scorer advisory]"
research_only: true
execution_allowed: false
score_claim: false
pointer_moved: false
main_review_required: true
---

# IC2 optimal-incumbent pose-typed composition: measured non-incumbent

## Outcome

The strict typed `W_seg -> PA1(frame_0)` packet is real, receiver-closed, parse-back
exact, and measured at batch 32 over all 600 pairs. It is **not** a new incumbent:

| Endpoint | d_seg | d_pose | archive bytes | advisory S |
|---|---:|---:|---:|---:|
| IC2 `W_seg -> PA1(frame_0)` | 0.024124510023328993 | 65.03498712932134 | 131,154 | 28.00173925293584 |
| incumbent v0 `W_joint -> PA1(frame_0)` | 0.07051923116048177 | 27.298487616378203 | 131,582 | 23.66179213623354 |
| IC2 minus v0 | -0.046394721137152775 | +37.73649951294313 | -428 | +4.339947116702302 |

The IC2 score terms are `2.412451002332899` Seg,
`25.501958185465156` Pose, and `0.08733006513778527` rate. PA1 reduces
the measured W_seg Pose endpoint from `146.36493245487773` to
`65.03498712932134` (delta `-81.3299453255564`) while preserving W_seg's
Seg result exactly, but that interaction still underdelivers the incumbent's Pose
term. No smoothing, telescoping, or component-delta summation is used.

Verdict:
`INSTANCE:W_SEG_PLUS_ZERO_BYTE_PA1_FRAME0_MEASURED_NON_INCUMBENT`.
This closes neither compact xi/Pose carriers nor DDM. The contest pointer remains
`0.1910828242 [contest-CPU]`.

## Why the requested stronger state could not be emitted honestly

- E2's `nested_pose6` is **ABSENT, not inert**: 3,600 bytes are consumed by the
  inter-pair worldsheet before export, no packet member owns them, and no compact
  code-to-photometry inverse exists. Treating those bytes as a composable stream
  would fake receiver ownership.
- #601 is a one-depth planar `dy/ds/dpsi` plus lossless-correction formulation.
  #605 is an n16, single-ground-depth, q4 Screw6 instance. They remain valid scoped
  controls, not an n600 additive carrier that can be attached to W_seg.
- The hood/gauge/blind ideas have no separate admitted, receiver-owned, negative
  marginal over this W_seg parent. Adding generic paint or copying a measured
  component delta would violate the scorer-recursive and non-telescoping laws.
- Substituting W_joint would reproduce the known weaker-state route and is
  expressly forbidden by the delegated contract.

The exact remaining blocker is therefore:
`COMPACT_CODE_TO_PHOTOMETRY_POSE_INVERSE_WITH_W_SEG_PARENT_ABSENT`.
The family remains open. Its unlock is a typed, counted xi/YUV6 carrier whose
decode maps code to both frames, survives R/uint8/parse-back, and is exact-replayed
on the W_seg parent before admission.

## Runtime and custody

- Source W_seg: 138,031 bytes,
  `264a09abb8f614eca104eb4ab1d0a12005ba65ec6a4fbc6620ff92f1c73281a9`.
- IC2 archive: 131,154 bytes,
  `be1989fe16a2983b291c85c5e58f3e2db74edd3123a5ed19add11c7e9800f97e`.
- Inflated raw: 3,662,409,600 bytes,
  `56e2b32a25805d06487a5309ab7bf6435a9ad6a596deda311c28d8caf4b797d7`.
- Export receipt:
  `97b88cef99cda2232253a5a0bb14f31992f098e3c5b70bf7fb76c30a7a7eb6d3`.
- Batch-32 receipt:
  `9638f3ad4920afb3d0cf563c8ced2d61ee02157785bd525ee001acbde600f10d`.
- Runtime bundle declares `numpy`, `torch`, `Brotli==1.2.0`, and
  `opencv-python-headless==4.11.0.86`. SciPy is encode-only and lazy; Pydantic
  is replaced by the bundled runtime compatibility surface when absent.
- A clean venv started without Brotli, cv2, SciPy, or Pydantic. `inflate.sh`
  self-bootstrapped its declared packages, decoded byte-identically with 38
  base plus 38 composed stage checkpoints preserved, and took 445.03 seconds.
  SciPy and Pydantic remained absent. Three immutable harness failure receipts
  retain the successive scorer-environment dependency misses; they are
  infrastructure failures, not scientific negatives.
- After completing the scorer-only environment, the frozen upstream harness
  passed in `1014.041449` seconds with `d_seg=0.02412450`,
  `d_pose=65.03498077`, and rounded score `28.00`; receipt SHA
  `c71582f287ca6337c2d097850f642e39c9841bb22513802b4a9465873a45044e`.
- The staged Modal wrapper runtime tree is
  `019b03f4790c43cf2a8a9e1676bea26798e760a60075651b404295de46703a3a`;
  the separately recomputed inner tree is
  `701bb0e0beaafeea13f2d3eff0c563224b884413973029e398ae4f7a6bdc5829`.
  Their content-tree SHA is identical
  (`3520e864bacbe9aa39332be42014d38684afb692ee69ae5bf464d1adb88ecbd4`).
  The prior `1976cdb2...` inner prefix was not reused because it did not match
  this recomputed runtime.

All bulky outputs and stage checkpoints live under
`/Volumes/VertigoDataTier/pact/evidence/ddm_ic2_optimal_incumbent_pose_typed_20260724`.
No source or proof bytes were deleted.

## Triality

- DSL: `DDMIC2RuntimeExporterConfigV1`,
  `DDMIC2PacketRemeasureConfigV1`, and the clean upstream harness schema bind
  W_seg, PA1, batch geometry, source/archive SHAs, runtime dependencies, and
  false-authority fields without widening legacy IC1/E4 schemas.
- DAG: W_seg exact parent -> receiver-owned WS1 streams -> PA1 frame-0 affine
  -> E4 Brotli-Q11 packet -> clean decode/parse-back -> n600 batch-32 Seg/Pose
  -> incumbent comparator -> nonpromotion. See
  `ddm_ic2_optimal_incumbent_pose_typed_DAG_FEED_20260724.md`.
- Equations: the score, moment-derived PA1 transform, and non-telescoping
  admission law are frozen in
  `ddm_ic2_optimal_incumbent_pose_typed_canonical_equations_20260724.md`.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`;
the v7.5/v8 specs; W_seg producer and E5/E4 export receipts; E2 pose-stream
receipt; #601/#605 scope audit; PA1/E3 and MC1 receipts; DM4 findings;
IC1 scoreboard and Modal stage bundle; lane/subagent/cost-band/continual-learning
state; both live directive inboxes.

## MAIN landing review

MAIN should review the strict legacy-schema separation, runtime import closure,
clean dependency bootstrap, exact batch-32 custody, scoreboard append, and the
scoped claim that the missing compact pose inverse blocks promotion. The staged
contest-CPU bundle is deliberately `PREPARED_NOT_DISPATCHED` and should not be
fired for a known non-incumbent without a separate calibration decision.
