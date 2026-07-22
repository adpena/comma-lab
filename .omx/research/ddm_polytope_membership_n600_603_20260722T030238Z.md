---
title: DDM polytope-membership curve and n600 same-artifact closure
utc: 2026-07-22T03:02:38Z
task: 603
master_task: 578
lane_id: lane_ddm_polytope_membership_n600_603_20260722
verdict: MEMBERSHIP_MEASURED_AND_N600_SAME_ARTIFACT_CLOSURE_GREEN
verdict_scope: local frozen-SegNet batch16 argmax-cell membership at n64/n256 plus scorer-free n600 same-artifact archive closure
research_only: true
execution_allowed: false
main_landing_review_required: true
---

# Outcome

The DDM objective is now measured as membership in the exact C1 solved member's frozen-SegNet
argmax cell, rather than smooth RGB reconstruction. The bounded curve is:

| Pairs | Exact archive bytes | Membership fraction | Pose-code completeness | Evidence |
|---:|---:|---:|---:|---|
| 64 | 274,664 | 0.493605613708 | 1.000000000000 | local frozen-SegNet batch16 advisory |
| 256 | 1,095,272 | 0.494461019834 | 1.000000000000 | local frozen-SegNet batch16 advisory |
| 600 | 2,565,528 | not run by design | 1.000000000000 | scorer-free same-artifact closure |

The n600 result is genuine describe/decode closure for one archive, not a size projection. It
describes and decodes all 600 pairs with one described chunk and one source chunk resident, compiles
deterministically twice, decodes deterministically twice, parse/re-encodes identically, assigns all
2,565,528 ZIP bytes exactly one home, and reproduces identical terminal archive and history after a
disk resume from stage 1. The archive is explicitly `not_a_candidate`; no scorer weights ship.

# Formulation verdict

The old RGB-channel diagnostic does **not** collapse into true argmax-cell membership. At n256 the
tie-first RGB-channel disagreement is 0.229040582975, while frozen-SegNet cell escape is
0.505538980166, a 0.276498397191 absolute under-report. The measured membership is also sharply
class-skewed: Undrivable membership is 0.999996584559, while Road and Lane are 0 and MyCar is
0.000011158723. Therefore the scoped membership rung is green as a measurement surface, but the
current chart grammar is not an effective all-class member solver. This is a FORMULATION result,
not a family-level negative and not SegNet score authority.

Freshly evaluated C1 target cells match retained `gt_n600.lstars` on 0.999844054381 of n256 sites.
The membership numerator uses target and described logits evaluated under the same frozen scorer,
batch size, device, seed, and arithmetic; the small retained-cache crosscheck gap is reported rather
than silently coerced.

# Exactness to membership, n256

| Stratum | RGB-pixel exact | Same C1 cell | Membership minus exactness |
|---|---:|---:|---:|
| overall | 0.001795709133 | 0.494461019834 | +0.492665310701 |
| boundary codim-1 | 0.000174505419 | 0.129902776900 | +0.129728271481 |
| cell interior | 0.001830591387 | 0.502304952747 | +0.500474361360 |
| margin [0,0.1) | 0.000144919135 | 0.155809808127 | +0.155664888992 |
| margin [0.1,0.5) | 0.000164794673 | 0.160311887806 | +0.160147093133 |
| margin [0.5,1) | 0.000221656640 | 0.166960898826 | +0.166739242186 |
| margin [1,inf) | 0.001838753154 | 0.503346185558 | +0.501507432403 |

`Pose-code completeness = 1` means every pair's six counted ordinal Pose codes are present. It is
not a PoseNet distortion or pose-tube claim. Because SegNet has spatial receptive fields, local
pixel exactness is not logically sufficient for same-cell membership; the receipt preserves the
separate exact-and-member, inexact-but-member, and escape counts.

# Custody and bounded re-derivation

- Receipt SHA-256: `3b6a7d41c7c6e63dd4ac02c252b52d96de7355d7c9c4b5508300d4f2939920b8`.
- n600 archive SHA-256: `63ce33d7cf78d77477fe3eb68ea2f6d7dd202a3a3036b5c76e1b58ed1ec2f65d`.
- Frozen weights SHA-256: `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`;
  38,502,892 local bytes, never copied into the archive.
- Source cache is read-only; the retained 5,078,017,610-byte SHA-bound cache is reused without
  materialization, mutation, deletion, or movement.
- Measured wall time in this worktree was 202.2 seconds, below the delegated ten-minute bound.

```text
/usr/bin/env python3 tools/run_direct_description_polytope_membership.py --config .omx/research/ddm_polytope_membership_n600_603_20260722T023416Z.config.json --output-dir .omx/research/ddm_polytope_membership_n600_603_20260722T023416Z_artifacts --execution-allowed false
```

Outputs are immutable by design; MAIN must use a fresh reviewed output directory for re-derivation.

# Blocker and pointer delta

- `N600_SAME_ARTIFACT_ARCHIVE_CLOSURE`: `RED -> GREEN_MEASURED_APPARATUS_SCOPE`.
- Existing 19-row register: `7/19 -> 8/19` green; 11 remain red.
- `POLYTOPE_MEMBERSHIP_RUNG`: new supplemental green row at local frozen-scorer measurement scope;
  it is not stretched into an existing PRIMARY blocker.
- Pointer remains `0.1910828242 [contest-CPU]`; no contest score or candidate promotion occurred.

The remaining 11 blockers are PRIMARY execution authority, live V8/V9 owner receipts, canonical
resume-registry integration, canonical PRIMARY typed-compiler integration, governed launcher/memory
adapter, heavy-run cleanup/cold-store integration, contest-CPU replay, contest-CUDA replay, healthy
completion certificate, externally attested failure token, and separate SHA-bound operator GO.

# STORES CONSULTED

Delegated authority, `CLAUDE.md`, `AGENTS.md`, project memory top, `PROGRAM.md`, operating manual,
v7.5/v8 contracts, DDM specs/DAG/equations, #547/#549/#580/#602 and rungs-1-to-3 artifacts, current
lane/task/pointer state, exact C1 target receipt and SSD chunks, retained `gt_n600` cells/margins,
frozen upstream scorer source/weights, and both inboxes.

Implementation commit: `67e35b897a`. MAIN landing review is required.
