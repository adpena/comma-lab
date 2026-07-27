---
title: DDM MR2 pricing-wave independent review and serial-merge disposition
date_utc: 2026-07-26T14:46:00Z
reviewer: mr2-independent-approver
delegation_checkpoint_key: codex_delegate:ddm_mr2_pricing_wave_merge:20260726T142758Z
research_only: true
score_claim: false
pointer_moved: false
main_landing_review_required: true
verdict: CONTENT_CLEAN_SERIAL_MERGE_BLOCKED_AT_PF3B_NONTRIVIAL_CONFLICT
---

# Outcome

Fresh adversarial review found the three incoming entities substantively
clean at their exact commit bytes:

- PF3B `074955c6ad0900268b01b7be6e359677e254b0d0`
- WF7 `bdc3f7261bad9d5d7c2fb755b79469f29560d399` plus normalization tip
  `e3c2140d3a9f096082efa7dfb938fd6d7f9b31db`
- CB1 `2721704ab215806d34788b29bae227554a6b9b50`

The mandated serial merge did not land. The first three-way merge, PF3B into
MR2 base `5a55aa5914dc5675d2dd0fbe8bc225c77e2d9163`, produced nontrivial conflicts
in the shared lane registry and an independently added PF3 materializer. It
was aborted unchanged, as delegated. WF7 and CB1 were not merged out of order.

The competitive frontier is the official leaderboard displayed `0.172`.
`0.1910828242` remains only a custody-local contest-CPU baseline. No launch,
exact evaluation, fire, reseal, paid dispatch, score claim, promotion, or
pointer mutation occurred.

# Per-entity independent verdicts

## PF3B — CONTENT CLEAN; LANDING BLOCKED

Commit `074955c6ad0900268b01b7be6e359677e254b0d0` correctly:

1. ranks correction-first, then strict joint distortion, spill, support,
   AT1 trace, event count, and stable identities;
2. stops at the first measured strict `delta_D_joint < 0` row and measures
   the complete same-coordinate two-magnitude/two-sign neighborhood;
3. retains the predecessor row and measures the other three neighborhood
   members, yielding 19 measured of 68 and leaving 49 unmeasured;
4. invalidates v1 before v2 authority because v1 incorrectly subtracted
   PF2/MS6 events from the composite support; and
5. labels the negative at instance scope without a score or promotion claim.

The independently rederived winning row is:

- `delta_D_joint = -0.00010799244434957068`
- `delta_archive_bytes = +860`
- `delta_S_rate = 25*860/37_545_489 = 0.0005726386996850674`
- `delta_S_total = +0.0004646462553354967`
- exact break-even `162.185...` bytes, hence at most `162` integer bytes.

The four-point neighborhood is `mag1/-`, `mag1/+`, `mag2/-`, `mag2/+`;
only `mag1/-` is distortion-downhill, and all four are total-score uphill.

Pinned Python:

- `tools/hunt_ddm_pf3b_joint_improving_edge.py`: SHA-256
  `71cfd1570c44f84313e334a0fa1254cc71080b0311ec79a1f0a9a200fa4ac2f5`
- `tools/materialize_ddm_pf3_finite_prices.py`: SHA-256
  `bdf3b82084098bae649d290994903cbbfb118ff15a16347e847acf506c54ac0d`
- `tools/tests/test_hunt_ddm_pf3b_joint_improving_edge.py`: SHA-256
  `ece92eae72518176dbffe04becc5f54ffdb9b8453ef96c227a99362cf59fe475`

The incoming Python blobs compile. The branch-wide diff check is not clean:
two Markdown lines in the PF3B findings use trailing spaces. This is a
mechanical landing cure, not a scientific finding.

## WF7 — CONTENT CLEAN; UNMERGED BEHIND PF3B

Tip `e3c2140d3a9f096082efa7dfb938fd6d7f9b31db` correctly treats the EV2
`+270` lane seed as a logical delta distributed across physical ZIP homes:

- manifest `+34`
- lane member `+155`
- central directory and EOCD `+81`
- total `+270`.

The DWF7 parser is fail-closed on magic/version, codec IDs, reserved bits,
canonical ULEB128 lengths, overruns, trailing data, exact seven-home
reconstruction, and sealed state identity. The directory is independently
rederived as eight fixed bytes plus ULEB128 lengths `2+3+3+1+2+2 = 13`,
therefore exactly `21` bytes. The selected payload saves `1,797` bytes;
including the directory, the candidate is `132,435` bytes, a `-1,776`-byte
lossless delta.

CC3 was independently reaggregated from its 27 physical leaves:
eight selected leaves total `-3,422` bytes, of which three nested v15 leaves
total `-2,302` and exterior wrappers total `-1,120`. WF7 correctly treats CC3
as an alternative same-pool object and never adds that credit to the seven
home row.

The state diagnostic explicitly says
`NOT_AN_E4_OR_CONTEST_PACKET_TRIPLE`; it cannot inherit runtime or contest
authority from exact state restoration.

Pinned Python:

- `src/tac/optimization/ddm_wf7_seven_home_stream_waterfill.py`: SHA-256
  `364c80dcc11eaa2a6dadad82f76312a38dd50914e16157a995694fce55b6c463`
- `src/tac/optimization/tests/test_ddm_wf7_seven_home_stream_waterfill.py`:
  SHA-256
  `b4ce6040df0ca71c7f52d9254a66bdcddcc12e8a7e14e5af894ef2799f2e7c17`
- `tools/run_ddm_wf7_seven_home_stream_waterfill.py`: SHA-256
  `e08c31ba342a9db749ed777e17980d29ac151024ffdfbe0b8fd351e2415a4e09`

All incoming Python blobs compile and the full WF7 branch diff check is clean.

## CB1 — CONTENT CLEAN; UNMERGED BEHIND PF3B

Commit `2721704ab215806d34788b29bae227554a6b9b50` was checked from the
preserved primary receipts, not merely its summary:

- 114 JSON batch receipts and 114 matching NPZ checkpoints exist;
- each candidate has 38 contiguous batches covering pairs `0..599`;
- each candidate has exactly `117,964,800` Seg sites and `3,600` Pose
  coordinates;
- every per-class before/corrected/introduced/after identity conserves;
- per-class sites and errors sum exactly to each global measurement; and
- fresh hashes of all three `3,662,409,600`-byte raw outputs match custody.

The MC1 target is not hardcoded. The tool requires exactly one evidence row
for every canonical class ID, computes MC1's
`bottom_share * static_iou` law, rejects a tie, and verifies the rederived ID
against the upstream detection. The observed unique winner is canonical class
ID `4`, but the counted static rule stores that detected target explicitly as
a wildcard transition and the receiver paints the payload target.

The three committed packet ZIPs match their byte/hash custody and contain
only `manifest.json` and `state/rg4.ddr4`. The emitted runtime consumes exactly
those members, verifies the framed source, canonical source-local re-emission,
parent re-emission, source/packet hashes, staged output shape, and final raw
identity.

The objective was independently recomputed from absolute control/candidate
terms:

- MyCar: Seg `-0.001052008734808707`, Pose `-0.05080601512212013`,
  rate `+0.00021240900604597266`, total
  `-0.051645614850883974`; `+319` bytes; **ADMIT**.
- Lane: total `+9.21156940553832`; `+1,530` bytes; **REJECT / INSTANCE**.

Only MyCar has `waterfill_eligible=true`. Lane remains a typed rejected row;
its failure does not close the Lane carrier family.

Pinned primary code SHAs are recorded in the companion conflict receipt. All
incoming Python blobs compile and the full CB1 branch diff check is clean.

# Merge conflict and exact cure

`git merge --no-commit --no-ff 074955c6ad` produced:

- `.omx/state/lane_registry.json`: content conflict. The registered merge
  driver attempted `.venv/bin/python tools/merge_lane_registry.py`, but this
  isolated worktree has no `.venv`.
- `tools/materialize_ddm_pf3_finite_prices.py`: add/add conflict. Current
  stage-2 blob `3d79b278b39427243e94a9b78c6c8134f20a920b`; incoming stage-3 blob
  `07e7f7fdde5c532890cc1684dcbb538d0525d1d4`. PF3B adds configurable
  `run_id`, `lane_id`, `checkpoint_schema`, and `candidate_prefix` behavior
  required by its hunter; resolving this by whole-file selection would risk
  clobbering the current independently added materializer.

MAIN's cure is a quiet-boundary three-way remerge on current main, with a
working registry merge-driver interpreter and a line-level materializer
resolution that preserves both current-base behavior and PF3B's generalized
checkpoint/identity parameters. Remove the two PF3B Markdown trailing spaces,
run touched-surface tests and `git diff --check`, obtain a fresh greenup
credential for the resolved hashes, then continue serially WF7 and CB1. No
whole-file ancestor checkout is admissible.

# Review credential record

Reviewer identity: `mr2-independent-approver`.

The source-commit review result is `CLEAN` for PF3B, WF7, and CB1 at the
commit and file hashes above. No `review_tracker.py` greenup credential was
issued, fail-closed: none of the incoming versions entered this working tree
after the mandatory merge abort, and marking current-base entities would
credential different bytes. MAIN must review the actual conflict resolution
and ingest a greenup pass against the final merged versions. No
`REVIEW_GATE_OVERRIDE` was used.

# STORES CONSULTED

`PROGRAM.md`; byte-identical `CLAUDE.md` and `AGENTS.md`; the complete
NO-FAKE section; `docs/operating_manual_craft_handoff.md`; delegated
authority SHA-256
`fd21e7dca8e7097ec18815b9820c75f938e537d2ce3204b510dba72fd5172e02`;
both watched inboxes through operator directive
`2026-07-25T19:52:29Z`; routing card section 8; all three pinned branch
diffs, configs, receipts, findings, DAG feeds, tests, and integrity manifests;
PF3B v1 invalidation and four-point rows; WF7 C1/EV2/CC2/CC3/E4 custody;
CB1 MC1 support/receipt, RG4 source-local receipt, exact packet ZIPs, emitted
runtime, 114 batch receipts, 114 NPZ checkpoints, and three preserved raw
outputs.

# DAG FEED

Three useful facts remain queued behind one merge-boundary blocker:

1. PF3B found a real distortion-downhill coordinate but its `+860` bytes make
   the exact row score-uphill; the break-even is at most `162` bytes.
2. WF7 provides a lossless `-1,776`-byte state-container recode, but it is
   rate-only and not an E4/contest packet.
3. CB1 MyCar provides the only strict negative measured joint row,
   `delta_S=-0.051645614850884` advisory at `+319` bytes; Lane stays rejected.

These rows are non-additive until c1 remeasures their composition on one
merged byte-closed base. The c1 co-measure remains a separate post-merge fire.
Pointer unchanged.

# MAIN landing review required

MAIN must review this MR2 commit and the eventual conflict resolution, verify
the final resolved file hashes, issue the greenup credential on those bytes,
preserve serial order, and then decide whether to merge the resulting branch
to main. This memo is not landing authority by itself.
