---
schema: sh1_findings.ddm_sh1_compose_and_local_exact.v1
date_utc: 2026-07-27
arm: ddm_sh1_compose_and_local_exact (claude worktree arm, 1:1 codex replacement)
branch: ddm/sh1_integration_20260727
base_main: ad464e269c
axis: "[macOS-CPU advisory — real evaluator, real bytes]"
research_only: true
score_claim: false
pointer_moved: false
paid_dispatch: false
competitive_bar: "min(0.15, official leaderboard best 0.172)"
main_landing_review_required: true
verdict: ROW_LANDED_S_23.91_ADVISORY_DECOMPOSITION_IS_THE_DELIVERABLE
---

# DDM sh1 — compose incumbent_v2 + FULL local exact protocol

## 1. Integration branch (charter step 1) — DONE

Branch `ddm/sh1_integration_20260727` from main `ad464e269c`, three pricing-wave
branches merged serially per the mr2 review package
(`.omx/research/codex_findings_ddm_mr2_pricing_wave_merge_20260726_codex.md`, e71e99dab1):

1. **PF3B** `074955c6ad` (commit 0bb5b9579c): two conflicts resolved exactly as mr2
   prescribed — `.omx/state/lane_registry.json` -> main's version (the
   `ddm_pf3_finite_price_materialization` row was byte-identical on both sides; the
   only branch-only row is carried in the patch file below); the
   `tools/materialize_ddm_pf3_finite_prices.py` add/add resolved LINE-LEVEL to PF3B's
   stage-3 blob after verifying it is a strict keyword-defaulted generalization
   (`run_id`/`lane_id`/`checkpoint_schema`/`candidate_prefix` params whose defaults equal
   the module constants — base behavior preserved; compiles). The two trailing-space
   markdown lines in the PF3B findings were stripped; `git diff --check` clean.
2. **WF7** `e3c2140d3a` (commit 8dd643c03a): clean merge; all three pinned Python SHAs
   re-verified identical to mr2's pins (364c80dc…, b4ce6040…, e08c31ba…).
3. **CB1** `2721704ab2` (commit 3963938cc8): clean merge; packet ZIPs + emitted runtime
   + receipts landed; python compiles.

Registry patch file (MAIN applies at a quiet boundary; this arm never wrote main's
live registry): `.omx/research/ddm_sh1_registry_rows_owed_20260727.json` — one owed
row (`ddm_pf3b_52probe_joint_improving_hunt`); WF7/CB1 merges touched no registry rows.

## 2. Composed incumbent_v2 (charter step 2) — identity + double-count guard

RECALLED receipts (not guessed):
- ws2 W_seg: d_seg 0.024124510 @ 138,031 B, sha 264a09ab… — measured Pose endpoint
  146.36 raw / 65.03 with PA1 -> advisory S 28.00 (IC2) — NON-incumbent.
- ic1 incumbent v0 = W_joint->PA1(frame_0): 131,582 B, d_seg 0.07051923,
  d_pose 27.29849, advisory S 23.66179 — best measured receiver-closed joint state,
  but its W_joint state format does not accept the CB1 carrier (which is byte-closed
  to the merged RG4 source-local PC1 state, `state/rg4.ddr4`).
- CB1 fresh byte-closed control (RG4 base, no carrier): 131,301 B,
  d_seg 0.061912604, d_pose 31.281041, advisory S 23.96504.
- CB1 MyCar static-mask carrier ON that control (the only ADMITTED strict-negative
  joint row): 131,620 B, d_seg 0.061902084, d_pose 31.101584, delta joint S
  -0.0516456 measured control-vs-candidate on the SAME base (non-additive-pools
  honored: this is a joint n600 remeasure through the composite receiver /
  R / uint8 / frozen scorers — not a summed delta).

**incumbent_v2 (sh1) := the CB1 `mycar_static_mask` packet** — merged RG4
source-local PC1 base + admitted MyCar hood carrier (+319 B), byte-closed through the
CB1/E4 framed exporter (deterministic Brotli-Q11, ImportError-only LZMA1 fallback —
the e4 leg of the charter), with the emitted self-contained runtime.

- archive: 131,620 bytes, SHA-256
  `5e1441180f83a6d1d12dc72b574d6801f815c555ede3ca2f56ccb228bc45c3b3`
  (re-hashed in this arm from the merged tree; matches the CB1 receipt byte custody).
- **Double-count guard (measured, not assumed):** the ic1 piece manifest records MC1
  hood static reassert as `EXCLUDED_MEASURED_CONFLICT` from incumbent v0's lineage,
  and the CB1 control (RG4 base WITHOUT carrier) vs candidate rows measure the carrier
  marginal directly on this base — the base does not already contain the hood
  reassert; the +319 B carrier is the only hood assert in the packet. Not redundant.
- Why not W_joint->PA1 + MyCar: no admitted typed compose route exists between the
  MyCar carrier (RG4 state rewrap) and the W_joint state; composing them would
  require a new byte-close + full n600 joint remeasure = a successor unit, recorded
  in section 6, not faked here.

## 3. Typed blockers (honest, per the charter's "typed blocker = full success")

- **WF7 seven-home repack leg: BLOCKED — `WF7_LEG_NON_E4_STATE_CONTAINER`.**
  WF7's -1,776 B lossless recode is measured on the exact 134,211-byte seeded C1
  state container (its own verdict scope: "INSTANCE x exact 134211-byte seeded C1
  state x settled five-coder menu"), and its own findings state the WF7 diagnostic
  "is not presented as an E4 or contest packet triple" and that the
  "state-container-to-E4 runtime binding remains explicit" (i.e. missing). The CB1/E4
  packet's receiver consumes `manifest.json` + `state/rg4.ddr4` and has no DWF7
  parser; wiring DWF7 into the packet runtime is a re-byte-close engineering unit
  with its own parse-back proof — composing the -1,776 B credit here without that
  would be a fake (summing pool deltas / unconsumed bytes). Exact unlock: bind the
  DWF7 seven-home container as the serialization of `state/rg4.ddr4` inside the E4
  exporter + emitted receiver, re-prove double-compile identity + raw identity, then
  re-measure.

## 4. THE ROW (charter step 4) — FULL 600-sample upstream/evaluate.py

Protocol: exact archive bytes staged at
`/Volumes/VertigoDataTier/pact/experiments/results/ddm_sh1_local_exact_20260727T2210Z/submission/archive.zip`
(SHA re-verified 5e144118…); the packet's own emitted `inflate.sh`/`inflate.py`
decode the EXTRACTED members to `submission/inflated/0.raw` (raw identity checked
against the CB1 receipt hash a6cee040…); then
`upstream/evaluate.py --device cpu --batch-size 16` over all 600 samples with the
frozen upstream scorers, GT `upstream/videos/0.mkv`, rate =
archive.zip bytes / 37,545,489. MPS explicitly refused (`--device cpu`).

RESULTS — `[macOS-CPU advisory — real evaluator, real bytes]`, NOT a contest-CPU
score claim (Apple Silicon is never a 1:1 axis):

- Inflate: decoded in 19 resumable stages via the packet's own emitted runtime
  (first attempt harness-killed at rc=144; relaunched via
  `tools/launch_detached_process.py` and RESUMED from stage checkpoints — the
  resumability non-negotiable paid for itself); final raw 3,662,409,600 bytes,
  SHA-256 `a6cee0402433f079107e890a4570541de8ff5171f9ac8b1ae1a716e2d02c4302` =
  BIT-IDENTICAL to the CB1 receipt (deterministic decode reproduced on this host).
- Evaluator: 600 samples, 38 batches, 12m55s wall, seed 1234.

| term | value |
|---|---:|
| d_seg (600-sample average) | 0.06190208 |
| d_pose (600-sample average) | 31.10157967 |
| archive bytes | 131,620 |
| rate | 0.00350561 |
| seg term (100·d_seg) | 6.190208 |
| pose term (sqrt(10·d_pose)) | 17.635640 |
| rate term (25·rate) | 0.087640 |
| **S (recomputed from components)** | **23.913488** |
| report.txt rounded | 23.91 |
| **distance to the 0.172 bar** | **+23.741488** |

Cross-check against CB1's frozen-scorer MENU1 path (independent measurement
chain, same bytes): d_seg identical to 8 printed decimals (0.06190208...);
d_pose 31.10157967 vs 31.10158359 (delta -3.9e-6, reduction-order precision).
The two independent measurement paths agree — the CB1 chain is validated
end-to-end through the actual upstream harness.

**Per-component decomposition names exactly which term is over (charter step 5,
S >> 0.18 branch — no Modal staging packet):**

1. **Pose term is 74% of S** (17.636 of 23.913; d_pose 31.10 vs a sub-0.01 need).
   The original DDM line's binding gap is the pose/photometric axis — consistent
   with IC2's `COMPACT_CODE_TO_PHOTOMETRY_POSE_INVERSE_WITH_W_SEG_PARENT_ABSENT`
   blocker and the #366 joint-descent line.
2. **Seg term is 26% of S** (6.190; d_seg 0.0619 vs the ~0.0008-0.001 need —
   ~70x over). The describe-line box (finite-price materialization) remains the
   crux, per routing cards section 5/section 8.
3. **Rate term is NOISE at this operating point** (0.0876; even WF7's blocked
   -1,776 B would move S by only -0.00118). Rate polish is economically
   irrelevant until distortion falls by orders of magnitude — the measured
   confirmation of the borrowed-incumbent-rate-polish-dead doctrine, now on OUR
   original line's own composed candidate.

## 5. SHA-pinned receipt

`.omx/research/ddm_sh1_local_exact_receipt_20260727.json`

## 6. Successor first-rungs (findings name their next measurement)

1. **DWF7-into-E4 binding** (unblocks the -1,776 B rate leg on the composed packet):
   implement `state/rg4.ddr4` serialization via the WF7 seven-home container in the
   E4 exporter + receiver; re-byte-close; expected rate delta approximately
   -1,776 B ~= -0.00118 S — only meaningful after the distortion terms are addressed.
2. **MyCar carrier on the W_joint->PA1 parent** (the better base, S 23.66): needs a
   typed compose route W_joint x hood-static rule + full joint n600 remeasure.
3. **The real gap is NOT rate:** see decomposition — the pose term and seg term are
   orders of magnitude over the bar; the #366 descent line / describe-line box
   remain the binding cruxes, exactly as the council routing cards say.

## STORES CONSULTED

Charter `.omx/tmp/codex_prompts/ddm_sh1_compose_and_local_exact.md`; CLAUDE.md
(NO-FAKE, THE GOAL, storage/provenance, MPS-never); mr2 review package e71e99dab1
(findings + conflict receipt); PF3B/WF7/CB1 branch findings + receipts at their exact
merged bytes; ws2 custody producer receipt; ic1 incumbent scoreboard + piece
manifest; ic2 optimal-incumbent findings (W_seg->PA1 non-incumbent + clean upstream
harness protocol); upstream/evaluate.py + frame_utils.py (read for the exact
invocation contract); council routing card §9 re-anchor (bar = min(0.15, 0.172);
0.19108 never used as a target).

## DAG FEED

- sh1 closed the ship-loop: the campaign's first END-TO-END composed original-line
  candidate (merged RG4 base + admitted MyCar carrier, E4 byte-closed, 131,620 B,
  sha 5e144118…) now has a REAL `upstream/evaluate.py` 600-sample row on exact
  archive bytes: **S = 23.913488 `[macOS-CPU advisory — real evaluator, real
  bytes]`** = 6.190 seg + 17.636 pose + 0.088 rate; +23.741 above the 0.172 bar.
  Raw decode bit-identical to the CB1 receipt; evaluator d_seg/d_pose agree with
  the independent CB1 frozen-scorer chain to <=4e-6 — the measurement apparatus
  is cross-validated, the candidate is not competitive: the gap is 74% pose,
  26% seg, 0.4% rate.
- The three pricing-wave branches are integrated on one branch with mr2's exact
  conflict cures; the WF7 rate leg is typed-blocked on the DWF7-to-E4 binding, not
  silently summed.
- Pointer UNMOVED; no score claim; no paid dispatch; Modal staging packet NOT
  assembled (S far above the 0.18 staging gate — decomposition is the deliverable).
