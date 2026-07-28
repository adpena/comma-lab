---
schema: ddm_cs1_consolidation_harvest.v1
date_utc: 2026-07-28
arm: ddm_cs1 (consolidation-harvest arm; read-mostly from main checkout)
axis: "[macOS-CPU advisory — no new measurement; harvest/routing only]"
research_only: true
score_claim: false
pointer_moved: false
paid_dispatch: false
competitive_bar: "min(0.15, official leaderboard best 0.172)"
pointer_state: "0.19108 UNMOVED (custody-specific local anchor); competitive target 0.172"
main_landing_review_required: true
verdict: SIGNAL_HARVESTED_NO_MERGE_TO_MAIN_NO_STRAND_OWED
---

# DDM cs1 — consolidation harvest of unmerged worktrees + branches

Operator 2026-07-28: "There is likely useful signal on the unmerged worktrees and
branches." The consolidation monitor read `CONSOLIDATE-NOW` (pile_files 245, pile_lines
3167, stale_commits 200). This arm worked READ-MOSTLY from the main checkout, made NO
edits to main's working tree (a parallel session owns a ~244-file WIP there — clobbering
it is the cardinal sin), and merged NOTHING to main. It enumerated all unmerged branches
+ dirty worktrees into one disposition table, harvested the sh1 integration signal, and
verified there is no unambiguous uncommitted strand owed on any inspectable branch.

## 1. Disposition table — all unmerged branches (`git branch --no-merged main`)

Main HEAD at harvest: `9a4e4e2b46`. Six branches are not reachable from main:

| branch / tip | disposition | rationale |
|---|---|---|
| `codexwt/ddm_cb1_perclass_carrier_byteclose_…` `2721704ab2` | **harvest-signal-only; contained-in-sh1** | tip is an ancestor of sh1 `6a77427ca1` (verified `merge-base --is-ancestor`); its bytes/receipts are the incumbent_v2 base. Worktree CLEAN but branch UNMERGED-to-main → not pruned. |
| `codexwt/ddm_pf3b_52probe_joint_improving_hunt_…` `074955c6ad` | **harvest-signal-only; contained-in-sh1** | ancestor of sh1; PF3B pricing row. Worktree CLEAN, UNMERGED → not pruned. |
| `codexwt/ddm_wf7_seven_home_stream_waterfill_…` `e3c2140d3a` | **harvest-signal-only; contained-in-sh1** | ancestor of sh1; WF7 rate leg. Worktree CLEAN, UNMERGED → not pruned. |
| `ddm/sh1_integration_20260727` `6a77427ca1` (worktree `agent-ae029…`) | **merge-blocked-on-`direct_description_carrier_compose.py`** | sh1's 10-commit diff overlaps `src/tac/optimization/direct_description_carrier_compose.py`, which is DIRTY in main's WIP (parallel session). DO NOT merge, DO NOT touch that file. Signal harvested read-only into this note (§3–§4). Worktree CLEAN but branch UNMERGED → not pruned (harvest source). |
| `worktree-agent-a377bd0a960d56789` `e8e0566e81` | **live-arm-do-not-touch (fd1)** | ddm_fd1 family-d GN/CG launcher wire-in; sub-45-min file activity confirmed. Named live arm. |
| `worktree-agent-ac3ed6708b4934989` `41f2affaca` | **live-arm-do-not-touch (ee1, locked)** | ddm_ee1 fresh-eyes optimal-codec theory; sub-45-min activity; worktree `locked`. Named live arm. |

## 2. Disposition table — dirty / clean worktrees (merged branches)

The two 245-dirty worktrees named in the cs1 charter (`agent-a278d…`, `agent-a703d…`)
NO LONGER EXIST in the worktree list — reconciled: **main's own working tree is the
245-dirty tree** (the parallel session's WIP). Not touched.

| worktree | dirty | branch merged? | disposition |
|---|---:|---|---|
| `agent-a4ba21155753780e9` | 5 (pdw1_*) | yes (`377df45980`, 07-19) | **superseded-by-main** — all 5 dirty files already tracked on main HEAD; 4 byte-IDENTICAL, `pdw1_fp32_realization_first_inbox_point.py` main=1389 lines vs worktree draft=539 (worktree is the STALE earlier draft; main superseded it via commit `237b955ef7` "recover PDW1 fp32 realization arm" + refinement). NO strand owed; committing the draft would REGRESS. |
| `agent-a1f080a4426f446f9` | 3 (v10_ratecrush*) | yes (`bc1caf7f84`) | **superseded-by-main** — all 3 already on main HEAD; `v10_jxl_plane_codec.py` main=335 lines vs worktree=286 (main has the imagecodecs backend the draft lacks). Stale earlier draft. NO strand owed. |
| `agent-a2d41…`,`a5559…`,`a55b0…`,`a78be…`,`a8621…`,`aa118…`,`ac6d1…`,`acef6…`,`ae50f…` (9) | 0 | yes | **merged-but-fresh; NOT pruned** — every one committed TODAY 2026-07-28 (rp1 07:35, gc5 06:28, sc1 06:31, ar1 06:03, da1 06:06, oc1 07-27 23:27, fc1 00:36, …) = the active arms in MEMORY current-state. No `.done` marker, no arm registry certifying death. Clean+merged ⇒ signal already in main, but recency ⇒ plausibly-live parallel sessions. Pruning risks the cardinal sin for ~zero benefit. |
| `agent-ae029…` (sh1) | 0 | no | harvest source (§3–§4); not pruned. |
| `agent-ac3ed…` (ee1) | 0 | no | live arm, locked; not touched. |
| `agent-a377…` (fd1) | 3 | no | live arm; not touched. |
| stale `codexwt/*` `2026-07-19..23` (v4,v5,v7,v8,v13,v15,v18b,einstein,mdl,pose_attach,target_receipt,measurement_ladder) | 1–25 EACH | yes | **dirty-leftover-scratch; NOT prunable** — branches merged (signal in main) but worktrees carry uncommitted scratch → "never dirty" rule forbids pruning. Left in place; not touched. |
| `clwt/km1,pi1,r6cal` | 0 | yes | **merged-but-recent CLI arms** (07-27/07-28); plausibly live; not pruned. `clwt/rc1` 1-dirty → not pruned. |

**Net prune action: ZERO worktrees pruned.** Every clean+merged worktree is a fresh
(today) arm plausibly-live; every stale worktree is dirty. There is no worktree that is
simultaneously clean AND merged AND certifiably-dead. Honest boundary, not a miss.

## 3. sh1 harvest — the incumbent_v2 END-TO-END negative (the deliverable)

sh1 (`ddm/sh1_integration_20260727`, base main `ad464e269c`) composed the campaign's
FIRST end-to-end original-line candidate and ran it through the FULL 600-sample
`upstream/evaluate.py`. **incumbent_v2 = the CB1 `mycar_static_mask` packet** — merged
RG4 source-local PC1 base + admitted MyCar hood carrier (+319 B), E4 Brotli-Q11
byte-closed with self-contained emitted runtime.

- archive: **131,620 B**, SHA-256 `5e1441180f83a6d1d12dc72b574d6801f815c555ede3ca2f56ccb228bc45c3b3`
- **S = 23.913488** `[macOS-CPU advisory — real evaluator, real bytes]` (report.txt 23.91)
  = **6.190208 seg** (100·d_seg, d_seg 0.06190208)
  + **17.635640 pose** (√(10·d_pose), d_pose 31.10157967)
  + **0.087640 rate** (25·131620/37,545,489)
- distance to the 0.172 bar: **+23.741488**.

**Evaluator-chain proof (measurement apparatus cross-validated):**
- Inflate decoded in 19 resumable stages via the packet's own runtime (first attempt
  harness-killed rc=144, RESUMED via `tools/launch_detached_process.py` from stage
  checkpoints — the resumability non-negotiable paid for itself). Final raw
  3,662,409,600 B, SHA-256 `a6cee040…` = **BIT-IDENTICAL to the CB1 receipt**
  (deterministic decode reproduced on this host).
- Independent cross-check vs the CB1 frozen-scorer MENU1 chain (same bytes): d_seg agrees
  to ≤4.4e-9, d_pose to ≤3.9e-6. **The two independent measurement paths agree to ≤4e-6**
  — the frozen-scorer chain is validated end-to-end through the real upstream harness.
- `--device cpu` (MPS explicitly refused; Apple Silicon is never a 1:1 axis → advisory,
  NOT a contest-CPU score claim). Pointer UNMOVED.

## 4. Routing the incumbent_v2 negative (what the S decomposition proves)

1. **Pose term is 74% of S** (17.636 of 23.913; d_pose 31.10 vs a sub-0.01 need).
   → **Confirms the pose-terminal-solve staging law** (`pose_is_a_terminal_six_equation_solve_on_conditioned_seg_base`,
   `#383`): pose CANNOT be composed post-hoc as a carrier on a seg-shaped base; it must be
   the TERMINAL joint 6-equation solve on a frozen-seg base. A pose-blind composition
   (this candidate never shaped photometry for pose) lands d_pose ≈ 31 — exactly the
   `COMPACT_CODE_TO_PHOTOMETRY_POSE_INVERSE` / `#366` joint-descent blocker. This is the
   custody-grade empirical confirmation of the staging law on OUR original line's own
   composed candidate.
2. **Seg term is 26% of S** (6.190; d_seg 0.0619 vs the ~0.0008–0.001 need, ≈70× over).
   → The describe-line box (finite-price materialization) remains the binding seg crux
   (council routing card §5/§8; sister of the `ms2r_r3` box-solve law). d_seg alone is
   34× the whole 0.172 bar — distortion, not rate, is the entire gap.
3. **Rate is NOISE at this operating point** (0.0876; even WF7's blocked −1,776 B moves S
   by only −0.00118). → **Confirms `borrowed_incumbent_rate_polish_permanently_dead`** now
   on OUR original line's OWN composed candidate: rate polish is economically irrelevant
   until distortion falls orders of magnitude. Do NOT spend on rate here.

**One-line consumption:** the first real end-to-end row on our original line is a
custody-grade NEGATIVE that quantitatively re-proves the two standing doctrines (pose =
terminal joint solve; rate polish dead) and re-anchors the binding cruxes as pose
(`#366`/`#383`) + describe-line seg box — NOT rate, NOT composition of more carriers.

## 5. Pricing-wave deliverables inventory (integrated on sh1, harvest-signal-only)

| deliverable | key result | scope / status |
|---|---|---|
| **CB1** per-class carrier byte-close (`codex_findings_…cb1…`) | On merged RG4 PC1 base: MyCar static-mask carrier = the ONLY admitted strict-negative joint row (+319 B, ΔS_joint −0.0516 advisory); inherited polished v13 Lane program strongly uphill → REJECTED from #613 waterfill. Fresh byte-closed control 131,301 B (d_seg 0.06191, d_pose 31.281). | n600 rows through emitted CB1/E4 runtime + composite receiver + uint8 + frozen CPU scorers. `CB1_HAS_STRICT_NEGATIVE_JOINT_ROW`. Advisory. |
| **PF3B** 52-probe joint-improving hunt (`codex_findings_…pf3b…`) | Sealed support-positive RG3 alphabet DOES contain a strict joint-distortion-improving edge (rank-2: `pair523.class0_4.boundary…NEGATIVE_ONE_QUANTUM`): ΔD_joint −1.08e-4. But E4 parseback price +860 B → ΔS **+0.000465** (distortion/byte 1.26e-7 < 6.66e-7 break-even). **Real joint gain, NOT a total-score gain.** rank_v1 invalidation receipt included. | Closes the "0/162 null table was a granularity issue" question. Not pointer-eligible. |
| **WF7** seven-home stream waterfill (`codex_findings_…wf7…`, canonical-equations memo) | Seven-home granularity is actionable for LOSSLESS rate: 5 improving physical-home rows recode the exact 134,211-B seeded C1 state → 132,435 B (−1,776 B, byte-for-byte restore), ΔS −0.00118. CC3 sister falsifier −3,422 B (only −2,302 attributable to v15 leaves; no double-count). **Rate-only; NO #613 box member; d_seg/d_pose unchanged.** | `STREAM_PRICE_DOMAIN_NONEMPTY_RATE_ONLY;NO_613_BOX_MEMBER`. |

**WF7 rate leg is TYPED-BLOCKED, not silently summed** —
`WF7_LEG_NON_E4_STATE_CONTAINER`: WF7's −1,776 B is measured on its own seeded C1 state
container; the CB1/E4 packet receiver consumes `manifest.json` + `state/rg4.ddr4` and has
no DWF7 parser. Unlock = bind the DWF7 seven-home container as the serialization of
`state/rg4.ddr4` in the E4 exporter + receiver, re-prove double-compile + raw identity,
re-measure. (First-rung successor; only meaningful after distortion is addressed per §4.)

## 6. Registry rows owed (do NOT apply to main here)

sh1 carries ONE owed lane row for MAIN to apply at a quiet boundary
(`.omx/research/ddm_sh1_registry_rows_owed_20260727.json`, on the sh1 branch):
`ddm_pf3b_52probe_joint_improving_hunt` (phase 1, level 0, all gates false, research-only
notes). WF7/CB1 merges touched no registry rows. This cs1 arm did NOT apply it (main's
live `.omx/state/lane_registry.json` is owned by the parallel session).

## 7. a4ba modal-prep finding (charter assumption falsified, honestly)

The cs1 charter flagged `agent-a4ba…` as "modal_auth_eval_cpu C1 receiver custody prep —
REAL R6-chain signal, inspect + commit its strands." Verified: the a4ba branch's COMMITTED
work (mount tool_bootstrap / measure_uint8_lattice_feasibility / C1 receiver sources for
`modal_auth_eval_cpu`, commits `377df45980`/`178432e4c9`/`6a3cd12159`) is ALREADY MERGED
into main. The 5 DIRTY files are pdw1_* — a different topic, and each already exists on
main HEAD (4 byte-identical; the 5th is a stale 539-line draft superseded by main's
1389-line version via commit `237b955ef7`). **No strand owed; the a4ba signal already
landed on main.** Committing the drafts would regress. Same for a1f080's v10_ratecrush
drafts (§2).

## STORES CONSULTED

CLAUDE.md + AGENTS.md (NO-FAKE, THE GOAL, storage/provenance, MPS-never, main-SoT,
serializer discipline); MEMORY current-state (`pose_is_a_terminal_six_equation_solve…`,
`borrowed_incumbent_rate_polish_permanently_dead`, `box_retired_min_s_target…`,
`goal_is_sub015_or_below_official_leaderboard_best`); sh1 branch findings + receipt
(`ddm_sh1_compose_and_local_exact_findings_20260727.md`, `ddm_sh1_local_exact_receipt_20260727.json`,
`ddm_sh1_registry_rows_owed_20260727.json`); CB1/PF3B/WF7 branch findings memos at their
merged bytes; `git branch --no-merged main`, `git worktree list --porcelain`, per-worktree
`git status` + `merge-base --is-ancestor` + per-file sha256/line-count diffs; consolidation
monitor `tools/consolidation_debt.py` (before/after).

## DAG FEED

- **cs1 harvested the unmerged surface with ZERO merges to main and ZERO worktree
  prunes** (every clean+merged worktree is a fresh today-arm plausibly-live; every stale
  worktree is dirty — no clean∧merged∧dead worktree exists). The high-value signal is
  sh1's incumbent_v2 END-TO-END negative: first real `upstream/evaluate.py` 600-sample row
  on our original line, **S = 23.913488 `[macOS-CPU advisory]`** = 6.190 seg + 17.636 pose
  + 0.088 rate, +23.741 above the 0.172 bar; raw decode bit-identical to CB1, evaluator vs
  frozen-scorer chain agree to ≤4e-6 (apparatus cross-validated). The decomposition
  RE-PROVES on our own composed candidate: pose = terminal joint solve (74% of S, `#366`/
  `#383`), seg describe-box is the crux (26%), rate polish is dead (0.4%). sh1 stays
  UNMERGED (blocked on `direct_description_carrier_compose.py` dirty in the parallel
  session's WIP); its 3 pricing-wave branches (CB1/PF3B/WF7) are contained in it and are
  harvest-signal-only. PF3B: real joint-distortion edge but +0.000465 total-score (below
  break-even). WF7: −1,776 B lossless rate but typed-blocked on the DWF7→E4 binding and
  irrelevant at this operating point. a4ba/a1f080 dirty = already-on-main superseded
  drafts, no strand owed. Pointer **0.19108 UNMOVED**; no score claim; no paid dispatch;
  no merge to main; parallel-session WIP untouched.
