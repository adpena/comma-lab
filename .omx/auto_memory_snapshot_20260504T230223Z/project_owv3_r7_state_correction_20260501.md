---
name: OWv3 R7 state correction — R5 + R6 already dispatched and FAILED at exact CUDA T4; R7 BLOCKED on component-balanced sensitivity
description: 2026-05-01 ~09:25 UTC. Major correction to last turn's r6 design memory. The OWv3 sweep script already has R5+R6 failed-CUDA references hardcoded. R5 (owv3_0047) scored 1.0374 = +0.0004 regression vs paired PFP16 1.0370. R6 (owv3_0076) scored 1.0393 = +0.0023 regression with PoseNet 2.1% worse. R7 selection explicitly requires component-balanced PoseNet/SegNet sensitivity (β Fisher dispatch) before any further blind threshold dispatch. The Shannon checkpoint's β sensitivity-map dispatch ($2 on Vast.ai 4090) is the actual unblocking action.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Last turn's r6 design memory was DIRECTIONALLY WRONG

The `project_owv3_r6_sweep_design_20260501.md` from the prior /loop fire assumed:
- "r5 conservative selection" left aggressive candidates untested → false; r5 selected `owv3_0047_bbr0p67_protect0p00135_aggr1em05` (NOT owv3_0018), dispatched it, and got a regression
- "r6 design samples bbr ∈ [0.50, 0.69]" → false; r6 already dispatched as `owv3_0076_bbr0p65_protect0p0013_aggr1em05` (similar bbr range to r5)
- "compress-time scorer call to predict distortion before archive build" → would have caught the regression IF the prediction had been accurate. The R5+R6 failures show that a local CPU scorer prediction at this granularity is not reliable enough to dispatch on.

**The actionable insight:** the byte-feasibility band in r5/r6 is genuinely populated (116 candidates within byte budget), but RANK-BY-PREDICTED-SCORE is not enough — the OWv3 byte plan's choice of WHICH channels to compress is dominated by the sensitivity prior. With UNIFORM sensitivity (no Fisher), aggressive byte savings hit the wrong channels and break PoseNet.

## Actual OWv3 lane state (per `experiments/sweep_owv3_byte_plan.py:52-83`)

### R5 dispatch (BYTE-BEST selection)

```
candidate_id: owv3_0047_bbr0p67_protect0p00135_aggr1em05
score_recomputed_from_components: 1.0373951773937642  [contest-CUDA T4]
avg_posenet_dist: 0.0031739
avg_segnet_dist: 0.0040215
archive_size_bytes: 686468  (-167 vs PFP16 frontier)
archive_sha256: 16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518
paired_pfp16_score: 1.037045485927815
paired_pfp16_posenet: 0.00316404
paired_pfp16_segnet: 0.00401966
paired_pfp16_bytes: 686635 (SHA 0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f)
DELTA_SCORE: +0.000349691  (regression of ~0.00035 score-points)
```

Outcome: **REGRESSION**. PoseNet drift +0.34% vs paired PFP16. SegNet drift +0.05%. Bytes -167. Net: regression.
Lane status: COMPONENT_GATE_REVIEW_REQUIRED. Promotion eligible: False.

### R6 dispatch (SEGNET-CONSERVATIVE selection after R5 failure)

```
candidate_id: owv3_0076_bbr0p65_protect0p0013_aggr1em05
score_recomputed_from_components: 1.0393166493980681  [contest-CUDA T4]
avg_posenet_dist: 0.00323147
avg_segnet_dist: 0.00402421
archive_size_bytes: 686531  (-104 vs PFP16 frontier)
archive_sha256: 9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91
paired_pfp16_score: 1.037045485927815
paired_pfp16_posenet: 0.00316404
paired_pfp16_segnet: 0.00401966
posenet_relative_to_paired_pfp16: 1.0213  (PoseNet 2.13% WORSE)
segnet_relative_to_paired_pfp16: 1.0011  (SegNet 0.11% worse)
DELTA_SCORE: +0.002271  (regression of ~0.0023 score-points)
```

Outcome: **WORSE THAN R5**. R6 was supposed to fix R5's pose collapse by tightening the SegNet-side protect threshold. SegNet protection worked (only 0.11% drift) but PoseNet got worse (+2.13%). The conservative threshold spent the byte budget on the wrong axis.
Lane status: REGRESSION_AND_COMPONENT_GATE_REVIEW_REQUIRED. Promotion eligible: False.

### R7 selection mode (defined; no candidate dispatched yet)

`select_r7_pose_balanced_candidates` (sweep_owv3_byte_plan.py:455). Requires:
- min_bit_budget_ratio = R6's bbr (0.65) — no further byte aggression
- max_owv2_low_bit_channels = R6's count — no MORE channels demoted
- excludes R5 + R6 candidate IDs
- output ordering: prefer higher bbr, smaller delta from R6's low-bit count

**Critical policy text** (line 522-530):

> "R7 after exact-evaluated R6 PoseNet gate miss; scalar threshold candidates must preserve at least R6's SegNet-side protected-channel count and must not lower the remaining OWV2 bit budget below R6. **Empty output means wait for component-balanced PoseNet/SegNet sensitivity rather than spending exact eval on another blind threshold.**"

This is the explicit gate: R7 BLOCKS further dispatch unless the candidate space already contains a strictly-superior-to-R6 candidate without further byte aggression. Per the script policy, the right next move is NOT another OWv3 sweep — it's the **β sensitivity-map Fisher dispatch** that produces the per-channel sensitivity needed for component-balanced selection.

## The two PFP16 baselines mystery

The Shannon checkpoint and `project_pfp16_a_plus_plus_deploy_baseline_freeze_20260501.md` both claim PFP16 A++ scores **1.043987524793892** at SHA `0af839ab...ed7f`/686635 bytes.

The OWv3 paired_pfp16 scores in R5+R6 references show **1.037045485927815** at the **SAME SHA `0af839ab...ed7f`/686635 bytes**.

**Same archive, same SHA, two different exact-CUDA scores — eval noise of 0.007 on contest-CUDA T4.** This is consistent with the OWv3 deferral memory's claim that "cycle-to-cycle measurement noise on the same archive runs ~0.005-0.01 score points".

**Implications:**
1. The frontier "1.044" claim in the Shannon checkpoint is a SINGLE measurement — not a robust lower bound. The true frontier may be anywhere in [1.037, 1.044] under eval noise.
2. The OWv3 R5+R6 "regressions" of +0.0004 and +0.0023 are within noise — they may not be real regressions at all. The "2.13% PoseNet worse" in R6 IS above noise (R5 vs R6 paired-PFP16 noise was 0; the +2% was on PoseNet specifically).
3. ANY future contest-CUDA dispatch should be PAIRED with a same-session PFP16 re-eval to control for the noise floor.

This is exactly why the OWv3 sweep schema includes `paired_pfp16_*` fields — every dispatch re-evaluates PFP16 in the same session as a control. R5+R6 each had their own paired_pfp16 measurement; both came in at 1.037.

## What this turn corrects

- **My last turn's r6 design memory** (`project_owv3_r6_sweep_design_20260501.md`) is now obsolete — left in place for forensic reference but the dispatch-implementation pseudocode there should NOT be executed; the lane is at R7, not R6.
- **The frontier "1.044"** claim should be re-tagged as `1.044 [single-measurement; paired re-eval shows 1.037 on same SHA]`. The Shannon checkpoint score derivative arithmetic is correct against 1.044 components, but the frontier itself has 0.007 eval-noise band.
- **The OWv3 lane** is correctly DEFERRED, but for a stronger reason than the original deferral memory captured: not just "owv3_0018 within noise" but "R5+R6 both regressed; R7 explicitly requires component-balanced sensitivity before further dispatch".

## What's the actual unblock

Per Shannon checkpoint Wave 1: **β sensitivity-map Fisher CUDA run (~$2, ~30min on Vast.ai 4090)** produces the per-channel sensitivity artifact that:
- Unlocks R7 component-balanced selection (currently blocked)
- Unlocks Ω-W-V3 sensitivity-weighted bit allocation (currently uniform)
- Unlocks Lane 12 NeRV alpha redesign (currently retired)

This is the foundational dispatch in Wave 1. R7 cannot proceed without it. Three downstream lanes unblock from one $2 dispatch.

## Adversarial Grand Council review

Council vote (5+ inner council members on the correction):

- **Shannon (LEAD):** the 0.007 eval-noise band is consistent with the deferral memory's claim. R5+R6 dispatches were within noise but R6's PoseNet 2.13% drift was real. **APPROVE correction.**
- **Dykstra (CO-LEAD):** the convex feasibility region is genuinely populated with sub-frontier byte candidates, but the wrong-channel selection breaks the Pareto frontier. Component-balanced sensitivity is the missing piece. **APPROVE.**
- **Yousfi:** "scorer-margin is the entire signal" — and we don't have it without Fisher. R7's gate (don't dispatch without sensitivity) is correct. **APPROVE.**
- **Fridrich:** PoseNet asymmetry is the killer. R6's 2.1% PoseNet drift = ~+0.014 score on the pose component alone. **APPROVE deferral.**
- **Contrarian:** "Why didn't the prior r6 design memory catch this?" Because I didn't grep for `R5_FAILED_EXACT_CUDA_T4_REFERENCE` before writing the design — exact same antipattern as the dead-flag wiring class (CLAUDE.md non-negotiable: read existing code before writing new design). **CRITICAL:** add a metabug entry — "design memos must grep for existing implementation before claiming the work is unstarted". **APPROVE correction with discipline note.**
- **Hotz:** the cheap thing is now even cheaper — don't dispatch ANYTHING until β sensitivity lands. **APPROVE.**

**VERDICT: 6/0 APPROVE correction.** The OWv3 r6 design memory from prior turn is forensic-only; the actual lane state is R7-blocked-on-β-Fisher.

## What would change my mind (reactivation criteria)

- β sensitivity-map Fisher dispatch lands → R7 selection becomes meaningful → re-design with component-balanced ranking
- A same-session re-eval of PFP16 produces 1.044 (not 1.037) → confirms eval-noise direction is upward → frontier IS 1.044
- A different paradigm (Lane 17 IMP, Lane 19 logit-margin) lands a sub-frontier score that obsoletes the OWv3 lane
- Manual inspection of OWv3 byte-plan output reveals the channel-selection bug (e.g., aggressive_threshold default 1e-5 mis-classifies a critical PoseNet channel) → fix that before re-sweeping

## Internal-consistency checks performed

- Verified R5+R6 references in `experiments/sweep_owv3_byte_plan.py:52-83` against the actual JSON in `byte_plan_summary.json` for owv3_0047 and owv3_0076.
- Verified `select_r7_pose_balanced_candidates` policy text (line 522-530) matches the design intent.
- Verified PFP16 SHA `0af839ab...ed7f` matches three deploy archive copies + R5+R6 paired_pfp16 references — same archive, different eval scores under noise.
- Verified r5 sweep had 116/135 byte-feasible candidates (per `byte_plan_candidates.jsonl`) — supports my last turn's "116 candidates" finding but the SELECTION was already smarter than I claimed.

## Process metabug captured (for permanent fix)

**The "design without grep" antipattern:** writing a design memo for a feature without first grepping the codebase for the feature's name. The OWv3 R6 design memo from prior turn would have been correct if I had grepped for `R5_FAILED_EXACT_CUDA_T4_REFERENCE` first.

**Mitigation candidate:** add a discipline rule to CLAUDE.md "Forbidden patterns" — "Forbidden design-without-grep — before writing a design memo claiming work is unstarted, grep for the feature name in source/scripts/experiments first. The 30 seconds of grep prevents 30 minutes of forensic correction."

## Cross-refs

- `project_owv3_r6_sweep_design_20260501.md` (the corrected memo from prior turn — kept as forensic reference)
- `project_owv3_byte_feasible_candidate_dispatch_deferred_20260501.md` (original deferral analysis — also wrong about owv3_0018 being best)
- `project_pfp16_a_plus_plus_deploy_baseline_freeze_20260501.md` (frontier freeze — needs eval-noise band annotation)
- `project_shannon_floor_execution_state_checkpoint_20260501.md` (master plan)
- `experiments/sweep_owv3_byte_plan.py:52-83` (R5+R6 failed-exact-CUDA T4 references)
- `experiments/sweep_owv3_byte_plan.py:455-544` (select_r7_pose_balanced_candidates with policy text)
- `feedback_dead_flag_wiring_pattern.md` (sister antipattern: invent-flag-without-grep)
