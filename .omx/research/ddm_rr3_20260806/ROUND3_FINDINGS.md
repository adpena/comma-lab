# ddm_rr3 Round 3 PR130 Lift Wave Adversarial Review

Date: 2026-08-06

Reviewer: ddm_rr3

Charter: `.omx/tmp/codex_runs/rr3_prompt.md`

Common contract: `.omx/tmp/codex_runs/_common_contract.md`

Scorer use: none. No `upstream/evaluate.py`, no n600 scorer job, no exact score claim.

Verdict: NOT-CLEAN

Clean counter: 0/3

## Summary

Round 2 repairs improved the top-level MX1 launch-ticket artifact, but round 3 is not clean. Two critical blockers remain:

1. MX1's source emit path still regenerates a single-arm ticket with bare `argv_n32` / `argv_n120` keys on the next probe run, so the artifact-level two-arm amendment is not durable.
2. ET4's live row ledger already contains duplicate pair rows, while the final archive builder requires raw `len(rows) == 600`. The run can finish unique-pair coverage and still be blocked or produce corrupted aggregate metrics unless rows are repaired by unique pair.

No PR130 lift-wave component is promoted here. The campaign projection remains arithmetic-only and coupled to unmeasured local legs.

## Typed Findings

| ID | Severity | Component | Verdict | Evidence | Required disposition |
|---|---:|---|---|---|---|
| RR3-F1 | CRITICAL | MX1 launch ticket | NOT-CLEAN | `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_ticket.json` now has `argv_n32_arm_cap`, `argv_n32_arm_veh`, `argv_n120_arm_cap`, and `argv_n120_arm_veh`, with no bare top-level `argv_n32` / `argv_n120`. However `experiments/ddm_mx1_pr130_semantic_renderer.py:510-628` still defines `launch_ticket()` that returns bare `argv_n32` at line 534 and bare `argv_n120` at line 570. The main path writes this generated ticket into result JSON, so the next source-run probe can regress to a single-arm contract. | QUEUED to MAIN. Patch the emit path or gate it off, and add a static/generated-ticket check that fails on bare `argv_n32` / `argv_n120`. Do not fire n120 from MLX telemetry before CPU-torch verdict selection. |
| RR3-F2 | CRITICAL | ET4 n600 row ledger | NOT-CLEAN | Live snapshot during review: `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_solve_within_cvp_rows.jsonl` had 706 row entries, 555 unique pairs, 151 duplicate extra rows, and 45 missing pairs. `shards_rc.txt` recorded `shards rc: 1 0 0` and final stage skipped, while `shard_a.log` continued appending. Code evidence: `experiments/ddm_et4_solve_within_cvp_n600.py` resumes with `done = {row["pair"]}` at line 706, aggregates on raw `len(rows)` at lines 360-438, and `build_archive()` refuses unless `len(rows) == 600` at lines 596-597. | QUEUED to MAIN. Do not treat current ET4 aggregate as valid. Quarantine or repair rows by unique pair, decide duplicate resolution against the per-pair `.npz` patch records, fill the missing pairs, and add a uniqueness/locking guard before any final stage. Final completion should require exactly 600 unique pairs and one receiver-consumable patch per pair, not raw row count alone. |

## Verified Boundaries

| Check | Outcome | Evidence |
|---|---|---|
| MX1 artifact-level two-arm amendment | CLEAN at artifact level only | The checked top-level ticket contains separate cap/veh argv keys for n32 and n120, distinct input caches, distinct run dirs, and an arm-selection rule requiring both n32 arms first and no n120 until CPU-torch verdict selection. All flags in the four argv lists are accepted by `experiments/ddm_mx1_pr130_semantic_renderer.py` argparse. RR3-F1 blocks durability because the source emitter can regenerate the old shape. |
| HB1 resume caveat | BOUNDED-CLEAN for current Stage 2 state | `RESUME_CAVEAT.md` correctly labels any crash-before-epoch-30 restart as clean restart and any later warm-start continuation as `FORM_DEVIATED_RESUME`. Current `driver.log` grep found no `resume from latest` hit. Final Stage 2 checkpoints are absent, so no HPAC byte race row exists yet. Stage 3/4 remain governed by their rc/report receipts, not by this caveat alone. |
| HB1 Stage 3/4 argv | CLEAN | `pack_hpac_self_compress.py` and `codec_hpac_integer.py` both accept the driver flags used in `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/hpac_our_labels_driver.sh`. No invented flag was found. |
| MX2 source custody | CLEAN within scope | The vendored PR130 pose files under `src/tac/pr130_lift/pose/lifted/` match the declared upstream source shas after accounting for the inserted `borrowed_substrate_accounting` header. The source head in `vendor_manifest.json` is `2f94596bb0136d342254022a5c9584756eae0468`. |
| MX2 CPR1 repack race | CLEAN negative | `.omx/research/ddm_mx2_20260806/REPACK_RACE.md` reports 17 candidates, `cpr1_applied=0`, and no adopted byte delta. The harness only admits a CPR1 rewrite after symbol decode/encode/decode equality, and no current banked section matched the legacy carrier shape. No ghost CPR1 saving was found. |
| ET4 cache-lineage threads | CLEAN but superseded by RR3-F2 | The parent argmax rebuild receipt records batch=1, threads=4, old cache preservation, new sha, and `score_claim=false`. The ET4 driver uses `--threads 4` for shards and final stage. This does not cure the row-ledger duplicate blocker. |
| Campaign arithmetic | CLEAN arithmetic, not promotion | Recomputed score formula `100*d_seg + sqrt(10*d_pose) + 25*bytes/37545489`: PR130 external components `d_seg=0.00029660`, `d_pose=0.00002331`, `bytes=191052` give `S=0.172141297492`; the TY2 projection at `168892` bytes gives `S=0.157385863091`. The delta is a pure rate delta of `-0.014755434401`. No double-count was found, but the table itself says the top PR130 legs are coupled and local legs remain unmeasured. |

## Recall Evidence

Stores consulted:

- `.omx/tmp/codex_runs/rr3_prompt.md`
- `.omx/tmp/codex_runs/_common_contract.md`
- `PROGRAM.md`
- `CLAUDE.md` / `AGENTS.md`
- `docs/operating_manual_craft_handoff.md`
- `.omx/state/main_hot_state.md`
- `.omx/research/CANONICAL_RESEARCH_INDEX*`
- `.omx/research/sub015_DAG_*`
- `.omx/state/canonical_task_status.jsonl`
- `.omx/state/operator_p0_ledger.jsonl`
- Round 1 and round 2 PR130 lift-wave receipts
- Current MX1, HB1, MX2, ET4, and EH1 receipts and live SSD run artifacts
- Canonical equation listing via `tools/list_canonical_equations.py --json`

Queries included:

- `PR130|lift wave|MX1|MX2|HB1|ET4|EH1|ROUND2|round 2|CPR1|HPAC`
- `argv_n32|argv_n120|argv_n32_arm_cap|argv_n32_arm_veh|launch_ticket`
- `resume from latest|FORM_DEVIATED_RESUME|stage3|stage4|rc`
- `parent_argmax|batch1|threads 4|et4_solve_within_cvp_rows|patch_records`
- `0.157385863|0.172141|168892|191052`
- canonical equations for score formula, exact-score authority, and pointer separation

Beyond-charter findings that changed the plan:

- ET4 row duplication is a fresh live-run blocker and is stricter than the original cache-lineage question.
- MX1's artifact-level repair is insufficient because the source generator can recreate the old single-arm shape.
- No PR130-specific canonical equation superseded the local score formula or authority split found in the governing docs.

## Verification Commands And Results

- `git diff --cached --name-status`: no pre-existing staged files before this rr3 deliverable.
- Python argparse check over MX1's four top-level argv lists: no missing flags.
- `nl -ba experiments/ddm_mx1_pr130_semantic_renderer.py | sed -n '510,630p'`: confirmed bare `argv_n32` / `argv_n120` source emitter.
- `grep -n "resume from latest" /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/driver.log`: no hits at review time.
- Python argparse checks for HB1 Stage 3/4 scripts: no missing flags.
- Source sha/header-only comparison for MX2 vendored PR130 pose files: declared upstream shas match; vendored diffs are header-only.
- ET4 ledger counter over live rows: 706 rows, 555 unique pairs, 151 duplicate extras, 45 missing pairs at the cited snapshot.
- `cat /Volumes/VertigoDataTier/pact/ddm_et4_20260806/shards_rc.txt`: `shards rc: 1 0 0`; final stage skipped.
- Campaign score recomputation: `0.172141297492` for PR130 external bytes and `0.157385863091` for the TY2 projection bytes.

## Follow-On Disposition

- RR3-F1: QUEUED to MAIN. Non-trivial source-contract fix; rr3 did not edit MX1 live/probe source.
- RR3-F2: QUEUED to MAIN. Non-trivial live-run repair; rr3 did not kill or mutate ET4 run artifacts.
- HB1 resume and Stage 3/4 argv checks: FOLDED as bounded-clean evidence; no exact row exists yet.
- MX2 custody/repack checks: FOLDED as bounded-clean evidence; no CPR1 saving exists on current banked sections.
- Campaign arithmetic: FOLDED as projection-only; no pointer movement and no additive independent-leg claim.

## Boundaries

- No scorer or `upstream/evaluate.py` was run.
- No exact archive was built or evaluated.
- No contest-CPU, contest-CUDA, or promotion-eligible row is claimed.
- No live driver source or protected file was edited.
- No running process was killed or modified.
- No `/tmp` path is used as persisted evidence.
- All absence statements are bounded to the stores and commands listed above.

## Pointer Delta Honesty

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.

Contest pointer unchanged: borrowed `S = 0.19108`; PR130 official comment row remains external, not our custody frontier.
