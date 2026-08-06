# ddm_rr6 Round 6 PR130 Lift Wave Adversarial Review

Date: 2026-08-06
Reviewer: ddm_rr6
Axis: scorer-free source/artifact review
Score claim: false
Promotion eligible: false
Tokens: [no-triality] [p0-ledger-ok]

## Verdict

NOT-CLEAN.

The RR5 pgrep capture fix is verified by source inspection and MAIN's two branch
receipts cover the liveness guard's reachable pgrep cases that can be tested
before ET4 truly finishes. I found one remaining fail-closed but real repair
script defect: the final repair/evaluate Python command is followed by `rc=$?`
under `set -e`, so a nonzero final-stage command exits before the script prints
its custom `final(repaired) rc=...` and tail diagnostics. This does not create a
ledger-corruption or false-success path, but it is still an unfixed control-flow
defect in the final handoff. I did not patch it because the RR6 charter says not
to touch ET4 files or run ET4's repair.

Clean counter remains 0/3. No scorer job was run, no archive was built, and no
live ET4/MX1/MX2 driver artifact was edited.

## Typed Findings

| id | severity | surface | verdict_scope | status | finding | required action |
|---|---:|---|---|---|---|---|
| RR6-F1 | LOW | ET4 repair script final-stage diagnostics | INSTANCE: `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_repair_rows_and_final.sh` | QUEUED-WITH-A-FIRE-ORDER | The RR5 pgrep guard is now errexit-safe, but the final `.venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --resume --threads 4 --build-archive --run-inflate --run-evaluate > $LOG/final_stage_repaired.log 2>&1` command is immediately followed by `rc=$?` while `set -euo pipefail` is still active. If that final command returns nonzero, bash exits before the custom `final(repaired) rc=$rc` echo and `tail -20` run. The failure is fail-closed and the Python rc still becomes the script rc, but the handoff loses the intended diagnostic tail exactly when MAIN needs failure evidence. | Before running the real final repair, wrap the final command in the same errexit-safe pattern used for `pgrep` (`rc=0; cmd || rc=$?`) so nonzero final-stage failures still emit `final(repaired) rc=...` and the final log tail. Then run only at the ET4 boundary after 600 unique pairs, no live writer, and stable ledger mtime. |

## Round-5 Fix Verification

| item | verdict | evidence | boundary |
|---|---|---|---|
| RR5 pgrep capture under `set -e` | VERIFIED | Current script uses `prc=0; pgrep -f "ddm_et4_solve_within_cvp_n600" > /dev/null || prc=$?`, then refuses every `prc != 1`. That reaches explicit discrimination for rc=0, rc=1, and rc>=2 without bare-command errexit. | I did not rerun the repair. Verification is source inspection plus MAIN branch receipts named in the RR6 charter. |
| Branch-A worker-alive receipt | COVERS rc=0 guard-1 | Charter records branch-A (worker alive) returned rc=4 via guard-1. This covers the live-writer refusal path. | Receipt supplied by MAIN; not re-executed here. |
| Branch-B no-match copy receipt | COVERS rc=1 through guard-2 | Charter records branch-B (sed-patched no-match copy) survived pgrep and refused rc=4 via the ledger mtime guard ("modified 52s ago"). This proves the intended no-match path reaches the second guard after RR5's fix. | It does not cover the Python rows audit or final-stage command. |
| rc>=2 process-enumeration errors | VERIFIED BY SOURCE | With the current `prc=0; pgrep ... || prc=$?` shape, any pgrep rc>=2 sets `prc` and then `if [ "$prc" -ne 1 ]` refuses with exit 4. | Not branch-tested this round; prior RR4/RR5 host behavior showed rc=3 can occur. |
| `date` / `stat` / rows-file absence | FAIL-CLOSED | `now=$(date +%s); mt=$(stat -f %m "$ROWS")` runs after the pgrep guard. If `date` or `stat` fails, `set -e` exits before any rewrite. A missing rows file therefore fails closed, though without a custom message. | No mutation; no missing-file control was run. |
| Python heredoc missing-pair exit | FAIL-CLOSED | The heredoc exits `sys.exit(5)` before backup/rewrite when pairs are missing. Under `set -e`, that stops the script before the final stage. | The branch past mtime remains untested on the live run. |
| Python backup/rewrite success path | UNTESTED UNTIL ET4 ENDS | The branch still not covered is: pgrep rc=1, ledger mtime >=120s, rows audit executes, and either missing-pair rc=5 or 600-unique backup/dedupe/final-stage execution occurs. The true final success path cannot be tested until the run reaches 600 unique pairs and the writer is quiescent. | Do not claim full coverage before this branch is exercised. |

Current ET4 snapshot during RR6: summary captured `2026-08-06T22:01:24.950129Z`, `n_rows=729`, `remaining_count=22`, `archive=null`, `aggregate.S=null`, and `full_population_complete=false`; `shards_rc.txt` still says `shards rc: 1 0 0` and "SHARD FAILURE - final stage skipped; rerun driver to resume."

## Fresh-Hunt Checks

| surface | result | evidence | MAIN read / fire instruction |
|---|---|---|---|
| ET4 final-stage consume path | CLEAN READ CONTRACT; no final row present | `experiments/ddm_et4_solve_within_cvp_n600.py` writes `byteclose_archive_receipt.json` to both bulk and receipt dirs after archive/build/inflate/evaluate. `EvaluateResult` fields from `src/tac/submission_chain.py` are `score_axis`, `authority`, `d_seg`, `d_pose`, `rate_from_evaluate`, `evaluate_final_score`, `recomputed_score`, `recomputed_vs_evaluate_delta`, `n_samples`, `archive_bytes_scored`, `report_path`, `score_claim`, and `promotion_eligible`. The current summary has no archive and no S. | When final lands, read durable JSON, not stdout. Required fields: `archive.archive_bytes`, `archive.archive_sha256`, `archive.path`, `evaluate.recomputed_score`, `evaluate.d_seg`, `evaluate.d_pose`, `evaluate.score_axis`, `evaluate.authority`, `evaluate.n_samples`, `evaluate.archive_bytes_scored`, `evaluate.report_path`, `evaluate.score_claim`, and `evaluate.promotion_eligible`. Cross-check summary fields `aggregate.S`, `aggregate.dS_vs_named_baseline`, `aggregate.baseline.S`, and `aggregate.baseline.archive_bytes`. Treat `evaluate.evaluate_final_score` only as rounded evaluate.py cross-check. |
| ET4 axis and baseline | CLEAN | Default `--eval-device cpu` routes through `axis_and_authority`; on this host CPU is advisory, not contest authority. ET4 imports `BASELINE_S=0.7534578126155775`, `BASELINE_BYTES=357837`, `BASELINE_D_SEG=0.004305419922`, and `BASELINE_D_POSE=0.000716508925`. | Delta-S should cite live best `0.7534578126155775 @ 357,837 B`, not the older common-contract `0.7539807296911207 @ 357,836 B`. The expected axis for default host CPU is `[macOS-CPU advisory]` / advisory; no contest claim. |
| Row-1 v2 two-arm fire checklist | CHECKLIST RE-EMITTED | The nested dedicated ticket has four arm argv keys and no bare `argv_n32`/`argv_n120`, but its nested schema is still `ddm_mx1_row1_launch_ticket.v1`; RR4's dry source artifact emitted `ddm_mx1_row1_launch_ticket.v2_two_arm`. No `launch_arm_cap`/`launch_arm_veh` dirs exist under the MX1 run root at RR6 inspection. Host parity n4 exists and clears Row-1 as a training substrate only; CPU-torch verdict remains required. | At ET4 boundary, regenerate/update the real MX1 ticket from current source and require schema `ddm_mx1_row1_launch_ticket.v2_two_arm`. Fire both n32 arms: ARM-CAP `input=GT,target=GT` and ARM-VEH `input=tq1c,target=GT`, same stratified n32/seed policy, separate `launch_arm_cap/` and `launch_arm_veh/` run dirs. Re-hash tq1c cache, GT cache, and init checkpoint; refuse any existing run-dir collision unless it is an explicit compatible resume. Require Metal probe success and record first real Metal s/step; abort if projected wall clock exceeds 4x the CPU-derived estimate from the amended ticket. Consume only CPU-torch verifier results; MLX telemetry stays `[macOS-MLX research-signal]`. |
| MX2 pose fit-target constructibility | BLOCKED; no immediate long-run fire | The tq1c archive exists at `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes` (357,837 B), and the inflated parent raw exists at `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_tq1c_decode/submission/inflated/0.raw` (3,662,409,600 B). However the named adapter outputs are absent: `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/inputs/gt_pose_cache_600.pt` and `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/master_cache/OUR_SURFACE_MASTERS.pt` did not exist at RR6 inspection. The source trainer requires a target cache with `seg` and `pose`, and rejects a reused master cache whose `source_checkpoint` differs from `--master-checkpoint`. The MX1-renderer surface cannot exist before Row-1 produces it. | Row-2 can immediately fire only adapter/resume prep, not the 20k-step pose fit. First construct the tq1c master-surface adapter and target cache with hashes, plus a true resume wrapper/patch for `train_pose_carrier_full.py`. The MX1-surface variant must wait for Row-1 output. Do not run n600 or long nonresumable pose work while ET4 owns the scorer slot. |

## Mandatory New Assumption

Rounds 1-5 did not explicitly name this assumption:

> A source surface that exists as archive/raw bytes is equivalent to a fireable pose fit target.

RR6 refutes it for MX2. The tq1c frame-1 surface exists as bytes, but the PR130 pose trainer cannot consume it until an adapter materializes the `target-cache` and a checkpoint-bound `master-cache`, with provenance hashes and a true resume path. The mx1-renderer fit target is even later: it must wait for Row-1 to produce the renderer surface.

## Recall Evidence

| source searched | query / command | found beyond charter seeds | changed plan |
|---|---|---|---|
| Memory registry | `rg -n "rr6|common_contract|codex_runs|basis crop|crop|bulk basis|reverse-reducer|reducer" /Users/adpena/.codex/memories/MEMORY.md` | No RR6-specific prior memory hit; unrelated road/lane crop memory appeared. | No plan change; live repo artifacts governed. |
| Governing files | `rr6_prompt.md`, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | Live board supersedes the common-contract older frontier line with `S=0.7534578126155775 @ 357,837 B`; ET4 owns scorer slot; Row-1 fires after ET4. | Used live baseline and stayed scorer-free. |
| Prior rounds | Round 1 memo plus `ddm_rr2` through `ddm_rr5` findings | RR5 named pgrep errexit bug and final-stage read checklist; RR5 also warned the dedicated MX1 JSON still needs fire-time regeneration. | Verified RR5 fix first, then widened to final consume, Row-1, and MX2. |
| Full corpus/state | `rg --max-filesize 1M "PR130|lift wave|mx1|mx2|et4|hb1|Row-1|Row-2|pose carrier|v2_two_arm|final_stage_repaired|repair_rows|pgrep|HPAC|fit-target|gt_pose_cache|OUR_SURFACE_MASTERS" ...` over research/state/DAG/task ledgers | Found active Campaign #984 sequence, MX1 host parity ruling, MX2 launch blockers, and old pose-carrier caveats. | Added MX2 fit-target constructibility and adapter/resume gate to the RR6 readout. |
| Canonical equations registry | `.venv/bin/python tools/list_canonical_equations.py --json` with PR130/lift/axis/authority terms inspected | No PR130-lift-specific equation superseded contest score arithmetic or authority separation. | Kept recomputed-score and axis-field read instructions. |
| Live ET4/MX1/MX2 artifacts | ET4 repair script, ET4 summary/shard logs, MX1 launch ticket/parity docs, MX2 launch/receipt/parity/source trainer | Confirmed current ET4 is incomplete; MX1 has host parity but no arm dirs; MX2 placeholder input paths absent and trainer cache guard blocks direct reuse. | Produced RR6-F1 and the final fire/read contracts. |

## Verification Commands And Results

| command | result |
|---|---|
| `bash` source inspection of `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_repair_rows_and_final.sh` | Current pgrep capture is errexit-safe; final command rc capture remains unsafe for diagnostics under `set -e`. |
| `jq` over `.omx/research/ddm_et4_20260806/et4_solve_within_cvp_summary.json` and bulk mirror | No final row: `archive=null`, `S=null`; live snapshot later showed `n_rows=729`, `remaining_count=22`, `full_population_complete=false`. |
| `cat /Volumes/VertigoDataTier/pact/ddm_et4_20260806/shards_rc.txt` | `shards rc: 1 0 0`; final stage skipped; resume still required. |
| `tail -n 40 /Volumes/VertigoDataTier/pact/ddm_et4_20260806/shard_a.log` | Shard A had appended through pair 183 / rows 729 in the inspected tail, confirming no final consumption yet. |
| `jq` over `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_ticket.json` | Top-level driver schema v1 contains nested `launch_ticket` schema v1 with four arm argvs and no bare n32/n120 keys; fire-time v2 regeneration remains required. |
| `find /Volumes/VertigoDataTier/pact/ddm_mx1_20260806 -maxdepth 3 ...` | No `launch_arm_cap`, `launch_arm_veh`, `n32_metal`, or `n120_metal` arm run dirs found. |
| `stat` on tq1c archive and parent raw | Archive exists at 357,837 B; inflated raw exists at 3,662,409,600 B. |
| `test -e` on MX2 adapter paths | `gt_pose_cache_600.pt` absent; `OUR_SURFACE_MASTERS.pt` absent. |
| `rg`/`sed` over `train_pose_carrier_full.py` | Trainer requires `seg`/`pose` target cache, shape checks 600 pairs, and rejects `--reuse-master-cache` if `source_checkpoint` mismatches `--master-checkpoint`; latest/best saves exist but no true resume load path. |
| `git diff --cached --name-status` | Empty before this RR6 artifact; staged index was not touched. |

## Follow-On Disposition

| follow-on | disposition |
|---|---|
| ET4 RR5 repair/final | QUEUED-WITH-A-FIRE-ORDER: first patch final command rc capture for diagnostics, then only run after 600 unique pairs, pgrep rc=1, ledger mtime >=120s, backup+sha, dedupe keep-last, archive, inflate, and evaluate. |
| ET4 final result consumption | QUEUED-WITH-A-FIRE-ORDER: consume `byteclose_archive_receipt.json` and repaired summary fields listed above; never consume current partial aggregate or rounded stdout as a row. |
| Row-1 ARM-CAP / ARM-VEH | QUEUED-WITH-A-FIRE-ORDER: regenerate v2 two-arm ticket at the ET4 boundary, re-hash inputs, check Metal and run-dir collisions, run both n32 arms, choose n120 only from CPU-torch verifier results. |
| MX2 Row-2 pose | QUEUED-WITH-A-FIRE-ORDER: build adapter caches and true resume before any long fit; tq1c-surface adapter can be built from existing bytes, mx1-surface fit must wait for Row-1 output. |
| HPAC harvest | FOLDED from RR5: no new RR6 HPAC finding; existing per-stage rc/exact-decode harvest contract still binds. |

## Boundaries

This review did not run `upstream/evaluate.py`, did not run an n600 scorer job,
did not build or evaluate an archive, did not mutate the ET4 rows ledger, did
not edit live driver scripts, and did not touch the staged git index. All
absence statements are bounded to the files and commands listed above.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
