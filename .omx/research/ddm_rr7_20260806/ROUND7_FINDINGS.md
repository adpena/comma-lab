# ddm_rr7 Round 7 PR130 Lift Wave Adversarial Review

Date: 2026-08-06
Reviewer: ddm_rr7
Axis: scorer-free source/artifact review
Score claim: false
Promotion eligible: false
Tokens: [no-triality] [p0-ledger-ok]

## Verdict

NOT-CLEAN.

RR6's one-line final-command fix is verified in the live repair script: the
final repaired Python command is now captured as `rc=0; command || rc=$?`, then
`final(repaired) rc=...` and the log tail always emit. That fix closes the RR6
diagnostic loss.

Round 7 still found boundary-sequence hazards. The original ET4 driver can
report a successful detached job after its own final stage fails, because it
does not propagate the final command rc. The repair script is fail-closed, but
on this host the pgrep quiescence proof returns rc=3 (`sysmon request failed`),
so the MAIN repair handoff can block without an alternate durable no-writer
proof. Row-1 MX1 also still has no executable CPU-torch verifier command in the
ticket/source path even though CPU-torch verdicts are the required selection
authority.

Clean counter remains 0/3. No scorer job was run, no ET4 repair was run, no live
ET4 file was edited, no archive was built, and no MX1 arm was fired.

## Typed Findings

| id | severity | surface | verdict_scope | status | finding | required action |
|---|---:|---|---|---|---|---|
| RR7-F1 | HIGH | ET4 original driver final-stage rc propagation | INSTANCE: `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_n600_driver.sh` | QUEUED-WITH-A-FIRE-ORDER | The driver uses `set -uo pipefail`, runs the final `experiments/ddm_et4_solve_within_cvp_n600.py --resume --threads 4 --build-archive --run-inflate --run-evaluate` command, then only echoes `final rc: $?` to `shards_rc.txt`. Because `set -e` is not active and the script does not `exit $final_rc`, a final-stage failure after all shards return 0 can make the child shell exit 0. With the known duplicate rows, the driver's final stage is expected to fail before archive write on raw `len(rows) != 600`; a canonical `--done-receipt` supervisor would then record `rc=0` for a boundary that did not produce `byteclose_archive_receipt.json`. This does not corrupt rows or create a false archive because `build_archive()` raises before receipt writes, but it can falsely notify the fleet that ET4 reached the boundary. | Patch the driver before any further driver-owned boundary use: `final_rc=0; ... > final_stage.log 2>&1 || final_rc=$?; echo "final rc: $final_rc" >> shards_rc.txt; exit $final_rc`. MAIN boundary consumption must ignore detached rc alone and require `byteclose_archive_receipt.json` plus `evaluate.recomputed_score`; if final rc is nonzero, fire the repair path instead of Row-1. |
| RR7-F2 | MEDIUM | ET4 repair quiescence proof | INSTANCE: `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_repair_rows_and_final.sh` on this host/sandbox | QUEUED-WITH-A-FIRE-ORDER | The repair script is now safety-correct but has an execution dependency that is not proven live: `pgrep -f ddm_et4_solve_within_cvp_n600` returns rc=3 here with `sysmon request failed with error: sysmond service not found` and `pgrep: Cannot get process list`. The script will refuse on rc=3, which is correct for data safety, but it means MAIN cannot assume "shard A ended" implies the repair can run from this execution context. Current live state also proves the boundary has not arrived yet: 734 raw rows, 583 unique pairs, missing 189-205, no final-stage or byteclose receipt, and no `.omx/tmp/codex_runs/et4_chain_v3.done.done`. | At the ET4 boundary, require a durable no-writer proof in addition to the 120s rows-mtime guard: either run the repair from a context where pgrep enumeration succeeds, or replace the pgrep-only proof with a driver/done receipt plus stable ledger SHA interval. Do not hand-wave rc=3 as "no process"; rc=3 remains REFUSE. |
| RR7-F3 | MEDIUM | MX1 Row-1 post-train authority path | FORMULATION: current Row-1 ticket/emitter contract, not the MLX trainer itself | QUEUED-WITH-A-FIRE-ORDER | The fixed source emitter now creates schema `ddm_mx1_row1_launch_ticket.v2_two_arm` with four MLX train argvs and no bare single-arm keys. The real run-root JSON is still the older nested schema `ddm_mx1_row1_launch_ticket.v1`, so regeneration is required. More importantly, I did not find an executable CPU-torch post-train verifier argv in `experiments/ddm_mx1_pr130_semantic_renderer.py`, the real MX1 artifacts, or the MX1 research docs. The ticket says n120 selection comes from the two n32 CPU-torch verdicts and MLX telemetry is research-signal only, but it only emits the MLX train commands. That leaves the Row-1 decision authority as prose unless MAIN supplies a separate verifier command at fire time. | Regenerate the ticket from the fixed emitter, then add or attach a per-arm CPU-torch verifier fire order before any n120 selection. The verifier must consume each arm's trained checkpoint/run dir, render through exact R/uint8, compare against `gt_seg_cache.pt` with frozen CPU-torch SegNet, emit axis-labeled result JSON, and be the only scale-up authority. |

## RR6 Fix Verification

| item | verdict | evidence | boundary |
|---|---|---|---|
| Final repaired command rc capture | VERIFIED | `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_repair_rows_and_final.sh:50-58` now sets `rc=0`, runs the final command with `|| rc=$?`, echoes `final(repaired) rc=$rc`, tails `final_stage_repaired.log`, then `exit $rc`. `bash -n` returned 0. | I did not run the repair. Verification is source inspection only, as chartered. |
| Current ET4 final receipt | ABSENT | `final_stage.log`, `final_stage_repaired.log`, and `byteclose_archive_receipt.json` do not exist in the ET4 bulk dir. `shards_rc.txt` still says `shards rc: 1 0 0` and `SHARD FAILURE - final stage skipped; rerun driver to resume`. | No ET4 row is available to consume. |

## Boundary Execution Walk

| transition | reviewed state | what breaks / remains unverified | disposition |
|---|---|---|---|
| Shard A ends -> original driver's final stage | Current run has not reached this transition: row ledger snapshot was 734 raw / 583 unique / 151 duplicate extras / 17 missing pairs (189-205). The driver script would only enter final stage if all shard rc values are zero. | If a resumed driver reaches all-shards-zero with duplicate raw rows still present, `build_archive()` refuses `len(rows) != 600`. The Python failure is fail-closed and happens before archive receipt writes, but RR7-F1 means the shell driver can still exit 0. | NOT-CLEAN. Treat final-stage rc and receipt presence as authority, not detached child rc alone. |
| MAIN repair script -> dedupe -> final stage | Repair script performs backup, keep-last dedupe, `sha_before`, `sha_after`, then repaired final stage. Missing pairs exit 5 before rewrite; final command rc capture is fixed. | In this sandbox pgrep cannot enumerate processes and returns rc=3. That is correctly refused, but it is an unproven execution dependency for MAIN. The current ledger mtime was recent and pairs 189-205 were missing, so a repair run now would also refuse before rewrite. | QUEUED. Run only after 600 unique pairs, durable no-writer proof, stable rows mtime/SHA interval, and pgrep rc=1 or equivalent stronger proof. |
| MAIN consumes `byteclose_archive_receipt.json` | Source writes the receipt to both bulk and receipt dirs only after archive build, optional inflate, and optional evaluate complete. `run_upstream_evaluate()` recomputes S from components and labels CPU on this host as advisory. | No receipt exists now. `evaluate.py`'s `--num-threads` controls dataset worker threads, while HB1 exports `OMP_NUM_THREADS=4` inside its own process; ET4 does not set OMP/MKL/VECLIB around evaluate. Co-running HB1 on CPU can contend for host threads even though it is not a scorer job. | QUEUED. Consume durable JSON fields: archive bytes/SHA/path plus evaluate recomputed score, d_seg, d_pose, axis, authority, n_samples, archive_bytes_scored, report path, score_claim, promotion_eligible. Do not consume rounded `Final score`. |
| MAIN regenerates MX1 v2 ticket and fires both n32 arms | Source emitter lines 510-595 returns `ddm_mx1_row1_launch_ticket.v2_two_arm`; argparse supports probe mode and all emitted arm flags. Existing real run-root artifacts have no `launch_arm_cap/` or `launch_arm_veh/` dirs. Cache hashes were rechecked: GT `286fe40a...`, tq1c `11fd8901...`, init `1549607d...`. | Existing `/Volumes/.../ddm_mx1_20260806/launch_ticket.json` is still nested schema v1. The exact regeneration command from argparse should be run at fire time, for example: `.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode probe --device gpu --run-dir /Volumes/VertigoDataTier/pact/ddm_mx1_20260806 --out /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/row1_probe_v2_two_arm_result.json`. Then read `.launch_ticket.schema == ddm_mx1_row1_launch_ticket.v2_two_arm` and fire `.launch_ticket.argv_n32_arm_cap` plus `.launch_ticket.argv_n32_arm_veh`. CPU-torch verifier command remains absent per RR7-F3. | QUEUED. Fire both n32 MLX arms only after ET4 boundary and no run-dir collision; do not select n120 until CPU-torch verifier JSON exists for both arms. |

## Recall Evidence

| source searched | query / command | found beyond charter seeds | changed plan |
|---|---|---|---|
| Memory registry | `rg -n "ddm_rr|ROUND[0-9]_FINDINGS|rr7|et4|byteclose_archive_receipt|v2_two_arm|BOUNDARY EXECUTION|final-command|errexit" /Users/adpena/.codex/memories/MEMORY.md` | No direct RR7/ET4/MX1 memory hit. | No memory-derived facts used; live repo artifacts govern. |
| Governing files | `rr7_prompt.md`, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md` (identical), `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | Live board orders ET4 evaluate before Row-1 and says ET4 owns the scorer slot; common contract requires scorer-free work unless charter owns the slot. | Stayed scorer-free and reviewed only source/artifacts. |
| Prior review rounds | Round 1 memo plus `ddm_rr2` through `ddm_rr6` findings | RR6 fixed final repaired rc capture; RR5/RR6 already require `byteclose_archive_receipt.json`; prior rounds did not close original driver rc propagation or emit a CPU-torch verifier command. | Verified RR6 first, then attacked boundary transitions. |
| Canonical equations registry | `.venv/bin/python tools/list_canonical_equations.py --json` filtered for PR130/lift/ET4/MX1/HB1/authority/score/launcher terms | Found general pointer/score/axis and launcher/done-receipt context; no PR130-lift-specific equation superseded the score authority or boundary contracts. | Kept recomputed score + axis labels as consumption authority; treated launcher rc as insufficient. |
| Research index / DAG / live state | Filtered `rg` over `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, `.omx/state/*`, and MX1/ET4 docs for PR130, ET4, MX1, byteclose, v2_two_arm, done receipt, CPU-torch verifier | Found Campaign #984 order, PR130 external separation, current hot-state ET4/MX1 sequencing, and BL1 detached-launcher rc-amplifier hardening. | Added RR7-F1 rather than trusting detached done receipts. |
| Live ET4 artifacts | Driver, repair script, rows JSONL, shard logs, summary, launch manifests, pgrep/ps behavior | Rows are incomplete; final receipt absent; `ps` is denied and `pgrep` returns rc=3 here; repair script fixed RR6 but depends on process enumeration or an equivalent proof. | Added RR7-F1/RR7-F2 and did not run repair. |
| Live MX1 artifacts | Source emitter, argparse, existing launch JSON, launch docs, parity receipt, cache/input files | Source emits v2 two-arm argvs; existing real JSON remains v1; no arm dirs exist; no executable CPU-torch verifier command found in searched scope. | Added RR7-F3 and named exact probe regeneration command. |

## Verification Commands And Results

| command | result |
|---|---|
| `cmp -s CLAUDE.md AGENTS.md` | Identical files. |
| `bash -n /Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_repair_rows_and_final.sh` | `bash_n_rc=0`. |
| Source inspection of ET4 repair script | RR6 fixed final repaired rc capture at lines 50-58. |
| Source inspection of ET4 driver | Final command rc is echoed but not propagated; driver can exit 0 after final-stage failure. |
| ET4 rows counter over live JSONL | 734 raw rows, 583 unique pairs, 151 duplicate extras, 17 missing pairs `[189..205]`, sha256 `0fccb8ee6807726b33b2b8b5789be1bd3b3024a0c7de0da7de71acdfe2ec19b3`. |
| `cat /Volumes/VertigoDataTier/pact/ddm_et4_20260806/shards_rc.txt` | `shards rc: 1 0 0`; final stage skipped. |
| file existence checks for ET4 final artifacts | No `final_stage.log`, no `final_stage_repaired.log`, no `byteclose_archive_receipt.json`. |
| `.omx/tmp/codex_runs/et4_chain_v3.done.done` check | Done receipt absent during review. |
| `ps -p <et4 pid>` | Denied by sandbox: `operation not permitted`. |
| `pgrep -f "ddm_et4_solve_within_cvp_n600"` | rc=3 with `sysmon request failed with error: sysmond service not found` and `pgrep: Cannot get process list`. |
| `jq` / source inspection of MX1 | Existing real launch JSON nested ticket schema is v1; source emitter returns v2 two-arm; no `launch_arm_cap` or `launch_arm_veh` dirs exist. |
| `shasum -a 256` over MX1 inputs | GT cache `286fe40a2a29aa6950684f43229fce3a4a284ac7ffc65040e7e18953b95787d4`; tq1c cache `11fd89016ab33a4b221975dafac0a572d66c372db1accf62284a3e81acddcc54`; init `1549607db224ea2c4681738dbcc80d2ba9dd453de72db1cf60309985d0602eaf`. |
| `git diff --cached --name-status` | Empty before this RR7 artifact; staged index was not touched during review. |

## Follow-On Disposition

| follow-on | disposition |
|---|---|
| ET4 original driver boundary | QUEUED-WITH-A-FIRE-ORDER: patch rc propagation or require MAIN to ignore driver rc and consume only `shards_rc.txt` plus final JSON receipts. |
| ET4 repair/final | QUEUED-WITH-A-FIRE-ORDER: after unique=600 and no-writer proof, run repair; current run is incomplete and pgrep rc=3 blocks this context. |
| ET4 final result consumption | QUEUED-WITH-A-FIRE-ORDER: consume `byteclose_archive_receipt.json` fields and recomputed score only. |
| MX1 Row-1 | QUEUED-WITH-A-FIRE-ORDER: regenerate v2 ticket via the probe command above, fire both n32 arms, add/attach CPU-torch verifier commands before n120 selection. |
| HB1 co-run during ET4 evaluate | QUEUED-WITH-A-FIRE-ORDER: if HB1 is still training, either pause/schedule around ET4 evaluate or record host contention as a boundary caveat; ET4 evaluate uses CPU advisory path with `--num-threads 2`, HB1 exports 4 thread env vars inside its process. |

## Boundaries

This review did not run `upstream/evaluate.py`, did not run an n600 scorer job,
did not build or evaluate an archive, did not run the ET4 repair script, did
not mutate the ET4 rows ledger, did not fire MX1, did not edit live ET4 files,
and did not touch the staged git index. Absence statements are bounded to the
files and commands listed above.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
