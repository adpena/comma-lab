# ddm_rr5 Round 5 PR130 Lift Wave Adversarial Review

Date: 2026-08-06
Reviewer: ddm_rr5
Axis: scorer-free source/artifact review
Score claim: false
Promotion eligible: false
Tokens: [no-triality] [p0-ledger-ok]

## Verdict

NOT-CLEAN.

RR4's ET4 pgrep fail-open class is closed on data safety, but the repaired script is still not executable as the final repair handoff: `pgrep` is a bare simple command under `set -euo pipefail`, so rc=1 (the only intended "no worker, proceed" path) exits the script before `prc=$?`. rc>=2 enumeration errors also exit before the intended `REFUSE ... exit 4` message. This is fail-closed, not a ledger-corruption path, but it blocks the repair/final stage and leaves the rc-classification fix unverified.

No scorer job was run. No live driver script, live rows ledger, upstream file, or staged index entry was edited. Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.

## Typed Findings

| id | severity | surface | verdict_scope | status | finding | required action |
|---|---|---|---|---|---|---|
| RR5-F1 | HIGH | ET4 repair script liveness guard | INSTANCE: `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_repair_rows_and_final.sh` | QUEUED-WITH-A-FIRE-ORDER | Lines 15-16 attempt `pgrep ...; prc=$?` under `set -euo pipefail`. In bash, a bare `pgrep` rc=1 exits immediately, so the only intended proceed path cannot reach the `if [ "$prc" -ne 1 ]` branch. On this host the same shape with a no-match pattern returned the process-enumeration failure rc=3 and exited before any custom REFUSE line. The current script is fail-closed, but not a working repair/final command. | Capture `pgrep` without `errexit` interception, e.g. `set +e; pgrep ...; prc=$?; set -e`, or `if pgrep ...; then prc=0; else prc=$?; fi`. Then run controls: rc=0 refuses with exit 4, rc=1 reaches the ledger-mtime/missing-pairs guards, rc>=2 refuses with exit 4 and the explicit message. Do not run final repair until the no-worker rc=1 path is proven. |

## Round-4 Fix Verification

| item | verdict | evidence | boundary |
|---|---|---|---|
| RR4-F1 pgrep fail-open repair | NOT-VERIFIED | The current script comments lines 13-14 correctly state `0=found, 1=no-match, >=2=ENUMERATION ERROR`, and line 17 would refuse every rc except 1. However line 15 is a bare `pgrep` under `set -euo pipefail` at line 8, so rc=1 or rc>=2 exits before `prc=$?`. | Safety is fail-closed, but the final repair path still cannot be trusted to execute on a quiet host. RR5-F1 blocks CLEAN. |
| RR4-F1 independent 120s ledger-mtime guard | VERIFIED-BUT-UNREACHED | Lines 20-24 refuse when the rows ledger was modified within 120s. A missing `ROWS` file would also fail closed via `stat -f %m "$ROWS"` under `set -e`, before any rewrite. Current rows ledger exists and was still advancing during review. | This guard is sound after pgrep capture is fixed; today it is unreachable for rc=1 because of RR5-F1. |
| RR4-F1 Python dedupe missing-pair refusal | VERIFIED | Lines 33-35 compute missing pairs and `sys.exit(5)` before backup/rewrite. As a simple command under `set -e`, a nonzero Python exit prevents the final stage. | Not executed against the live ledger. Current ledger was incomplete during review, so this refusal would be expected if reached. |
| RR3-F2 dedupe keep-LAST assumption | VERIFIED | `append_jsonl()` opens the rows path in append mode at `experiments/ddm_et4_solve_within_cvp_n600.py:138-142`; the runner loads prior rows on resume and builds `done` from row pair ids at lines 705-720; new records are appended at lines 907-908. The driver shard ranges are disjoint (`8..206`, `206..403`, `403..600`). A resumed duplicate row is therefore chronologically later in file order; no source path rewrites earlier JSONL lines. | This verifies the ordering premise, not final-row correctness before 600 unique pairs exist. |
| RR4-F2 HB1 harvest caveat sharpening | VERIFIED | `.omx/research/ddm_hb1_20260806/RESUME_CAVEAT.md:14-17` says `all done` is only a loop-completion marker and requires per-stage rc plus exact encode/decode reports. The live driver still logs stage rc separately; the review snapshot had no Stage 3/4 lines and no `all done`. | No HB1 byte row exists yet. Harvest remains blocked until per-payload pack/encode/decode receipts parse. |

## Fresh-Hunt Checks

| surface | result | evidence | action |
|---|---|---|---|
| ET4 final-stage result consumption | CLEAN-CHECKLIST; no final row present | No `final_stage_repaired.log` or `byteclose_archive_receipt.json` existed in the ET4 run dirs during review. When it lands, MAIN should read durable JSON, not stdout: `byteclose_archive_receipt.json` fields `archive.archive_bytes`, `archive.archive_sha256`, `evaluate.recomputed_score`, `evaluate.d_seg`, `evaluate.d_pose`, `evaluate.score_axis`, `evaluate.authority`, `evaluate.n_samples`, `evaluate.report_path`, and `evaluate.score_claim`. `evaluate.evaluate_final_score` is only the rounded evaluate.py cross-check. Cross-check `.omx/research/ddm_et4_20260806/et4_solve_within_cvp_summary.json` fields `aggregate.S`, `aggregate.dS_vs_named_baseline`, and `aggregate.baseline`. | The row axis should be `evaluate.score_axis == "[macOS-CPU advisory]"` for the default CPU run on this host; the row is not contest authority. Delta-S baseline is verified as `BASELINE_S=0.7534578126155775`, `BASELINE_BYTES=357837`, parent sha `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`. |
| ET4 live state during review | NOT-COMPLETE | Snapshot: rows ledger had 721 raw rows, 570 unique pairs, 151 duplicate extras, and 30 missing pairs; `shards_rc.txt` said `shards rc: 1 0 0` and final stage skipped. Shard A later appended pair 176 and 177 in the viewed log tail, so the run was still not quiescent. | Do not consume any ET4 final result until 600 unique pairs, no live writer, dedupe, archive, inflate, and evaluate receipts exist. |
| Row-1 v2 two-arm fire checklist | CHECKLIST-EMITTED | Source now emits `ddm_mx1_row1_launch_ticket.v2_two_arm` with four argv keys and no bare `argv_n32`/`argv_n120`. The RR4 dry artifact sha is `e9c74ee31e428aed755c0d7e52cd530f16939829ba5a25acfd4ffed44d245ee4`. Cache identity verified now: tq1c `11fd89016ab33a4b221975dafac0a572d66c372db1accf62284a3e81acddcc54`, GT `286fe40a2a29aa6950684f43229fce3a4a284ac7ffc65040e7e18953b95787d4`, init checkpoint `1549607db224ea2c4681738dbcc80d2ba9dd453de72db1cf60309985d0602eaf`. No `launch_arm_cap` or `launch_arm_veh` dirs existed under `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806`, and no collision with ET4 dirs was found. | At fire time: regenerate/update the dedicated ticket from current source under the real MX1 run root, verify schema `v2_two_arm`, re-hash both caches and init, require Metal probe non-blocked, record first real Metal s/step before n120, refuse any resume whose checkpoint metadata does not match argv, and consume only CPU-torch verifier verdicts. |
| HPAC epoch trajectory | BOUNDED-CLEAN | Driver log epochs `[0,1,2,4,6,8]` had estimated joint bytes `[129274,129274,128134,124774,123082,122128]`, nonincreasing through epoch 8. Token bytes and bpp were not monotone (`102248,102248,102336,99928,98940,98981`; bpp rose at epochs 2 and 8), so only the joint estimate is monotonically descending. `hpac_selfcompress_e60.latest.pt` existed at 172,219 bytes with mtime `2026-08-06T21:37:05Z`, matching the epoch-8 driver log mtime and the `--eval-every 2` cadence. | Continue to treat trainer estimates as estimates only. No exact HPAC byte row before Stage 3 pack and Stage 4 encode/decode `--require-exact` receipts. |

## Mandatory New Assumption

Rounds 1-4 did not name this assumption:

> Capturing a command's rc on the next line is equivalent to capturing it under `set -e`.

RR5-F1 refutes it. The guard now says the right thing, but bash exits before the rc-classification branch on exactly the rc=1 path that is supposed to proceed.

## Recall Evidence

| source searched | query / command | found beyond charter seeds | changed plan |
|---|---|---|---|
| Governing files | `rr5_prompt.md`, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | ET4 owns the scorer slot; Row-1 waits for ET4; no scorer use allowed here; current own-vehicle frontier is `0.7534578126155775 @ 357,837 B`. | Kept review scorer-free and made ET4 repair/final the critical gate. |
| Prior round receipts | Round 1 memo, `ddm_rr2/ROUND2_FINDINGS`, `ddm_rr3/ROUND3_FINDINGS`, `ddm_rr4/ROUND4_FINDINGS` | Prior clean counter is 0/3; RR4 fixed pgrep semantics in intent, sharpened HB1 harvest, and verified MX1 source-level v2 two-arm generation. | Verified fixes instead of reopening settled MX2 and MX1 source-emitter points. |
| Full corpus/state | `rg "PR130|lift wave|ddm_rr[234]|ET4|HB1|MX1|final_stage_repaired|repair_rows|pgrep|all done|HPAC|Row-1|v2_two_arm" .omx/research/CANONICAL_RESEARCH_INDEX* .omx/research/sub015_DAG_* .omx/state/*` | Live board still orders ET4 evaluate -> Row-1 -> HPAC; DAG/index reinforce external PR130 separation and byte-closed row authority. | Final result checklist reads durable JSON fields and axis labels, not projection prose or rounded stdout. |
| Canonical equations registry | `.venv/bin/python tools/list_canonical_equations.py --json | rg "PR130|lift|et4|mx1|hb1|HPAC|baseline|score|axis|evaluate|authority|MPS|CPU|CUDA"` | No PR130-lift-specific equation superseded the score formula or authority split. Authority surfaces reinforce advisory-vs-contest separation and recomputed score from components. | Kept `evaluate.recomputed_score` and `score_axis` as the consumption contract. |
| ET4 live artifacts | Repair script, ET4 runner, driver, rows ledger, shard logs, summary JSON | Found the `set -e` rc-capture bug, current incomplete ledger, append-only row writer, disjoint shard ranges, and no final-stage receipt. | Added RR5-F1 and refused CLEAN. |
| HB1 live artifacts | Driver, `driver.log`, `RESUME_CAVEAT.md`, `BYTE_RACE_TABLE.md`, checkpoint mtime | Found epoch-8 latest checkpoint; joint estimate descending but token/bpp not monotone; no Stage 3/4 harvest receipts yet. | Narrowed HPAC statement to estimated joint bytes only. |
| MX1 Row-1 artifacts | Source launcher, dedicated launch artifact, RR4 v2 dry artifact, cache/input files | Verified source-level v2 two-arm ticket and current cache/init hashes. Found the dedicated older `launch_ticket.json` still nested as schema v1 despite four arm keys, so fire-time regeneration is required. | Added explicit FIRE_CHECKLIST rather than treating any old JSON/prose copy as sufficient. |

## Verification Commands

| command | result |
|---|---|
| `bash -n /Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_repair_rows_and_final.sh` | `bash_n_rc=0` |
| `bash -c 'set -e; pgrep -f "definitely_no_such_process_for_rr5_guard_check_20260806" >/dev/null; prc=$?; echo would_proceed_rc=$prc'; echo outer_rc=$?` | On this host, `pgrep` could not enumerate processes and exited before `prc=$?`; outer rc was `3`. This proves rc>=2 is still not classified by the script. Bash semantics also make rc=1 exit before the intended proceed branch. |
| ET4 rows counter over live JSONL | Snapshot: 721 rows, 570 unique pairs, 151 duplicate extras, 30 missing pairs, sha256 `107cf7c61e90d0aa943d34ac5a6952b5c6dbfefb6597d4fed08a330130725959`. |
| `cat /Volumes/VertigoDataTier/pact/ddm_et4_20260806/shards_rc.txt` | `shards rc: 1 0 0`; final stage skipped. |
| Import baseline constants | `BASELINE_S=0.7534578126155775`, `BASELINE_BYTES=357837`, `BASELINE_D_SEG=0.004305419922`, `BASELINE_D_POSE=0.000716508925`, parent sha `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`. |
| `shasum -a 256` over MX1 Row-1 inputs | tq1c cache `11fd89016ab33a4b221975dafac0a572d66c372db1accf62284a3e81acddcc54`; GT cache `286fe40a2a29aa6950684f43229fce3a4a284ac7ffc65040e7e18953b95787d4`; init checkpoint `1549607db224ea2c4681738dbcc80d2ba9dd453de72db1cf60309985d0602eaf`. |
| HPAC driver-log parser | Epochs `[0,1,2,4,6,8]`; estimated joint bytes nonincreasing; token bytes and bpp not nonincreasing; latest checkpoint mtime `2026-08-06T21:37:05Z`. |

## Follow-On Disposition

| follow-on | disposition |
|---|---|
| ET4 repair/final | QUEUED-WITH-A-FIRE-ORDER: fix pgrep rc capture under `set -e`, run rc controls, wait for 600 unique rows and no live writer, then backup+sha, dedupe keep-last, build archive, inflate, and evaluate. |
| Row-1 ARM-CAP/ARM-VEH | QUEUED-WITH-A-FIRE-ORDER: fire only after ET4 boundary; regenerate v2 ticket under real run root; reverify cache/init hashes and Metal availability; run both n32 arms before any n120. |
| HB1 harvest | QUEUED-WITH-A-FIRE-ORDER: parse stage2/stage3/stage4 rc and exact encode/decode reports for both payloads; ignore `all done` as a byte-row receipt. |

## Boundaries

This review did not run `upstream/evaluate.py`, did not run an n600 scorer job, did not build or evaluate an archive, did not mutate the ET4 rows ledger, did not edit live driver scripts, and did not touch the staged git index. All absence statements are bounded to the files and commands listed above.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
