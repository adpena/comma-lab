# ddm_rr4 Round 4 PR130 Lift Wave Adversarial Review

Date: 2026-08-06
Reviewer: ddm_rr4
Axis: scorer-free source/artifact review
Score claim: false
Promotion eligible: false
Tokens: [no-triality] [p0-ledger-ok]

## Verdict

NOT-CLEAN.

RR3-F1 is VERIFIED at source and regenerated-artifact level. RR3-F2 is NOT-VERIFIED because the live-worker guard in the repair script fails open when `pgrep` cannot inspect the process table; the ET4 row ledger advanced during this review, proving a worker was still appending while that guard could not see it. A second fresh HB1 harvest-safety finding is queued: the host driver can exit 0 and print `all done` even if Stage 3/4 pack/encode/decode return nonzero, so the done receipt must not be consumed without parsing the per-stage rc lines and required reports.

No scorer job was run. No live driver script or live rows ledger was edited. Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.

## Round-3 Fix Verification

| item | verdict | evidence | boundary |
|---|---|---|---|
| RR3-F1 MX1 source emitter | VERIFIED | `experiments/ddm_mx1_pr130_semantic_renderer.py:524-572` now emits schema `ddm_mx1_row1_launch_ticket.v2_two_arm`, four arm argv keys, and `arm_selection_rule`; no bare `argv_n32` / `argv_n120` keys. | MLX training telemetry remains research-signal; n120 still waits for CPU-torch n32 verdict selection. |
| RR3-F1 regenerated probe artifact | VERIFIED | Dry probe wrote `/Volumes/VertigoDataTier/pact/ddm_rr4_20260806/mx1_probe_emit_result.json` sha256 `e9c74ee31e428aed755c0d7e52cd530f16939829ba5a25acfd4ffed44d245ee4`. Parsed ticket had exactly `argv_n32_arm_cap`, `argv_n32_arm_veh`, `argv_n120_arm_cap`, `argv_n120_arm_veh`; argparse AST check found zero unknown flags. | Probe mode only; local MLX device still unavailable. |
| RR3-F1 arm routing | VERIFIED | Generated cap arms use input cache = target cache = `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt`; veh arms use input cache `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt` and target cache GT. | Source defaults and dry artifact agree. |
| RR3-F2 dedupe transform | PARTIAL-VERIFIED | Duplicate rows are content-identical except `elapsed_s` on the current ledger: 151 duplicate pairs, 0 non-elapsed differing pairs. `experiments/ddm_et4_solve_within_cvp_n600.py:705-720` reloads JSONL under `--resume`, builds `done` from pair ids, and would skip all pair work after a complete 600-unique dedupe. `aggregate_rows` uses only loaded `rows`; no `.bak` consumer was found. | This verifies transform semantics only, not safe execution. |
| RR3-F2 repair script safety | NOT-VERIFIED | See RR4-F1. The script can proceed past a failed process-table query. | Do not run the script as-is on this host. |

## Findings

| id | severity | surface | verdict_scope | status | finding | required action |
|---|---|---|---|---|---|---|
| RR4-F1 | CRITICAL | ET4 repair script live-ledger safety | INSTANCE: `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_repair_rows_and_final.sh` | QUEUED to MAIN | The pgrep guard at lines 13-15 treats every nonzero `pgrep` result as "no worker." On this host `pgrep -f ddm_et4_solve_within_cvp_n600` returned rc=3 with `sysmon request failed... Cannot get process list`; stdout was empty. During the review the ledger advanced from 713 rows / 562 unique to 715 rows / 564 unique, last pair 169, so a shard was live while pgrep could not see it. At completion, this script could rewrite the rows ledger while a worker is still appending. | Replace the guard with fail-closed semantics: rc=0 means refuse-live, rc=1 can mean no match only if process enumeration succeeded, rc>=2 refuses. Also require a durable driver completion condition (`shards_rc.txt` all zero or operator-confirmed no live writer) and a stable rows-file interval before rewriting. |
| RR4-F2 | MEDIUM | HB1 host-chain harvest safety | INSTANCE: `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/hpac_our_labels_driver.sh` | QUEUED to MAIN | The driver separates `tq1c` and `gt` state correctly, but Stage 3/4 failures are logged and then ignored: lines 45-53 log pack rc, lines 55-68 log encode/decode rc, and line 71 prints `all done` regardless. The launcher done receipt points to the driver process; if the driver exits 0 after a Stage 3/4 rc!=0, a harvest that trusts only process rc / `all done` can accept a missing pack, missing token stream, or failed `--require-exact` decode. | Harvest must require per-payload `stage3 rc=0`, `stage4 encode rc=0`, `stage4 decode rc=0`, payload-prefixed reports, per-payload `artifacts/$payload/tokens.bin`, and `--require-exact` decode PASS before closing the HPAC race. Future driver edit should fail nonzero on any Stage 3/4 failure or emit a structured rc summary. |

## Fresh-Hunt Checks

| surface | result | evidence |
|---|---|---|
| ET4 final evaluate invocation | CLEAN | `build_archive()` calls `run_upstream_evaluate(... device=args.eval_device, batch_size=args.eval_batch_size, num_threads=args.eval_threads)` at `experiments/ddm_et4_solve_within_cvp_n600.py:665-680`; defaults are `--eval-device cpu`, `--eval-batch-size 16`, `--eval-threads 2`. `src/tac/submission_chain.py:956-1034` labels CPU on this host as `[macOS-CPU advisory]`, refuses MPS, parses components, and recomputes S from components plus archive bytes. |
| ET4 baseline arithmetic | CLEAN | `BASELINE_S = 0.7534578126155775` and `BASELINE_BYTES = 357_837` come from `experiments/ddm_et2_projected_phase_field.py:60-63`; ET4 aggregate computes `dS_vs_named_baseline = measured_s - BASELINE_S` at `experiments/ddm_et4_solve_within_cvp_n600.py:401-427`. No wrong `0.7539807296911207` baseline consumer was found in ET4 final-stage code. |
| ET4 overlay frame contract | CLEAN after re-derivation | Runtime patches frame_1 at `experiments/ddm_et4_overlay_inflate_runner.py:46-48` and preserves parent frame_0 by calling parent `f0` on the base parent `f1` at lines 50-52. This matches the per-pair scorer path, which measures `score_pair(dec_f0=dec[0], cam_f1=cam_cvp)` at `experiments/ddm_et4_solve_within_cvp_n600.py:830-838`. I initially challenged this as suspicious, then re-derived the scorer contract and did not change it. |
| ET4 aggregate double-weighting | CLEAN conditional on safe dedupe | `aggregate_rows()` double-weights raw duplicate rows by construction (`len(rows)`, sums, and means at `experiments/ddm_et4_solve_within_cvp_n600.py:360-438`). A completed 600-unique keep-last rewrite cures the aggregate and `build_archive()` length gate because both consume `load_jsonl(args.rows_path)` after repair. The remaining blocker is safe timing of the rewrite. |
| HB1 state bleed between payloads | CLEAN for named bleed class | Driver uses per-payload checkpoint and artifact roots (`checkpoints/$payload`, `artifacts/$payload`) at `/Volumes/.../hpac_our_labels_driver.sh:15-22`, payload-specific caches at lines 27-39 and 59-67, per-payload `tokens.bin` under `$ART`, and payload-prefixed encode/decode reports at lines 55-58. No tq1c/gt token overwrite or report-name reuse was found. |
| Campaign #984 sequence | CLEAN ordering, blocked by findings above | `.omx/state/main_hot_state.md:37-42` orders ET4 evaluate before Row-1 ARM-CAP/ARM-VEH, HPAC race, Row-2 pose race, composition, export, byte-close, n600 evaluate. I did not find a dependency inversion. RR4-F1 blocks ET4 repair/final unless fixed; RR4-F2 constrains HB1 harvest. |

## Recall Evidence

| source searched | query / command | found beyond charter seeds | changed plan |
|---|---|---|---|
| Prior review receipts | `rg "RR[123]-F|argv_n32|argv_n120|dedupe|keep-last|repair" .omx/research/ddm_rr2_20260806 .omx/research/ddm_rr3_20260806 ...` | RR3 already isolated source-emitter durability and live duplicate rows; RR2 had HB1 resume caveat but not Stage 3/4 rc consumption. | Verified exact prior fixes instead of re-opening settled MX2/HB1 resume points. |
| Full corpus | `rg "PR130|pr130|lift wave|ddm_rr[1234]|et4|mx1|hb1|aggregate_rows|load_jsonl|arm_selection_rule|payload=gt" .omx/research .omx/state docs experiments src tools` | Found live hot-state Campaign #984 order, ET4 first-8 receipt, HB1 receipt/caveat, and the source/runtime files that actually consume rows. | Added ET4 final-stage and HB1 harvest lenses, not just RR3 fix checks. |
| Canonical equations registry | `.venv/bin/python tools/list_canonical_equations.py --json | rg "PR130|pr130|lift|et4|mx1|hb1|aggregate|payload|baseline|score|axis|evaluate"` | No PR130 lift-specific equation changed the plan; score-axis and pointer rules reinforced component recomputation and advisory labels. | Kept scorer-free boundary and recomputed/axis-label focus. |
| Live board | `.omx/state/main_hot_state.md` | Current own-vehicle baseline is `0.7534578126155775 @ 357,837 B`; ET4 owns scorer slot; HPAC follows; Row-1 waits for ET4 boundary. | Treated ET4 repair/final as the critical gate and did not run any scorer job. |
| Live ET4 artifacts | `et4_repair_rows_and_final.sh`, `et4_n600_driver.sh`, current rows JSONL, shard logs | Current ledger was still live and advancing; duplicate rows content-identical except `elapsed_s`; repair script lacks fail-closed pgrep handling. | RR3-F2 moved from likely verified to NOT-VERIFIED. |
| Live HB1 artifacts | `hpac_our_labels_driver.sh`, `driver.log`, launch manifest, receipt/caveat | Per-payload roots are clean; process rc / `all done` is insufficient because Stage 3/4 rc is not propagated. | Added RR4-F2 harvest-safety finding. |

## Verification Commands

| command | result |
|---|---|
| `.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode probe --run-dir /Volumes/VertigoDataTier/pact/ddm_rr4_20260806/mx1_probe_emit --out /Volumes/VertigoDataTier/pact/ddm_rr4_20260806/mx1_probe_emit_result.json` | passed; emitted v2 two-arm ticket; ignored atexit Metal-device warning only. |
| argparse AST check over generated MX1 argv lists | passed; zero unknown flags across all four arm argv lists. |
| `pgrep -f "ddm_et4_solve_within_cvp_n600"` | rc=3, stderr sha256 `ab9ec9da6f215c2a2a8c372292779fc7d710e2bc06ab2f5b3cfc04257a439e0f`; proves process enumeration failure is distinguishable from no workers but repair script does not distinguish it. |
| ET4 rows snapshot at 2026-08-06T21:47:11Z | rows `715`, unique pairs `564`, missing `36`, last pair `169`, sha256 `fe137449524a79d6331c449610de3a07c240151f7822245f672f93f912904471`. |
| duplicate-content audit on ET4 rows | duplicate pairs `151`, non-`elapsed_s` differing duplicate pairs `0`, duplicate range `328..599`. |
| `.venv/bin/python -m pytest experiments/tests/test_ddm_et4_overlay_codec.py -q` | `2 passed in 0.15s`; no code edits retained from this check. |

## Follow-On Disposition

| follow-on | disposition |
|---|---|
| ET4 repair/final | QUEUED-WITH-A-FIRE-ORDER: fix fail-closed liveness guard first, then only after unique=600 and no live writer, backup+sha rows, dedupe keep-last, rerun final stage. |
| MX1 Row-1 fire | FOLDED into existing fire order: source emitter and dry artifact are verified; fire still waits for ET4 boundary and CPU-torch verdict protocol. |
| HB1 harvest | QUEUED-WITH-A-FIRE-ORDER: parse per-stage rc and exact decode reports for both `tq1c` and `gt`; do not trust `all done` or wrapper rc alone. |
| Campaign #984 ordering | FOLDED: no order inversion found; respect the existing sequence after ET4/HB1 blockers are cleared. |

## Boundaries

This review did not run `upstream/evaluate.py`, did not run a scorer job, did not mutate the ET4 live rows ledger, and did not edit live driver scripts. It did not produce a byte-closed archive or move any pointer.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
