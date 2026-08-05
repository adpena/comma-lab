# ddm_rr1 recursive adversarial review receipt

Status: REVIEW arm. Cycle 1 covered rounds 2-7 fresh-eyes over the JD3 v3 chain and DY1 scope-law resolver; cycle 2 rounds 2-5 reviewed the JD4 landing wave, n600 both-bases probe variant, sealed JD4 ticket, fire record, plateau policy, endpoint exit paths, and live-window telemetry claims. I ran no scorer, no Metal/MLX launch, no archive build, and no exact evaluation. Live full-v3/JD4 dirs/files were read-only.

Axis: source/receipt/telemetry review plus derived arithmetic. Any score arithmetic below is non-promotable `[macOS-CPU/MLX advisory]` or derived from existing receipts; `score_claim=false`.

## RECALL EVIDENCE

| Surface | Query / artifact | Result beyond charter seeds | Plan change |
|---|---|---|---|
| Pact memory registry | `rg -n "rr1|jd3|dy1|scope-law|chain_both_bases|0.0037233" /Users/adpena/.codex/memories/MEMORY.md` | No rr1-specific prior memory found; unrelated #899/#904 memory only. | Kept review scoped to live artifacts, not memory conclusions. |
| Governing docs | `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | Hot state says JD3 full window owns scorer slot and the live dirs are sacred; recursive review requires assumption-challenge and measured-runnability axes. | No scorer work; all findings are review-only. |
| Full corpus recall | targeted `rg` over `.omx/research`, `.omx/state`, docs/reports for `JD3`, `full_v3`, `dy1`, `chain_both_bases`, `ep1349`, `0.0037233` | Found cx1 fire-gate receipt and dy1 build receipt as additional context. Chain-sweep best-live caveat lives inside the JSON finalize note; no uncaveated downstream citation found in searched md/json surfaces. | Added cx1 gate verification and DY1 pre-merge review. |
| Round-3 recall refresh | targeted `rg` over `.omx/research`, `.omx/state`, docs/reports for `JD3`, `full_v3`, `scope-law`, `inertness`, `chain_both_bases`, `0.0037233`, `ru_maxrss`, `peak RSS`, `VRAM` | Found `ddm_gc20_20260805/RECEIPT.md`, which consumes the two-smoke matrix and explicitly says the live full-v3 lane must not be mutated. The only current peak-memory rows found were unrelated older receipts; JD3 manifests/telemetry still do not record peak RSS/VRAM. | Raised the custody severity of the full-window out-dir alias; kept measured-runnability downgraded to executed-without-peak-memory. |
| Round-4 recall refresh | targeted `rg` for `full_v3`, `entry_cont`, `stage_ema`, `0.996667`, `0.999555`, `jd1_stage_ema_reanchor`, `scope_law`, and `0.0037233`; inspected the full-window launch manifest, ticket, top-level mixed telemetry, smoke snapshot receipt, TP1, GC20, and the EMA evaluator source. | Found a new live-window control-transfer defect: the full continuation resumes from the entry smoke final checkpoint with `stage_ema_reanchored=true` and keeps the 8-epoch smoke decay `0.9966666667` (`U=1200`) even though the 60-epoch full window would derive `0.9995555556` (`U=9000`) if re-anchored at resume. | Raised RR1-R4-F1 as endpoint-blocking for consuming the full continuation as a full-window stage-scoped EMA run. |
| Round-5 recall refresh | targeted current-surface `rg` over `.omx/research`, `.omx/state`, docs, and reports for `full-window EMA`, `EMA-cured`, `smoke-EMA continuation`, `stage_ema_reanchored`, `jd1_realized_hold`, `scope_law`, `inertness`, and `0.0037233`; re-read TP1, main hot state, DY1 clone `a9eac92166`, mixed telemetry, live manifest, tickets, `SCOPE_MARKER.md`, and preserved scratch probe variants. | No missed current consumer of the wrong full-window EMA label was found in the searched decision surfaces. The relabel/gating language is present in TP1 and main hot state; DY1 and regenerator production fixes remain unimplemented and correctly blocked. Other carried resume-state fields are classified below rather than silently treated as new full-window derivations. | Round 5 produced zero new findings; clean-pass counter advances to 1/3. |
| Round-6 recall refresh | endpoint-path lens over `jd3_full_v3_entry_cont.done`, the r2 launch manifest, the fired full ticket, `tools/launch_detached_process.py`, `experiments/train_tr1_partition_renderer_mlx.py`, the mixed top-level receipt/telemetry/checkpoints, `SCOPE_MARKER.md`, TP1, main hot state, `MAIN_ADDENDUM_OPTIMAL_CONVERSION.md`, the old JD1 ticket, and targeted `rg` for `recursive_encode_pass_loop` / `next_resume_from_template` / `tr1_window_receipt.json`. | The live full-v3 window was still unharvested in this sandbox: no `.done` marker, telemetry/log updating at 2026-08-05 15:06 CDT through epoch 1395, and `ps` blocked by sandbox. The terminal receipt is written only on trainer exit, so the stale smoke `tr1_window_receipt.json` is covered by RR1-R3-F1 as long as consumers wait for `.done`. New gap: the fired full ticket's recursive continuation fire-order still points at the old JD1 pass directory/template, not the actual JD3 continuation out-dir. | Raised RR1-R6-F1; clean-pass counter resets to 0/3. |
| Canonical equations | `tools/list_canonical_equations.py --json`, plus source grep for `ema_decay_run_geometry_v1` and `src/tac/canonical_equations/evaluators.py` | EMA law is executable; JD3 stage EMA derivation is `d = 1 - 2/(0.5*1200) = 0.9966666667`. | Re-derived EMA seed-retention rather than trusting the receipt wording. |
| Source and receipts | TP1 receipt, cx1 receipt, JD3 smoke telemetry/verdicts, fired ticket, launcher manifests, DY1 clone at `a9eac92166` | Rounds 2-4 found artifact/status and enforcement gaps; round 5 found no new gap in the searched scope. | Counter is now 1/3; fixes remain recommended unless explicitly marked implemented. |
| Source and receipts, round 6 update | Fired full ticket plus live continuation manifest/telemetry and trainer receipt-write source | Round 6 found one new follow-on fire-order defect in the machine-readable recursive loop metadata; no completed endpoint was consumed. | Counter is now 0/3; next clean pass is round 7. |
| Round-7 recall refresh | Re-read `NEXT_IF_RESUMED.md`, TP1 round-6 disposition, the old JD1 ticket, all three regenerated JD3 tickets, `experiments/ddm_jd1_ticket_regenerate.py`, the r2 launch manifest, `tools/launch_detached_process.py`, live telemetry, and current `.done` absence; targeted `rg` for `recursive_encode_pass_loop`, `pass0_input`, `next_resume_from_template`, `winner_dir`, `smoke_epochs`, `full_confirm`, and `full_v3_entry_cont`; ran a structured `argv` vs `levers[*].overrides` diff. | TP1's consumption contract voids the stale recursive resume template for the fired ticket, and `launch_detached_process.py` consumes only final argv. New gap: the regenerator updates argv/child paths but leaves the inherited `levers[*].overrides` value ledger stale across all JD3 regenerated tickets. | Raised RR1-R7-F1; counter remains 0/3. |
| Cycle-2 round-2 recall refresh | Read `rr1_prompt.md` cycle-2 addendum, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, craft handoff, hot state, TP1 cycle-2 round 1, JD4 receipt, sealed JD4 ticket, `experiments/ddm_jd1_ticket_regenerate.py`, trainer force-reanchor source, the n600 probe variant, launch manifest/log, and completed `jd3_endpoint_n600_both_bases_verdict.json`; diffed probe variant vs committed source and checked ticket lever overrides. | JD4 ticket repairs are real in the emitted artifact (`0` lever mismatches; recursive template under child out-dir; forced reanchor flag present). New gap: the completed n600 probe receipt has 600 pair ids and both live/EMA rows, but its `axis` still says `36 gd1-designed gate pairs`. | Raised RR1-C2-R2-F1; cycle-2 counter resets to `0/3`. |
| Cycle-2 round-3 recall refresh | Re-read `rr1_prompt.md` round-3 addendum, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md` recursive-review lines, craft handoff, hot state, TP1 fire record, JD4 receipt/NEXT, sealed JD4 ticket, JD4 launch manifest, first JD4 telemetry, `tools/launch_detached_process.py`, fixed committed endpoint instrument, stale SSD n600 variant, correction manifest, canonical EMA evaluator, regenerator source/tests, and targeted full-corpus `rg` over `.omx/research`, `.omx/state`, docs, reports, tools, experiments, and src for `jd4`, `jd3_endpoint_n600`, `RR1-C2-R2-F1`, `51c64222`, `6e4a6e24`, and force-reanchor terms. | The committed endpoint class fix is real; the stale SSD variant remains append-only/correction-only and is explicitly barred by TP1 for future probes. JD4 ticket/launch custody matches the fire record, the R4 cure is verified in live telemetry, and I found no consumer in searched current decision surfaces that routes future endpoint probes through the stale SSD source. Process liveness remained unmeasured because `kill -0` is sandbox-blocked and `.done` is absent. | No new finding; cycle-2 clean counter advances to `1/3`. |
| Cycle-2 round-4 recall refresh | Re-read `rr1_prompt.md` round-4 addendum, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, craft handoff, hot state, TP1 plateau policy/fire record, JD4 receipt/NEXT, sealed JD4 ticket, JD4 launch manifest, live telemetry, trainer exit/receipt source, launcher done-receipt source, fixed endpoint instrument, stale SSD n600 variant/correction manifest, and targeted `rg` over `.omx/research`, `.omx/state`, docs, reports, tools, experiments, and src for `jd4`, `plateau policy`, `1.5e-3`, `0.00144`, `max_wall_minutes`, `stop_reason`, `tr1_window_receipt`, and `jd4_cont_ep1406`. | The JD4 landing fixes remained real; the live window had gates through ep1414 and no `.done`/terminal receipt. New low-severity record gap: TP1's Case A pose satisficing shorthand treated `d_pose <= ~1.5e-3` as `<=0.12`, but exact arithmetic gives `sqrt(10*0.0015)=0.122474`; the hard `<=0.12` gate is `d_pose <=0.00144`. | Raised RR1-C2-R4-F1 and appended the TP1 correction note; cycle-2 counter resets to `0/3`. |
| Cycle-2 round-5 recall refresh | Re-read `rr1_prompt.md` round-5 addendum, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, craft handoff, hot state, TP1 Case A and fire-record sections, JD4 launch manifest, JD4 telemetry through ep1429, trainer A1/refuse source, canonical equations registry for EMA terms, and targeted `rg` over TP1 plus current decision surfaces for `1.5e-3`, `0.00144`, `0.12`, `jd4_cont_ep1406`, `A1_REALIZATION_GAP_ALARM`, and force-reanchor terms. | The R4 arithmetic correction is present at both TP1 gate sites and no third TP1 gate site still carries the loose inequality. The R4 cure live telemetry matches the fire record: forced_start 1406 vs legacy 1407, U=18000, decay 0.9997777777777778, warmup 9000 updates. Derived tau is 29.9967 epochs, so endpoint ep1526 is about 2.0002 tau after warmup end ep1466. A single ep1424 A1 alarm recovered at ep1429 and no typed exit/refuse row exists. | Cycle-2 round 5 found zero new findings; clean-pass counter advances to `1/3`. Carry the ep1424 alarm as a watch note only, not an endpoint verdict. |
| Cycle-2 round-6 recall refresh | Re-read `rr1_prompt.md` round-6 addendum, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, craft handoff, hot state, TP1 MAIN-R5X block and plateau policy, trainer source at the force-reanchor predicate/resume gate basis/A1 gate sites, JD4 ticket and launch manifest, JD4 telemetry through ep1439, `current_focus.md`, OP2 gate-basis receipt, JD3 receipt, `src/tac/ddm_costate_organ.py`, canonical equations registry, and targeted `rg` over `.omx/research`, `.omx/state`, docs, reports, tools, experiments, and src for `MAIN-R5X`, `resume.*warm-shadow`, `read LIVE`, `reads LIVE`, `jd1_live_basis_gate`, `gate_basis`, `stage_ema_reanchored`, and force-reanchor terms. | The MAIN-R5X correction matches source: resumed runs start with `global_step=ema_warmup_updates`, forced reanchor resets EMA to live and bumps `global_step`, so normal `a1_gate` rows are `ema_shadow` from the first gate. Telemetry confirms paired rows through ep1439: `jd1_live_basis_gate` uses `live_weights`, while `a1_gate` uses `ema_shadow`; JD4 ticket/manifest carry `--jd1-live-gate-telemetry`. The TP1 plateau policy still intentionally uses live-basis slopes, but that is executable via the live channel and does not mean the normal A1 gate is live. No uncorrected "read LIVE through warmup" consumer was found in TP1 or hot state. | Cycle-2 round 6 found zero new findings; clean-pass counter advances to `1/3` after MAIN-R5X reset. Future endpoint/plateau consumers must keep `jd1_live_basis_gate` (live-slope policy) separate from `a1_gate` (EMA-shadow A1/hold gate). |

## ROUND 2 FINDINGS

Counter state after this round: **0/3 clean passes**. This round found new issues, so it is not a clean pass.

| ID | Severity | Finding | Evidence | Required disposition |
|---|---|---|---|---|
| RR1-F1 | MEDIUM | DY1's inertness control is fake-passable if merged as-is. `inertness_violations()` exists and is unit-tested, but no production caller in `tools/launch_tr1_run.py` or the trainer consumes it. A ticket can declare scope laws and hash-bind them, then still launch without any fail-closed check that every declared law resolved. | `src/tac/witness_dsl/scope_laws.py:447-468` defines the check; `rg inertness_violations` finds only tests. Launcher validates schema/hash at `tools/launch_tr1_run.py:169-178`; it does not check runtime resolution rows. | Before merge or next scope-law launch, add a production verification point that compares ticket `scope_laws` to checkpoint/receipt `resolved_scope_laws` and refuses selection on any declared-but-unresolved law. |
| RR1-F2 | MEDIUM | Round 1's measured-runnability axis overclaims memory evidence. The smokes and live window executed, but I found no peak RSS/VRAM/ru_maxrss field in launcher manifests, run logs, or telemetry. The phrase "peak memory unremarkable" is therefore unverified. | `rg "peak|RSS|VRAM|ru_maxrss|memory" ...` found no measurement rows in JD3 logs/telemetry. `ps` was blocked by sandbox, and in any case would not recover peak usage. | Downgrade to "runnability executed; peak memory not recorded." Next governed run should emit peak RSS/VRAM in the manifest or receipt if the review counter consumes measured-runnability. |
| RR1-F3 | LOW | `jd3_smoke_entry_pose_verdict.json` is internally stale: it contains all four ckpt rows used by adjudication, but `status` remains `"running"` and it lacks final summary/deltas. | `jq` shows `status=running`, `ckpt_count=4`, no `deltas`/`gate_positive_control`; final smoke verdict is `status=complete`. | Correct or supersede the entry verdict status in a doc/metadata-only follow-up. Decision arithmetic remains reconstructable and unchanged. |
| RR1-F4 | LOW | The partial-S winner is correct, but the "omit rate term because identical payloads" premise is not literally true against telemetry counted bytes. Entry endpoint bytes and refuse-final endpoint bytes differ by 1,193 B on the gate telemetry. | Entry ep1344 `total_counted_bytes=306891`; refuse-final ep1363 `305698`; rate delta = `25*1193/37545489 = 0.000794 S`, far below the `0.125096 S` partial-S gap. | Keep the winner; phrase this as "rate delta is negligible for this pick," not "identical payloads." |
| RR1-F5 | LOW | Scratch-variant instrument custody is incomplete. The committed instrument physics block is inspectable, but I did not find the scratch variant source or command that proves the JD3 pose verdicts changed only `CKPTS`/repo path/summary stripping. | `rg ddm_jd1_endpoint_verdict|jd3_smoke_*pose` found the committed source, output JSONs, and endpoint run logs, but no preserved scratch script. The JSONs share schema/pairs/semantics with the committed instrument. | Treat the claim as not fully source-custodied. Preserve scratch scripts or a command manifest for future config-only probe variants. |

## ROUND 3 FINDINGS

Counter state after this round: **0/3 clean passes**. This round found new issues, so it is not a clean pass.

| ID | Severity | Finding | Evidence | Required disposition |
|---|---|---|---|---|
| RR1-R3-F1 | MEDIUM | The full-v3 continuation writes into the entry smoke trainer out-dir, making the post-fire `tr1_jd3_v3_smoke_entry_ep1336` directory mixed-scope. The smoke adjudication state is preserved by `smoke_snapshot_ep1344/`, so the winner is not invalidated, but consumers that read the top-level smoke telemetry/checkpoints after fire can silently consume full-window continuation rows as "smoke" evidence. | `full_v3_entry_cont_mainlaunch_r2/launch_manifest.json` has launcher `output_dir=/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/full_v3_entry_cont_mainlaunch_r2`, but trainer `--out-dir=/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/tr1_jd3_v3_smoke_entry_ep1336`. The full ticket differs from the entry smoke ticket only in `--epochs`, `--resume-from`, and `--max-wall-minutes`; it leaves `--out-dir` unchanged. Top-level entry telemetry now has gates through ep1369, while `smoke_snapshot_ep1344/telemetry_smoke.jsonl` preserves only ep1339/1344 smoke rows. | Endpoint harvest must explicitly treat the top-level entry directory as `entry_smoke_plus_full_continuation`, use `smoke_snapshot_ep1344/` for the two-smoke adjudication evidence, and create a unique trainer out-dir for any future full continuation or add a manifest field declaring same-dir continuation intentionally. |
| RR1-R3-F2 | LOW | Smoke-start labels are source/resume labels, not saved checkpoint epochs, and that is not explicit in the verdict JSONs. This can shift slope/lineage arithmetic by one epoch if a reader treats `smoke_start_ep1336` or `smoke_start_ep1355` as the saved checkpoint metadata epoch. | `stage_joint_pose_finish_entry.npz` in the entry smoke has `meta::epoch=[1337]`, while the parent source checkpoint has `meta::epoch=[1336]`. The refuse-final smoke entry has `meta::epoch=[1356]`, while the parent final source checkpoint has `meta::epoch=[1355]`; the pose verdict tag is `smoke_start_ep1355`. | Future tables should split `source_checkpoint_epoch` from `saved_reanchor_checkpoint_epoch`; do not use the `smoke_start_*` tags alone for epoch-delta or slope derivations. |

## ROUND 4 FINDINGS

Counter state after this round: **0/3 clean passes**. This round found a new issue, so it is not a clean pass.

| ID | Severity | Finding | Evidence | Required disposition |
|---|---|---|---|---|
| RR1-R4-F1 | CRITICAL | The 60-epoch full-v3 continuation did not re-derive/re-anchor the active EMA for the full window. It resumes from the entry smoke final checkpoint with `stage_ema_reanchored=true`, so the trainer carries the smoke-window decay `0.9966666667` (`U=1200`, warmup 601 updates) into the full continuation. The full ticket's own epoch geometry (`--epochs 1406`, resume event epoch 1346, 150 steps/epoch) would derive `U=9000`, decay `0.9995555556`, warmup 4500 updates. The two-smoke adjudication remains valid; the full continuation is endpoint-blocked as evidence for a "full-window stage-scoped EMA" law until relabeled or re-run/re-anchored. | `jd3_ticket_v3_full_entry_cont.json` and the r2 launch manifest show `--epochs 1406` and `--resume-from .../tr1_jd3_v3_smoke_entry_ep1336/checkpoints/stage_joint_pose_finish_final.npz`. Top-level telemetry has start indices `[0,30]`; the full resume row at epoch 1346 reports `stage_ema_reanchored=true`, `active_ema_decay=0.9966666667`, provenance `U=1200`. The only `jd1_stage_ema_reanchor` event in the mixed telemetry is the smoke event at epoch 1337. `eval_ema_decay_run_geometry` gives `d = 1 - 2/(0.5*U)`, so a 60-epoch full window at 150 steps/epoch would use `0.9995555556`. The top-level `tr1_window_receipt.json` still records `stage_ema_reanchored_epoch=1337`, active decay `0.9966666667`, and `ema_warmup_updates=601`. | Do not byte-close, promote, or endpoint-route the full continuation as a full-window EMA-cured v3 result. If the owner harvests it, label it `smoke-EMA continuation` and measure live/EMA basis explicitly; otherwise regenerate into a unique out-dir or force a fresh full-window reanchor at continuation resume. Add a production inertness/refuse check that compares declared full-window EMA scope to the resolved active EMA row before checkpoint selection. |

## ROUND 5 FINDINGS

Counter state after this round: **1/3 clean passes**. This round found zero new findings; it re-verified the round-4 disposition boundary and checked the broader resume-state class.

| ID | Severity | Finding | Evidence | Required disposition |
|---|---|---|---|---|
| RR1-R5-clean | NONE | No new blocker, regression, or missed consumer was found in the searched round-5 scope. | The current hot state and TP1 receipt label the live continuation as `smoke-EMA continuation`, block #961/DY1 on production inertness and geometry-keyed EMA scope, and require endpoint adjudication to measure both live/EMA bases. The searched current md/json/jsonl decision surfaces did not contain an uncaveated full-window EMA-cured consumption of the continuation. | Continue the recursive review with counter `1/3`; do not treat DY1/regenerator production debts as fixed until code or launch gates actually enforce them. |

## ROUND 6 FINDINGS

Counter state after this round: **0/3 clean passes**. This round found a new issue, so it is not a clean pass.

| ID | Severity | Finding | Evidence | Required disposition |
|---|---|---|---|---|
| RR1-R6-F1 | MEDIUM | The fired full-v3 ticket carries a stale recursive continuation fire-order. If the current full window exits by `epochs_complete` or `max_wall_minutes` and someone follows the ticket's `recursive_encode_pass_loop`, the next pass template points to `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/tr1_joint_pose_finish_after_tp1_lane_on/checkpoints/stage_joint_pose_finish_final.npz`, while the actual fired JD3 continuation runs in `/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/tr1_jd3_v3_smoke_entry_ep1336`. This is not a current-run physics bug and not a completed-endpoint claim, but it is orphan-prone follow-on metadata: the machine-readable continuation route is neither FOLDED nor a valid fire-order for this lane. | `jq` on `jd3_ticket_v3_full_entry_cont.json` shows `child_out_dir=/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/tr1_jd3_v3_smoke_entry_ep1336` and `child_resume_from=.../tr1_jd3_v3_smoke_entry_ep1336/checkpoints/stage_joint_pose_finish_final.npz`, but `recursive_encode_pass_loop.continue_policy.next_resume_from_template=/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/tr1_joint_pose_finish_after_tp1_lane_on/checkpoints/stage_joint_pose_finish_final.npz`; the old `.omx/research/ddm_jd1_20260805/JD1_TICKET.json` has that same template, so this is inherited stale JD1 metadata. The r2 launch manifest confirms the actual full-v3 trainer out-dir/resume path. | Before any continuation or recursive pass from the full-v3 endpoint, replace or supersede the ticket's recursive loop with a JD3-specific fire-order: next resume source must be the actual completed full-v3 endpoint checkpoint (or explicitly FOLD continuation). Add a cheap preflight/refuse in the ticket regenerator or launch gate: if `recursive_encode_pass_loop.next_resume_from_template` is present, it must resolve under the ticket's actual `child_out_dir` or a declared new continuation out-dir, never an unrelated ancestor lane. |

## ROUND 7 FINDINGS

Counter state after this round: **0/3 clean passes**. This round found a new issue, so it is not a clean pass.

| ID | Severity | Finding | Evidence | Required disposition |
|---|---|---|---|---|
| RR1-R7-F1 | MEDIUM | The JD3 ticket regenerator leaves the inherited `levers[*].overrides` value ledger stale after changing final argv. The launched command uses the argv and is not invalidated by this finding, but the ticket's machine-readable provenance no longer binds the values it claims. This is the same inheritance genus at the value-ledger surface, not just path templates. | In `experiments/ddm_jd1_ticket_regenerate.py`, the v3 branch does `regen = dict(t)`, updates `argv`, `child_resume_from`, `child_out_dir`, and `regenerated_from`, but never rebuilds `levers`. Structured diff against the fired full ticket shows substantive mismatches: `tr1_window_ep1076_b4 --epochs` says `1076` while argv says `1406`; `--max-wall-minutes` says `130.0` while argv says `168`; `tr1_ema_decay_parent_tp1_pinned --ema-decay` says `0.9999436222692036` while argv says `0.999960019990005`; `tr1_jd1_joint_pose_finish --jd1-seg-hold-floor-source` says `checkpoint_tail_ep_loss` while argv says `last_pre_pose_epoch_loss`. The two smoke tickets have the same stale fields with argv epochs `1345`/`1364` and wall `23`. Boolean `--lane-guard` case-only differences were ignored. | Do not merge or consume future regenerator output as value-custodied until it either rebuilds `levers[*].overrides` from final argv or explicitly demotes/removes that ledger as non-authoritative. Add a cheap refuse check: for every flag declared in `levers[*].overrides`, the declared value must match the final argv value after all lane/geometry rewrites. The regenerator cure should validate the whole emitted ticket surface, not only `recursive_encode_pass_loop.next_resume_from_template`. Current fired ticket remains hash-custodied; supersede rather than mutate it. |

## CYCLE 2 ROUND 2 FINDINGS

Counter state after this round: **0/3 clean passes** for cycle 2. This round found a new issue, so it is not a clean pass.

| ID | Severity | Finding | Evidence | Required disposition |
|---|---|---|---|---|
| RR1-C2-R2-F1 | MEDIUM | The completed JD3 n600 both-bases probe receipt has the wrong denominator/axis label. The run measured all 600 pair ids and wrote both `endpoint_ep1405_live` and `endpoint_ep1405_ema`, but the receipt and source variant still say `[macOS-CPU frozen-scorer advisory] 36 gd1-designed gate pairs, NON-PROMOTABLE`. That is a false denominator label on the exact artifact MAIN is supposed to consume before firing JD4. | `jd3_endpoint_n600_both_bases.py` sets `GATE_PAIR_IDS = list(range(600))` at line 46, but line 101 still hardcodes the 36-pair axis string and line 61 still defaults `--out` to the old JD1 3-ckpt path. The r2 launch manifest did override `--out` to `jd3_endpoint_n600_both_bases_verdict.json`, so the path default did not corrupt this run. The completed JSON has `status=complete`, `pair_count=600`, `ckpt_keys=[endpoint_ep1405_ema, endpoint_ep1405_live]`, live d_seg `0.0071503364`, live d_pose `0.5740917290`, EMA d_seg `0.0057479858`, EMA d_pose `0.1288530915`, and the stale 36-pair axis string. | Superseding axis-only correction landed at `.omx/research/ddm_rr1_20260805/jd3_endpoint_n600_axis_correction.json`; raw SSD receipt bytes were not mutated. Before JD4 fire, cite that correction or patch the source receipt label, and patch/preserve the variant with a correct default output path and axis string before any rerun. Treat the numeric rows as existing `[macOS-CPU frozen-scorer advisory] n600 training-vehicle objective` evidence only with the correction; still not archive-byte-closed or contest authority. |

## CYCLE 2 ROUND 3 FINDINGS

Counter state after this round: **1/3 clean passes** for cycle 2. This round found zero new findings; it re-verified the round-2 class fix, JD4 fire custody, fire-gate adjudication, and live R4 cure evidence.

| ID | Severity | Finding | Evidence | Required disposition |
|---|---|---|---|---|
| RR1-C2-R3-clean | NONE | No new blocker, regression, or missed consumer was found in the searched cycle-2 round-3 scope. The only live boundary remains the already-recorded raw SSD n600 receipt label defect; future probes must derive from the fixed committed instrument, not from the stale SSD copy. | Committed `experiments/ddm_jd1_endpoint_verdict.py` derives `_axis_set_desc` from `GATE_PAIR_IDS` and self-labels `range(600)` as `all 600 pair ids (0..599)`. `git diff` on the fixed source files was empty; `git show 6e4a6e24fe` is the source class fix. The raw SSD variant `/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/jd3_endpoint_n600_both_bases.py` still hardcodes the stale 36-pair axis, but TP1 says endpoint obligations use the fixed committed instrument and never stale copies. Targeted current-surface search did not find a future consumer routing through the stale SSD source. | Keep the raw n600 receipt append-only and consume it only with `.omx/research/ddm_rr1_20260805/jd3_endpoint_n600_axis_correction.json`. For future endpoint probes, copy or invoke the fixed committed source and require the derived axis label. Continue cycle 2; next clean pass would be `2/3`. |

## CYCLE 2 ROUND 1 DISPOSITION RE-VERIFIED IN ROUND 2

| Prior item | Round-2 state | Evidence |
|---|---|---|
| JD4 force-reanchor repair | Verified in the emitted ticket and source. The ticket has `--jd1-force-ema-reanchor-on-resume`, resume start `1406`, `new_window_u=18000`, derived stage EMA decay `0.9997777777777778`, and zero lever-vs-argv mismatches. | `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd4_ticket_cont_ep1406.json` SHA-256 `a22783a...`; structured argv/lever check returned `mismatches 0`. Trainer predicate allows forced reanchor only for `resume_inside_joint_pose_finish` under window scope. |
| Recursive template and out-dir | Verified. The JD4 recursive template resolves under `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1406/checkpoints/`, and the child out-dir is unique in the ticket. | JSON field `recursive_encode_pass_loop.continue_policy.next_resume_from_template` points under the JD4 child out-dir. |
| Wall cap arithmetic | Verified as derived from the measured full-v3 timing, not a stale constant. | TP1 records full-v3 endpoint `3316s` over 60 epochs = `55.27 s/epoch`; JD4 wall cap computes `55 s/epoch * 120 * 1.5 / 60 = 165 min`. |
| Plateau-policy slow-EMA arithmetic | Verified. For `U=18000`, decay is `1 - 4/U = 0.9997777778`; code warmup `ceil(2/(1-decay)) = 9000` steps, which is 60 epochs at 150 steps/epoch. | `derive_jd1_stage_ema_decay(120, 150)` and `ema_warmup_updates` formula in the trainer source. |
| Probe status semantics | Partially verified. The receipt writes `status=running` during incremental rows and `status=complete` at the end; hot state explicitly says not to consume partial rows. The final receipt is complete now. | The completed receipt mtime is 2026-08-05 15:47 local, `.omx/tmp/codex_runs/jd3_endpoint_n600_probe_r2.done` exists, and run log printed both endpoint rows plus deltas. The remaining issue is RR1-C2-R2-F1's stale axis label. |
| Axis correction artifact | Landed as a record fix, not a raw-artifact mutation. | `.omx/research/ddm_rr1_20260805/jd3_endpoint_n600_axis_correction.json` records original receipt SHA-256 `fda3451c...`, source variant SHA-256 `dfb912b...`, the stale axis string, corrected n600 axis string, pair count 600, complete status, and both measured rows. |

## CYCLE 2 ROUND 2 DISPOSITION RE-VERIFIED IN ROUND 3

| Prior item | Round-3 state | Evidence |
|---|---|---|
| RR1-C2-R2-F1 axis class fix | Fixed in committed source; raw SSD receipt/variant intentionally remain append-only/stale and require the correction manifest. | `experiments/ddm_jd1_endpoint_verdict.py` computes `_axis_set_desc` from `GATE_PAIR_IDS`; `range(600)` yields `all 600 pair ids (0..599)`. Diff against the SSD variant shows the stale literal only in the preserved copy. TP1 endpoint obligations name the fixed committed instrument `6e4a6e24fe`, never the stale copies. |
| JD4 ticket custody | Verified. | Ticket SHA-256 is `a22783a9340c13e60fc8e79dc6f186d0570e0054f43cb79fe1a89c15ab171130`, internal `ticket_hash` is `51c64222b432b1abfac8cdb0d72ba39622573ce8a27c8e868e7144df26f93076`, argv0/argv1 are `.venv/bin/python` and `experiments/train_tr1_partition_renderer_mlx.py`, child out-dir is `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1406`, and `levers[*].overrides` vs final argv mismatches were `[]`. |
| JD4 launch/fire record | Verified with one sandbox boundary. | Launch manifest SHA-256 is `857f17708ab513ccd6aacddfffa946ceea8e157ddc3ea5c49dec66825d6c17d6`, `generated_utc=2026-08-05T20:55:58Z`, `pid=90157`, `done_receipt_path=.omx/tmp/codex_runs/jd4_cont_ep1406.done`, `git_sha=6e4a6e24feb86782dec2fe3dad2e94856cba9280`, and the manifest argv matches the ticket. `.done` was absent during this review and `kill -0 90157` was blocked by sandbox `operation not permitted`, so live process liveness was not measured by rr1. |
| JD4 R4 cure live telemetry | Verified from first telemetry rows. | `jd1_force_resume_epoch_reanchor` reports forced_start `1406`, legacy `1407`, tail_last `1405`. `resume` carries old JD3 smoke decay `0.9966666667` with `U=1200`. `jd1_stage_ema_reanchor` at epoch `1406` records `forced_from_resume=true`, new derived decay `0.9997777777777778`, `U=18000`, and warmup `9000`. This matches `ema_decay_run_geometry_v1` with `decay_from_warmup_fraction`, `phi=0.5`, `U=18000`. |
| Fire-gate adjudication | No gate-bending found. | The committed gate text was "review clears the LANDING." TP1 round 1 and rr1 round 2 found zero findings against the JD4 landing itself; rr1's sole round-2 finding was on the n600 probe artifact label and was fixed at the source class before fire. Under that wording, continuing the broader 3-clean-pass seal in parallel while firing the landing was an honest interpretation, not a silent weakening. |

## ROUND 2 DISPOSITIONS RE-VERIFIED IN ROUND 3

| Prior item | Round-3 state | Evidence |
|---|---|---|
| RR1-F1 DY1 inertness production caller | Correctly gated, not fixed. The TP1 receipt converts it to a merge-blocking condition on #961, and DY1's own handoff says future v3 smoke must require all five `scope_law_resolution` rows. The production refuse point is still absent. | In the DY1 clone at `a9eac92166`, `inertness_violations()` is defined in `scope_laws.py` and tested in `test_scope_laws.py`; `rg inertness_violations` finds no launcher/trainer production caller. `tools/launch_tr1_run.py` validates `scope_laws` schema and ticket hash only. |
| RR1-F2 peak memory wording | Corrected in prose; measurement still absent. | `rg "peak|RSS|VRAM|ru_maxrss|memory"` over JD3 launch dirs, smoke dirs, manifests, receipts, and telemetry found no peak RSS/VRAM measurement rows. |
| RR1-F3 entry verdict status | Fixed enough for custody, not restored to full summary schema. | `jd3_smoke_entry_pose_verdict.json` now has `status="complete_summary_stripped_see_rr1_f3"` and four ckpt rows. It still has no `deltas` or `gate_positive_control`, which is consistent with the preserved scratch variants stripping those summaries. |
| RR1-F4 rate wording | Corrected in TP1. | TP1 now says the 1,193 B telemetry delta is negligible for the pick rather than saying rate cancels. Recomputed sensitivity still leaves entry ahead by `0.124301811510 S` when telemetry counted-byte rates are included. # MAGNITUDE_DISMISSAL_OK: 0.000794 S = 0.63% of the 0.125096 S decision gap (157x under flip threshold) AND 0.136% of the 0.5818394 S remaining goal gap; verdict_scope INSTANCE (candidate pick only); nothing orphaned — rate re-enters IN FULL at byte-close (tp1 compliance note, committed) |
| RR1-F5 scratch variants | Fixed. | `/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/jd3_smoke_verdict_{entry,final}.py` are preserved. Diffs against `experiments/ddm_jd1_endpoint_verdict.py` show repo-path edits, `CKPTS` replacements, and summary-strip removal of `deltas`/`gate_positive_control`; the physics block remains unchanged. |
| Chain-sweep best-live caveat | Still honestly caveated in searched current surfaces. | `chain_both_bases_sweep.json` finalize note says process killed rc=-15, 7/8 rows valid, single-ckpt live values carry `+/-0.0002` noise, and best live ep1349 `0.0037233` is noise-caveated. I did not find an uncaveated downstream citation in the targeted current `.md`/`.json`/`.jsonl` searches. |

## ROUND 3 DISPOSITIONS RE-VERIFIED IN ROUND 4

| Prior item | Round-4 state | Evidence |
|---|---|---|
| RR1-R3-F1 mixed-scope full continuation | Marker fixed the citation hazard, but not the underlying same-dir/continuation semantics. The finding remains correctly disposed for smoke evidence only. | `/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/tr1_jd3_v3_smoke_entry_ep1336/SCOPE_MARKER.md` explicitly says the top-level directory is `entry_smoke_plus_full_continuation` and instructs smoke claims to use `smoke_snapshot_ep1344/`. |
| RR1-R3-F2 source-vs-saved epoch labels | Still a documentation rule, not a completed schema fix. | The preserved pose verdicts still use tags `smoke_start_ep1336` and `smoke_start_ep1355`; no new verdict schema field splits source checkpoint epoch from saved reanchor checkpoint epoch. |
| DY1 scope-law branch | Unit tests pass; the merge-blocking production caller gap remains. | In the DY1 clone at `a9eac92166`, `PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python -m pytest src/tac/witness_dsl/tests/test_scope_laws.py` passed `5/5`. `rg inertness_violations` still finds no production refuse caller outside tests/helper definition. |

## ROUND 4 DISPOSITION RE-VERIFIED IN ROUND 5

| Prior item | Round-5 state | Evidence |
|---|---|---|
| RR1-R4-F1 wrong-geometry EMA carry | Correctly relabeled and endpoint-blocked in the governing surfaces searched. The continuation remains a real run, but only as a `smoke-EMA continuation`, not as a full-window EMA-cured result. | TP1 says the window runs on under the relabel, endpoint adjudication must measure both live and EMA bases, DY1/#961 must resolve declared-vs-active EMA scope before selection, and regenerator debt remains. Main hot state repeats `smoke-EMA continuation` and keeps #961 blocked on production inertness plus geometry-hash-keyed resolution. |
| Missed consumers of the wrong label | No missed current consumer found in searched decision surfaces. | Targeted `rg` over `.omx/research`, `.omx/state`, docs, and reports found TP1, main hot state, RR1, cx1 pre-fire context, and older historical EMA mentions. cx1 predates the full-continuation defect and consumes smoke/fire gates, not the full endpoint. |
| DY1 scope-law merge condition | Disposition complete as a gate, not implemented as a cure. DY1 remains blocked. | In clone `a9eac92166`, `ScopeLawResolution.with_hash()` hashes the full row, but `inertness_violations(declared_scope_laws, resolved_scope_laws)` compares names only. A wrong-geometry `jd3_stage_ema_decay` row would therefore satisfy the helper if only that helper were wired. This matches the existing TP1/NEXT_IF_RESUMED demand for a positive control where full-window declared EMA refuses when only a smoke-scope row exists. |
| Regenerator state-flag debt | Still debt, not fixed. | `experiments/ddm_jd1_ticket_regenerate.py` can preserve a same-dir continuation shape by changing only `--epochs`, `--resume-from`, and wall minutes; the full ticket confirmed this. Future continuation tickets that change geometry must force fresh reanchor or emit a scope-law refusal. |
| Other resume-carried state flags | No new hidden full-window derivation claim found. `jd1_realized_hold` floor/margin, `engaged`, and effective pose-controller state carry through the same-stage continuation; these are not EMA-law geometry claims. They must still be labeled as inherited/static if endpoint rows are harvested. | Trainer resume code carries `jd1_pose_finish_state`, `jd1_realized_hold_state`, and `effective_w_pose`. Round 1/TP1/CX1 already classified the static realized floor as grandfathered and non-ratcheting; no rollback/refuse row was observed in the smokes. |
| Live endpoint status | Still not consumed by rr1. | The expected `.done` marker for `jd3_full_v3_entry_cont` was absent in this sandbox check; top-level telemetry was inspected only as control-flow evidence, not as an endpoint result. |

## ROUND 5 DISPOSITION RE-VERIFIED IN ROUND 6

| Prior item | Round-6 state | Evidence |
|---|---|---|
| RR1-R5-clean wrong-label sweep | Still no completed endpoint consumed, and the governing label boundary remains intact for the searched surfaces. | Main hot state and TP1 still require `smoke-EMA continuation` handling and no byte-close before endpoint probes. The live r2 manifest records `.omx/tmp/codex_runs/jd3_full_v3_entry_cont.done`, but that marker was absent at 2026-08-05 15:06 CDT; telemetry/log were updating through epoch 1395. |
| Mixed top-level smoke/full directory | Covered by RR1-R3-F1, not a new Round-6 finding by itself. | `tr1_window_receipt.json` in the mixed top-level dir still has the smoke receipt mtime/content while `telemetry.jsonl` has full-continuation rows through epoch 1395. Trainer source writes `tr1_window_receipt.json` only at terminal exit; consumers must wait for the `.done` marker and use `SCOPE_MARKER.md`/`smoke_snapshot_ep1344/` split. |
| Endpoint follow-on fire-order | Not clean: RR1-R6-F1 found the stale recursive continuation template. | Fired full ticket's actual `child_out_dir` is the JD3 smoke-entry dir; its `next_resume_from_template` still points to the old JD1 `tr1_joint_pose_finish_after_tp1_lane_on` dir. |

## ROUND 6 DISPOSITION RE-VERIFIED IN ROUND 7

| Prior item | Round-7 state | Evidence |
|---|---|---|
| RR1-R6-F1 stale continuation template | Dispositioned as a consumption contract in TP1, not code-fixed. The current fired ticket remains unmutated and hash-custodied; any continuation from the full-v3 endpoint must derive from the actual endpoint checkpoint, not the ticket's stale template. | TP1 round-6 disposition declares the template VOID and adds regenerator debt item #4. `tools/launch_detached_process.py` consumes only the final command argv for the current run, so the recursive loop did not affect the launched physics. |
| Regenerator emission-surface sweep | Not clean: RR1-R7-F1 found stale inherited `levers[*].overrides` values in the same regenerated tickets. | All three JD3 tickets update final argv values but keep old lever override metadata for epochs, wall cap, EMA decay, and seg-hold floor source. |

## ROUND 1 F1-F4 VERIFICATION

| Round-1 item | rr1 verdict | Evidence |
|---|---|---|
| F1 EMA-only v2 endpoint / v3 controller non-intervention | VERIFIED with caveat. The v3 smokes have zero rollback/refuse rows; realized hold stood guard but did not intervene. Existing v2 discriminator is EMA-basis for its three preserved ckpts, with live values only at entry/final per TP1. | `jq` over smoke telemetry found only `jd1_stage_ema_reanchor` and `jd1_realized_hold_latch`; no `jd1_realized_hold_rollback`, `jd1_realized_hold_refuse`, or `a1_refuse`. |
| F2 static realized floor | VERIFIED and quantified. The entry-chain static threshold is `0.007122746220341435 + 0.0002168586037040609 = 0.007339604824045496`. At ep1359 the chain could regress by `0.0017822973 d_seg = 0.1782297 S_seg` before the realized hold tripped. A1 covers only persistent smooth-descent/realized-nondrop gaps; it is not a ratcheting absolute floor. | Derived from `jd1_realized_hold_latch` and later `a1_gate` rows. A1 code classifies by interval drops at `a1_adjudicate`, not by best-so-far floor. |
| F3 budget asymmetry | VERIFIED. Entry pick uses an 8-epoch smoke from the fresh entry state; refuse-final already carries the v2 lineage before its 8-epoch smoke. The pick still stands on same-instrument endpoint value and tail slopes. | Entry verdict rows: `1336 -> 1339 -> 1344`. Refuse-final rows: `1355 -> 1359 -> 1363`. |
| F4 n36 scope label | VERIFIED. Both smoke verdict JSONs state 36 gd1-designed gate pairs and non-promotable advisory axis. This is not an n600 result. | `axis="[macOS-CPU frozen-scorer advisory] 36 gd1-designed gate pairs, NON-PROMOTABLE"`, `pairs=36`, `score_claim=false`. |

## ADJUDICATION ARITHMETIC

Partial-S formula used by the smoke adjudication: `100*d_seg + sqrt(10*d_pose)`, excluding rate.

| Candidate endpoint | d_seg | d_pose | pose term | partial-S |
|---|---:|---:|---:|---:|
| entry `gate_ep1344` / smoke final | 0.006819972286 | 0.101172111458 | 1.005843484136 | 1.687840712688 |
| refuse-final `gate_ep1363` / smoke final | 0.005571012144 | 0.157712325404 | 1.255835679555 | 1.812936893929 |

Including telemetry counted-byte rates for a sensitivity check gives entry `1.892186832671` and refuse-final `2.016488644181`; entry still wins by `0.124301811510 S`. This is advisory telemetry rate, not an archive score.

## EMA DERIVATION

For the smoke windows, `remaining_epochs=8` and `steps_per_epoch=150`, so `U=1200`.

`d = 1 - 2/(0.5*1200) = 1 - 4/1200 = 0.9966666666666667`.

Seed retained by the reanchored EMA:

| Updates after reanchor | Seed retained `d^h` | New-weight mass `1-d^h` |
|---:|---:|---:|
| 450 | 0.222571791836 | 0.777428208164 |
| 1200 | 0.018193670527 | 0.981806329473 |
| 1950 | 0.001487203947 | 0.998512796053 |

Implication: the reanchored EMA materially reduces the parent-horizon mismatch, but an early gate can still carry a real live-vs-EMA gap. Gate decisions must keep live and EMA rows separate.

## CX1 FIRE-GATE CHECK

| Gate | rr1 check |
|---|---|
| FG1 lane-guard ratchet | Verified not triggered in the smokes: entry lane S-units fell `0.1335427 -> 0.1231158`; refuse-final fell `0.1149495 -> 0.1070093`. Ratchet remained disabled. |
| FG2 realized-hold rows | Verified: both smokes emitted `jd1_realized_hold_latch` with `sd/sqrt(36)` margins. |
| FG3 rollback | Verified n/a: no rollback/refuse rows in either smoke telemetry. |
| FG4 deterministic R | The TP1 fire record explicitly records the Metal nondeterminism decision; exact authority remains CPU/CUDA. I did not find a deterministic-R enablement artifact. |
| FG5 v4 riders | Verified absent from v3 argv/config: `margin_weighted_loss=off`, distill weights zero, no silent stacking of EN1/SL2/PE3 riders in the v3 tickets. |

## DY1 PRE-MERGE REVIEW

| Question | Verdict |
|---|---|
| Does OFF path leave trainer byte-identical? | Source/test-supported, not runtime-measured. DY1's ticket test shows adding `scope_laws` does not change argv; legacy tickets without laws omit `scope_laws`. I did not run a byte-for-byte trainer-output comparison. |
| Does seal-binds-laws / receipt-binds-values split hold? | Partially. `ticket_payload_hash()` includes `scope_laws`, and `ScopeLawResolution` rows carry resolution hashes suitable for receipt/checkpoint metadata. |
| Are inertness controls real? | Not yet. See RR1-F1: the inertness positive-control helper is not wired into production launch/selection. |

DY1 should not merge as a complete inertness cure until RR1-F1 is closed.

## ASSUMPTION-CHALLENGE AXIS

Shared assumption: the n36 gd1-designed gate sample and training-objective EMA-basis smoke rows are sufficient to choose the next resume candidate.

Would violating it unlock breakthrough? For the candidate pick, likely not enough to overturn the entry-vs-refuse-final decision because the observed partial-S gap is `0.1251 S` and the rate sensitivity is only `0.000794 S`. For promotion or pointer movement, yes: n36/training-objective evidence must be replaced by n600 byte-closed archive evaluation on authority hardware.

## ROUND 4 ASSUMPTION-CHALLENGE AXIS

Shared assumption: a smoke-final checkpoint with a stage-scoped EMA reanchor can be used as the start of a longer continuation without re-solving the EMA scope law.

Would violating it unlock breakthrough? It will not itself move the score, but it prevents the same cross-regime constant-transfer class that v3 was built to cure. The full continuation may still be a real candidate if measured as the actual checkpoint it is; it cannot be cited as a full-window EMA-cured result without a fresh reanchor or an explicit relabel.

## ROUND 5 ASSUMPTION-CHALLENGE AXIS

Shared assumption: documented relabel/gating is enough to prevent the wrong full-window EMA claim while the production guard is still pending.

Would violating it unlock breakthrough? Not directly on score, but it would reduce recurrence risk in the same state-flag class. The next higher-value cure is code-level: scope-law selection must compare declared full-window EMA geometry against the resolved active row's geometry/hash, not just law names or a boolean `stage_ema_reanchored` flag.

## ROUND 6 ASSUMPTION-CHALLENGE AXIS

Shared assumption: machine-readable follow-on metadata in a regenerated ticket is harmless unless `launch_now=true`.

Would violating it unlock breakthrough? It will not move the score by itself, but it prevents an endpoint-governance failure: stale continuation templates can silently route the next pass away from the measured full-v3 endpoint or into an ancestor lane. Follow-on fire-orders must be validated as executable artifacts even when the current run is still live.

## ROUND 7 ASSUMPTION-CHALLENGE AXIS

Shared assumption: final argv is the only machine-readable value authority, so inherited lever metadata can be treated as documentation.

Would violating it unlock breakthrough? Not directly on score, but it closes a NO-FAKE provenance hole. This project uses tickets as value-custody artifacts; if `levers[*].overrides` remains present, downstream checks and readers can reasonably treat it as authoritative. The cure is structural: either the regenerator makes the value ledger match final argv after rewrites, or it removes/demotes that ledger so it cannot pass as value custody.

## CYCLE 2 ROUND 2 ASSUMPTION-CHALLENGE AXIS

Shared assumption: a probe copied from a 36-pair discriminator can safely become an n600 endpoint probe through CKPTS/pair-list edits alone.

Would violating it unlock breakthrough? Not directly on score, but it prevents denominator laundering at the artifact boundary. For endpoint governance, the receipt's denominator label is part of the measured claim; changing the population requires updating the source-level axis, output default, and receipt schema, not only the pair list.

## MEASURED-RUNNABILITY AND MEASURED-SCORED-QUANTITY AXIS

- MEASURED from existing artifacts: both v3 smokes executed at `num_pairs=600`, `batch_pairs=4`, emitted gates, and preserved checkpoints. The full-window retry launched with `.venv/bin/python` prepended.
- NOT MEASURED by this arm: peak RSS/VRAM, any new scorer output, any archive bytes, contest CPU/CUDA score, or full-window endpoint.
- Measured scored quantity in scope: n36 advisory `d_seg`/`d_pose` through the existing smoke probe only. Rate sensitivity used telemetry counted bytes, not an exact archive.

## ROUND 4 MEASURED-RUNNABILITY AND MEASURED-SCORED-QUANTITY AXIS

- MEASURED from existing artifacts: the full continuation wrote live telemetry through at least epoch 1377, with active EMA decay `0.9966666667` on every inspected epoch row. The expected done marker `jd3_full_v3_entry_cont.done` was absent at 2026-08-05T19:50:32Z, so I did not inspect or consume a completed endpoint.
- NOT MEASURED by this arm: process liveness via `ps` (`operation not permitted` in this sandbox), peak RSS/VRAM, a completed full-window endpoint, any archive bytes, or contest CPU/CUDA score.
- Measured scored quantity in round 4: none new. The review result is source/telemetry control-flow evidence only.

## ROUND 5 MEASURED-RUNNABILITY AND MEASURED-SCORED-QUANTITY AXIS

- MEASURED from existing artifacts: current-source and receipt labels, ticket diffs, telemetry control-flow rows, DY1 helper semantics, preserved scratch-probe source diffs, and absence of a local full-window `.done` marker.
- NOT MEASURED by this arm: no new scorer output, no process liveness, no peak RSS/VRAM, no completed endpoint, no archive bytes, and no contest CPU/CUDA score.
- Measured scored quantity in round 5: none new. This was a review-only clean pass over control-flow and custody surfaces.

## ROUND 6 MEASURED-RUNNABILITY AND MEASURED-SCORED-QUANTITY AXIS

- MEASURED from existing artifacts: the r2 launch manifest, the absence of `.omx/tmp/codex_runs/jd3_full_v3_entry_cont.done`, the current-minute run log summary, live telemetry through epoch 1395, trainer source showing terminal receipt write semantics, and ticket/manifest path consistency.
- NOT MEASURED by this arm: process liveness via `ps` (`operation not permitted`), peak RSS/VRAM, a completed full-window endpoint, any new scorer output, archive bytes, or contest CPU/CUDA score.
- Measured scored quantity in round 6: none new. The finding is a follow-on metadata/fire-order defect, not a scored result.

## ROUND 7 MEASURED-RUNNABILITY AND MEASURED-SCORED-QUANTITY AXIS

- MEASURED from existing artifacts: the r2 launch manifest, ticket SHA-256s, structured ticket-field diffs, absence of `.omx/tmp/codex_runs/jd3_full_v3_entry_cont.done`, and live telemetry through epoch 1402 in this sandbox check.
- NOT MEASURED by this arm: process liveness via `ps`, peak RSS/VRAM, a completed full-window endpoint, any new scorer output, archive bytes, or contest CPU/CUDA score.
- Measured scored quantity in round 7: none new. The finding is a ticket value-custody/provenance defect, not a scored result.

## CYCLE 2 ROUND 2 MEASURED-RUNNABILITY AND MEASURED-SCORED-QUANTITY AXIS

- MEASURED from existing artifacts: JD4 ticket fields, trainer/regenerator source, n600 probe launch manifest, completed probe log, completed probe JSON, and SHA-256s for the probe script/receipt/ticket.
- Existing scored quantities consumed for review only: n600 all-pair training-vehicle objective rows from `jd3_endpoint_n600_both_bases_verdict.json`: live d_seg `0.0071503364`, live d_pose `0.5740917290`, live pose term `2.3960211372`; EMA d_seg `0.0057479858`, EMA d_pose `0.1288530915`, EMA pose term `1.1351347562`. These are `[macOS-CPU frozen-scorer advisory]`, non-promotable, not byte-closed archive rows, and currently denominator-mislabeled in the source receipt.
- NOT MEASURED by this arm: no new scorer output, no Metal/MLX launch, no peak RSS/VRAM, no archive bytes, no byte-close, and no contest CPU/CUDA score.

## CYCLE 2 ROUND 3 ASSUMPTION-CHALLENGE AXIS

Shared assumption: "landing review clear" can be scoped to the JD4 code/ticket landing while n600 probe-label correction and the 3-clean-pass seal continue in parallel.

Would violating it unlock breakthrough? Not directly on score, but it would tighten launch governance if future wording says "all artifacts clear" instead of "landing clears." For this fire, the narrower reading is supported by the written gate and by source-level disposition of the only non-landing finding. The safer future rule is to name the gated surface explicitly: landing-only, all-artifact seal, or endpoint-consumption seal.

## CYCLE 2 ROUND 3 MEASURED-RUNNABILITY AND MEASURED-SCORED-QUANTITY AXIS

- MEASURED from existing artifacts: committed endpoint source class fix, stale SSD variant diff, correction manifest, JD4 ticket SHA/fields, launch manifest SHA/fields, first JD4 telemetry rows, checkpoint file presence, canonical EMA law source, and current-surface consumer search.
- Existing scored quantities consumed for review only: same n600 all-pair training-vehicle objective rows from the corrected JD3 endpoint receipt as round 2; no new score was produced.
- NOT MEASURED by this arm: process liveness (`kill -0` blocked by sandbox `operation not permitted`), completed JD4 endpoint, peak RSS/VRAM, any new scorer output, archive bytes, byte-close, or contest CPU/CUDA score.

## CYCLE 2 ROUND 4 FINDINGS

Counter state after this round: **0/3 clean passes**. This round found one low-severity record issue, so it is not a clean pass.

| ID | Severity | Finding | Evidence | Required disposition |
|---|---|---|---|---|
| RR1-C2-R4-F1 | LOW | The TP1 plateau policy's Case A pose satisficing bar has a small exact-arithmetic overclaim. The text says `d_pose <= ~1.5e-3 -> contribution <=0.12`, but the formula is `sqrt(10*d_pose)`: at `d_pose=0.0015`, the pose contribution is `0.1224744871`, not `<=0.12`. The strict `<=0.12` cutoff is `d_pose <= 0.00144`. This does not invalidate the live CONTINUE decision because the endpoint d_pose is still ~86x above the neighborhood; it only affects the future hard threshold wording. | `.venv/bin/python` derivation: `sqrt(10*0.0015)=0.1224744871391589`; `0.12**2/10=0.00144`. TP1 Case A used the shorthand in the plateau policy. | Appended a TP1 correction note. Future hard gates should use `d_pose <=0.00144` for `pose_term <=0.12`, and may call `1.5e-3` only an approximate satisficing neighborhood (~0.1225 S). |

## CYCLE 2 ROUND 4 ENDPOINT-PATH CHECK

| Exit path | Artifact consumed | Geometry / done verdict |
|---|---|---|
| `epochs_complete` at ep1526 | Wait for `.omx/tmp/codex_runs/jd4_cont_ep1406.done`, then read `tr1_window_receipt.json` first and `checkpoints/stage_joint_pose_finish_final.npz` second. Endpoint probe must run from the fixed committed `experiments/ddm_jd1_endpoint_verdict.py` copied/adapted after the completed receipt, not from the stale SSD n600 variant. | Source writes the terminal checkpoint before the terminal receipt; the checkpoint embeds `telemetry_tail[-4:]`. With final tail max ep1525 and `meta::epoch=1526`, future force-resume geometry resolves from tail+1 = 1526, so the exclusive-epoch mismatch remains covered. Launcher supervisor writes the `.done` marker on child exit. |
| `max_wall_minutes` at 165 min | Same consumption order: `.done` marker, then terminal receipt, then final checkpoint if the receipt declares a usable endpoint. | The trainer checks wall cap at the top of the epoch loop, then still saves `stage_joint_pose_finish_final.npz` and writes `tr1_window_receipt.json` with `stop_reason=max_wall_minutes`. Baked tail geometry remains tied to the last completed gate; the next forced resume should derive from the tail, not blindly from `meta::epoch+1`. `.done` fires through the launcher supervisor for any child rc. |
| typed exits (`a1_realization_gap_refuse`, realized-hold refusal, `nonfinite_loss`, or basin handoff) | Receipt is the authority. Do not treat `stage_joint_pose_finish_final.npz` alone as an endpoint row; branch by `stop_reason` before any probe/continuation. `basin_entry_handoff` has its own handoff receipt; refusal/nonfinite paths are fold/refuse evidence unless a later adjudicator records a bounded-positive reason. | Source still writes a terminal checkpoint and receipt after typed breaks, and the supervisor still writes `.done`. For nonfinite, no endpoint score should be consumed without an explicit safety adjudication because the break happens after the batch update path; receipt-first branching is mandatory. This is a boundary rule, not a new JD4 finding because the current live window has not reached a typed exit. |

## CYCLE 2 ROUND 4 ASSUMPTION-CHALLENGE AXIS

Shared assumption: plateau-policy thresholds can be carried as rounded operational bands without changing future hard gates.

Would violating it unlock breakthrough? Not directly on score, but it prevents threshold drift at the exact branch where training stops and terminal solvers take over. The hard branch predicate should use the exact formula (`d_pose <=0.00144` for pose contribution `<=0.12`); rounded neighborhoods are acceptable only when explicitly labeled approximate.

## CYCLE 2 ROUND 4 MEASURED-RUNNABILITY AND MEASURED-SCORED-QUANTITY AXIS

- MEASURED from existing artifacts: JD4 ticket SHA/fields, launch manifest schema/argv, live telemetry through ep1414, force-reanchor telemetry, committed endpoint source diff against the stale SSD variant, stale raw n600 receipt plus correction manifest, trainer terminal checkpoint/receipt source, launcher `.done` supervisor source, and the exact arithmetic above.
- Existing scored quantities consumed for review only: the previously completed n600 training-vehicle live/EMA endpoint rows from `jd3_endpoint_n600_both_bases_verdict.json`; no new scorer output was produced.
- NOT MEASURED by this arm: process liveness (`ps` is sandbox-blocked with `operation not permitted`), completed JD4 endpoint, peak RSS/VRAM, any new scorer output, archive bytes, byte-close, or contest CPU/CUDA score.

## CYCLE 2 ROUND 4 DISPOSITION RE-VERIFIED IN ROUND 5

| Prior item | Round-5 state | Evidence |
|---|---|---|
| RR1-C2-R4-F1 Case A arithmetic correction | Fixed in TP1 primary wording and adjudication echo; no third TP1 gate site still carries the loose inequality as a hard gate. | `rg` over `.omx/research/ddm_tp1_boundary_receipt_20260805.md` found the corrected Case A policy line and endpoint-adjudication paragraph: strict gate `d_pose <= 0.00144` gives contribution `<=0.12`; `1.5e-3` is labeled only as the rounded `~0.1225` neighborhood. Recomputed values: `sqrt(10*0.00144)=0.1200000000`, `sqrt(10*0.0015)=0.1224744871`, and `0.12**2/10=0.00144`. |
| JD4 R4 cure live telemetry | Verified from the actual run telemetry, not the commit message. | Telemetry rows record `jd1_force_resume_epoch_reanchor` forced_start `1406`, legacy `1407`, saved_epoch `1406`, tail_last `1405`; the following `jd1_stage_ema_reanchor` row records new derived decay `0.9997777777777778`, old carried decay `0.9966666666666667`, `U=18000`, and `ema_warmup_updates=9000`. |
| Warmup and tau claims | Correct when derived per update and converted through measured `steps_per_epoch=150`. | Ticket/run geometry is ep1406->1526, so `U=(1526-1406)*150=18000`. The derived decay is `1-4/U=0.9997777777777778`; warmup is `U/2=9000` updates = 60 epochs, ending near ep1466. The EMA time constant is `-1/log(decay)=4499.49998` updates = `29.9967` epochs; ep1526 is `(1526-1466)/29.9967 = 2.0002` tau after warmup. |
| Live A1 status | Watch note only, not a finding or endpoint verdict. | JD4 telemetry through ep1429 has one `A1_REALIZATION_GAP_ALARM` at ep1424 (`interval_dseg_delta=+4.1114e-05`), followed by a non-alarming `COUPLED_DESCENT` gate at ep1429 (`interval_dseg_delta=-1.0399e-04`). There are zero `a1_stage_exit_refuse`, `jd1_realized_hold_refuse`, or `jd1_realized_hold_rollback` rows; `.done` and `tr1_window_receipt.json` remain absent. |

## CYCLE 2 ROUND 5 FINDINGS

Counter state after this round: **1/3 clean passes**. This round found zero new findings.

| ID | Severity | Finding | Evidence | Required disposition |
|---|---|---|---|---|
| RR1-C2-R5-clean | NONE | No new blocker, regression, gate-bending, arithmetic error, or missed endpoint consumer was found in the searched round-5 scope. | R4 correction is real in TP1; JD4 force-reanchor telemetry and EMA/warmup/tau arithmetic match the fire record; endpoint artifacts are still absent, so no partial telemetry was consumed as an endpoint row. The single ep1424 A1 alarm recovered at ep1429 and did not produce a typed exit/refuse row. | Continue recursive review with cycle-2 counter `1/3`. Next round must start from live `.done`/receipt status if the window has completed; otherwise keep partial telemetry advisory and non-endpoint. |

## CYCLE 2 ROUND 5 ASSUMPTION-CHALLENGE AXIS

Shared assumption: a clean review round can accept a live telemetry warning as non-finding if the trainer's typed-exit contract did not fire and the next gate recovered.

Would violating it unlock breakthrough? Not directly on score, but it prevents two opposite errors: laundering a partial warning into a terminal fold, or ignoring the early signal when endpoint governance later branches. The correct boundary is receipt-first: a single A1 alarm is a watch condition; a typed `a1_realization_gap_refuse` or terminal receipt `stop_reason` is the consumable endpoint/fold signal.

## CYCLE 2 ROUND 5 MEASURED-RUNNABILITY AND MEASURED-SCORED-QUANTITY AXIS

- MEASURED from existing artifacts: TP1 correction text, JD4 launch manifest, JD4 telemetry through ep1429, trainer A1/refuse source, canonical EMA formula output, and absence of `.omx/tmp/codex_runs/jd4_cont_ep1406.done` plus `tr1_window_receipt.json`.
- Existing scored quantities consumed for review only: none new; the telemetry d_seg values are gate advisory rows on the running training vehicle, not endpoint scores and not archive scores.
- NOT MEASURED by this arm: process liveness via `ps`, completed JD4 endpoint, peak RSS/VRAM, any new scorer output, archive bytes, byte-close, or contest CPU/CUDA score.

## CYCLE 2 ROUND 6 DISPOSITION RE-VERIFIED

| Prior item | Round-6 state | Evidence |
|---|---|---|
| MAIN-R5X source correction | Verified from source, not from the commit message. | `jd1_should_reanchor_stage_ema()` only forces over a carried `stage_ema_reanchored` latch when `--jd1-force-ema-reanchor-on-resume` is set and reason is `resume_inside_joint_pose_finish`. On resume, `global_step = ema_warmup_updates`; the forced reanchor resets EMA equal to live weights, derives the new decay, recomputes warmup, and bumps `global_step = max(global_step, ema_warmup_updates)`. Gate selection then reads normal `a1_gate` rows from `ema_shadow` when `global_step >= ema_warmup_updates`. |
| TP1 and hot-state correction text | Correct at the three named TP1 sites plus hot state. | TP1 Round-1 check 5 now labels the old "continuation gates MUST read LIVE" inference as wrong; the adjudication watch item says `a1_gate` reads `ema_shadow` from the first gate and names the parallel `jd1_live_basis_gate` channel; endpoint obligations say in-window gate basis is ep1406-reanchored `ema_shadow`, with live basis logged in parallel. Hot state repeats the same corrected basis and says not "live through warmup". |
| 4-tau arithmetic | Verified. | With `active_ema_decay=0.9997777777777778`, `1/(1-d)=4500` updates = `30` epochs at `150` steps/epoch; the exact log time constant is `4499.49998` updates = `29.9967` epochs. A 120-epoch JD4 window is `4.0000` reciprocal-tau, or `4.0004` log-tau, from the ep1406 anchor. |
| Residual live-basis plateau policy | Not a finding. | TP1 Case 0/B/C still intentionally specify live-basis plateau slopes because slow EMA can manufacture phantom plateaus, but JD4 ticket/manifest enable `--jd1-live-gate-telemetry`, trainer source requires it for realized hold, and telemetry writes paired rows: `jd1_live_basis_gate gate_params=live_weights` plus `a1_gate gate_params=ema_shadow`. The policy is executable if consumers read the live channel for plateau slopes and the normal A1/hold channel for EMA-shadow gate safety. |
| JD4 live endpoint status | Still incomplete in this sandbox check. | `.omx/tmp/codex_runs/jd4_cont_ep1406.done` and `tr1_window_receipt.json` were absent. Telemetry through ep1439 showed normal `a1_gate` basis `ema_shadow`, no typed exit/refuse rows, and one current `FLAT` classification at ep1439, which is not a three-gate plateau or endpoint verdict. |

## CYCLE 2 ROUND 6 FINDINGS

Counter state after this round: **1/3 clean passes**. This round found zero new findings after MAIN-R5X reset the counter.

| ID | Severity | Finding | Evidence | Required disposition |
|---|---|---|---|---|
| RR1-C2-R6-clean | NONE | No new blocker, regression, gate-bending, stale live-basis consumer, arithmetic error, or endpoint-consumption defect was found in the searched round-6 scope. | MAIN-R5X matches trainer source and live telemetry; TP1/hot-state correction sites are accurate; the plateau policy's live-basis slopes have a real telemetry channel; no `.done` or terminal receipt exists, so no partial telemetry was consumed as an endpoint row. | Continue recursive review with cycle-2 counter `1/3`. Next round should start from the latest `.done`/`tr1_window_receipt.json` status and preserve the live-vs-EMA channel split. |

## CYCLE 2 ROUND 6 ASSUMPTION-CHALLENGE AXIS

Shared assumption: live-basis plateau detection can coexist safely with EMA-shadow A1/hold gating as long as both channels are logged and consumers keep them separate.

Would violating it unlock breakthrough? Not directly on score, but it prevents a silent governance failure. If consumers collapse `jd1_live_basis_gate` and `a1_gate` into one "gate" series, they can either manufacture a plateau from slow EMA or miss a hold/refuse event by reading the wrong basis. The decision surface remains sound only with a typed split: live channel for plateau slope policy, EMA-shadow normal A1 channel for safety/hold telemetry, and both-bases n600 endpoint probes before any byte-close or candidate row.

## CYCLE 2 ROUND 6 MEASURED-RUNNABILITY AND MEASURED-SCORED-QUANTITY AXIS

- MEASURED from existing artifacts: trainer source for force-reanchor and gate-basis semantics, JD4 ticket/manifest live-gate flag, TP1/hot-state correction text, canonical gate-basis recall surfaces, and JD4 telemetry through ep1439.
- Existing scored quantities consumed for review only: none new. Gate `d_seg` rows in telemetry are `[macOS-CPU/MLX advisory]` training-vehicle steering rows, not endpoint scores, not archive-byte-closed, and not contest authority.
- NOT MEASURED by this arm: completed JD4 endpoint, peak RSS/VRAM, any new scorer output, archive bytes, byte-close, process liveness via `ps`, or contest CPU/CUDA score.

## BOUNDARIES

- No `.py` file was edited; no review override was used.
- No protected file was touched.
- The existing dirty worktree was left intact.
- Live full-v3/JD4 SSD artifacts were read-only.
- The scorer slot was not claimed by this arm; I only inspected the existing completed n600 probe artifact.
- No byte-close, archive build, or exact eval was run.
- The contest pointer remains borrowed/unmoved.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
