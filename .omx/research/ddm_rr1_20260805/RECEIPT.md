# ddm_rr1 recursive adversarial review receipt

Status: REVIEW arm, rounds 2-3 fresh-eyes pass over the JD3 v3 chain and DY1 scope-law resolver. I ran no scorer, no Metal/MLX launch, no archive build, and no exact evaluation. Live full-v3 dirs/files were read-only.

Axis: source/receipt/telemetry review plus derived arithmetic. Any score arithmetic below is non-promotable `[macOS-CPU/MLX advisory]` or derived from existing receipts; `score_claim=false`.

## RECALL EVIDENCE

| Surface | Query / artifact | Result beyond charter seeds | Plan change |
|---|---|---|---|
| Pact memory registry | `rg -n "rr1|jd3|dy1|scope-law|chain_both_bases|0.0037233" /Users/adpena/.codex/memories/MEMORY.md` | No rr1-specific prior memory found; unrelated #899/#904 memory only. | Kept review scoped to live artifacts, not memory conclusions. |
| Governing docs | `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | Hot state says JD3 full window owns scorer slot and the live dirs are sacred; recursive review requires assumption-challenge and measured-runnability axes. | No scorer work; all findings are review-only. |
| Full corpus recall | targeted `rg` over `.omx/research`, `.omx/state`, docs/reports for `JD3`, `full_v3`, `dy1`, `chain_both_bases`, `ep1349`, `0.0037233` | Found cx1 fire-gate receipt and dy1 build receipt as additional context. Chain-sweep best-live caveat lives inside the JSON finalize note; no uncaveated downstream citation found in searched md/json surfaces. | Added cx1 gate verification and DY1 pre-merge review. |
| Round-3 recall refresh | targeted `rg` over `.omx/research`, `.omx/state`, docs/reports for `JD3`, `full_v3`, `scope-law`, `inertness`, `chain_both_bases`, `0.0037233`, `ru_maxrss`, `peak RSS`, `VRAM` | Found `ddm_gc20_20260805/RECEIPT.md`, which consumes the two-smoke matrix and explicitly says the live full-v3 lane must not be mutated. The only current peak-memory rows found were unrelated older receipts; JD3 manifests/telemetry still do not record peak RSS/VRAM. | Raised the custody severity of the full-window out-dir alias; kept measured-runnability downgraded to executed-without-peak-memory. |
| Canonical equations | `tools/list_canonical_equations.py --json`, plus source grep for `ema_decay_run_geometry_v1` | EMA law is executable; JD3 stage EMA derivation is `d = 1 - 2/(0.5*1200) = 0.9966666667`. | Re-derived EMA seed-retention rather than trusting the receipt wording. |
| Source and receipts | TP1 receipt, cx1 receipt, JD3 smoke telemetry/verdicts, fired ticket, launcher manifests, DY1 clone at `a9eac92166` | Found new artifact/status and enforcement gaps below. | Counter remains 0/3; fixes are recommended, not applied. |

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

## ROUND 2 DISPOSITIONS RE-VERIFIED IN ROUND 3

| Prior item | Round-3 state | Evidence |
|---|---|---|
| RR1-F1 DY1 inertness production caller | Correctly gated, not fixed. The TP1 receipt converts it to a merge-blocking condition on #961, and DY1's own handoff says future v3 smoke must require all five `scope_law_resolution` rows. The production refuse point is still absent. | In the DY1 clone at `a9eac92166`, `inertness_violations()` is defined in `scope_laws.py` and tested in `test_scope_laws.py`; `rg inertness_violations` finds no launcher/trainer production caller. `tools/launch_tr1_run.py` validates `scope_laws` schema and ticket hash only. |
| RR1-F2 peak memory wording | Corrected in prose; measurement still absent. | `rg "peak|RSS|VRAM|ru_maxrss|memory"` over JD3 launch dirs, smoke dirs, manifests, receipts, and telemetry found no peak RSS/VRAM measurement rows. |
| RR1-F3 entry verdict status | Fixed enough for custody, not restored to full summary schema. | `jd3_smoke_entry_pose_verdict.json` now has `status="complete_summary_stripped_see_rr1_f3"` and four ckpt rows. It still has no `deltas` or `gate_positive_control`, which is consistent with the preserved scratch variants stripping those summaries. |
| RR1-F4 rate wording | Corrected in TP1. | TP1 now says the 1,193 B telemetry delta is negligible for the pick rather than saying rate cancels. Recomputed sensitivity still leaves entry ahead by `0.124301811510 S` when telemetry counted-byte rates are included. |
| RR1-F5 scratch variants | Fixed. | `/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/jd3_smoke_verdict_{entry,final}.py` are preserved. Diffs against `experiments/ddm_jd1_endpoint_verdict.py` show repo-path edits, `CKPTS` replacements, and summary-strip removal of `deltas`/`gate_positive_control`; the physics block remains unchanged. |
| Chain-sweep best-live caveat | Still honestly caveated in searched current surfaces. | `chain_both_bases_sweep.json` finalize note says process killed rc=-15, 7/8 rows valid, single-ckpt live values carry `+/-0.0002` noise, and best live ep1349 `0.0037233` is noise-caveated. I did not find an uncaveated downstream citation in the targeted current `.md`/`.json`/`.jsonl` searches. |

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

## MEASURED-RUNNABILITY AND MEASURED-SCORED-QUANTITY AXIS

- MEASURED from existing artifacts: both v3 smokes executed at `num_pairs=600`, `batch_pairs=4`, emitted gates, and preserved checkpoints. The full-window retry launched with `.venv/bin/python` prepended.
- NOT MEASURED by this arm: peak RSS/VRAM, any new scorer output, any archive bytes, contest CPU/CUDA score, or full-window endpoint.
- Measured scored quantity in scope: n36 advisory `d_seg`/`d_pose` through the existing smoke probe only. Rate sensitivity used telemetry counted bytes, not an exact archive.

## BOUNDARIES

- No `.py` file was edited; no review override was used.
- No protected file was touched.
- The existing dirty worktree was left intact.
- Live full-v3 SSD artifacts were read-only.
- The scorer slot remained with the live full-v3 window.
- The contest pointer remains borrowed/unmoved.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
