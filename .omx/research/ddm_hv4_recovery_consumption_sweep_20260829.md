# ddm_hv4 recovery-consumption sweep — plan surface drained; no unowned FIRE-NOW head

Date: 2026-08-29  
Actor: `ddm_hv4`  
Authority: source/receipt, Git-custody, queue-state, and memo↔task join audit; scorer-free

## Outcome

The exact frontier did **not** move. The landed extractor still exposed **357 / 357 live
`NEXT_IF_RESUMED` plan rows** at the frozen input snapshot. Every row now has one typed, owned exit:

| Exit | Rows | Meaning |
|---|---:|---|
| `FIRED` | 63 | A later source, receipt, task, or current-object artifact already consumed the row. |
| `FOLDED` | 28 | A named later row or current-object successor subsumes the source order. |
| `QUEUED-W-FIRE-ORDER` | 7 | The action remains live with owner, consumer store, and explicit trigger. |
| `CLOSED` | 11 | A scoped negative or terminal cancellation closes this formulation. |
| `STALE` | 248 | The precondition moved; the historical instruction cannot be carried onto GB1/post-FCD2 without re-derivation. |
| **Total** | **357** | Complete plan-surface denominator; **0 unowned**. |

The ranked score-moving `FIRE-NOW` head is **empty: 0 / 357**. The current material opening is
already owned by `ddm_fcd3`, not hidden in the plan surface: its real re-encode retained
`-2,965 B` versus JT21 at the `tau_1e-6` rung, while the fresh selected-body pose solve and publish
gate remain in progress. The only immediate non-score action is SW1 credential revocation, owned by
the credential operator. Therefore the charter's prior-law prediction is **falsified on this frozen
357-row surface**: hv4 found no measured opening of at least 30 B or `1e-5 S` that was both live and
unowned.

## Frozen authority and boundaries

Snapshot time: `2026-08-29T20:39:52Z`.

| Input | Bytes | SHA-256 | Use |
|---|---:|---|---|
| `.omx/state/codex_arm_queue.next_if_resumed.jsonl` | 628,762 | `4842bcf3d6aacbe39c5ef038757038062e121d1ef722153d718b838aef1afdb9` | Landed extractor authority; 357 live of 360 total, 3 superseded, 1 amend-required. |
| `.omx/state/canonical_task_status.jsonl` | 837,902 | `bf9cf49a1522e3aa95296cc7851aa0ace795e7f91d3e5315206383cb077a648d` | Latest task rows for the #880 join. |
| `ddm_hv2_harvest_consumption_ledger_20260826.jsonl` | 100 rows | `b883f380de9e0aaef5ca9c2614417606adb54dcd4ceb30dae6de7343a9cb15e3` | Prior exit baseline plus hv3 post-baseline receipt rows. |
| `au1_corrections_index.jsonl` | 4,220,615 | `ec3fe2415e327c3658b8a864eed0993db68d6ab023026a38b75d5a8e96b72c11` | Stale-headline instrument; never treated as verdict authority. |
| `.omx/state/graph_memory/nodes.jsonl` | 18,137,447 | `ffb0f0c5e0d667af470d9b617aaa7a5eea5cbafc91ee9ce8ef8fbf449a99dd20` | Frozen graph-presence check. |

The correction index had 223 candidate correction rows across 120 / 355 distinct plan-source files.
Those are occurrences, not 223 facts. Each disposition used the retained source block and a later
receipt/task/current-object join; no memo headline or correction-keyword occurrence was promoted to a
verdict.

The retained evidence root is
`/Volumes/APDataStore/pact/ddm_hv4_recovery_consumption_sweep/`; its machine-readable inventory is
`HV4_RECEIPT_MANIFEST.json`. The manifest records each retained file's bytes and SHA-256 and preserves
the reduced SSD-census scope and fire trigger rather than certifying an unmeasured zero.

## Orphan recovery

### Repository corpus delta from OS1

The current repository population is **9,411 present research Markdown files**. Its full nontracked
candidate set is **36 / 9,411**:

| Class | Rows | Graph-absent | Signal/scratch disposition |
|---|---:|---:|---|
| Ordinary untracked Markdown | 1 | 1 | `ddm_fcd3_pose_screened_reselection_20260829.md` is active owned signal. It is corpus-present, graph-cache-newer, and folded to the current fcd3 arm; it is not an ownerless orphan. |
| Ignored untracked Markdown | 35 | 0 | The unchanged OS1 raw artifact/extraction class; `FOLDED_OS1_CERTIFIED_EXCLUDED`, never force-added or deleted. |
| **True unowned orphan signal** | **0** | **0** | Drained on this repository scope. |

Per-file path, SHA-256, bytes, birth mtime, graph presence, signal/scratch classification, owner,
consumer, and trigger are in
`/Volumes/APDataStore/pact/ddm_hv4_recovery_consumption_sweep/orphan_recovery_receipt.jsonl`
(`36 / 36` rows, SHA-256
`9dfb1f7dd19a673bd42f369ba0d42905b70d92f999f97cca0b832d8b73adc91d`). No file was moved or
deleted. The OS1 26-row ordinary-untracked baseline was not re-drained; its already-exited rows are
folded, and the single current delta birth is fcd3.

### SSD-tier delta

This leg is a **scope reduction, not a zero**. A read-only full metadata walk completed on
`/Volumes/APDataStore/pact`: **213,129 files**, **29,144 directories**, and **14,361 broad
candidates** (Markdown plus receipt/manifest/certificate JSON born after the OS1 cutoff). The scan
did not durably emit its per-file ledger before the subsequent Vertigo walk stalled. Two narrower
incremental/file-list retries also failed to produce a complete durable denominator and were stopped
instead of adding further metadata contention to the active fcd3 workload.

Therefore AP has an aggregate denominator only, the Vertigo denominator is **NOT MEASURED**, and the
SSD-tier true-orphan count is **UNKNOWN / NOT ADJUDICATED**. No SSD file was moved or deleted, and hv4
does not claim that the cross-tier orphan class is drained. The exact deferred action is
`BLOCKED-WITH-FIRE-ORDER`: owner `next hv4 custody successor`; consumer store
`/Volumes/APDataStore/pact/ddm_hv4_recovery_consumption_sweep/`; trigger `fcd3 lane terminal and both
volume metadata paths responsive`; action `rerun a per-root incremental scan that journals each row
before crossing to the next volume`.

## Plan-surface drain

The full machine ledger is
`.omx/research/ddm_hv4_recovery_consumption_ledger_20260829.jsonl`: **357 / 357 rows**, 357 unique
extractor row IDs, source SHA equality **357 / 357**, no missing source, no invalid exit type, and no
owner equal to blank/`none`/`unowned`. Its SHA-256 is
`82f2bb3402c63daa81180203b958153b3b6ad29d4a26f7a50f504b6f5add6588`; the byte-identical retained
copy is `/Volumes/APDataStore/pact/ddm_hv4_recovery_consumption_sweep/plan_exit_ledger.jsonl`.

The large `STALE` class is not a family kill. It says only that a plan written before RC2/GB1, or
before the post-FCD2 selected-body change, cannot be fired from carried prose. Its fire condition is
fresh re-derivation on the live object. Scoped negative rows are `CLOSED`; actual later consumers are
`FIRED` or `FOLDED`; seven still-valid actions retain explicit orders.

### Full 357-row exit table

| # | Arm | Exit | Owner | Consumer / evidence | Trigger or reason |
|---:|---|---|---|---|---|
| 1 | `tc1` | `CLOSED` | MAIN history custodian | `hv4 ledger` | source retained block explicitly closes rerun |
| 2 | `au1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 3 | `dw1` | `CLOSED` | MAIN history custodian | `hv4 ledger` | source retained block explicitly closes rerun |
| 4 | `pj1` | `CLOSED` | MAIN history custodian | `hv4 ledger` | source retained block explicitly closes rerun |
| 5 | `su2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 6 | `np1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 7 | `al1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 8 | `am1` | `FIRED` | canonical task owner | `.omx/state/canonical_task_status.jsonl` | exact task-ledger terminal join |
| 9 | `bd1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 10 | `bf1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 11 | `ca1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 12 | `cf1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 13 | `ci1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 14 | `ek1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 15 | `ig1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 16 | `od4` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 17 | `od5` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 18 | `od6` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/ddm_od6_20260805/NEXT_IF_RESUMED.md` | later same-name row b06a9f997e45 subsumes this row |
| 19 | `od7` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/ddm_od7_20260805/NEXT_IF_RESUMED.md` | later same-name row ca5c4ccfdbd1 subsumes this row |
| 20 | `od8` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 21 | `pe1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 22 | `pe2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 23 | `pe3` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 24 | `pe4` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 25 | `sd1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 26 | `sj1` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/ddm_sj1_20260805/SJ1_CROSSWALK_RECEIPT.md` | later same-name row fdadb71facc1 subsumes this row |
| 27 | `sj1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 28 | `st1` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/ddm_st1_20260805/NEXT_IF_RESUMED.md` | later same-name row 525c8f1916d5 subsumes this row |
| 29 | `na3` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 30 | `od2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 31 | `od3` | `CLOSED` | MAIN history custodian | `hv4 ledger` | source retained block explicitly closes rerun |
| 32 | `od6` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 33 | `od7` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 34 | `st1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 35 | `st2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 36 | `tj1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 37 | `kt1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 38 | `uf1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 39 | `jb1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 40 | `ddm_gc19` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 41 | `ddm_cx1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 42 | `ddm_na5` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/arm_final_messages/ddm_na5_20260809T122047Z.md` | later same-name row 211e645bad74 subsumes this row |
| 43 | `ddm_gc20` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 44 | `ddm_rr1` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/arm_final_messages/ddm_rr1_20260809T151025Z.md` | later same-name row a0df1979347f subsumes this row |
| 45 | `ddm_rr1` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/arm_final_messages/ddm_rr1_20260809T151025Z.md` | later same-name row a0df1979347f subsumes this row |
| 46 | `ddm_rr1` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/arm_final_messages/ddm_rr1_20260809T151025Z.md` | later same-name row a0df1979347f subsumes this row |
| 47 | `ddm_rr1` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/arm_final_messages/ddm_rr1_20260809T151025Z.md` | later same-name row a0df1979347f subsumes this row |
| 48 | `gc20` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 49 | `dy2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 50 | `fx1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 51 | `ed2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 52 | `ce1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 53 | `tq1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 54 | `wl1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 55 | `us2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 56 | `lt1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 57 | `na6` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 58 | `hv1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | the exact task join spans a later census and an older terminal receiver rather than one live arm row |
| 59 | `if1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 60 | `lw1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 61 | `ty2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 62 | `ddm_eh1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 63 | `ddm_hb1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 64 | `ddm_mx2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 65 | `ddm_cons1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 66 | `ddm_fa1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 67 | `ddm_et6` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 68 | `ddm_aa1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 69 | `ddm_cf2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 70 | `ddm_cr1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 71 | `ddm_gc21` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 72 | `ddm_hb2_hpac_pack_roundtrip` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 73 | `ddm_lx1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 74 | `ddm_m1c1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 75 | `ddm_m1r5a` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 76 | `ddm_m1r5b` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 77 | `ddm_m1r5c` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 78 | `ddm_ng1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 79 | `ddm_oh1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 80 | `ddm_rr17` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 81 | `ddm_rr18` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 82 | `ddm_rv2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 83 | `ddm_tr2p1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 84 | `ddm_wc2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 85 | `ddm_zc1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 86 | `ddm_mp2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 87 | `ddm_na5` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/arm_final_messages/ddm_na5_20260809T122047Z.md` | later same-name row 211e645bad74 subsumes this row |
| 88 | `ddm_na5` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 89 | `ddm_ax2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 90 | `ddm_pp2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 91 | `ddm_nb2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 92 | `ddm_cb2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 93 | `ddm_pq1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 94 | `ddm_rr1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 95 | `ddm_rr2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 96 | `ddm_rr3` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 97 | `ddm_rr4` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 98 | `ddm_fx4` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 99 | `ddm_fx1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 100 | `ddm_fx2` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/arm_final_messages/ddm_fx2_20260809T233625Z.md` | later same-name row 64e89e747773 subsumes this row |
| 101 | `ddm_fx5` | `FIRED` | MAIN/current successor | `Pact HEAD` | later source/receipt consumes this row |
| 102 | `ddm_fx3` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 103 | `ddm_fx5b` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 104 | `ddm_rr1_v7v18_recall` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 105 | `ddm_rc1_receiver` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 106 | `ddm_pk2` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/arm_final_messages/ddm_pk2_20260809T222049Z.md` | later same-name row f72d7b3dafb0 subsumes this row |
| 107 | `ddm_pk2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 108 | `ddm_pk2r_impl` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 109 | `ddm_cl1_capacity` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 110 | `ddm_sd1_semantic` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 111 | `ddm_dv1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 112 | `ddm_fx2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 113 | `ddm_dt1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 114 | `ddm_ap1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 115 | `ddm_cx2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 116 | `ddm_tm1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 117 | `ddm_pr1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 118 | `ddm_sm3` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 119 | `ddm_sr1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 120 | `ddm_sg2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 121 | `ddm_cp2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 122 | `ddm_vp1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 123 | `ddm_lt1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 124 | `ddm_hp3` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 125 | `ddm_sd2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 126 | `ddm_vh2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 127 | `ddm_rc2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 128 | `ddm_ai1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 129 | `ddm_sd2r` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 130 | `ddm_vh2r` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 131 | `ddm_sm4` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 132 | `ddm_pz2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 133 | `ddm_sv3` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 134 | `ddm_pz3` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 135 | `ddm_gp2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 136 | `ddm_hm1` | `FIRED` | canonical task owner | `.omx/state/canonical_task_status.jsonl` | exact task-ledger terminal join |
| 137 | `ddm_lc2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 138 | `ddm_ah2_arm_harvest` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 139 | `ddm_pi135_pr135_intake` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 140 | `ddm_pi136_leaderboard_breadth` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 141 | `ddm_fd135_fractal_decomposition` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 142 | `ddm_cp135_rate_compose` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 143 | `ddm_ps135_pose_resolve` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 144 | `ddm_rc64p_native_cpu_decode` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 145 | `ddm_lp135_lossless_pack` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 146 | `ddm_pz4p_pose_gauge_preproof` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 147 | `ddm_sr1_implicit_edge_conditioning` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 148 | `ddm_pz4r_pgq1_receiver` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 149 | `ddm_ps135b_leg_a_fire` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 150 | `ddm_cn4_arc_consolidation` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 151 | `ddm_na6_arc_negative_audit` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 152 | `ddm_hy1_capstone_hybrid` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 153 | `ddm_hr1_realization_engineering` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 154 | `ddm_hr2_prestage_build` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 155 | `ddm_rvs1_realization_survival_harvest` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 156 | `ddm_rvs2_geometry_survival_crosswalk` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 157 | `ddm_lv2_terminal_campaign_completeness` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 158 | `ddm_ip1_decision_integrity_ports` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 159 | `ddm_t0r1_intake_rehearsal` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 160 | `ddm_rho1_survival_prior_harvest` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 161 | `ddm_sr2_vertigo_space_reclaim` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 162 | `ddm_t1r1_container_build_rehearsal` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 163 | `ddm_js1_global_joint_solve` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 164 | `ddm_js2_implicit_edge_conditioning` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 165 | `ddm_js2b_edge_conditioning_relative_gauge` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 166 | `ddm_js3_learned_implicit_conditioning` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 167 | `ddm_js4_pose_null_projected_conditioning` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 168 | `ddm_js5_projector_distilled_conditioning` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 169 | `ddm_tf1_theoretical_floor_and_beyond` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 170 | `ddm_xi1_screw_conditioned_learned_prior` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 171 | `ddm_ec1_event_coordinate_producer` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 172 | `ddm_xi1f_leg_a_pack_schema_fix` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 173 | `ddm_js6_event_proposal_acceptance` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 174 | `ddm_se1_shipping_axis_survival_resolve` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 175 | `ddm_js7_acceptance_sweep_and_compose` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 176 | `ddm_xi2_xi_context_full_scale_promotion` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 177 | `ddm_dg1_pinkall_elastica_willmore_crosswalk` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 178 | `ddm_eu3_fresh_eyes_eureka_hunt` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 179 | `ddm_hp4_frame_embedding_prediction` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 180 | `ddm_ec2_sparse_event_hpac_conditioning` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 181 | `ddm_jo1_joint_probability_object` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 182 | `ddm_vd1_modal_batch_event_validator` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 183 | `ddm_vd1b_worker_adapted_decode_fix` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 184 | `ddm_cp5v_compose_five_validated_events` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 185 | `ddm_gv2_lane_road_grammar_v2` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 186 | `ddm_js1b_cuda_argmax_field_materializer` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 187 | `ddm_sa1_shipping_axis_seg_actuator` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 188 | `ddm_po1_t4_error_feedback_pose_compensation` | `CLOSED` | canonical task owner | `.omx/state/canonical_task_status.jsonl` | exact task-ledger terminal join |
| 189 | `ddm_hv1_fresh_eyes_hybrid_corpus_pass` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 190 | `ddm_pz4r_full_n600_eval` | `CLOSED` | canonical task owner | `.omx/state/canonical_task_status.jsonl` | exact task-ledger terminal join |
| 191 | `ddm_re1_realization_engineered_candidate` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 192 | `ddm_hg1_named_followons` | `FIRED` | canonical task owner | `.omx/state/canonical_task_status.jsonl` | exact task-ledger terminal join |
| 193 | `ddm_cn5_arc_consolidation` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 194 | `ddm_re1x_round1_full_n600_eval` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 195 | `ddm_vh2_vehicle_harvest_drain` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 196 | `ddm_vh3_ms_family_drain` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 197 | `ddm_re1t_t4_gate_prep` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 198 | `ddm_js6_seg_representation_join` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 199 | `ddm_pr135ps_truncated_search_resume` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 200 | `ddm_js6b_pose_screened_compile` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 201 | `ddm_qs1_frame0_schur_coupled_solve` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 202 | `ddm_sc3_storage_custody_move` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 203 | `ddm_qs2_compensation_rate_rung` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 204 | `ddm_gca1` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 205 | `ddm_qs3` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 206 | `ddm_qs4` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 207 | `ddm_eu4` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 208 | `ddm_qs5` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 209 | `ddm_pk3` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 210 | `ddm_pk4` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 211 | `ddm_js8` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 212 | `ddm_js1c_cuda_custody_stage0` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 213 | `ddm_ec1_implicit_edge_conditioning` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 214 | `ddm_ec2_oriented_adapter_trainer` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 215 | `ddm_rfo1_fresh_hybrid_compose` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 216 | `ddm_bg1_bilinear_gate_pricing` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 217 | `ddm_bg2_postmortem_execute` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 218 | `ddm_mc35_micro35_union_build` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 219 | `ddm_ac1_automatic_endpoint_closure` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 220 | `ddm_dt1_repeated_lesson_determinizer` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 221 | `ddm_mt1_978_multitoken_screen` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 222 | `ddm_mc36_micro35_variants` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 223 | `ddm_gs1_gestalt_signal_census` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 224 | `ddm_fs1_fire_seal_adapters` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 225 | `ddm_f26p_runtime_cpu_lift` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 226 | `ddm_f26q_rc64_native_lowering` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 227 | `ddm_f26r_hpac_final_rung` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 228 | `ddm_js8_implicit_edge_conditioning` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 229 | `ddm_na7_negative_signal_audit` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 230 | `ddm_dio1_dion3_crosswalk` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 231 | `ddm_dr1_dispatch_infra_repair` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 232 | `ddm_wc1_hpac_trainer_throughput_port` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 233 | `ddm_lh1_watched_launch_hardening` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 234 | `ddm_wc2_hpac_mps_port_build` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 235 | `ddm_mz1_model_section_rate_race` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 236 | `ddm_lh2_continuation_launch_determinizer` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 237 | `ddm_mz2_frozen_section_representation_attack` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 238 | `ddm_pq1_submission_packet_prep` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 239 | `ddm_rfo2_fresh_eyes_gestalt_synergy` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 240 | `ddm_wd2_width_distillation_build` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 241 | `ddm_av1_wd2_earlystop_adversarial_review` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 242 | `ddm_av2_fresh_eyes_and_distillation_reopen` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 243 | `ddm_wd3_scorer_aware_width_distillation` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 244 | `ddm_mp2_mixed_precision_receiver_close` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 245 | `ddm_wc1_advisory_decode_wallclock` | `STALE` | MAIN current-object router | `.omx/state/main_hot_state.md` | source predates RC2/GB1 and has no exact current task join |
| 246 | `ddm_gt2` | `FIRED` | MAIN history custodian | `git:HEAD` | git:HEAD |
| 247 | `ddm_rvf1` | `FIRED` | MAIN history custodian | `git:HEAD` | git:HEAD |
| 248 | `ddm_rv16_round3_finding_wave` | `FIRED` | MAIN history custodian | `git:HEAD` | git:HEAD |
| 249 | `ddm_rc2_composed_clean_decode_and_seal` | `FIRED` | MAIN history custodian | `.omx/state/canonical_frontier_pointer.json` | .omx/state/canonical_frontier_pointer.json |
| 250 | `ddm_pq9_pr_final_polish` | `STALE` | MAIN history custodian | `.omx/state/main_hot_state.md + .omx/state/canonical_frontier_pointer.json` | .omx/state/main_hot_state.md + .omx/state/canonical_frontier_pointer.json |
| 251 | `ddm_sw1_portable_paths_secrets_scrub` | `QUEUED-W-FIRE-ORDER` | credential owner/operator | `.omx/state/operator_p0_ledger.jsonl` | immediately rotate or revoke the historical GCP/JWT credentials and record revocation evidence without secret values |
| 252 | `ddm_pq10_codex_packet_review_round` | `STALE` | MAIN history custodian | `.omx/state/main_hot_state.md + .omx/state/canonical_frontier_pointer.json` | .omx/state/main_hot_state.md + .omx/state/canonical_frontier_pointer.json |
| 253 | `ddm_dj1_dual_lineage_carrier` | `FIRED` | MAIN history custodian | `git:HEAD` | git:HEAD |
| 254 | `ddm_r012_rate_representation` | `FIRED` | MAIN history custodian | `.omx/state/canonical_frontier_pointer.json` | .omx/state/canonical_frontier_pointer.json |
| 255 | `ddm_jo1_joint_objective_design` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md#row-1` | .omx/research/ddm_no1_new_object_derivation_20260826.md#row-1 |
| 256 | `ddm_gs3_unbridled_gestalt` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 257 | `ddm_jo1u_payload_unblock` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md#row-1` | .omx/research/ddm_no1_new_object_derivation_20260826.md#row-1 |
| 258 | `ddm_dc1_decode_time_compute` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 259 | `ddm_jo1u2_materializer_cure` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md#row-1` | .omx/research/ddm_no1_new_object_derivation_20260826.md#row-1 |
| 260 | `ddm_wd4_warm_lineage_width` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 261 | `ddm_dc1s_sparse_grid_sweep` | `FIRED` | MAIN history custodian | `git:HEAD` | git:HEAD |
| 262 | `ddm_jo2_solve_reseal` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md#row-1` | .omx/research/ddm_no1_new_object_derivation_20260826.md#row-1 |
| 263 | `ddm_jo3_entrypoint_and_final_reseal` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md#row-1` | .omx/research/ddm_no1_new_object_derivation_20260826.md#row-1 |
| 264 | `ddm_jo4_certified_retention_reseal` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md#row-1` | .omx/research/ddm_no1_new_object_derivation_20260826.md#row-1 |
| 265 | `ddm_dx2_cabac_receiver_fold` | `FIRED` | MAIN history custodian | `.omx/state/canonical_frontier_pointer.json` | .omx/state/canonical_frontier_pointer.json |
| 266 | `ddm_jo5_determinism_cure_reseal` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md#row-1` | .omx/research/ddm_no1_new_object_derivation_20260826.md#row-1 |
| 267 | `ddm_es1_end_state_characterization` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 268 | `ddm_ws0_worldsheet_grammar_price` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 269 | `ddm_wc2_jo1_wallclock` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 270 | `ddm_ig1_implicit_carriage_gestalt` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 271 | `ddm_nt1_naive_toy_generic_audit` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 272 | `ddm_ws1_optimal_worldsheet_grammar` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 273 | `ddm_jo6_receiver_container_compat` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 274 | `ddm_tl1_teacher_ledger` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 275 | `ddm_ht1_hard_tail_student` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 276 | `ddm_nr1_taskcell_quotient_prebuild` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/arm_final_messages/ddm_nr1_taskcell_quotient_prebuild_20260822T151557Z.md` | later same-name row 32f807955335 subsumes this row |
| 277 | `ddm_rb1_rate_bound_decomposition` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 278 | `ddm_ec2_collateral_suppressed_conditioner` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 279 | `ddm_xt1_exact_solve_teacher_student` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 280 | `ddm_nl1_never_fired_levers` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 281 | `ddm_db1_decode_boundary_families` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 282 | `ddm_vf1_evaluator_visible_floor` | `FOLDED` | MAIN same-arm history custodian | `.omx/research/arm_final_messages/ddm_vf1_evaluator_visible_floor_20260822T150521Z.md` | later same-name row c06df46a2089 subsumes this row |
| 283 | `ddm_jx1_joint_exchange_envelope` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 284 | `ddm_vf1_evaluator_visible_floor` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 285 | `ddm_rc1_rate_crush` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 286 | `ddm_nr1_taskcell_quotient_prebuild` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 287 | `ddm_os1_orphan_signal_reconciliation` | `FIRED` | MAIN history custodian | `git:HEAD` | git:HEAD |
| 288 | `ddm_ni1_nr1_k32_receiver_distortion` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 289 | `ddm_cb2_class_balanced_dictionary` | `FIRED` | MAIN history custodian | `.omx/research/ddm_tb2_token_bit_attribution_20260823.md` | .omx/research/ddm_tb2_token_bit_attribution_20260823.md |
| 290 | `ddm_ad2_addressing_cost_decomposition` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 291 | `ddm_lq1_lane_quotient_representability` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 292 | `ddm_to2_token_ordering_race` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 293 | `ddm_cx3_context_axis_ceiling` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 294 | `ddm_ef1_token_entropy_floor` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 295 | `ddm_ms9_dx2_seg_manufactured_fraction` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 296 | `ddm_bl1_per_position_bit_allocation` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 297 | `ddm_lx2_lane_bit_budget_exchange` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 298 | `ddm_ae1_anti_predicted_excess` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 299 | `ddm_xs1_cross_section_conditioning` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 300 | `ddm_mst1_manufactured_stage_split` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 301 | `ddm_ld1_lane_lossy_drop_exchange` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 302 | `ddm_ar1b_archive_residue_purchase` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 303 | `ddm_oe1_online_escape_member` | `FIRED` | MAIN history custodian | `git:HEAD` | git:HEAD |
| 304 | `ddm_wj1_cost_error_position_join` | `CLOSED` | MAIN history custodian | `hv4 ledger` | later receipt closes the source formulation |
| 305 | `ddm_rx3_receiver_precompensation` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 306 | `ddm_jf1_joint_field_model_refit` | `FIRED` | MAIN/current successor | `Pact HEAD` | later source/receipt consumes this row |
| 307 | `ddm_ap1_residue_purchase_scorer` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 308 | `ddm_rj1_renderer_joint_move` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 309 | `ddm_sy2_composition_synergy_deep_pass` | `FIRED` | MAIN/current successor | `Pact HEAD` | later source/receipt consumes this row |
| 310 | `ddm_ny1_live_lineage_toy_and_reactivation_audit` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 311 | `ddm_et1_edge_topology_container_gate` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 312 | `ddm_hg1_heterogeneous_analytic_generator_gate` | `FIRED` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 313 | `ddm_mp3_hpac_member_prune` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 314 | `ddm_na12_post_sy2_negative_regrade` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 315 | `ddm_tb2_token_bit_attribution` | `QUEUED-W-FIRE-ORDER` | MAIN-designated CB2 task-weighted K2048 successor | `/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/reactivated_task_weighted_refit/` | TB2 verification and handoff hashes match; the source fire trigger is MET |
| 316 | `ddm_mf1_manufactured_seg_repair` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 317 | `ddm_hr3_residual_implicit_carrier` | `STALE` | MAIN history custodian | `.omx/research/ddm_no1_new_object_derivation_20260826.md` | .omx/research/ddm_no1_new_object_derivation_20260826.md |
| 318 | `ddm_s1_trained_renderer_diagonal` | `FIRED` | MAIN history custodian | `.omx/research/ddm_s1e_stage_a_off_floor_verdict_20260825.md` | .omx/research/ddm_s1e_stage_a_off_floor_verdict_20260825.md |
| 319 | `ddm_rj2_joint_renderer_object_change` | `FIRED` | MAIN history custodian | `.omx/research/ddm_s1e_stage_a_off_floor_verdict_20260825.md` | .omx/research/ddm_s1e_stage_a_off_floor_verdict_20260825.md |
| 320 | `ddm_s1a_stage_a_adapter` | `FIRED` | MAIN history custodian | `.omx/research/ddm_s1e_stage_a_off_floor_verdict_20260825.md` | .omx/research/ddm_s1e_stage_a_off_floor_verdict_20260825.md |
| 321 | `ddm_s1e_off_floor_adjudicator` | `FIRED` | MAIN history custodian | `.omx/research/ddm_s1e_stage_a_off_floor_verdict_20260825.md` | .omx/research/ddm_s1e_stage_a_off_floor_verdict_20260825.md |
| 322 | `ddm_wh1_wrong_half_decomposition` | `FIRED` | MAIN history custodian | `.omx/research/ddm_s1e_stage_a_off_floor_verdict_20260825.md` | .omx/research/ddm_s1e_stage_a_off_floor_verdict_20260825.md |
| 323 | `ddm_cc2_catalog_consolidation` | `QUEUED-W-FIRE-ORDER` | MAIN plus operator | `docs/meta_bug_class_catalog.md + src/tac/preflight.py` | operator approves the named catalog replacement/consolidation set |
| 324 | `ddm_cm1_coder_matched_surrogate` | `QUEUED-W-FIRE-ORDER` | MAIN | `.omx/state/canonical_task_status.jsonl::ddm_no1_row1_three_term_objective` | a restartable F26/HPAC exact-increment cache validates rank correlations at stratified-random n>=32, or Metal plus the outstanding wd3 seed control becomes available |
| 325 | `ddm_d3_alphabet_merge` | `FIRED` | MAIN history custodian | `.omx/research/ddm_d3a_analytic_lane_carrier_20260826.md + experiments/ddm_d3b_lossless_lane_factorization.py` | .omx/research/ddm_d3a_analytic_lane_carrier_20260826.md + experiments/ddm_d3b_lossless_lane_factorization.py |
| 326 | `ddm_d3a_analytic_lane_carrier` | `STALE` | MAIN history custodian | `experiments/ddm_d3b_lossless_lane_factorization.py + .omx/state/main_hot_state.md` | experiments/ddm_d3b_lossless_lane_factorization.py + .omx/state/main_hot_state.md |
| 327 | `ddm_pc2_pose_carrier_live_remainder` | `FIRED` | MAIN history custodian | `/Volumes/APDataStore/pact/ddm_pc2_commit.uRh8we/ddm_pc2_b9ae14c5bc.bundle` | Pose/carrier drain found no live current-body candidate; D3B later made the post-reaccount representation rate-negative. |
| 328 | `ddm_hv2_harvest_consumption_sweep` | `FOLDED` | MAIN/current successor | `.omx/research/ddm_hv2_harvest_consumption_ledger_20260826.jsonl` | later current-object receipt subsumes this row |
| 329 | `ddm_d3c_class_pyramid_peel_order` | `CLOSED` | MAIN history custodian | `hv4 ledger` | later receipt closes the source formulation |
| 330 | `ddm_or1_orthogonal_representation_regime` | `QUEUED-W-FIRE-ORDER` | MAIN boundary-grammar successor | `/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/global_region_grammar_reference/` | QUEUED-W-FIRE-ORDER-double_decode_exact_le_47696B |
| 331 | `ddm_jf2_terminal_diagonal_harvest` | `FIRED` | MAIN history custodian | `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/BYTE_DIAGONAL_TERMINAL.json` | Three byte winners were real, but MAIN later measured all ordered advisory rows and closed the trained diagonal; WJ1 is mooted. |
| 332 | `ddm_d3b_lossless_lane_factorization` | `QUEUED-W-FIRE-ORDER` | MAIN | `/Volumes/APDataStore/pact/ddm_d3b_lossless_lane_factorization/followons/fr0/` | QUEUED-W-FIRE-ORDER-body_le_64000_or_64080_integrated |
| 333 | `ddm_w96a_aligned_config_renderer_window` | `FOLDED` | MAIN/current successor | `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/` | later current-object receipt subsumes this row |
| 334 | `ddm_hd1_apparatus_two_landings` | `FIRED` | MAIN/current successor | `Pact HEAD` | later source/receipt consumes this row |
| 335 | `ddm_bs3_born_small_resolved_carrier` | `CLOSED` | MAIN history custodian | `hv4 ledger` | later receipt closes the source formulation |
| 336 | `ddm_fb2_route_table_gb1` | `FOLDED` | MAIN/current successor | `.omx/state/main_hot_state.md` | later current-object receipt subsumes this row |
| 337 | `ddm_w96b_aligned_loss_implementation` | `FOLDED` | MAIN/current successor | `.omx/state/main_hot_state.md` | later current-object receipt subsumes this row |
| 338 | `ddm_bs4_born_small_stage_fire` | `CLOSED` | MAIN history custodian | `hv4 ledger` | later receipt closes the source formulation |
| 339 | `ddm_mg1_mps_gate_burndown` | `FIRED` | MAIN/current successor | `Pact HEAD` | later source/receipt consumes this row |
| 340 | `ddm_rb1_born_small_renderer_build` | `QUEUED-W-FIRE-ORDER` | MAIN renderer-training successor | `/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/next_renderer_born_small/` | QUEUED-W-FIRE-ORDER-after_both_W96_seeds_and_fresh_Metal_scorer_claims |
| 341 | `ddm_fc1x_serializer_fatclone_cure` | `FIRED` | MAIN/current successor | `Pact HEAD` | later source/receipt consumes this row |
| 342 | `ddm_pf2x_preflight_chain_burndown` | `FIRED` | MAIN/current successor | `Pact HEAD` | later source/receipt consumes this row |
| 343 | `ddm_sr3_ap_certify_compress_reclaim` | `FIRED` | MAIN/current successor | `Pact HEAD` | later source/receipt consumes this row |
| 344 | `ddm_pf4x_bare_round_burndown` | `FIRED` | MAIN/current successor | `Pact HEAD` | later source/receipt consumes this row |
| 345 | `ddm_bs4x_stage0_cure_and_stage_fire` | `CLOSED` | MAIN history custodian | `hv4 ledger` | later receipt closes the source formulation |
| 346 | `ddm_hv3_done_arm_consumption` | `FOLDED` | MAIN/current successor | `.omx/state/main_hot_state.md` | later current-object receipt subsumes this row |
| 347 | `ddm_bd1_decode_time_structure` | `FOLDED` | MAIN/current successor | `.omx/research/ddm_no2_quotient_born_object_20260827.md` | later current-object receipt subsumes this row |
| 348 | `ddm_no2_quotient_born_object` | `FOLDED` | MAIN/current successor | `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/` | later current-object receipt subsumes this row |
| 349 | `ddm_qbw1_builder_first_rung` | `FOLDED` | MAIN/current successor | `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/` | later current-object receipt subsumes this row |
| 350 | `ddm_qbw2_temporal_bound` | `FOLDED` | MAIN/current successor | `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/` | later current-object receipt subsumes this row |
| 351 | `ddm_qbflow_rate_first_rung` | `FOLDED` | MAIN/current successor | `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/` | later current-object receipt subsumes this row |
| 352 | `ddm_qbt1_qbflow_trainer_build` | `FOLDED` | MAIN/current successor | `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/` | later current-object receipt subsumes this row |
| 353 | `ddm_qbt2_class_birth_curriculum` | `FOLDED` | MAIN/current successor | `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/` | later current-object receipt subsumes this row |
| 354 | `ddm_qbt2b_inherited_palette_birth` | `FOLDED` | MAIN/current successor | `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/` | later current-object receipt subsumes this row |
| 355 | `ddm_qbt2b_r7_lane_constrained_margin` | `FIRED` | MAIN/current successor | `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/` | later source/receipt consumes this row |
| 356 | `ddm_fcd1_field_for_coder_diagonal` | `FIRED` | MAIN/current successor | `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/` | later source/receipt consumes this row |
| 357 | `ddm_fcd2_distortion_legs_execute` | `FOLDED` | MAIN/current successor | `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/fcd3_pose_screened_reselection/` | later current-object receipt subsumes this row |

## Memo↔task ledger join gap

The two stores still lack a shared primary key. Using only exact source equality, explicit numeric
task ID, exact task/session name, or the arm slug at a path/ID token boundary:

| Join result | Rows | Denominator |
|---|---:|---:|
| Plan rows with at least one exact task join | 26 | 357 plan rows |
| Plan-only rows | 331 | 357 plan rows |
| Latest task rows reached by those joins | 58 | 275 latest task rows |
| Task-only rows | 217 | 275 latest task rows |

The full 548-row symmetric difference is retained at
`/Volumes/APDataStore/pact/ddm_hv4_recovery_consumption_sweep/memo_task_join_gap.jsonl`, SHA-256
`0864b4fc823bb3fcf7f409d1b078811856bcf98a555a952ffbc0289656f41214`. No substring/fuzzy join was
allowed to manufacture ownership. This confirms the #880 defect remains structurally live even though
hv4 assigned an exit to every plan row through source/receipt joins.

## Ranked FIRE-NOW head

| Rank | Plan row | Measured opening | Live object | Cheap deciding step | Owner state | Verdict |
|---:|---|---|---|---|---|---|
| — | none | — | — | — | — | **0 qualifying rows / 357** |

The near-heads fail one required predicate:

- `ddm_fcd3` has a real `-2,965 B` re-encode opening, but it is already owned and its exact-object
  fresh solve is running; hv4 must not duplicate or fire it.
- RB1 has a measured 18,811 B complete-archive byte credit on its smaller-renderer body, but its next
  step is a governed training/scorer program rather than a `$0` or cheap falsifier, and fcd3 owns the
  scorer lane.
- TB2, OR1-global-grammar, D3B-FR0, and CM1 retain hypotheses or gates, not a current measured
  same-object opening that can be falsified cheaply now.
- SW1 is immediate P0 security work but is not a score/byte opening and therefore is not laundered
  into the ranked frontier head.

## RECALL EVIDENCE

Recall was not charter-only. It covered:

- the governing contract, `PROGRAM.md`, byte-retention/certify-or-block and NO-FAKE sections,
  `docs/operating_manual_craft_handoff.md`, and `.omx/state/main_hot_state.md`;
- the complete hv2 memo and 100-row hv2/hv3 ledger, OS1's 26/9,119 baseline and producer-side cure,
  WA1's later recovery of the RD2 full memo and GB1 self-correction, and current fcd1/fcd2/fcd3
  primary bodies and retained receipts;
- the landed `codex_arm_queue.py next` resolver including retraction debt, all 357 retained plan
  blocks, current canonical task status, and the AU1 corrections index;
- content queries over `.omx/research/`, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks,
  charters/design docs, and canonical equations for harvest/consumption, pair-set component pricing,
  real marginal re-encode, and section-coding closure.

Beyond the charter seeds, the material changes were: fcd2's union is instance-refused at 42.96x the
base pose MSE; fcd3 already owns the derived pair-screened route and has retained a real `-2,965 B`
rung; later hv3 receipts consume the JF1/SY2/WJ1 cluster; current ordinary Git-untracked research is
one active fcd3 birth rather than a 42-memo orphan reservoir; and exact memo↔task reachability remains
only 26/357 plan rows. These facts emptied the unowned FIRE-NOW head instead of respawning historical
rows.

## Verification boundary

- Measured here: extractor/task/corpus denominators, source and receipt SHA equality, graph presence,
  the AP aggregate metadata census, typed exits, owners, consumer stores, and exact join gaps.
- Not measured here: a durable per-file SSD-tier census or any Vertigo denominator; that leg is
  explicitly blocked and retained above rather than represented as zero.
- Not measured here: any new archive distortion, scorer component, exact contest score, or frontier
  delta. No scorer, Metal training, Modal, evaluator, or archive mutation ran.
- `upstream/` remained read-only. No retained payload was deleted or moved.
- Worktree changes outside hv4—including the AU1 correction outputs, active lane ledger, untracked
  fcd3/LB1 memos and runners, BHW1 runner, and WD3 runner—were preserved and excluded from the
  landing. LB1/BHW1 were born after the frozen `20:39:52Z` corpus snapshot and are not silently
  absorbed into its denominator.

## NEXT_IF_RESUMED

- **BLOCKED-WITH-FIRE-ORDER** — owner: next hv4 custody successor; consumer store: `/Volumes/APDataStore/pact/ddm_hv4_recovery_consumption_sweep/`; fire trigger: fcd3 is terminal and both volume metadata paths are responsive; run a per-root incremental SSD scan that journals every classified row before crossing volumes.
- **PENDING** — owner: current `ddm_fcd3` arm; consumer store: `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/fcd3_pose_screened_reselection/`; fire trigger: the running `tau_1e-6` public decode completes; finish the fresh selected-body Schur chain and obey its publish gate.
- **QUEUED-W-FIRE-ORDER** — owner: credential owner/operator; consumer store: `.omx/state/operator_p0_ledger.jsonl`; fire trigger: immediately; rotate/revoke the historical SW1 credentials and record non-secret evidence.
- **QUEUED-W-FIRE-ORDER** — owner: MAIN-designated CB2 successor; consumer store: `/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/reactivated_task_weighted_refit/`; fire trigger: TB2 hashes revalidate and a complete receiver can close at or below 137,986 B before any scorer request.
- **QUEUED-W-FIRE-ORDER** — owner: MAIN plus operator; consumer store: `docs/meta_bug_class_catalog.md` and `src/tac/preflight.py`; fire trigger: operator approval of the CC2 replacement/consolidation set.
- **QUEUED-W-FIRE-ORDER** — owner: MAIN; consumer store: `.omx/state/canonical_task_status.jsonl::ddm_no1_row1_three_term_objective`; fire trigger: an exact-increment cache validates CM1 rank correlations at stratified-random `n>=32`, or the named Metal/WD3 control becomes available.
- **QUEUED-W-FIRE-ORDER** — owner: MAIN boundary-grammar successor; consumer store: `/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/global_region_grammar_reference/`; fire trigger: a counted double-decode-exact global grammar reaches at most 47,696 B.
- **QUEUED-W-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_d3b_lossless_lane_factorization/followons/fr0/`; fire trigger: a zero-side-information derivation proves the 64,000/64,080 B body gate.
- **QUEUED-W-FIRE-ORDER** — owner: MAIN renderer-training successor; consumer store: `/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/next_renderer_born_small/`; fire trigger: fcd3 releases the scorer lane, the W96 order is terminal, and fresh Metal/scorer claims plus storage preflight permit the four sealed RB1 configurations.

## LIVE-HYPOTHESES

- FCD3 pair-level pose screening may preserve material coder credit through publish. It is plausible
  because the retained compensated pose excess is heavy-tailed and the real `tau_1e-6` re-encode keeps
  4,194 / 5,268 B positions while retaining 2,965 B of rate credit; the fresh selected-body solve is
  the unresolved authority step.
- RB1's smaller trained renderer may buy a genuine object change after fcd3. It is plausible because
  its receiver-closed body has 18,811 B of measured rate credit, but distortion is unmeasured and the
  first faithful step is not cheap enough for hv4's FIRE-NOW head.
- A stable memo↔task primary key could prevent another 357-row accumulation. It is plausible because
  331 / 357 plan rows and 217 / 275 latest task rows are unmatched even after exact arm-token joins,
  while source/receipt adjudication still recovered ownership for all 357.

## DEAD-ENDS

- Claiming that hv4 drained the SSD orphan class is closed: AP has only an aggregate candidate count,
  Vertigo has no complete denominator, and neither tier has a durable per-file adjudication from this
  run.
- Re-draining OS1's 26 ordinary-untracked births is closed: those rows already exited, and the current
  ordinary delta is one active, owned fcd3 memo.
- Treating Git-untracked as graph-absent is closed: 35 / 36 current nontracked Markdown candidates are
  already in the frozen graph; the one graph miss is the newer active fcd3 body.
- Treating the AU1 correction index's 223 occurrences as 223 corrected facts is closed: they collapse
  to 120 source files and require body/receipt adjudication.
- Carrying a pre-RC2/GB1 fire order onto the live body is closed: 248 rows are stale until re-derived on
  the current object; this is not a family kill.
- Re-firing the FCD2 union or its three batch ladder rows is closed on that instance: the fresh union
  solve missed the pose publish band by 26,710.49x, and fcd3 owns the materially different screened body.
- Claiming a hidden unowned score head is closed on this frozen surface: 0 / 357 rows satisfy measured
  opening, live object, cheap deciding step, and no current owner simultaneously.

Own-vehicle frontier: **GB1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]; unchanged by hv4.**

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

<!-- # FORMALIZATION_PENDING:sister sweep of hv3 over recovered artifacts; same class, same reason. It re-grades consumption status and measures no object. The consumption-rate law it would anchor is not yet derivable for the same reason: two censuses do not make a rate without a matched denominator of arm output between them. -->

**No canonical equation.** Sister sweep of hv3 over recovered artifacts; same class, same reason. It re-grades consumption status and measures no object. The consumption-rate law it would anchor is not yet derivable for the same reason: two censuses do not make a rate without a matched denominator of arm output between them.
