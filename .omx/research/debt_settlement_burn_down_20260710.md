# Debt-settlement burn-down — 2026-07-10 (debt-burn-down executor)

Operator directive (verbatim): *"Pay all owed, burn it down and don't accumulate debt."*

STORES CONSULTED: `.omx/state/deferral_ledger.md` (D1-D20 + this pass's D21-D28) ·
`.omx/state/harness_failure_ledger.jsonl` (all classes) · `.omx/research` 07-08..10 memos
(t5_crucible2/ADVISORY_v752_fresh_eyes · t5_crucible2/ADVISORY_v753_texture_trunk_fresh_eyes ·
t5_crucible3/ADVISORY_v8_fresh_eyes · philosophy_pass_v752 · philosophy_pass_v8 ·
fullstack_fractal_optimal_synthesis · reactivation_campaign_397 · v8_unlock_398a ·
owed16_bounded_ab_and_drystart · DUAL_CHAIN_BRIEF_385) · `tools/check_tac_terminology.py --strict` ·
CLAUDE.md (ANTI-SIGNAL-LOSS / anti-deferral, NO-FAKE, serializer discipline) ·
`docs/operating_manual_craft_handoff.md`.

**Pointer 0.19110 UNMOVED — this pass is hygiene + inventory, not an exact-score mover (means, stated plainly).**

## Counts per bin

| Bin | Count | Meaning |
|---|---:|---|
| **PAY-NOW (settled this pass, $0)** | 1 class (18 findings) | terminology gate → clean, committed |
| **GATED (machine / spec / byte-close / operator)** | 6 | named trigger + owner, re-pointed in ledger |
| **IN-FLIGHT (builder-owned — NOT touched)** | 4 | listed with owner; hard rule honored |
| **Verified-already-closed / re-confirmed** | D3/D4/D10/D20 + D1/D2/D9/D15-D19 armed | no stale trigger, no orphan |
| **Could-not-classify honestly** | 0 | — |

## PAY-NOW — settled this pass

| Item | Source | Action taken | Commit |
|---|---|---|---|
| 18 stale `https://github.com/adpena/tac` public-doc URLs (README + 10 docs) | `tools/check_tac_terminology.py --strict` (warned repeatedly) | repointed to `[comma-lab/src/tac](.../comma-lab/tree/main/src/tac)` — the canonical boundary the gate names + the existing precedent in `comma_pr_archive_dataset_card.md`; `adpena/molt` left alone (real separate repo, not flagged). Gate now **PASSES**. | `ffccb2725` |

## GATED — re-pointed with named trigger + owner (deferral ledger D21-D27 + owed-16 re-status)

| Item | Source | Bin | Named trigger | Owner |
|---|---|---|---|---|
| owed-16 realized directional-basis A/B verdict | owed16_bounded §MEASURED VERDICT | **MEASURED-SETTLED** (was OWED-BLOCKED) | verdict is ROBUST at ep675 (≈0 realized, >70× sep) — only the ep700 ON cell is GATED on ~15 GiB baseline freed (6× governed REFUSE at 119 GiB) → D21, LOW-PRI | owed-16 A/B (built, queue-ready) |
| Dual-chain DOC-gaps: P3 end-to-end tolerance ledger + P12 5-lever composition-sign matrix | philosophy_pass_v752 §P3/§P12 + philosophy_pass_v8 §Dual-chain | GATED (spec/byte-close) | SPEC_v8.1 authoring + v752/v8 byte-close (shared owed-gate) → D22 | SPEC_v8.1 author + #385 brief owner |
| reactivation_campaign_397 machine-bound queue (13 pinned governed cmds) | reactivation_campaign_397 §3 (b) | GATED (machine) | machine free (owed16v2 + v752 release) → drain per §3; heavy = operator-GO → D27 | reactivation campaign / operator |
| Harness residual: serializer `--base-content-sha256` hot-file MANDATE/default promotion | harness_failure_ledger `serializer_whole_file_staging_absorbs_sibling_hunks` (class-fixed opt-in 56fc64e19; mandate owed) | GATED (session-quiesce) | next apparatus-maintenance batch AFTER live multi-agent commit session quiesces (promoting a default mid-session risks sibling commits) → D25 | serializer maintenance (#354) |
| Harness residual: false-dead liveness process-tree-walk gate | harness_failure_ledger `false_dead_diagnosis_incomplete_process_tree_walk` (worked-around 2026-07-10, no gate) | GATED (apparatus batch) | next apparatus-maintenance batch → D26 | liveness-tooling maintenance |
| Harness class: `daemon_5min_harness_long_call_sweep_kill` | harness_failure_ledger | **NOT $0-gate-payable** (environmental, not class-fixable from repo) | permanent answer = chunked RESUMABLE foreground (already the workaround); WARNING docstring landed 91b4b5db1 — no further gate owed | (environmental) |

## IN-FLIGHT — builder-owned, NOT touched (hard rule honored)

| Item | Source | Owner (do-not-touch files) |
|---|---|---|
| v752 P0-1..P0-5 (launcher-facing #383 program · epoch-backstop vs banked-fallback · R1 pose composability · chroma A/B rung · amber admission precondition) | ADVISORY_v752_fresh_eyes | launch executor — witness_autoconfig.py, launch_witness_run.py, levelset trainer, test_crucible2_v752_dsl_wirein.py |
| v752 config bare-literal NOTEs (taper knobs strength/scale/floor · OI-2 λ computing-FEED cite · `--dseg-aware-taper*` activation-ledger key resolution) → D28 | philosophy_pass_v752 §CONSTANT + P1 | launch executor (witness_dsl / config) |
| v753 P0-1..P0-3 (3-vehicle MLX/deploy/inflate reconciliation · fixed-bank byte-close vs inflate · `derive_crucible_v753_config` + A2 matched-capacity control) | ADVISORY_v753_texture_trunk_fresh_eyes | v753 builder — witness_autoconfig.py, witness_dsl |
| v8 P0-1..P0-4 (kill-inequality direction · matched-compute experiment · executable byte closure · edge-vs-class architecture) + decoupled-field trainer BUILD + `--head` Lever + owed-9 1a-BLOCKING carrier → D24 | ADVISORY_v8_fresh_eyes + v8_unlock_398a + DUAL_CHAIN_BRIEF | v8 decoupled-field build — trainer, witness_dsl |
| v8 bare-literal code NOTEs (param_tolerance=0.05 · _SEG_H/_SEG_W · dilate=2) → D23 | philosophy_pass_v8 §6 (+ #393-D4 note) | v8 build / inc1a sibling — decoupling_screen.py, road_undriv_bulk_field.py, movable_deshare.py |
| D10 marimo / molab contest #347 | deferral_ledger D10 (RESOLVED-VERIFIED) | sibling Marimo viz agent — figures, witness-machine repo, dashboard |

## Could-not-classify honestly

None. Every enumerated owed item resolved to exactly one bin.

## NEW-DEBT DISCIPLINE (standing rule)

**Every new owed item is, at creation time, either (a) PAID in the same turn, or (b) LEDGERED in
`.omx/state/deferral_ledger.md` with a NAMED trigger + a NAMED owner.** There is exactly ONE home for
open debt: the deferral ledger. **No owed item may live only in a memo body.** A research memo may
*describe* an owed item, but the moment it is owed it must also be a ledger row — the memo is the
rationale, the ledger is the debt of record (single source of truth, surfaced by `costate_digest` at
session start).

Enforcement of the rule:
- A "named trigger" is a concrete condition that WILL fire (`fires when the chosen-chain FINAL ckpt
  lands`, `~15 GiB baseline freed`, `next apparatus-maintenance batch`), never "later" / "eventually".
- A "named owner" is a person/agent/task-# (`launch executor`, `v8 build`, `serializer maintenance`,
  `operator`). "TBD" owner = the row is not closed.
- GATED rows carry the machine/spec/operator blocker verbatim; IN-FLIGHT rows carry the owning agent +
  the do-not-touch files.
- Any negative that becomes an owed reformulation keeps its `verdict_scope: <level>` line.
- Sister of CLAUDE.md ANTI-SIGNAL-LOSS (DEFER is forbidden for READY ∧ high-EV; this rule is the
  book-keeping half — a READY node with a $0 first step is PAID, not ledgered) + the default-off
  activation-ledger discipline (an owed item held-but-never-fired is still orphaned until the ledger
  tracks it) + "Results must become system intelligence" (a chat/memo-only owed item is a lost item).
