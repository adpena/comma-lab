# Deferred-Blocked Inventory — Task Triage — 2026-05-20T12:06:07Z

> Companion to `.omx/research/task_triage_inventory_20260520T120607Z.md`.
> Lists 2 DEFER-BLOCKED tasks with explicit blocker + canonical-state
> pointer + reactivation criteria + watch trigger. Per CLAUDE.md "Forbidden
> premature KILL without research exhaustion" non-negotiable: these are
> dormant-with-reactivation, NOT killed.

## Routing rationale (per task brief Rule 6)

Both DEFER-BLOCKED tasks below qualify because:
- Explicit blocker prevents progress
- Reactivation criteria pinned in canonical state
- Per CLAUDE.md "Forbidden premature KILL": preserve as dormant-with-reactivation

## Decision summary table

| # | Task | Blocker | Canonical state pointer | Reactivation criterion | Watch trigger |
|---|------|---------|---|---|---|
| 1 | `paid_dispatch_batch::ITEM_4` Catalog #204 A1 passthrough recovery | Catalog #313 DEFER predecessor `harvest_e8_sgld_1_instant_crash_20260519` | `.omx/state/probe_outcomes.jsonl` rows for probe_id | (a) Address E.8 SGLD #1 root cause (`tools/operator_authorize.py --capture-output` reproducing) OR (b) supersede the DEFER row via fresh anchor OR (c) operator-frontier-override per Catalog #300 | DEFER expires 2026-06-02T06:10:00Z (14d staleness window per Catalog #313); operator may re-evaluate at expiry |
| 2 | `overconservative_authority_bottlenecks::DETERMINISTIC_PACKET_RUNTIME_AUTHORITY` | Partner-active high-churn surface on `tools/build_deterministic_packet.py` + sidecar; backed off after local review + sidecar audit | Lane registry + commit-churn audit on `tools/build_deterministic_packet.py` | (a) Partner sister-subagent completes their deterministic_compiler.py work OR (b) Codex re-routes via separate non-partner-active sister (e.g. procedural candidate authority surface) | Watch git-log for `tools/build_deterministic_packet.py` last-commit timestamp; reactivate when 7 days since last commit (surface stabilized) |

---

## Defer 1 — Catalog #204 A1 passthrough recovery

**Explicit blocker**: `tools/operator_authorize.py --yes` path refused before provider spawn/claim/spend on Catalog #313 DEFER predecessor `harvest_e8_sgld_1_instant_crash_20260519`. The refusal is STRUCTURAL per Catalog #313 (`check_dispatch_target_has_no_predecessor_adjudicated_outcome`) — the DEFER blocker is in the canonical probe-outcomes ledger.

**Canonical state pointer**:
```bash
.venv/bin/python -c "
from tac.probe_outcomes_ledger import latest_blocking_outcome_by_substrate
v = latest_blocking_outcome_by_substrate('stack_of_stacks')
print(f'verdict={v.verdict}, expires={v.expires_at_utc}, notes={v.notes[:120]}')
"
```
Expected output: `verdict=DEFER, expires=2026-06-02T06:10:00Z, notes=E.8 SGLD #1 fc-01KRZCHVY6C1TSFNNS6KN13G70: trainer crashed at 2.1s...`

**Sister advisory row**: same probe_outcomes.jsonl has a sister advisory-grade row from codex session `019de465` documenting the refusal observation (NOT adding new blocker; preserving no-spend ITEM_4 refusal evidence).

**Reactivation criteria** (per CLAUDE.md "Forbidden premature KILL" non-negotiable):
1. **(preferred)** Re-dispatch SGLD #1 with `--capture-output` to surface root cause for the 2.1s instant crash → adjudicate as PROCEED OR re-DEFER with specific failure-mode classification per Catalog #307
2. **(alternative)** Operator-frontier-override per Catalog #300 §"Mission alignment" Consequence 1 → requires verbatim operator quote + memo at `.omx/research/operator_authorizations/`
3. **(natural expiry)** DEFER expires 2026-06-02T06:10:00Z; operator may re-evaluate at that point

**Watch trigger**: monitor `.omx/state/probe_outcomes.jsonl` for any new event with `probe_id=harvest_e8_sgld_1_instant_crash_20260519` (re-adjudication) OR for the `expires_at_utc` to pass.

**Per CLAUDE.md "Apples-to-apples evidence discipline"**: the SGLD trainer paradigm IS intact (sister E.8 #2 reached auth_eval proving the trainer code path is sound). This is an IMPLEMENTATION-LEVEL FALSIFICATION per Catalog #307 — the opaque-dispatch implementation falsified; the SGLD paradigm preserved.

---

## Defer 2 — Deterministic packet compiler runtime authority hardening

**Explicit blocker**: `tools/build_deterministic_packet.py` + sidecar surfaces are partner-active/high-churn per local review + sidecar audit. The Codex session pivoted to procedural candidate authority rather than overlap with a partner-active surface (per CLAUDE.md "Subagent coherence-by-default" anti-duplication primitive + Catalog #230 sister-subagent ownership map + Catalog #340 sister-checkpoint guard).

**Canonical state pointer**:
```bash
git log --oneline --all tools/build_deterministic_packet.py | head -20
```
Watch this list for activity; if the file goes 7+ days without a commit, the surface has stabilized.

**Sister advisory cross-reference**: `.omx/research/codex_findings_overconservative_authority_bottlenecks_20260519T014528Z_codex.md` documents the original codex finding + the pivot to procedural candidate authority.

**Reactivation criteria**:
1. **(preferred)** Partner subagent completes their work on `tools/build_deterministic_packet.py` and lands a stable canonical contract → THIS task may then layer on without collision
2. **(alternative)** Codex re-routes the deterministic-packet-runtime-authority work to a separate non-collision surface (e.g. procedural candidate authority sister surface already exists per the pivot)
3. **(operator decision)** Operator declares the partner-active surface boundary explicitly (e.g. via canonical helper namespace partition per CLAUDE.md `tac` package hygiene rule)

**Watch trigger**: `git log --since="7 days ago" tools/build_deterministic_packet.py` returns empty → surface stabilized → reactivate.

**Per CLAUDE.md "Subagent coherence-by-default"**: the structural protection is the sister-subagent ownership map + checkpoint discipline + pre-commit guard. The deferral honors that discipline.

---

## Discipline attestation

- Catalog #229 PV: read blocker text + probe-outcomes events + sister codex findings
- Catalog #287: no phantom-API citations (every Catalog # cited exists; every probe_id verified)
- Catalog #292: per-deliberation assumption surfacing applied (this triage assumes the DEFER blockers are HARD-EARNED per Catalog #313 + #340 sister discipline; both verified via ledger reads)
- Catalog #307: paradigm-vs-implementation classification applied (Defer 1: paradigm intact + implementation falsified)
- Per CLAUDE.md "Forbidden premature KILL without research exhaustion": both tasks remain in dormant-with-reactivation state per the non-negotiable
- Per CLAUDE.md "Subagent coherence-by-default": Defer 2 honors the sister-subagent ownership map anti-duplication primitive

## Cross-references

- Catalog #313 (`check_dispatch_target_has_no_predecessor_adjudicated_outcome`)
- Catalog #245 modal_call_id_ledger
- Catalog #307 paradigm-vs-implementation falsification
- Catalog #340 sister-checkpoint guard
- CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable

## Watch cadence recommendation

- **Weekly** (every Sunday): check `tools/operator_authorize.py --dry-run --recipe substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch` for current blocker state
- **At 2026-06-02 DEFER expiry**: re-evaluate Defer 1 reactivation paths
- **Daily via git log**: check Defer 2 partner-activity stability


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:deferred-blocked-inventory-audit-memo-trigger-tokens-describe-inventory-items-not-new-empirical-finding -->
