# Legacy allowlist backfill cadence ledger — 2026-08-25 wave file

**Created**: 2026-08-25 by the #842 preflight full-enumeration loop (Catalog #183 fired at the
prescan-r5 frontier: newest ledger `legacy_allowfill…20260513.md` was 105 days old vs the 30-day
window). **Predecessor**: `.omx/research/legacy_allowlist_backfill_cadence_20260513.md` —
HISTORICAL_PROVENANCE, immutable, carries the founding contract + the R3-2 wave row. Per the
Catalog #183 gate's own cure menu ("add a new ledger file with today's date"), this NEW dated file
continues the ledger; the freshness gate keys on the newest filename date.

## Contract (inherited unchanged from the 2026-05-13 founding ledger)

HISTORICAL_PROVENANCE ledger (Catalog #113): each backfill wave appends one row to the wave log;
previous rows are immutable. The newest dated ledger file MUST be ≤30 days old (by FILENAME date)
and carry the "Backfill wave log" heading, or Catalog #183
(`check_legacy_allowlist_backfill_cadence_ledger_current`) refuses. Target invariant:
|`_CHECK_176_LEGACY_ALLOWLIST`| → 0. Each FIX-WAVE shipping a new Catalog # MUST backfill ≥1
allowlist entry; backfill-only waves (no new gate) are allowed and reduce |allowlist| by N ≥ 1.

## Backfill wave log

| Wave | Date | Entries backfilled | Allowlist size after | Notes |
|------|------|-------------------|---------------------|-------|
| R3-2 | 2026-05-13 | 3 (#162 `check_operator_authorize_canonical_use`, #165 `check_modal_mount_builder_uses_mtime_stability_check`, #167 `check_substrate_dispatch_uses_smoke_before_full_pattern`) | 76 | Carried forward verbatim from the founding ledger (immutable there; repeated here so this file is self-contained). |
| W2 | 2026-08-25 | 3 (#409 `check_dispatch_cli_shell_hazards`, #410 `check_feature_flags_have_live_objective_effect`, #411 `check_evidence_row_has_falsification_scope_when_negative`) | 73 | #842-loop wave. Selection = the R3-2 leverage rule (test coverage + strict wiring + operator-visible gap): #409 is named in CLAUDE.md "Operator gates must be wired and used" and wired strict in `preflight_all()`; #410 guards the dead-objective-flag genus (config-orphan sister); #411 enforces `falsification_scope` on negative evidence per forbidden-premature-KILL. Rows landed in `docs/meta_bug_class_catalog.md` (the canonical pointer-backed catalog surface), honestly labeled BACKFILLED (authored from the gates' own docstrings + code, not fresh landings). Catalog numbers claimed via `tools/claim_catalog_number.py` (serialized; counter now 412). Measured context: 0 of the 76 allowlisted names had catalog rows anywhere in the corpus at wave start — the backfill debt is real era-debt, not extraction loss. |

## Reference: remaining 73 entries

Source of truth remains the frozenset constant — deliberately NOT enumerated here:

```bash
.venv/bin/python -c "from tac.preflight import _CHECK_176_LEGACY_ALLOWLIST; [print(n) for n in sorted(_CHECK_176_LEGACY_ALLOWLIST)]"
```

## Exit criteria

Unchanged: this ledger family CLOSES (and Catalog #183 may retire) when
`|_CHECK_176_LEGACY_ALLOWLIST| == 0`.

## Cross-references

- `.omx/research/legacy_allowlist_backfill_cadence_20260513.md` — founding ledger (contract + R3-2).
- Catalog #176 (`check_strict_preflight_callsites_have_claude_md_catalog_row`) — the META gate whose allowlist this tracks.
- Catalog #183 (`check_legacy_allowlist_backfill_cadence_ledger_current`) — the freshness gate that demanded this file.
- `docs/meta_bug_class_catalog.md` rows 409–411 — the W2 backfill rows.
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" — the structural parent discipline.
