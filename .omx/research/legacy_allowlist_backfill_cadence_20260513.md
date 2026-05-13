# Legacy allowlist backfill cadence ledger

**Created**: 2026-05-13 by FIX-WAVE-3 (R3-1 closure).
**Source finding**: `feedback_recursive_review_r3_LANDED_20260513.md` — Catalog #176's `_CHECK_176_LEGACY_ALLOWLIST` silently absorbs 79 strict-mode preflight gates not in the CLAUDE.md numbered catalog table; operator-facing protection surface understated by 46%.

## Contract

This is a HISTORICAL_PROVENANCE ledger (per CLAUDE.md artifact-lifecycle taxonomy, Catalog #113). Each backfill wave appends one row below; previous rows are immutable. The ledger MUST contain a session entry within the last 30 days OR the META gate `check_legacy_allowlist_backfill_cadence_ledger_current` (Catalog #183) refuses with a freshness violation. The 30-day window is the operator's enforced "shrink toward zero" cadence per Boyd's adversarial-review verdict.

## Target invariant

|`_CHECK_176_LEGACY_ALLOWLIST`| → 0 (every entry must eventually have a numbered row in the CLAUDE.md "Meta-bug class catalog" table).

## Cadence rule (NON-NEGOTIABLE)

Each FIX-WAVE that ships ANY new Catalog # MUST also backfill ≥ 1 entry from `_CHECK_176_LEGACY_ALLOWLIST` to the CLAUDE.md numbered table (with a corresponding allowlist removal). This is the structural "slack must shrink" rule per Boyd's verdict. The R3-2 sub-finding (3 entries with test files + STRICT wire-in but no CLAUDE.md row) is the prioritized highest-leverage subset per Karpathy/Hassabis/Tao.

A FIX-WAVE that ships zero new Catalog #s is exempt from the backfill cadence rule (e.g. pure documentation / pure test additions) — but the 30-day overall ledger-freshness rule still applies. Operators may also land "backfill-only" waves that ship no new Catalog #s but reduce |allowlist| by N ≥ 1.

## Backfill wave log

| Wave | Date | Entries backfilled | Allowlist size after | Notes |
|------|------|-------------------|---------------------|-------|
| R3-2 | 2026-05-13 | 3 (#162 `check_operator_authorize_canonical_use`, #165 `check_modal_mount_builder_uses_mtime_stability_check`, #167 `check_substrate_dispatch_uses_smoke_before_full_pattern`) | 76 | First backfill wave; prioritized highest-leverage subset (recent catalog numbers with full test coverage where operator-visible gaps were most jarring) per Karpathy/Hassabis/Tao verdict. |

## Reference: remaining 76 entries (snapshot at R3-2 close)

The remaining 76 entries are defined by `_CHECK_176_LEGACY_ALLOWLIST` in `src/tac/preflight.py`. They are NOT enumerated here because the source of truth is the frozenset constant; enumerating here would duplicate state and invite drift. To inspect the current set:

```bash
.venv/bin/python -c "from tac.preflight import _CHECK_176_LEGACY_ALLOWLIST; [print(n) for n in sorted(_CHECK_176_LEGACY_ALLOWLIST)]"
```

## Reactivation / exit criteria

This ledger CLOSES (and Catalog #183 may be retired) when `|_CHECK_176_LEGACY_ALLOWLIST| == 0`. Until then the ledger is active and the 30-day freshness rule is binding.

## Cross-references

- CLAUDE.md "Meta-bug class catalog" — the canonical numbered table this ledger tracks convergence against.
- Catalog #176 (`check_strict_preflight_callsites_have_claude_md_catalog_row`) — the META gate whose `_CHECK_176_LEGACY_ALLOWLIST` this ledger tracks.
- Catalog #183 (`check_legacy_allowlist_backfill_cadence_ledger_current`) — the freshness self-protection sister of this ledger.
- `feedback_recursive_review_r3_LANDED_20260513.md` — the adversarial-review finding that originated this ledger.
- `feedback_fix_wave_3_r3_findings_LANDED_20260513.md` — the landing memo for the R3-2 first backfill wave.
- CLAUDE.md "Strict-flip atomicity rule" — sister rule that prevents warn-only-purgatory; this ledger prevents legacy-allowlist-purgatory.
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable — the structural reason this ledger has a self-protection gate.
