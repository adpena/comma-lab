# Codex Routing Directive — TOP-9 Arbitrariness Extinction: Early-stopping Patience Undeclared

**Subagent**: `lane_arbitrariness_extinction_meta_lens_systematic_audit_20260518`
**Value ID**: `early_stopping_patience_undeclared`
**Resolution path**: `formula`
**Predicted ΔS**: [-0.003, -0.0005]
**Cost envelope**: **$0 (NET-NEGATIVE — saves money)**
**Rank score per dollar**: 3.0

## Bug class

Most substrate trainers lack any early-stopping mechanism. Runs to full `args.epochs` regardless of convergence.

## Solution

Same as TOP-2 (`epochs_wildly_varies_*`) — this is the SISTER row. Both consume the same canonical helper `tac.early_stopping.SlopeWatcher`.

## Concrete next step

Land canonical `tac.early_stopping.SlopeWatcher` from TOP-2 directive; this row inherits.

## Net effect

NET-NEGATIVE COST (saves GPU dollars). Improves convergence (avoids overfit at frontier substrates).

## Exit criteria

Same as TOP-2 — single canonical helper closes both rows.
