# PRE-REGISTERED harvest adjudication for ddm_rxc1 (written BEFORE SCREEN.json exists — no-toy law applied at MY charter's own defect)

Date: 2026-09-01 · Author: MAIN · Status: BINDING at rxc1 harvest · Trigger: operator "No naive
or toy" (2026-09-01) → charter-time audit of the live SCMDL chain found ONE vacuity hazard in
MAIN's own rxc1 charter, filed here before the verdict can be born.

## The hazard (m50 vacuity==PASS, NO-FAKE-adjacent)

The rxc1 charter's Stage-2 gate ("incremental delta vs FULL exact re-encode, Pearson/Spearman
≥0.9") is only meaningful if the incremental leg is APPROXIMATE. rxc1's live design (BASELINE +
NULL_REPLAY receipts, 2026-09-01 00:18–00:38) restarts from checkpointed exact state and
re-encodes the SUFFIX — which is EXACT BY CONSTRUCTION. Exact vs exact ⇒ correlation ≡ 1.0 ⇒ the
gate passes VACUOUSLY. A "GATE-1-PASSED" claimed on that correlation alone would be a toy verdict
on the campaign's critical path.

## The binding adjudication rule (pre-registered)

1. **If SCREEN.json's incremental leg is an exact suffix re-encode** (deltas byte-identical to
   full): the correlation row is recorded as VACUOUS-BY-CONSTRUCTION and carries NO gate
   authority. Gate-1 is then adjudicated on the COST criterion alone: the measured per-proposal
   wall cost at the best stride (null-replay receipts: 716.4 s full · 475.1 s @start 200 ·
   356.0 s @start 300 · 237.7 s @start 400 — cost ∝ suffix length) vs what an SCMDL outer loop
   can afford. Expected verdict shape: GATE-1-PARTIAL — exactness proven, economics NOT solved
   (average suffix cost ≈ half of 897.675 s is no outer-loop enabler).
2. **The family's OPTIMAL FORM is named now**: restart + **state-reconvergence splice** — after
   re-encoding the edited region forward, detect when the adaptive HPAC/corrector context state
   RECONVERGES to the original trajectory; from that point the original stream suffix is reused
   byte-identically (splice), giving EXACT deltas at LOCAL cost. If the context never reconverges
   (permanently shifted count tables), that non-reconvergence is itself the decisive measured
   fact: exact incremental coding is then structurally suffix-priced on this coder, and SCMDL's
   loop granularity must be batch-of-proposals-per-suffix-re-encode, not per-proposal. EITHER
   measured outcome routes the successor; neither may be skipped by citing a vacuous 1.0.
3. **If the incremental leg is halo-approximate** (deltas differ from full): the chartered ≥0.9
   Pearson/Spearman gate applies exactly as written, plus max-abs-error and sign-agreement.
4. **No GATE-1-PASSED headline may be written at harvest without quoting this rule** and stating
   which branch (1/2/3) the receipts land in.

## Scope

verdict_scope: INSTANCE (rxc1 on the AFR1 stream). sfp1 audited in the same pass: its charter
already carries the anti-toy structure (real field-edit programs over the real dense field,
positive control with a KNOWN realized outcome, PROPOSAL-SPACE-EMPTY honest exit, token-GT
excluded by construction) — no amendment needed.
