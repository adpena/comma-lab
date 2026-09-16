# ddm_jrx2 — the joint (field + renderer) exchange probe with the REPAIRED row-1 instrument (charter, MAIN 2026-09-16; codex astra xhigh; --owns-scorer)

jrx1 (landed 3c5669723; receipt `.omx/research/ddm_jrx1_20260916/RECEIPT.md`) finished the 24 reference searches (215
single-token + 121 two-token resolved rows, retained under `/Volumes/APDataStore/pact/ddm_jrx1/`, read-only) and stopped
INSTRUMENT_INCOMPLETE because MAIN's Definition 2 compared a random-pair median to pd4's SELECTED pooled 12.0 bits/token —
a different estimand — and pairs 502 / 547 had no admissible single anchor (62 moves fail the two-error screen). This arm
IS jrd1's charter (1c1aba7ce) + jrx1's Definition 1 (zero-net bytes = real re-encode within ±35 B, residual charged at
6.658589531221714e-7 S/B) with the row-1 instrument REPLACED by jrx1's proposal (`.omx/research/ddm_jrx1_20260916/CONTROL_REPAIR_PROPOSAL.md`, sha 0771c3870b348029), adopted verbatim:

## Row 1 (instrument) — MAIN's decisions
1. **Reproduction on identical retained pd4 proposals** for the pairs common to pd4's pool and the K24 draw: same
   resolved-pose + seg numerator, same coded-price denominator, per pair. Tolerance: each overlapping pair's price within
   ±10 % and its credit within ±15 % of pd4's retained row. Report k of n overlapping pairs inside tolerance; the
   instrument passes at k/n ≥ 0.8 with the failures listed. The K24 discovery economics (random-pair medians) are reported
   SEPARATELY as a distribution, never compared to pd4's pooled figure.
2. **Null-pair policy**: pairs 502 / 547 stay in the 24 denominator as null observations (no changed token → no price);
   every median quotes k of 24. No substitution of pairs; no no-op called a changed token.
3. **Bounded extension** (jrx1 decision 3): for pair 547 use its least-damaging rejected single anchor (cell [343,231],
   4→3, +5 cells) and test its radius-1 second-token neighbourhood under the unchanged ≤2-cell gate on the combined
   render; same procedure for 502 from its least-damaging anchor. Report whether either becomes admissible.

## Rows 2 and 3 — only after row 1 passes
Row 2 (renderer-only at held field; ALL 600 pairs' collateral priced) and Row 3 (joint zero-net direction; real tail +
int4 bytes by twins; resolved pose; residual bytes charged; undefined ratios reported as undefined when the better single
row is null) exactly as jrd1's charter §The probe. Gate: joint realized exchange ≥ 2× the better single row on the median
of the pairs where all three rows are finite (quote k of 24). Falsifiers as jrd1's. A fired falsifier closes the joint
direction on this object; a pass charters the burn (jrd2).

## Apparatus, boundaries, retention
As jrd1/jrx1: no burn, Modal, fire, packet, authorize_*; never edit upstream/, the PR tree, sealed trees, contract or
receiver code, the shipped renderer/basis/prior in the tree; pd1–pd6, jrd1, jrx1 stores read-only; write under
`/Volumes/APDataStore/pact/ddm_jrx2/`; report free space before every heavy step (APDataStore ~18 GiB); retain ≤ 3 GiB
with sha256; Vertigo untouched. Retained-bytes accounting must skip `.pending` temp files and ExFAT `._` stubs and
tolerate a sibling's rename (jrx1's shard race, cured in its v2). Heavy steps via `tools/launch_detached_process.py
--done-receipt`; background receipt waits. Serializer commits with post-edit shas; two visible review passes per .py;
`[no-triality] [p0-ledger-ok]`; NEVER a co-author trailer or AI attribution; rc 17 is NOT a stop (continue, bundle,
MAIN lands; commit LAST). Checkpoint `ddm_jrx2`; lane `ddm_jrx2_joint_exchange_probe_repaired_row1_20260916`.
Reuse jrx1's code (`experiments/ddm_jrx1_field_control.py`, `ddm_jrx1_price_control.py`) — its pricing adapter's
encoder and winner-selection stages are source-reviewed but UNEXECUTED; execute them and record the first real run.

## OPTIMAL FORM
As jrd1's charter (reference forms pd4 + rw1/ren2 + real coder; SCOPE deltas K = 24 and one discrete step; no MECHANISM
reduction). The row-1 change is an INSTRUMENT correction (estimand match), not a reduction. Provenance pins: jrx1
landing 3c5669723; jrx1 charter e9c4f75c0; jrd1 charter 1c1aba7ce; proposal sha 0771c3870b348029; pd4 memo sha 48767941b8a9cf1e; move 52
archive ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e, pointer commit d1fc2a1c2.

## Prior negatives accounted (operator 2026-08-15)
jrx1 (estimand mismatch; null pairs; shard accounting race); jrd1 (no scorer slot; int4 step −4…+29 B); rw1 / ren2 / cb1 /
pd4 / pd5 (credit does not compound — this arm does not extend runs) / m164 / m132 / m88.

Final message: as jrd1's, plus the row-1 k/n table and the null-pair dispositions, ending with the current own-vehicle
frontier line (re-derive at start).

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
