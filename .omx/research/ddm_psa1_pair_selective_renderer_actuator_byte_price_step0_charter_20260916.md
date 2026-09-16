# ddm_psa1 — a PAIR-SELECTIVE renderer actuator as a new object part: STEP 0, its real byte price under the shipped weight coder (charter, MAIN 2026-09-16; codex astra xhigh; SCORER-FREE — pd6 (Opus) owns the scorer slot; do not run the scorer)

## Why (gs3 Addendum 67; rw1 OWED1f; jrx2 live hypothesis)
jrx2 measured at n600 that the smallest GLOBAL int4 weight step credits 0 / 0 / −3 target Seg cells against 1,740–2,438
added errors on the other 599 pairs: a global weight is not a per-pair actuator, and the shipped object has NO per-pair
renderer conditioning except the token field itself. The joint field+renderer family therefore stays open only through
a new object part: a pair-selective delta on the renderer applied to ONE pair's render and to nothing else, with its
bytes COUNTED in the archive. Whether that part can ever pay is a price question first: how many real coded bytes does
the cheapest pair-selective actuator cost per pair, under the shipped SM1S int4 weight coder or a new counted section?
If the cheapest per-pair actuator costs more than the pair's whole seg debt is worth (the median pair's seg debt at move
52 is ~0.6 bar ≈ 15 B at the exchange 6.658589531221714e-7 S/B — the arm re-derives this from the per-pair residual
receipts in pd4/pd5's retained rows), the part is closed before any scorer runs. That is the only question here.

## The probe (SCORER-FREE; byte legs by the REAL coder; no render, no seg, no pose)
Base = move 52 (tree `/Volumes/APDataStore/pact/ddm_pd3/candidate/candidate_runtime`, read-only, copy; archive sha
ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e; 179,332 B; tail 119,097 B). Reuse jrd1's byte
machinery (`experiments/ddm_jrd1_byte_preflight.py`, landed c9bf03e62) and jrx2's renderer byte legs
(`experiments/ddm_jrx2_renderer.py`, 8a689a70d) — do not rewrite. Price, per pair, on the 24 seeded pairs jrd1 drew
(seed 20260916; the same K24) plus the 12 heaviest-residual pairs from pd4's pool, four actuator forms:
A. a rank-1 delta on ONE layer's weight (the layer jrx2's byte legs found cheapest to perturb), stored as int4 vectors
   with the layer's fp16 scale, in a NEW counted section keyed by pair id;
B. a per-pair additive bias on the head layer (5 channels) at int4;
C. a per-pair FiLM-style scale+shift on the last block's output (2 × channels) at int4;
D. a per-pair 1-cell-radius spatial mask on the head bias (the boundary-jitter locus, memory `residual_seg_debt…86 %`)
   encoded as run-length + int4 amplitude.
For each form and pair: the exact section bytes under (i) a standalone brotli-q11 section, (ii) appended to the SM1S
stream with the shipped coder (twins), (iii) the shipped tail's context model where applicable. Report per-form median
and IQR bytes per pair and the fixed section overhead (header, index). Then the DERIVED break-even: bytes per pair ×
exchange vs the pair's own measured seg debt (pd4/pd5 retained rows) — a form is viable only where the price is below
the pair's whole debt for ≥ 10 % of pairs; state k of 36.

## Pre-registered gate and falsifier
The part survives step 0 only if at least one form prices below the pair's total seg debt on ≥ 10 % of the 36 pairs
(k/36 quoted) AND the fixed section overhead is < 200 B. Otherwise: closed at formulation scope (pair-selective
renderer actuator, these four forms, on this object); say so with verdict_scope. If it survives, MAIN charters the
scorer step (psa2: realize the surviving form on those pairs, resolve pose, seg, all-600 collateral) with --owns-scorer.

## Boundaries, retention, process
No scorer, render, pose solve, training, Modal, fire, packet, authorize_*; never edit upstream/, the PR tree, sealed
trees, contract or receiver code, the shipped renderer/basis/prior IN THE TREE (all under
`/Volumes/APDataStore/pact/ddm_psa1/`); pd*/jrd1/jrx* stores read-only; report free space before every heavy step
(APDataStore ~17 GiB — pd6 is writing too); retain ≤ 1 GiB with sha256; retained-bytes accounting skips `.pending`
and `._` and tolerates sibling renames. Heavy steps via `tools/launch_detached_process.py --done-receipt`; background
receipt waits. Serializer commits, post-edit shas, two visible review passes per .py, `[no-triality] [p0-ledger-ok]`,
never a co-author trailer or AI attribution; rc 17 is NOT a stop — continue, bundle, MAIN lands, commit LAST.
Checkpoint `ddm_psa1`; lane `ddm_psa1_pair_selective_renderer_actuator_byte_price_20260916`.

## OPTIMAL FORM
Reference forms: the shipped SM1S int4 coder + fp16 scales (the real price), jrd1/jrx2 byte legs, brotli-q11 for the
standalone control. Declared deltas: 36 pairs (SCOPE); four actuator forms (SCOPE — the family is wider; a closure is
scoped to these four); no scorer (STAGE split, not a mechanism reduction — the price is the question). Provenance
pins: jrx2 landing 8a689a70d; jrd1 landing c9bf03e62; rw1 memo `.omx/research/ddm_rw1_boundary_local_renderer_weight_foldback_20260909.md`
sha 233ed1b53a35585b; pd5 memo sha 8b596398ad31ed69; move 52 archive ae59c510…, pointer commit d1fc2a1c2.

## Prior negatives accounted (operator 2026-08-15)
rw1 (global int4 breaks 240 cells; jrx2 reproduced at n600 — hence PAIR-selective); ren2 (refit in place loses — this is
an added part, not a refit); cb1 (bytes buy a position, not a spectrum — form A is a position); jrd1 (int4 step −4…+29 B);
m111 (a byte is not fungible 26× — price per pair, not pooled); m132 (collateral — by construction zero for a
pair-keyed part; the scorer step verifies); the 34.8 B container lottery (twins).

Final message: the per-form byte table (median/IQR per pair, fixed overhead), the break-even k/36 per form, the gate
verdict in one sentence with verdict_scope, paths + shas, serializer rc, every boundary, ending with
`composition S 0.13620226906030858 @ 179,332 B [contest-CUDA T4 n600] (move 52)`.

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
