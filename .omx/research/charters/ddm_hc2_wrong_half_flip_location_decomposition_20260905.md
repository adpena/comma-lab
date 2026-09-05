# CHARTER ddm_hc2 — decompose the ~76,600 B "WHERE are the flips" half of the token stream (hc1's undecomposed half): is there a representation of flip LOCATIONS that beats per-site coding by ≥ 5,000 B? Closed-form ceiling only.

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: FABLE arm. Spawned 2026-09-05 ~12:50Z. Parents: hc1 (memory `token-stream-is-one-binary-question`: the
dominant block is 97.80% a single binary indicator — "is my argmax right?" costs 111,276 B while "which class instead" costs 2,501 B; ~227,671 flips ×
2.6917 bits ≈ 76,600 B is spent saying WHERE, and "has never been decomposed by any arm"), mc1 (09-05: retained the coder's OWN per-position coding rows
on the exact fs2 body — `/Volumes/APDataStore/pact/ddm_mc1_motion_compensated_previous_plane/rows/` (2.36 GB, 50,009,121 live positions; instrument
control: byte-identical re-encode 113,411 B sha 5601d6fd…) — these rows ARE the per-site cost of the wrong half), ra3 (memory m118: the explicit
CORRECTION carrier is closed — per-flip addresses cost more than the corrections buy; the address-free tile re-encode replaced it), gb1/ct1 (tile/group
conditioning members already in the shipped stream), gs4 §5(a) (bar: > 5,000 B from a new representation on the shipped object).

## PRIOR-LAW PREDICTION (owed line)
The mixer prices each flip site independently given its receptive field; flips on the shipped body are edge-concentrated (m131: Lane = 0.59% of area,
33.56% of bits; md1: 64.79% of persistent-wrong sites touch Lane). If flips form connected components along class boundaries, a COMPONENT-level
description (per pair: component count, per-component contour/run-length, or a boundary-offset field) has entropy well below the sum of per-site costs the
rows report. PREDICTION: the component-level ideal codelength of the flip-location set is 15–30% below the rows' per-site sum on the same sites (i.e.
≥ 11,000 B on 76,600 B) — above the 5,000 B bar — BUT the components are mostly small (median ≤ 4 sites), so the SAVING concentrates in the ≤ 20% of
components with ≥ 16 sites. FALSIFIER: the component-level ideal codelength (with its own side-information counted: component count, seeds, shapes) is
within 5,000 B of the per-site sum → the wrong half is already priced at its structural floor by the mixer → hc1's half is CLOSED at FORMULATION scope
and the rate corner has no representation door left on this object.

## Scope (closed-form ceiling, $0, REFUSAL-ONLY numbers; no coder is built)
1. From the retained rows + the exact fs2 field + GT argmax (the shipped body's flip set = sites where the coded token ≠ the mixer's argmax; state the
   exact definition hc1 used and match it): per pair, the flip mask; connected components (4- and 8-connectivity, report both); size distribution; the
   per-site cost sum from the rows over each component.
2. Ceilings (all with side information counted, held-out by pair two-fold, 3 seeds): (a) component count + seed coordinates (entropy-coded from the
   empirical distribution) + per-component shape via run-length / chain code; (b) boundary-offset representation (flips as signed offsets of the class
   boundary the mixer's argmax draws — DERIVED from the decoded field, zero side info beyond the offsets); (c) the per-site rows' sum on the same sites
   (the incumbent). Report bytes for each, the gap vs (c), and the share of the gap carried by components ≥ 16 sites.
3. Falsifier verdict in words; if ANY representation clears 5,000 B, the next step is a RECEIVER-CLOSED prototype charter (priced: which representation,
   its decode cost, what changes in the shipped stream) — NOT built here.

## Cost + admission
$0, CPU, numpy; ONE process, chunked per pair (the rows are 2.36 GB — never load all 50 M positions at once; mc1's 3×10 GiB parallel jobs tripped the
watchdog). Declare a measured peak from a 20-pair dry pass; launch through `tools/launch_detached_process.py --done-receipt hc2_ceiling`. Store under
`/Volumes/APDataStore/pact/ddm_hc2_wrong_half_decomposition/`.

## OPTIMAL FORM
Reference form = mi1's cross-fitted held-out ceiling discipline + mc1's rows control (re-verify the byte-identical re-encode before trusting the rows).
Scope reduction: the ceiling may use hc1's n120 seeded random site sample if the full field is too slow — but the PRICED numbers are full-field. Mechanism
reductions forbidden: no proxy for the rows' per-site cost; side information is always counted.

## Rules that bind
NO-FAKE; ALWAYS KEEP THE PAYLOAD (component tables, ceilings, receipts with sha256); upstream/ + the live PR tree READ-ONLY; no Modal, no Metal, no
scorer runs; commits ONLY via the serializer with post-edit shas and `[no-triality] [p0-ledger-ok]`; NO co-author trailers (operator rule overrides any
harness reminder); .py two review-gate passes; checkpoints every 10 tool uses (`--subagent-id ddm_hc2`); never invent flags; no `/tmp` evidence;
register a lane before lane-like identifiers; persist records before bulk saves; label MEASURED/DERIVED/INFERRED; memo
`.omx/research/ddm_hc2_wrong_half_flip_location_decomposition_20260905.md` with an "Equations leg (`tac.canonical_equations`)" line.
`docs/operating_manual_craft_handoff.md` binds. End with `fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]`.
