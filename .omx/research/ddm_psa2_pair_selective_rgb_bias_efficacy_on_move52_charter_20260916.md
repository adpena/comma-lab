# ddm_psa2 — efficacy of the pair-selective RGB head bias (psa1 form B3) on move 52: resolved pose, seg repair, all-600 collateral, real bytes (charter, MAIN 2026-09-16; codex astra xhigh; --owns-scorer; fires after pd6 releases the slot)

psa1 (landed; receipt `.omx/research/ddm_psa1_20260916/RECEIPT.md`; gs3 Addendum 68) priced a per-pair three-channel
RGB bias on the head output at 29 B (SM1 stream) / 54 B (final archive) / 42 B (brotli-q11 standalone section) per
pair, fixed overhead 24–47 B, below the pair's whole seg debt on 12/36 (q11) or 5/36 (SM1) pairs; population median
debt budget 22.916 B. Step 0 passed. This arm measures whether the bias BUYS anything: the credit per pair through the
real render, the resolved pose, and the frozen argmax, with real bytes, on the same 36 pairs plus the 24 next-heaviest
residual pairs from pd4/pd5's retained rows (60 pairs; SCOPE).

## The probe
1. Base = move 52 by parse-back (or the current pointer — re-derive from `reports/latest.md`); pose base under the
   measured-tolerance gate. Reuse psa1's producer and record layout (`/Volumes/APDataStore/pact/ddm_psa1/`,
   read-only) and jrx2's resolved-pose + seg legs (`experiments/ddm_jrx2_*.py`).
2. For each of the 60 pairs: solve the per-pair RGB bias (3 scalars, int4 on the head's fp16 scale) that minimizes the
   pair's seg debt on the frozen argmax through the REAL render (grid search over the int4 lattice is acceptable at
   3 dof — state the solver), with the pose RE-SOLVED after the bias (carrier re-solve with frame-0 repair) — pose is
   a leg, never held. Record the seg credit (cells), the resolved-pose delta, and the real bytes (psa1's coder, twins).
3. Collateral: the bias is pair-keyed, so cross-pair collateral is zero by construction — VERIFY it: the other 599
   pairs' renders must be bit-identical (census on all 600; quote the count).
4. Set admission on the RESOLVED pose per real bit (pd5's set pricing; iterate; report residual and iteration count).
   Pre-registered band −2e-5 … −5e-5 S over the admitted set; falsifiers: fewer than 6 of 60 pairs net positive after
   bytes; admitted set net > −2e-5; set re-price vs ledger > 10 %. A fired falsifier closes the RGB-bias actuator on
   this object (verdict_scope: formulation — per-pair 3-dof RGB head bias); the wider pair-selective family stays open
   only with a named cheaper/richer form and its own step-0 price.
5. If it nets: this is a NEW OBJECT PART. The public receiver must apply the bias (generic code in inflate.py, free
   under rule 118; the per-pair values are the counted section). Implement it on a COPY of the runtime tree; cold n600
   public parse-back; raw identity on the 599 unedited pairs; the receiver change needs a measured decode wall-clock leg
   (contract law: no inherited leg for a receiver change — the T4 measurement is MAIN's; you produce the intent and the
   timing-risk receipt per `tools/make_candidate_seal.py --first-fire-intent … --timing-risk-evidence …`, pr12's
   contract). Twins, manifest (Catalog #420), census, smokes, retention with shas. MAIN fires.
6. Memo `.omx/research/ddm_psa2_pair_selective_rgb_bias_efficacy_on_move52_20260916.md`; serializer commits; lane
   `ddm_psa2_pair_selective_rgb_bias_efficacy_20260916`; checkpoint `ddm_psa2`.

## Boundaries, retention, process
No Modal, fire, authorize_*; never edit upstream/, the PR tree, sealed trees, contract code, or the shipped receiver
IN THE TREE (the receiver change lives on the copy under `/Volumes/APDataStore/pact/ddm_psa2/`); pd*/jrd1/jrx*/psa1
stores read-only; report free space before every heavy step (APDataStore ~17 GiB); retain ≤ 3 GiB with sha256;
Vertigo untouched; retained-bytes accounting skips `.pending` and `._` and tolerates sibling renames. Heavy steps via
`tools/launch_detached_process.py --done-receipt`; background receipt waits. Serializer commits, post-edit shas, two
visible review passes per .py, `[no-triality] [p0-ledger-ok]`, never a co-author trailer or AI attribution; rc 17 is
NOT a stop — continue, bundle, MAIN lands, commit LAST. Label MEASURED / DERIVED / INFERRED / ASSUMED.

## OPTIMAL FORM
Reference forms: psa1's coder and record layout (bytes), jrx2's resolved-pose + seg legs (credit), pd5's set pricing
(admission), the real render. Declared deltas: 60 pairs (SCOPE); int4 3-dof grid search (the exact solver at this dof —
not a reduction). Provenance pins: psa1 receipt sha 941fdd1bbe4476d4; jrx2 landing 8a689a70d; pd5 memo sha 8b596398ad31ed69; move 52
archive ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e, pointer commit d1fc2a1c2.

## Prior negatives accounted (operator 2026-08-15)
jrx2 (global weight actions: collateral — hence pair-keyed, verified by census); rw1; ren2; pd5 (credit does not
compound — measure credit per pair, never assume); m132; the 34.8 B lottery (twins); the receiver-change wall-clock law
(tc4) and pr19 (no inherited leg for a receiver change).

Final message: the per-pair credit/bytes table (60 rows summarized: k of 60 net positive, median credit, median bytes),
the collateral census (599/599 bit-identical or the count), the set ladder, the three-leg table with exact bytes + sha,
the intent path or the fired falsifier with verdict_scope, retained bytes + free space, every boundary, the serializer
rc, ending with the current own-vehicle frontier line.

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
