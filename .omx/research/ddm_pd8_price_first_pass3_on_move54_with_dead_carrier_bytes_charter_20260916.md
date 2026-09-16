# ddm_pd8 — price-first pass 3 on move 54's re-rendered field, with the 16 unread carrier bytes and the unused header flag bit removed in the same byte close (charter, MAIN 2026-09-16; Opus)

## Why (gs3 Addendum 70; pd7 memo `.omx/research/ddm_pd7_price_first_pass2_on_move53_20260916.md`)
Move 54 (S 0.13605599532783202 @ 179,266 B; archive sha 5c6bf403b4cb4554fe24a46bdf5b46d62876854764a10a22d8c90d76a7292ee6;
tree `/Volumes/APDataStore/pact/ddm_pd7/candidate/candidate_runtime`, read-only; leg 1,112.2 s at
`/Volumes/APDataStore/pact/ddm_pd7/SEAL_ddm_pd7_price_first_pass2_contest_cuda.json.decode_wall_clock.json`, unconsumed;
re-derive from `reports/latest.md`). The cheap half is a property of the field and re-prices after every move (rank-0
sheet 2.163 → 3.075 bits/token between passes 1 and 2; paying pairs 128 → 72): pass 3 is expected to yield less than
pass 2 (−4.5e-5) and must be run cheaper. Separately, the second fresh reader of the minimal receiver found 16 bytes in
the carrier section (offsets 123–138, zeros) that no decoder reads, and header flag bit 32 set but unread
(`.omx/research/ddm_mrs6_20260916/RATE_LEVER_unread_carrier_bytes.md`): −16 B with an identical decode, IF the sealed
receiver (the tree above) also never reads them — verify by reading `runtime/` in the sealed tree; if it reads them,
they are not dead and the lever is closed at instance scope.

## The rung
1. Base = move 54 by parse-back; pose base under the measured-tolerance gate; pricer proved by byte-identical repack
   (twins, two processes); rows re-captured (reproduce per_frame_bits ≤ 1e-6 relative or STOP).
2. Cheap-half enumeration as pd7 (≤ 8 bits/token, singles + 2-token, 588 pairs) — report the rank-0 sheet cost vs
   3.075 (the decay curve is a finding in itself); timing smoke first; budget declared.
3. Credit afterwards → resolved-pose admission per real bit → SET pricing (iterate) → closed-archive size ladder.
4. THE DEAD BYTES: in the same byte close, produce the archive with the 16 carrier bytes and the flag bit removed
   (the carrier section shrinks by 16 B; the header flag byte loses bit 32) and PROVE the sealed receiver decodes it to
   the identical raws (600/600) — this is a container change: twins, cold n600 public parse-back, census; if the
   receiver reads either, do NOT ship it and report exactly where. Report the rate leg with and without.
5. Three legs on the shipped bytes' own cold decode (render from the SHIPPED field — pd7's law; never an overlay of an
   earlier set); band −1e-5 … −6e-5 S (pass 3 is expected to be smaller); falsifiers: < 30 pairs paying at their own
   price; net > −1e-5 (a pointer move is still a pointer move at 0.6 bar — the bar for "closes pass 3" is −1e-5, and any
   net > −2e-5 is reported as "below the 2e-5 admit bar" — MAIN decides whether to fire); set re-price vs ledger > 10 %;
   projection-vs-decode pose gap > 1.5×.
6. If it nets: byte-close on move 54, twins, cold n600 parse-back, manifest (Catalog #420), census (the RLC1 rider must be
   present), smokes, retention ≤ 2 GiB with shas, NORMAL seal inheriting move 54's leg (pr18 digest 9f6e7168… must match
   — unless the container change alters the receiver behaviour digest; it should not, the receiver is unchanged). MAIN
   fires — CUDA first, then the CPU sibling after DISPATCHED.
7. Memo `.omx/research/ddm_pd8_price_first_pass3_on_move54_20260916.md`; serializer commits; lane
   `ddm_pd8_price_first_pass3_20260916`; checkpoint `ddm_pd8`.

## Boundaries
As pd7's charter (no Modal, fire, packet, authorize_*; upstream/, the PR trees, sealed trees, contract/receiver code,
renderer, basis, prior read-only; pd1–pd7/jrd1/jrx*/psa*/mrs* stores read-only; heavy steps via
`tools/launch_detached_process.py --done-receipt <bare-name>`; never SIGTERM a running detached encode; the pricer refuses
restarts over partial checkpoints; report APDataStore free space (≥ 8 GiB + retention must remain; ~24 GiB now); retain
≤ 2 GiB; never write Vertigo; no ScheduleWakeup; two review passes per .py; `[no-triality] [p0-ledger-ok]`; never a
co-author trailer or AI attribution). psa2's resumed exhaustive search runs at nice 10 on the same host and also uses
the scorer — expect contention; your admission legs are not timing-gated. Label MEASURED / DERIVED / INFERRED / ASSUMED.

## OPTIMAL FORM
Reference forms: pd7's producer (`experiments/ddm_pd6_price_first.py` lineage + pd7's fixes: the `setabsorb` iteration
defect and the premature bulk prune are documented in pd7's memo — do not repeat them), row capture, sheet pricer, set
pricing, closed-archive ladder, render-from-shipped-field. Declared deltas: the field is move 54's (the object moved);
the dead-byte removal (a container change proven by identity). Provenance pins: pd7 memo sha 0205df352e662866; packet move 54
commit (record from `git log`); archive sha above; leg sidecar sha 609addb9acb938a9; reader lever memo sha da9ac35fffff9020.

## Prior negatives accounted (operator 2026-08-15)
pd5 (credit does not compound — singles + 2-token only); pd4 (selection bias — set pricing); pd6/pd7 (overlay ≠ shipped
render — render from the shipped field; rider drop — census); pd7 (setabsorb no-op; premature prune); pass 8/pp1 floor
pairs (excluded); m132 collateral; m88 (no prefix stop); the 34.8 B lottery (twins); pr19 (inherit the leg once).

Final message: as pd7's, plus the rank-0 decay figure, the dead-byte verdict (identity with/without), ending with the
current own-vehicle frontier line (re-derived at start).

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
