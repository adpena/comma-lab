# ddm_tk1 next if resumed

## Folded

- Fold the small static learned-prior label coder at this form. Exact TK1 frame
  bytes are 700111 B on tq1c and 713345 B on GT, both far worse than PP1 KT.
- Fold raw dense raster compressors for this object. Best generic row is bz2-9
  at 285394 B on tq1c and 338593 B on GT, still worse than PP1 KT.
- Do not rerun the pp1 GT partition price unless changing the coder family. The
  n600 GT KT price remains 173617 B here and the tq1c shipped-vehicle partition
  price is 142001 B.

## QUEUED-WITH-FIRE-ORDER

- Owner: next scorer-lane claimant. Fire first: D1 semantic renderer
  source-forward n600 SegNet-only closure. Claim a scorer lane before firing.
  This is the cheapest discriminator for the open renderer cell.
- Owner: D1 discriminator owner after a GO. Fire second only if D1 lands near
  `d_seg=0.0002966`: compose a byte-closed Route S candidate using the 142001 B
  semantic stream and the cheapest renderer that survives parse-back.
- Owner: next byte-coder claimant after a D1 NO-GO. Fire third only if Route S
  stays at the flat-paint floor: move to Route H and price only the residual
  cells in the boundary/appearance-critical annulus. Use 2436.19 GT-boundary
  px/frame as the starting denominator, not full-frame pixels.
- Owner: H residual claimant. Budget condition before firing H against 0.19110:
  with PR130-pose base, residual budget is only 50633 B at `d_seg=0.0002966`
  and 20086 B at `d_seg=0.0005`; above that, H may beat live L but does not
  beat the competitive target.

## Guardrails

- This unit was scorer-free. Any resumed scorer run must claim the lane and
  record command, source hashes, and hardware axis.
- Keep et2, rw1, and vo2 live surfaces read-only unless the operator assigns
  those lanes.
- Do not cite the `learned_static_prior` estimate-only rows in
  `semantic_stream_race.json` as measured bytes. Use
  `learned_prior_exact_addendum.json` for exact learned-prior rows.
- A composed archive score is still absent. Pointer delta remains unmoved.
