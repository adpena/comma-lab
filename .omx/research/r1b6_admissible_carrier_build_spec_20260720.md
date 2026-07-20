# R1b6 admissible carrier — bounded implementation spec

UTC: 2026-07-20. `lane_id=r1b6_admissible_carrier`.

## Authority and outcome

Consume the R1b5 Fisher ordering and the exact factor-2 singleton preimage
solver, but measure the resulting bytes only after they pass through the
landed R1b4 receiver and the hard CPU-Torch oracle.  The first deliverable is
a real prefix receipt that recomputes the realized score recovery and the
conditional break-even byte budget.  It is `[macOS-CPU advisory]`, never a
contest score or pointer move.

## Bounded implementation

1. Decode a capped R1b4 baseline with zero boundary coefficients and the same
   xi0 payload used in both arms.
2. Read the SHA-bound 38,077-row Fisher ordering.  Preserve its global order,
   filter to the measured pair prefix, and preserve the Road-Lane-first tier.
3. For each selected scorer cell, solve both exact same-rounded-bin factor-2
   uint8 preimages encoder-side and choose the source-closest sign.  Record
   every exact numerator and write the chosen 2x2xRGB camera bytes into the
   existing canonical `R1K1` replay.  The receiver performs assignments only;
   its search count stays zero.
4. Decode the candidate through R1B4, verify two deterministic receiver runs,
   then score baseline and candidate with hard CPU Torch at seed 1234 and
   batch 16.
5. Recompute `B = max(0, Delta S_realized) * 37,545,489 / 25`, keeping the old
   1,852.091296-byte anchor alongside the new scoped value.  A prefix result
   remains prefix-scoped; only an n600 replay may refine the canonical law.

## Stop and admission rules

- If the selected receiver-bound replay is inert or has non-positive realized
  recovery, stop this formulation with a narrow `verdict_scope`; do not kill
  the boundary/full-kernel family.
- The 1,273-byte compact-binary-v2 projection remains inadmissible until its
  binary descriptor, direct shifts, and every counted section parse back and
  actuate in R1B4.  A successful absolute-write prefix does not launder that
  projected number into measured bytes.
- Rank-4 custody requires both a typed Frechet tangent and a distinct realized
  uint8 endpoint secant.  The prefix can anchor the latter, but cannot close
  n600 production custody unless all 16,319 moderate cells and their exact
  endpoints are materialized.
- Full VJP sidecar rehash stays fail-closed deferred unless producer inputs
  are actually present.

## Verification and landing

The measurement tool must use SSD scratch, storage preflight, atomic receipts,
deterministic model settings, source hashes, success-only scratch cleanup, and
tests for stream/order/refusal arithmetic.  Python changes require two clean
review-tracker passes and serializer commit.  MAIN must independently review
the branch and rerun focused tests before landing.
