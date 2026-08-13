# JO1 six-event T4 row — adjudication (2026-08-13, MAIN)

## Finding
The vh3 harvest characterized JO1's T4 fire-order as HELD/UNFIRED. That was stale: the row
was already bought at 2026-08-13T01:03:56Z (run-id
`ddm_jo1_calibrated_6event_paired_modal_auth_20260813T010356Z_cuda`, 430.2 s wall).
Receipt: `/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/main_t4/MODAL_CUDA_ROW.json`
(`score_claim: true`, axis `contest_cuda`, recomputed from components).

## The row [contest-CUDA T4, n600]
- Archive: sha `cbcbb9ec22f81ad6ce2f8f97c976148831e825ba64312878a798d92a46907c8f` @ 186,253 B
  (cp135 base + six calibrated events, +1 B).
- S (recomputed from components): **0.1621711682636563**
- avg d_seg 0.0002965 · avg d_pose 7.23e-06 · rate 25·186,253/37,545,489 = 0.1240135

## Verdict vs the floor
cp135 composed floor: 0.16195513827824176 @ 186,252 B (seg 2.9643e-4 · pose 6.8856e-06).
JO1 is **+0.000216 WORSE**. Decomposition of the delta:
- seg: ~unchanged (2.9643e-4 → 2.965e-4; +0.7e-9 in d_seg, ~noise-scale)
- pose: **+0.34e-06 d_pose → +0.000205 S** — the dominant term
- rate: +1 B → +0.0000007 S

**JO1's open question ("exact seg/pose interaction of the six events") is ANSWERED by
measurement: the events are pose-harmful and seg-neutral at the exact instrument. DEAD at
INSTANCE scope (this six-event selection on this base). No pose job, no further spend.**

Scope note (verdict ladder): INSTANCE — the event-carrier FAMILY is not closed by this row;
a pose-null-projected event selection (Q3-projected placement, #889/#932 lineage) remains
the named reopening path, priced against the ~+0.0002 bar this row measured.

## Bookkeeping
- #381 spend: this dispatch (~430 s T4) was bought before this session's tally; treat the
  ~$2.4 running figure as approximate-low by ≤ ~$0.16.
- The T4 "batching decision" (re1t × JO1) is CLOSED: re1t fired standalone (in flight,
  run-id `ddm_re1_round1_t4_gate_20260813`); JO1 needs nothing.
- Cadence doctrine: this IS a byte-closed complete-S row — an honest negative, banked.
