# Einstein-Kolmogorov crux v3 external review

**UTC:** 2026-07-20  
**Review instance:** exactly one, `ek3_review_same_packet`  
**Session:** `019f7eb2-2598-7c22-9256-3157afb88252`  
**Verdict:** `FIX_ONCE`  
**Authority:** `[macOS-CPU advisory] NON-PROMOTABLE`

## Immutable scope reviewed

The reviewer independently inspected commits `9e8b302a56`, `80842be655`,
`87cf1bda86`, and `cbbac4837c`. It made no edits and created no commit.
Focused verification was 8 passed.

The review independently confirmed:

- old and repaired archives are byte-identical at 91,062 bytes, SHA-256
  `3555bafcccac0827225a87f07dc5b093381de3188560cb7002f2bf9ac2b37c6a`;
- the receiver constants are self-contained and hash-bound;
- the hard CPU-Torch score used all 600 pairs and is honestly labeled local
  macOS advisory;
- derived `S=35.955425463668846`, delta `+35.76434263946884` versus the
  0.1910828242 bank, and 173,258-byte headroom versus 264,320 bytes;
- classification A (measured but negative), certified raw cleanup, and an
  unmoved pointer are honest.

## FIX_ONCE findings

1. A capped decode was confirmed against the original full blob rather than
   the exact capped `archive/0.bin`. The fix must parse and confirm the exact
   scored blob.
2. The old receipt's `max_pairs=24` label enclosed only a two-pair/four-frame
   bit-exact gate. Strict evidence must check all 24 pairs / 48 frames, and the
   validator must reject two-pair evidence.
3. Resume validation must bind the emitted receiver, GT cache, byte-close
   scorer surface, hard-oracle module, and contest-score source; the completed
   receipt fast path must revalidate those bindings and finish interrupted raw
   cleanup.

## Bounded resolution

Commit `261a2c8296049d6bef6a43f1aa545d653618a398` applies exactly that fix set.
The refreshed capped diagnostic confirms 24/24 frame-zero pairs with maximum
absolute uint8 difference zero. A separate direct strict gate confirms all 24
pairs / 48 frames bit-exact with zero differing frames and maximum difference
zero. The 91,062-byte packet is unchanged, so the already completed n600 hard
score was not rerun.

The new operator redirect dated 2026-07-20T08:54:58Z through 08:59:03Z makes
this row terminal evidence: subsequent work must compose the proven C1
distortion capstone and frontier rate solver at their joint KKT optimum, not
iterate this collapsed-distortion packet.

## Review custody

- Final review text SHA-256:
  `08686e1c1eeedcb38baabe2d85f07f7a56f23a83466d0878bc686cce27e32129`
- Full review log SHA-256:
  `839c91873b5d9508d237978c3648f7f033b00e78a3c3ab8f4a65f8ba244f53e4`
- Reviewer final:
  `.omx/tmp/codex_runs/ek3_review_same_packet_20260720T084348Z.last.txt`
- Reviewer log:
  `.omx/tmp/codex_runs/ek3_review_same_packet_20260720T084348Z.log`

This branch is not MAIN. MAIN must review the bounded fix commit, refreshed
receipts, regression tests, and the operator redirect before landing.
