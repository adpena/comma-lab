# jrd1 byte preflight reviews

Research-only source: `experiments/ddm_jrd1_byte_preflight.py`.
These reviews cover the byte instrument, not an implemented joint descent probe.

## Pass 1 — source and mechanism review

Traced move-52 runtime copying, SM3R walk, fixed fp16 scales, SM1S encoding,
CK2 layout, Brotli settings, ZIP header length, and receiver parse-back.
The original rw1 loader requires RC1S; move 52 uses SM1S. The new instrument
uses the copied receiver and ren2's actual representation reader. It neither
changes the old helper nor transfers its old price law. Replaced an assumed
receiver-side interleave function with the existing up3 encoder function.
Added immediate body/metadata retention and capacity checking before ZIP writes.
The final code requires exact full-archive null identity and refuses unsupported
semantic split formats. Marked all 9 current entities reviewed through
`tools/review_tracker.py mark-file` after those edits.

## Pass 2 — actual artifact and refusal review

The actual move-52 body passed: 14/14 archives (control and six mutations, each
twice) re-hashed to their receipts, 7/7 twin pairs agreed, and the null archive
reproduced `ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e`.
The shipped receiver restored each SM1S body exactly and retained all unchanged
HPAC, token-stream, carrier, mixer-weight, and residual sections byte-for-byte.
The draw is 12 distinct pairs from the 156-pair pd4 pool and 12 from its 444-pair
complement, using PCG64 seed 20260916; it is neither a prefix nor winner-selected.

Negative checks executed against the real instrument: wrong archive pin refuses;
an attempted Vertigo destination refuses before writing; a capacity request of
the full 3 GiB allowance refuses while existing payloads are retained. No foreign
file was created. No scorer or gradient result is inferred from these controls.
Marked the same 9 entities reviewed a second time. No review override used.

Remaining limits: global scorer ownership is unassigned, the three exchange rows
are unimplemented/unmeasured, and no pose solve or global collateral check ran.
The six seeded byte actions are not the charter's gradient-selected actions.
<!-- # FORMALIZATION_PENDING: review receipt, no scientific law or score -->
