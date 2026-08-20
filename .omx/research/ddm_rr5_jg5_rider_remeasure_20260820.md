# ddm_rr5 rider RE-MEASURED on the jg5 pointer — 169 B, ΔS −1.125302e-04, lossless

`date_utc: 2026-08-20` · `owner: MAIN` · `axis: [byte-exact, lossless — no scorer run, decode
proven identical]` · `score_claim: false` · `promotable: false` · `frontier_moved: false`
· cost: **$0** (no Modal, no GPU, no scorer forward)

## THE ANSWER, FIRST

The CPR1 lossless rider, re-measured on the LIVE jg5 pointer body instead of the up3-era body it
was staged against, **saves 169 archive bytes — ΔS = −1.125302e-04 — and all three losslessness
controls PASS.** That is roughly 15× the entire banked micro-edit pool (qs2 −4.375e-6 + re1
−1.207e-6 ≈ −5.6e-6) from a transform that changes no decoded byte.

| Quantity | Value |
|---|---|
| base archive (jg5 pointer) | `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e` @ 180,625 B |
| rider archive | `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080` @ 180,456 B |
| REALIZED archive delta | **−169 B** |
| ΔS (rate term, 25/37,545,489 per byte) | **−1.125302e-04** |
| derived S | **0.14827847122030854** |

Basis payload 12,277 → 12,046 B (−231 B); Huffman table dropped, 16 packed-table bytes zeroed;
carrier body 22,026 B, −231 B. The archive-layer delta (169 B) is SMALLER than the payload-layer
delta (231 B) because Brotli responds to the change — this is `ddm_up3`'s lesson, and the tool
measures at the archive layer for exactly that reason. **169 B is the number; 231 B is not.**

## WHY d_seg AND d_pose ARE EXACTLY UNCHANGED — not projected

All three controls executed and PASSED on these bytes:

- **C1 arithmetic round-trip** — 27,648/27,648 symbols decode back exactly.
- **C2 carrier-body identity** — the decoder's `restore_carrier_body` returns the shipped body
  BYTE-FOR-BYTE, so every stage downstream of the carrier section is bit-identical *by
  construction*.
- **C3 receiver decode identity** — the REAL receiver
  (`runtime.residual_archive.read_residual_archive`) was run on BOTH archives and all **10** parsed
  parts compared byte-for-byte. Proof sha `b435877fd9ff59f4644544e8d2ab5e5502e9fd86b1ec3534275314d665bc3c40`.

Because the decoded stream is byte-identical, the scorers see identical inputs: d_seg and d_pose
are unchanged EXACTLY, and only the rate term moves. The S arithmetic above is therefore exact
given the controls, **not a projection**. It is still NOT an admitted pointer move: per the
authority rules a pointer move requires the exact `upstream/evaluate.py` row on the shipped bytes.
That row is owed and queued (single-flight — the rr8 wall-clock row is in flight).

## THE CROSS-REGIME CORRECTION THIS UNIT PAID

`ddm_rr5`'s own memo (2026-08-19) measured **183 B / ΔS −1.2185e-4** and labelled it "the pointer
body". That pointer was the up3-era 176,450 B body. The pointer has since moved twice (to1 →
jg5). Transferring 183 B onto jg5 would have been the [[cross-regime constant transfer]] genus —
the same genus rr5's memo itself flagged when it corrected the chain's inherited 278 B / −1.85e-4
down to its own measured 183 B. Re-measured on jg5 the payload saving is **larger** (231 B vs the
old 183 B at payload layer) but the realized archive saving is **169 B**, so the honest ΔS is
−1.125302e-04, not −1.2185e-4. The direction of the error was not predictable from the old number;
that is why it had to be re-measured rather than carried.

## FINDING — the tool carries a STALE POINTER-ERA PIN (#1138 genus, third instance)

`tools/ddm_rr5_rider_apply.py:82-85` hardcodes
`POINTER_ARCHIVE_SHA256 = "7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f"`,
documented as "the pointer body this rider was pre-staged against (ddm_up3 candidate)", and uses it
as the argparse default for `--expect-sha256`. On the live pointer the tool therefore REFUSED:

    RiderApplyError: input archive sha256 f3bce5d2… != expected 7ce46fd7…;
    refusing to touch a body the caller did not name

The refusal is *correct behaviour for a safety prompt* and I do not want it removed — but the
constant goes stale on every honest pointer move, which is exactly the defect already filed as
#1138 (`test_candidate_seal.py` hardcodes a pointer baseline). The jg5 runtime contains no such
pin, so this is tool-local, not a custody problem with jg5.

**How this run cleared it, and why that is safe:** the pin is a "did you mean this body?" prompt;
the three controls are the correctness proof. I passed `--expect-sha256` with the value **derived
from the file** (`shasum -a 256` of the jg5 archive), never hand-typed, so the pin became a no-op
for this call while C1/C2/C3 stayed fully armed — and they are what would have caught a carrier
layout the rider does not understand. They passed.

**Cure owed (two-landing):** derive the expectation from the named runtime's own pin, or re-label
the constant as `VALIDATED_AGAINST_ARCHIVE_SHA256` and make a mismatch a loud WARN that still runs
the controls, refusing only if a control fails. Not applied in this unit; filed so the next
pointer move does not re-pay the diagnosis.

## COMPOSITION NOTE — the rider and the rr8 port BOTH move the runtime tree

The rider emits its own `rider_runtime/` including a modified `inflate.py` (2,282 B), because the
decoder needs the adaptive-arithmetic restore path. So the rider moves BOTH the archive sha and the
runtime tree hash. `ddm_rr8`'s native corrector port moves the runtime tree alone (archive
byte-identical, tree `b8a43c6b…` vs shipped `2103073d…`).

They are orthogonal in mechanism and compose, but they are NOT two independent rows: the shipping
candidate, if both hold, is ONE object — {rider archive `df7fd266…` × ported+rider runtime} — and
should be sealed and fired as ONE T4 row, not two. Firing them separately would buy a row for a
tree we do not intend to ship. Fire order: adjudicate the rr8 wall-clock row first (it decides
whether the port ships at all), then compose.

## Own-vehicle frontier

**S 0.14839100138338618 @ 180,625 B [contest-CUDA T4 n600] — UNMOVED by this unit.** This unit
produced a candidate and an exact rate arithmetic, not a row.
