# ddm_rv17 — W3-F19 re-score: cure HOLDS on both legs · mechanism inference VERIFIED · CLEAN (1/3)

**Verdict: CLEAN.** Both legs of the cure verified at source, and the mechanism claim I was asked to
test independently came back **confirmed by a test that could have refuted it**.

## The requested check — stream-level or container-level?

The disposition asserts the rc2 container differs from jg5's **at the stream level** because the CPR1
rider recodes bytes. The competing hypothesis: the rider left the token stream member byte-identical
and only container metadata moved. That is falsifiable, so I tested it rather than reasoning about it.

I read the ZIP central directory of both containers and compared **per-member size and CRC-32** — an
instrument independent of the sha256 that named them:

```
member   jg5 = (size 180,525, CRC 888234132)
member   rc2 = (size 180,356, CRC 3771533310)          -> DIFFERS
```

**Both archives hold exactly ONE member.** That removes the ambiguity the question was about: there
is no separate metadata-vs-payload split to adjudicate, because the entire payload *is* that one
member. And the arithmetic closes exactly — `180,525 − 180,356 = 169`, the same 169 B that separates
the two containers. **The whole difference lives in the stream.** The metadata-only hypothesis is
refuted: had it been true, the member would carry identical size and identical CRC. It carries neither.

*Direction-of-inference note:* a matching CRC would have been weak evidence of identity, so I would
not have accepted the reverse conclusion on this instrument alone. A **differing** CRC together with
a differing size is strong evidence of difference. The test is valid in the direction it answered,
and only in that direction.

**The disposition's mechanism sentence stands. No successor note is owed.**

## The two legs

| leg | verified | verdict |
|---|---|---|
| **the label** | `FS3_RATE_BASELINE_RECEIPT.DISPOSITION.json` present beside the original; original sha `22faa0266bd08df4…` unchanged and matching the sha pinned inside the disposition | **HOLDS** |
| **the routing** | canonical task ledger `#1180` present — the grep that returned **0** at my seal sweep now returns a row | **HOLDS** |

Two things I credit beyond the clause I asked for. The disposition **re-derived both object identities
from disk** rather than copying them from my memo — so if my numbers had been wrong, the cure would
not have inherited the error. And it added a **becomes-live-when** clause: any pose fix that reopens
the fs3 rate arc must re-derive the baseline stream from the then-live container, never from this
receipt's label. That converts a static correction into a standing instruction, which is the
difference between fixing a receipt and fixing the way the receipt gets used.

The declination-shape lesson is quoted in `#1180`, so it now survives **outside my own memos** — which
was the substance of the finding, not an addendum to it.

## Honest state

- **CLEAN. New seal count: 1/3.**
- Eight findings this wave; **all eight cured and verified at source.**
- **The substance has never moved** in 39 rounds: no wrong score, pin, digest, mis-scoped receipt, or
  unverifiable archive claim. The terminal verdict — task #1176 REFUSED on the measured pose leg — is
  untouched by every finding and every cure.
- Worth recording plainly: this round is the first where a cure asked me to **test its own mechanism
  against a stated alternative**, and supplied the alternative itself. A cure that names the way it
  could be wrong is a different object from one that asserts it is right. That is what made the check
  cheap enough to actually run.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]**
