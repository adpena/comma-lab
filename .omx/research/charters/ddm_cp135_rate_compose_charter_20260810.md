# ddm_cp135 — RATE COMPOSITION ON THE PR135 BASE: the fastest sub-bar row

## Mission (operator 2026-08-10: "laser focused on frontier score lowering... all signal from everywhere")

Build the first candidate BELOW the custodied bar by composing OUR measured lossless rate
levers onto the PR135 base (off the shelf, operator-granted). The bar in OUR custody:
**S = 0.16226942370411543 @ 186,724 B** [contest-CUDA T4, locked env, n600] — replayed
2026-08-10, archive sha `12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`.
Every −1,494 B on this base = −0.001 S. A −1,000 B lossless composition ≈ S 0.16160 = NEW RANK-1
CANDIDATE by rate alone, before any pose work.

## The levers (each MEASURED on our side; the question is ABSORPTION + composition on THEIRS)

1. **lc2 same-state ANS token recode** — our lc2 cross-state token payload measured 178 B
   SMALLER than F26's (pi135 receipt). Race our ANS coder against their RC64/WANS1/CAP1 stack
   ON THEIR TOKEN STREAM (decode their sections to symbols with their own runtime code, re-encode
   with ours, round-trip symbol-identical on CPU).
2. **VP1 split-model −903 B lossless** — FIRST run the absorption check (ah2 fire-order 1): is the
   split-model gain already inside PR135's ~4.3KB lossless improvements (WANS1/CAP1/RC64/
   fixed-schema)? Only compose if NOT absorbed. Evidence: /Volumes/VertigoDataTier/pact/
   ddm_ah2_lossless_compose_20260810/ + the vp1 arm receipts.
3. **HP3 small lossless lever** (ah2 survivor) — same absorption-check-first discipline; its bytes
   must be RE-CODED into the composed container, never added linearly (ah2's warning).
4. Any additional lossless slack fd135's L1 section-vs-memoryless-bound table surfaces when it
   lands (read `.omx/research/ddm_fd135_*` if present; do NOT block on it).

## Method + constraints

- Custody: `/Volumes/VertigoDataTier/pact/pr135_intake_20260810/` — archive + pr135_src runtime
  (READ-ONLY intake; COPY to a work dir before modifying). Their inflate is CUDA-locked at the
  renderer (f26_inflate.py:105) — you CANNOT decode frames locally. Verification ladder that
  needs NO GPU: (a) entropy layer is CPU — decode each section to SYMBOLS with their code,
  re-encode with ours, assert symbol-stream BIT-IDENTITY; (b) adapt their inflate.py to parse the
  recoded sections (rule-118: coder CODE free, payload counted; keep their CUDA path untouched
  otherwise); (c) full parse-back of the composed archive. Frames identical BY CONSTRUCTION when
  (a) holds and the receiver consumes identically-valued symbols.
- Deliverable: composed `archive.zip` (bytes < 186,724 or the honest absorption verdict per lever)
  + adapted runtime dir + parse-back receipt + per-lever byte ledger {lever, −B, absorbed?}.
  MAIN fires the ONE Modal CUDA row (single-flight; do NOT dispatch Modal yourself).
- ALWAYS KEEP THE PAYLOAD: persist every composed candidate + intermediate section payloads to
  /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/ with sha256+bytes in the result JSON.
- borrowed_substrate_accounting REQUIRED (NO-FAKE #7 honesty-half): the base + runtime are
  codexblack's (PR135); our original bytes = the ANS recode + VP1/HP3 levers. Say so in the memo.
- Durable memo `.omx/research/ddm_cp135_rate_compose_20260810.md`, serializer commit
  (post-edit --expected-content-sha256, tags [no-triality] [p0-ledger-ok]). Checkpoint per protocol.

## OPTIMAL FORM

Reference: the #996 coder-axis protocol (race per-section vs own memoryless bound, real coders,
whole-container recount — never linear addition of per-lever savings) + the ah2 absorption-check
discipline. SCOPE = one archive, 3 named levers + fd135 extras; NO mechanism reduction (real
coders, real container, real parse-back — no estimated byte counts). Pins: PR135 archive sha
`12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`; our lc2 anchor sha
`f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`; PR130 bar sha
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
PRIOR-LAW PREDICTION (m24/#940 races-not-reputation + ah2): at least one of {lc2-ANS, VP1} will
be PARTIALLY absorbed by their 4.3KB lossless stack — predicted composed win −300 to −1,100 B,
NOT the naive −1,081+ sum; if ALL levers absorb to ≤0, that is the honest verdict and the rate
axis on this base is CLOSED at their form.

## Falsifier

Composed candidate ≥ 186,724 B after all three levers raced → PR135's lossless stack dominates
ours on its own sections; record per-lever absorption evidence and close the rate leg honestly.
