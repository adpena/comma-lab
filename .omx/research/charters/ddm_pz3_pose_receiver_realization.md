# ddm_pz3 — REALIZE the 2,860 B pose packet through a real receiver

## WHY THIS IS RANK 1

PZ2 measured (scorer-free, official DALI GT, n600) that PR130's **23,384 B** pose carrier can be
replaced by a **2,860 B** direct-precision packet at quantization MSE **6.912240592300149e-07**.

Its projections against the PR130 base (S = 0.17214129749189644 @ 191,052 B):

| projection | S |
|---|---:|
| perfect realization | **0.14583670958983525** |
| additive error | 0.15869992400122784 |
| worst-aligned error | 0.16110432236983460 |

**All three clear the bar. The optimistic one clears sub-0.15 alone.** Nothing else on the board
is this large: −20,524 B is 66.5% of the 30,842 B that separates our best measured candidate
(S = 0.170536856816211 @ 188,636 B) from sub-0.15.

And every one of those numbers is stamped
`projection_axis = [TOY-BRACKET over contest-CUDA,DALI,n600 base components; no receiver/scorer]`.
The packet exists; the frames it implies do not. **PZ3's whole job is to close that gap or to
measure honestly why it cannot close.** A projection is not a row.

## OPTIMAL FORM

**Reference form.** The family is *stored-target pose carriage realized through a decoder* — the
mechanism PR130 already ships and proves at 23,384 B / d_pose 2.331e-05 (receipt:
`.omx/research/ddm_pz2_pose_representation_20260810/PZ2_COMPACT_RECEIPT.json`, base_score
0.17214129749189644). PR130's own carrier IS the reference implementation; we are changing its
REPRESENTATION (bit allocation per pose dimension), not inventing a new carriage mechanism.

**Deltas, each declared:**
- SCOPE reduction (legal): a first realization pass MAY run the n=120 seeded stratified subset
  (`selection.seed = 20260809`, `selection.receipt_sha256 =
  2e2778fd65e69c2af3ddcd1bff1bed3db3737a54743df89393e1e8f673a90f99`) before n600. Per m88/m96 the
  POSE axis is the one where prefix bias is worst — so the subset MUST be the seeded stratified
  selection, never a prefix. Any headline row is n600.
- MECHANISM reduction: **NONE PERMITTED.** The receiver must be the real decode path that
  `upstream/evaluate.py` will run. A harness that computes d_pose from the packet without
  materializing frames reproduces PZ2's toy bracket and answers nothing.

**Provenance pins.**
- base archive `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`, 191,052 B
- packet `f332b3f6a52cdb4661baf35d6767d1cb6325d422d9c91ebcd5d061794174e668`, 2,860 B,
  `repeat_byte_identical = true`
- official DALI GT cache `382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195`
- retained pose targets `23ae28d20aee8697d87e015c1145b248c111bc3ce61b9b66793e770d65522b2a`, 14,400 B
- PR130 intake git head `e34f31bc4969042c0051ac81aa3c56884419a231`
- PR130 code is OFF-THE-SHELF AUTHORIZED (operator 2026-08-06). Reuse its carrier/receiver
  directly; the honesty half stands — `borrowed_substrate_accounting` + attribution in the receipt.

## THE WORK

1. **Read PR130's actual pose receiver** in the intake clone (READ-ONLY; never `git add` inside it).
   Establish exactly how the 23,384 B carrier reaches frames the scorer reads. That path is the
   thing you are re-pointing at a smaller packet.
2. **Build the receiver** that consumes the 2,860 B packet and produces the pose-carrying frames.
   Byte-identical when the packet is the base one; that equivalence is your positive control.
3. **MEASURE d_pose through the real path** — packet → receiver → frames → the exact PoseNet the
   scorer runs. Not the quantization MSE. The REALIZED d_pose.
4. **Byte-close.** Real `archive.zip`, real bytes, real sha. Report S from
   `tac.contest_score` / the canonical component recompute, never a rounded display field.
5. Retain EVERY payload to `/Volumes/VertigoDataTier/pact/ddm_pz3_20260810/retained/` with
   sha256 + bytes in the result JSON (P0 DEF CON 1000: a run that materializes a payload and
   persists only its length is FORBIDDEN).

## FALSIFIER (pre-registered)

The projection assumed realization is free. It is not. Declare the family **REALIZATION-LIMITED**
if realized d_pose exceeds the quantization MSE by more than the additive-error bracket already
priced — i.e. if realized S ≥ 0.16110432236983460 (the worst-aligned projection). At that point the
gap is receiver loss, not representation, and the honest next object is the receiver, not a
smaller packet.

If realized S < 0.17214129749189644 you have a real candidate: say so plainly, hand MAIN the
byte-closed archive path + sha + bytes, and STOP. MAIN owns the exact-eval dispatch.

## BOUNDARIES

- **NO Modal, NO paid dispatch, NO contest-CPU/CUDA claim.** MAIN holds that slot and its claim.
- Local rows are `[macOS-CPU advisory]` with `score_claim=false`. The bar moves only through
  `upstream/evaluate.py` on contest hardware, and only MAIN fires it.
- upstream/ is IMMUTABLE. Public-PR intake clones are READ-ONLY.
- If d_pose cannot be measured through a real receiver in this arm, say that in one sentence and
  report what you did measure. A named blocker beats a projection dressed as a result.
