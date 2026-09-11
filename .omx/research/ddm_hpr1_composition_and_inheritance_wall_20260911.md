# ddm_hpr1 — the composition row on move 47, and the inheritance wall it ran into

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false ·
`# FORMALIZATION_PENDING: the composition price is one measured row and the inheritance finding is a reading of the shipped contract's own two refusals; neither is a law until a second instance prices it.`

Base: move 47, S 0.13654774984742127 @ 179,359 B [contest-CUDA T4 n600], archive `d1fab05d69f31c90…`.

---

## 1. The composition row — PRICED, PROVEN, and BLOCKED ONE STEP FROM THE SEAL

ntb2's even-rounding of the frame embedding (the 2,320-of-4,800 rule as it shipped), re-applied ON TOP
of the retrained prior. The base is read out of move 47's own promoted archive, so this composes
rather than re-does.

| leg | composition | move 47 | Δ |
|---|---:|---:|---:|
| `hpac` member | 11,629 B | 12,262 B | **−633 B** |
| RLC1 stream | 118,896 B | 118,511 B | **+385 B** |
| **archive** | **179,111 B** | 179,359 B | **−248 B** |
| sha256 | `d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c` | — | — |

**ΔS = −1.651330204e-04**, projected **S = 0.13638261682704697**, **8.3× the −2e-5 bar**. Values moved:
**2,310 of 4,800**.

**The composition is very nearly additive, and that is itself the finding.** ntb2's identical rule on
the OLD prior bought −603 hpac / +358 tail = −245 B; on the RETRAINED prior it buys −633 / +385 =
−248 B. **The rounding's value did not depend on the prior being stale** — the two levers are
independent in the way the naive reading hoped but nobody had measured. (Still not a sum: −887 and
−245 do not add to −1,132; the composition's true position is 179,111 B against move 45's 180,246, a
joint −1,135 B, and only the exact encode says so.)

**Proof status — everything except the seal:**

- **Cold n600 public parse-back PASSES.** Raw **3,662,409,600 B**, sha
  `2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc` — identical to the pointer's own
  retained raw. Cold start, no resume, token cache DISABLED, `F26_TOKEN_DECODER=python`, 985 s, no
  scorer. **Four archives now decode to one raw** (45, 46, 47, composition).
- Staged from move 47's promoted tree; content diff **exactly `{archive.zip, inflate.py,
  MANIFEST.sha256}`**; manifest regenerated outside the tree; all rows re-hash clean.
- Public-entrypoint smokes with the pointer read live: **problems `[]`**.
- Receiver behavior digest **`9f6e71680a13d859…`** — equal to moves 46 and 47 and to the measured leg.
- Twins agree; output-lossless by in-loop assertion; twelve receipts staged in `seal_inputs_comp/`.

## 2. The inheritance wall — the pr19 question, answered by the contract itself

MAIN asked me to seal inheriting move 47's leg and, if a second hop is refused, to STOP and report the
exact refusal. **Both available routes refuse, and the two refusals together are the answer.**

**Route A — inherit from move 47's leg (the second hop).** Verbatim:

```
decode_wall_clock: decode_wall_clock: inheritance must point directly to a measured or
t4_direct leg (never to an inherited one)
```

`src/tac/decode_wall_clock.py:574` — `_require(source.get("mode") in {"measured", "t4_direct"}, …)`.
**Chains are forbidden by contract.** Move 47's leg is `mode: inherited`, so it can never be a source.

**Route B — inherit directly from move 46's `t4_direct` leg**, which is what the first refusal's own
wording points at, and the same leg move 47 used. Verbatim:

```
decode_wall_clock: decode_wall_clock: source measurement is not the pointer archive
```

`decode_wall_clock.py:578` — `_require(source.get("archive_sha256") == pointer_archive_sha256, …)`.
The measured leg was taken on move 46's archive `a0de607d…`; the pointer is now move 47's
`d1fab05d…`. **The measured leg is no longer the pointer's.**

### What the two conditions mean together

An inherited row is legal only while **the pointer IS the archive the measured leg was taken on**.
Move 47 consumed move 46's leg by becoming the pointer. Therefore:

> **The inheritance contract admits at most ONE inherited row per measured leg, and only as that
> leg's immediate successor.**

That is not a bug I can route around, and I did not try to. The behavior digest is identical across
move 46, move 47 and this candidate (`9f6e7168…`), so the *physics* of the inheritance is satisfied —
the receiver code that was timed is byte-for-byte the code that would run. What refuses is the
contract's **identity** clause, not its **timing** clause.

**MAIN decides.** The three resolutions I can see, stated without advocating one:

1. **Measure a t4_direct leg on move 47's archive.** Restores inheritance for its successors at the
   cost of one paid T4 decode. Cleanest, and the only one needing no contract change.
2. **Fire the composition as a first-measurement row**, as ntb2's frame_even did at move 46. It is not
   a receiver change, so this over-pays procedurally, but it is available today.
3. **Amend the contract** so a leg may be inherited transitively while the receiver BEHAVIOR digest is
   unchanged — the digest the contract already computes and already compares. This is the pr19
   question proper, and it is a contract decision, not an arm's.

**The candidate is otherwise complete.** The moment MAIN resolves this, the seal is one command with
the inputs already staged.

## 3. The step-4 variant and the q rung

Both are IN FLIGHT on the same rail, detached with done-receipts, and their prices land in this file.

**`retrain_frame_quad`** (step 4): 3,543 of 4,800 values moved, hpac **11,146 B**, model leg
**−1,116 B** — MEASURED. Its tail cost, and therefore its joint, is owed. Note it must beat the step-2
row's −248 B, not merely move 47: a bigger model saving is not a win if the tail gives more back.
Launch history worth carrying: its first run died at frame 400 on the 40 GiB reserve and its RESUME
then correctly REFUSED, because I had changed the producer in between and the checkpoints were no
longer bound to it. The immutability guard did its job; the rerun is from zero.

**The q rung** — the coder's scalar q re-solved on the REFIT prior, against mxo3's prior negative of
**−183.53 B** on the OLD prior. Instrument: `experiments/ddm_hpr1_q_rung.py`, scoring a 64-bin
calibration table that `--collect-q` accumulates online from the same hook the arithmetic encoder is
fed from; folds by frame PARITY so they interleave the video; each fold coded under the other fold's
KT table with back-off, **using mxo3's own `cross_bits`, imported and not re-derived**, because a
prior negative and its re-test must be one instrument. The scorer carries its own control: on a
synthetic perfectly-calibrated table it returns **−0.04 B**, so it cannot manufacture a gain.

**I threw away my first version of this statistic.** It collected the probability the coded row gave
the symbol *actually coded* — an event whose frequency is 1 by construction, so any refit of it is
degenerate and would have produced a confident number about nothing. I stopped my own 90-minute run
rather than let it finish into a wrong answer, and replaced it with the row's own confidence against
whether its argmax was right — an event whose realisation varies. The exact binary baseline is
accumulated alongside, because binning it after the fact would have left the baseline approximate
while the challenger stayed exact, a comparison tilted by construction.

## 4. One storage decision, certified not silent

The 40 GiB Vertigo reserve fired on the q-collection launch. **I did not lower it.** Instead I
certified and reclaimed ONE payload: move 47's own cold-decode raw, which was a byte-for-byte
duplicate of the pointer's retained raw `2b762eba…` that another arm still holds — verified by
streaming both immediately before reclaim. Certificate:
`/Volumes/VertigoDataTier/pact/ddm_hpr1/CERTIFIED_RECLAIM.json`, recording path, bytes, sha256, the
producing command, the surviving identical copy, and why the CLAIM those bytes support is preserved
(RESULT.json, the committed seal's RAW_IDENTITY receipt, and the landed move-47 pointer row). **The
composition row's raw is NOT reclaimed**: its row is unsealed and its bytes are live evidence.
