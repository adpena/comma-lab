# ddm_dx1 — the dxi lever priced bit-exactly, and a fruit sweep that found no fruit

**Axis:** `[exact local byte arithmetic, no scorer]` for every byte number below. No score claim.
**Base:** jg5 — `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e`, 180,625 B,
S `0.14839100138338618` `[contest-CUDA T4]`, lane `lane_ddm_jg5_waterfill455_t4_20260820`.
**Store:** `/Volumes/VertigoDataTier/pact/ddm_dx1/retained/`.

---

## Headline

**The dxi lever is essentially closed. Its measured ceiling is −18 B (ΔS −1.199e-5), not the
−1,787.7 B (ΔS ≤ −0.00119) ov1 published — 99.0% of the claimed headroom does not exist.**
Two independent errors produced the old number: it was computed on the wrong object, and it was
a per-context bound that ignores model cost. The positive control ov1 could not pass now passes
bit-exactly, on jg5, which is what makes this a price rather than a bound.

**The fruit sweep found no fruit.** Every named candidate is already spent, already refused, or
already inside jg5. No composed candidate is admissible, so this arm seals none.

**One real thing was discharged.** rung-4's second door — the carrier re-code byte budget, which
pq6 records as UNMEASURED — is now measured on the shipping body: **72.7 B projected at 600 pairs
against a 4,873 B budget, clearing by 67.0×.** Its first door does not clear: the 99.9874%
cancellation that reopened rung-4 rests on 3 pairs, and jg5's own 454-pair re-solve of the same
mechanism lands at **99.7254%** — the n=3 figure is optimistic by **21.80×** in residual.

---

## §1 The dxi lever — the control that ov1 could not pass

`experiments/ddm_dx1_dxi_recode_race.py`, receipt `retained/DX1_RECODE_RACE.json`.

ov1 reconstructed the coded symbols as per-coordinate temporal deltas of the raw codes and landed
+1.19% off the shipped bit count, so it reported a bound and said so honestly. The coder does not
code that array. CAP1 codes

    U = zigzag( forward_ar1( codes ) )

where `forward_ar1` inverts `restore_ar1_bias` — a per-dimension q8 factor plus bias under
`signed_mod`. Coding **that** array through the receiver's own `carrier_repack._rice_encode`
reproduces the shipped stream exactly:

| control | value |
|---|---|
| Rice bits re-encoded / shipped | **78,628 / 78,628** — match |
| Rice `ks` re-encoded / shipped | `[9,9,9,8,8,9,9,9,9,9,9,9]` — match |
| Rice payload byte-identical to the shipped tail | **True** |

Note the shipped payload on jg5 is **9,829 B**, not the 9,754.5 B ov1 quoted — that number is
the rr4 body's. jg5 is a carrier re-solve; its carrier is a different object. The cross-body
re-verification the charter demanded was not a formality.

## §2 What the headroom actually is

| bound, on the TRUE coded symbols | bytes | headroom vs 9,829 |
|---|---:|---:|
| order-0, one global model | 9,523.6 | 305.4 B (**3.11%**) |
| order-0, per dimension | 7,968.2 | 1,860.8 B (18.93%) |

The per-dimension figure is what ov1's 18.3% corresponds to, and **it is not reachable**. It is a
static per-context bound over a 4,096-symbol alphabet estimated from 600 samples per dimension.
The model costs more than the structure it buys. This is not an inference — it is measured, and
the sign is emphatic: the coder built to chase that bound lands at **11,419 B**, which is
**+1,590 B worse than Rice** and **3,451 B above its own bound**.

**A per-context entropy bound that omits model cost is not merely unreached here; it is
anti-predictive.** Rice wins because a one-parameter-per-dimension model is the right size for
600 samples.

## §3 The race — 16 coders, real bytes, decode-identity enforced

Every row round-trips to the exact coded-symbol array and through it to the exact shipped lattice.
Byte counts are real encoder output plus any side table the decoder needs.

| coder | total B | ΔB |
|---|---:|---:|
| SHIPPED Rice, segments=1 | 9,829 | — |
| **adaptive-ctx Rice (CABAC prefix, cap=8/16/24)** | **9,811** | **−18** |
| Rice segments=2 / 4 / 8 (adaptive k per stripe) | 9,839 / 9,857 / 9,891 | +10 / +28 / +62 |
| adaptive-ctx Rice, k−1 / k−2 | 9,854 / 10,018 | +25 / +189 |
| adaptive Golomb, JPEG-LS (reset 32/64/128/512) | 10,435–10,449 | +606 … +620 |
| arithmetic, order-0 global | 10,714 | +885 |
| arithmetic, ctx = coeff × prev-magnitude | 11,243 | +1,414 |
| arithmetic, ctx = coeff index (12 models) | 11,419 | +1,590 |

Two structural notes. Per-stripe adaptive `k` **is already decodable** — `_rice_decode` accepts
`ks.shape[1]` segments — but CAP1 hard-codes a 12-byte `ks` table, and the 12 B per extra segment
costs more than the stripe adaptation saves. And the zero-side-info parametric alternative
(JPEG-LS adaptive Golomb) loses by 606 B: the source is stationary enough per dimension that
re-deriving `k` per symbol is strictly worse than transmitting it once.

**Best: −18 B, ΔS −1.198546e-05.** It passes the −3.5e-6 admission bar 3.4× over, and it needs a
CABAC decoder in the receiver. Recommendation in §5.

## §4 The fruit sweep — nothing transfers

| item | claimed | base it was measured on | verdict on jg5 |
|---|---|---|---|
| **qs2** micro-edit | −4.375e-6 @ +34 B | ⚠ cp135, 186,252 B | **SPENT** — consumed into mc35→mc36 Variant C, fired on T4, promoted at −2.068e-5. Falsified-premise key `qs2_re1_bank_union_is_held_and_unfired_20260817` already records that the "bank" does not exist. |
| **re1** | −1.207e-6 @ 0 B | ⚠ cp135 | **SPENT** — same consumption, pairs 96 and 7. Its probability object no longer exists on this HPAC model. |
| **ma1** | −105 B | ck1/ck2 | **ALREADY IN jg5** via to1 (twelfth move). jg5's pre-edit tail carries 109,696 B of RC64 — ma1's own number. Ledger says plainly: do not re-mine. rc64 cure landed (`a6e07d42df`); task #1131 does not exist in the repo ledger. |
| **fx2 D1 remainder** | −151 B (D1) | fx1, 180,601 B | **D1 ALREADY IN jg5.** The remainder is E1/E6/E3/E5 (−238/−211/−196/−104 B), refused on decode margin. jg5 is *already* decode-REFUSED at 1,419.9 s; E1 adds +89 s. Strictly worse on the binding constraint. |
| **mz2** q3/q4, FiLM sparsity | −823 B, −130…−2,051 B | ⚠ e480b, 183,502 B | **FIRED AND REFUSED** (+0.0467 / +0.0443 / +0.0414). Family CLOSED with a measured dose-response. Rate credit realised exactly; the pose/seg leg buried it. |
| **rr5 / CPR1 rider** | −1.85e-4 | ⚠ to1, 176,420 B | **DECLINED for this packet** by pq3, and the number is wrong: measured −1.2185e-4 (−183 B), 66% of what the chain budgeted. Not proven across a carrier re-solve — and jg5 *is* a carrier re-solve. |
| **ck2 −657 B + to1 −105 B** | −5.074e-4 | ck1/ck2 | **ALREADY IN jg5's ancestry.** |
| **ec2** sparse-event HPAC | 413 B payload | ⚠ pre-cp135 | READY_TO_FIRE, never fired; base predates cp135, no jg5-compatible container. |
| **hy1** capstone hybrid | +11 B (wrong sign) | ⚠ cp135 | PARKED; needs a jg5 rebuild before it is a candidate. |

Nine items, zero admissible. The container axis on this lineage is recorded as exhausted, and this
sweep independently agrees.

## §5 The two rung-4 doors

`experiments/ddm_dx1_carrier_resolve_price.py`, receipt `retained/DX1_CARRIER_RESOLVE_PRICE.json`.

**Door 2 — the carrier re-code byte budget — DISCHARGED.** pq6 records it as unmeasured across 600
pairs. It is measurable exactly, because jg5's build already ran a full carrier re-solve and both
bodies are retained. Both pass the bit-exact control:

| | archive | carrier blob | Rice payload |
|---|---:|---:|---:|
| body before re-solve (`body_subset455.zip`) | 180,580 | 22,241 | 9,774 |
| jg5 final, 454 pairs re-solved | 180,625 | 22,296 | 9,829 |
| **delta** | **+45** | **+55** | **+55** |

**0.1211 carrier B per resolved pair → 72.7 B projected at 600 pairs, against a 4,873 B budget.
Clears by 67.0×.** The carrier really does absorb a pose re-aim for almost nothing, and that is now
measured at n=454 on the shipping body rather than extrapolated from n=3.

**Door 1 — the pose residual — DOES NOT CLEAR, and the n=3 number is why.** jg5's own build is a
454-pair instance of the same mechanism: token edits damage pose, the carrier re-solves against the
new frame 1.

| quantity | value |
|---|---:|
| pose damage with the stale carrier | 2.317861e-03 |
| residual after the 454-pair re-solve | 6.365768e-06 |
| **cancellation** | **99.725360%** |
| residual fraction | 2.746398e-03 |
| the n=3 claim (99.9874%) as a residual fraction | 1.260000e-04 |
| **optimism factor of the n=3 figure** | **21.80×** |
| rc4's fraction door, 99.807% | **FAILS** at n=454 |
| rc4's absolute door, 6.431e-6 | passes (6.3658e-6) — see caveat |

pq6 suspected the 3-pair aggregate was the prefix-bias trap. It is, and the size is now known:
**21.80× in residual.** The absolute door is passed, but that bar was calibrated to rc4's drop
amplitude on the hv1 body (182,759 B); jg5's perturbation is a token *edit* at a different
amplitude on a different body. Transferring it is exactly the cross-regime constant transfer this
campaign has been burned by five times. **I do not transfer it, and rung-4 should not be folded on
this evidence.**

This also means the demanded n≥60 seeded-random re-measure has, in effect, already been answered at
n=454 — for the *edit* perturbation. A drop-specific leg would still need the drop deltas rebuilt on
jg5's token field, which do not exist (rc4's are hv1's), and a token drop is a receiver change on
this receiver. That is a build, not a re-measure.

## §6 What this arm seals

**Nothing.** No admissible fruit, and the one measured win needs a receiver revision:

* The −18 B CABAC re-code is real, decode-identical, and clears the admission bar 3.4×. It is
  **0.018% of the archive** and requires a new decoder in the runtime tree.
* jg5 is already **decode-REFUSED** (1,419.9 s vs CI `[822,1302]`), so no candidate on this body
  ships until the rr2 native port lands — and that port rebuilds the receiver anyway.

**Recommendation: fold the −18 B into the rr2 receiver revision, when the tree is being rebuilt
for reasons that already justify a seal chain. Do not spend a seal chain on 18 bytes now.**

## §7 Retained payloads

`/Volumes/VertigoDataTier/pact/ddm_dx1/retained/` — 13 files: `DX1_RECODE_RACE.json`,
`DX1_CARRIER_RESOLVE_PRICE.json`, `dx1_coded_symbols_U.int32.npy` (the exact coded-symbol array,
sha `0bfe31cf9586104f4308329fec8f76f748c56441ac5bd85b824dfcca3434db50`), and **every re-coded
payload** with its sha256 recorded in the race receipt — not merely their lengths.

## §8 Corrections owed to the record

1. **ov1 §5's ΔS ≤ −0.001190 ceiling is withdrawn.** Measured ceiling on jg5: **−1.199e-5**, 99.0%
   smaller. Wrong object, plus a per-context bound that ignores model cost. ov1's F1 fire order is
   **FOLDED**, not fired — it was honest about being a bound; the bound was simply far off.
2. **ov1's "18.3% above its order-0 bound" should read 3.11%** against the reachable (global)
   bound on the true coded symbols.
3. **The qs2+re1 "banked pool" does not exist** and has not since 2026-08-17. This charter asked
   me to re-check it on jg5; the correct answer is that it was consumed, not that it failed to
   transfer. A falsified-premise key already records this — it should be consulted at charter time,
   which is where this arm's own recall caught it.
4. **pq6 §B item 2's "carrier re-coding cost across 600 pairs is unmeasured" is closed:** 72.7 B at
   600 pairs, 67.0× inside budget.
5. **rung-4's reopening should be re-scoped.** The negative genuinely no longer binds on bytes. It
   binds on pose, and the reopening's headline cancellation is 21.80× optimistic at scale.
