# ddm_rc4 — rung 4 (token drop × Schur compensation) priced on the LIVE hv1 vehicle

**Date:** 2026-08-16
**Base:** hv1 ep0634, `S = 0.15959729295498598` @ 182,759 B `[contest-CUDA T4, n600]`,
archive sha256 `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`
(receipt `experiments/results/ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/MODAL_REMOTE_RESULT.json`).
**Score claim:** false. No Modal dispatch, no new archive, no exact eval. Every distortion number
below is `[macOS-CPU advisory]` on a stratified-random pair sample; every rate number is exact.
**Store:** `/Volumes/APDataStore/pact/ddm_rc4_rung4_token_drop_20260816/`

---

## VERDICT

**Rung 4 is REFUSED as an uncompensated drop — on the POSE leg, by 517×.** The rate leg is
exact and favourable and the seg leg is genuinely net-negative (best rung −3.243e-3 S, 34% of
the gap), but the pose leg costs **+0.17432 S**, which is **53.8× the entire rate+seg gain**
and **517× the pose headroom the gain buys**. Measured `delta_d_pose` = 3.3279e-3 against an
allowed 6.431e-6.

That pose figure is scored against the **authority-lineage** GT per ddm_pi2's fix
(`gt_cache_dali.pt["pose"]`, sha `a91d9825…`, tracks contest authority at 1.00081×). Scoring
the same retained vectors against the older PyAV-lineage GT gives `delta_d_pose` 3.4366e-3 →
`dS_pose` +0.17727, i.e. **1.7% different**. The verdict does not depend on the GT-lineage
question at all: the drop is measured as a *paired differential*, so a GT-lineage offset
cancels in the difference. Both numbers are recorded in `POSE_RESCORED_DALI.json`.

`verdict_scope: FORMULATION` — *uncompensated* confidence-threshold token drop on the hv1
vehicle. This is NOT a family kill. The charter's own composition — qs5's **in-compile frame-0
Schur compensation** — is untested at this amplitude and is the one door left. It must cancel
**99.807%** of the pose perturbation to make the rung net-negative. qs5 achieved *full*
cancellation at micro scale, and the structural case is strong (the 12-coefficient carrier
already drives `d_pose` to 6.88e-6 from scratch, so it demonstrably has the authority to
re-hit the pose target for a perturbed frame_1 — 6 pose equations, 12 free coefficients per
pair). What is unmeasured is its reach at this amplitude and the carrier's re-coding cost
across all 600 pairs.

Three things are now settled and should not be re-derived:

1. **The token coder is exactly optimal** — measured at 1.00000 of the model's cross-entropy,
   with a total per-symbol coding tax of 0.32 bits over the whole 117,964,800-symbol stream.
   (Independently reproduced by sister arm ddm_dc1 to the byte: 112,109.578 B vs my 112,109.6 B,
   from a different instrument.)
2. **No token-drop scheme can buy rate at zero distortion**, by the source coding theorem —
   see the theorem section. Every byte must be purchased with label flips.
3. **The seg amplification is A ≈ 0.79, flat in the threshold** (0.785 / 0.798 / 0.807 across
   `p_max` 0.969 → 0.997). Even with pose entirely free, closing the gap on this rung alone
   needs A ≤ 0.5084, so rung 4 was never going to close it alone.

Pointer UNMOVED: `S = 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]`. No fire-order
emitted — there is no candidate archive, and the uncompensated candidate is refused on
measurement.

---

## Stage 0 — the census, re-derived on hv1's actual bytes

Independent parse (`rc4_census.py`), driven from the SHIPPED receiver's own constants in
`runtime/residual_archive.py`, never from an inherited table. `CENSUS_hv1.json`.

| section | bytes | share | rate term | shipped bits/byte | H0 slack |
|---|---:|---:|---:|---:|---:|
| RX1M header | 14 | 0.008% | 0.0000093 | 3.5216 | 7.8 B |
| HPAC stream | 13,515 | 7.395% | 0.0089991 | 7.9844 | 26.3 B |
| semantic stream | 34,763 | 19.021% | 0.0231473 | 7.9951 | 21.1 B |
| carrier stream | 22,161 | 12.126% | 0.0147561 | 7.9927 | 20.2 B |
| RCF1 residual table | 96 | 0.053% | 0.0000639 | 6.1368 | 22.4 B |
| **RC64 token stream** | **112,110** | **61.343%** | **0.0746494** | 7.9985 | 21.4 B |
| ZIP framing | 100 | 0.055% | — | — | — |
| **total** | **182,759** | 100% | **0.12169171641365491** | | |

Unaccounted bytes: **0**. Recomposed rate term matches the T4 receipt's 0.1216917.

**Cross-check against sister arm ra2** (MAIN relay): ra2 reported token+residual 112,206 /
semantic 34,763 / carrier 22,161 / HPAC 13,515 / header 14 / framing 100. My split is
token 112,110 + residual 96 = **112,206**. Every row agrees exactly. Two independent parses,
zero disagreement.

**#996 reproduced on hv1, and its weakness named.** Every compressed section sits within
20–27 B of its own order-0 byte entropy, so at the byte level the coder axis is closed here
too. That is a WEAK bound — order-0 on already-compressed bytes is near-vacuous by
construction. The strong statement comes from the replay below, and it is stronger than #996:
the token coder is measured at **1.00000 of the model's cross-entropy**.

**One structural fact the census settles for every downstream rate arm:** the ZIP member `p`
is **STORED**, not deflated (182,759 − 182,659 = 100 B of framing: 30+1 local, 46+1 central,
22 EOCD). The RC64 token stream is appended raw after the model sections. Therefore **a byte
removed from the token stream is a byte removed from `archive.zip`, 1:1**. No
compression-of-compressed uncertainty stands between a token-rate result and the rate term.

---

## The mechanism, read at source (not inherited)

`cpr1/inflate.py:289–345`, the shipping renderer:

```
output[2*i + 1] = master   <- rendered from the TOKEN field   (frame_1)
output[2*i]     = slave    <- rendered from the CARRIER only  (frame_0)
```

Consequences that govern this rung:

1. **SegNet reads `x[:, -1]`** (`upstream/modules.py:105`) = frame_1 = the token-rendered
   master. The token field is **100% of the seg surface**.
2. **PoseNet reads both frames.** A token change reaches pose through the frame_1 half only;
   frame_0 is carrier-painted and untouched.
3. **The frame-0 carrier is exactly the qs5 in-compile Schur compensation channel**
   (`compensation_overlay` adjusts carrier coefficients, `f26_inflate.py:310–330`). The
   charter's composition is structurally correct: the drop damages frame_1, and the
   compensation channel that qs5 PROVED can absorb the pose leg acts on frame_0.
4. **The drop threshold ships for zero bytes.** `RX1_MODEL_HEADER` is `<4sBBBBHHH` and its
   `reserved` byte is currently required to be 0. A version bump can carry the threshold
   exponent in that existing byte. The receiver change is a skip-decode branch in
   `decode_production_tokens`, and `inflate.py` is unsized.

---

## The theorem before the measurement — why zero-distortion drop is impossible

The decoder computes the full probability table *before* decoding each symbol
(`residual_archive.py:595–602`), so a rule "drop iff `p_max >= tau`" is free to signal. But a
drop rule that is free to signal necessarily drops the positions where the model is *wrong*
as well as those where it is right, and each of those is a label flip by construction.

The zero-distortion alternative — drop only the positions the model already gets right — is
not free: the decoder must be told the exception set. Any "drop plus exception list" scheme is
simply an alternative code for the same source, so by the source coding theorem it cannot beat
the code already in use. Measured here: the exception-set identification floor alone is
`log2 C(117,964,800, 223,694) = 2,345,184 bits = 293,148 B`, against a whole token stream of
112,110 B.

**So no token-drop scheme of any kind can buy rate at zero distortion. Rung 4 must purchase
every byte with label flips, and the only open question is the exchange rate.** That question
is answered by measurement below, not by argument.

---

## Stage 1 — the exact rate–distortion structure (`TOKEN_RD_REPLAY.json`)

The shipped decoder was replayed forward with teacher forcing against the RETAINED decoded
token field of the frontier archive
(`.../ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8`,
sha `9ba2e52b…`, bound to archive sha `80d9c8c6…`).

**Positive control — PASSED, and it is a bit-identity, not a similarity.** The replay
recomputed the decoder's own `corrected_quantized_logit_sha256` and
`corrected_cdf_input_sha256` and matched both (`562ac652…`, `dd48843b…`). Every probability
table priced below is byte-identical to the one the shipping RC64 decoder consumed. The
ladder script refuses to run if this control fails.

| quantity | value | label |
|---|---:|---|
| positions | 117,964,800 | 600 × 384 × 512 |
| ideal code bits `Σ −log2 p(x_i)` | 896,877 | MEASURED |
| shipped token bits | 896,880 | MEASURED |
| **coder efficiency** | **1.00000** | DERIVED |
| model top-1 error | 0.0018963 | MEASURED (223,694 disagreements) |
| bits in disagreeing positions | 627,909 = **70.011%** | MEASURED |
| bits in agreeing positions | 268,967 = 33,621 B | MEASURED |
| mean bits per disagreement | 2.807 | MEASURED |

The coder is spending exactly the model's cross-entropy — three bits out of 896,880. RC64's
own frequency floor is `RC64_TOTAL = 2^31` with minimum frequency 1, so the per-symbol coding
tax over the whole stream is **0.32 bits total**. There is no per-symbol overhead pool for a
drop rung to drain. Every one of the 112,110 B is information the model does not already
predict.

### The exchange rates (DERIVED, exact, at this operating point)

| quantity | value |
|---|---:|
| S per archive byte | 6.658589531221714e-7 |
| S per SegNet argmax flip | 8.477105e-7 (100 / 117,964,800) |
| **breakeven** | **1.273108 bytes, or 10.185 bits, saved per net seg flip created** |
| pose marginal | 602.80 S per unit `d_pose` |
| base seg term as flips | 34,930.6 |

### The ladder (`DROP_LADDER.json`)

Drop rule: substitute the model argmax wherever `p_max >= tau`, parameterised by
`u = −log2(1 − tau)`. Rate is exact (it is the code length the stream no longer spends, and
token bytes are archive bytes 1:1). `A` is the seg amplification — net SegNet argmax flips
against GT per token flip — measured in Stage 1b.

| A | best `p_max` | bytes saved | token flips | ΔS | S after | sub-0.15 alone? |
|---:|---:|---:|---:|---:|---:|---|
| 0.30 | 0.727373 | 81,321 | 123,772 | −2.267e-2 | 0.1369261 | YES |
| 0.40 | 0.863687 | 57,873 | 70,294 | −1.470e-2 | 0.1448974 | YES |
| 0.50 | 0.925675 | 42,652 | 43,629 | −9.908e-3 | 0.1496895 | YES |
| **0.5084** | — | — | — | — | **0.150000** | **critical** |
| 0.60 | 0.959474 | 32,959 | 29,658 | −6.861e-3 | 0.1527362 | no |
| 0.72 | 0.979737 | 24,658 | 19,570 | −4.474e-3 | 0.1551230 | no |
| 0.85 | 0.990709 | 18,869 | 13,711 | −2.685e-3 | 0.1569127 | no |
| 1.00 | 0.997238 | 11,901 | 7,791 | −1.320e-3 | 0.1582774 | no |
| 1.50 | 0.999914 | 1,538 | 759 | −5.879e-5 | 0.1595385 | no |
| **2.5078** | — | 0 | 0 | 0 | 0.1595973 | **dead above here** |

Gap to 0.15 from hv1: **0.009597292954985986**.

**Pre-registered falsifier, fixed before Stage 1b ran:**
* `A <= 0.5084` → rung 4 alone crosses sub-0.15;
* `0.5084 < A < 2.5078` → rung 4 is a real but partial supplier;
* `A >= 2.5078` → rung 4 is dead at every threshold and the rate ladder loses its last
  representation-side supplier.

**Scope on the rate leg (honest):** the ladder prices the drop against the BASE decode's
probability tables. A live drop encoder perturbs the context of later positions. At the
thresholds of interest the drop changes 7,791–25,619 of 117,964,800 positions
(0.0066%–0.0217%), so the ladder is a first-order figure; the sign of the second-order term
is not established. Label it DERIVED-first-order, not MEASURED-closed-loop.

**The charter's pure-rate byte equivalence does NOT apply to this rung.** Token drop moves the
decoded frame_1 field, so seg and pose both move; every row above therefore carries the full
three-component arithmetic. (MAIN's correction of 2026-08-16 sharpens this: the bar is
invariant under pure-rate moves *and only those*, with cp135 → MC36 as the worked
counterexample. Future rows should read the bar from
`tac.canonical_equations.sub015_pure_rate_archive_byte_bar_20260816.pure_rate_byte_bar_from_pointer()`
rather than any literal.)

---

## Stage 1b — the seg amplification A, MEASURED (`AMPLIFICATION.json`)

Rendered the SHIPPED semantic renderer on a **stratified-random n=120 pair sample** (seed
20260816, one pair drawn per 5-wide block — never a prefix; prefix bias is a measured law on
this vehicle), pushed both frames through the exact upstream scorer preprocess
(`interpolate` to 384×512 → SegNet → argmax) and counted flips against the retained GT SegNet
argmax field. Every comparison is a **paired differential** on the same pair, so the only
difference between arms is the token field.

| `p_max >=` | token flips | net seg flips | B (benef.) | H (harmful) | W (wrong→wrong) | **A** | bytes / net flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.968750 (u=5.0) | 5,324 | 4,178 | 1,454 | 5,632 | 33 | **0.78475** | 1.4900 |
| 0.992188 (u=7.0) | 2,689 | 2,147 | 939 | 3,086 | 18 | **0.79844** | 1.7564 |
| 0.997238 (u=8.5) | 1,662 | 1,341 | 677 | 2,018 | 7 | **0.80686** | 1.8959 |

**Positive control on the seg instrument.** Base seg flips over the sample: 7,107 for 120
pairs = 59.2/pair → 35,535 at n600, against the contest-CUDA seg term's 34,930.6. My local
advisory instrument reproduces the authority base to **1.7%** (rn1 measured 2.5% at n=96 on an
independent build).

**A defect found in my own instrument during review, and measured to zero.** The amplification
and pose harnesses built their probability tables WITHOUT calling
`optimize_sparse_evaluator`, which the shipping decoder — and my bit-identical replay — always
call. That would make the drop set they applied differ from the one a real decoder would build.
Rather than argue it was harmless, I measured it (`OPTIMIZE_SPARSE_CONTROL.json`,
`experiments/ddm_rc4_optimize_sparse_control.py`): on frames 0, 137 and 411 the argmax field is
**identical**, `p_max` max-absolute-difference is **exactly 0.0**, and the drop sets at both
u=5.0 and u=7.0 are **identical**. It is purely a speed path. The landed instruments call it
anyway, because the decoder does.

**A is flat in the threshold** — 0.785 → 0.807 over a 20× range of `1 − p_max`. Dropping a
token the model is *more* confident about does not make the flip cheaper in score. So there is
no "safe" confident tail to harvest, and the 1-D ladder's shape is governed by rate alone.

**The seg market is thin and two-sided at ≈1.5 B per flip.** Rung 4 *sells* seg flips at
1.49–1.90 B/flip against a 1.273 B/flip bar (a 17–49% margin). qs4/qs5 *bought* flips via
micro-edit at ≈1.53 B/flip and were REFUSED for it. The vehicle sits at a local RD optimum
where the marginal price of a seg flip is ≈1.5 B in both directions. That is why every
micro-edit row lands at ±1e-6…1e-5: the margin is thin and the volume was 17 flips. **Rung 4's
only real advantage was volume** — the same thin margin applied to 12,902 flips instead of 17,
which is exactly how it reaches −3.24e-3.

---

## Stage 1c — the pose leg, MEASURED (`POSE_LEG.json`)

Same construction at `p_max >= 0.9921875`, stratified-random **n=48**, frame_0 rebuilt through
the exact shipping carrier path (including the compensation overlay) so that frame_0 is
byte-identical between arms and the differential isolates frame_1.

Priced per the binding ddm_pi2 rule — **absolute delta only, converted at the AUTHORITY
baseline**, never a ratio and never rescaled:

```
dS_pose = sqrt(10*(6.88e-06 + delta_d_pose_abs)) - sqrt(10*6.88e-06)
```

**The GT-lineage fix, applied.** ddm_pi2's final verdict is that the advisory pose gap was our
own tooling reading two ground truths — the seg half off a DALI-lineage cache, the pose half
decoding GT fresh with PyAV. The FIX is to score pose against
`/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt["pose"]`
(sha `a91d9825…`, 1.00081× of authority). **Because this arm retained every per-pair pose
vector, applying the fix cost a re-score, not a re-render** — `POSE_RESCORED_DALI.json`,
`experiments/ddm_rc4_pose_rescore_dali.py`. That is what ALWAYS-KEEP-THE-PAYLOAD buys.

| quantity | **FIX: authority-lineage GT** | prior PyAV-lineage GT |
|---|---:|---:|
| base `d_pose` | **3.33887e-6** (0.49× authority) | 1.38622e-4 (20.15× authority) |
| dropped `d_pose` | 3.33124e-3 | 3.57523e-3 |
| **`delta_d_pose` absolute** | **3.327899e-3** | 3.436612e-3 |
| **`dS_pose` at the authority baseline** | **+0.174319** | +0.177272 |
| pose headroom the rate+seg gain buys (exact sqrt inverse) | 6.431e-6 | 6.431e-6 |
| **over budget by** | **517.5×** | 534.4× |
| pose cost ÷ rate+seg gain | **53.8×** | 54.7× |

The fix removes the floor entirely — base `d_pose` drops from 20.15× authority to 0.49×
(the residual 0.49× is the n=48 population effect on a heavily skewed pose distribution, not
an instrument gap) — and moves `dS_pose` by **1.7%**. The verdict is invariant to it.

**GT-lineage control on the SEG half.** MAIN's correction warns that a hand PyAV GT decode
inflates a seg number by 1.4425×. My seg half used a *cached* `gt_argmax.npy`, and I verified
its lineage directly rather than assuming: it differs from the DALI authority cache's `seg`
field at **3 sites out of 117,964,800** (2.54e-6 %) — exactly the 3-site agreement MAIN quoted
for the authority cache. My seg GT *is* the authority lineage. No seg number here is inflated.

**Determinism repeat.** The n=48 pose run was executed twice (the second time to regenerate a
retained payload a smoke test of mine had overwritten — my error, caught and repaired). Both
runs returned `delta_d_pose_absolute = 0.003436611547583462` and
`delta_S_pose = 0.1772718953507871`, **bit-identical**, and the regenerated
`retained/pose_leg/pose_u7.0.json` reproduces its original sha256 `c8d44ba626091576` exactly.
The n=120 amplification sample was likewise regenerated from its recorded seed and reproduced
sha `9b11dd644d344b19`. No payload was lost.

**Positive control on the pose instrument.** ddm_pi2 predicts the local advisory base to be
authority + floor = 6.88e-6 + 1.4061e-4 = **1.47490e-4**. I measured **1.38622e-4** from a
completely independent code path (my own carrier/renderer reconstruction, my own PoseNet call,
mt1's retained GT pose). Error **−6.0%** at n=48. That is an independent confirmation of pi2's
additive-floor attribution, and it is the reason this pose number is quotable at all.

**Why the pose leg is so much larger than intuition.** A token flip is not a 1-LSB dither: it
changes a pixel's class, which the renderer smears into a ~21×21 EVAL-grid RGB neighbourhood
(4 blocks at dilations 1,1,2,4 plus a 3×3 head). At u=7.0 that is ~21 flips per pair — roughly
9,000 strongly-changed pixels — against rn1's ±1-LSB dither over 196,608 weakly-changed ones.
rn1's dither measured `delta_d_pose` 1.216e-4; this measures 3.437e-3, 28× larger, which is the
right order for the perturbation-energy ratio.

---

## CONCLUSION

The rate ladder's last representation-side supplier on the 61.3% token section is real on two
legs and dies on the third:

| leg | value at the optimum (`p_max >= 0.9921875`, 17,985 B, 12,902 token flips) | axis |
|---|---:|---|
| rate | **−1.19754e-2** | EXACT (token bytes are archive bytes 1:1; ZIP member is STORED) |
| seg | **+8.7325e-3** | `[macOS-CPU advisory]` n=120, A=0.79844 |
| rate + seg | **−3.2430e-3** → S 0.1563543 | |
| pose | **+0.174319** | `[macOS-CPU advisory]` n=48, absolute delta at authority baseline, authority-lineage GT |
| **all three** | **+0.171076** → S 0.330673 | REFUSED |

Rung 4 has no owner because it does not deserve one in its uncompensated form. It deserves
exactly one more measurement, and that measurement is not another drop sweep — it is whether
the frame-0 Schur compensation can absorb a per-pair `delta_d_pose` of 3.4e-3.

**What this closes for the campaign.** The rate ladder that rfo2 opened is now fully
adjudicated on the hv1 base: rung 1 (mixed precision) ran, rung 2 (carrier rank) is refused,
rung 3 (width distillation) was refused at ep60, and rung 4 is refused uncompensated. The
coder axis is closed to 7.8 B (dc1). **The representation side of the rate ladder has no
measured supplier left that does not first go through a pose compensator.** That is the
re-routing this arm produces: pose compensation is no longer an optional composition step for
frame_1 levers — it is the gate every remaining frame_1 lever must pass.

---

## NEXT_IF_RESUMED

Bars must be read from
`tac.canonical_equations.sub015_pure_rate_archive_byte_bar_20260816.pure_rate_byte_bar_from_pointer()`,
never from a literal (MAIN 2026-08-16; the 186,269 B literal went stale under six arms). Note
that this rung is NOT pure-rate, so that bar bounds only its rate leg.

| # | row | owner | fire condition | READY? |
|---|---|---|---|---|
| 1 | **Schur-compensated drop reach test.** At `p_max >= 0.9921875`, re-solve the 12 frame-0 carrier coefficients per pair against the 6 PoseNet equations *in-compile* (never carried), and measure the residual `delta_d_pose`. PASS iff it falls below 6.431e-6 (99.807% cancellation) **and** the re-coded carrier section grows by less than 17,985 B − 12,902·0.79844·1.2731 = 4,873 B. | qs5 successor / pose owner | immediate; $0 local, reuses `experiments/ddm_rc4_pose_leg.py` for the verdict half | code READY, solver owed |
| 2 | **The compensator is now a GATE, not a composition step.** Every remaining frame_1 lever (semantic width, token representation, renderer edits) inherits this 517× pose exposure. Before any of them spends a measurement, the compensator's reach must be characterised once. | pose owner | fires with row 1 | — |
| 3 | **Guarded 2-D drop — DEMOTED by this result.** `experiments/ddm_rc4_guarded_drop.py` tests whether a decoder-free boundary-bucket guard lowers A below 0.5084. It is built and unfired. It only improves the SEG leg, which is not what binds; fire it only if row 1 passes. | ddm_rc4 successor | row 1 PASSES | READY, unfired |
| 4 | **HPAC model-size sweep — the genuinely unowned representation lever.** The 13,515 B model buys 51,484 B of token rate (dc1) and pays for itself 3.8×, but `d(token bytes)/d(model bytes)` has never been measured: the checkpoint selector optimises the joint at *fixed architecture*, never across `HPAC_CHANNELS`/`HPAC_PATCH`. This is pure-rate at fixed decoded field, so it carries none of rung 4's pose exposure. | rate owner | needs a burn slot | not ready (training) |
| 5 | **Closed-loop rate correction.** The ladder is DERIVED-first-order; a live drop encoder perturbs 0.007–0.022% of positions' contexts. Only worth resolving if row 1 passes. | ddm_rc4 successor | row 1 PASSES | — |

**Retracted / not claimed:** no fire-order, no candidate archive, no exact row. The pointer did
not move and this arm did not move it.

---

## Artifacts (ALWAYS KEEP THE PAYLOAD)

Store root `/Volumes/APDataStore/pact/ddm_rc4_rung4_token_drop_20260816/`. Retained payload
total 213,145 B. sha256 shown as first 16 hex.

| result | bytes | sha256 |
|---|---:|---|
| `CENSUS_hv1.json` | 4,818 | `b55369f72720d098` |
| `TOKEN_RD_REPLAY.json` | 2,370 | `5db3bfb9f19fe636` |
| `DROP_LADDER.json` | 167,199 | `196a67996a2add87` |
| `AMPLIFICATION.json` | 2,164 | `77518fb7ac584524` |
| `POSE_LEG.json` | 1,808 | `d33694523d59c79e` |
| `JOINT_VERDICT.json` | 3,642 | `0bbc8b49f69e6c7f` |

| retained payload | bytes | sha256 |
|---|---:|---|
| `retained/census_sections/token_stream.bin` | 112,110 | `73a878891a31c366` |
| `retained/census_sections/semantic_stream.bin` | 34,763 | `4099eab6fc18af5b` |
| `retained/census_sections/carrier_stream.bin` | 22,161 | `fd14aabcb9daa5f1` |
| `retained/census_sections/hpac_stream.bin` | 13,515 | `602115b323b0e403` |
| `retained/census_sections/residual_table.bin` | 96 | `8ab2fe748ab7d69d` |
| `retained/census_sections/rx1_header.bin` | 14 | `43fb77d81b2c45f0` |
| `retained/token_rd/hist_{n,bits,n_disagree,bits_disagree}.npy` | 3,216 ea | `04d2cf3c…`, `20b27805…`, `8310af81…`, `6ed9e135…` |
| `retained/token_rd/cost_bins_disagree.npy` | 2,184 | `415430599e61a653` |
| `retained/pose_leg/pose_u7.0.json` | 14,350 | `c8d44ba626091576` |
| `retained/amplification/sample_pairs.npy` | 1,088 | `9b11dd644d344b19` |

**Third positive control (parse):** the retained `token_stream.bin` hashes to
`73a878891a31c3668a0403f842740f21598999fee5c8afd8982fb2ca31125829`, byte-identical to the
`token_stream_sha256` in the shipping decoder's own checkpoint receipt. The census parse is
the decoder's parse.

**Landed instruments** (`experiments/ddm_rc4_*.py`). The store copies are the exact producers
of the receipts above; the landed copies differ only by lint-clean no-ops (an unused `noqa`, an
unused local, and a `next(...)` idiom), so their shas differ:

| instrument | store sha256 | landed sha256 |
|---|---|---|
| `token_rd_replay.py` | `730154c6ac543cc4` | `0c7fc3d2927b2c4e` |
| `drop_ladder.py` | `e46a16ba94c092f8` | `9488a7290c3fdca2` |
| `amplification.py` | `18a62512065463bb` | `620017cfb625f1ed` |
| `pose_leg.py` | `a285e1050521088f` | `659b2449ca888de5` |
| `guarded_drop.py` (unfired) | `8d68f9d602f87f53` | `50717cedc0195cbc` |
| `joint_verdict.py` | `7256f34ba89b33fb` | `7256f34ba89b33fb` |

`ddm_rc4_amplification.py` and `ddm_rc4_pose_leg.py` are the reusable pieces: any future
frame_1 lever on this vehicle needs exactly this paired-differential seg instrument and this
pose instrument, and the pose one has the ddm_pi2 additive-floor rule enforced in code rather
than left to the reader.
