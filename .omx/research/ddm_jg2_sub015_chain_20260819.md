# ddm_jg2 — the sub-0.15 chain: replace the modelled rate leg with a measurement

- **arm** `ddm_jg2` (task #1139 — the successor to `ddm_jg1`'s joint solve)
- **date** 2026-08-19
- **axis** every number is `[macOS-CPU advisory]` unless it carries an explicit DALI-lineage
  tag. `score_claim=false` · `promotable=false`. This arm fires **no Modal job**; MAIN owns
  the T4 slot.
- **cost** $0.
- **store** `/Volumes/APDataStore/pact/ddm_jg2/`
- **status** IN PROGRESS — written incrementally, committed at every stage boundary.
  **Pointer UNMOVED** at contest-CUDA `0.15652626435208142` until a T4 row says otherwise.

## THE BASE (re-read from `.omx/state/canonical_frontier_pointer.json` at arm start)

| term | value | S contribution |
|---|---:|---:|
| `d_seg` | 0.00030309 | 0.030309 |
| `d_pose` | 7.649246787e-06 | 0.008746 |
| archive | 176,420 B | 0.117471 |
| **S** | | **0.15652626435208142** |

`archive.zip` sha `7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f`.
**Gap to sub-0.15 = 0.006526.**

## THE INHERITED PROJECTION, AND THE ONE LEG THAT IS NOT MEASURED

`ddm_jg1` (memo `.omx/research/ddm_jg1_joint_solve_20260819.md`) established, at $0:

1. a **validated** local contest-axis seg instrument (`0.99995x` of the T4 seg leg);
2. the **move-class law** — single-cell token coordinate moves repair ~1.5 argmax cells per
   changed token and compose within a sparse pass; block/dilation moves realize worse;
3. the **hard negative and its reversal** — token seg edits destroy pose (`x387`), but
   re-running the carrier's own coordinate descent against the edited frame recovers
   `d_pose` to `1.073x` of original at ~0 bytes. **The actuators compose.**

Its rate leg is **modelled, not measured**: `+4.718 bits` per changed token, computed from
the **hm1/182,759 B body's** probability model, then transferred to the to1/up3 body we
actually ship. jg1 names three reasons that constant is suspect, and **all three point the
same way — the real price is likely HIGHER**:

| # | risk (jg1 S1d caveats 3-5) | direction |
|---|---|---|
| 3 | cross-body transfer: to1's model is **sharper** (0.007446 vs 0.007603 bits/token) | costs MORE |
| 4 | context coupling: the HPAC model decodes in 190 groups, feeding decoded tokens forward | costs MORE |
| 5 | the table correction is omitted from the marginal number | unknown sign |

Two extrapolations exist and they disagree, which is itself information:

| source | repaired cells | changed tokens | net S |
|---|---:|---:|---:|
| jg1 §S1e "honest extrapolation" | ~11,400 | ~7,800 | **-0.0066** |
| jg1 §S2 first-pass scale-up (the charter's headline) | ~18,000 | ~11,600 | **-0.0104** |

The gap is 0.006526. **The honest one barely clears it; the headline clears it with room.**
Both rest on the same modelled constant. That is why S1 runs before anything else.

## STAGE LEDGER

| stage | what it settles | status |
|---|---|---|
| S1 | REAL `ΔB` for jg1's retained 3-pair edit set, through a real encoder on the to1 body | IN PROGRESS |
| S2 | n600 joint solve, seeded-random pair order, rate-aware acceptance | GATED on S1 |
| S3 | byte-close + identity control + determinism + seal | GATED on S2 |

**HONESTY RAIL (charter, binding).** `-0.0104 S` is a 3-pair extrapolation. Realized-vs-
projected is printed at every scale rung. A smaller honest win still seals and fires; an
honest refusal with the measured curve is a first-class landing.

STORES CONSULTED: `.omx/state/canonical_frontier_pointer.json` (re-read at arm start) ·
`.omx/research/ddm_jg1_joint_solve_20260819.md` (full) ·
`/Volumes/APDataStore/pact/ddm_jg1/JG1_RETENTION_MANIFEST.json` + all 12 retained files ·
memory `pose_gap_was_gt_cache_lineage_not_cuda_20260819` ·
memory `the_denominator_and_the_falsifier_can_both_be_vacuous_20260816` ·
memory `concavity_helps_when_you_pay_the_axis_upward_20260818`.

---

## S1 — THE REAL RATE

### S1a — the coordinate correction, measured before anything was encoded

Two things jg1 recorded needed fixing, and both are structural rather than cosmetic:

1. **The tail is not all coder output.** `read_residual_archive` (`runtime/residual_archive.py:478-494`)
   splits it: **109,792 B tail = 96 B compact fixed residual table + 109,696 B RC64
   stream.** A re-encoder that compares its output against the whole tail is 96 B off by
   construction. This module carries the 96 B prefix through untouched and byte-checks
   only the stream.
2. **The pointer body is `ddm_up3/candidate_runtime/`, not the to1 tree.** The to1 tree's
   `inflate.py:19` pins `ARCHIVE_SHA256 = "50e56145..."` and would refuse the pointer
   archive. MEASURED here: the two archives' `runtime/` and `cpr1/` trees are identical,
   and section-wise `hpac`, `semantic` and `tail` are **byte-identical** — **only the
   carrier differs** (`f59210d7...` vs `4ef50093...`). So jg1's tail work transfers, but
   the splice target is the pointer.

### S1b — THE FINDING THAT DECIDES THE METHOD: there is no per-token price

jg1's `+4.718 bits/token` is a **per-symbol marginal**: the cost of flipping one token
holding every other probability fixed. Reading the shipped decoder
(`decode_production_tokens`, `:600-649`) shows that quantity does not exist for this
coder. **Four independent feedback paths make a token's price depend on other tokens'
VALUES:**

| # | path | file:line | reach |
|---|---|---|---|
| 1 | `sparse.selected_logits(current, context, group)` — `current` is the partially-decoded frame, one-hot'd into the conv | `residual_archive.py:617`, `cpr1/hpac_integer_sparse.py:161-170` | the 189 later groups of the SAME frame |
| 2 | `context = model.prepare_frame_context(index, previous)` — `conv_past` + SPM on the previous decoded frame | `:603`, `cpr1/hpac_integer.py:330-362` | all of frame n+1 |
| 3 | `boundary = _boundary_buckets(previous_cpu)` — the fixed-table row index is a distance-to-class-edge map of the previous frame | `:605-608`, `:515-534`, `:621` | all of frame n+1 |
| 4 | `FreeCorrector` Krichevsky-Trofimov counters, updated only from decoded symbols and **never reset per frame** | `free_corrector.py:290-308`, `173-175` | **the entire remaining stream** |

Path 4 alone makes the blast radius of one changed token **global and unbounded** — one
edit in pair 283 perturbs the probability of every symbol through pair 599. There is no
local window to price, and `ddm_hm1`'s retained `base_logits_int16_n600.i16` — which
`ddm_rr2`'s encoder memmaps — becomes **wrong** the moment a token changes. `ddm_rr2`
states its own precondition at `:40-43`: its logits are valid *"because the decoded field
is unchanged by construction."* A jg2 token edit is exactly what breaks that.

**So the modelled 4.718 was never going to be checkable by a better model. It is
replaceable only by encoding the whole 600-frame stream along the new trajectory and
stat'ing `archive.zip`.** That is what `experiments/ddm_jg2_tail_reencode.py` does: it is
`decode_production_tokens` line for line with the decode call replaced by an encode of the
known symbol, importing the model, group plan, boundary map, fixed table, corrector and
probability quantization from the shipped runtime rather than reimplementing any of them.

### S1c — the encoder is the exact inverse (smoke, n=2 frames)

The RC64 encoder half is not in the shipped tree (`runtime/entropy/rc64_backend.c`, sha
`05839d14...`, is **decoder-only**). It comes from the `ddm_rr2` lineage source pinned at
sha `5c75e2c7...` plus `route_b_rc64.RC64_CHECKPOINT_EXTENSION`. That pairing is not
assumed to invert this decoder — it is **tested**:

| quantity | value |
|---|---:|
| frames encoded | 2 |
| emitted bytes | 555 |
| **prefix bytes agreeing with the shipped stream** | **554** |
| ideal code length from the probability rows | 554.78 B |
| wall clock | 2.88 s (1.44 s/frame -> ~14.4 min for n600) |

The single trailing byte is the coder's end-of-stream flush at frame 2, which the shipped
stream does not have there. **554 of 555 bytes agree**, and the emitted length sits 0.22 B
above the ideal code length — the coder tax, measured rather than assumed. The full
600-frame control (which must be byte-identical, not prefix-identical) is the binding
proof and is running.

### S1d — PRE-REGISTERED PREDICTION, written before the n600 encode returned

Pre-registering because the honest reading of the mechanism says the modelled number
should be **too low**, and I would rather be on record than fit a story afterwards.

**The edit set.** jg1's retained 3-pair payload changes **58 tokens** (pair 283: 20,
pair 468: 19, pair 513: 19). At jg1's modelled `+4.718 bits/token` that is
**+273.6 bits = +34.2 B -> +0.0000228 S**.

**Why I expect the real price to be higher, and it is not only the three caveats jg1
listed.** jg1's own S0 measured that the stored tokens are **99.9985% identical to the
DALI GT argmax** — the shipped label field is essentially the true segmentation. The seg
actuator is therefore **PRE-DISTORTION**: it deliberately moves tokens *away* from the
natural field so that after `render -> re-segment` the argmax lands closer to GT. But the
IHS1 model is a prior fitted to the natural field. **So every edit moves a token toward
what the model finds less likely, and the four causal paths propagate that surprise
forward.** The cascade is not sign-symmetric; it should cost.

| | prediction |
|---|---|
| sign of `archive_delta_bytes` | **positive** (costs bytes) |
| realized / modelled ratio | **> 1**, and 2-5x would not surprise me |
| what falsifies "the axis survives" | realized cost above the exchange rate at jg1's 1.55 cells/token, i.e. **> ~15.8 bits per changed token** |

The measurement decides it either way, and the curve gets reported whichever way it falls.

### S1e — THE CONTROL PASSED, BYTE-IDENTICAL

| quantity | value |
|---|---|
| frames encoded | 600 |
| emitted stream | **109,696 B** |
| emitted sha256 | `15054e5da33640bcb2e9d4589615c3b89b1312ce27fd9aa8e2a0ec0284b506f2` |
| shipped stream sha256 | `15054e5da33640bcb2e9d4589615c3b89b1312ce27fd9aa8e2a0ec0284b506f2` |
| `byte_identical` | **true** (all 109,696 B) |
| wall clock | 969.8 s |

**This encoder is the exact inverse of the shipping decoder on the pointer body.** Same
model, same 190-group wavefront, same boundary map, same fixed table, same corrector, same
probability quantization, same coder, same flush. Every byte delta below is therefore a
MEASUREMENT of `archive.zip`, not a model of it.

### S1f — THE REAL RATE. MY PRE-REGISTERED PREDICTION WAS WRONG.

| quantity | value |
|---|---:|
| tokens changed | 58 (pairs 283/468/513) |
| token stream | 109,696 -> **109,726 B** |
| **archive.zip** | 176,420 -> **176,450 B** |
| **archive delta** | **+30 B** |
| **measured bits per changed token** | **4.1379** |
| jg1 modelled | 4.718 |
| **realized / modelled** | **0.877** |
| `ΔS_rate` | **+0.0000200** |

**I predicted the real price would be HIGHER than modelled, and named 2-5x as unsurprising.
It is 0.877x — 12.3% CHEAPER.** The prediction is recorded above and is not being edited.

**Where my reasoning failed, precisely: I inferred MAGNITUDE from REACH.** The four causal
paths are real and the cascade is genuinely global — the per-frame ledger shows **317 of
600 frames carry a nonzero bit delta, the first at 283 and the last at 599**, exactly as
the mechanism says. But reach is not cost:

| where the bits are paid | bits | share |
|---|---:|---:|
| at the three EDITED frames | +236.577 | **101.9%** |
| **cascade (the other 314 perturbed frames)** | **-4.409** | **-1.9%** |
| total (ideal code length) | +232.168 | 100% |

**The cascade is structurally unbounded and numerically negligible — here a small CREDIT,
not a cost.** Per-edit: pair 283 +94.2 bits, pair 468 +77.6, pair 513 +64.7; the 30 frames
after pair 513 come back **-15.4 bits**. Re-labelling a boundary cell toward the true class
makes the following frames slightly MORE predictable, and that nearly cancels the local
surprise the later edits create.

The local price also came in under the model, for the reason jg1 itself flagged: **4.718
was the mean over all four neighbour candidates, and a solver pays the accepted one.**
Measured at the sites: **4.079 bits/token**. So jg1's constant was directionally right and
mildly conservative, and its three named caveats (cross-body, context coupling, omitted
table) net out to **-12.3%**, not the multiple I expected.

| bits/token, three ways | value |
|---|---:|
| at the edit sites only | 4.0789 |
| including the whole cascade (ideal) | 4.0029 |
| **realized on `archive.zip`** | **4.1379** |

The 0.135 bits/token between ideal and realized is the coder tax — measured, not assumed.

### S1g — THE RE-PRICED PROJECTION, at the measured rate

Same first-pass rates jg1 measured (58 tokens / 90 repaired cells over 3 pairs), scaled to
600 pairs, with the rate leg now MEASURED and **with a pose charge the -0.0104 headline
omitted** (the carrier re-solve recovers `d_pose` to 1.073x, which is not free through the
sqrt):

| leg | value | S |
|---|---|---:|
| seg | 18,000 repaired cells | **-0.015259** |
| rate | 11,600 tokens @ 4.1379 bits = 6,000 B | **+0.003995** |
| pose | carrier re-solve 1.073x on `d_pose` | **+0.000314** |
| **net** | | **-0.010950** |
| **projected S** | | **0.145576** |

**That clears sub-0.15 with 0.004424 of margin — and it is still a PROJECTION.** The rate
leg is now measured; the seg leg is 90 cells extrapolated 200x, and the pose leg is n=3.

**What would have to be true to miss 0.15** (rate and pose held at their measured values,
seg yield swept):

| realized yield (cells/changed token) | repaired | net S | S |
|---|---:|---:|---:|
| **1.5517 (jg1 measured, first pass)** | 18,000 | -0.010950 | **0.145576** |
| 1.2000 | 13,920 | -0.007491 | 0.149035 |
| **1.0000 (break-even for the goal is near here)** | 11,600 | -0.005525 | **0.151002** |
| 0.8000 | 9,280 | -0.003558 | 0.152968 |
| 0.3900 (jg1's 8-pass iterated yield) | 4,524 | +0.001033 | 0.157559 |

**The goal survives if and only if the first-pass yield holds above ~1.06 cells/changed
token at n600.** jg1 measured 1.462 and 1.500 on single passes and 0.390 when it iterated
one pair to exhaustion — so the stopping rule is not a refinement, it is the whole game.
Anything that pushes past the first pass walks the score back up.
