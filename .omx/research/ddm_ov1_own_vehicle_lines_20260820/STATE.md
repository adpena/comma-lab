---
arm: ddm_ov1
title: "Every own-vehicle line has an owner-state. TR1's closure SURVIVES its own configuration cure and gets STRONGER — the cure flips the binding term from seg to rate, and TR1's byte floor fails by 1.68x even at d_seg=0. Semantic-primary WON by adoption: the live archive codes the label half at 123,211 B, 29.0% below the best coder tk1/hb1 ever raced, so hb1's fire order would measure a number the shipping archive already beats. Two orphans found and discharged: hg1 arm_b (margin_hinge beats tau_softplus 1.466x on d_seg at 0.828x bytes, paired t=-10.85, never folded) and the dxi Rice payload (18.3% above its own order-0 bound, never priced)."
utc: 2026-08-20
axis: "[macOS-CPU advisory / retained-telemetry advisory] unless a row says [contest-CUDA T4 n600]. Real coders for bytes -- NEVER a score."
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "jg5 waterfill-455 -- S 0.14839100138338618 @ 180,625 B [contest-CUDA T4 n600], archive f3bce5d2... -- UNMOVED by this unit"
verdict_scope_default: "INSTANCE/FORMULATION unless a row names FAMILY"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_ov1 — the own-vehicle lines, with a current owner-state each

## ANSWER FIRST

**Nothing in the own-vehicle set is orphaned any more, and nothing in it reopens.** Eight lines,
eight owner-states. The two that mattered are both *closed harder than before* — and the closing
arguments are new, not repeated:

1. **TR1 survives its own cure and still loses.** The #1091 configuration cure (13.6069x) was
   measured on a *different trainer*; granting it to TR1 anyway — an illegal cross-vehicle
   transfer I grant deliberately, to lose the argument on purpose — **flips the binding term from
   seg to rate**. Post-cure TR1's seg is only **0.2036x** of budget, but its smallest-ever
   byte-closed archive (353,805 B) carries a rate term of **0.235584 = 1.588x the ENTIRE jg5
   score**. Even at **d_seg = 0 exactly**, TR1's byte floor is **1.68x** over the ceiling. TR1 is
   closed on an axis no seg work can touch.
2. **Semantic-primary already won — by adoption, not by a race.** The live archive codes its
   label half (109,696 B RC64 tokens + 13,515 B HPAC model) at **123,211 B**, which is **29.0%
   below** the best coder tk1/hb1 ever raced on the same class of object (PP1-KT, 173,617 B on GT
   `lstars`) and **9.2% below** hb1's own projected target (135,732 B). **hb1's fire order would
   measure a number the shipping archive already beats.**

Two real orphans existed, and I discharged both with measurements rather than filing them:

3. **hg1 arm_b was never folded.** tc1 §8 row 5 named the fold-point on 08-17; arm_b finished
   ep399 the same day and sat unread for 3 days. Folded here: **`margin_hinge` beats
   `tau_softplus` by 1.466x on d_seg (0.011571 vs 0.016963) at 0.828x the bytes** — a dominating
   win on both axes, paired over an identical 36-pair gate, **t = -10.854, better on 33/36**.
   It does *not* reopen TR1 (8.2x over budget). It *does* independently corroborate #1091's axis
   on a second vehicle.
4. **The dxi/pose Rice payload had never been priced.** Measured: it sits **18.3% (1,787.7 B)
   above its own order-0 bound on temporal deltas**, worth **ΔS ≤ -0.001190**. Separately and
   decisively: **post-hoc recompression of that section is DEAD** (-37 B, -0.2%, measured) — the
   headroom is only reachable by changing the coder inside the encoder.

**The charter's premise that two SMEVR legs are unfired is FALSE for one of them.** The Lane-crop
SMEVR leg was fired twice already (cg3 08-04: 252,434 B; bf1 08-05: 252,417 B) and **SMEVR lost
to Brotli-q11 by ~45,700 B both times**. I did not re-run it.

**Pointer: UNMOVED.** No line in this memo is a score.

---

## §1 The owner-state table

Status vocabulary: **LIVE** (work in flight) · **PARKED** (named trigger, not yet fired) ·
**CLOSED** (verdict + scope on the ladder) · **ORPHANED** (no owner, no trigger — the finding).

| # | line | last measured state (receipt) | status | trigger / scope | cheapest next measurement |
|---|---|---|---|---|---|
| 1 | **solver line** (up2/up3/jg2/jg3/jg5/qs5) | **S 0.14839100138338618 @ 180,625 B [contest-CUDA T4 n600]**, archive `f3bce5d2…`; `experiments/results/modal_auth_eval_mirror/contest_auth_eval_ddm_jg5_t4_r1.json` (seg 0.00020139 · pose 6.37e-06 · rate exact) | **LIVE — the winner** | — | not mine; the ship wall (`rr2` corrector native port) is MAIN's critical path |
| 2 | **TR1** partition renderer | best-ever d_seg **0.00389011** (n600 `evaluate.py`, pfs1 D1 `624ffe57…`); smallest byte-closed **353,805 B** (cx1/ix2) | **CLOSED** | **FAMILY on the frontier question** (§2); INSTANCE elsewhere. Reactivation: a TR1 archive **< 167,934 B counted** with d_seg ≤ 2.86e-4 — i.e. 2.11x below its own smallest-ever | none. Do not spend on TR1 |
| 3 | **semantic-primary tokens** (tk1) | PP1-KT: **142,001 B** (tq1c argmax) / **173,617 B** (GT `lstars`); `scorer_forwards_run: 0` (`.omx/research/ddm_tk1_20260806/semantic_stream_race.json`) | **CLOSED-BY-ADOPTION** | the live vehicle **is** semantic-primary and codes the same half at **123,211 B** (§3). Scope: FORMULATION on hand-coder label coding | none |
| 4 | **hb1 HPAC-on-our-labels** | "not measured, BLOCKED locally: no CUDA" (`.omx/research/ddm_hb1_20260806/RECEIPT.md`); later partial **135,732 B** at ep54/60, `rc=143`, flagged **UNSAFE** by op1r ADDENDUM §2c | **CLOSED-BY-SUPERSESSION** (was: PARKED on CUDA + #906) | both blockers moved: CUDA available via Modal, and the #906 GT-lineage blocker was lifted 08-19 (jg1 DALI instrument, 0.99995x seg / 1.00081x pose). **But the target is now dominated by 9.2%** — the trigger firing does not make the row worth firing | none. §3 fire-order F3 records the reversal |
| 5 | **capstone hybrid** (hy1) | C1 solved plane **114,717 B** under frozen F26 HPAC+RC64 (**+11 B** vs 114,706); decoder reproduced **117,964,800/117,964,800** tokens exactly; grammar expresses **27,351/27,351** changes (`ddm_hy1_capstone_hybrid_20260811.md`) | **PARKED** | named in-memo: *"a whole-container rebuild and joint scorer replay are mandatory"* — the +11 B is **not additive** to the live base. **Zero occurrences of "hybrid" in `main_hot_state.md`** | rebuild against the **jg5** container, not cp135, then one joint scorer replay |
| 6 | **SMEVR race legs** | leg (a) Lane-crop: **FIRED TWICE** — cg3 252,434 B / bf1 252,417 B vs Brotli-q11 206,688 / 205,135 → SMEVR **loses ~45,700 B**. Base-rule race: **+5,183 B** on IX2TOK01 (sv2). cp135: **0 wins / 14 sections** | **CLOSED-REFUTED** | FORMULATION: SMEVR loses where the live coder pays for **LZ match structure**, not symbol rank. One standing win: sparse stroke payload 1,392 B (st2) | none |
| 6b | **pose/dxi coder leg** | **NEVER PRICED** until this unit. fx2 R5 mapped the 22,161 B carrier and priced only the semantic blob | **ORPHANED → discharged here** (§5) | — | done: 18.3% headroom, ΔS ≤ -0.00119 |
| 7 | **LOTTO** | `lotto` renderer **3,284 B** vs `plain` **20,214 B = 6.2x** (tc1 §3.1); ix2: lottery mask is **1 B above its combinatorial bound** | **CLOSED-OPTIMAL** | INSTANCE: at the bound, nothing to win. Per-pair phase index into a shared codebook (ph1 fire-order 3) is **NOT RACED** and belongs to the ph1 owner, not here | none |
| 8 | **witness / level-set** | last touch **2026-08-10** (`ddm_lt1_levelset_longtail_forces_port_20260810/`) | **CLOSED-BY-SUPERSESSION** | superseded twice: witness → TR1 → PR130-lift. Lessons transfer (m18/L18); numbers do not | none |
| 9 | **hg1 arm_b hinge fold** | ep399 complete, telemetry retained, **never read** | **ORPHANED → discharged here** (§4) | — | done: 1.466x d_seg at 0.828x bytes, t=-10.854 |

---

## §2 THE TR1 REACTIVATION ADJUDICATION — the cure fires, and TR1 still loses

### §2.1 What is being adjudicated, and against which bar

tc1 (08-17) closed TR1 against the **hv1** bar (0.15959729295498598). The bar has since moved to
**jg5, 0.14839100138338618 @ 180,625 B [contest-CUDA T4 n600]**. A delta without its baseline is
unanchored, so every number below is re-derived against jg5, not carried.

The identity reconstructs to the last digit, so custody is sound:

```
seg  100 x 0.00020139        = 0.0201390000
pose sqrt(10 x 6.37e-06)     = 0.0079812280
rate 25 x 180,625 / 37,545,489 = 0.1202707734
                          SUM = 0.14839100138338618   (residual 0.000e+00)
```

**seg+rate budget, granting TR1 jg5's pose for free = 0.1404097734.**

### §2.2 The scope alarm I am deliberately overriding

`#1089` (CE takes **81.1497%** of the LR budget) and `#1091` (aligned-objective aim closes
**13.6069x**, 92.651%) were measured on `src/tac/pr130_lift/train_semantic_quantized_resumable.py`.
**Verified at source in this unit:** `--ce-fraction`, `--softplus-fraction` and
`--band-objective-weight` return **1 hit each** in that trainer and **0 hits each** in
`experiments/train_tr1_partition_renderer_mlx.py`. Transferring the 13.6069x to TR1 is a
cross-regime constant transfer — the poison class.

**I grant it anyway.** Granting the opponent an illegal favour and still winning is the only form
of this argument that cannot be reopened by curing the illegality.

### §2.3 Pre-cure — unchanged in sign, re-anchored in magnitude

| TR1 row (all n600) | d_seg | seg | B | rate | seg+rate | x budget |
|---|---:|---:|---:|---:|---:|---:|
| BEST-EVER d_seg — pfs1 D1 | 0.00389011 | 0.38901 | 569,996 | 0.37954 | 0.76855 | **5.47x** |
| ep854 counted (SMEVR price) | 0.003943 | 0.39430 | 275,381 | 0.18336 | 0.57766 | **4.11x** |
| terminal frontier — qo1 | 0.00431179 | 0.43118 | 357,836 | 0.23827 | 0.66945 | **4.77x** |
| smallest byte-closed — cx1/ix2 | 0.00431179 | 0.43118 | 353,805 | 0.23558 | 0.66676 | **4.75x** |
| bc1 ep399 from-birth counted | 0.005169 | 0.51690 | 253,858 | 0.16903 | 0.68593 | **4.89x** |

Seg term alone / budget = **2.7705x** (tc1 measured 2.571x against the hv1 bar; the move to jg5
tightens it). Free archive + jg5's pose: **S = 0.396992 = 2.675x** the bar.

### §2.4 Post-cure — THE BINDING TERM FLIPS

Cured d_seg = 0.00389011 / 13.6069 = **2.85892e-04** → seg term **0.028589** = **0.2036x** of
budget. **Seg stops binding.** Rate takes over, and rate is where TR1 dies:

| TR1 byte row + cured seg + jg5 pose free | S | x bar |
|---|---:|---:|
| FREE archive (B = 0) | 0.036570 | 0.246x |
| jg5's own archive size (180,625 B) | 0.156841 | **1.057x** |
| bc1 smallest counted (253,858 B) | 0.205604 | **1.386x** |
| smallest byte-closed (353,805 B) | 0.272155 | **1.834x** |
| gr1 cell-drop50 (359,221 B) | 0.275761 | **1.858x** |

**The byte ceiling.** Solving for the largest archive that clears the bar:

| assumption | ceiling |
|---|---:|
| cured seg (2.86e-4) + jg5 pose free | **167,934 B** |
| **d_seg = 0 exactly** (perfect seg, impossible) + jg5 pose free | **210,870 B** |

TR1's smallest-ever byte-closed archive is **353,805 B** — **2.11x** over the cured ceiling and
**1.68x** over the perfect-seg ceiling. Its rate term alone is **0.235584 = 1.588x the ENTIRE jg5
score**; bc1's counted 253,858 B is **1.139x** the entire score.

**Verdict.** *Even with a perfect segmenter and a free pose section, TR1's byte floor exceeds the
whole jg5 budget by 1.68x.* The configuration cure changes the arithmetic materially — it moves
the binding term from seg to rate — and the closure **strengthens**, because rate is the axis TR1
already exhausted (SMEVR raced 9-way, quant-levels swept, carriers refit, cell-drop measured;
tc1 §3.4, ra2 "≤1.93% of the bar"). **No seg cure can reach a rate wall.**

**Scope: FAMILY, on the question "can TR1 be the frontier vehicle".** INSTANCE elsewhere — TR1's
mechanisms remain lesson-eligible (§4 is one). **Reactivation condition, stated as an obligation
on any future claimant:** a TR1 archive under **167,934 B counted** *with* d_seg ≤ 2.86e-4,
byte-closed through a TR1-native exporter that does not exist yet (tc1 G5). Absent that, TR1 does
not get another slot.

**And I checked the direction of its own R–D curve.** tc1 measured, on the live control lineage,
that fidelity is **bought** with bytes (d_seg improved 28.9x while counted bytes grew 2.68x).
Shrinking toward 167,934 B therefore moves d_seg the *wrong* way. I quote only the **sign**: tc1
self-falsified its own log–log extrapolation (inverting it returns d_seg 7.79 > 1.0), and I do not
resurrect the magnitude.

### §2.5 What this does to lc1's "TR1-primary" crowning

`lc1` (08-05) crowned the TR1 learned carrier PRIMARY on `net_fixed = -12,884` at n32 — but that
was a routing decision **between PE3 target labels and the TR1 carrier**, both inside the
own-vehicle family. It never compared TR1 to the inherited body. It is **not superseded as a
PE3 verdict** and **is superseded as a routing claim**: the primary is the live vehicle.

---

## §3 SEMANTIC-PRIMARY, tk1 AND hb1 — won by adoption; the fire-order reverses

### §3.1 The decisive comparison

jg5's section census (`/Volumes/APDataStore/pact/ddm_jg5/retained/final/CLOSE.json`, schema
`ddm_jg5_close.v1`): tail **113,943** · semantic renderer **30,856** · pose carrier **22,197** ·
HPAC model **13,515** · container **114** = 180,625 B. The tail splits at
`runtime/residual_archive.py:478-494` into a 96 B fixed table + the RC64 stream; the base body
(pre-edit) carries **109,696 B** of RC64 tokens (jg2, byte-identical control, sha `15054e5d…`).

**The live label half = 109,696 + 13,515 = 123,211 B.**

| coder raced on the same class of object | bytes | live is |
|---|---:|---|
| tk1 PP1-KT temporal context-arith on **GT `lstars`** (lossless) | 173,617 | **29.0% better** (ΔS -0.033563) |
| tk1 PP1-KT on tq1c parent argmax | 142,001 | **13.2% better** (ΔS -0.012511) |
| hb1's own projected our-GT-HPAC target | 135,732 | **9.2% better** (ΔS -0.008337) |
| PR130's shipped GT-HPAC | 137,159 | **10.2% better** (ΔS -0.009287) |

### §3.2 Why lossless label coding cannot fit, independent of coder

Ship lossless GT labels at PP1-KT's 173,617 B and grant jg5's pose free: rate 0.115604 + pose
0.007981 = **0.123586**, leaving **0.024805** S-units for seg *and* the renderer *and* the model
*and* the edits. jg5's actual seg term is 0.020139, leaving **0.004666 = 7,006 B** for a renderer
(PR130's is 40,252 B) plus a 13,515 B model. **It does not fit, by ~48,700 B.**

And the reason it can never fit is measured, not argued: **the realization gap.** hv1's token
field reproduces the GT SegNet argmax to **8.48e-06** while its realized d_seg is **2.9611e-04**
— **34.9x**. Label fidelity is 34.9x past mattering. *Paying lossless label bytes buys accuracy
the renderer throws away.* The live stack wins precisely because it is lossy down to the
realization floor and not one byte past it.

### §3.3 hb1 (#982) — the in-progress row, dispositioned not skipped

hb1's 08-06 receipt is honest and complete: it did **not** produce an HPAC row on our payloads
(`cuda_available: false`), and it wrote a 5-stage fire order. Its two blockers have since moved —
CUDA is reachable via Modal, and the #906 GT-lineage blocker that made the "135,732 vs 137,159 B"
comparison UNSAFE was lifted on 08-19 (jg1's DALI instruments: seg 0.99995x, pose table 1.00081x).

**So the trigger fired — and the row still should not run.** Its target (135,732 B) is **9.2%
worse than what already ships**. Firing it would spend a Modal slot to measure a number the
shipping archive beats.

**Disposition: CLOSED-BY-SUPERSESSION**, scope **FORMULATION** (PR130-recipe HPAC retrained on
our label caches as a *replacement* for the live token stream). **Not closed:** hb1's stretch rows
4 and 6 — CPR1-style Huffman/Rice repack and bit-depth self-compression of the HPAC model — which
target the **13,515 B model**, a different object. Those are alive and belong to the `hm1`/`cl1`
line, whose measured law already governs them (HPAC returns **3.810 token-B per counted byte**;
correction-table branch CLOSED at FORMULATION scope on hv1).

---

## §4 ORPHAN DISCHARGED #1 — the hg1 arm_b fold-point (NEW MEASUREMENT)

tc1 §8 row 5 named this fold-point on 2026-08-17 and marked it *"arm_b fired 18:52Z, lands after
this seal."* arm_b reached ep399, wrote 492 telemetry rows, and **no one read them for three
days**. Discharged here at $0 from retained telemetry.

**Design.** Both arms fresh-start, 400 epochs, one axis: `--seg-form`. arm_a ran CE→`tau_softplus`
(the PR95-inherited default); arm_b ran `margin_hinge` throughout. Gate set **identical** — the
same 36 pair ids in both arms, verified by set equality.

**Result at ep399, matched gate:**

| | arm_a `tau_softplus` | arm_b `margin_hinge` | ratio |
|---|---:|---:|---:|
| `realized_gate_dseg_mean` | 0.016963393599898728 | **0.011571389657479746** | **0.6821x** (-31.8%) |
| `total_counted_bytes` | 523,315 | **433,226** | **0.8278x** (-17.2%) |

**Paired test over the 36 shared pairs:** mean(B-A) = **-0.00539200**, se = 0.00049677,
**t = -10.854**, B better on **33/36**, worse on 3, tied on 0.

**This is a win on both axes at once** — not a bytes-for-fidelity trade. Per-class, arm_b improves
every one of the five GT classes, most on Lane (0.004626 → 0.002378, 1.945x) and MyCar
(0.001898 → 0.000529, 3.591x).

**Denominator honesty.** n = **36 gate pairs, not 600**. The sample is **not a prefix**: ids span
1..566, mean 322.6, all ten per-60 deciles populated. The prefix-bias law ([[m88]]/[[m96]]) does
not fire here.

**Axis + scope.** `[retained MLX/Metal training-telemetry advisory]`, **not** n600
`upstream/evaluate.py`. tc1's EMA-warmup defect (G2) contaminates each arm's *absolute* endpoint;
both arms share it, so the **difference** survives and the absolutes may not be quoted against any
other lineage. d_pose (142.96 vs 144.61) carries its own label
`advisory_trend_channel_n600_probe_authority` and is **used for no conclusion** — raw TR1 ships an
inert 83 B pose stub. Scope: **INSTANCE** on the TR1 vehicle at 400 fresh epochs.

**What it does and does not mean.**
- It **does not reopen TR1.** arm_b's seg term would be 1.1571 — **8.2x** the jg5 seg+rate budget.
  And arm_b is fresh-start: it is still **2.97x worse** than TR1's warm best-ever (0.00389011),
  exactly the fresh-vs-warm floor `wd3` measured.
- It **does corroborate #1091's axis on a second, independent vehicle.** Two trainers, two
  objective families, same sign: the **form of the seg objective dominates**. Magnitudes differ
  **9.3x** (1.466x here vs 13.6069x there) and **neither transfers** — that is the whole point of
  the constant-transfer law. The transferable claim is the *axis*, and it now has two witnesses.
- tc1's §5.3 falsifier for the sealed `class_field` ticket asked for a comparison *at matched
  counted bytes*. arm_b wins at **17.2% fewer** bytes, so that falsifier is satisfied in the
  strong direction for `margin_hinge` — a **different lever** from the ticket's `class_field`,
  which remains unfired.

---

## §5 ORPHAN DISCHARGED #2 — the pose/dxi coder leg (NEW MEASUREMENT, and a failed control)

The charter named a "pose/dxi coder race leg". Only one of the two named SMEVR legs was genuinely
unfired, and **it is not an SMEVR leg at all**.

**SMEVR is structurally inapplicable.** `experiments/ddm_r7_token_coder.py:258-270` refuses any
input that is not `uint8 [P,H,W,C]` with `2 <= levels <= 16` and `size <= 16,000,000`. The pose
codes are `(600,12) int32` in **[-2047, 2047]** with **2,357 unique values** — a 4,095-level
alphabet, ~256x outside the coder's domain. **CLOSED, structural, FAMILY scope for this object.**
(The same gate refuses a full `[600,384,512]` label map twice over: rank ≠ 4, and 117,964,800 is
7.37x the cap.)

**What I raced instead**, on the retained shipped codes
(`/Volumes/APDataStore/pact/ddm_t1h/shipped_base_codes.int32.npy`):

**Positive control — FAILED, and that bounds the claim.** T1H's receipt records the shipped
carrier as `rice_bits 78,036` (= 9,754.5 B) + a 12,467 B fixed prefix. My best reconstruction —
per-coordinate temporal-delta zigzag-Rice at the shipped `ks` — gives **78,963 bits, +1.19%**.
Sixteen other formulations (raw zigzag, sign-magnitude, uniform k = 4..12) are all further away.
So I have the **structure** right and **bit-exactness wrong**. Per the verdict-clearance rule, a
race whose positive control fails **cannot price the shipped section**. I am reporting it as a
bound, not a price.

| measurement | bytes |
|---|---:|
| shipped Rice payload (T1H receipt) | 9,754.5 |
| my temporal-delta Rice reconstruction (control **FAIL**, +1.19%) | 9,870.4 |
| **order-0 entropy bound, per-coordinate, on temporal deltas** | **7,966.8** |
| order-0 bound, per-coordinate, on raw codes | 7,976.6 |
| best real coder on the code array (brotli11 / delta-zigzag-u16, column-major) | 10,101 |

**Headroom: 9,754.5 - 7,966.8 = 1,787.7 B = 18.3% of the Rice payload = ΔS ≤ -0.001190**
(0.99% of the archive). Rice is within ~0.03 bits/symbol of optimal *for a geometric source*; the
deltas are a heavy-tailed mixture, which is exactly where an adaptive arithmetic/ANS model
collects the gap. **This is the price fx2's unfired R5 leg was missing.**

**And the post-hoc route is DEAD — measured, not assumed.** Recompressing the shipped
22,183 B carrier section: **-37 B (-0.2%)** whole-section; the 12,467 B fixed prefix **-87 B
(-0.7%)**; the Rice tail **+4 B (+0.0%)**. At H0 ≈ **7.97 bits/byte** the section is already
entropy-dense. This is consistent with ra2's standing "lossless post-hoc ≤ ~1.93% of the bar" and
it **sharpens** it: the 1,787.7 B is reachable **only** by changing the coder that *produces* the
section, never by recompressing its output. Those are different claims and prior memos have
conflated them.

**Payloads persisted** (ALWAYS KEEP THE PAYLOAD): 12 coder outputs + manifest with per-file
sha256 at `/Volumes/APDataStore/pact/ddm_ov1/retained/` (`DXI_CODER_RACE.json`, schema
`ddm_ov1_dxi_coder_race.v1`, carrying `positive_control_pass: false`).

---

## §6 Fire-or-fold, per parked row

| # | row | verdict | detail |
|---|---|---|---|
| **F1** | **dxi in-encoder re-code** | **FIRE — $0, local, scorer-free** | Reproduce the shipped 78,036 Rice bits **bit-exactly first** (my control failed at +1.19%; without it no price is admissible). Then race an adaptive arithmetic/ANS model on the temporal deltas against the Rice payload. **Ceiling 1,787.7 B, ΔS ≤ -0.001190.** Admission bar: derive from `canonical_frontier_pointer.json` **at fire time**, never from this memo. Receiver cost is **not** included and will offset part of it. |
| **F2** | **hg1 arm_b → the live vehicle** | **FIRE as a lesson, not a lever** | Two vehicles now agree the seg-objective *form* dominates. The live trainer already exposes `--ce-fraction` / `--softplus-fraction` / `--band-objective-weight`; F2 is to make sure the #1091 sweep covers a **hinge-shaped** aim, not only CE/softplus fractions. **Carry no magnitude** from either vehicle. |
| **F3** | **hb1 HPAC-on-our-labels** | **FOLD — do not fire** | Target 135,732 B is **9.2% worse** than the shipping 123,211 B. Scope FORMULATION. Stretch rows 4/6 (model-byte self-compression) stay alive under `hm1`/`cl1`. |
| **F4** | **hy1 capstone hybrid** | **FIRE ONLY ON REBASE** | The +11 B result is against **cp135**, whose HP3 object and 115,231 B token stream are **not** jg5's. Rebuild the container against jg5 and run **one** joint scorer replay. Until then it is not a candidate — hy1 says so itself. |
| **F5** | **TR1 (all rows)** | **FOLD** | §2. Includes tc1's sealed `class_field` ticket (`tc1_launch_now: false`) and the 4 surviving PRE candidates from the oq1 fold — real work, retired vehicle. |
| **F6** | **SMEVR Lane-crop leg** | **FOLD — already fired** | cg3 252,434 B / bf1 252,417 B vs Brotli-q11 206,688 / 205,135. Correct the charter premise at its source. |
| **F7** | **LOTTO** | **FOLD** | Mask is 1 B above its combinatorial bound. The unraced ph1 phase-codebook rung belongs to ph1. |
| **F8** | **witness / level-set** | **FOLD** | Superseded twice; lessons only (m18/L18). |

---

## §7 Orphans found, and how each was disposed

| orphan | how it was orphaned | disposition |
|---|---|---|
| **hg1 arm_b fold-point** | tc1 §8 row 5 named it, arm_b landed **after** the seal, nobody returned | **DISCHARGED** — measured and folded in §4 |
| **pose/dxi coder price** | fx2's R5 charter named "dxi/pose"; the delivered memo priced only the semantic blob | **DISCHARGED** — priced in §5, scoped by a failed control |
| **hb1 (#982) in-progress** | receipt said BLOCKED-no-CUDA; both blockers later moved and nobody re-checked | **DISPOSITIONED** — CLOSED-BY-SUPERSESSION, §3.3 (not silently skipped) |
| **23 TR1-class rows routed by oq1 to `ddm_tc1`** | oq1 landed `f9cf240434` 14:08:06, tc1 `7763583f6d` 14:11:41 — 3m35s apart | **NOT an orphan** — tc1 §9 consumed all 35, narrowed to 4 PRE survivors. Verified, not assumed |
| **charter premise "two unfired SMEVR legs"** | leg (a) fired twice, 08-04 and 08-05 | **CORRECTED** at source, §6 F6 |

---

## §8 What this unit did NOT establish

* **No score, no pointer move.** Every number here is advisory, retained-telemetry, or arithmetic
  on existing receipts. jg5 stands **UNMOVED**.
* **My dxi positive control failed** (+1.19%). §5's 1,787.7 B is a **bound derived from an
  entropy calculation on the code array**, not a measured price on the shipped section. F1 states
  reproduction as a precondition, not a nicety.
* **The 13.6069x is not a TR1 number** and I never treat it as one — §2.4 grants it *in order to
  lose the argument on purpose*. If it is later re-measured smaller, §2's verdict only hardens.
* **I did not run the semantic head**, did not fire any TR1 arm, launched nothing, and spent $0.
* **I did not re-derive** qs5's Schur result, up2's basis-limited finding, jg2's 4.1379 bits/token,
  or the tk1 KT internals. I read them, scoped them, and used them at the precision they were
  published.
* **jg5's section census is DERIVED where it splits the tail** (113,943 = 96 B table + ~113,847 B
  RC64), from jg2's source read at `runtime/residual_archive.py:478-494`, corroborated by
  `S1_encode_jg5_subset455.json` (`code_bytes_ideal 113846.99`). The four top-level sections are
  MEASURED in `CLOSE.json`.
* **A `455` vs `454` label discrepancy stands** between the jg5 memo/hot-state (455 admitted) and
  `CLOSE.json`'s winning level (`pairs_admitted: 454`, path `archive_level_0454.zip`). Same bytes,
  same sha; the label differs by one. Routed to MAIN — not resolved here.

---

**Own-vehicle frontier: jg5 waterfill-455 — S 0.14839100138338618 @ 180,625 B
[contest-CUDA T4 n600], archive `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e`
— UNMOVED by this unit.**
