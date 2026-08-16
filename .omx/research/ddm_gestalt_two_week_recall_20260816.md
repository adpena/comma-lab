# Two-week gestalt recall — rate attack · representation/serialization · regime · the patterns

`date_utc: 2026-08-16` · `owner: MAIN` · operator directive: *"Remember all signal over the course
of the past two weeks bearing on understand and gestalt especially the rate attack and
representation serialization and regime level work and patterns and everything"*

Recalled from the corpus (337 docs, 2026-08-02 → 08-16), not from working memory. This is a
CONSUMPTION document: its job is to state what the arc measured, and to close the rows today's
measurement closes.

---

## 0. The frontier, and the shape of the remaining gap

`hv1 ep0634 — S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]`, sha `80d9c8c6…`
= seg 0.029611 + pose 0.0082945765 + rate 0.1216917. Gap to 0.15 = **−0.0095973**.

The single most important structural fact of the fortnight, measured repeatedly: **pose→0 buys
only −0.0083.** It cannot reach 0.15 alone. Seg and rate are both load-bearing. Every "one axis
closes it" framing that appeared this fortnight is arithmetically false, and the arc corrected
itself on this point more than once.

---

## 1. THE RATE ATTACK — the arc, and what closed today

**The binding number (rfo2, 08-15): −15,157 B at unchanged distortion** to be strictly sub-0.15
from the e480b 183,502 B object. Everything below is measured against that.

### What the fortnight CLOSED on rate

| line | verdict | receipt |
|---|---|---|
| same-state lossless coder race | **CLOSED, 0 B saved** — 8/8 complete alternatives, split-Brotli won | MZ1 |
| exact semantic re-representation | **CLOSED** — all 38/38 semantic tensors receiver-required, 0 derive-at-decode, dense/sparse/row-dict/hybrid each **+340 B** | MZ2 (#1060) |
| wrapper / ZIP / packaging | **CLOSED** — the entire header+residual+ZIP budget is **210 B** | rfo2 |
| pz4a absolute-code coarsening | **CLOSED, +2,232 B net** (a loss) | rfo2 |
| basis rotation | **CLOSED** — 0.83% to IDENT; basis is not the lever | #918 |
| SMEVR vs the token bulk | **CLOSED** — SMEVR LOSES +5,183 B on IX2TOK01; the live coder pays for **LZ match structure**, not symbol rank | sv2 (#859/#940) |
| frozen tq1c prior → MC36 | **INSTANCE/FROZEN-TRANSFER refuted** — the prior does not transfer | rx1 |

**The law that fell out:** *another lossless coder is not the rate lever.* Every remaining
multi-kilobyte mechanism must **change learned state or architecture** — trained width reduction,
carrier rank/atom reduction **with re-fit**, or token-drop/waterfill followed by a coder re-fit.

### TODAY CLOSES RUNG 2 (the truncation half), MEASURED

rfo2's ladder rung 2 was *"Carrier atom/rank reduction plus coefficient re-fit"*, projecting
rank 10/8/6/4 → `3,694 / 7,387 / 11,081 / 14,774 B` gross, `−0.00246 / −0.00492 / −0.00738 /
−0.00984 S`, explicitly labelled **CONJECTURE**, with falsifier: *"the nonlinear Pose increase
costs at least the rate saving."*

**That falsifier FIRED today, measured** (`ra2c` rank-4, advisory n600, archive byte-identical):

```
rank-4 returns 14,662 B  →  rate credit          −0.009763 S   (rfo2 projected 14,774 B: good)
d_pose 1.4747e-4 → 0.35402399  →  pose term  0.038402 → 1.881553
                                  Δ pose term      +1.843151 S
                                  NET              +1.833388 S      falsifier fired by 188.8×
```

Extended to the gentlest rung with the corrected damage constant `K_eff = 37,953` (§4):
rank-11 (1,514 B back, 4.23% Frobenius error) → ratio 68.9 → pose term 0.31876 → Δ **+0.28036 S**
against a **−0.001008 S** credit — **fires by 278×**.

**Every rung of the pure-truncation carrier ladder is refuted, by 189–278×.**

**The surviving half is exactly named.** rfo2's rung 2 was *rank reduction **plus coefficient
re-fit***. I measured **truncation without re-fit**. Re-fit is a different mechanism — and it is
precisely the pose-metric-optimal direction the ra2c erratum named this morning. So:

- truncation → **REFUTED**, bounded at all ranks by Eckart–Young + K_eff;
- **re-fit → OPEN**, and it is now the same object as the pose-metric rung. Two independently-derived
  lines converged on one measurement.

### Why one measurement closed four rungs — the ordering law paying rent

This morning's law (`rate_credit_ladders_run_largest_first…`): rate-credit ladders run
**largest-cut-first** because the affordance bar is **quadratic in returned bytes**. Rank-4 is the
largest cut and therefore has the **loosest** bar (1.5731× advisory / 4.7394× T4). It failed
anyway, by 189×. Testing the loosest rung first collapsed the entire ladder in one run. The law
was banked at 09:xx and earned its keep by 21:xx the same day.

### The live rate ladder AFTER today

| rung | status |
|---|---|
| 1 — MZ2 mixed q3/q4, −823 B materialized | **QUEUED, fire-ordered** (needs shipping-receiver check) |
| 2 — carrier rank/atom **truncation** | **REFUTED TODAY**, 189–278× |
| 2′ — carrier rank **+ coefficient re-fit** (= the pose-metric rung) | **OPEN — the named successor** |
| 3 — joint-proxy checkpoint selection | FIRED-IN-CODE (`tools/select_hpac_checkpoint.py`) |
| 4 — nested-width semantic distillation + QAT, 4–12 KB target | QUEUED (wd3 built) |
| 5 — coder×drop token waterfill on the surviving state | QUEUED |
| 6 — #978 semantic-vs-latent × #982 trained receiver, 8–20 KB | CONJECTURE, sealed-fire only |

**No measured combination supplies 15,157 B.** That was rfo2's honest headline and it is still
true tonight — with rung 2's truncation half now removed from the candidate pool.

---

## 2. REPRESENTATION / SERIALIZATION — what the arc established

- **Receiver-requiredness is the binding constraint, not entropy.** MZ2's decisive result: all 38
  semantic tensors are *receiver-required*; nothing derives at decode. Serialization cleverness
  cannot beat a receiver that needs the bytes.
- **The coder pays for STRUCTURE, not for symbol statistics.** sv2's mechanism: IX2TOK01 moved the
  win from symbol-rank cost to **LZ match structure**; the winning mode maximizes exact-zero runs.
  This single mechanism explains a whole family of negatives, and it explains why marginal-entropy
  surrogates are blind to permutations that move 13–13,466 real bytes (#862).
- **Description ≠ realization.** Repeated across rl1/#939, dm1→dm2, v14/v15: a stream's
  *description* price and its *realized correction* price are different quantities, and only the
  second is admissible. The 2,524× semantic-vs-realized ratio (dm2) is the anchor.
- **d_seg is EXACTLY invariant to frame_0 carrier edits** — now confirmed on THREE independent
  treatments (α=0, α=1, rank-4; 0.00042714 to 8 s.f. each time). Every carrier question is a pure
  **(pose, rate)** trade. This is a confound class permanently removed.
- **Precision is a per-stage waterfill**, not a global setting (operator doctrine 08-11, encoded as
  js1 A10): *as much precision as possible, only as much as necessary and optimal, at every stage.*

---

## 3. REGIME-LEVEL WORK — the honest state

The regime line asked whether a **metric-aimed training window** could change the vehicle's
editability or its exponent. Measured results:

- **b2e (08-16): REGIME_THESIS_INSTANCE_REFUTED.** The "train-for-editability" F2-alone window did
  **not** make semantic weight edits cheaper. Pre-registered bar: pose-damage excess collapses
  **≥50×**. Measured: **0.75×–1.06×** — no collapse at all, and **two of three edits are worse** on
  the burn-2 model than on the shipped hv1 model.
- **rg1 / lr1**: the band-objective probe and lr ladder ran as the gates for that charter.
- **lr1 (08-02) LATTICE-SOLVE REBASE REFUTED (FAMILY)** — the teacher is a **noisier copy of GT**,
  and GT is already in the loss.

**The regime lesson, stated plainly:** training the vehicle to be *more editable* is refuted at
instance scope on this vehicle. The edits do not get cheaper by re-training the substrate under a
band objective. That closes a hoped-for multiplier and returns the campaign to direct rate/pose
mechanisms.

---

## 4. THE PATTERNS — the genus layer this fortnight produced

Five cross-cutting laws, each measured, each now load-bearing:

1. **The instrument's units, level, and aggregation are part of the claim.** Genus of five 08-16
   maxims: units · level (per-ROLE, not per-granularity) · aggregation (pose = mean-of-d_pose,
   NEVER mean-of-ratios — the ratio mean flips the sign) · dispersion≠resolution (σ_log is
   scatter) · ordering (largest-cut-first, quadratic bar).
2. **The control plane fails silently — make the silence loud AT LAUNCH.** Governor stuck-throttle
   (state `T` = suspended, not slow) · no-silent-failures · monitor sleep-loops die to SIGURG.
   Confirmed twice more today: two launch-time instruments caught two of my own defects in seconds.
3. **A prefix of a skewed population is a different population**, and the bias **inverts by axis**:
   pose prefixes 2.5–4.2× HARDER, seg ≈0.96× easier, rate ≈neutral (m88/m96/na4 — the axis triple
   is now complete).
4. **Euclidean is not optimal but is still useful signal** (operator, today). Measured Euclid-vs-
   Fisher cosine SIGN FLIP; today's 9.23× damage-law over-prediction is a third instance on a new
   surface. Report the dual-metric pair; never one alone.
5. **Cross-regime constant transfer is a recurring killer.** qs4's stale Schur compensation
   (+2.4e-4 disaster, cured in-compile by qs5) and today's `K` over-prediction are the same shape:
   a constant fitted in one regime, spent in another.

**Correction banked today under pattern 5:** the carrier damage law's `K = 350,427` (two-point fit
at the α endpoints) **over-predicts intermediate damage by 9.23×** — `K_eff = 37,953` at rank-4's
error. Every closure margin published this morning loosens by 3.04×. The ladder still closes; the
margin is thinner than claimed, and the memory has been corrected at source.

---

## 5. What this recall changes tonight

1. **Rate ladder rung 2 (truncation) is closed** — remove from the candidate pool; do not re-derive.
2. **Rung 2′ (rank + re-fit) is promoted** and is the SAME object as the pose-metric rung. One arm
   serves both lines.
3. **Regime multiplier is refuted** — no editability dividend; rate/pose mechanisms carry the gap.
4. **The gap is still unallocated.** −0.0095973 needed; pose caps at −0.0083; the measured rate
   pool does not reach −15,157 B. Nothing tonight changes the pointer.

## 6. Honest limits

- The rank-4 closure is advisory-axis; the T4 column assumes ratio transfer (weaker here than at
  α=0). Not a score claim.
- The 189–278× closure covers **Frobenius-optimal truncation of the shipped 12-dim carrier**. It
  does **not** bound the re-fit mechanism, and Eckart–Young gives no theorem in the pose metric.
- b2e's regime refutation is INSTANCE-scoped to the F2-alone window on this vehicle.
- This document consumed 337 windowed docs by targeted grep on the operator's named axes, then read
  the gestalt/rate/regime heads in full. It is a synthesis of what those docs say, not an
  independent re-measurement of their claims.
