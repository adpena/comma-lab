# RECALL SAVE #5 — `ddm_cl1` is CLOSED BY RECALL at 4.63× ($0, no burn) · and the same memo convicts my gestalt's central number: the free-derivation seam has ~211 B left, not 2,009 B

Date: 2026-08-31 · Author: MAIN · Cost: **$0** (corpus recall + arithmetic over landed receipts)
Axis: no new measurement. `score_claim=false` · `promotable=false`
`verdict_scope`: **CLOSED-BY-RECALL** for `ddm_cl1` (§1) · **erratum** against
`ddm_gest1_decoupling_gestalt_20260831.md` §4/§5/§6 (§2). Every number is CITED from a landed memo.
Append-only: `gest1` stands as written; this supersedes its seam-size row.

---

## 1. `ddm_cl1` — CLOSED BY RECALL, no burn fired

Following the gestalt's own logic into `ddm_afc1`'s 23-family census, exactly one live strict member
had apparatus built and **zero rungs ever run** — row 12, HPAC nonlinear retrain/widen capacity,
`afc1` verbatim:

> *"CL1 preregistered apparatus but ran no rung; receipt remains blocked and tied to PR130 … **FOLDED;
> requires current-body re-derivation, not duplicate fire**."*

I read `BLOCKED_RECEIPT.md` and `PREREGISTRATION.md` at source and confirmed the apparatus is
genuinely fireable: fail-closed on exact cache/init/source/config, hard governed launch, disabled MPS
CPU fallback, live MPS, canonical EMA after every optimizer step, atomic fsynced immutable
checkpoints, full RNG preservation. It is *not* a stub. An hours-long Metal burn was one GO away.

**The $0 pre-check found the answer already written.** `ddm_mi1_indicator_model_axis_20260824.md`
§(e) names `ddm_cl1` explicitly, quotes its preregistered question, and states the consequence:

> *"**There are TWO never-fired queued experiments on this question, not one.** Besides `#938`/`eu2`,
> `ddm_cl1` … preregistered "whether allocating more serialized bits to the HPAC prior saves more
> serialized token bytes than it costs" with break-even `Δtoken / Δmodel < −1` — **QUEUED-WITH-A-FIRE-ORDER,
> sandbox fire REFUSED, never run.** Both were written against pre-DX2 bodies. §5 prices the question
> they share on the live body, at $0, and **the answer removes the reason to fire either**."*

### The pricing, in cl1's own break-even currency

cl1's admission condition is `Δ(token bytes) / Δ(model bytes) < −1` — the prior must save strictly
more than it costs. `mi1` §6 measured both terms:

| term | measured | source |
|---|---:|---|
| whole conditioning target (realised excess over the model's own entropy) | **2,162.13 B**, z = +11.89 | `mi1` §3 |
| cost of the cheapest paid model in the corpus (10K int8) | **10,000 B** | `eu2` / `mi1` §6 |
| **best achievable ratio** | **2,162 / 10,000 = 0.216** | DERIVED |
| **required** | **< −1** | `cl1` PREREGISTRATION |

**4.63× short, and that grants the paid model a perfect capture of every bit of excess that exists.**
`mi1`'s own sharper framing: a paid model *"would have to be 4.63× more wrong than the shipped model
actually is before this family cleared its own byte cost"* — and the closest measured formulation,
`cx3`, came in at **worse than nothing** (+11,433 B).

### Why the transfer to the current body is sound (not assumed)

`mi1` measured on the DX2/lb1-lineage body. `afr1` (today's pointer move) proved **decoded symbols
bit-identical to lb1** — 600/600 pairs, 117,964,800 tokens, 3,662,409,600 raw bytes, 0 differing. The
FIELD `mi1` measured *is* the field on the current body. The only change since is that afr1 collected
81 B more conditioning, which **shrinks** the remaining excess. The transfer is conservative in the
direction that matters, so `afc1` row 12's *"requires current-body re-derivation"* is **discharged by
bit-identity**, not waived.

**`ddm_cl1`: CLOSED BY RECALL. Disposition `QUEUED-WITH-A-FIRE-ORDER` → `CLOSED_BY_RECALL`. Pricing
authority: `mi1` §5/§6. Burn not fired. $0.** `afc1` row 12's STALE/re-derive disposition is
superseded by this row.

---

## 2. ERRATUM — my gestalt's seam size is overstated by ≥9.6×

`ddm_gest1_decoupling_gestalt_20260831.md` (commit `dc8f53e344`) states in §4, §5 and §6 that
decoupling (A), the free-derivation seam, is **"4.8% from empty (2,009 B / 42,016 B)."** Reading `mi1`
and `afc1` at source for the cl1 check shows that number is **the wrong object**.

### What the two figures actually measure

| figure | what it is | source |
|---|---|---|
| **≈2,009 B** | remaining lossless total from the gb1 pointer, **whole model axis** — a CEILING that includes shipping a paid probability model | `m144` |
| **2,162.13 B** | realised excess over the shipped model's own entropy — the same ceiling, measured independently, **and the thing a paid model would have to capture** | `mi1` §3 |
| **+211.13 B** | **the richest UNCONSUMED zero-stored context**, held-out (`patch192`) — the free half | `mi1` §5 |

`mi1` split the model axis into a **free** half (zero-stored receiver-derived conditioning; the
decoder computes context from `(x,y)` + already-decoded symbols) and a **paid** half (ship a model),
and measured the paid half **4.63× underwater**. **I quoted the ceiling over BOTH halves as if it were
free headroom.**

### The corrected seam size

```
richest unconsumed free context (mi1, held-out)        +211.13 B
already collected by afr1 today (tile48 × groupbin8)     −81    B   (byte-closed, exact)
                                                    ------------
remaining free seam                                    ≤ 211.13 B, and strictly less
```

Strictly less because `patch192 = tile48 × subtile4` shares the `tile48` factor afr1 just collected —
these contexts **overlap and are not additive**. Against the 42,016 B demand:

| | published in gest1 | corrected |
|---|---:|---:|
| remaining free seam | 2,009 B | **≤ 211 B** |
| as % of demand | **4.8%** | **≤ 0.50%** (net of afr1, ~0.31%) |

**A ≥9.6× overstatement, mine.** And it converges with `afc1`'s own one-word verdict on the same
family, which I had flagged as a framing note without quantifying: **"Family is drained."**

### The mechanism of my error, named

This is [[m99]] (units × level × role is part of the claim) and [[m124]] (the floor you divide by
decides the answer) firing on my own arithmetic — a **ceiling over a paid-plus-free axis** quoted as
**free headroom on the free half**. It is the third conviction against my own numbers today, after
`lfb1`'s toy-priced 9.262× and `hyb1`'s area-priced token split. The genus is identical each time:
**I inherited a number from a neighbouring object without re-deriving what it measures.**

`mi1`'s cumulative line is the honest scale of this seam and I should have quoted it instead: every
zero-byte model-axis win the campaign has ever produced — `fx1` −560.07 B (byte-closed), `fx2` D1
−151 B, `ma1` −104.58 B, `mi1` −211.13 B — **totals ≈1,027 B, of which only 560 B is byte-closed.**
That is the whole historical yield of decoupling (A), and it is 2.4% of the demand.

---

## 3. WHAT THIS CHANGES — decoupling (B) is not "the other one," it is the only live one

The gestalt's law is unaffected: *every win is a decoupling of cost from harm; every closure died on
collinearity.* What changes is the ledger under it.

| decoupling | gest1 said | corrected |
|---|---|---|
| **(A) free derivation** | 4.8% from empty, "keep collecting" | **≤0.50% of demand, `afc1`: DRAINED**; historical total yield ≈1,027 B |
| gf1 horizon | bounded 2.04× of 10.30× | unchanged |
| **(B) fcd1 win-win cone** | "the one live head" | **the ONLY live decoupling** — 5,268 edits / −3,756 B / 0 harm, realized d_seg still `NOT MEASURED` |

`ddm_wwc1_winwin_cone_sweep` is LIVE (1/4 codex slots) and its falsifier is now carrying more weight
than when I wrote it: if realized d_seg fails to improve on fcd1's banked union, the label-benefit →
realized-benefit transfer is broken, decoupling (B) closes, and **both** measured decouplings are
exhausted with 42,016 B of demand outstanding. That is the campaign-shaping question, and it is
already running with a −3,756 B receipt attached.

The paid-model route stays closed by `mi1` at 4.63×, `cx3` at worse-than-nothing, and now `cl1` by
recall. **Do not re-open the model axis without a formulation that beats the entire history of the
axis by 9.7×** — that is `mi1`'s bar, and nothing measured is within 100× of the bar above it.

---

## 4. Denominator

Census families examined for a live never-fired member: **23** (`afc1`). Strict members with
apparatus-built-zero-rungs: **1** (row 12 / `cl1`). Closed by recall: **1**. Burns fired: **0**.
Metal hours spent: **0**. Own published numbers corrected: **1** (the seam size, ≥9.6×) — third
today. Dollars: **0**.

The exact pointer did not move in this unit. It **prevented** an hours-long Metal burn on a question
already priced 4.63× underwater on the live body, and corrected the central quantity of the synthesis
I published an hour earlier — in the harder direction.

`[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25; sub-0.12 gap 0.027976171255591042.`
