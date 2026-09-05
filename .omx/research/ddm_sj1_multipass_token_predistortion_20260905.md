# ddm_sj1 — multi-pass token PRE-DISTORTION to convergence (measured on the cl2 body, built and sealed on the rc1 pointer)

Tokens: `[no-triality] [p0-ledger-ok]` · Arm: ddm_sj1 (Opus) · Date: 2026-09-05 · Owner: MAIN dispatches T4.
Axes: d_seg `[macOS-CPU advisory, jg1 instrument, DALI GT lineage, cpu_torch argmax]`; bytes exact;
`score_claim=false`, `promotable=false` until a T4 row.

Charter: `.omx/research/charters/ddm_sj1_multipass_token_predistortion_to_convergence_20260905.md`.
Lane: `lane_ddm_sj1_multipass_token_predistortion_20260905`.

**Canonical equations this arm measures against** (`tac.canonical_equations`):

* `cw1_realized_acceptance_monotonicity_v1` — the acceptance law this arm's engine is an
  instance of; its canonical producers are `experiments.ddm_up2_shipping_pose_solve` and
  `experiments.ddm_jg1_seg_solve`, and §2 below reports a MEASURED refinement of its
  admissible attribution scope (a windowed proxy for the realized objective is not
  admissible on this actuator).
* `token_rate_model_direction_dependence_v1` — why every byte number here comes from a REAL
  re-encode and never from a `−log2 p` sum; producers `experiments.ddm_jg2_tail_reencode`,
  `experiments.ddm_fs3_jg5_real_price_reopen`.

A multi-pass YIELD law (`token_predistortion_multipass_yield_v1`) is registered in §3 once
pass 2 has produced its measured decay, not before.

---

## 0. Custody — every identity verified before anything was proposed

| object | path | sha256 | bytes |
|---|---|---|---|
| GT argmax table (DALI, T4-scored) | `/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt` | `a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994` | 117,980,732 |
| body archive (cl2 λ=1.0 sealed candidate) | `…/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/receiver_copy_runtime/archive.zip` | `08ec85333d13d71344b4482cf261e3b2d508725e49f3ca05971265a81498ad4e` | 179,982 |
| token field (held bit-identical across all five cl2 copies) | `…/rungs/lambda_1p0/retained/decoded_tokens.u8` | `cc10a7b09353c0af…` | 117,964,800 |
| receiver's own decode | `…/parseback/lambda_1p0/0.raw` | `f86bfaf39f83bcccb1df14ed3cf982767dc94d3a91cd956f00be923612fec4e0` | 3,662,409,600 |
| base T4 row (fs2) | archive `a8f3a3791499b2b6…` @ 180,023 B | S 0.14784474152757654, d_seg 0.00020139, d_pose 6.14e-06 | — |

`ddm_up2.verify_gt_lineage(axis="contest_cuda", declared_lineage="dali")` was run and PASSED on every
load. MAIN's pin (2026-09-05) binds this: the PyAV table `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`
differs from the DALI table at 20,671 argmax sites — 87% of the whole 23,757-flip budget — so acceptance
computed against it would aim at sites the contest does not score. This arm never opened it.

## 0a. Object change 2026-09-05 — the pointer moved to rc1, and the edits still compose

MAIN moved the pointer to `ddm_rc1`'s lossless recode of the two RX1 MODEL sections:
**S 0.14666350774473783 @ 178,249 B**, sha `1438049e3655fbcfa8eb289fa51ac58f834d72d8a09586353663cea68e57c122`,
tree `/Volumes/VertigoDataTier/pact/ddm_rc1_model_section_adaptive_recode/staged_runtime/`.
This arm now BUILDS and SEALS on that tree and admits against that score. I verified the
composition premise rather than taking it, because everything measured above was measured
on cl2's body:

| section | cl2 | rc1 | identical? |
|---|---:|---:|---|
| carrier | 22,031 | 22,031 | **yes** |
| tail (96 B residual + 113,419 B token stream) | 113,515 | 113,515 | **yes** |
| hpac (model) | 13,466 | 12,343 | no — lossless recode |
| semantic (renderer weights) | 30,856 | 30,246 | no — lossless recode |
| header | 14 | 14 | no — RX1 reserved `0x1A` → `0x7A` |
| **archive** | **179,982** | **178,249** | −1,733 B |

Three MEASURED facts make the composition safe:

1. **The token coder and the field it codes are untouched** — the tail is byte-identical,
   so the stream cl2's encoder mirrors is the stream rc1's receiver decodes. That is why
   the edited field is priced through cl2's path (as chartered) and the emitted stream is
   spliced into rc1's member: asking jg2's encoder to materialise a model out of a section
   coded by a codec it has never seen would be the mistake.
2. **The renderer weights are the same object.** rc1's SEAL falsifier 1 states both model
   sections are restored byte-for-byte BEFORE any parsing, proven three ways, so the
   distortion legs are zero BY CONSTRUCTION. Every render and every argmax in §1–§3 —
   taken on cl2's tree — therefore describes rc1's body too. **Pass 2a needed no restart.**
3. **The score reproduces exactly.** `100·0.00020139 + √(10·6.14e-06) + 25·178249/37545489`
   = 0.1466635077447378, matching MAIN's quoted value to the last digit.

And one trap, checked before it could bite: rc1 flips the RX1 reserved byte, so a rebuild
that reset it would silently produce different bytes. MEASURED: `up3.parse_shipped_body` +
`build_archive` rebuild rc1's body from its OWN carrier codes to sha `1438049e3655fbcf…` at
exactly **178,249 B** — byte-identical. The carrier splice survives the recode, and the
close path's identity control (which refuses unless that holds) is the standing guard.

## 1. Step 0 — the instrument reproduces the frontier body's seg leg (MEASURED)

**Forward-model control:** re-rendering the shipped tokens through the receiver's own
`SemanticTokenRenderer` (batch 1, as `cpr1/inflate.py:312` runs it on CPU) reproduces the shipped frames
**byte for byte** — `max_abs_delta = 0` over 6,104,016 pixels on 2 seeded-random pairs.
Receipt: `step0/forward_control_smoke.json`.

**Seg leg, n600, full field, `cpu_torch` argmax from the receiver's own `0.raw`:**

| lineage | d_seg MEASURED | flipped cells | vs published |
|---|---|---|---|
| **DALI** (contest-CUDA axis) | **0.00020132276746961808** | **23,749** | T4 row 0.00020139 → residual **−0.033%** |
| PyAV (advisory axis) | 0.00034740871853298610 | 40,982 | — (1.726× the DALI leg) |

Receipt: `step0/STEP0_RESULT.json`, argmax field sha `68f5ad9604090ebd…`.

MAIN's pin quoted the jg1 step-0 DALI reading as 2.0387e-4 (1.23% above T4). This arm MEASURES
2.01323e-4, i.e. **−0.033% from T4 and −1.25% from that pin**. The instrument therefore reproduces the
T4 seg leg to five figures on this body, and the 1.23% gap in the pin does not reproduce here. Every
projection below carries the −0.033% residual explicitly rather than absorbing it.

**Flip structure (MEASURED, n600, DALI):**

* 90.5% of flipped cells (21,489 / 23,749) already carry the RIGHT token — the debt is
  render→re-segment loss, confirming jg1's governing law on this body (jg1 measured 95.9% on the
  ancestor body; pass 1 spent 9,179 deliberately-wrong tokens, which is why the fraction fell).
* Pass 1's footprint: **9,179 tokens differ from GT**, on 590 of 600 pairs. Carrying jg1's base leg
  0.00030307 to this body's 0.00020132 gives 12,003 cells repaired for 9,179 changed tokens =
  **1.308 cells per changed token** for pass 1 (DERIVED from the two measured legs).
* Edge census (gt row, ours column): Road is on 80.5% of flips; Lane is 44.6% of flips while being
  0.59% of area. The two dominant edges are Lane→Road 5,707 and Road→Lane 4,527.
* The residue is a **speckle, not a ribbon**: 89.9% of 8-connected flip components are a single cell,
  mean component size 1.16, 32.9 components per pair, 39.6 flipped cells per pair.

## 2. The influence probe — the measurement that redesigned the search (MEASURED)

1,296 realized single-token moves over 36 seeded-random sites on 18 pairs (six disjoint seeds), full
3×3 × 4-class family per site. Receipt: `probe/probe_{0..5}.json`.

| quantity | MEASURED |
|---|---|
| moves that change the argmax at all | 85.6% |
| argmax cells changed per move | p50 **2**, p90 6, max 18 |
| Chebyshev radius of the furthest changed cell | p50 **19**, p90 **155**, p99 371, max **416** |
| fraction of moves whose whole response fits inside radius 8 | **34.2%** |
| moves that repair ≥1 flip | 18.1% · moves that make it worse: 55.3% |

**The response to a single token change is SPARSE but LONG-RANGE.** A move moves a median of two
argmax cells, and the furthest of them sits a median of 19 and a p90 of 155 token cells away — SegNet's
receptive field, not the renderer's (the renderer's own radius is 9, DERIVED from `cpr1/inflate.py`:
coord_mix 1×1 + depthwise 3×3 at dilations 1,1,2,4 + head 3×3).

This **falsified the affordable design I started from.** Batching many spatially-separated moves into
one render and attributing each inside a local window would have mispriced two thirds of all moves by
ignoring exactly the far-field cells that decide their sign. The search was rebuilt as true greedy
coordinate descent on the pair's WHOLE realized flip count — one render + one SegNet forward per move,
with SegNet batched ACROSS pairs (MEASURED 0.315 s/frame at batch 8 vs 0.526 s/frame at batch 1, 3
threads) while the render stays batch 1 because `ddm_up2` sec.6 measured batch 8 as byte-changing.

Yield by class role (the role, not the class index, because Road is the hub of 80.5% of flips):

| role | moves | hit rate | mean flips repaired |
|---|---|---|---|
| `gt` (widen the class SegNet is missing) | 159 | **28.9%** | +0.377 |
| `other` | 969 | 17.8% | +0.274 |
| `ours` (widen the class it wrongly sees) | 168 | 10.1% | +0.161 |

`verdict_scope`: SIZING instrument (n = 36 sites). It ORDERS the sweep; no combination was excluded
from the n600 pass on this evidence.

## 3. The engine, and why it is the shape it is

Greedy realized coordinate descent, per pair, on the pair's whole realized flip count:

```
A = argmax(render(T));  F = |A != G|
for each move m in the ordered 36-family:
    skip m if its SITE already reads GT (an earlier move's FAR FIELD may have repaired it)
    A' = argmax(render(T + m))            # receiver's own renderer, frozen CPU SegNet
    if |A' != G| < F:  T += m;  A = A';  F = |A' != G|
```

Monotone by construction: a pair's flip count never rises. The render runs at batch 1
(byte-identity contract); SegNet is batched ACROSS pairs, 8 in flight per shard.
Shards are STRIDED over all 600 pairs, never contiguous blocks.

**Shakedown (4 seeded pairs, `gt` stage only, MEASURED):** 166 flips → 83, i.e. **50.0%
repaired**, 72 changed tokens, **1.153 cells per changed token**, ~109 realized evaluations
per pair. The charter's FALSIFIER (a full pass repairing < 8% of remaining flips) is cleared
by 6× on the `gt` slice alone.

**Close-path identity control (MEASURED, before any candidate exists):**
`ddm_up3_carrier_splice.parse_shipped_body` + `build_archive` on the cl2 body's OWN carrier
codes rebuilds `archive.zip` to sha `08ec85333d13d71344b4482cf261e3b2d508725e49f3ca05971265a81498ad4e`
at exactly **179,982 B** — byte-identical. The carrier splice is therefore byte-anchored on
this body, so any later byte delta is attributable to the re-solve and not to the rebuild.

### 3a. The one place the instrument is NOT the receiver, and why it does not leak

`cpr1/inflate.py:312` sets `semantic_batch = 8 if cuda else 1`, and `ddm_up2` sec.6 MEASURED
that batch shape as byte-changing on this half (1,326 pixels move by ±1 through the
`clamp/round`). The contest-CUDA row is therefore scored on batch-8 frames, while this
instrument renders at batch 1 — the only shape that reproduces the CPU decode it is
controlled against.

That gap is not hidden, it is the residual: this instrument reads 0.00020132277 where the
T4 row reads 0.00020139, i.e. **−0.033%**, and the batch-shape effect is already inside
that number. Every projection below carries the seg leg onto T4 through the SAME-INSTRUMENT
ratio 0.00020139 / 0.00020132277, never by quoting an advisory number as a T4 one. Using
jg5's ancestor ratio instead (0.00030309 / 0.00030307) would misprice by 2.8e-4 relative =
5.6e-6 in score units — a quarter of the 2e-5 admission bar, which is why this arm computes
its own ([[binding-instruction-numbers-expire-and-nobody-rederives-them]]).

## 4. Operating record (honest)

* **RSS under-declaration.** I declared 14 GiB for 5 shards; MEASURED per-shard RSS at
  `--batch 8` is 5.0–5.5 GB, so the real footprint is ~25 GiB. The system admission gate
  then correctly refused every 6th process at 120.6 GiB used (pc1 holds 4 × 4.6 GB). The
  refusal is information, not an obstacle: the exact-byte pricing runs after pass 2a frees
  its memory. Next launch of this family declares 5.5 GiB per shard.
* **Throughput MEASURED:** 1.40–1.77 s per realized evaluation per shard at 3 torch threads
  under contention, 5 shards → ~3.6 evals/s aggregate; ~1.8 pairs/min over the field.

### 4a. MEASURED CPU + memory footprint of one pass shard (for the governor's model)

MAIN asked for this after accepting, by decision, that this arm's shards plus pc1's solvers
would slow md3's Metal cell to 1–2 steps/min. TWO `ps` samples, minutes apart, ~51 min into
pass 2a on an 18-core / 128 GiB box — reported as a range because one sample of a sawtooth
allocator is not a footprint:

| | procs | CPU each | CPU sum | RSS each | RSS sum |
|---|---|---|---|---|---|
| **ddm_sj1 pass shard** (`--threads 3 --batch 8`), sample A | **5** | 46.5–113.4%, mean **95.8%** | 479% ≈ 4.8 cores | 3,420–5,068 MB, mean 4,509 MB | **22.5 GB** |
| **ddm_sj1 pass shard**, sample B | 5 | 49.2–228.3%, mean **133.7%** | 668% ≈ 6.7 cores | 2,588–5,671 MB, mean 3,866 MB | **19.3 GB** |
| ddm_pc1 solver (context, sample A) | 8 | 42.6–71.7%, mean 59.7% | 478% ≈ 4.8 cores | 3,386–5,283 MB, mean 4,377 MB | 35.0 GB |

System at sample A: **load average 19.99 on 18 cores**, 915 free pages (14 MB) with
35.5 GiB inactive/reclaimable.

**Constant for the governor to model:** one shard of this family at `--threads 3 --batch 8`
costs **~1.0–1.3 cores and 2.6–5.7 GB RSS (mean ≈ 4.2 GB, peak ≈ 5.7 GB)**. Declare on the
PEAK, 5.7 GiB, not the mean — the allocator sawtooths with the in-flight batch.

Two corrections to what the launch declared, both mine:

1. I ran **five** shards, not six — the governor refused nine at a declared 36 GiB and I
   dropped to five after `pgrep -f ddm_pc1` showed pc1 active (the charter caps me at ≤ 6
   while pc1 runs).
2. I declared **14 GiB** for those five. The MEASURED peak is **22.5 GB** — a 1.6×
   under-declaration. I had extrapolated 2.8 GiB/shard from a `--batch 4` smoke; RSS
   scales with the in-flight batch, so a declaration must be taken at the batch the run
   will actually use, on a PEAK sample. The governor's later refusals of a sixth process
   at 120.6 GiB used were therefore CORRECT — my declaration was the wrong number, not
   the gate.
3. **Waiter hygiene.** I armed eight artifact-bound waiters on the pass-2a receipt and each
   one resumed me with no new information, which is the #1121 orphan-waiter genus wearing a
   different hat: bounded, artifact-bound loops still re-invoke the caller on expiry. All
   eight are stopped; only the launcher supervisor (which WRITES the receipt) remains on my
   side. The cure is one watcher at the coordinator, not N at the arm.

---

*(Sections 5+ — the n600 pass table, the persistent partition, admission, exact ΔS — are
appended as each lands. Nothing is written here before it is measured.)*

---

## Frontier line

`rc1 S 0.14666350774473783 @ 178,249 B [contest-CUDA T4 n600]` (live pointer, sha `1438049e3655fbcf…`)

Lineage: fs2 0.14784474152757654 @ 180,023 B → cl2 0.14781744131049854 @ 179,982 B → rc1 (above).
