# ddm_sj1 — multi-pass token PRE-DISTORTION to convergence on the cl2 frontier body

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

## 4. Operating record (honest)

* **RSS under-declaration.** I declared 14 GiB for 5 shards; MEASURED per-shard RSS at
  `--batch 8` is 5.0–5.5 GB, so the real footprint is ~25 GiB. The system admission gate
  then correctly refused every 6th process at 120.6 GiB used (pc1 holds 4 × 4.6 GB). The
  refusal is information, not an obstacle: the exact-byte pricing runs after pass 2a frees
  its memory. Next launch of this family declares 5.5 GiB per shard.
* **Throughput MEASURED:** 1.72–1.77 s per realized evaluation per shard at 3 torch threads
  under contention (load average ~30 on 18 cores), 5 shards → 2.87 evals/s aggregate.

---

*(Sections 5+ — the n600 pass table, the persistent partition, admission, exact ΔS — are
appended as each lands. Nothing is written here before it is measured.)*

---

## Frontier line

`cl2 S 0.14781744131049854 @ 179,982 B [contest-CUDA T4 n600]`
