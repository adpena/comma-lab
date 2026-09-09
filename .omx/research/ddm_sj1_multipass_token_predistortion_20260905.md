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

## 0b. Object change #2 — the pointer is pc1's V3, and the CARRIER lattice moved

MAIN moved the pointer again, this time to `ddm_pc1`'s V3 built on rc1's body:
**S 0.1451981569076111 @ 176,448 B**, sha `891add546f5cf0943929b566f29dd4318f1d8b2ab76ae05183d8189098880f40`,
tree `/Volumes/VertigoDataTier/pact/ddm_pc1_pose_carrier_efficiency/retained/v3_on_rc1_candidate_runtime/`.
The coefficients are re-quantised on a lattice coarsened ×4 and RE-SOLVED for all 600
pairs: d_pose 6.14e-06 → **5.73e-06**, carrier section 22,031 → **20,230 B** (−1,801).

| section | rc1 | pc1 V3 | identical |
|---|---:|---:|---|
| hpac (model) | 12,343 | 12,343 | **yes** |
| semantic (renderer weights) | 30,246 | 30,246 | **yes** |
| tail (token stream) | 113,515 | 113,515 | **yes** |
| carrier | 22,031 | 20,230 | no — ×4 lattice, re-solved |
| header | 14 | 14 | no |
| **archive** | **178,249** | **176,448** | −1,801 B |

MEASURED before anything was rebased onto it:

* pc1's codes read **absmax 542 / absmean 129.4** against cl2's 2,048 / 515.0 — exactly
  the ÷4 projection — and pc1's `coefficient_scales` are **exactly ×4** cl2's on all 12
  coordinates.
* The **container is still signed int12**: pc1 clips to `(-2048, 2047)` like everything
  else (`ddm_pc1_pose_carrier_efficiency.py:1041`), so "10-bit" names the OCCUPIED range,
  not a new bound, and `br1.realize`'s clamp is still the correct one. That mattered —
  a coarser *container* would have silently let the GN solve escape the alphabet.
* `up3.parse_shipped_body` + `build_archive` rebuild pc1's body from its own codes to sha
  `891add546f5cf094…` at exactly **176,448 B**.
* `100·0.00020139 + √(10·5.73e-06) + 25·176448/37545489` = 0.14519815690761112, matching
  MAIN's quoted value to the last digit.
* hpac, semantic and tail byte-identical to rc1 ⇒ renderer, token field and every seg
  number above carry over. **Pass 2a needed no restart, again.**

### The silent-revert trap, and the structural cure

MAIN named the failure precisely: a carrier re-solve seeded from cl2's int12 codes on
cl2's scales would **revert the whole V3 move while looking like it succeeded**. Nothing
downstream could catch it — every such code is a perfectly valid int12, so no container
check, no parse-back and no byte count would fire. The number would simply be wrong.

The cure is not to remember: `assert_carrier_is_pointer()` hashes the carrier runtime's
`archive.zip` and refuses unless it is the live pointer's. Both carrier entry points in
the joint half call it before `load_carrier_state`. VERIFIED: it accepts pc1's tree and
REFUSES cl2's and rc1's. The re-solve therefore starts from pc1's coefficients on pc1's
lattice by construction, because `load_carrier_state` reads both the codes and the scales
out of the section bytes it is pointed at.

**Split of trees, deliberate:** this arm RENDERS and measures d_seg on cl2's tree (whose
semantic-section coding jg1's loader understands) and takes the CARRIER, the build and the
seal from the pointer tree. That is not mixing two bodies — the three sections involved
are byte-identical objects — it is the same discipline MAIN prescribed for the token
stream: read each section from the tree whose codec the instrument speaks.

## 0c. Object change #3 — pc1's ×8 rung, and the pointer becomes a checkable row

MAIN moved the pointer a third time inside one session, to `ddm_pc1`'s ×8 lattice rung:
**S 0.1445177913121716 @ 175,576 B**, sha `f7e0bb793645894b2f6885fca82b98cab3067837bd66181e222f3d4b1f43e1ff`,
tree `…/ddm_pc1_pose_carrier_efficiency/retained/v3x8_on_rc1_candidate_runtime/`. Carrier
coefficients re-quantised ×8 and re-solved for all 600 pairs: d_pose 5.73e-06 → **5.58e-06**,
carrier 20,230 → **19,358 B** (−872).

MEASURED before adoption, same drill as ×4: hpac / semantic / tail byte-identical to the
×4 rung; codes **absmax 275, absmean 64.9** against cl2's 2,048 / 515.0; `coefficient_scales`
**exactly ×8** cl2's on all 12 coordinates; every code still inside signed int12; up3 rebuilds
the body from its own codes to `f7e0bb793645894b…` at exactly **175,576 B**; and
`100·0.00020139 + √(10·5.58e-06) + 25·175576/37545489` = 0.1445177913121716 exactly.
**Pass 2a needed no restart for the third time.**

### The recurrence is the finding: a pointer is a ROW, not five constants

Three pointer moves landed on this arm in one session and MAIN signalled a fourth (×16).
Re-editing five scattered constants by hand each time is exactly the shape that goes
half-applied and is never re-derived ([[binding-instruction-numbers-expire-and-nobody-rederives-them]]).
So the pointer is now a `PointerRow` in an ordered `POINTER_LINEAGE`, and the live pointer
is `POINTER_LINEAGE[-1]` — one row to add, nothing to keep in sync:

| row | S | bytes | d_pose | what moved |
|---|---|---:|---:|---|
| fs2_base | 0.14784474152757654 | 180,023 | 6.14e-06 | — |
| cl2_lambda1_repack | 0.14781744131049854 | 179,982 | 6.14e-06 | HPAC prior re-fit |
| rc1_model_section_recode | 0.14666350774473783 | 178,249 | 6.14e-06 | both MODEL sections, lossless |
| pc1_v3_lattice_x4 | 0.1451981569076111 | 176,448 | 5.73e-06 | carrier lattice ×4, re-solved |
| pc1_v3x8_lattice_x8 | 0.1445177913121716 | 175,576 | 5.58e-06 | carrier lattice ×8, re-solved |
| pc1_v3x16_lattice_x16 | 0.14411787458634504 | 174,786 | 5.77e-06 | carrier lattice ×16, re-solved |
| **sj1_token_predistortion_joint (LIVE)** | **0.1398140172839628** | **180,904** | **5.4e-06** | **token pre-distortion + carrier re-solve — the first row whose SEG leg moved** |

Two checks now stand between this arm and a wrong number, and both were tested to BITE:

* `PointerRow.verify_arithmetic()` runs on **every row at import** and refuses unless the
  declared S recomputes from its own three legs. All five rows pass. VERIFIED to fire on a
  one-byte change to `archive_bytes` and on a one-ULP-ish change to `d_pose`.
* `assert_carrier_is_pointer()` hashes the carrier runtime's `archive.zip` before any
  `load_carrier_state`. VERIFIED: accepts the live tree, refuses cl2, rc1 and pc1-×4 on the
  sha, and refuses a tree that is not on disk rather than raising a bare `FileNotFoundError`.

Together they make the fourth pointer move a one-row edit that cannot land half-applied.

## 0d. Object change #4 — ×16, the pose leg rises, and the row design pays for itself

**S 0.14411787458634504 @ 174,786 B**, sha `1de6c5d7186a0b31e5cc085bb6d2baab8275ee0d9de4d509f4d8add13695a629`,
tree `…/retained/v3x16_on_rc1_candidate_runtime/`. Carrier 19,358 → **18,568 B** (−790).

This rung is the first where **the pose leg ROSE**: d_pose 5.58e-06 → 5.77e-06. It is
admitted on the EXCHANGE, not on the leg — MEASURED: pose **+1.2611e-04 S** against rate
**−5.2603e-04 S**, net **−3.9992e-04 S**, twenty times the 2e-05 bar. Worth naming because
a reader scanning the lineage for a monotone pose column would read this row as a
regression; it is a priced trade, and the ladder is now past its pose knee (MAIN reports
×32 is not pre-registered as paying).

Verified before adoption, same drill: hpac / semantic / tail byte-identical to ×8; codes
**absmax 142 / absmean 32.6** — half the ×8 rung's, as an ×2 coarsening should give;
`coefficient_scales` **exactly ×16** cl2's on all 12 coordinates; codes inside signed
int12; up3 rebuilds from its own codes to `1de6c5d7186a0b31…` at exactly **174,786 B**;
S recomputes to the last digit. **Pass 2a needed no restart for the fourth time.**

**The row design paid for itself immediately.** Adopting this pointer was ONE appended
`PointerRow` — no other edit anywhere. All six rows passed the import-time arithmetic
self-check, and `assert_carrier_is_pointer` accepted the new tree and REFUSED all five
predecessors, including the ×8 rung that had been live minutes earlier. That refusal is
the whole point: the ×8 tree was the correct answer one message ago, which is exactly the
condition under which a remembered constant silently ships the wrong one.

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

## 5. PASS 2a — the measured n600 row (`gt` slice of the 36-family)

| quantity | MEASURED |
|---|---|
| flips before → after | 23,749 → **14,156** |
| flips repaired | **9,593 = 40.39%** |
| tokens changed | **7,804** |
| **cells per changed token** | **1.229** |
| break-even bits per changed token | **12.52** |
| d_seg (instrument, DALI) | 0.00020132277 → **0.00012000190** |
| seg gain | 12,213 B-equivalent = **0.008132 S** |
| pairs edited | 600 / 600 |
| moves accepted | 7,806 |
| proposals enumerated / realized | 96,764 / 68,782 |
| realized evaluations | 69,382 |
| wall clock | 16,580 s on 5 shards |

Field sha `e107b5ab9701f7cc…`.

**Against the charter's PRIOR-LAW PREDICTION** (`20–35%` of remaining flips, `4,750–8,300`
cells, at `1.3–1.6` cells per changed token): the repair fraction **BEAT** the band —
**40.39%, 9,593 cells** — while the efficiency came in **BELOW** it at **1.229**. Both
residuals point the same way: this actuator finds more repairable sites than predicted and
pays slightly more tokens for each. And this is the `gt` slice alone — 9 of the 36 combos.
The charter's FALSIFIER (a full pass repairing < 8%) is cleared by **5.0×**.

Accepted moves by offset (all `gt` role), which is a physical signature not a curiosity:

| offset | accepted | flips repaired |
|---|---:|---:|
| (−1, 0) | 2,099 | 2,900 |
| (1, 0) | 1,552 | 1,900 |
| (0, −1) | 1,105 | 1,282 |
| (0, 1) | 880 | 991 |
| (0, 0) | 528 | 735 |
| (−1,−1) | 473 | 514 |
| (1, −1) | 456 | 488 |
| (1, 1) | 365 | 403 |
| (−1, 1) | 348 | 380 |

The two VERTICAL neighbours carry **46.8%** of all accepts and **50.1%** of all repairs:
a dashcam's class boundaries run mostly horizontally, so the productive move is to push
the painted boundary UP or DOWN. The centre cell is only 6.8% of accepts, confirming from
the other side that the debt is render→re-segment loss and not a wrong stored label.

### The rc=1 that was mine

The pass receipt returned rc=1 with every shard at 120/120 and a clean per-shard receipt.
The failure was `KeyError: 'local_repaired'` in `cmd_pass_merge`: the engine records the
realized whole-pair repair as `repaired`, and `local_repaired` was a leftover name from the
windowed attribution §2's probe had already falsified. The merge is idempotent and re-ran
from the shard receipts, so no compute was lost — but a summary step that reads a field the
producer stopped writing is a real defect, and it only surfaced after 4.6 h of shard time.
Fixed at the reader.

## 6. EXACT pricing of the pass-2a field, and the pose damage (MEASURED)

### 6a. The encoder control — the thing that makes every byte below a number

`ddm_jg2 --stage control` re-encoded the UNEDITED field through the shipped decode
trajectory and emitted **113,419 B, byte-identical** to the shipped token stream
(`prefix_bytes_matching = 113,419`, `full_run = true`). The encoder inverts the shipping
decoder, so the deltas below are the pointer's own bytes and not a look-alike's.

### 6b. The rate cost — MEASURED by real re-encode, never by −log2 p

| quantity | MEASURED |
|---|---|
| token stream, base → candidate | 113,419 → **119,497 B** |
| **stream delta** | **+6,078 B** |
| tokens changed | 7,804 |
| **bits per changed token (marginal, realized)** | **6.2307** |
| jg1's modelled bits per changed token | 4.718 |
| realized / modelled | **1.321** |
| ΔS rate | **+0.004047090717076558** |

Break-even was **12.52 bits per changed token**. The realized cost is **6.23** — the pass
pays with a **2.01× margin**, spending less than half its budget. (An interim read on the
first 25 pairs gave 7.42 bits/token, i.e. the prefix OVERSTATED the cost by 19%; the
prefix-bias law holds here and, unusually, in the conservative direction.)

**The seg/rate composition, exact on both legs:**

| leg | ΔS |
|---|---|
| seg (9,593 cells repaired) | **−0.008132086859809028** |
| rate (+6,078 B) | **+0.004047090717076558** |
| **net before pose** | **−0.004084996142732470** |

That is **204×** the 2e-05 admission bar.

### 6c. The pose damage — why the re-solve is not optional

| leg | MEASURED |
|---|---|
| base d_pose (x16 carrier, base renders) | **5.7675e-06** |
| the x16 seal's own d_pose | 5.77e-06 → **ratio 0.99957** |
| stale d_pose (x16 carrier, CANDIDATE renders) | **2.482268e-03** |
| damage factor | **430.4×** |
| pairs damaged | **597 / 600** |
| pose leg | 0.00759441 → 0.15755214 = **+0.149958 S** |

The base control anchors the pose half on the live pointer to **−0.043%**. The damage
factor reproduces jg4's independently measured ×387 on a different body — same mechanism,
same order: frame 2p is a photometric probe solved against the ORIGINAL frame 2p+1, so
editing tokens strands every edited pair's carrier by construction. Un-resolved, the
+0.14996 S pose damage dwarfs the −0.00408 S seg/rate gain by 37×. The carrier re-solve is
the whole composition, not a polish step.

## 7. The carrier re-solve, the admission, and the composed candidate (MEASURED)

### 7a. Re-solve — the composition holds, and then some

`jg5.refine_pair` verbatim, on the candidate's OWN renders, starting from the ×16
pointer's coefficients on its lattice (`assert_carrier_is_pointer` checked the tree at
load). 600/600 pairs, 6,300 s on 6 shards, **4,622 of 7,200 coordinates changed**.

| leg (matched batch shape, `cpu_torch`, n600) | d_pose | pose leg |
|---|---|---|
| base (×16 carrier, base renders) | 5.767500e-06 | 0.00759441 |
| stale (×16 carrier, CANDIDATE renders) | 2.482268e-03 | 0.15755214 |
| **resolved** | **5.398060e-06** | **0.00734715** |

The re-solve does not merely recover the 430× damage — it lands **0.9359× the base**, a
pose **GAIN of −0.00024726 S**, with 339/600 pairs below their base value. The reason is
structural, not luck: the ×16 carrier was solved against the BASE frame 1, and this solve
sees the frame 1 the candidate actually ships. Every stop was PHYSICAL — 552
`no_improving_step`, 20 `lattice_floor`, 28 `converged_below_materiality_floor`, and
**zero** iteration-budget backstops, so the stopping rule bound on every pair.

The solver's own `final_d_pose` values were re-measured at the base leg's batch shape
before being used, per [[batch_shape_is_part_of_the_forward_instrument_20260806]]; the
matched re-measurement is the number above.

### 7b. Admission — the per-pair rate leg is MEASURED, not apportioned

The first sweep priced each pair's rate at a uniform 0.7788 B per changed token. That is
the average-for-marginal substitution `token_rate_model_direction_dependence_v1` warns
about, and the encodes had already written the honest answer: two per-frame bit ledgers
whose difference is each pair's marginal cost under the shipped model. Their sum is
**6,078.76 B** against the exact stream delta of **6,078 B** — 0.013% — so the
decomposition is faithful. The sweep now consumes the ledgers.

| subset | pairs | d_seg (T4-carried) | d_pose | bytes | S |
|---|---:|---|---|---:|---|
| drop everything (anchor) | 0 | 2.01390e-04 | 5.767500e-06 | 174,786 | 0.1441162286 |
| **full edit set** | **600** | **1.20042e-04** | **5.398060e-06** | **180,864.8** | **0.1397817675** |
| sweep optimum | 566 | 1.22153e-04 | 5.296566e-06 | 180,573.6 | 0.1397296788 |

The anchor row reproduces the pointer's 0.14411787 to 1.7e-06 — that gap is the pose
instrument's own −0.043% residual, carried openly rather than absorbed.

**I sealed the FULL 600, not the sweep optimum**, and the reason is a measurement fact
rather than a preference: the 600-set's rate leg is EXACTLY measured (+6,078 B, from a
real re-encode whose control was byte-identical), while the 566-subset's is a SUM of
per-pair deltas taken along the full-edit coder trajectory. Dropping 34 pairs changes the
context the coder carries into every later frame, so that sum is an estimate, not the
subset's stream. Its 5.21e-05 advantage is real but must be bought with its own encode
pair; it is recorded here as an available follow-on, not claimed.

### 7c. The composed candidate

| | value |
|---|---|
| candidate archive | **180,904 B**, sha `42aa84b59f71d83b8f11a26c635a7af8f32dcfdf183e3fea4bb2007e74a5f2f8` |
| delta vs pointer | **+6,118 B** = 6,078 (token tail) + 40 (re-solved carrier) |
| identity control | **PASS** — the tail-staged body rebuilds from its own codes to `180c64ac…` |
| frame-1 section identity | **PASS** — hpac, semantic and tail byte-identical; only the carrier moved |
| d_seg (instrument → T4-carried) | 0.00012000190 → 1.20042e-04 |
| d_pose | 5.398060e-06 |
| **projected S** | **0.13980789447084238** |
| **net ΔS vs pointer** | **−0.004309980115502654** |

Every leg is measured on this body: seg by realized argmax through the receiver's own
renderer, pose by the frozen CPU-torch PoseNet on the DALI table, bytes by a real
re-encode with a byte-identical control. `score_claim=false` until MAIN fires T4.

## 8. SEALED — `ddm_sj1_token_predistortion_joint`, contest-CUDA

`SEAL_ddm_sj1_token_predistortion_joint_contest_cuda.json`, seal sha
`6a4eb2a953989d130a539954420aa78321ecdcf2ec21837fd610c4e327dca9a5`, **SEAL_VALID**.
Archive **180,904 B** sha `42aa84b59f71d83b8f11a26c635a7af8f32dcfdf183e3fea4bb2007e74a5f2f8`;
runtime 43 files, 924,722 B, digest `4b871196ebc653ce…`.

| leg | pointer (x16) | candidate | ΔS |
|---|---|---|---|
| seg (T4-carried, same-instrument ratio 1.0003339539) | 2.0139e-04 | **1.2005045391e-04** | **−0.00813395** |
| pose (`cpu_torch`, DALI, n600) | 5.77e-06 | **5.398060e-06** | **−0.00024890** |
| rate | 174,786 B | **180,904 B** | **+0.00407373** |
| | 0.14411787458634504 | **0.1398087424644421** | **−0.0043091321219029255** |

215× the 2e-05 admission bar.

**The seg leg is measured on the SHIPPED bytes,** not on the encoder's field: SegNet argmax
over the candidate's own parse-back `0.raw` reads **0.0001200103759765625 = 14,157 flipped
cells**, against the pass ledger's predicted 14,156 — a **one-cell** difference (+0.0071%).
That single cell is the residual/corrector path the full inflate walks and the batch-1
re-render does not; it is carried, not absorbed.

**Every gate that stands behind the row:**

| gate | result |
|---|---|
| encoder control (unedited field) | **byte-identical**, 113,419 B |
| determinism twin (2 independent encodes) | **byte-identical** stream `c97c78c3…`, equal `code_bits` |
| carrier identity control | **PASS**, rebuilds to `180c64ac…` |
| frame-1 section identity | **PASS**, only the carrier moved |
| receiver decode identity | **PASS**, `decoded_field_matches_admitted = true` |
| full CPU inflate | **2,101.7 s**, `0.raw` sha `5aa5ffe5…`, 3,662,409,600 B |
| public entrypoint `bash inflate.sh` | reached the receiver's CUDA gate in **1.363 s** |
| pose instrument vs the x16 seal | **0.99957** |
| seg instrument vs the T4 base | **−0.033%** |

Nine falsifiers are pre-registered in the seal, including the one this arm cannot supply:
the report-8dp bound is deliberately NOT a hand-typed number, because this arm does not
hold the x16 base row's auth-eval receipt path. MAIN composes that two-row sentence with
`tools/report_8dp_delta_bound.py` once the candidate row lands — bounds ADD for a delta,
so the margin is judged against base + candidate, never against one row.

`score_claim=false`, `promotable=false` until MAIN fires T4.

## 9. FIRED AND PROMOTED — the row

`ddm_sj1_token_predistortion_joint` fired on contest-CUDA T4 and PROMOTED:
**S 0.1398140172839628 @ 180,904 B**, d_seg **0.00012009**, d_pose **5.4e-06**,
call `fc-01M1T6TCW2JS1JEW5CSZH3FVBY`, lane `ddm_sj1_t4_token_predistortion_joint_20260906`.
The pointer is now this arm's own tree, `…/candidate/candidate_runtime/`, sha `42aa84b5…`.

**Net against the ×16 pointer it replaced: −0.0043038573023822 S**, and it is the first row
in this lineage whose SEG leg moved — the six rows before it all held d_seg at 0.00020139
and bought bytes.

### Prediction vs measurement — the calibration this arm owes

| | projected | MEASURED on T4 | residual |
|---|---|---|---|
| S | 0.1398087424644421 | **0.1398140172839628** | **+5.2748e-06 (+0.0038%)** |
| d_seg | 1.2005045391e-04 (14,157 cells) | 0.00012009 (14,166 cells) | **+9 cells** of 117,964,800 |
| d_pose | 5.398060e-06 | 5.4e-06 | +1.32e-06 S |
| bytes | 180,904 | 180,904 | 0 |

The residual decomposes exactly: +3.955e-06 S of seg and +1.32e-06 S of pose, summing to
the +5.2748e-06 observed. So the advisory instrument called the contest row to **four
significant figures**, and the whole error is nine argmax cells plus a pose print. Both
printed legs reproduce the reported score to 1e-16, so this row needs no rounding
allowance — the score and its legs are mutually consistent.

Three instrument residuals are now measured end to end on this body and all point the same
way — the advisory rig reads very slightly OPTIMISTIC on distortion:

* seg instrument vs the T4 base row: **−0.033%**
* pose instrument vs the ×16 seal: **−0.043%**
* composed projection vs the fired row: **−0.0038%**

That is the honest calibration to carry into the successor: expect the T4 row to land a few
parts in 10⁵ ABOVE the projection, not below.

## 10. PASS 3 — the convergence loop still says CONTINUE

600/600 pairs, 4 shards, 15,041 s. Coverage verified (600 rows, 600 distinct pairs).

| quantity | pass 2a | **pass 3** |
|---|---|---|
| flips before → after | 23,749 → 14,156 | **14,157 → 12,710** |
| flips repaired | 9,593 = **40.39%** | **1,447 = 10.22%** |
| tokens changed THIS pass | 7,804 | **1,339** |
| cells per changed token | 1.229 | **1.081** |
| break-even bits per changed token | 12.520 | **11.006** |
| d_seg | 0.00020132 → 0.00012000 | 0.00012001 → **0.00010774** |
| seg gain | 0.008132 S | **0.001227 S** |
| moves accepted | 7,806 | 1,339 |
| proposals enumerated / realized | 96,764 / 68,782 | 59,693 / 56,161 |

Cumulative against the original body: **23,749 → 12,710 flipped cells**, d_seg
0.00020132277 → **0.00010774400**, on 9,121 changed tokens.

**The convergence rule says CONTINUE.** The stop is "a pass repairs < 1% of remaining
flips"; pass 3 repaired **10.22%**, ten times that. The marginal case is still healthy but
it IS decaying, and both numbers move together: the repair fraction fell 40.39% → 10.22%
(3.95×) while efficiency fell only 1.229 → 1.081 cells/token (1.14×). So the loop is
running out of SITES, not out of leverage per site — each remaining site is nearly as
repairable as before, there are simply far fewer of them. On that shape pass 4 is worth
one round: at 1.081 cells/token the break-even is 11.0 bits against a pass-2a marginal of
6.23, and the rate only has to stay under roughly 1.8× its previous cost to keep paying.

### A second merge bug of the same genus, found and fixed

The merge reported pass 3's break-even as **1.616 bits/token**. That is a per-pass
numerator over a CUMULATIVE denominator — 1,447 repairs divided by the 9,121 tokens
changed since the ORIGINAL body — mixing two populations and understating the real figure
by **6.8×**. The correct value is 11.006 bits/token. This is the same shape as the
`local_repaired` defect in §5: a summary line reading a field whose meaning had moved
under it. Fixed by separating `tokens_changed_this_pass` from `tokens_changed_vs_base`
and naming the metric `break_even_bits_per_changed_token_this_pass`; both passes re-merged
from their shard receipts.

## 11. Pass-3 successor — exact bytes against the LIVE row

The control for this round did not need re-deriving: the live pointer's token stream IS
the pass-2a encode output, already on disk with its per-frame ledger. The control that
licenses this encoder at all — the ORIGINAL unedited field re-encoding byte-identical to
the shipped 113,419 B stream — was passed in the first round and still stands.

| quantity | MEASURED |
|---|---|
| live row's stream (banked control) | 119,497 B, sha `c97c78c3…` |
| pass-3 field, encode | **120,385 B**, sha `4c13538145f916d6…` |
| pass-3 field, twin encode | **120,385 B, sha `4c13538145f916d6…`, identical `code_bits`** |
| **stream delta vs the live row** | **+888 B** |
| tokens changed this pass | 1,339 |
| **bits per changed token (marginal)** | **5.3055** |
| break-even for this pass | 11.006 |
| **margin** | **52%** |

The marginal cost FELL from pass 2a's 6.2307 to **5.3055** bits per changed token. That is
the opposite of the intuition that later edits get dearer, and it has a mechanism: the
context model has already absorbed pass 2a's neighbourhood structure, so a pass-3 token
landing beside those edits is a less surprising symbol than the same token would have been
on the untouched field. The actuator gets cheaper as it goes, not more expensive.

| leg | ΔS |
|---|---|
| seg (1,447 cells repaired) | **−0.00122664** |
| rate (+888 B) | **+0.00059128** |
| **net before pose** | **−0.00063535** |

That is 32× the 2e-05 bar before the pose leg is composed.

Stale pose damage this round is **56.18×** (3.032640e-04 against the live row's own
5.398060e-06), with 433/600 pairs damaged — far milder than pass 2a's 430× over 597 pairs,
because only 1,339 tokens moved rather than 7,804. The re-solve is still mandatory: 0.0477 S
of un-resolved pose damage would swamp a 0.00064 S gain by 75×.

## 12. The subset question, SETTLED by measurement

Last round I sealed the full edit set and recorded the sweep optimum as an untested
follow-on, on the argument that a subset's rate is a SUM of per-pair deltas taken along the
FULL-edit coder trajectory and therefore an estimate. This round I measured it.

| field | stream | archive | ledger-sum PREDICTION | error |
|---|---:|---:|---:|---|
| full edit set (445 pairs) | 120,385 B | 181,792 B | — | — |
| **admitted subset (370 pairs)** | **120,225 B** | **181,632 B** | 181,612.4 B | **+19.6 B (+0.0108%)** |

Both encodes of the subset are byte-identical to their twins. **The ledger sum is a good
RANKER and a slightly OPTIMISTIC pricer** — it under-charged the real subset stream by 19.6
bytes, which is 1.3e-05 S: below the 2e-05 admission bar, but the same sign as every other
residual this arm has measured. So the sweep may choose on the ledger; the seal must not.

The subset still wins on EXACT bytes:

| candidate | exact bytes | d_seg (T4-carried) | d_pose | S | net vs live row |
|---|---:|---|---|---|---|
| full edit set | 181,792 | 1.07815e-04 | 5.263729e-06 | 0.1390845360 | −0.00072948 |
| **admitted subset** | **181,632** | **1.09139e-04** | **5.092802e-06** | **0.1389915600** | **−0.00082246** |

It trades a little seg (dropping 75 pairs' edits) for more pose and rate than it gives up —
those are the pairs whose carrier re-solve went badly, and dropping them recovers 1.71e-07
of d_pose.

### A third silent-revert class, caught before it shipped

The subset writer emitted only the ADMITTED pairs' planes. An edit npz is spliced onto the
ORIGINAL base field, so a pair merely ABSENT reverts all the way back — which on a
SUCCESSOR round undoes what the live pointer already banked. MEASURED here: **230
non-admitted pairs would have reverted 2,337 pass-2a tokens**, invisibly, because the
result is a perfectly valid token field that no byte count, parse-back or size check could
flag. Cured by making the subset CARRY the live row's plane for every dropped pair, with a
fail-closed check that every carried pair matches.

The same round surfaced a sibling: `stage-tail`'s baseline check asserted the pointer's
tail EQUALS the encoder tree's tail. True in round one (both trees held the same stream),
false the moment the pointer's tail became this encoder's own previous output. Replaced by
`--expect-pointer-stream`, which checks the pointer's tail suffix against the RETAINED
bytes it should carry — correct in both rounds, and stronger than the sibling comparison it
replaces.

That is three silent-revert defects in this family now: the carrier start codes (MAIN
caught it), the subset writer, and the tail baseline. All three share one shape — **a check
that encoded a round-one premise, still passing after the premise moved.**

## 13. SEALED — `ddm_sj1_token_predistortion_pass3`, contest-CUDA

`SEAL_ddm_sj1_token_predistortion_pass3_contest_cuda.json`, seal sha
`748e4737a2137ffbf11148360304da096e472b75ea92d92517bc65963528bc56`, **SEAL_VALID**.
Archive **181,645 B** sha `06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3`;
runtime 43 files, 925,463 B, digest `2435dab86725ed8e…`.

| leg | live row | candidate | ΔS |
|---|---|---|---|
| seg (T4-carried, ratio 1.0006634762) | 1.2009e-04 | **1.0913879636e-04** | **−0.00109521** |
| pose (`cpu_torch`, DALI, n600) | 5.398060e-06 | **5.0928018e-06** | **−0.00021076** |
| rate | 180,904 B | **181,645 B** | **+0.00047217** |
| | 0.1398140172839628 | **0.13900021608143795** | **−0.0008138012025248609** |

41× the admission bar. Bytes itemised: +728 token stream, +13 re-solved carrier.

**The seg leg matched the admission EXACTLY** — 12,866 flipped cells predicted, 12,866
measured on the shipped bytes, **zero cells of disagreement**. Round one carried a 9-cell
gap because its base was a batch-1 re-render; this round's base was itself a parse-back
measurement, and the gap closed completely. That is the instrument agreeing with itself
across a full encode → decode → re-segment round trip.

| gate | result |
|---|---|
| subset encode twin | **byte-identical**, 120,225 B |
| full-set encode twin | **byte-identical**, 120,385 B |
| subset priced by REAL encode, not ledger | 181,632 B measured vs 181,612.4 predicted |
| carrier identity control | **PASS**, rebuilds to `9157920b…` |
| frame-1 section identity | **PASS**, only the carrier moved |
| receiver decode identity | **PASS**, `decoded_field_matches_admitted = true` |
| full CPU inflate | **705.7 s**, `0.raw` sha `bfe96bac…`, 600 pairs, token cache DISABLED |
| public entrypoint `bash inflate.sh` | reached the CUDA gate in **0.835 s** |

Ten falsifiers pre-registered, including one that exists only because this round found the
defect: *non-admitted pairs must still carry the live row's edits* — if the fired d_seg
lands near the ORIGINAL body's 0.00020139 rather than near 0.00010914, the silent revert is
the first thing to look for.

### Where the campaign stands

| | d_seg | flipped cells | archive | S |
|---|---|---:|---:|---|
| body as this arm found it | 0.00020132277 | 23,749 | 179,982 (cl2) | — |
| pass 2a candidate (PROMOTED, move 31) | 0.00012009 | 14,166 | 180,904 | 0.1398140172839628 |
| **pass 3 candidate (PROMOTED, move 32)** | **0.00010913** | **12,866** | **181,645** | **0.13900437796841966** |

Cumulative: **45.8% of the flipped argmax cells this arm inherited are gone**, bought with
741 bytes on top of the first candidate's 6,118.

---

## 14. FIRED AND PROMOTED — pointer move #32

The pass-3 seal fired on T4 as call `fc-01M1TFD35EPY2YZHNV3VKJP6MG`, lane
`ddm_sj1_t4_token_predistortion_pass3_20260906`, and was promoted:

**S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]**, archive sha
`06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3`, −8.096393e-04 S
against move 31. The three legs move together and sum to exactly that delta:

| leg | move 31 | move 32 | ΔS |
|---|---:|---:|---:|
| seg | 0.00012009 | 0.00010913 | −1.0960e-03 |
| pose | 5.4e-06 | 5.1e-06 | −2.0704e-04 |
| rate | 180,904 B | 181,645 B (+741) | +4.9340e-04 |
| | | **total** | **−8.0964e-04** |

### Prediction vs measurement — the calibration, now with two points

| | projected | MEASURED | residual | rel |
|---|---|---|---:|---:|
| move 31 (pass 2a) | 0.1398087424644421 | 0.1398140172839628 | +5.2748e-06 | +0.0038% |
| move 32 (pass 3) | 0.13900021607682325 | 0.13900437796841966 | +4.1619e-06 | +0.0030% |

Both are OPTIMISTIC by the same small amount, and the decomposition differs:

- **seg came in BETTER than projected on move 32** — −8.796e-07 S, about 1.04 cells of
  117,964,800. On move 31 it came in worse by 9 cells. The seg instrument is therefore
  accurate to ±10 cells (±8.5e-06 S) on this body, in both directions.
- **pose came in WORSE both times** — +1.32e-06 S then +5.042e-06 S. The arm measures
  pose on CPU; T4 reads it on CUDA. This is a one-signed drift, not a print artefact,
  and it is the larger half of the residual.

The seg leg landed EXACTLY where the admission put it: **12,866 flipped cells predicted
on the shipped bytes, 12,866 measured, zero disagreement.**

---

## 15. Pass 4 — the CONTINUE/STOP arithmetic, PRE-REGISTERED

Written before anything is launched, so the decision cannot be back-fitted to the result.
Every input below is MEASURED (`passes/*/PASS_RESULT.json`, the exact re-encodes, and the
two banked T4 rows); only the pass-4 column is projected.

### The banked passes

| | sites | repaired | frac | tokens | cells/token | bits/token (EXACT encode) | break-even | margin | pose leg (T4) | pose leg (local) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pass 2a | 23,749 | 9,593 | 40.39% | 7,804 | 1.2292 | 6.2307 | 12.520 | 2.01× | −2.4758e-04 | −1.500e-04 |
| pass 3 | 14,157 | 1,447 | 10.22% | 1,339 | 1.0807 | 5.3055 | 11.006 | 2.07× | −2.0704e-04 | −9.199e-05 |
| ratio | | | 0.2530 | | 0.8791 | 0.8515 | | | 0.8363 | 0.6133 |

Two facts the table makes plain, and neither was assumed in advance:

1. **The marginal price FALLS pass to pass** — 6.23 → 5.31 bits per changed token. The
   context model absorbs the earlier pass's edits as structure, so the next pass's edits
   sit in a cheaper conditional. This is why the exchange has not closed even though the
   repair yield collapsed 4×.
2. **The price margin is stable** — 2.01× then 2.07× against break-even. That is the
   least risky input in the whole projection.

Break-even is not a constant: it is `cells_per_token × 1.27311 B/cell × 8`, so it falls
with the yield. Both banked passes reproduce it to 3 decimals, which is the check that
the formula, not a remembered number, is what is being used.

### The projection on the 12,866-cell residual

| quantity | projected | how |
|---|---:|---|
| repair fraction | 2.586% | 10.22% × 0.2530 |
| cells repaired | 332.8 | 12,866 × 2.586% |
| cells/token | 0.9500 | 1.0807 × 0.8791 |
| tokens changed | 350.3 | 332.8 / 0.9500 |
| bits/token | 4.5176 | 5.3055 × 0.8515 |
| break-even | 9.6760 | 0.9500 × 1.27311 × 8 → **margin 2.14×** |
| **stream delta** | **+197.8 B** | 350.3 × 4.5176 / 8, against the LIVE row's own stream |
| seg ΔS | **−2.8208e-04** | 332.8 × 8.477105e-07 |
| rate ΔS | **+1.3170e-04** | 197.8 × 6.658590e-07 |
| seg+rate | **−1.5038e-04** | |

The pose leg is projected four ways rather than one, because the arm has two disagreeing
estimators for it and one adverse precedent (the ×16 rung, where the leg ROSE):

| pose leg | net ΔS | basis |
|---:|---:|---|
| −1.7314e-04 | **−3.2352e-04** | central — T4-realized trend, ×0.8363 |
| −5.6414e-05 | **−2.0679e-04** | conservative — local CPU trend, ×0.6133 |
| 0 | **−1.5038e-04** | floor — the re-solve pays nothing |
| +2.0704e-04 | **+5.6660e-05** | adverse — the leg rises by pass 3's whole magnitude |

### The margin the rule must clear

`projection residual (worst banked) 5.2748e-06` + `pose 3-sig-fig print band 7.143e-06`
= **1.242e-05 S**. The print band is real: the T4 receipt carries `avg_posenet_dist` to
three significant figures, and the pointer's S is composed from that print, so half a
unit in the last place is 7.1e-06 S of unrecoverable quantisation at a pose near 4.9e-06.

### PRE-REGISTERED RULE and verdict

> **CONTINUE iff the predicted net ΔS is negative by more than 1.242e-05 S.**

- central −3.2352e-04 → **26.1× the margin**
- **floor (pose leg = 0) −1.5038e-04 → 12.1× the margin**

Even with the pose re-solve credited nothing at all, the seg/rate exchange alone clears
the margin by an order of magnitude. **Verdict: CONTINUE to pass 4.**

The single failure mode that flips the sign is the adverse pose row, and it is **caught
at admission, not at T4**: the admission prices the actual resolved pose before anything
is sealed. Pre-registered stop: *if the admission's total is not negative, no candidate
is built.*

### The convergence rule

The <1%-repaired-per-pass rule does **not** fire at pass 4 (projected 2.586%). Carrying
the same ratio one step further projects **0.654% at pass 5 — the rule fires there.**
So pass 4 is, on the current trend, the LAST single-cell pre-distortion pass this object
supports, and the residual it leaves is what the renderer door inherits.

### Cost, honestly

The search does **not** shrink with the yield. Proposals per flipped cell are *rising*
(4.074 → 4.217), so pass 4 enumerates ~54,200 proposals ≈ **0.91× pass 3's search** to
repair ~4.3× fewer cells. Pass 4 costs about what pass 3 cost and returns about a
quarter as much. That is still 12–26× the bar, but it is the honest shape of the tail:
the per-pass return is falling much faster than the per-pass cost.

---

## 16. The residual partition — a MEASURED census of the 12,866 cells

Measured on the SHIPPED object: `seg_final_pass3/argmax_n600.npy` (the parse-back argmax
of the promoted archive) against the DALI GT table, with the shipped token field
`admission_pass3/field_admitted.npz` (sha `192ec469…`, the one the encode receipt names).
Residual reproduces at exactly **12,866** cells. Everything below is numpy on artifacts
already on disk; nothing here is projected.

### 16a. Shape — a one-pixel boundary displacement, not a class hallucination

| | measured |
|---|---:|
| residual cells lying ON a GT 4-neighbour class edge | **12,824 = 99.67%** |
| GT boundary cells as a share of the frame | 2.163% |
| **enrichment** | **46.1×** |
| our predicted class present in GT within Chebyshev r ≤ 1 | **12,812 = 99.58%** |
| confusion symmetry index (1 = perfectly reciprocal) | **0.9049** |

The residual is the argmax boundary sitting one pixel off. Road→Lane 2,710 against
Lane→Road 2,924; Undrivable→Movable 1,401 against Movable→Undrivable 1,080. A class
collapse would be one-sided; this is reciprocal jitter on a codim-1 curve.

### 16b. Where it lives

**Vertical band.** Rows 128–319 of 384 carry **100.00%** of it. Rows 0–127 (sky) and
320–383 (ego hood) carry **ZERO**. Row centroid 200.9, column centroid 271.0.

| row band | share | | column band | share |
|---|---:|---|---|---:|
| 0–127 | 0.00% | | 0–127 | 14.74% |
| 128–191 | 54.52% | | 128–255 | 26.92% |
| 192–255 | 34.83% | | 256–383 | 39.13% |
| 256–319 | 10.66% | | 384–511 | 19.21% |
| 320–383 | 0.00% | | | |

**Class**, in the comma10k canonical order (never luma-sorted):

| class | GT-side cells | share | frame area | over-representation |
|---|---:|---:|---:|---:|
| Road | 5,289 | 41.11% | 23.234% | 1.77× |
| **Lane** | **3,025** | **23.51%** | **0.586%** | **40.15×** |
| Undrivable | 2,388 | 18.56% | 49.517% | 0.37× |
| **Movable** | **1,740** | **13.52%** | **1.238%** | **10.92×** |
| MyCar | 424 | 3.30% | 25.426% | 0.13× |

Lane at 40× and Movable at 11× are the whole story; Undrivable and MyCar are 3–8× *under*
represented. This is the same lane-orbit long tail the campaign has measured elsewhere,
now read on the residual of a twice-pre-distorted object.

**Per pair — diffuse, not concentrated.** Mean 21.44, median 19, max 100, min 4, and
**zero pairs are clean**. The worst 100 pairs of 600 carry only **30.87%**. There is no
"fix the bad pairs" strategy available: the residual is spread over every pair.

### 16c. Granularity — the residual is 11,859 isolated specks

4-connected components, per pair:

| component size | components | cells | share |
|---|---:|---:|---:|
| 1 | 11,135 | 11,135 | **86.55%** |
| 2 | 559 | 1,118 | 8.69% |
| 3–4 | 145 | 467 | 3.63% |
| 5–9 | 17 | 104 | 0.81% |
| 10–24 | 3 | 42 | 0.33% |
| ≥25 | 0 | 0 | 0.00% |

**Mean component size 1.08 cells.** Pass 3's measured yield was **1.0807 cells per
changed token.** Those are the same number, and that is not a coincidence: the search
has converged to repairing exactly one connected speck per token it spends. There is no
remaining structure for a single move to catch two of.

### 16d. Why single-cell token moves cannot repair them

**(i) It is not unexplored ground.** Pass 3's per-pair rows record `sites_persistent`
summing to **12,710**, and `flips_after` summing to **12,710** — identical. The search
enumerated 4.217 proposals per flipped cell and **failed on 100% of what it left behind.**
Pass 4's projected yield therefore comes ENTIRELY from field-change re-opening — the
pass-3 edits moved the render, so some sites become newly reachable — and not from sites
the search has yet to visit. That is a real mechanism (it is what produced pass 3 from
pass 2a's residual) but it is a decaying one, and this measurement is why.

**(ii) The token is already right at 86.39% of them.** Against the SHIPPED field:

| | cells | share |
|---|---:|---:|
| stored token ALREADY equals GT | **11,115** | **86.39%** |
| stored token is wrong | 1,751 | 13.61% |
| …of which this arm deliberately set the lie | 388 | |
| …inherited from the base field, GT tried and rejected | 1,363 | |

At 86% of the residual there is nothing to correct *in the token* — the token says the
right class and the renderer's output at that cell still argmaxes wrong. The only lever
the token field has left is to LIE in a neighbour, and that is exactly the move family
the search has exhausted.

> **Provenance note, and a small correction to a shipped receipt.** The seg-final
> receipt's `flips_where_stored_token_already_equals_gt` reads **11,447**, not 11,115.
> Traced: that field is computed against `body.tokens` — the BASE field — because
> `step0` loads the base body by design. It is not the shipped object's number. The
> shipped archive stores the pass-3 admitted field, whose number is 11,115 / 1,751
> (86.39% / 13.61%). Same genus as
> [[available-field-vs-authoritative-field]]: the field exists, is correctly computed,
> and names a different object than its name suggests. Use 11,115 for the shipped row.

**(iii) The collateral ratio is the wall.** A token move perturbs the renderer's whole
influence footprint (r = 9 tokens, DERIVED from `cpr1/inflate.py`: coord_mix 1×1 +
depthwise 3×3 at dilations 1,1,2,4 + head 3×3). Counting what sits inside that footprint
around each residual cell:

| footprint | CORRECT boundary cells at risk | other residual cells in reach | ratio |
|---|---:|---:|---:|
| r = 1 (3×3) | 5.29 mean / 5 median | 0.225 mean, 17.17% have ≥1 | **23.5 : 1** |
| r = 9 (19×19) | 68.75 mean / 70 median, min 4 | 1.544 mean, 57.03% have ≥1 | **44.5 : 1** |

Acceptance is a composite re-render whose flip count must go DOWN. So every candidate
move is a bet that repairs ≥1 speck while breaking none of ~69 equally fragile correct
boundary cells sitting in the same footprint. Early passes won that bet often because the
board was full of specks; at 12,866 the correct-to-residual ratio inside the footprint has
risen to 44.5:1 and the bet is mostly lost. **This ratio, not the search, is what closes
single-cell pre-distortion.**

### 16e. What this hands the renderer door

The residual is not a token-coding problem. Stated as constraints on any successor:

1. **It is a renderer problem at 86.39%** — the correct token is already stored and the
   render still argmaxes wrong. A better token code cannot reach these cells.
2. **It is one-pixel boundary jitter (99.58%) on a codim-1 curve (99.67% on a GT edge,
   46.1× enriched), reciprocal (symmetry 0.905).** The renderer's boundary is in the
   right place to within a pixel and lands on the wrong side of it. The lever that fits
   that shape is sub-pixel boundary placement, not class capacity.
3. **It is Lane (40.15×) and Movable (10.92×)** and essentially nothing else. Undrivable
   and MyCar are already at 0.37× and 0.13×.
4. **It is confined to rows 128–319** — the horizon band. Two thirds of the frame
   contributes zero, so any capacity spent outside that band is spent on a solved region.
5. **It is atomised** — 86.55% isolated single cells, mean component 1.08, max 17. Any
   mechanism whose unit of repair is larger than ~1 cell pays for coverage it cannot use.
6. **It is diffuse across pairs** — every one of the 600 pairs carries some, worst 100
   carry 30.87%. Per-pair specialisation buys almost nothing.

The composition that fits all six is a renderer fold-back at optimal form that moves the
boundary sub-pixel in the horizon band on Lane and Movable edges, with token
pre-distortion composed on top of it to catch whatever single cells the better renderer
still leaves. The pre-distortion cannot lead; on this measurement it has ~12,866 cells
left and can reach a few hundred of them per pass at rising cost.

---

## 17. A FOURTH silent-revert route, closed at the resume surface

Found while pass 4 was in flight, checking whether the 6-hour walltime cap was safe to
hit. It was not.

**The mechanism.** `cmd_pass` appends a ledger row for every completed pair but rewrites
`planes_shard_N.npz` only every `--checkpoint-every` pairs (default 5). Any kill inside
that window — the cap, an OOM, an operator cut — leaves up to 4 pairs per shard with a
ROW and no PLANE. On resume, `done` is built from the rows, so those pairs are skipped
and never re-planned. At merge, a pair absent from the npz is simply absent from
`field_after.npz` — and **an absent pair does not fall back to the prior pass, it reverts
to `BODY_TOKENS`**, because every downstream consumer splices the npz onto the base. So
the loss is not this pass's edits for those pairs; it is *every banked pass's* edits.

**Why it is the dangerous shape.** The row ledger stays complete, so every printed number
stays right. DEMONSTRATED: dropping three planes from a pass-3 shard and re-merging
reproduces `PASS_RESULT.json` **field for field** — 1,447/14,157 repaired, 1.0807
cells/token, `d_seg` 0.00012001 → 0.00010774 — on a field that has silently lost three
pairs. The receipt cannot see it.

This is the same genus as the three already closed on this arm (the carrier silent
revert, the subset writer that spliced onto the ORIGINAL base, and the stage-tail
baseline) — a valid-looking artifact that quietly reverts banked work — now at the
resume/crash surface rather than a writer.

**The cure, fail-closed at both surfaces.**

1. `cmd_pass` REFUSES to resume when any `done` pair lacks a plane, naming both files to
   delete and redo. Guessing (re-running only the orphans, or carrying the prior plane)
   would be repair-by-approximation on an inconsistent state.
2. `pass-merge` REFUSES a merged field that omits any of the 600 pairs, reporting how
   many of the missing ones accepted moves. `--allow-partial-planes` is the documented
   escape and is legal ONLY for a first pass with no prior field, where an untouched pair
   legitimately has no plane.

**Proof both ways.** Re-merging pass 3 under the new checks reproduces `PASS_RESULT.json`
field-for-field (transparent on complete data); the positive-control resume with rows and
planes agreeing completes with an empty `todo` in 0.0012 s; and both falsifiers bite with
actionable messages. Landed at commit `3afe99635`.

---

## 18. The §17 refusal fires for real, 4 hours after landing

Pass 4 launched 2026-09-08T23:06:46Z and returned **rc = 120** at 12:48:12Z — 13.7 h wall
for what should have been 2.5 h. Not a run defect. MAIN traced it from `pmset -g log`:
the machine entered Low Power Sleep on battery at 18:53:45 −0500 for 46,467 s and woke
from hibernate on AC attach; at wake the external SSDs re-enumerated (Vertigo
disk6 → disk5) and every detached process on the fleet died in the same second. This
arm's receipts agree independently: `resource_safe_run_status.json` froze at
`elapsed_s` **2,793** / `2026-09-08T23:53:19Z` with `status: running`, while the
supervisor's own clock ran to 49,286 s. **A frozen sampler beside a running wall clock is
the hibernate signature**, and it is worth keeping: the run receipt says rc=120 with no
detail, and only the two clocks disagreeing say *why*.

**MEASURED, and a correction to my own declaration.** Peak RSS for 5 shards at batch 8
was **34,812 MiB (34.0 GiB)** — about **7.0 GiB per shard**, not the 5.7 GiB this arm
declared from the pass-2a measurement. The relaunch declares 36 GiB. The governor was
never at risk (ceiling 116 GiB), but the declaration was low and is now right.

**Integrity, checked before trusting anything.** A plane file written into a SIGBUS
window can be truncated, so every `planes_shard_*.npz` was fully decompressed rather than
merely opened: **170 planes, all `(384, 512)` uint8 with max < 5, zero malformed**, and
all 182 rows parse. Nothing was lost to corruption.

**And then the §17 refusal earned itself.** The shards were left with **182 rows against
170 planes — 12 orphaned pairs** (180, 175 / 191, 201, 181, 176 / 162 / 133, 153, 163,
193 / 179). Resuming without the check would have skipped all 12, and the merged field
would have reverted them to `BODY_TOKENS`, losing every banked pass's edits for them
**while the ledger read complete**. The check was written four hours earlier on the
theory that the walltime cap might fire; what actually fired was a hibernate. The genus
was right even though the trigger was not.

`pass-repair-shard` (commit `a1a582935`) then made the recovery **exact** rather than
total: it dropped precisely those 12 rows, kept UTC-stamped backups of the originals, and
left rows == planes == 170 verified consistent across all five shards. **Recovery cost 12
pairs of recompute instead of 182.**

---

## 19. PASS 4 — SEALED, and the pre-registration graded honestly

`SEAL_ddm_sj1_token_predistortion_pass4_contest_cuda.json`, seal sha `c887162d…`,
**SEAL_VALID**. Archive **181,521 B**, sha `b0ca809ce2c657df…`, runtime digest
`453d191e…`. Re-based onto pointer move 34 (pc2) and run ONCE against it.

### 19a. The n600 pass, against what was pre-registered at §15

| | projected | MEASURED | ratio |
|---|---:|---:|---:|
| repair fraction | 2.586% | **3.288%** | 1.271× |
| cells repaired | 332.8 | **423** | 1.271× |
| tokens changed | 350.3 | **415** | 1.185× |
| cells/token | 0.9500 | **1.0193** | 1.073× |
| bits/token (EXACT encode) | 4.5176 | **6.4000** | **1.417×** |
| break-even bits/token | 9.676 | **10.3812** | 1.073× |
| price margin | 2.14× | **1.62×** | 0.757× |

`d_seg` 0.00010907 → 0.00010548; flips 12,866 → 12,443. Twin encode byte-identical.

**The falling-price model is FALSIFIED.** Three points now — 6.2307, 5.3055, **6.4000** —
and the third went back above the first pass's neighbourhood. The story that the context
model absorbs earlier edits as structure does not survive it. The simpler account fits and
predicts a RISING price: the search takes the cheapest repairs first, so each later pass
buys progressively dearer tokens. On that reading pass 3's dip is the anomaly to explain,
not pass 4's rise. Pass 5's arithmetic must use a rising-price model.

### 19b. The subset sweep is what admitted this pass

| composition | seg ΔS | pose ΔS | rate ΔS | net | vs bar |
|---|---:|---:|---:|---:|---:|
| full 600-pair field | −3.585815e-04 | **+1.043752e-04** | +2.210652e-04 | −3.314120e-05 | 1.66× |
| **112-pair admitted subset** | −2.137476e-04 | **−3.525803e-05** | +9.854713e-05 | **−1.504585e-04** | **7.52×** |

Keeping only the pairs where the seg gain and the re-solve BOTH help flips the pose leg
from a **+1.04e-04 cost to a −3.53e-05 credit** — a 1.40e-04 swing, seven times the admit
bar, **from selection alone**. It also explains the full-field weakness: 488 of the 600
pairs were carrying pose damage that 171 marginal repaired cells did not pay for.

**Two of the three pre-registered inputs were wrong** — bits/token by 1.417× and the pose
leg by its SIGN. The CONTINUE call still stands on its own margin (−1.50e-04 against a
1.242e-05 threshold), but the reasoning offered for it was not the reasoning that made it
right, and what I called the "floor" case — *seg+rate with zero pose credit* — was **not a
floor**: pose went positive, which only the adverse row in §15's table contemplated.

### 19c. The base-pose instrument trap (the expensive near-miss)

`admit` composes pose PER PAIR — `pose = np.where(keep, resolved_pose, base_pose)` — so
`base_pose` supplies the actual pose of every NON-admitted pair, not a comparison constant.
A wrong base poisons the majority of the vector.

The first attempt measured it with `pose --tag base` and **no `--overlay`**, which pairs the
live carrier against the ORIGINAL odd frames — frames carrying none of three passes'
pre-distortion. It returned `d_pose` **2.559e-03, 500× the T4 print**, and would have shown
pass 4 as a **−0.153 S, 7,643×-the-bar** gain. **What caught it was the magnitude being
absurd, not any gate.** Nothing in the chain refuses a base measured on the wrong decode:
the number is a well-formed float from a correctly functioning instrument. At 2× instead of
500× it would have sealed.

Three borrowable bases were all inadmissible: the **T4 print** (different instrument, and a
scalar where a 600-vector is needed); **pass 3's local resolved pose** (different carrier —
pc2 re-solved 17 coordinates and set scales to 1.0); and **pc2's own `base_d_pose.json`** (a
scalar, from their `base/` rung rather than the shipped `rebase_scales_resolve/`). Even
`pose_pass3/overlay` is wrong: it renders the FULL 445-pair pass-3 field while the pointer
ships the 370-pair admitted subset, differing on 75 pairs.

The correct object — pc2's shipped codes on the renders of `admission_pass3/field_admitted.npz`
— measured **5.090165e-06**, which cross-checks against pc2's own scalar 5.09276404439735e-06
to **0.05%** and against the T4 print to 0.2%. Same genus as §16's receipt correction: a
field that exists, is correctly computed, and names a DIFFERENT OBJECT.

### 19d. Gates, all green on the shipped bytes

| gate | result |
|---|---|
| exact subset stream | 120,367 B, twin BYTE-IDENTICAL, **+142 B** vs the 120,225 control |
| ledger sum | predicted +136.30 B — **under-charged by 5.70 B** (pass 3: 181,612.4 vs 181,632). Sums RANK, encodes PRICE |
| stage-tail | 181,515 B, `tail_baseline_check: PASS` |
| close identity control | **PASSED** — body rebuilds byte-identically from its own codes |
| carrier splice | 112 pairs, 661 coordinates |
| candidate runtime | built FROM THE POINTER TREE, re-pinned `e138ee09…` → `b0ca809c…` @ 181,521 |
| parse-back | `decoded_field_matches_admitted: true`, receiver pin PASS, 881.9 s |
| **seg final on shipped bytes** | **12,614 cells predicted, 12,614 measured — zero disagreement** |
| public-entrypoint smoke | REACHED_TOKEN_DECODE 240.01/240.02 s; REACHED_CUDA_GATE 1.78/1.78 s rc=1, both roles |

Projected row **S 0.13867280587520867**, net **−1.504585e-04** vs pc2. Both banked
projections came in optimistic (+5.2748e-06, +4.1619e-06), so T4 is expected near
**0.13867781 ± 1.0e-05**; a residual outside that band is a new effect, and is
pre-registered as a falsifier rather than absorbed.

### 19e. Where the lever stands

The convergence rule does not fire (3.288% vs 1%), but one more step extrapolates to
**1.093% — essentially ON the bar**. And the residual is **not concentrating**: Gini
0.2542 → 0.2518, top-100 share 31.01% → 30.74%, still **zero clean pairs**, minimum still
4, and only **206 of 600 pairs** received any repair. Pre-distortion is skimming a thin
uniform layer off a floor it cannot reach — which is §16's census read from the other side.
Pass 5 is a genuine decision, not a formality, and on a rising-price model it looks
marginal.

---

## 20. PASS 5 — the CONTINUE/STOP arithmetic, PRE-REGISTERED (rising-price model)

Written while the pass-4 T4 row is in flight, before anything is launched. The model
changed because §19a falsified the old one: the marginal price RISES.

### 20a. Inputs (all MEASURED)

| pass | sites | repaired | frac | tokens | cells/tok | bits/tok | break-even |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2a | 23,749 | 9,593 | 40.393% | 7,804 | 1.2292 | 6.2307 | 12.5197 |
| 3 | 14,157 | 1,447 | 10.221% | 1,339 | 1.0807 | 5.3055 | 11.0063 |
| 4 | 12,866 | 423 | 3.288% | 415 | **6.4000** | 1.0193 | 10.3812 |

Ratios: repair fraction 0.2530 then **0.3217** (the decay is SLOWING); cells/token 0.8791
then 0.9432; bits/token last step **×1.2063 — rising**, per the cheapest-repairs-first
reading. The pose leg is taken as a **COST** in the central case (pass 4's full field
measured **+1.043752e-04**), with the subset sweep's selection as the only credit.

### 20b. Projection on the 12,614-cell shipped residual

repair fraction **1.058%** → 133.4 cells; cells/token 0.9614 → 138.8 tokens; price
**7.7203 bits/token** against a 9.7916 break-even (**margin 1.27×**, down from 1.62×);
133.9 B.

| composition | seg ΔS | rate ΔS | pose ΔS | **net** |
|---|---:|---:|---:|---:|
| **full 600-pair field** | −1.1308e-04 | +8.9162e-05 | **+3.4898e-05** | **+1.0977e-05 — POSITIVE** |
| **Lagrange subset** | −6.7369e-05 | +3.8135e-05 | −1.1789e-05 | **−4.1022e-05** |

**The full field FAILS outright.** Only the subset admits, at **3.30× the 1.242e-05 margin
and 2.05× the −2e-05 bar**. The subset is modelled from pass 4's MEASURED subset structure:
it kept 59.6% of cells but only 42.8% of tokens, i.e. **1.420 cells/token against the full
field's 1.019** — the sweep picks the efficient pairs, and that is the entire source of the
remaining margin.

### 20c. The margin was under-specified, and this is the correction

The 1.242e-05 margin was built from the projection-vs-T4 residual (5.27e-06) and the pose
print band (7.14e-06). Those are **measurement** errors. Pass 4 showed the dominant term is
**MODEL** error: projected −3.2352e-04, realized **−1.5046e-04** — a **1.7306e-04 miss,
53% of the projection**, thirteen times the margin. A rule that compares a projection
against a measurement-error margin is under-specified, and §15's "floor" case inherited
that flaw.

Carrying pass 4's own 53.5% band onto pass 5 gives **[−6.2968e-05, −1.9075e-05]**. The
whole band clears the 1.242e-05 CONTINUE margin. But the **pessimistic end misses the
−2e-05 admit bar** (0.95×), so pass 5 can fail its own admission after the search is spent.

### 20d. VERDICT: CONTINUE, marginal, subset-only

- central **−4.1022e-05**, band [−6.30e-05, −1.91e-05], CONTINUE margin cleared throughout;
- **full-field composition must not be sealed** — it is projected POSITIVE;
- convergence rule: **1.058%**, above the 1% bar but sitting ON it; pass 6 projects ~0.34%
  and fires;
- cost ≈ 0.9× pass 4's search (proposals per flipped cell keep rising) plus the chain;
- pre-registered stop UNCHANGED: if the admission's total is not negative past the bar, no
  candidate is built and no T4 row is bought.

### 20e. AMENDMENT before launch — the placement-price caveat (MEASURED at §24b)

The 7.7203 bits/token this projection uses is extrapolated from passes whose tokens were
chosen GREEDILY: the search tries moves in a fixed order and keeps whatever lowers the
realized flip count, so it implicitly harvests cheap placements first. §24b measured what
happens when placements are FORCED instead — the slide family's writes, pinned to specific
offsets, priced at **10.86 and 11.20 bits, about 1.73× pass-4's greedily-chosen 6.4000**.

So marginal price depends strongly on WHERE a token sits, not only on how many there are,
and **7.7203 is optimistic to the extent pass 5's remaining sites are more constrained than
pass 4's were**. The pass-5 rate leg is +3.8135e-05 against a central net of −4.1022e-05: at
1.3× the assumed price the net reaches the −2e-05 bar, and at ~1.7× — the factor §24b
actually measured on forced placements — it goes POSITIVE. This does not change the CONTINUE
call, which is made on the pre-registered rule, but it means the pass-5 admission is more
likely to bind than §20d's band alone suggests, and the stop is expected to do real work.

Binding, unchanged and reinforced: **the seal prices by real encode under whichever coder is
live at pricing time — never by this projection, and never by a number carried from pass 4.**

**Do not launch until MAIN confirms the pointer moved**: the pass-5 field must be proposed
on the promoted tree's parse-back argmax, because the base changes on promotion.

---

## 21. The pass-4 residual census — 12,614 cells, and the reachable/unreachable split

Measured with the §16 instrument on the SHIPPED parse-back (`argmax_n600.npy`) against the
DALI GT and the shipped field `admission_pass4/field_admitted.npz`.

### 21a. THE NEW COLUMN — what the search has already refused

| | cells | share |
|---|---:|---:|
| **TOUCHED** — was a flip at pass-2a start, so all three passes enumerated proposals on it and rejected them | **11,285** | **89.46%** |
| **BORN** — created by this arm's OWN edits; never a flip when the search enumerated | **1,329** | **10.54%** |
| …of the BORN, created since the pass-3 row shipped | 52 | |
| base flips now REPAIRED | 12,464 of 23,749 | **52.48%** |

**Nearly nine tenths of the residual is not unexplored — it is explicitly refused.** Three
passes proposed on those 11,285 cells at ~4.2 proposals each and every proposal lost the
composite re-render test. That is the unreachable set for single-cell token pre-distortion,
and it is what the renderer door inherits.

The other tenth is **collateral this arm created**: 1,329 cells born from our own edits
against 12,464 repaired — a **9.4 : 1** repair-to-collateral ratio. Honest but not free.

### 21b. The §16 tables, one pass later

| facet | pass-3 residual (12,866) | **pass-4 residual (12,614)** |
|---|---:|---:|
| token ALREADY == GT | 86.39% | **86.25%** |
| one-pixel boundary displacement | 99.58% | **99.56%** |
| on a GT class edge (enrichment) | 99.67% (46.1×) | **99.66% (46.1×)** |
| Lane over-representation | 40.15× | **40.43×** |
| Movable over-representation | 10.92× | **10.88×** |
| rows 128–319 share | 100.00% | **100.00%** |
| components / mean size | 11,859 / 1.08 | **11,635 / 1.08** |
| **isolated singletons** | 86.55% | **93.92%** |
| per-pair Gini | 0.2518 | **0.2494** |
| pairs with zero residual | 0 | **0** |

Everything holds — and the two facets that MOVED both say the same thing. **Singletons rose
86.55% → 93.92%**: the search eats clusters first and leaves specks. **Gini fell again**:
the residual keeps getting more uniform, never concentrating. There is still no clean pair
and the minimum is still 4.

### 21c. What the renderer door's charter should carry

1. **12,614 cells, of which 11,285 (89.46%) have been proposed on and refused** by three
   passes of single-cell search. Not a coverage gap — a capability limit.
2. **86.25% have a CORRECT stored token and a wrong render.** No token code reaches them.
3. **99.56% is one-pixel boundary jitter** on a codim-1 curve (99.66% on a GT edge, 46.1×
   enriched). The lever that fits is sub-pixel boundary placement, not class capacity.
4. **Lane 40.43× and Movable 10.88×**; Undrivable 0.37× and MyCar 0.13× are solved.
5. **Rows 128–319 carry 100%.** Two thirds of the frame contributes nothing.
6. **93.92% isolated singletons, mean component 1.08.** Any mechanism whose unit of repair
   exceeds ~1 cell pays for coverage it cannot use.
7. **Diffuse**: worst 100 of 600 pairs carry 30.60%, zero clean pairs. No per-pair
   specialisation available.
8. A composed successor must also not re-create collateral: pre-distortion currently runs
   **9.4 repairs per cell it breaks**, and the broken ones land in the same fragile band.

---

## 22. PRE-REGISTERED — the two-cell SLIDE sizing (before it runs)

§21 measured two things that together are a falsifiable claim about move SHAPE, not about
coverage: the residual is **99.56% a one-pixel boundary displacement**, and **89.46% of it
has already been proposed on and REJECTED** by three passes of the single-cell family.
Those cells are not unexplored. The available move is the wrong shape for them.

**The family.** A single move WIDENS one class by a token — it changes the class balance
across the edge. A displacement does not need widening, it needs TRANSLATION. So a slide
writes TWO tokens straddling a flipped site `s` reading `o` where GT says `g`: for each of
eight directions `d`, advance `g` into `s + d` and retreat it at `s − d` by writing `o`.
The pair of writes moves the token boundary one step along `d` while leaving the number of
`g` tokens across the cut unchanged. That degree of freedom is provably outside a
one-token family, and it is distinct from jg1's block/dilation moves, which set a whole
neighbourhood to one class.

**Price, stated before the result.** Two changed tokens per proposal, so break-even is
**~2× per repaired cell — ≈19.58 bits at pass 5's projected 9.7916 bits/token.** A slide
pays only where it repairs **≥2 cells**, or where the second token is one the coder charges
near zero. That second condition is real but is NOT settled here: per-token −log2 p is
direction-dependent ([[fs2]]) and average ≠ marginal ([[fs3]], 2.24×), so it is a question
for a real re-encode, never for a model. This sizing decides the MECHANISM only.

**Scope.** 12 SEEDED pairs (never a prefix — [[m88]]), ≤2 procs, the same instrument every
other row on this arm uses: `argmax_for_tokens`, i.e. the receiver's own `render_frame1` at
batch 1 then the frozen CPU SegNet, judged on the REALIZED whole-pair flip count. REFUSED
is defined as still-flipped on the shipped bytes AND flipped at pass-2a start, so all three
single-cell passes enumerated proposals on it and lost.

**PRE-REGISTERED STOP RULE.** The family is DEAD and is excluded from pass 5 unless BOTH:
**≥5% of refused cells repaired** AND **≥1.5 cells per accepted slide.** Below either, it
cannot clear a 2× break-even, and pass 5 stays singles-only as §20 planned.

**What each outcome means.** Clearing says the census's diagnosis was right and the wall was
the move shape — pass 5 becomes singles + slides under one acceptance and one subset sweep.
Failing says the wall is the RENDERER, not the family: if translating the boundary at the
token grid cannot move the argmax onto the right side, no token-grid move will, and the
89.46% refused set belongs wholly to the renderer door.

---

## 23. Move 35 — this arm's pass 4, PROMOTED, and the calibration flips sign

**S 0.13867171823146562 @ 181,521 B [contest-CUDA T4 n600]**, sha `b0ca809c…`, lane
`ddm_sj1_t4_token_predistortion_pass4_20260909`, **−1.515461e-04 S on +148 B**. Verified at
source: the lane's own `MODAL_REMOTE_RESULT.json` carries the same sha, size and
`score_recomputed_from_components`, and the tree on disk hashes to the same sha at the same
size. Lineage to date: **0.14784474 → 0.13867172 = −9.173023e-03 over 11 rows.**

**Projected 0.13867280587520867 → MEASURED 0.13867171823146562, residual −1.0876e-06.** The
first PESSIMISTIC projection of the wave, against +5.2748e-06 (move 31) and +4.1619e-06
(move 32). So the one-signed +0.0030%..+0.0038% band this arm carried is **not one-signed**;
the honest reading is a residual of order 1e-06 to 5e-06 with EITHER sign. Seg came in about
one cell better than projected (0.00010698 vs 0.00010699252).

## 24. The two-cell SLIDE family — DEAD on both clauses, MEASURED

### 24a. Mechanism (12 seeded pairs, 260 refused cells)

| | measured | bar |
|---|---:|---:|
| refused cells repaired | **28 = 10.77%** | ≥5% ✔ |
| accepted slides | 21 | |
| cells repaired | 24 | |
| **cells per accepted slide** | **1.143** | ≥1.5 ✘ |

The slide **does** reach cells that three passes of singles refused — 10.77%, twice the bar,
so the census's diagnosis of the residual as a boundary DISPLACEMENT was right about shape.
It reaches them **one at a time**, and two of twelve pairs accepted nothing at all.

### 24b. Rate — the clause I nearly failed to test

§22 pre-registered *"pays at ≥2 cells/slide **OR** where the second token is near-free"*, and
after 24a I was ready to call DEAD on the first clause alone. That would have been a
half-applied rule. The second clause is a property of the CODER and had to be encoded.

The sizing then surfaced something that made it genuinely live: the 21 slides cost only
**34 token writes, not 42** (advance 14, retreat 20), because one target token was often
already in the wanted class. At **1.62 tokens/slide** and pass 4's 6.40 bits/token the
modelled cost is **10.4 bits** against an **11.19-bit** break-even — under it by 7%, far too
close to call from a model.

MEASURED, four concurrent encodes against the live 120,367 B control, twin byte-identical:

| field | bytes | Δ | per slide |
|---|---:|---:|---:|
| control (live row) | 120,367 | — | — |
| **both** | 120,412 | **+45 B** | **17.143 bits** |
| advance only | 120,386 | +19 B | 7.238 bits |
| retreat only | 120,395 | +28 B | 10.667 bits |

**17.143 bits/slide against an 11.192-bit break-even — over by 1.53×.** The family would
need **1.751 cells/slide** and delivers 1.143.

**The standing hypothesis is FALSIFIED.** The retreat token was supposed to be cheap or
negative because it restores context earlier pre-distortion had broken. It costs **11.20
bits per write against the advance token's 10.86 — 1.03×, i.e. the same.** Restoring
context buys nothing here.

**A third number, larger than the verdict.** Both slide writes price at **~1.73× pass-4's
single tokens** (10.9–11.2 bits versus 6.40). Pass-4's singles were chosen greedily and the
search implicitly preferred cheap placements; the slide writes are FORCED to specific
offsets. So **the marginal price of a token depends strongly on WHERE it is, not only on how
many there are** — which means pass 5's projected 7.72 bits/token, extrapolated from
greedily-chosen placements, may itself be optimistic if pass 5's remaining sites are more
constrained. Additivity holds to 4.3% (+47 B separately vs +45 B together), so the two
writes barely interact and per-token accounting is sound.

### 24c. Verdict and consequence

**DEAD on both clauses. Pass 5 is singles-only, as §20 planned.** No slide code enters the
pass-5 family.

This is a complete negative with a named mechanism, and it sharpens the renderer door's
brief rather than merely closing a lane: **the residual is not reachable by token-grid moves
of either shape.** Singles refused 89.46% of it; slides reach 10.77% of that refused set at
1.53× over break-even. The token grid has been tried in both the widening and the
translating direction, and the remaining 12,443 cells belong to the renderer.

---

## 25. The pass-5 split of the shipped residual — and a level switch corrected in §21

Measured before the pass-5 rate leg returned, on artifacts already on disk: the SHIPPED
parse-back argmax of the live body (`pass4/seg_final/argmax_n600.npy`) against the DALI GT
table, split by the 235 accepted pass-5 moves. Instrument:
`experiments/ddm_sj1_pass5_census.py`; receipt
`/Volumes/VertigoDataTier/pact/ddm_sj1_pass5_price/CENSUS.json`.

Two gates passed before any row below is readable. The residual reproduces at **exactly
12,614** cells — the same number §21 measured and the same `flips_before` the pass-5 search
reported on the jg1 instrument. And **every one of the 234 distinct sites the accepted moves
aim at is a residual cell of the shipped body** (`repaired_sites_not_in_shipped_residual =
0`): the search and the parse-back are describing one object, not two.

(235 moves land on 234 distinct sites, and two moves repaired two cells each, so the site
mask is a lower bound on `PASS_RESULT`'s 237 repaired cells. The kind-comparison below is
unaffected by the three-cell difference.)

### 25a. THE CORRECTION — §21's "singletons rose" is a change of statistic, not of object

§21 reads: *"**Singletons rose 86.55% → 93.92%**: the search eats clusters first and leaves
specks."* Both numbers are real and neither is wrong; they are **different statistics**.
§16's 86.55% is the share of residual CELLS that sit in a size-1 component (11,135 of
12,866). §21's 93.92% is the share of COMPONENTS that are singletons (11,135 of 11,859 =
93.89%). Measured on the SAME pass-4 object, here, both ways:

| statistic | pass-3 residual (12,866) | pass-4 residual (12,614) | movement |
|---|---:|---:|---|
| singleton share of **cells** | 86.55% | **86.63%** | +0.08 pp |
| singleton share of **components** | 93.89% | **93.92%** | +0.03 pp |
| components | 11,859 | **11,635** | |
| cells per component | 1.0849 | **1.0841** | |

**On either statistic held fixed the granularity is FLAT.** The residual did not get more
speck-like between pass 3 and pass 4; §21's sentence compared a cell share against a
component share and read the level difference as a trend. The claim is withdrawn. What
§21's other facets say — 99.56% one-pixel displacement, Lane 40.43×, rows 128–319 carrying
100%, no clean pair — is unaffected and stands.

This is the [[m99]] genus (units × level × aggregation are part of the claim) landing inside
this arm's own memo, and the reason the census now emits both numbers side by side so the
next pass cannot repeat it.

### 25b. What the last reachable cells ARE — they do differ in kind

The interesting question is not the sizes. It is whether the 234 cells one more token move
can still reach differ from the 12,380 it cannot. **They do, on three facets.**

| facet | repaired (234) | remaining (12,380) | ratio |
|---|---:|---:|---:|
| **Lane over-representation** | **56.93×** | 40.11× | **1.42×** |
| Movable over-representation | 6.21× | 10.97× | 0.57× |
| Road over-representation | 1.56× | 1.77× | 0.88× |
| **singleton share of cells** | **95.30%** | 86.47% | **1.10×** |
| cells per component | 1.0271 | 1.0853 | 0.95× |
| **row centroid** | **208.1** | 200.9 | +7.2 rows |
| rows 128–191 | 40.17% | 54.57% | 0.74× |
| rows 192–255 | **45.30%** | 34.79% | **1.30×** |
| pairs contributing | 126 of 600 | 600 of 600 | |

**What a single token move can still reach is an ISOLATED, LOW-IN-FRAME, LANE speck.** It is
1.42× more Lane, 1.10× more likely to be a lone cell, and sits 7.2 rows lower — nearer the
camera, where a token covers fewer pixels of world and the boundary it moves is coarser
relative to the error. What it cannot reach is the clustered, higher-in-frame, Movable-
bearing part: Movable is *under*-represented among the reachable by 0.57×, and the
128–191 band — the far field, which carries the majority of the residual — is 0.74×.

That is a sharper handoff than §21c's list, because it is a statement about the SUB-family
rather than the family: the token grid is not merely out of budget on a homogeneous
residual; it has been eating a specific, identifiable corner of it — near-field isolated
lane specks — and the corner is nearly gone. A representation-level successor that only
reproduces this corner's capability inherits 234 cells out of 12,614 and nothing else. The
one it must beat is the far-field cluster: **rows 128–191 hold 54.57% of what remains, and
the token grid's yield there is 0.74× its own average.**

---

*(Section 22+ — the pass-4 T4 row and whatever follows it — are
appended as each lands. Nothing is written here before it is measured.)*

---

## Frontier line

`sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]` (live pointer, sha `06c44dc464038649…`)

Lineage: fs2 0.14784474152757654 @ 180,023 B → cl2 0.14781744131049854 @ 179,982 B → rc1 0.14666350774473783 @ 178,249 B → pc1 ×4 0.1451981569076111 @ 176,448 B → pc1 ×8 0.1445177913121716 @ 175,576 B → pc1 ×16 0.14411787458634504 @ 174,786 B → sj1 pass 2a 0.1398140172839628 @ 180,904 B → sj1 pass 3 (above).
