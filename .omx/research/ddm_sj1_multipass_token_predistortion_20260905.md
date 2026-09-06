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

---

*(Section 11+ — the n600 pass table, the persistent partition, admission, exact ΔS — are
appended as each lands. Nothing is written here before it is measured.)*

---

## Frontier line

`pc1 x16 S 0.14411787458634504 @ 174,786 B [contest-CUDA T4 n600]` (live pointer, sha `1de6c5d7186a0b31…`)

Lineage: fs2 0.14784474152757654 @ 180,023 B → cl2 0.14781744131049854 @ 179,982 B → rc1 0.14666350774473783 @ 178,249 B → pc1 ×4 0.1451981569076111 @ 176,448 B → pc1 ×8 0.1445177913121716 @ 175,576 B → pc1 ×16 (above).
