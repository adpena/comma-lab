# ddm_gdc2 — categorical Cool-Chic distillation of the retained K=8 scanline teacher (arm memo, 2026-09-10)

`research_only=true` · `score_claim=false` · `promotable=false` · `frontier_moved=false`
Axis: `[macOS-MLX research-signal]` training · `[macOS-CPU scorer-free exact-field measurement, n600]` verdicts.
No scorer, no Modal, no candidate archive, no MPS number anywhere in this arm.

Frontier line unchanged by this arm:
`composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43)`.

## Provenance pins

| object | path | bytes | sha256 |
|---|---|---:|---|
| move-43 field | `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/pass6.u8` | 117,964,800 | `78e57545439515eb29f806cc5a5f7d8b14acf658955cdd7561debb4edf3b7db6` |
| K=8 teacher render | `…/ddm_gdc1_generator_door/scanline_v1/k08/receiver_render.u8` | 117,964,800 | `abd130921beec23255112e8b56148d55c49cf4fede03e99e0e99b55378b16ea2` |
| GDC1 memo | `.omx/research/ddm_gdc1_generator_door_construction_design_20260910.md` | 24,749 | `c924bb6a831e7a8fa1a48e5f8d8e2f5433ef64e95fea84cbf06bca7f6e67c439` |
| GDC1 RESULT | `.omx/research/ddm_gdc1_20260910/RESULT.json` | 2,634 | `c40caf67ff344281a61ea4d5d96c76922c3a0a5f6fe973c7adcc3ffd3044f316` |

Pointer commit `48109233e`; archive sha `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`;
seed `20260910` everywhere; custody root
`/Volumes/VertigoDataTier/pact/ddm_gdc2_categorical_coolchic_k8_distill/`.

## Finding 1 (MEASURED, first burn) — the charter's clean screen was a transferred rate; the real one is 2.04x tighter

GDC1 screened this door at `packet <= 68,322 B`, which is `94,010 - 0.2909 * 88,304` using GF1's
transferred rate. GDC1 itself flagged that transfer as optimistic (its own K=6 point measured 0.4508).
The first GDC2 burn replaced the transfer with the measurement on the exact object being distilled:
the real exact residual that carries the K=8 teacher render back to the move-43 field, raced over all
eight retained residual orders and three physical coders, with exact-closure proof.

| residual order | raw B | coded B | winner |
|---|---:|---:|---|
| frame_raster | 200,321 | 64,140 | lzma2_extreme |
| class_frame_raster | 205,339 | 62,880 | lzma2_extreme |
| tile8_time | 240,001 | 67,379 | brotli_q11 |
| tile16_time | 226,437 | 62,755 | brotli_q11 |
| tile32_time | 217,562 | 61,632 | lzma2_extreme |
| **tile64_time** | 211,155 | **60,520** | lzma2_extreme |
| class_tile16_time | 231,689 | 62,721 | brotli_q11 |
| pair_tile16 | 208,831 | 65,152 | lzma2_extreme |

- `R_exact(K=8 teacher) = 60,520 B`, exact field closure verified, elapsed 11.9 s.
- Measured rate **0.6854 B per mismatch** over 88,304 mismatches. GF1's 0.2909 is **0.4244x** of it —
  the transfer was **2.356x optimistic**. The spread across all eight orders is only 60,520–67,379 B
  (11.3%), so no ordering rescues the rate; the cost is the addressing entropy of a sparse error set,
  not the scan.
- Teacher mismatches by field class 0..4: `[7,914 · 63,143 · 4,291 · 12,530 · 426]` — Lane (class 1)
  carries **71.5%** of them at 0.59% of the area, exactly the MD1–MD4 Lane concentration.
- **Real packet cap at exact teacher identity = 94,010 − 60,520 = 33,490 B**, against 292,493 B of raw
  counted decoder state (230,400 B `z0` + 57,600 B `z1` + 5,303 B parameters). That is **8.74x**
  compression, or **0.917 bits per counted latent symbol**, at zero added mismatch.
- The teacher itself sits at `306,042 + 60,520 = 366,562 B` = **3.90x** the 94,010 B gate.

Retained: `…/ddm_gdc2_categorical_coolchic_k8_distill/teacher_residual_probe/` (RESULT.json,
eight `residual.raw`, 24 coded payloads plus deterministic repeats).

## Finding 2 (MEASURED) — the exact-residual law R(M) on this field, over three decades

The same eight-order race was run against every retained GDC1 render, so the residual rate is now a
measured curve on the move-43 field rather than a transferred constant. Every row proves exact field
closure. K=6 reproduced GDC1's own 105,628 B (GDC1 raced three orders, this raced eight, same winner)
— an independent cross-check of the residual coder.

| render | mismatches M | real residual R (B) | B/mismatch | winning order | its packet | packet+R | **packet cap at this M** |
|---|---:|---:|---:|---|---:|---:|---:|
| K=4 | 673,602 | 174,680 | 0.2593 | tile64_time | 133,426 | 308,106 | −80,670 |
| K=6 | 234,295 | 105,628 | 0.4508 | tile64_time | 223,494 | 329,122 | −11,618 |
| **K=8 (teacher)** | **88,304** | **60,520** | **0.6854** | tile64_time | 306,042 | 366,562 | **33,490** |
| K=12 | 9,311 | 11,772 | 1.2643 | frame_raster | 421,886 | 433,658 | 82,238 |
| K=16 | 561 | 1,238 | 2.2068 | frame_raster | 459,394 | 460,632 | 92,772 |

Local log-log exponents of `R ∝ M^b`, in order of decreasing M: **0.4763 · 0.5708 · 0.7278 · 0.8017**.
The residual is strongly sublinear and *stiffening*: **a better generator pays more per remaining
error**, and the marginal return on accuracy keeps falling. Halving a student's mismatches near the
teacher's operating point buys back only about a third of its residual bytes. Any successor that
budgets residual with a rate borrowed from a coarser render under-charges itself — GF1's 0.2909
corresponds to a render around 6.7e5 mismatches, not to the K=8 operating point where the truth is
2.36x higher. (Same genus as `[[binding-instruction-numbers-expire-and-nobody-rederives-them]]` and
`[[cross-regime constant transfer]]`; the winning order even switches from `tile64_time` to
`frame_raster` below ~1e4 mismatches, so the order roster must be re-raced too.)

### The admissible frontier this defines for every successor

The gate is one curve in the `(mismatches, packet bytes)` plane:
`packet <= 94,010 - R(M)`, with `R(M)` measured above. Reading it off:

| operating mismatch count M | admissible packet |
|---:|---:|
| 0 (exact program, no residual) | 94,010 B |
| 561 | 92,772 B |
| 9,311 | 82,238 B |
| 88,304 (the teacher) | 33,490 B |
| 234,295 | infeasible |

The move-43 archive already describes the same 117,964,800 token cells exactly in a **119,969 B**
replaceable tail. So the whole GDC1/GDC2 door reduces to one sentence: **describe this token field
exactly in at most 94,010 B, i.e. beat the shipped tail by 21.6%** — either as one exact program, or
as a program plus its exact residual anywhere on the curve above. The near-exact corner (M ~ 1e3, up
to ~92.8 kB of program) is far more forgiving than the teacher's corner (M ~ 8.8e4, only 33.5 kB), and
no arm before this one had the curve to aim at.

## Construction as built (GDC1 governed spec, verbatim where it is fixed)

Counted source object = one packet with three raced sections: `z0` int8 `[75,24,32,4]` (230,400 B raw),
`z1` int8 `[150,12,16,2]` (57,600 B raw), and parameters (5,303 B raw: int8 stem/depthwise/pointwise/head
weights, int32 biases, three uint8 requant shifts). Trilinear resampling to `(600,384,512)`, five
width-24 depthwise-separable 3x3 blocks, five-logit integer head, argmax. Free generic decoder code:
the resampling lattice, the convolution algebra, the requant arithmetic, the argmax, the three coders.

**Arithmetic contract.** Every intermediate the receiver computes is an integer of magnitude below
2^24, so float32 and int64 evaluation agree exactly and the argmax is summation-order independent.
The guard is executable, not a comment: `_matmul_exact` recomputes in int64 whenever the derived bound
could be violated.

**MLX/NumPy parity (MEASURED).** The MLX training window reproduces the receiver's frame-border zero
pad by masking out-of-frame sites to zero after every layer, and uses valid convolutions so a 74x74
window yields the receiver's exact 64x64 interior. Argmax identity was verified exact at an interior
window and at both frame corners (`[10,128,192]`, `[0,0,0]`, `[592,320,448]`), before and after
training. Full-n600 receiver decode: **0.309 s/frame -> ~185 s**, inside the 900 s budget.

## Two recorded design decisions the spec left open (both would have made the run measure nothing)

1. **Trainable chart.** The counted parameters are integers on a ±127 lattice, but AdamW takes a
   normalised step of order the learning rate. In the integer chart the spec's `lr=3e-3` moved an
   integer weight by 3e-3 per step: MEASURED cross entropy 1.70 -> 1.26 over 200 steps, essentially the
   class prior (1.129 nats). Every trainable tensor is therefore carried unit scale and multiplied by
   its fixed quantiser scale (127 for latents/weights, 1024 for biases) before rounding. In the unit
   chart the same 300 steps reach cross entropy 0.298 and 90.7% training accuracy. Architecture,
   latent sizes and loss are unchanged; only the optimiser's coordinate chart is.
2. **Rate-proxy unit.** The spec fixes `lambda in {1e-4, 3e-4, 1e-3}` but not the unit of the measured
   symbol-rate proxy, and the unit decides whether the sweep is informative. Expressed in the gate's
   own unit — estimated kilobytes of the counted latent stream, `N * H / 8000` — the three lambdas put
   the rate term at 0.020 / 0.061 / 0.204 against a cross entropy of order 0.05–0.30 nats: weak /
   moderate / strong. In bits-per-symbol all three would have been inert and the sweep would have
   measured one point three times.

Per-layer initialisation scales are likewise DERIVED from the fixed requant shifts
(`sigma = 2**shift / sqrt(n/2)`): a single global scale decays the activation to zero by block five and
kills the gradient (MEASURED).

## Parity smoke (declared SCOPE reduction: step counts only; produces no verdict)

`…/ddm_gdc2_categorical_coolchic_k8_distill/smoke_parity_scope/`, 300 / 200 / 100 steps, one lambda,
elapsed 661 s, peak RSS **0.67 GiB**. It proved the whole chain before the governed burn spent an hour:

| check | result |
|---|---|
| packet built, deterministic repeat byte-identical | PASS (3 stage ends) |
| receiver parse-back exact (z0, z1, parameters) | PASS |
| full-n600 receiver render | PASS, decode **172 / 202 / 192 s** (budget 900 s) |
| MLX float vs NumPy integer argmax identity | **exact**, 262,144 sampled sites, agreement 1.000 |
| resumable atomic checkpoints every 250 steps | PASS |

A separate 1,100-step probe measured the optimisation itself: the run sits on the class-prior plateau
(cross entropy 1.129 nats) for roughly 300 steps, escapes, and reaches **98.4% training accuracy by
step 700**, holding 97–98% after. The 300-step smoke had simply not escaped yet — the plateau is a
symmetry-breaking delay, not a pathology.

## Stage table

*(filled by the governed burn)*

## Closure arithmetic

The authority gate is `packet_bytes + real_exact_residual_bytes <= 94,010 B`, with receiver parse-back
exact, deterministic packet repeat, and full-n600 decode under 900 s.

*(filled by the governed burn)*

## Retained custody

Root `/Volumes/VertigoDataTier/pact/ddm_gdc2_categorical_coolchic_k8_distill/` (storage waterfall:
Vertigo first; 36.4 GiB free at launch, projected output under 8 GiB, live floor 12 GiB enforced by
the driver, which refuses to start below it). Nothing deleted; every coder repeat kept beside its
payload.

| directory | contents | files | size |
|---|---|---:|---:|
| `teacher_residual_probe/` | K=8 exact-residual race, 8 orders x 3 coders + repeats | 64 | 4.9 M |
| `residual_law/k04 k06 k12 k16/` | the same race at four more mismatch counts | 64 each | 21 M / 9.9 M / 1.0 M / 292 K |
| `smoke_parity_scope/` | declared-SCOPE parity smoke: packets, renders, checkpoints | — | 386 M |
| `governed_v1/` | the governed burn: tile schedule, per-stage checkpoints every 250 steps, per-branch stage ends, per-evaluation packets/renders, heartbeats | — | *(burn)* |

Launch receipts: `.omx/tmp/codex_runs/gdc2_teacher_residual_probe.done`,
`gdc2_smoke_parity.done`, `gdc2_residual_law_k{04,06,12,16}.done`, `gdc2_governed_v1.done`.
Lane claim `ddm_gdc2_categorical_coolchic_k8_distill_20260910`.

## Boundaries

- Nothing here is a score. No scorer, no Modal, no candidate archive, no MPS number. The pointer is
  untouched.
- The residual rate is measured for THIS field and THIS render family; it does not transfer to a
  different generator's mismatch geometry (that is the whole content of Finding 1).
- The teacher is a smoothed K=8 scanline render, so a decoder distilled from it inherits the teacher's
  88,304 mismatches as a floor against the field. Training against the field directly is a strictly
  different object and is not what this charter measured.
- Training blocks are sampled uniformly, as specified. Lane occupies 0.59% of the area and 71.5% of the
  teacher's mismatches, so uniform sampling under-serves exactly the class that dominates the residual.
  Importance-weighted sampling is the named successor lever, not a silent change here.

<!-- # FORMALIZATION_PENDING: arm process memo; the closure arithmetic lands in the equations leg with the burn result -->
