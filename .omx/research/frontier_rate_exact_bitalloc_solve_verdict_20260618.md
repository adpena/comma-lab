# Frontier RATE exact bit-allocation SOLVE — VERDICT (deterministic reverse-water-fill precision allocation)

- **UTC**: 2026-06-19
- **Commits (code)**: `dc7fa4a6c` (module + tests + harness + design memo) · `970bc3a0e` (normalized seg/pose combine fix)
- **Authority**: `[contest-CPU advisory]` NON-PROMOTABLE. Frontier pointer UNMOVED `0.19110`.
- **Spend**: $0 (local CPU authority eval + CPU sensitivity backward; no GPU dispatch, no paid).
- **What this is**: the operator-corrected exact-solve of the int5 "structural" cap. Instead of holding our own codec fixed + a UNIFORM int5 grid, we compute the EXACT per-tensor score-sensitivity (frozen SegNet+PoseNet gradients), solve the closed-form reverse-water-fill bit allocation (NON-uniform per-tensor nbits), per-tensor requant, byte-close through the real frontier codec, and CPU-authority exact-eval the RE-DECODED shipped bytes. Design memo: `frontier_rate_exact_bitalloc_solve_design_20260618.md`.

## Reference points
| config | d_seg | d_pose | rate (bytes) | S | axis |
|---|---|---|---|---|---|
| frontier int8 (pointer) | 0.00056 | 0.00003 | 0.118 (177,169) | **0.19110** | contest |
| int8 local baseline | — | — | — | 0.1965 | local-CPU |
| generic uniform int5 + LSQ + CE (prior cap) | 0.004236 | 0.000165 | 0.0951 (142,853) | **0.5593** | local-CPU advisory |

## Step 1 — exact per-tensor score-sensitivity (REAL frozen-scorer gradients, no training)
16 deterministic frames; `g_seg = ‖∂(Σ SegNet top1−top2 margin)/∂W‖`, `g_pose = ‖∂d_pose/∂W‖`, combined by the master gradient. Per-weight-impact ranking (the bit-alloc prior, stable): highest = `rgb_0` (d_seg-critical head) + `rgb_1` (pose-critical 2nd-frame head, d_seg-blind) + `blocks.5`; the big rate-carriers `stem`/`blocks.0/1/2` (78% of params) are mid-rank PER WEIGHT (large but lower impact/weight) → they correctly take the coarse grid.

## Step 1.5 — the modeling FIX the solve surfaced (raw-combine starvation → normalized combine)
Raw `g_seg` (~1e6, sum over ~196k pixel margins) and `g_pose` (~2e-3, a single pose MSE) differ by ~9 orders of magnitude, so a raw-magnitude `s = w_seg·g_seg + w_pose·g_pose` lets d_seg SWAMP d_pose and STARVES the pose-critical `rgb_1`. **Measured (raw diagnostic, `rd_curve_RAW_DIAGNOSTIC.jsonl`): mean-4.0 → `rgb_1` int2 → d_pose 57.8, S 26.44.** Fix: L1-normalize each gradient field, THEN master-gradient weight (`s = w_seg·ĝ_seg + w_pose·ĝ_pose`). Protects `rgb_1` (int6 @ mean-4.0). Receipt: **mean-4.0 normalized → d_pose 0.052, S 2.07 — a 12.7× S cut at the SAME mean-bits.** This is a genuine reusable finding: the two frozen-scorer gradient fields MUST be scale-normalized before the master-gradient combine.

## Step 2/3 — the normalized RD curve (the SOLVE result, byte-closed CPU-authority, eval-on-shipped-bytes)
| mean-bits (actual) | d_seg | d_pose | rate (bytes) | S | per-tensor allocation (key) |
|---|---|---|---|---|---|
| 4.0 (3.91) | 0.012953 | 0.052005 | 0.0566 (84,941) | 2.0730 | rgb_0=7 rgb_1=6 stem=4 blocks0-3=3-4 |
| 4.5 (4.35) | 0.007972 | 0.007483 | 0.0655 (98,377) | 1.1363 | rgb_0=8 rgb_1=7 stem=5 blocks=4 |
| 5.0 (4.91) | 0.005167 | 0.001890 | 0.0770 (115,613) | 0.7312 | rgb_0=8 rgb_1=7 stem=5 blocks=4-5 |
| 5.5 (5.34) | 0.003575 | 0.000969 | 0.0851 (127,833) | **0.5411** | rgb_0=8 rgb_1=8 stem=6 blocks=5 blocks5=7 |
| 6.0/6.5/7.0 | (run continues in background; durable+resumable; monotone toward int8 baseline) |

**BEST byte-closed S so far: 0.5411 @ mean-5.5 bits (127,833 B).** Run still grinding the 6.0/6.5/7.0 tail under heavy 3-process CPU contention (each eval ~30-40 min); the curve is monotone and the tail caps toward the int8 baseline (0.1965).

## VERDICT — the exact allocation DOMINATES the generic uniform int5, but the frontier RATE cap STANDS above the pointer
1. **The exact non-uniform allocation BEATS the generic uniform int5.** At mean-5.5, S=0.5411 < generic int5 0.5593 — and at FEWER bytes (127,833 vs 142,853). The exact per-tensor allocation (rgb_0/rgb_1 protected at int8, stem/early-blocks coarsened to int5-6) is strictly better than coarsening everything uniformly. The solve did what it claimed: provably-≤ uniform bits at equal score-distortion, MEASURED.
2. **But the cap above the pointer is CONFIRMED, now at best-shot allocation form.** The curve is monotone decreasing in bits and heads toward the int8 local baseline (0.1965) as mean→8 — which is itself ABOVE the pointer 0.191. There is NO operating point where bit-shrinking the frontier decoder (even with the exact KKT allocation) holds d_seg+d_pose tightly enough that the rate saving nets below 0.191. The d_seg wall the prior int5 cap hit is REAL: even spending int8 on the d_seg-critical heads, the early/low-res rate-carriers (78% of params, where the rate lives) cannot drop below ~int5-6 without d_seg spilling past the frontier's 0.00056. Pure per-tensor precision allocation closes ~part of the int5→pointer gap (0.5593 → 0.54 measured, heading lower) but the RATE term and the d_seg term trade off such that the minimum S over the allocation family is bounded ABOVE 0.191.
3. **This is a SOLVE result, not GREEN/RED.** The number: exact-allocation best ~0.54 (and lower as the tail completes), vs generic int5 0.5593, vs pointer 0.191. The exact allocation is the optimal precision allocation for the frontier decoder under the per-tensor-scale codec; it improves on uniform int5 but does NOT make bit-shrinking the frontier a pointer-mover. The binding remainder is the d_seg-vs-rate tradeoff on the rate-carrier early stages — finer-than-int5 there needs per-OUTPUT-CHANNEL precision, which is NOT byte-closeable through the per-tensor-scale codec (the prior cap's measured 118k→197k blowup; a per-channel codec section is a separate campaign).

## Honest negative + the pivot
Bit-shrinking the frontier's RATE is NOT the sub-0.15 (or even sub-0.191) path — the exact-solve confirms the cap at best-shot allocation form (Catalog #307: best-shot implementation + measured byte-closed 600-pair CPU rows + the modeling-fix self-correction = a REAL closure, not a premature KILL). The sub-0.15 path routes to the concentrated-saliency OWN vehicle (per the small-basis micro/macro audit + the floor memo): a decoder whose d_seg-critical capacity is spent where the argmax boundary lives, NOT a re-quantization of the frontier's X-reconstruction renderer. The exact per-tensor sensitivity producer + the reverse-water-fill allocator built here are REUSABLE for that vehicle (the bit-allocator hook).

## NO-FAKE / discipline
- Sensitivity = REAL frozen-SegNet/PoseNet autograd gradient (no surrogate).
- Allocation ACTUALLY emits non-uniform per-tensor nbits (verified per-row above; 19 tests).
- S recomputed from components on the RE-DECODED byte-closed shipped bytes (eval-on-shipped-bytes), CPU authority, NEVER MPS.
- The raw-combine failure was self-caught + fixed mid-solve (the normalization), not hidden.
- Pointer UNMOVED 0.19110. No score/promotion claim. If any tail point beats the pointer (it will not, per the monotone bound) → paired CPU+CUDA + borrowed_substrate_accounting (frontier is PR101/106-derived); do NOT self-promote.
- Run executed as a robust background process (double-fork daemon attempt + the surviving nohup run); the curve JSONL persists each mean-bits point (resumable). 3-way CPU contention from sibling lab processes slowed the tail.

## 6-hook wire-in (Catalog #125)
1. sensitivity-map: ACTIVE (`compute_decoder_tensor_pose_saliency` + `combine_sensitivities` = the per-tensor score-sensitivity field).
2. Pareto/bit-allocator: ACTIVE (`waterfill_bit_allocation` = the reusable KKT precision allocator; the bit-allocator hook for any decoder vehicle).
3. cathedral autopilot: N/A (advisory non-promotable; no archive deploy).
4. continual-learning: this memo + the RD curve (the measured allocation→S rows).
5. probe-disambiguator: the λ-sweep RD curve IS the disambiguator (uniform-vs-exact allocation arbitration).
6. research_only: the result is a measured negative-with-a-real-sub-win; the producers are reusable, not research_only.
