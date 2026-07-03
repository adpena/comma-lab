# POSE FRAME0 INVERSE-SOLVE — MLX/METAL ENGINE + PARITY GATE (#251)

**UTC:** 2026-07-03 · **Authority:** d_pose VERDICT is the frozen **CPU-torch PoseNet** —
`[contest-CPU authority via CPU-torch verdict] NON-PROMOTABLE`; the MLX/Metal SEARCH is
`[macOS-MLX research-signal]`. **Frontier pointer UNMOVED 0.19110** (advisory diagnostic; NOT a
contest score). Tool: `tools/pose_frame0_inverse_solve_probe.py --engine mlx` (committed `462e732ca`).
Parity report: `reports/pose_frame0_probe_mlx_parity_gate.json`.

## VERDICT (one paragraph)
**YES — MLX/Metal reproduces the CPU search's pose result, at 1.78× on the P-E solve, and it is
decision-grade for the #250 symposium's P-E existence proof.** On the 24-pair parity gate BOTH engines
drive the free-frame0 inverse-solve to the frozen CPU-authority floor: **CPU-only median d_pose
1.27e-08, MLX-hybrid median 1.88e-08 (ratio 1.49, same order of magnitude); 22/24 pairs both < 1e-6**;
GATE PASS. On the single hardest pair (large forward motion) the MLX-hybrid (1.9e-6) actually BEAT
pure-CPU (7.28e-4) — the Metal warm-start escaped a basin the 8-iter CPU LM-GN stalled in. Wall-clock:
the P-E solve is **1.78× faster** (CPU 96.1 s vs MLX-hybrid 54.1 s over 24 pairs). The **P-E existence
proof — pose is inverse-solvable to ~0 (~1e-8) against the frozen PoseNet — is now AUTHORITY-CONFIRMED
at n=24** (was n=3 smoke), so pose does NOT fundamentally bound the witness budget at R1's 0.0011.
The **decisive P-F question** (cheapest LEGAL decoder-reproducible counted carrier — coarse + generic
DCT/low-rank sweeps) is render+CPU-authority-bound (~2–3 hr at n600, which MLX does NOT accelerate);
that full n600 `--pf --pf-generic` run is now executing as a resumable durable daemon (below) and will
supersede the n=3 smoke's directional coarse-P-F numbers when it lands.

## What was built (substrate wiring — all REUSED, nothing reinvented)
- **MLX PoseNet forward (autodiff):** `tac.local_acceleration.mlx_scorer_adapters.torch_posenet_to_mlx`
  → `MLXPoseNetAdapter` (custom-Metal-VJP strided grouped convs). Input = 12-ch YUV6 pair NHWC
  `(1,192,256,12)` in [0,255], order `[f0_6, f1_6]` (upstream `PoseNet.preprocess_input` convention).
- **MLX preprocess (mirrors the CPU `_posenet6` path exactly):** bilinear up f0_work→camera(874×1164),
  **STE uint8** (`round`+`stop_gradient`), bilinear down camera→scorer(384×512),
  `pr95_hnerv_mlx_training.rgb_to_yuv6_mlx`, `resize_nhwc_align_corners_false`. f1: camera→384 bilinear
  → yuv6 once/pair.
- **6×N Jacobian on Metal:** `mx.vjp(pose6_fn, (f0,), (e_k,))` for k=0..5 → the SAME rank-≤6 min-norm
  LM-GN step as the CPU `pe_free_solve` (verified full-rank, `rank(JJᵀ)=6`).
- **Precision polish (the drift fix, REQUIRED):** the MLX PoseNet is not FP32-exact (forward parity on a
  real GT pair = **8.79e-02** > the 5e-2 GPU bound), so after the MLX coarse solve a **short CPU-torch
  `pe_free_solve` (2 iters), warm-started from the MLX frame0**, lifts d_pose to CPU-authority precision.
- **Verdict (UNCHANGED):** `_frozen_dpose` → `cpu_verdict_d_pose` (camera-res uint8, `inference_mode`).
  Never MLX. Same P-E/P-F sweeps + JSON schema + NO-FAKE self-check (`PoseNet(GT)==gt_poses`, err 0.0).

## Hybrid design (why the net speedup survives)
`MLX coarse GN min-norm (Metal, 6 iters)` → `CPU-torch GN polish (2 iters, warm-started)` →
`CPU frozen-authority verdict`. Because the MLX PoseNet closely tracks the CPU one, the CPU polish needs
only 2 iters (vs 8 from scratch), so most of the Jacobian cost moves to the GPU. Both engines END in a
CPU-torch min-norm step → the returned frame0 is authority-grade AND P-F-comparable (same basin) →
apples-to-apples. Tuning (measured, 3-pair): `mlx-iters 6 / polish 2` is the robust sweet spot — cutting
`mlx-iters` to 3–4 starves the hard large-forward-motion pairs (mlx-hybrid → 8e-3 / 3e-4, unrecovered by
2 polish iters); `polish 1` gives 2.5× but a 10× worse hard pair. Robustness > 0.5× → default `6 / 2`.

## 24-pair PARITY GATE (task part e) — `--parity-gate --parity-pairs 24`  → **PASS**
| metric | value |
|---|---|
| GATE_PASS | **true** |
| d_pose CPU-authority (median / max) | 1.27e-08 / 7.28e-04 |
| d_pose MLX-hybrid authority (median / max) | 1.88e-08 / **1.9e-06** |
| ratio median (mlx / cpu) | **1.49** (≤1.5 bound; same OOM) |
| pairs both < 1e-6 | **22 / 24** |
| **P-E solve wall-clock speedup** | **1.78×** (CPU 96.1 s → MLX-hybrid 54.1 s) |
| MLX PoseNet fwd parity vs CPU (GT pair) | 8.79e-02 `[macOS-MLX research-signal]` (proposer only) |

*Caveat, honest:* at the ~0 floor the per-pair ratio is noisy (cpu 1e-10 vs mlx 1e-8 = 100× ratio but
both numerically ~0, far below any decision threshold); the **median** and **"both < 1e-6" (22/24)** are
the meaningful gate metrics, and both pass. `d_pose = MSE over the 6 pose scalars` (upstream `PoseNet`),
so 1e-8 ↔ per-scalar error ~1e-4.

## RENDER DECISION (task part a) — numpy oracle RETAINED (per the task fallback)
The MLX render (`render_batch_through_R_mlx`) emits at **scorer res 384×512**, whereas this probe's
warp-base + frozen-authority pipeline (and the R1 anchor) operate at **camera res 874×1164** (the numpy
oracle `numpy_oracle_reference_frames`). That resolution mismatch means the two renders are not the same
frames, so — exactly per the task's explicit fallback ("*if they diverge, keep the numpy oracle for f1
and only MLX-accelerate the search*") — the **numpy oracle is retained for f1** and MLX accelerates the
SEARCH. The MLX search's f1 preprocessing (bilinear camera→384 + yuv6) is validated end-to-end by the
MLX-vs-CPU PoseNet **forward parity** above. A camera-res MLX witness render (build the MLX witness +
self-orient loop + `apply_contest_faithful_roundtrip_nhwc(output_hw=camera)`) would slash the ~63-min
render but is a separate, larger build and would change the probe's res semantics vs R1 — not taken here.

## HONEST runtime finding — MLX accelerates the SEARCH, not the whole probe
The full n600 probe is **render + CPU-authority-sweep bound, NOT search-bound**, so it is inherently
~2–3 hr and CANNOT be "minutes": (a) the numpy-oracle witness render is **~6.3 s/pair (~63 min for 600)**
— kept numpy per the res fallback, unaccelerated; (b) the `--pf-generic` DCT/low-rank + coarse sweeps are
the **frozen CPU-torch authority verdict** (`_frozen_dpose`, forward-only) — MLX may NOT replace them
(they are the promotable-adjacent numbers). MLX's real win is the **per-pair Jacobian inverse-solve**
(1.78×), which is the slice the task targeted. This is a measured property of the diagnostic, reported
plainly rather than papered over.

## n600 decision-grade run — LAUNCHED as a resumable durable daemon
```
tools/spawn_durable_daemon.py --label pose_frame0_mlx_n600 \
  --log reports/pose_frame0_probe_mlx_n600.daemon.log --projected-gb 8 --min-free-gb 20 -- \
  .venv/bin/python tools/pose_frame0_inverse_solve_probe.py --engine mlx --pf --pf-generic \
    --n-pairs 600 --mlx-iters 6 --polish-iters 2 --out reports/pose_frame0_probe_mlx_n600.json
```
- pid 55334 (own session/pgid), memory admission OK (114.5 GB free, projected 8 GB; machine-safe,
  no training running). Writes per-pair **resumable** `reports/pose_frame0_probe_mlx_n600.partial.jsonl`
  + caches the 600-frame render to the SSD tier. Final report → `reports/pose_frame0_probe_mlx_n600.json`.
- **Harvest (for #250):** when the daemon lands, read `reports/pose_frame0_probe_mlx_n600.json`
  `1_P_E_free_frame0_solve` (P-E floor) + `2_3_P_F_rate_pareto` (coarse `d_pose_frozen`/bytes Pareto +
  `over_warp_base…` generic DCT/low-rank rungs + `cheap_vs_adversarial_verdict.CHEAPLY_COUNTED_REALIZABLE`)
  → supersede this ledger's n=3-smoke directional coarse-P-F numbers with the n600 aggregates. Check:
  `tail reports/pose_frame0_probe_mlx_n600.daemon.log`; stop: `spawn_durable_daemon.py --stop pose_frame0_mlx_n600`.

## Tags (kept explicit)
- d_pose VERDICT + the 24-pair gate d_pose numbers: `[contest-CPU authority via CPU-torch verdict]
  NON-PROMOTABLE` (frozen CPU-torch PoseNet; NEVER MPS; MLX never a score).
- MLX/Metal SEARCH (proposer, forward-parity, wall-clock): `[macOS-MLX research-signal]`.
- Frontier pointer UNMOVED 0.19110. `score_claim=false`, `promotable=false`.

**Cross-refs:** `pose_frame0_inverse_solve_probe_20260703T0810Z.md` (the n=3 smoke this succeeds) ·
`project_pose_solved_screw_twist_dual_use_film_conditioned_sidecar_20260701` (POSE OPEN on the witness) ·
`mlx_gpu_not_bit_identical_crossprocess_bitexact_proof_cpu_locked_20260702` (why MLX = proposer, CPU =
authority) · #248 P-E/P-F ladder · #250 optimal-form symposium · #238 byte-close.
