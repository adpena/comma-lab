# THE CLOSED-FORM ATLAS — the operator's claim verified at source and made binding: every upstream stage is frozen piecewise-analytic math; 10/12 stages HELD with receipts, 2 PARTIAL (certified radii), and the doctrine now rides every scaffolded charter

Date: 2026-08-31 · Author: MAIN · Cost: **$0** (source verification + receipt audit)
Axis: doctrine + audit. `score_claim=false` · `promotable=false`
`verdict_scope`: operator steer 2026-08-31 verbatim *"All upstream can be closed form"* —
adjudicated TRUE in a precise sense, encoded structurally, gaps priced honestly.

## 1. The precise statement (verified at source this turn)

The ENTIRE scoring functional `S = 25·|archive|/37,545,489 + 100·d_seg + sqrt(10·d_pose)` is a
**frozen, deterministic, piecewise-real-analytic closed form** of the decoded frames. There is no
stochastic element and no oracle. Every non-analytic locus is EXACTLY characterized:

- **PoseNet body is real-analytic**: `upstream/modules.py:25` sets `ACT_LAYER='gelu_tanh'` and
  `:66` builds `fastvit_t12` with it — GELU-tanh compositions, RepMixer branches fuse to plain
  convs at inference (closed-form reparameterization). The Hydra head (`:38-57`) is **ReLU =
  exactly piecewise-affine**: within a fixed activation pattern the head is EXACTLY
  linear, so the 6-dim pose output is exactly CPWL in the vision features.
- **SegNet** (`:105`, `tu-efficientnet_b2`): SiLU encoder (analytic), smp decoder ReLU blocks
  (CPWL), bilinear upsample (linear), **affine 5×144 terminal head** — the argmax partition is
  EXACT Laguerre cells at the head (long-established; #559 rank-4 contrast structure, ux1-L3).
- The remaining kinks are exactly known combinatorics: **uint8 lattice** (#532), **argmax cell
  boundaries** (analytic margin zero-sets), **tie structure** (deterministic corrector #26), and
  the **coder's integer arithmetic** (exact bit cost by real re-encode, fs2).

Consequence: exact derivatives a.e., exact local expansions with real convergence radii (no
ReLU-kink obstruction in the bodies), and solve-not-train licensed at every upstream-adjacent
surface — limited by combinatorics and THE LAW, never by missing math.

## 2. The atlas — per-stage closed-form holdings

| # | stage | status | receipt |
|---|---|---|---|
| 1 | rate numerator (`archive.zip` st_size) | HELD | trivial exact; ux1-L1 |
| 2 | rate denominator (dynamic rglob) | HELD | #812 guard |
| 3 | GT decode lineage (DALI=CUDA authority / PyAV=CPU) | HELD as fixed constants | gti1; cached lstars + DALI pose6 table; lb1 error 1,717 |
| 4 | resize `D` (shared by BOTH scorers) | HELD | #580 exact separable kernel + adjoint + ker(A) 80.67%; pz1 shared-D; ux1-L2 live-spelling identity 0/353,894,400 |
| 5 | uint8 quantization | HELD as exact lattice | #532 (incl. the measured range(A)-exactness break Δ=62.74) |
| 6 | rgb→yuv6 | HELD | exact affine; differentiable twin in `tac.differentiable_eval_roundtrip` |
| 7 | SegNet body (SiLU analytic + smp ReLU decoder) | **PARTIAL** | exact forwards/Jacobians; ms3/ms4 margin-Fisher row-Gram + per-bucket composite-R Hessian/adjoint; validity radii MEASURED (v16/v17 curves), not certified |
| 8 | SegNet terminal head (affine 5×144) | HELD | #559; argmax = exact Laguerre cells at the head |
| 9 | d_seg (argmax count vs fixed lstars + ties) | HELD | #26 deterministic tie-corrector; canonical compute path #168 |
| 10 | PoseNet body (gelu_tanh analytic; RepMixer fuses) + ReLU Hydra head (exact CPWL) | **PARTIAL** | ms3/ms4 n600 batch32 ≤6-dim quadratic in active-tube form (some NON_CONVERGED blocks); same certified-radius gap |
| 11 | d_pose (exact quadratic vs fixed 6-dim table) | HELD | DALI table wired (#1142); up2 exact GN converged on this structure |
| 12 | coders (HPAC/Brotli/LZMA integer arithmetic) | HELD-BY-COMPUTATION | fs2 real re-encode prices (no cheap analytic form; fs3 average≠marginal 2.24×); cm1 coder-matched surrogate for search only |

## 3. The two honest gaps, priced under the laws

- **G1 — certified trust radii** for the analytic bodies (stages 7/10): v16/v17 measured the
  validity-radius curves empirically; a derived certificate (Hessian-Lipschitz/interval bounds
  through the frozen weights) would convert trust-region choices from measured to derived.
  Consumer: lc3 solve steps + any future ms2-class box solve. PRICED HONESTLY: instrument
  refinement, not a score mover — the solver axis is measured CLOSED (#897: 96.6% of realization
  flips cured, remaining optimality elsewhere; #930: search quality is not the seg lever).
- **G2 — exact-CPWL pose-head cells**: within one Hydra activation pattern the terminal pose
  solve is EXACTLY linear-quadratic. Likely already implicitly exploited by up2's converged GN;
  does not reopen the measured-terminal pose families (pk4/ps135b/js-line failed on HELDOUT
  generalization and realization, not on solver exactness).

Neither gap changes THE LAW (33.7× exchange), the sharp optimum (#1214), or the pose absolute
budget ([[m110]]). Closed forms make search cheap and acceptance exact; they do not make the
trade-offs kinder. The measured walls stand.

## 4. Structural encoding (the doctrine outlives this memo)

1. **Charter template** — `tools/codex_arm_queue._CHARTER_TEMPLATE` HARD CONSTRAINTS now carries
   CLOSED-FORM-FIRST: derive/solve against the exact upstream operators before any fit; a fitted
   stage owes a one-line reason. Every scaffolded charter inherits it automatically.
2. **Live consumer** — lc3 (the D3 Lane-carriage rung) already embodies it: arithmetic floors
   before builds, exact adjoints, real coders, bit-identity mandatory.
3. **Memory** — `closed_form_first_all_upstream_20260831` + MEMORY.md row.

## 5. Denominator

Stages audited: 12 (10 HELD · 2 PARTIAL). Source facts verified this turn: 2 (`gelu_tanh` at
modules.py:25/66 · ReLU Hydra head :38-57). New arms spawned: 0 (the doctrine's consumer is
live; G1/G2 are priced as instrument work and ride the next solve arm's charter, not a slot).
Dollars: 0. Pointer: UNMOVED (doctrine + audit).
