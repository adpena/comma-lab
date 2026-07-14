# Standalone DAG FEED — Task #494 throughput authority ladder

**UTC:** 2026-07-14T03:10:02Z  
**Feed:** `FEED-494-throughput-authority-ladder`  
**Lane:** `throughput_authority_ladder`  
**Status:** `FIXED+DYNAMIC_N600_MEASURED; INT64_CEILING_N600_RUNNING; HOST_GATES_OWED`  
**Authority:** `[research-only MEANS; score_claim=false; pointer_moved=false]`

The shared pursuit DAG is hot. This standalone feed is the collision-safe trajectory leg MAIN may
append after serializer review.

## Executable graph

```text
MEASURED one-axis R-adjoint:
  float atomic = 10 hashes / 10 processes
  Q15/int32 atomic = 1 hash / 10 processes
  bounded error 0.007969 < 0.011561
    -> BUILD four-axis real-n600 full-R probe
    -> MAIN Metal N=10 x {float,integer}
    -> {REAL-L70-LEVER-FULL-R-N600 | scoped failing axis/formulation}
    -> BUILD no-atomic int32 gather VJP
    -> exact NumPy state hash + bounded dequant + repeat + speed
    -> {default-off training backend | fixed-order fp32 fallback}

MEASURED n96 one-thread CPU-Torch verdict:
  59.615 s; 0.621 s/pair
  SegNet share 0.774; PoseNet share 0.226
  DERIVED n600 projection 372.6 s
    -> P0 = SegNet forward verdict

fixed-calibration real n600 W8..W24 QDQ:
  exact pair/hash custody
  W24 = 8,960 flips / 117,964,800
  no arm passes exact or 3.3e-5 tolerance
    -| fixed-calibration FORMULATION
    -> dynamic max(abs(x)) scale [distinct; label-free; order-independent selection]
    -> W16/W18/W20/W22/W24 real n600 [MEASURED]
       W20 first tolerance arm; W24 = 19 flips; no exact arm
    -> W25/W26 real n600 [RUNNING; finite single-int64 ceiling]
       W26 bound 4.782822519e18 < 2^63; W27 exceeds int64
    -> {minimum exact arm | uniform-single-int64 FORMULATION negative}
    -> custom Metal exact-int64 all-Conv2d host gate
    -> exact + certified + 10-process same digest + speed > 1
    -> {local candidate verdict backend | one-thread CPU fallback}

CoreML W8A8 settled 45.836809% held-out flips
  + public CoreML activation precision = int8
  + ANE forward selectors only; zero backward
    -| rerun settled W8A8
    -> if selected bits > 8: PUBLIC-API FORMULATION BLOCK
    -> retain CoreML fp32 advisory only

pose frozen pre-finish, w_pose=0
  + MEASURED Pose share 0.226 at n96
    -> typed PoseVerdictGate default OFF
    -> index0 live + live every K + labelled banked NON-LIVE otherwise
    -> MAIN governed n96 dry-start/resume/drift receipt
    -> {admit pre-finish K | widen cadence | CPU live fallback}
    -> pose engagement => CPU live always

complete QDQ + Metal + integer-R + pose receipts
    -> compile_throughput_authority_policy
    -> append empirical equation anchors
    -> V9.CGauge / pose-finish score-moving run
    -> byte-close archive
    -> exact contest CPU and separately CUDA
    -> pointer may move only here
```

## Triality links

- Equations: `exact_commutative_reduction_reorder_invariance_v1` and
  `interval_argmax_enclosure_certificate_v1`, via
  `throughput_authority_anchors_20260714.py`.
- DSL: `PoseVerdictGate`, `PoseVerdictGateDryStart`, and pure receipt-bound
  `compile_throughput_authority_policy`.
- Synthesis:
  `.omx/research/throughput_authority_ladder_20260714T031002Z.md`.

## Verdict-scope guard

- fixed calibration negative = `FORMULATION`, never family/paradigm;
- dynamic QDQ = numerical feasibility only, never native speed;
- custom direct-int64 Metal negative, if any = kernel formulation;
- public ANE higher-bit block = API/formulation, never ANE family;
- integer R = training reproducibility, never score authority;
- local Metal/ANE/MPS evidence never promotes contest CPU/CUDA by equivalence.
