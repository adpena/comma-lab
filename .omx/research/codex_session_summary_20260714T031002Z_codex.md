# Codex session summary — Task #494 throughput authority ladder

**UTC:** 2026-07-14T03:10:02Z  
**Lane:** `throughput_authority_ladder`  
**Status:** `BUILD_COMPLETE; QDQ_UNIFORM_AND_GEOMETRY_INT64_N600_MEASURED; WEIGHT_L1_INT64_N600_RUNNING; HOST_DEVICE_GATES_OWED`
**Flags:** `research_only=true` · `score_claim=false` · `pointer_moved=false`

## TIER-0 outcome

Task #494 now has a fail-closed heterogeneous authority policy and executable host packet rather than
an architectural conjecture. The fixed-calibration n600 formulation is decisively negative. A
distinct dynamic max-absolute QDQ formulation is closed through its W26 single-int64 ceiling. Full-R
integer accumulation, a no-atomic integer R-adjoint backend, an all-Conv2d custom-Metal fixed-point
SegNet, an ANE formulation ticket, a pose-frozen canary gate, policy compiler, and empirical equation
anchors are built and covered by CPU contract tests. Device execution remains with MAIN's M5-Max as
directed.

The pointer is unchanged: submittable `0.19108282419209976 [contest-CPU]`; exact
borrowed-lineage defensive bank `0.1880443979880752 [contest-CPU]`.

## Measured this session

- Fixed calibration, real n600, one-thread CPU-Torch control, W8..W24 QDQ/fp32 accumulation:
  **no admitted fixed-point precision**. W24 has 8,960 / 117,964,800 flips, aggregate
  `7.595486111111111e-05`, worst pair `9.358723958333334e-04`.
- Cache audit: one argmax pixel in one pair differs from the recomputed control; maximum margin delta
  `3.6239624e-05`. Computed control owns authority.
- Dynamic full n600: no exact arm through W24; W20 first tolerance arm; W24 has 19 flips /
  117,964,800, aggregate `1.6106499565972223e-07`, worst pair `5.086263020833333e-06`, and
  244 conservative uncertified pixels. Receipt SHA-256
  `feaf29ab8d1ca3fef20976586141b57dcfdb6da23c77140d142813c02f97fb5f`.
- n96 local verdict timing from MAIN: 59.615 seconds total, 0.621 seconds/pair, SegNet 77.4%, PoseNet
  22.6%, `[macOS-CPU Torch one-thread advisory]`. n600 372.6 seconds is DERIVED.

- Corrected finite single-int64 QDQ ceiling, exact pairs 0..599: W25 has 13 flips / 139 uncertified;
  W26 has 3 flips / 83 uncertified, aggregate `2.5431315104166668e-08`, worst pair
  `5.086263020833333e-06`. Corrected receipt SHA-256
  `a04a8e2672981faeda9a2a1adb086c8e1a4c073c0e1319dcd78ee1536c594c91`.
- Uniform exact-int64 W26 CPU twin, exact pairs 0..599: 4 flips / 117,964,800 at pairs 64, 362,
  371, and 507; aggregate `3.390842013888889e-08`, 77 conservative uncertified pixels, maximum
  logit error `2.525597810745239e-04`. This is an `INSTANCE` negative. Receipt SHA-256
  `b4bd48f580501926492d826a8a2504f5420fa266d6270f4aff915e7820f60af2`.
- Geometry-safe mixed exact-int64 successor, exact pairs 0..599: 1 flip / 117,964,800 at pair 11;
  aggregate `8.477105034722222e-09`, 38 conservative uncertified pixels, maximum logit error
  `7.62939453125e-05`. This is an `INSTANCE` negative; receipt SHA-256
  `129e9d39d09ff2e019cdab7ac04f699b64a846d319390d71d3bd12d9497959f5`.
- Frozen-weight-L1-safe successor passed the full geometry-negative gate and is running resumably:
  label/frame-free static int64 proof, histogram W27:4, W28:28, W29:32, W30:41, W31:20,
  worst bound `9,035,402,569,620,285,889`, with CPU/Metal twins, pair-atomic n600 runner, typed
  policy, and canonical anchor.

## Built

1. Full four-axis render-R adjoint N=10 cross-process real-n600 probe and host command.
2. Calibrated/dynamic frozen-scorer QDQ feasibility probe with exact row/hash custody, margin
   diagnostics, continuous Pose debt, resumability, and no native-speed claim.
3. Custom direct NHWC Metal fixed-point Conv2d with exact signed int64 accumulation, all 125 SegNet
   Conv2d replacement, NumPy integer reference, and full n600/cross-process/latency host gate.
4. No-atomic integer render-R VJP gather with raw int32 NumPy parity, static overflow, bounded
   dequant, repeat, and speed admission.
5. Settled-state-aware ANE ticket compiler that refuses duplicate W8A8 and public-API-unrepresentable
   higher-bit requests.
6. Typed pre-pose-finish Pose verdict gate wired across every trainer verdict branch, with index-0 and
   periodic live canaries, explicitly banked telemetry, resume-persisted monotonic index, and dry-start
   host command.
7. Pure receipt-bound op/substrate/precision policy with unconditional CPU fallback and explicit MPS,
   settled-W8A8, contest-CPU, and contest-CUDA authority separation.
8. Append-only empirical anchor builder plus collision-safe standalone DAG feed and synthesis memo.
9. Uniform and geometry-safe mixed exact-int64 CPU twins for all 125 Conv2d, per-pair resumable n600
   probes, bound Metal precursor gates, and distinct equation anchors.
10. Split-honest lowest-class epsilon tie-snap decision head: preregistered dyadic ladder,
    calibration-only selection, heldout validation without reselection, pair-atomic n600 probe,
    NumPy/MLX twins, canonical anchor, policy gate, and synchronized Metal-host binding.

## Adversarial bugs extinguished

- fp32 control falsely selected as the minimum fixed-point arm;
- incomplete/duplicate n600 pair sets could lack exact digest custody;
- custom-Metal fidelity child overwrote computed CPU labels with legacy cache labels;
- QDQ feasibility could be mistaken for native integer speed;
- full-R equation anchor schema mismatch;
- integer-R admission lacked exact raw-state parity and retained unnecessary full-frame views;
- Pose timing documentation mislabeled the DERIVED n600 projection as measured.
- W26 positive qmax was rounded upward by fp32 before clamping; all paths now clamp exact integers.
- QDQ/fp32 accumulation was at risk of being mislabeled as the actual direct-int64 formulation.

The detailed review is
`.omx/research/codex_findings_throughput_authority_ladder_20260714T031002Z_codex.md`.

## Host packet owed to MAIN

Run on the M5-Max host in this order after the dynamic QDQ receipt is complete:

```bash
tools/run_full_r_adjoint_bitident_host.command
tools/run_mixed_int64_scorer_forward_n600_host.command
tools/run_fixedpoint_authority_kernels_host.command
tools/run_integer_r_adjoint_backend_host.command
tools/run_pose_verdict_gate_dry_start_host.command
tools/run_ane_fixedpoint_authority_host.command
tools/run_throughput_authority_policy_host.command
tools/run_throughput_authority_anchor_registration_host.command
```

Paid/CUDA dispatch, live-run/config change, and any run stop remain operator-GO containment. Local
Metal results remain research-signal/local-candidate evidence and never promote contest axes by
equivalence.

## Recommended next action

The highest-EV next rung is conditional and fail-closed: finish the frozen-weight-L1-safe W27..W31
exact-int64 CPU twin. If exact, MAIN runs its bound full-n600 custom-Metal receipt (one fidelity
process, ten total processes, exact argmax, one digest, placement, and speed). If non-exact, do not
spend Metal time on it; the next formulation is multi-limb exact accumulation or a separately
measured sparse near-tie correction ladder.

## Inbox/directive custody

Consumed per-arm directives through `2026-07-14T02:12:22Z` and fleet-wide directives through
`2026-07-14T03:06:30Z`. The latter clarified that ordinal concordance is stronger than required and
that the throughput next step remains full-n600 fixed-point QDQ followed by true integer Metal/ANE
placement and latency. No stop directive was received.
