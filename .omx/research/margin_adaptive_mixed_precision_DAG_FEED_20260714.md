# Standalone DAG FEED — margin-adaptive exact-integer SegNet forward

**UTC:** 2026-07-14T11:22:42Z  
**Feed:** `FEED-494-margin-adaptive-mixed-precision`  
**Lane:** `margin_adaptive_mixed_precision`  
**Status:** `BUILT_AND_LOCAL_VERIFIED; MAIN_M5_MAX_N600_MEASUREMENT_OWED`  
**Authority:** `[research-only throughput MEANS; no Metal execution here]`  
**Flags:** `research_only=true` · `score_claim=false` · `pointer_moved=false`

## Pointer status

The submittable `0.19108282419209976 [contest-CPU]` pointer and the unsubmitted defensive
`0.1880443979880752 [contest-CPU]` bank are unchanged. This feed can move neither. A pointer can
move only after a score-moving run produces a byte-closed archive and exact contest-axis replay.

## Executable graph

```text
MEASURED real-n600 fixed-scale uniform W8..W24 QDQ/fp32 accumulation
  -> NO_ADMITTED_PRECISION_IN_LADDER
  -| calibrated uniform fixed-scale FORMULATION at this n600 INSTANCE
  -> retain distinct exact-integer suite

MEASURED frozen-weight-L1 exact-int64 W27..W31 CPU twin
  + per-output-channel weight scales
  + exact signed-int64 MAC with proved no-overflow bounds
  + frozen design-only ordered (4,0)->0 tie rule at gap <= 2^-19
  -> 0 source-n600 argmax flips in predecessor CPU receipt
  -> derive nested per-layer cap profiles 8,10,...,31
  -> lower a full frozen SegNet for every profile into the existing custom-Metal kernel
  -> pack each layer into the narrowest exact signed {int8,int16,int32} operand storage bucket
     while retaining exact signed-int64 accumulation
  -> design pairs 0..263:
       rank certificate-preserving profiles by measured Metal seconds/pair,
       then MAC-weighted logical bits; also report the distinct minimum-bit profile
  -> freeze selected profile (no second-validation reselection)
  -> second validation pairs 264..599
  -> full source n600 exact/certificate custody
  -> ten fresh-process selected-profile argmax digest
  -> synchronized CPU-Torch one-thread versus selected custom-Metal latency
  -> {ADMIT local candidate iff all gates and speedup>1 | scoped negative + refine profiles}

per-pixel margin waterfill over all measured profiles
  -> minimum average bits among certifying finite-ladder profiles at each source pixel
  -> report by margin band
  -| native spatial speed claim
     23 squeeze-excite global reductions + measured skip-inclusive halo685 close exact support
     to the full frame
  -> diagnostic lower bound only; executable treatment remains one frame-independent per-layer map

DERIVED frozen-model storage phase boundary
  -> cap8 = int8 operands; caps10..16 = int16 operands; caps18..31 = int32 operands
  -> cap18+ can admit exact-Metal placement but not margin-adaptive physical-width reduction
  -> receipt keeps these verdicts separate

admitted local candidate
  -> typed default-OFF authority integration
  -> score-moving witness run
  -> exact archive bytes + inflate receiver closure
  -> exact contest-CPU and separately contest-CUDA replay
  -> only then may the frontier pointer move
```

## Canonical law

For reference winner `a`, profile `k`, and corpus-observed classwise enclosure radius `e`:

```text
L[p,a,k] = z_fp32[p,a] - e[p,a,k]
U[p,c,k] = z_fp32[p,c] + e[p,c,k]
C[p,k]   = (L[p,a,k] > max_{c != a} U[p,c,k]) OR frozen_tie_rule[p,k]
k_native = argmin_k t[k]
           subject to C[p,k]=1 for every design and validation pixel,
                      H_k^(1)=...=H_k^(10), and t_fp32/t[k] > 1
k_pixel  = argmin_{k:C[p,k]=1} average_bits[k]
A_conv   = sum_i qx_i * qw_i, |A_conv| <= 2^63-1
```

The strict interval clause is a sufficient per-pixel certificate. The frozen tie clause handles
the one reference zero-margin boundary where strict separation is impossible. `e` is
`CORPUS_OBSERVED_PER_PIXEL_ABS_FP32_VS_FIXEDPOINT_LOGIT_ERROR`; this feed makes no unseen-input IBP
claim. Exact integer accumulation means exact, reorder-invariant arithmetic over the integer codes;
it does **not** mean the quantized logits equal fp32. Decision preservation comes from the margin
certificate plus the frozen tie rule.

## Triality

- **DSL:** `tac.witness_dsl.margin_adaptive_mixed_precision_20260714` seals n600, the split,
  exact-int64/per-channel rules, native int8/int16/int32 operand buckets, ten processes,
  resume/checkpoints, and no-score/pointer claims.
- **Equation:** `margin_adaptive_integer_profile_waterfill_v1` is registration-inert until MAIN
  supplies the complete M5-Max receipt; it composes
  `exact_commutative_reduction_reorder_invariance_v1`,
  `interval_argmax_enclosure_certificate_v1`, and `decode_determinism_integer_arithmetic_v1`.
- **DAG:** this collision-safe standalone feed; MAIN may merge it after review.

## Six-hook research-only wire-in

1. **Sensitivity map:** source winner/rival margin supplies the exact certificate priority; no
   input-gradient or unseen-input robustness claim.
2. **Pareto:** profile selection ranks certificate-preserving measured latency first, average bits
   second; unmeasured bit count cannot win.
3. **Bit allocator:** consumes per-layer safe ceilings and per-output-channel scales; the profile
   manifest exposes MAC-weighted average bits and measured cost.
4. **Cathedral/autopilot:** incomplete n600, validation reselection, missing digest, or no speedup
   fails closed and cannot dispatch or promote.
5. **Continual learning:** MAIN's complete receipt is the owed empirical anchor for the canonical
   equation and policy; no anchor is fabricated here.
6. **Disambiguator:** native frame-independent per-layer allocation and retrospective per-pixel
   waterfill remain separate named surfaces.

## Host handoff

MAIN on the M5-Max runs exactly:

```zsh
./tools/run_margin_adaptive_mixed_precision_n600_host.command
```

The command is resumable, checkpoints every pair and each terminal stage, binds the real n600 GT
cache plus all predecessor receipts by content hash, and writes durable output under
`experiments/results/margin_adaptive_mixed_precision_20260714/`.

## Verdict ladder and reformulation queue

If no native profile is admitted, the negative is the `FORMULATION-at-n600-INSTANCE` comprising the
supplied finite cap ladder, per-layer frame-independent mapping, current custom-Metal lowering, and
frozen tie rule. It does not kill per-channel/per-tile/per-pixel allocation, analytic affine bounds,
multi-limb accumulation, ANE/CUDA placement, mixed precision as a family, or the witness paradigm.
