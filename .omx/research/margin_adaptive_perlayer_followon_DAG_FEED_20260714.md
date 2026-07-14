# Standalone DAG FEED — per-layer/per-channel genuine adaptive width

**UTC:** 2026-07-14T12:16:00Z  
**Feed:** `FEED-494-margin-adaptive-perlayer-followon`  
**Lane:** `margin_adaptive_perlayer_followon`  
**Status:** `BUILT_REAL_RESUMABLE_EXECUTOR; V9_INTEGRATION_BLOCKED_OWNER; MAIN_NATIVE_METAL_N600_MEASUREMENT_OWED`  
**Authority:** `[research-only throughput MEANS; no Metal execution in this landing]`  
**Flags:** `research_only=true` · `score_claim=false` · `pointer_moved=false`

## Pointer status

The submittable `[contest-CPU Linux x86_64]` pointer remains
`0.19108282419209976`. The `0.1880443979880752` defensive bank is explicitly
non-submission. This lane changes neither: only byte-closed exact contest-axis
evaluation can move the pointer.

## Recalled receipt and scope

The predecessor receipt
`experiments/results/margin_adaptive_mixed_precision_20260714/margin_adaptive_mixed_precision_n600.json`
(SHA-256 `bd8e83704d23518a725bbea4f4404e84d1f07b6efc0b30c7e7104d20f8d56b4c`)
is an exact real-n600 `0..599` receipt. Its cap28/cap30/cap31 zero-flip rows
use physical int32 operands. Its cap16 physical-int16 row has 5,446 flips.
Therefore that global-cap result establishes an exact seed and a placement
candidate, not an admitted adaptive-width latency result.

This follow-on is the queued optimal formulation: finite per-layer activation
width plus per-output-channel weight width, physically typed native buffers,
exact signed-int64 accumulation, full-fp SE control, and a same-map
forced-int32 control. The public/monolithic int16 absence and uniform-QDQ
results are TESTED-BACKEND / naive-formulation instances only; neither closes
the adaptive-width family.

## Executable graph

```text
completed real-n600 exact-int predecessor receipt
  -> plan-only hash/custody rederivation (local, $0, atomic output)
  -> frozen cap28 seed; full-fp SE + high-width segmentation head control
  -> MAIN design split 0..263, finite registered ladder {8,16,28}
       -> execute/certify the seed under this exact kernel + full-fp SE form
       -> every lower per-layer candidate (no monotonic pruning)
       -> freeze selected layer map
       -> per-channel risk-prefix refinement + one exchange pass
       -> label BEST_FOUND_CERTIFIED, not global optimum
  -> untouched validation split 264..599 (no reselection)
  -> exhaustive tie-aware decision identity + exceptional witness
  -> ten fresh-process decision digest
  -> four arm timing:
       U32_M seed uniform int32 Metal
       A32_M selected adaptive codes/map, forced-int32 buffers
       AN_M same selected map, native int8/int16/int32 buffers
       U32_C int32-storage/int64-MAC CPU reference control with matching fp32 SE
  -> admit native-width timing only if A32_M vs AN_M has positive measured
     same-map latency improvement; report all components without clipping
  -> later receiver-close archive and contest CPU/CUDA replay
```

## Canonical law

For layer `ell`, channel `c`, activation qmax `q_a`, integer weight codes
`q_w`, and int64 accumulator:

```text
B[ell,c] = q_a(ell) * sum_i abs(q_w[ell,c,i]) <= 2^63 - 1
A[ell,c] = sum_i q_x[ell,i] * q_w[ell,c,i]          (signed int64)

rho[p,r] = abs(center[p,w] - center[p,r])
           + ||alpha[p,w] - alpha[p,r]||_1 + rem[p,w] + rem[p,r]
certificate[p,r] = margin_fp32[p,w,r] - rho[p,r]

S_width = t(A32_M) / t(AN_M)
S_place = t(U32_C) / t(U32_M)
S_total = t(U32_C) / t(AN_M)
f_width_latency = (t(A32_M)-t(AN_M))/(t(U32_C)-t(AN_M))
f_width_log = log(S_width)/log(S_total)
```

`A32_M` and `AN_M` retain the same logical codes, qmax values, scales,
channel partition, launch count, accumulation and finalization. They differ
only in physical operand storage; this is the genuine-width isolation. The
existing long-cast int64 MAC has unchanged conventional MAC/FLOP count, so
this lane reports conventional FLOP cut `0.0` unless a future native narrow-MAC
kernel proves otherwise.

`U32_C` also preserves the authority logical codes at 27/28 bits: it rounds
into int64, clamps against the exact integer qmax, stores int32, and expands
only at the MAC. Float-domain clamping is forbidden because those qmax values
are not exactly representable in fp32.

The primary finite-corpus certificate is exhaustive decision identity with
frozen lowest-index tie semantics. Robust observed intervals and affine
shared-symbol bounds are separate statistics. An affine result is
`UNANCHORED_SCREEN_ONLY` absent sound nonlinear remainder provenance.

## SE control and reformulation queue

The real model has 23 SE blocks / 46 SE convolution paths: 1,313,728 of
9,909,333,952 Conv MACs (`0.01325748%`, DERIVED). A real pair-0 CPU shape trace
measures 14,533,632 SE feature elements. With the recorded two-FLOP/MAC and
two GAP+gate scalar-op/feature convention, the whole SE control is
31,694,720 operations (`0.15992356%`), leaving `99.84007644%` in the bulk
adaptive path. It does not permit a spatial-skip claim: upstream error still
reaches global pooling and its broadcast gate.

If a selected finite map fails, scope it to the tested
`INSTANCE < FORMULATION < FAMILY < PARADIGM` level and retain these live
reformulations: stronger per-channel grouping/search; sound affine remainder
propagation; exact global-summary/gate closure for spatial allocation; and a
native packed narrow-MAC kernel. No first-cut negative is a family closure.

## Queued compositions (not measurements)

- **L70 determinism:** native-width operands sign-extend before the exact
  signed-int64 MAC. This makes the selected treatment a reorder-invariant
  decision-authority candidate, subject to n600 identity/digest proof; it is
  neither a score nor a latency claim.
- **Molt router analogy:** `segmentation_head.0` remains at the high-width
  seed while adaptive width targets the bulk feature graph. This is a typed
  composition boundary, not evidence from an LLM router.
- **Distilled-surrogate successor:** only after a frozen map hash and n600
  teacher identity may a successor train against this per-layer/per-channel
  quantized teacher. Decision-quotient fidelity and its estimator remain owned
  by `surrogate_vjp_fidelity_metric`; this lane neither duplicates nor promotes
  that estimator.

## Triality and six-hook wiring

- **DSL:** `tac.witness_dsl.margin_adaptive_perlayer_20260714` seals real n600,
  split, ladder, ten-process digest, resumable checkpoints, four arms, no score,
  plan-only versus MAIN Metal modes, and the only built SE mode (`full-fp`).
  Unimplemented `adaptive-int` SE is rejected at both DSL and plan parsing.
- **Equation:** `margin_adaptive_perlayer_channel_waterfill_v1` records finite
  search scope, physical-byte accounting, tie-aware corpus certificate, and
  the same-map decomposition; it is registration-inert awaiting MAIN.
- **DAG:** this collision-free feed records predecessor custody and remaining
  gates without mutating the hot shared DAG.
- **Sensitivity map:** winner/rival margin ranks finite-map candidates; this
  makes no unseen-input robustness claim.
- **Pareto / bit allocator:** select only decision-preserving maps, account
  physical operands with one shared per-layer activation buffer across all
  per-channel weight buckets, then rank direct bytes before modeled bits.
  Duplicate activation traffic is charged only by a separately named future
  per-channel-activation extension.
- **Autopilot / continual learning:** missing custody, validation reselection,
  absent ten-process identity, or missing same-map timing fail closed; a MAIN
  receipt is the owed empirical anchor.
- **Disambiguator:** native-storage and forced-int32 storage are both callable
  and make the width-versus-placement attribution observable.

## MAIN handoff

On the Metal host, after the typed DSL/compiler and plan-only custody check
pass, MAIN may emit/run:

```zsh
./tools/run_margin_adaptive_perlayer_host.command
```

The wrapper routes `MODE=plan-only` to the custody/SE remeasurement tool and
the default Metal mode to the real staged executor. The executor recomputes
live fp32 reference decisions (rather than trusting the one-pixel-stale cached
label at pair 195), validates its own seed, checkpoints every candidate and
validation pair, freezes before validation, and executes all four arms plus
ten fresh-process trials. Candidate bulk follows VertigoDataTier ->
APDataStore -> explicit-local-opt-in and is removed only after a durable
tree-hash cleanup certificate. Search, n600 identity, and timings remain
`OWED_MAIN` because this landing has no Metal. The highest-EV measured outcome
remains the same frozen map's `A32_M` versus `AN_M` timing.

## Post-build v9 integration gate

The binding `2026-07-14T13:22:43Z` fleet directive forbids promotion of a
standalone side module. This feed and its external probe policy are therefore
not a live v9·CGauge lever yet. The owner
`provenance_canonicalize_fix_all_fakes` must add the sole canonical
DSL/LawRef/consumer/receipt edge in its witness-autoconfig/config/bijection
surfaces; this lane must not collide there.

The consumer is the asynchronous/local SegNet verdict-forward path only. The
kernel is inference-only and cannot masquerade as the training-loss backward
path. The live edge must fail closed on receipt schema
`margin_adaptive_perlayer_metal_n600.v1` unless full n600 identity, ten fresh
processes, and positive same-map `A32_M/AN_M` timing all hold. Official
evaluator replay remains the only score authority. Until a compiled v9 argv and
consumer receipt exist, the honest status is `V9_INTEGRATION_BLOCKED_OWNER`.

## Stores consulted

`CLAUDE.md`, `AGENTS.md`, frozen build specification, canonical lane/ownership
surfaces, predecessor code/receipt, latest findings/session summaries, all
last-24-hour directives, and both live inboxes through
`2026-07-14T13:10:48Z` (per-arm empty). No provider, Metal, evaluator, or
live run was launched.
