# Build specification — per-layer and per-channel genuine adaptive width

**Date:** 2026-07-14  
**Lane:** `margin_adaptive_perlayer_followon`  
**Mode:** build + saved-artifact `$0` remeasurement; MAIN owns Metal execution  
**Pointer:** unchanged. Submittable `[contest-CPU Linux x86_64]` remains
`0.19108282419209976`; `0.1880443979880752` is a non-submission defensive
bank. This build has no archive or score authority.  
**Verdict ladder:** every negative is scoped
`INSTANCE < FORMULATION < FAMILY < PARADIGM`; a first-cut result cannot close
the adaptive-width family.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, the craft handoff, SPEC v7.5 §8,
  and SPEC v8.
- `reports/latest.md` plus a live `tac.frontier_scan` payload.
- `.omx/state/lane_registry.json`, `subagent_progress.jsonl`, gradient,
  dispatch, cost, continual-learning, probe, task, council, and operator
  authorization surfaces required by preflight.
- Latest Codex findings/session summaries, latest T3/design memo, all
  `*_directive_*` files from the preceding 24 hours, and live inboxes through
  `2026-07-14T11:57:05Z`.
- Commit `c12957757e` and its exact-int64 allocator, Metal adapter, probe,
  typed DSL, canonical equation, tests, and DAG FEED.
- Completed n600 receipt
  `experiments/results/margin_adaptive_mixed_precision_20260714/margin_adaptive_mixed_precision_n600.json`
  (SHA-256 `bd8e83704d23518a725bbea4f4404e84d1f07b6efc0b30c7e7104d20f8d56b4c`).
- Frozen source cache SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`
  and SegNet weight SHA-256
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`.

## Re-derived starting evidence

The completed receipt covers exact pair indices `0..599`. It measures cap28,
cap30, and cap31 at zero observed argmax flips, but all three use physical
int32 operands. Cap16 uses physical int16 operands and has 5,446 flips. The
existing zero-flip rows therefore establish an exact-cap seed and Metal
placement, not a genuine adaptive-width win. This is an `INSTANCE` result for
the global-cap formulation; the per-layer/per-channel optimal form is live.

The frozen model has 125 Conv2d layers and 9,909,333,952 Conv MACs per pair.
Its 23 SE blocks contain 46 Conv2d layers and 1,313,728 Conv MACs, a DERIVED
0.01325748% of Conv MACs. A real source-pair-0 CPU forward shape trace measures
14,533,632 SE feature elements. Under the explicit convention Conv MAC = two
FLOPs and GAP plus gate application = two scalar operations per feature
element (excluding implementation-dependent SiLU/sigmoid transcendental
cost), the full-fp SE control is DERIVED as 31,694,720 operations, 0.15992356%
of the Conv-FLOP baseline. Thus 99.84007644% remains in the bulk adaptive
treatment under that convention. This is an operation-count fact, not a
latency claim.

The existing Metal kernel loads narrower buffers but casts each operand to
signed `long` and executes the same int64 MAC loop. Consequently:

- conventional MAC count and conventional FLOP cut are exactly zero;
- narrower buffers can reduce operand traffic;
- only a same-logical-map native-storage versus forced-int32 Metal A/B can
  attribute latency to genuine width;
- CPU-to-Metal timing by itself is placement/determinism evidence, not width
  evidence.

## Objective and exact scope

Build the strongest executable finite-ladder formulation that can be reviewed
without Metal here:

1. choose activation and weight operand storage independently per SegNet
   layer over physical `{int8, int16, int32}`;
2. refine weight width per output channel inside a layer, with physically
   separate typed buffers rather than an int32 tensor plus width metadata;
3. preserve exact signed-int64 accumulation, per-output-channel scales,
   deterministic fp32 finalization, and a static no-overflow proof;
4. search on frozen design pairs `0..263`, freeze the map, and validate without
   reselection on untouched pairs `264..599`;
5. emit an exact map, a finite-corpus certificate, byte/traffic accounting,
   an SE-full-fp control, and a four-arm host timing receipt;
6. keep affine arithmetic honest: build a correlation-aware bound primitive,
   but admit it only when the caller supplies sound coefficients and a sound
   nonlinear remainder bound. Endpoint error may not be renamed affine.

"Minimum" means the coordinatewise/groupwise minimum produced by the registered
best-improvement finite-ladder search, not a global optimum over an exponential
layer/channel space. The receipt must preserve the search trace and use
`BEST_FOUND_CERTIFIED` unless every lower choice for the stated coordinate or
group was tested or soundly excluded. Width feasibility is not assumed
monotone.

## Owned files

Additive, collision-free files only:

- `src/tac/local_acceleration/perlayer_adaptive_width.py`
- `src/tac/local_acceleration/metal_perchannel_adaptive_width.py`
- `src/tac/local_acceleration/tests/test_perlayer_adaptive_width.py`
- `src/tac/local_acceleration/tests/test_metal_perchannel_adaptive_width.py`
- `tools/probe_margin_adaptive_perlayer_n600.py`
- `src/tac/tests/test_probe_margin_adaptive_perlayer_n600.py`
- `tools/probe_margin_adaptive_perlayer_metal_n600.py`
- `src/tac/tests/test_probe_margin_adaptive_perlayer_metal_n600.py`
- `tools/run_margin_adaptive_perlayer_host.command`
- `src/tac/witness_dsl/margin_adaptive_perlayer_20260714.py`
- `src/tac/witness_dsl/tests/test_margin_adaptive_perlayer_20260714.py`
- `src/tac/canonical_equations/margin_adaptive_perlayer_waterfill_20260714.py`
- `src/tac/canonical_equations/tests/test_margin_adaptive_perlayer_waterfill_20260714.py`
- `experiments/results/margin_adaptive_perlayer_followon_20260714/plan_only.json`
- `.omx/research/margin_adaptive_perlayer_followon_plan_only_20260714.json`
- `.omx/research/margin_adaptive_perlayer_followon_DAG_FEED_20260714.md`
- a dated `codex_findings_*_codex.md` and one TIER-0 session summary.

Do not edit the trainer, shared DAG, `src/tac/witness_dsl/__init__.py`, resume
registry, upstream, or sister-owned files. The lane registry/audit are already
mutated through `tools/lane_maturity.py` and stay outside the code commit.

## Core data model and arithmetic

### Layer/channel map

Represent each integer Conv2d with:

- one activation logical width and storage bucket;
- one logical weight width per output channel;
- channel partitions by `(logical_bits, physical_storage_bits)`;
- explicit full-fp paths for the SE control;
- exact map SHA-256 and complete path/channel coverage.

The executable ladder is `{8, 16, 28}` logical bits, mapping to physical
`{int8, int16, int32}`. Logical 28 is the lowest completed exhaustive-zero-flip
global-cap seed. A future caller may supply a different registered ladder, but
the host policy is frozen to this one for attribution.

`segmentation_head.0` stays at the exact high-width seed in the primary search.
This is the Molt-style exact-decision-head composition: the bulk feature graph
is the adaptive treatment and the five-class decision head is the high-width
control. A separately named head-adaptive formulation may be tested later but
cannot be silently mixed into the primary map.

For output channel `c`, validate

`B_lc = qmax(a_l) * sum_i abs(qweight_lc_i) <= INT64_MAX`.

No saturation, wrap, or implicit dtype coercion is permitted. Finalization is
one fp32 multiply by the dynamic activation scale and the channel weight scale,
then fp32 bias.

### Physically genuine per-channel execution

Build one typed activation buffer for the layer's single registered activation
width and share it across the physically separate per-output-channel weight
buckets. Launch one width-specialized kernel per weight bucket and restore
original channel order with one cached inverse permutation. Re-quantizing the
same activation once per weight bucket is avoidable overhead and is forbidden
in this formulation. A future per-channel-activation extension may materialize
multiple activation buffers, but it must account that overhead separately. A
forced-int32 mode must keep identical logical codes, qmax, scales, channel
partition, launch count, accumulation, finalization, and gather permutation
while widening only the shared activation and weight buffers to int32. Native
and forced-int32 outputs must be bit-identical in the deterministic NumPy
reference and digest-identical on MAIN.

Per-channel metadata over a single int32 weight tensor is explicitly
insufficient and must fail the `physical_width_implemented` gate.

### Cost accounting

Report separately:

- conventional MACs and FLOPs (unchanged; cut `0.0`);
- int-treated MAC share and full-fp SE MAC share;
- resident operand bytes;
- direct-kernel operand-load bytes, with one shared activation buffer per layer;
  a future multiple-activation-width extension must charge every duplicate;
- output/fp32/accumulator traffic held constant;
- a modeled logical operand-bit statistic, explicitly not hardware compute;
- measured native Metal latency, never inferred from byte count.

## Certificate case split

The final finite-corpus certificate has three non-interchangeable parts:

1. **Exhaustive identity:** candidate and reference decision arrays agree at
   every source-n600 pixel, with exact pair set, array hashes, map hash, and
   frozen tie law. This certifies the enumerated corpus only.
2. **Robust interval subset:** for reference winner `w`, require
   `L_w > max_{r != w} U_r` using componentwise observed final-logit error.
   Report covered coordinates separately. This is corpus-observed, not
   unseen-input IBP.
3. **Deterministic exceptional witness:** uncovered coordinates must still
   have exhaustive identity in ten fresh processes; exact zero-margin cases
   require the frozen ordered tie law. Store counts and hashes. Do not call
   this subset interval-robust.

The full n600 admission requires exhaustive identity, the exceptional witness,
ten-process digest identity, exact custody, and positive measured native-width
latency versus the same-map forced-int32 control. Strict interval coverage is a
robustness statistic, not an over-strong all-pixel replacement for the actual
argmax objective.

The optional affine primitive represents class error as
`e_c = center_c + alpha_c^T epsilon + rho_c`, `epsilon_j in [-1,1]`. For
winner `w` and rival `r`, it returns the correlated radius
`abs(center_w - center_r) + sum_j abs(alpha_wj - alpha_rj) + abs(rho_w) +
abs(rho_r)`, retaining common-mode cancellation that independent class
intervals lose. It may tighten a supplied sound interval by retaining shared
symbols. The probe must label it
`UNANCHORED_SCREEN_ONLY` unless dynamic absmax, SiLU, sigmoid, skips, resize,
and global SE remainder bounds are all supplied. No admission may depend on an
unsound linearization.

## Search and resumability

### Stage 0 — plan-only remeasurement

Verify the completed predecessor receipt and its custody hashes, extract the
lowest exhaustive-zero-flip cap seed, re-derive the exact per-layer seed map,
MAC/storage/SE accounting, and write `plan_only.json` atomically. The plan also
runs one real pair-0 CPU shape trace for the transparent SE operation count.
This is the only local execution required in this no-Metal lane. Missing or
drifting custody fails closed without deleting source bytes.

### Stage 1 — per-layer best-improvement waterfill on MAIN

Starting from the frozen exact seed, evaluate every registered lower width of
one layer on design pairs; never binary-search or prune by a monotonicity
assumption. Accept the feasible candidate with maximum direct-kernel byte
saving; use modeled bit statistic and measured design latency only as
deterministic tie-breaks. Repeat until no direct single-layer change in the
registered set is feasible. Preserve every candidate row and atomic checkpoint.
This produces a coordinatewise minimum over the explicitly tested layer
coordinates and ladder, not a global minimum.

### Stage 2 — per-channel refinement on MAIN

Within integer layers, rank output channels by a declared quantization-risk
proxy and test every registered width at every registered deterministic
channel-prefix fraction. Do not binary-refine or prune by monotonicity. Every
accepted change is replayed end to end on the design split. Then run one
pairwise exchange pass across adjacent risk groups. The receipt states this
finite groupwise search scope and uses `BEST_FOUND_CERTIFIED`; it does not
claim a global per-channel optimum.

### Stage 3 — freeze and untouched validation

Freeze the selected map before reading pairs `264..599`. Do not repair or
reselect from validation. A failed validation is an `INSTANCE` result for that
map/search trace and queues stronger search/sound-affine reformulations; it
does not close per-channel adaptive width.

### Stage 4 — identity and timing

After full n600 identity, run ten fresh-process digests and rotate arm order by
pair. Preserve all stage receipts. Resume fingerprints bind code, weights,
cache, predecessor receipt, map, candidate ladder, split, and host fingerprint.
Reference decisions are recomputed from live NumPy-fp32 logits and matched to
every predecessor hash; cached `lstars` is not authority because real design
pair 195 differs at one pixel. Candidate sidecars checkpoint every pair on the
SSD waterfall, bind the full run fingerprint, and are deleted only after a
machine-readable tree-hash rebuildability certificate is durable.

## Required timing arms and decomposition

MAIN must measure four arms on identical pair order and synchronized device
completion:

1. `U32_M`: exhaustive-exact cap seed, uniform int32 physical storage on Metal;
2. `A32_M`: selected adaptive logical map, same bucket partition, buffers forced
   to int32 on Metal;
3. `AN_M`: the same selected map with native int8/int16/int32 buffers on Metal;
4. `U32_C`: uniform-int32-storage CPU numerical control that expands operands
   into exact int64 accumulation while keeping SE full-fp.

`U32_C` logical codes MUST use the authority ordering: round to int64, clamp
against the exact integer qmax, then store int32. Float-domain clamping is
invalid at 27/28-bit qmax because those limits are not exactly representable
in fp32.

The decisive width isolation is `A32_M` versus `AN_M`. Report

- `width_speedup = t_A32_M / t_AN_M`;
- `metal_placement_speedup = t_U32_C / t_U32_M`;
- `total_speedup = t_U32_C / t_AN_M`;
- `genuine_width_fraction = (t_A32_M - t_AN_M) / (t_U32_C - t_AN_M)`.
- `genuine_width_log_fraction = log(t_A32_M / t_AN_M) / log(t_U32_C / t_AN_M)`.

Do not clip negative components. If the CPU path is only the deterministic
reference and not a kernel-isomorphic native implementation, label placement,
total, and fraction timing `CPU_REFERENCE_IMPLEMENTATION_BOUND`, not
hardware-optimal. Width admission still rests on the same-map Metal comparison.

The receipt must also report `U32_M` versus `A32_M`; this exposes any runtime
effect from logical-code/map structure that is neither placement nor physical
storage width.

## SE-full-precision control

One registered formulation leaves all `*.se.conv_reduce` and
`*.se.conv_expand` paths in fp32 and applies adaptive integer arithmetic to the
remaining full-frame convs. It reports the structural operation fraction and a
measured host A/B. It may be called a per-layer/per-channel precision
workaround. It may not be called an exact spatial-skip solution: upstream
perturbations still enter global pooling and the broadcast gate. The queued
optimal spatial reformulation is a separately certified exact global-summary
closure, not another annulus-only first cut.

## Triality

- **DSL:** a launch-inert policy binds exact n600 paths, split, ladder,
  checkpoints, process count, four timing arms, no-score flags, and plan-only
  mode. It emits only real probe arguments.
- **Equation:** register an inert canonical law for layer/channel waterfill,
  physical-byte accounting, finite-corpus certificate case split, and genuine
  width decomposition. Empirical status remains awaiting MAIN for the selected
  map/timing.
- **DAG:** standalone FEED records predecessor receipt, search stages,
  certificate gates, SE control, MAIN receipt, and pointer honesty.

## Acceptance tests

1. Complete layer/channel coverage and exact signed-int64 per-channel bounds.
2. Physically typed int8/int16/int32 buffers; metadata-only width fails.
3. Native versus forced-int32 NumPy outputs are bit-identical for identical
   logical codes and map.
4. A cached inverse permutation restores exact original output order without a
   per-channel Python/device-host scatter in the timed forward.
5. Cost accounting charges one shared activation buffer in this formulation,
   exposes the activation-buffer count, and reports conventional FLOP cut
   `0.0`; future multiple-activation-width extensions must charge duplicates.
6. Certificate keeps exhaustive, robust-interval, and exceptional subsets
   separate; zero margins cannot pass strict separation.
7. Affine primitive preserves shared-symbol cancellation and refuses missing
   remainder provenance.
8. Search is deterministic, checkpointable, inclusion-wise 1-opt on fixtures,
   freezes before validation, and never reselects from validation.
9. SE classifier finds 23 blocks/46 conv paths on the real frozen model,
   reproduces the exact Conv-MAC fraction, and records the real-frame shape
   trace plus explicit 0.15992356% whole-control operation convention.
10. Timing decomposition uses same-map `A32_M`/`AN_M`, never CPU-to-Metal alone.
11. Plan-only execution verifies all source hashes and writes an atomic durable
    receipt without Metal.
12. Host command is shell-clean, exact-n600, resumable, no-score, and points to
    the real Metal executor and a durable experiment directory; candidate bulk
    uses the SSD waterfall and certify-before-delete cleanup.
13. Typed DSL and equation tests are registration-inert and fail closed on
    weakened custody or authority.
14. Ruff, py_compile, focused pytest, `git diff --check`, three clean review
    passes, review-tracker evidence, serializer commit, and post-commit file
    verification all pass.

## Deliverable truth table

The final report must rank per-layer, per-channel, and SE-full-fp formulations
against: exhaustive n600 identity; strict interval coverage; physical storage
reduction; conventional FLOP cut; modeled operand traffic cut; measured width
latency; and the exact MAIN command. Every absent measurement is `OWED_MAIN`,
not guessed. The single highest-EV measurement is same-map forced-int32 Metal
versus physically native per-layer/per-channel Metal after the frozen n600 map
passes identity.

## Post-build binding amendment — 2026-07-14T13:22:43Z

Promotion now additionally requires live v9·CGauge wiring through the sole
canonical DSL/LawRef/consumer/receipt provenance path. The additive external
probe policy is not itself that live edge. The assigned
`provenance_canonicalize_fix_all_fakes` owner must wire the admitted
`margin_adaptive_perlayer_metal_n600.v1` receipt into the asynchronous/local
SegNet verdict-forward consumer and return compiled v9 argv plus a consumer
receipt. This lane must not edit that owner's witness-autoconfig/config/
bijection surfaces. Until then, the build is
`BUILT_AND_REVIEWED_BUT_NOT_V9_LIVE_WIRED`, not promoted apparatus.
