# Codex adversarial findings — margin-adaptive mixed precision

**UTC:** 2026-07-14T11:22:42Z  
**Lane:** `margin_adaptive_mixed_precision`  
**Status:** `BUILD_AND_LOCAL_PREFLIGHT_GREEN; MAIN_METAL_RECEIPT_OWED`  
**Authority:** `[code/source/receipt re-derivation; no Metal execution]`  
**Flags:** `research_only=true` · `score_claim=false` · `pointer_moved=false`

## Pointer and verdict scope

Pointer delta is zero. The submittable `0.19108282419209976 [contest-CPU]` and the unsubmitted
`0.1880443979880752 [contest-CPU]` defensive bank are unchanged. This landing is throughput
apparatus, not a byte-closed score row.

All negative language is scoped no broader than the supplied n600 instance and executable profile
formulation. No result here transfers from local M5-Max Metal to contest CPU/CUDA.

## Top discovery

The actionable allocation is **per-layer exact-int64 profile selection**, not literal annulus-only
execution. SegNet's 23 global squeeze/excite reductions and measured skip-inclusive halo685 make an
exact spatial annulus kernel full-frame. The per-pixel margin waterfill is still useful: it is the
finite-ladder information-theoretic lower bound, shows which margin bands consume bits, and identifies
whether finer execution granularity is worth building. It is not presently a native speed claim.

## Findings and guards

### F1 — The coarse-QDQ NO-GO was not rebuilt

The new converter composes the frozen-weight-L1 exact-int64 CPU/Metal suite. Every Conv2d gets an
explicit layer bit width, per-output-channel frozen weight scales, a dynamic per-layer activation
scale, the narrowest exact signed `int8`/`int16`/`int32` operand storage bucket, an exact signed-int64
multiply-accumulate, one fp32 scale/bias finalization, and a static no-overflow proof. Lower caps
coarsen only the integer representation; they cannot be admitted merely because they execute.

**Precision of language:** exact-int64 means the accumulation over integer codes is associative and
reorder-invariant. Quantization can still change logits. The all-pixel margin/tie certificate is what
licenses decision equality. Calling the lower-bit logits numerically lossless would be false.

### F2 — Minimum bits alone cannot select a profile

The design selector initially minimized average bits among empirically zero-flip profiles. Round-1
review tightened it twice: a selectable profile must also cover every design pixel by the strict
classwise interval test or the pre-frozen ordered class-pair tie rule, and the P0 ranking is measured
Metal seconds/pair first with logical bits only as tie-break. The distinct minimum-bit profile remains
reported. Only the timing-selected treatment is frozen and presented to pairs 264..599; the diagnostic
full-corpus minimum cannot reselect it. Profile execution order rotates by pair to avoid a fixed
thermal/cache position bias.

### F3 — Strict intervals and the zero-margin tie remain separate

The source corpus contains a reference zero-margin boundary, so `L_top1 > max(U_rival)` must refuse
that pixel. The already-measured predecessor froze `(4,0)->0` at gap `<=2^-19` from design pairs
0..263 and validated it without reselection on 264..599. The new admission requires exhaustive
equality and coverage by either the strict interval or that exact frozen rule. Ordinary empirical
equality is retained as a separate diagnostic and cannot masquerade as an interval proof.

### F4 — The waterfill optimum is exact only over the supplied ladder

`solve_finite_profile_waterfill` selects the lowest MAC-weighted average-bit certifying profile at
each pixel and is pointwise optimal over the measured finite set. It is not a continuous KKT optimum,
an unseen-input affine-arithmetic proof, or evidence that a per-pixel kernel exists. The receipt labels
all three boundaries explicitly.

### F5 — Wall clock, not bit count, is the P0 admission metric

The MAIN probe recomputes the one-thread CPU-Torch SegNet reference and synchronizes every Metal
candidate. It reports both MAC-weighted logical precision and physical storage width, measured n600
SegNet seconds, seconds saved, and speedup. A margin-adaptive admission additionally requires physical
width below the int32 baseline; if only the custom-Metal placement is faster, the separate verdict is
`EXACT_INT_METAL_CANDIDATE_ONLY_NO_MARGIN_ADAPTIVE_PHYSICAL_WIDTH_REDUCTION`. It separately reports
the combined n600 projection using the **DERIVED**, not measured, 372.6-second n96 extrapolation and
Pose share 0.226. Admission requires `speedup > 1`; a lower-bit but slower kernel is a NO-GO instance.

### F6 — Cross-process identity is load-bearing

The selected profile is rerun in ten fresh processes. A candidate requires one identical argmax
corpus digest across all ten, matching the parent run. Exact int64 accumulation is necessary but not
sufficient because scaling, fp32 finalization, tie handling, runtime/compiler behavior, and the full
graph still need target-device evidence.

### F7 — Custody and resumability fail closed

The probe requires exact pair indices 0..599, the real `gt_n600` cache, the measured uniform-QDQ
NO-GO, the dynamic calibration receipt, and the frozen class-pair exact-int predecessor. It hashes
the probe, allocator, cache, weights, and all three receipts into a resume fingerprint. Every pair,
the completed search, and each process trial is atomically checkpointed and preserved.

### F8 — External router replay is corroboration, not local evidence

NVIDIA NeMo's [labs-molt](https://github.com/NVIDIA-NeMo/labs-molt) describes fp32 router arithmetic
to avoid drift and Router Replay of the discrete expert selection. That is a useful analogy for
keeping the argmax decision exact while coarsening bulk arithmetic. It is an LLM/MoE/CUDA system,
not SegNet/MLX code or evidence for this candidate; no implementation was lifted.

### F9 — Physical storage has a real phase boundary — DERIVED from frozen model bytes

A blind structural re-derivation loaded the actual frozen SegNet and checked all 125 Conv2d layers
at every registered cap. Caps 8 and 10..16 use exact native int8 and int16 operand buffers,
respectively; cap 18 and every higher profile use int32 operands while retaining int64 accumulation.
All profiles passed the static signed-int64 bound. The maximum proved bound rises from `16,621,379`
at cap8 to `1,106,507,774,456` at cap16, `17,704,922,438,699` at cap18, and
`9,035,402,569,620,285,889` at caps30/31.

This makes the physical-width gate non-cosmetic: a design-selected cap18+ profile may still admit the
custom-Metal exact-integer placement, but cannot be called a margin-adaptive physical-width win. The
host receipt has a separate verdict for precisely that case.

## Built surface

- allocator/certificate/Metal adapter:
  `src/tac/local_acceleration/margin_adaptive_mixed_precision.py`
- decisive resumable probe: `tools/probe_margin_adaptive_mixed_precision_n600.py`
- exact host command: `tools/run_margin_adaptive_mixed_precision_n600_host.command`
- typed policy: `src/tac/witness_dsl/margin_adaptive_mixed_precision_20260714.py`
- canonical law: `src/tac/canonical_equations/margin_adaptive_integer_waterfill_20260714.py`
- standalone DAG feed: `.omx/research/margin_adaptive_mixed_precision_DAG_FEED_20260714.md`

## Verification disposition

- Focused allocator/probe/DSL/equation plus inherited fixed-point-kernel suite: `49 passed`.
- Fix-ALL review seal: three clean passes (adversarial selector/physical-width review, blind actual
  125-Conv structural derivation, authority/DSL/host integration).
- Ruff, `py_compile`, host-command shell syntax, and `git diff --check`: clean.
- Worker Metal execution: intentionally not run; environment has no authority and mission requires
  HAND MAIN.
- Empirical n600 selected bits, exact fraction, certificate fraction, digest, and latency:
  `OWED_MAIN_M5_MAX`, not guessed.

## STORES CONSULTED

- `reports/latest.md`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- `.omx/state/master_gradient_anchors.jsonl`
- `.omx/state/modal_call_id_ledger.jsonl`
- `.omx/state/cost_band_posterior.jsonl`
- `.omx/state/continual_learning_posterior.jsonl`
- `.omx/state/probe_outcomes.jsonl`
- uniform fixed-scale n600 NO-GO receipt
- dynamic-scale n600 calibration receipt
- frozen weight-L1 class-pair tie-snap n600 receipt
- latest Codex findings/session summary, latest T3 council/design memo, v7.5/v8 specs
- live per-arm and fleet inboxes through 2026-07-14T11:20:13Z
