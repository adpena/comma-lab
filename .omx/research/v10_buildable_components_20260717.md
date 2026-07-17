# v10 buildable components — RESULTS-INDEPENDENT builds (#520 P3 · P4 coder · P1 default) — 2026-07-17

Task #520 family · branch `claude/p0_v10_buildable_components_20260717` (from main) · joins the
boundary-merge queue AFTER `SPEC_v10` (`claude/p0_521_spec_v10_capstone_20260717`).
**Axis:** `[macOS-CPU advisory]` / MEANS — **pointer 0.19108 UNMOVED**; these are the two
RESULTS-INDEPENDENT unbuilt v10 components (+ the P1 structured-init default), built so the
post-v9c2 critical path shrinks to measurement-gated decisions only. NO score claim.

Subordination: NO-FAKE > THE GOAL (sub-0.15) > SPEC_v10 > this landing. Every number below is
MEASURED (donor byte-close), DERIVED (from the exact operator), or labeled.

## Per-component status

| # | component | status | evidence |
|---|-----------|--------|----------|
| C1 | range(A) render-target projection layer (`src/tac/boundary_math/range_a_projection.py`) | **DONE** (primitive + hook + DSL lever); render-loop consumption **OWED at v10 fold** (build-gated) | projector self-test residual max\|A(X−PX)\| = **1.65e-15** (reproduces #519 exactly); 106/140 exact blind rows/cols; 12 tests pass |
| C2 | content-priced coder (`src/tac/codec/content_priced_coder.py`) | **DONE** — real `encode`/`decode` with EXACT round-trip; wins bytes on the donor | donor content-stream **−2,944 B (−3.58%)** vs `quantize_levelset_blob`; full decodable blob **−2,369 B** even WITH scales+manifest; round-trip exact; 15 tests pass |
| C3 | structured-init-as-default (`src/tac/witness_dsl/spec_v10_structured_init_defaults.py`) | **DONE** as spec-module code; fail-closed; fold into `spec_v10_capstone` is a documented one-liner (not applied on this branch → clean merge) | 5 defaults ALL birth-on; all 5 machinery modules import + self-detect present; hood-tex-seed absence → 1 fail-closed blocker; 7 tests pass |

## C1 — range(A) render-target projection (#520, SPEC_v10 P3)

**What it is.** The EXACT orthogonal projector `P_range(A)(X) = Qr (Qrᵀ X Qc) Qcᵀ` onto the frozen
scorers' σ-algebra, where `A` is the shared bilinear resize `874×1164 → 512×384`
(`align_corners=False, antialias=False`), extracted via the #391 `resize_matrix_1d` machinery (NOT
re-derived). `X − P(X) ∈ ker(A)` exactly, so `A(P X) = A(X)` and `P` drops the MEASURED ~52%
scorer-invisible render energy (#519).

Three forms delivered:
- **(a) numpy authority** `apply_projection` — fp64-internal (exact) → fp32-out; handles
  `(H,W)`, `(H,W,C)`, `(...,H,W[,C])`. Self-test residual **1.65e-15** = the validated #519 number.
- **(b) MLX twin** `apply_projection_mlx` — fp32, parity-tested vs numpy fp32 (rel 3.7e-3 — fp32
  einsum accumulation; the numpy fp64 path is the authority, MLX is the training-time twin).
- **(c) DSL lever** `curriculum_dsl.RangeAProjection(cadence)` — DEFAULT-OFF duty-to-measure;
  emits `--range-a-projection` + `--range-a-projection-cadence`; `lever_registry.completeness()`
  maps both (no new unmapped/stale).

**Composition-note law (docstring, binding).** `P_range(A)` and the Laguerre cell-generator
description are the SAME projection onto the scorer σ-algebra: ker(A) removes what the scorer
cannot RESOLVE; the generator description removes what the argmax cannot DISTINGUISH within an
atom (SPEC_v10 §3.0 / P3 amendment).

**Trainer wiring (guarded, default-OFF, fail-closed).** Two argparse flags + one lazy-import
arm-and-validate hook in `main()` after `parse_args`. **Default False ⇒ the hook is skipped
entirely ⇒ byte-identical** (no import, no behavior). When armed, it reproduces the projector
self-test and REFUSES (raises) if the exact residual regresses, then records the cadence to run
provenance as `PREPARED_NOT_FIRED`.

**Honest boundary (why the render-loop consumption is OWED, not wired here).** Applied to the
SCORER INPUT the projection is scorer-invariant BY CONSTRUCTION (`A(PX)=A(X)`) — a null. Its
score-moving use is a TRAINING-RENDER restriction, which requires the render at CAMERA resolution
`874×1164` (the loss render is currently `render_h×render_w`, and `A` is only defined on the
camera grid). That render-at-camera-res path is a v10-fold decision (SPEC_v10 P3 "build owed"),
so the trainer hook ARMS + VALIDATES the primitive; the loss-loop consumption folds at v10 launch,
score EFFECT owed via a byte-closed n600 realized-through-R row. The bit-identity-critical #205
verdict path was deliberately NOT touched.

## C2 — content-priced coder (SPEC_v10 P4) — the Kolmogorov-violation fix

**What it is.** One `encode(checkpoint) -> blob` / `decode(blob) -> checkpoint` pair that prices
CONTENT not SHAPE, composing EXISTING primitives with EXACT round-trip verification built in:
- per-tensor symmetric int8 grid (`lever_b_levelset_generator._int8_symmetric`, REUSED — the
  canonical grid, byte-identical to the baseline ⇒ fidelity-matched comparison);
- **#461 cross-tensor structure** (`witness_crosstensor_codec`): exact post-Brotli axis-storage
  permutation + frame-separated mod-256 temporal delta on `code` (both round-trippable);
- adaptive lossless entropy backend: best-of {Brotli-q11, raw-LZMA(9|EXTREME), zlib-9} PER STREAM
  (Brotli-q11 ALWAYS a candidate ⇒ can never lose to the baseline);
- **#519 gauge/palette pre-canonicalization** — OPT-IN precision transform (default OFF; MEASURED
  byte-neutral in #519, so it is NOT the byte win — offered as a recorded, exactly round-tripping
  transform, never faked as bytes saved).

**Contract met (donor `mod32cap ep650 EMA BEST`, read-only, MEASURED):**

| quantity | baseline `quantize_levelset_blob` | content-priced | delta |
|---|---:|---:|---:|
| base int8+entropy | 61,842 | 61,413 (perm: `in_proj.weight`,`out_tex.weight`) | −429 |
| code int8+entropy | 20,355 | 17,840 (`frame_delta_mod256`) | **−2,515** |
| **content stream (accounting-matched)** | **82,197** | **79,253** | **−2,944 (−3.58%)** |
| full DECODABLE blob (incl. scales + 559 B manifest) | 82,197 (non-decodable measurement) | **79,828** | **−2,369** |
| round-trip | — | **EXACT (18 tensors)** | — |

The content-stream win is STRICTLY-BETTER-OR-EQUAL by construction (the permutation plan carries
the no-perm mask; the entropy best-of carries Brotli-q11) and MEASURED strictly better on the
donor. Notably the full decodable blob is SMALLER than the baseline byte-measurement even though
it honestly includes the per-tensor scales the baseline omits (the #461 code-delta win exceeds the
575 B scale/manifest overhead).

**#336 SKIPPED honestly (recorded).** Sensitivity bit-allocation needs MEASURED per-tensor
score-response RD rows a raw checkpoint does not carry, and downgrading below int8 is a FIDELITY
tradeoff outside the equal-fidelity byte contract. The coder fail-closes (raises the recorded
reason) if `bit_alloc` is passed and records the skip in `compare_bytes(...)["skipped"]`. Compose
it upstream (`apply_sensitivity_bitalloc_witness.py`) when measured RD rows exist; the coder then
prices whatever grid it emits.

**Applypass registration.** `witness_applypass_batch` already registers `bit_alloc_336` /
`low_rank_pose_140` / etc. as delegates; the content-priced coder is a rate lever whose ΔS row
rides the post-run batch via `compare_bytes` (a byte-close RATE measurement). Its registry entry
is a candidate for the next applypass-batch fold (delegate `content_priced_coder_p4` →
`compare_bytes`); NOT added on this branch to avoid touching the shared tool registry mid-flight —
recorded as the fold step. `# WIRE_IN_DEFERRED: applypass registry fold at boundary merge (rate
lever; compare_bytes is the measurement surface).`

## C3 — structured-init-as-default (SPEC_v10 P1)

`structured_init_defaults()` holds the 5 seeded statics (hood MyCar / sky Undrivable / lane
polynomial / per-dash anchors / hood-tex seed) as **birth defaults** (all `default_on=True`) — the
v10 compile-path default rather than opt-in flags. Each names its class-SELF-DETECTING machinery
entry (NEVER a hardcoded class index) + its provenance triple. `structured_init_status(root)`
probes (read-only, $0) machinery import + self-detect presence + required seed existence, emitting
typed fail-closed blockers exactly like the spec skeleton (the absent hood-tex seed → 1 blocker).
`structured_init_blockers(root)` is the one-call boundary fold surface:
`report.blockers.extend(structured_init_blockers(root))` in `spec_v10_status` — a single line,
`spec_v10_capstone` UNMODIFIED on this branch (clean merge; the cherry-picked skeleton is
byte-identical to its source).

## Residuals (honest)

- **C1** ker(A) has no exact weight-space projection (nonlinear render) — the projector is
  IMAGE-space; its training use is a render regularizer whose EFFECT is measurement-gated. MLX
  twin parity is fp32-level (0.37% rel), acceptable for a non-authority training twin.
- **C1** the trainer render-loop consumption is a v10-fold build (render-at-camera-res), not wired
  here; the hook is arm/validate only. This is the honest scope, not a shortcut.
- **C2** the byte win is on THIS donor's int8+brotli grammar (INSTANCE scope); the mechanism
  (permutation + code temporal-delta + entropy best-of) is general but the deltas are donor-
  specific. Gauge canonicalization is byte-neutral (#519), offered as a precision transform.
- **C3** the hood-tex seed is a real required counted artifact; its path is the calibration-family
  root (exact filename confirmed at harvest). Fold into `spec_v10_capstone` is owed at boundary
  merge (documented one-liner).

## Round-1 adversarial review (own attack — findings + fixes, landed)

1. **F1 (fake-consumption risk, C1):** an arm-hook that only validates could be read as claiming
   the render is projected. FIX: the hook prints `PREPARED_NOT_FIRED` + the memo/§ state plainly
   that render-loop consumption is a v10-fold build and that scorer-input projection is a null by
   construction. The real primitive (`maybe_project_render_target`) is separately tested.
2. **F2 (baseline-omits-scales, C2):** `quantize_levelset_blob` is a byte MEASUREMENT that omits
   per-tensor scales, so a naive "full blob vs baseline" would be apples-to-oranges. FIX:
   `compare_bytes` reports BOTH the accounting-matched content stream (the guaranteed ≤ claim) AND
   the full decodable blob (with scales), labeling the baseline "non-decodable measurement".
3. **F3 (gauge-as-byte-win overclaim, C2):** #519 measured gauge byte-neutral. FIX: gauge is
   default-OFF, opt-in, recorded, and the memo states the byte win comes from #461+entropy, not
   gauge.
4. **F4 (#336 fake composition):** claiming #336 is composed without measured RD rows would be a
   fake. FIX: #336 is SKIPPED with a recorded reason + fail-closed on `bit_alloc`.
5. **F5 (registry drift, C1):** a trainer flag without a DSL owner drifts (unmapped). FIX: the
   `RangeAProjection` factory in `curriculum_dsl` holds both flags; `completeness()` confirms
   mapped / no new stale.
6. **F6 (merge safety):** the cherry-picked `spec_v10_capstone` is UNMODIFIED (byte-identical both
   sides → clean 3-way); C3 lives in a NEW module (no add/add conflict); curriculum_dsl + trainer
   edits are localized additive blocks.

## Triality legs

- **DSL:** `curriculum_dsl.RangeAProjection` (new lever) + `spec_v10_structured_init_defaults`
  (new spec-module) + the two default-off trainer flags. `completeness()` clean.
- **equations:** consumed (no new law — these are BUILDS, not measurements):
  `null_subspace_rate_measure` (#519 gauge/blind, the projector residual + byte-neutrality),
  `segnet_head_rank4_linear_flipdist_v1`, `necessity_generator_seed_dseg_calibration_v1` (hood-tex
  seed). `# FORMALIZATION_PENDING: build-only-no-new-measured-finding (donor byte delta is
  INSTANCE-scope; register a canonical equation only if a second checkpoint reproduces the sign).`
- **DAG:** FEED block below (appended by the boundary integrator at merge — this worktree does not
  mutate the shared DAG file).

## DAG FEED block (ready to paste)

```
FEED-520-v10build (2026-07-17) — v10 RESULTS-INDEPENDENT components BUILT (task #520 family,
branch claude/p0_v10_buildable_components_20260717; joins boundary queue AFTER SPEC_v10).
C1 range(A) render-target projector src/tac/boundary_math/range_a_projection.py — EXACT P_range(A)
= Qr(QrᵀXQc)Qcᵀ via #391 resize kernels; self-test max|A(X-PX)|=1.65e-15 (reproduces #519);
numpy authority (fp64) + MLX twin (parity 0.37% rel) + DSL lever RangeAProjection (default-off,
lever_registry-mapped) + guarded default-OFF fail-closed trainer arm-hook (render-loop consumption
OWED at v10 fold: needs render-at-camera-res; scorer-input projection is a null by construction).
C2 content-priced coder src/tac/codec/content_priced_coder.py — real encode/decode EXACT round-trip
composing int8 grid + #461 cross-tensor (perm + code frame-delta) + entropy best-of {brotli,lzma,
zlib} + #519 gauge (opt-in, byte-neutral); donor mod32cap ep650: content-stream 82197->79253
(-2944, -3.58%), full decodable blob -2369 even WITH scales, round-trip exact; #336 SKIPPED
honestly (needs measured RD rows; fail-closed). C3 structured-init-as-default
src/tac/witness_dsl/spec_v10_structured_init_defaults.py — 5 seeded statics birth-ON, class-self-
detecting, fail-closed on absent hood-tex seed; folds into spec_v10_capstone via one line at merge.
36 tests pass; ruff-F clean. Pointer 0.19108 UNMOVED (MEANS/apparatus — builds, not a score row).
```

**Pointer 0.19108 UNMOVED — these are MEANS.** They exist to make the post-v9c2 v10 launch a
measurement-gated decision, not a build-gated one.
