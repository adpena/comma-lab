# SAFE-COMPILE v2 FINISH — auto-discovery + hot-loop wiring + per-chip trust (#252) — 2026-07-08  [no-triality]

Operator 2026-07-08: *"Finish engineering our own mx.compile."* v1 (commit 2950e6133) landed the
partitioner-by-allowlist + 3-certificate harness + manifest + fail-closed activation, hot-loop NOT
wired, auto-partitioner scoped to v2. This finishes the four missing pieces.

**MEANS, not ends.** Nothing here moves the pointer (0.19110). A certified region is bit-identical to
the uncompiled path — SPEED only ([macOS-MLX research-signal]; a bit-identity certificate is an
on-device FACT, NEVER a score). Only a byte-closed n600 `upstream/evaluate.py` row moves the pointer.
Default `--safe-compile-regions none` => empty set => every call site is a pass-through => BYTE-IDENTICAL
to the pre-#252 path (the live run-1 is unaffected — verified max|Δ| == 0 on the real levelset model).

## STORES CONSULTED
`.omx/research/mlx_safe_compile_20260708.md` (v1 memo — the region table + v2 scope note) ·
`src/tac/mlx_safe_compile.py` v1 (partitioner/harness/manifest/activation) ·
`src/tac/tests/test_mlx_safe_compile.py` v1 (22 tests) ·
`experiments/train_levelset_witness_realized_through_R_mlx.py` (the LIVE model = `build_levelset_rgb_witness`,
`_act` @1129 — NOT the base `train_witness` model; wired both) · `tools/launch_witness_run.py`
(admission chain b1/b1-sys — added b2) · `.omx/state/deferral_ledger.md` D17 (evidence gate) ·
`tools/mlx_gpu_determinism_probe.py` #348 (cross-process instrument) · live run-1
`levelset_n600_crucible_v6_run1_20260708T095730Z` (GPU occupied => governor: no GPU bench).

## PIECE 1 — AUTO-DISCOVERY (the v2 partitioner)
Replaced hand-nomination with a systematic sweep. Two complementary op-kind extractors whose UNION feeds
`classify_op_kinds`: **(i) AST scan** (`ast_op_kinds`) — catches operator-overload ops a runtime trace
CANNOT see (`@`→matmul, `*`→mul, `+`→add, `nn.Linear(...)`→matmul) — the safety net against a hidden
matmul misclassifying a region SAFE; **(ii) trace shim** (`trace_named_ops`, `_OpTraceShim`) — monkeypatches
the mx namespace to CONFIRM which named `mx.<op>` calls actually execute at real-shaped inputs (device=cpu
=> zero GPU contention). Added `STRUCTURAL_SAFE_OPS` (concatenate/reshape/broadcast/... = layout, no
arithmetic reorder => SAFE); `take`/gather DELIBERATELY EXCLUDED (its VJP is the #348 dup-index scatter hazard).

### CANDIDATE TABLE (10 registered witness hot-path regions; AST+CPU-trace sweep)
| region | union op-kinds | verdict | fma? | wired? |
|---|---|---|---|---|
| hosc_activation `tanh(β·sin(ω·u))` | tanh,sin,mul | SAFE_ELEMENTWISE | no | **WIRED** |
| siren_activation `sin(ω·u)` | sin,mul | SAFE | no | v3 (activation family) |
| wire_activation `sin·exp(-½(su)²)` | sin,exp,square,mul | SAFE | no | v3 |
| finer_activation `sin(ω(|u|+1)u)` | sin,abs,add,mul | SAFE | yes | v3 |
| film_affine_tail `x·s+b` | mul,add | SAFE | yes | MIXED→v3 (Linear rides outside) |
| film_modulate `h·s+b` | mul,add | SAFE | yes | MIXED→v3 |
| compose_rgb_tail `sigmoid(z)·255` | sigmoid,mul | SAFE | no | MIXED→v3 (self.out rides outside) |
| sigmoid_scale `sigmoid(b+t)·255` | sigmoid,add,mul | SAFE | yes | MIXED→v3 |
| achromatic_luma (BT.601 replicate) | mul,add,concatenate | SAFE | yes | control-arm |
| ce_reduction (logsumexp+sum+mean) | logsumexp,sum,mean,mul,sub | UNSAFE_REDUCTION | — | fixed-point route |

MIXED functions (`call_batch`'s film/compose tails) are reported as SAFE sub-chains but NOT split in v2
(their elementwise tail rides a Linear matmul — a v3 sub-chain split), per the operator's MIXED guidance.

## PIECE 2 — HOT-LOOP WIRING (flag-flip, wiring COMPLETE)
`install_safe_compiled_regions(model, enabled_regions, manifest)` sets `model._compiled_act =
mx.compile(pure_hosc)` ONLY when the CERTIFIED+fresh `hosc_activation` region is enabled. The levelset
model's `_act` (@1129, the LIVE hot loop) + the base `train_witness` model's `_act` both consult
`_compiled_act`; None (default) => plain path => BYTE-IDENTICAL. beta/omega pass as float32 array scalars
so the per-epoch beta anneal is a TRACED input (never a baked constant / per-epoch recompile). Call site:
the levelset trainer's existing safe-compile block installs onto `model` right after `resolve_enabled_regions`.
**Verified on the REAL levelset model (CPU): default-OFF plain vs compiled-array-scalar path max|Δ| == 0.**
Activation is now a pure flag-flip (`--safe-compile-regions hosc_activation`), evidence-gated for training
(D17 v7.1 arm) — wiring is complete, not a future build.

## PIECE 3 — WHOLE-STEP DELTA (governor: DEFERRED, honestly)
Live run-1 occupies the GPU (safe_run, 90 GB cap). A GPU whole-step bench beside it would BOTH corrupt
the measurement AND perturb run-1 → the governor-respect path = **defer the full-shape B=8 GPU whole-step
to the run-1 stop checklist** (queued: D17). Reduced-footprint CPU micro-bench (LABELED — NOT the GPU
number; tiny-shape CPU compile overhead dominates, so speedups are noisy 0.4–1.4×, unrepresentative):
per-region CPU `certify_region` timings recorded in `.omx/state/mlx_safe_compile_discovery_cpu_v2.json`.
**No projection is reported as a result.** The finished GPU whole-step number is a run-1-stop deliverable.

## PIECE 4 — PER-CHIP TRUST CLOSURE
- `host_fingerprint()` = `{chip, macos_build, mlx_version}` (this host: Apple M5 Max / 25E246 / 0.31.2),
  stamped into the manifest (schema bumped `v1`→`v2`). `manifest_fingerprint_ok` + `fingerprint_matches`:
  a NON-EMPTY recorded field that mismatches the host => STALE => `resolve_enabled_regions` returns empty
  (fail-closed, recertify). A legacy (fingerprint-absent) manifest is allowed by construction (back-compat).
- **Device match**: fp-contraction is device-specific (MEASURED below) — `resolve_enabled_regions(run_device=…)`
  refuses a CPU-measured cert on a GPU run and vice versa. Trainer passes `args.mlx_device`.
- **Launcher admission (b2, patch-file)**: when the launch.sh arms `--safe-compile-regions != none`,
  REFUSE (rc=4) an absent/stale-fingerprint manifest before spawn (advisory in --dry-run).
- **Failed-cert → Metal-kernel-candidate pipeline (D16 feed)**: `emit_kernel_candidates(manifest)` turns
  every FAILED bit-equality row into a ranked kernel candidate (fp_contraction ranked first by speedup
  potential). Emitted into `manifest.as_dict()["kernel_candidates"]`. LIVE, not dormant — see the finding.

## PIECE 5 — REMAINING REDUCTIONS (`STEP_REDUCTION_SITES`)
Enumerated the step's reduction sites: `ce_loss` (logsumexp+sum+mean), `eikonal_loss` (mean+square),
`length_loss` (sum+mean), `R_operator_reductions` (interpolation+mean). **All routed NATIVE** — the #348
cross-process probe proved native `mx.sum`/`mean`/`logsumexp` ALREADY cross-process bit-identical on MLX,
so a fixed-order rewrite is a FALLBACK, not needed. `certify_fixed_point_reduction('ce_reduction')` is the
armed fallback if a future host's determinism cert fails a native reduction. R roundtrip routes through the
#348 `--fused-r-kernel` (fixed-order VJP), NOT an mx.compile target (v7 audit lever #3 EXCLUDED it).
**Left unrouted: all native reductions — reason: #348 already certifies them bit-identical.**

## THE v2 FINDING (a real bug + a real measurement)
1. **v1 harness DEVICE BUG (fixed):** `_bit_equality_and_timing` measured bit-equality on the process
   DEFAULT device regardless of `device=` (which only reached the cross-process children) — so a
   `certify_region('cpu')` could silently measure GPU bit-equality. Fixed: `certify_region` now pins the
   in-process measurement to the requested device.
2. **hosc fp-contracts on CPU (MEASURED):** with the bugfix, CPU certification shows 8/9 SAFE elementwise
   regions bit-equal (max|Δ|=0) but **`hosc_activation` FAILS on CPU (max|Δ|=5.96e-8, 1 ULP)** — the exact
   activation run-1 uses. It auto-emits a rank-0 `fp_contraction` kernel candidate (the D16 pipeline working
   end-to-end on a genuine failure). On GPU, v1 certified hosc bit-equal at n=32 — but CPU's clean failure
   means the **empirical certificate is coverage-limited (a finite input sample, not a proof)**; GPU hosc
   warrants BROADER/adversarial coverage before the score-bearing run trusts it. This vindicates the v7
   audit's original MEASURED-EXCLUSION of blanket mx.compile from the R operator, and is exactly why the
   manifest is per-chip + per-device and activation stays evidence-gated (D17).

## v3 RESIDUALS (honest)
- GPU cross-process re-certification WITH fingerprint into the CANONICAL manifest (governor-deferred to
  run-1 stop); the v1 canonical manifest is device=gpu, fingerprint-absent (legacy-unscoped, back-compat).
- Larger / adversarial input coverage for the bit-equality certificate (hosc's CPU failure shows n=32 can
  pass while unsampled inputs diverge) — a coverage/robustness upgrade, not a correctness gap for default-OFF.
- MIXED sub-chain split (film/compose tails off their Linear) → wire film_affine_tail / compose_rgb_tail.
- The full-shape GPU whole-step B=8 bench (the finished speedup number).

## WHAT LANDED
- `src/tac/mlx_safe_compile.py` v2: STRUCTURAL_SAFE_OPS · ast_op_kinds · _OpTraceShim/trace_named_ops ·
  CandidateRow/discover_candidate/discover_candidates · host_fingerprint/fingerprint_matches/
  manifest_fingerprint_ok · CertificationManifest.fingerprint + failed_ids + kernel_candidates ·
  resolve_enabled_regions(enforce_fingerprint, host, run_device) · emit_kernel_candidates ·
  install_safe_compiled_regions/_pure_hosc/WIRED_MODEL_REGIONS · STEP_REDUCTION_SITES · `--discover` CLI ·
  device-bug FIX in certify_region.
- `src/tac/tests/test_mlx_safe_compile.py`: 22→45 tests (auto-discovery classification · fingerprint
  refuse/recertify · device-mismatch refuse · kernel-candidate ranking · install/flag-flip byte-identity).
- `experiments/train_{levelset,witness}_realized_through_R_mlx.py`: model `_compiled_act` slot + `_act`
  compiled-path branch (default-OFF byte-identical) + levelset install call (patch-file, my hunks only —
  a sibling mod-dim-dynamics landing shares the file).
- `tools/launch_witness_run.py`: admission step b2 (safe-compile manifest freshness, patch-file).
- `.omx/state/mlx_safe_compile_discovery_cpu_v2.json` (gitignored): the CPU discovery/candidate artifact.
- `.omx/state/deferral_ledger.md` D17 (gitignored): status → v2-WIRED.

Code changes carry their triality treatment (DSL lever `SafeCompileRegions` exists from v1; no new
equation — this is compute-facet apparatus, not a measured law). This memo = [no-triality].
