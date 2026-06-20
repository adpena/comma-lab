# Weight-entropy penalty — the Ballé end-to-end rate-distortion lever, wired into the PR95 torch-vehicle

**UTC:** 20260620T202332Z
**Authority:** `[contest-CPU advisory]` NON-PROMOTABLE. `$0`. NO paid dispatch. NO score claim.
**Pointer:** UNMOVED (this is an enabler landing — a training lever, not an exact row).
**Lane intent:** integrate the one un-integrated VCM / task-aware-compression lever — entropy-penalized
TRAINING (the Ballé rate-term-in-loss) — into `src/tac/torch_vehicle/driver.py`. We had the code
(`tac.entropy_bottleneck.EntropyBottleneck`) but it was NOT in the training loop.

## The math (why this is the real rate lever, not a fake)

Contest rate term `= 25·archive_bytes/37_545_489`. The decoder blob dominates; its byte count is
`archive_bytes(decoder) ≈ Σ_tensor H(quantized weight symbols)·n_elem/8`, where the symbols are EXACTLY
what the vendored codec codes: per Conv2d/Linear `weight`, `scale = max(|w|)/127`,
`symbol = round(w/scale) ∈ {-127..127}` (verified against
`…/hnerv_muon/src/codec.py::quantize_state_dict`). The post-hoc coder (PR112 constriction) is already at
the lossless floor (~7.999 bits/byte), so **a better coder is dead**. The ONLY way to lower the rate is to
lower `H` itself — which is set by TRAINING. The Ballé 2018 factorized-prior term `λ·E[−log2 p(w)]` under a
LEARNED prior pulls the weight-symbol distribution toward low entropy → lower `H` → lower deployed byte
floor. PR95 has only a weak MEMORYLESS shadow of this (`cat_entropy_v2` / `cat_lambda`); this is the proper
learned per-output-channel entropy model.

## What landed

1. **Module** `src/tac/torch_vehicle/weight_entropy_penalty.py`:
   - `WeightEntropyPenalty(nn.Module)` — one per-output-channel `EntropyBottleneck` per coded Conv2d/Linear
     `weight` tensor (channel dim = `weight.shape[0]`). `rate_bits(decoder)` runs each tensor's bottleneck on
     the **codec-grid representation** `w/scale` (scale `.detach()`-ed: only the symbol DISTRIBUTION is
     penalized, not the per-tensor magnitude) and returns `(total_bits, rate_term)` where `rate_term =
     total_bits/8/37_545_489·25` (contest rate scale). Training adds `U(-0.5,0.5)` (Ballé STE); eval rounds.
     The prior params (loc/raw_scale/raw_shape) are learnable.
   - `measure_decoder_weight_symbol_entropy(decoder)` — the REAL (hard codec-quantized, exact histogram)
     mean symbol entropy in bits/weight — the NO-FAKE metric the lever must LOWER (NOT the surrogate).
2. **Driver wiring** (`driver.py`): cfg fields `weight_entropy_penalty_lambda` (DEFAULT 0.0 = OFF /
   byte-identical), `weight_entropy_penalty_stage_min` (C1a-style late-stage gate), `_init_scale`.
   - `_build_stage_runtime`: when λ>0, build the penalty ONCE (persisted across stages; learned prior
     carries), add its params to a DEDICATED AdamW group at `spec.adamw_lr` (NOT in Muon partition / clip
     set — they are prior params, not decoder weights).
   - `_weight_regularizers`: append `λ·rate_term` (gated on λ>0 AND `_cur_stage_index >= stage_min`),
     alongside the C1a + Lever-1 terms — the same surface, same backward (split + fused paths).
   - `_final_decoder` exposed at each stage boundary for the post-run λ-on/off A/B (NOT a score surface).
3. **Launcher** (`experiments/launch_split_by_head_basin.py`): `--weight-entropy-penalty-lambda` (default
   0.0) + `--weight-entropy-penalty-stage-min` threaded into `TorchVehicleConfig`.

## Measured results (NO-FAKE)

- **λ=0 byte-identical**: two all-default runs produce bit-identical best score AND archive bytes; the
  penalty module is never built (`driver._weight_entropy_penalty is None`); `_weight_regularizers` returns
  the exact legacy `None`. The live MPS GREEN run (λ=0) is unaffected — its dir was never touched.
- **λ>0 lowers MEASURED codec entropy (headline)**: the driver-level A/B
  (`test_driver_lambda_positive_lowers_measured_entropy_vs_lambda_zero`, bit-shared init, λ=0 vs λ=50)
  ends with strictly LOWER measured hard-codec symbol entropy on the trained decoder. Pure-surrogate
  descent lowers it ~0.63 bits/weight in isolation. $0 CPU smoke (n8, 40ep) confirms the same in the full
  loop — see `## $0 smoke` below.
- **prior params in the optimizer + update**: verified the penalty params are a subset of the AdamW
  param-group set AND change across a step.
- **scale detached**: rescaling all weights by 3× (symbols invariant) does NOT change `total_bits`.
- 14 NO-FAKE tests pass (`src/tac/torch_vehicle/tests/test_weight_entropy_penalty.py`). Regression suites
  (`test_all_layer2_levers`, `test_export_and_faithful`, `test_driver_resume`) green.

## $0 smoke

Tiny synthetic CPU A/B (n8, 40ep, `base_ch20`), NOT MPS, separate temp dir; the live GREEN run was never
touched. Bit-shared init (same seed); the ONLY difference is λ:

```
[lam=0 ] best_score=81.74140  measured_entropy=7.4442 bits/wt
[lam=50] best_score=81.74166  measured_entropy=5.6767 bits/wt
ENTROPY DELTA (lam50 - lam0) = -1.7675 bits/wt   LOWERS_ENTROPY=True
```

λ=50 lowers the REAL codec weight-symbol entropy by **−1.77 bits/weight** (7.44 → 5.68) while the
synthetic task best_score is essentially unchanged (Δ ~3e-4 — the d_seg/d_pose proxies are NOT destroyed).
The lever measurably lowers `H` (the byte-floor driver) without harming the task signal.

## Regression status (post-commit, NO-FAKE honesty)

Full `test_all_layer2_levers` (96): **95 passed, 1 failed** in a 28-min run — the failure is
`test_r13_all5_descent_byteclose_parseback_on_real_scorer`, a REAL-scorer multi-step DESCENT assertion on
a tiny 8-pair/16-step slice. **This is a PRE-EXISTING FLAKE, NOT caused by this change** — proven three
ways: (1) my driver.py diff REMOVES ZERO lines (100% additive: new cfg fields + gated blocks + an inert
`_final_decoder` assignment), so the λ=0 path is bit-identical to pre-change; (2) the test uses the
weight-entropy lever at its DEFAULT (cfg λ=0.0; not even a StageSpec field), so my gated
(`lam_we > 0.0`) block is a strict no-op on its loss path — the penalty module is never built; (3) the test
PASSES in isolation (exit 0 on re-run), confirming flakiness. The test's own docstring acknowledges the
descent on this tiny slice is instrument-sensitive ("a 50× overshoot oscillates — instrument artifact, not
a lever defect"). The other 95 layer2 tests + `test_export_and_faithful` (7) + `test_driver_resume` (12) +
the new `test_weight_entropy_penalty` (14) + `test_from_scratch_launcher` (16) all pass.

## Honest verdict

The lever WORKS as designed: λ>0 measurably LOWERS the real codec weight-symbol entropy `H` (the quantity
that sets the deployed byte floor) — not just the surrogate. This is the orthogonal RATE axis that stacks
with the d_seg levers. It is `[contest-CPU advisory]` NON-PROMOTABLE: the ΔS claim (≈ −0.013 to −0.017
estimated from a 20–30% entropy cut) requires a byte-closed paired CPU/CUDA `upstream/evaluate.py` on a
real converged λ-on archive vs λ-off (the empirical bit-spend proof per Catalog #304). No score is
asserted from the flag alone.

**Follow-up (declared, not done):** the learned prior params are NOT yet persisted in the checkpoint, so a
RESUME of a λ>0 run rebuilds a fresh prior (loses the adapted prior; the decoder/EMA/optimizer resume
normally). The default λ=0 path is fully byte-identical incl. resume (penalty never built). Persisting the
prior in `_capture_state`/`_restore_into` is the resume-correctness follow-up for a long λ>0 campaign.

## 6-hook unified-Lagrangian wire-in (Catalog #125)

1. **Sensitivity-map** — N/A directly; the penalty IS a per-output-channel rate-sensitivity model on the
   decoder weights (a learned prior), complementary to the Lever-4 `||∂S/∂w||` EMA (sensitivity is on the
   distortion axis; this is on the rate axis).
2. **Pareto constraint** — ACTIVE in spirit: `rate_term` is on the contest rate scale, so λ trades rate vs
   d_seg/d_pose on the same axis the score's rate term lives — a tunable point on the R/D Pareto frontier.
3. **Bit-allocator hook** — ACTIVE (this IS a learned per-channel rate model; lowering `H` lowers the
   per-tensor byte budget the codec realizes).
4. **Cathedral autopilot dispatch** — N/A (training-time lever; no archive-deployable artifact from the
   flag alone; the campaign that USES it dispatches via the existing launcher).
5. **Continual-learning posterior** — N/A until a byte-closed exact row lands (then the λ-on/off
   archive_bytes-at-equal-distortion anchor seeds the rate posterior).
6. **Probe-disambiguator** — N/A (single defensible interpretation: lower symbol entropy → fewer bytes;
   the NO-FAKE test IS the disambiguation — measured codec entropy, not surrogate).

`council_predicted_mission_contribution: frontier_breaking_enabler` (an orthogonal rate axis that stacks
with the d_seg levers toward sub-0.15; the exact row is the follow-on campaign).
