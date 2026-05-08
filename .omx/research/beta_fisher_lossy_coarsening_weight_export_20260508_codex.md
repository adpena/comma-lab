# Beta-Fisher x Lossy Coarsening Weight Export - 2026-05-08

Scope: CPU-only scaffold for the user-requested beta-Fisher x
lossy_coarsening / boundary-mass weighting / deterministic film-grain
re-injection lane. No GPU jobs were launched. No score claim is made.

## New surfaces

- `src/tac/optimization/beta_fisher_lossy_weights.py`
  - Converts `tac.sensitivity_map` beta-Fisher/channel artifacts into
    per-tensor scalar allocator weights.
  - Optional boundary mass raises distortion cost for tensors carrying
    boundary-sensitive signal.
  - Optional texture/film-grain capacity lowers distortion cost where a
    deterministic, archive-accounted residual design or high local symbol
    variance can hide coarsening error. This is an allocator prior only; it
    does not add free random noise at inflate time.
  - Emits weights in the existing allocator convention:
    `cost = bytes + lambda * weight[t] * rel_err[t]^2`.
- `tools/build_beta_fisher_lossy_coarsening_weights.py`
  - Loads a PR101-style state dict and sensitivity map.
  - Exports `tensor_weights.allocator_input.weights`.
  - Runs the existing `LagrangianPerTensorAllocator` with those weights over
    lossy-coarsening K curves.
  - Exports `weighted_k_allocations[].selected_Ks` for no-dead-K archive
    rebuilding.

## Evidence class

`[CPU-planning beta-Fisher lossy-coarsening weight export]`

This is a planning artifact only:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `score_affecting_payload_changed=false` for the weight artifact itself
- `downstream_selected_Ks_can_change_charged_bits=true` once a byte-closed
  no-dead-K archive builder consumes the selected vector

Diagnostic/stub sensitivity maps are rejected by default. The tool requires
`--allow-diagnostic-sensitivity` to use a stub/proxy map, and the manifest
records `diagnostic_or_stub_sensitivity_map_not_score_authority`.

## Exact next integration point

Patch `tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py` at
`ADMM_PATH_B_STEP6_KS` to accept an optional
`--score-weights-json/--selected-Ks-json` argument. The argument should read:

```text
weighted_k_allocations[].selected_Ks
```

from the manifest produced by:

```text
.venv/bin/python tools/build_beta_fisher_lossy_coarsening_weights.py \
  --state-dict experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt \
  --sensitivity-map <certified_or_diagnostic_sensitivity_map.pt> \
  --rms-targets 0.0386 \
  --output-json reports/raw/beta_fisher_lossy_coarsening_weights/<run>/manifest.json
```

Then rebuild the no-dead-K archive and run static preflight. Only after the
archive bytes, SHA-256, runtime closure, and dispatch claim are recorded should
an exact CUDA auth eval be considered.

## Why this is the highest-EV unblocked patch

The no-dead-K path already has a byte-closed CPU candidate and the allocator
already supports per-tensor weights. The missing unblocked piece was the
score-aware producer that turns beta-Fisher/component sensitivity, boundary
mass, and texture/film-grain capacity into the exact weight vector the existing
allocator can consume. This patch does that without touching active archive
builders or GPU dispatch surfaces.

## Composition review

- Additive with Jacobian-pullback allocation: beta-Fisher weights are a coarse
  sensitivity prior; Jacobian-pullback can replace or calibrate the sensitivity
  scalars later without changing the archive-builder interface.
- Additive with no-dead-K archive surgery: the exported `selected_Ks` are meant
  to feed the existing no-dead-K packet path, not become a separate sidecar.
- Potentially antagonistic with pure UNIWARD variance: if boundary/scorer
  sensitivity is high, texture capacity must not override it. Score-aware
  sensitivity has priority over capacity.
- Orthogonal to entropy/packer tuning: this lane changes the K vector and
  therefore decoded tensor values; low-level packing can still run after a
  byte-closed candidate exists.
- HStack/VStack status: currently a VStack planning stage
  `sensitivity -> K allocation -> no-dead-K pack`. It is not yet an HStack
  component because no separate payload stream is emitted.

Retirement is not justified by any CPU output here. The artifact can only be
accepted or rejected after a byte-closed archive consumes the selected K
vector, passes static preflight, and returns exact CUDA auth-eval evidence.
