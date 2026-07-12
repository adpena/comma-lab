# Frozen-SegNet P0: ELM/INR affine-head seed implementation spec

Date: 2026-07-12  
Lane: `lane_frozen_segnet_gradient_replacement_elm_headsolve_20260712`  
Authority: operator P0 request plus `fast_witness_training_oss_survey_20260712.md`  
Execution scope: local build and faithful-slice measurement only; no n600 launch and no edit to the live V9 arm.

## Objective

Build the closed-form ELM/INR output-head solve as a deterministic, NumPy-portable seed for the already-settled #341 terminal Gauss-Newton finisher. The external paper uses a random-feature hidden basis; this integration ports its freeze-`H` plus local least-squares-head plus partition-of-unity mechanism onto the existing directional-feature/level-set hidden state so the result is a seed for the receiver that actually ships. It is not a reproduction of the paper's random-Fourier architecture. The feature/trunk state is frozen. Only the affine SDF output head is solved in closed form; `out_tex`, `palette`, FiLM, pair codes, and trunk weights remain unchanged for the subsequent exact through-R #341 polish.

This landing is not a score claim. Exact authority remains `upstream/evaluate.py` on the final archive bytes. A diagnostic pair slice is explicitly non-authority and cannot promote the lane.

## Owned files

- `src/tac/boundary_math/elm_inr_head_solve.py`
- `tools/elm_inr_head_seed.py`
- `src/tac/tests/test_elm_inr_head_solve.py`
- `src/tac/witness_dsl/elm_head_seed_policy.py`
- `src/tac/tests/test_elm_head_seed_policy.py`
- `src/tac/canonical_equations/elm_inr_affine_head_seed_20260712.py`
- `src/tac/canonical_equations/tests/test_elm_inr_affine_head_seed_20260712.py`

Do not edit the live MLX trainer, curriculum DSL, CUDA port, micro-batch modules, V9 result directory, or `tools/quadratic_basin_finisher_probe.py`. The seed artifact must instead be directly consumable by the latter's existing `--params` argument.

## Mathematical contract

For frozen hidden features `h_i in R^D`, append the intercept to obtain `x_i=[h_i,1]`. For class target `y_i in R^K`, solve the weighted ridge system per subdomain `s`:

`A_s = sum_i w_si x_i x_i^T + ridge * diag(1,...,1,0)`

`B_s = sum_i w_si x_i y_i^T`

`beta_s = A_s^+ B_s`

where the bias is unregularized and `pinv` is a deterministic fail-safe for singular systems. Targets are finite centered log-probabilities from label-smoothed categorical labels, not exact argmax derivatives:

`y_ik = temperature * (log(q_ik) - mean_j log(q_ij))`.

Rectangular partition-of-unity weights are nonnegative tent functions on normalized coordinates and sum to one for every sample. The local prediction is `sum_s w_si x_i beta_s`. Because the present witness has one global affine `out_sdf`, the partitioned field must be projected back to a global affine head with a second streaming ridge solve. Report the projection RMSE and do not claim the local POU field survived the current decoder unless that residual is accepted by a future governed A/B.

With one subdomain, the solve is exactly the directly deployable global affine seed. Returned parameter orientation must match MLX/NumPy custody: `out_sdf.weight` shape `(K,D)` and `out_sdf.bias` shape `(K,)` for `phi = h @ weight.T + bias`.

## Hidden-feature custody

Expose a NumPy hidden-feature forward that mirrors `levelset_rgb_forward_numpy` through the last frozen hidden activation, including legacy-compatible optional `film_pl.*` and `concat_pl.*` branches. A contract test must show that:

`hidden @ out_sdf.weight.T + out_sdf.bias == levelset_rgb_forward_numpy(...).phi`

within fp32 tolerance on deterministic fixtures. This prevents a seed generated from a subtly different trunk.

## Streaming, resumability, and storage

The CLI operates in bounded stages and writes an atomic state artifact after every processed pair. State contains complete normal equations, pair cursor, selected pair list, config, source checkpoint SHA-256, and feature-state SHA-256. Resume must reject changed custody/config. It must never materialize all P=600 hidden pixels at once.

Stages:

1. `accumulate`: stream pair features and smoothed label targets into the per-subdomain normal equations.
2. `project`: solve local heads, re-stream the same pairs, and accumulate the global fold normal equations.
3. `finalize`: solve the global head, write a complete checkpoint atomically with all source keys preserved, plus a JSON receipt containing condition/rank diagnostics, fit RMSE, projection RMSE, bytes, hashes, pair scope, and non-authority label.

Full-P=600 is the canonical scope. A smaller pair limit is allowed only through the typed policy with an explicit `diagnostic_slice=true` receipt. Loose semantic CLI flags are forbidden: ridge, target, grid, custody, and pair scope compile from the value-provenanced policy. Source labels must be class-range validated before lossless compaction to the smallest admissible integer dtype. The output filename must encode the stage/scope and be accepted by:

`.venv/bin/python tools/quadratic_basin_finisher_probe.py solve --params <seed.npz> --tag <features> --mask head --k-pairs 600 ...`

No automatic #341 solve is launched by this tool. A diagnostic policy emits no #341 command at all; only a full-P=600 policy may emit a suggested command, and it must spell `--k-pairs 600` explicitly. Immediately before command emission, rehash the canonical feature-state path and compare its bytes to the declared feature-state SHA; tag/path equality alone is insufficient.

## Verification gates

1. Exact synthetic affine recovery with noise-free data.
2. POU nonnegativity and row-sum-one across boundaries.
3. Streaming accumulation equals one-shot normal equations.
4. Singular system is finite and deterministic.
5. Hidden-forward parity with the canonical NumPy witness.
6. Atomic checkpoint preserves all non-SDF arrays and metadata while changing only `out_sdf.weight/bias`; file and parent-directory fsync make replacement durable.
7. A real CLI interruption/resume sequence gives byte-identical final direct/POU checkpoints to an uninterrupted CLI sequence and validates done-state re-entry custody.
8. Faithful local slice on the real #341 checkpoint/features records fit/projection engineering metrics. Any bounded frozen-SegNet comparison must independently bind its exact commands, scorer/GT/video/source/checkpoint/runtime custody, timing boundary, and pair-only non-authority scope; no d_seg/score claim without that receipt.

## Structural-gradient follow-on boundary

The scorer-gradient replacement prototype is a separate callable contract, not a live-trainer edit in this landing because the trainer is owned by sister agents. It must define a costate-injection identity and periodic full-teacher verification gate. Promotion requires measured input-gradient cosine/norm agreement plus one-step improvement under the real teacher loss; logit agreement alone is insufficient because the historical learnable-student loop was inert.
