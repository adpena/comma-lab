# Codex Findings: Embedding LR Policy Transfer Signal

UTC: 2026-05-23T01:04:54Z

## Evidence Axis

This memo records online research and planning-only optimizer-sweep wiring. It
is not score evidence, not promotion evidence, and not an exact-eval dispatch
trigger.

## Source Signal

Maxime Labonne's 2026-05-21 post:

`https://x.com/maximelabonne/status/2057602654151364899`

Fetchable mirror used for metadata and exact post text:

`https://api.fxtwitter.com/maximelabonne/status/2057602654151364899`

The post points to the paper:

`https://arxiv.org/abs/2605.21486`

Claim summarized for Pact: for AdamW-trained GPT-style models, much of muP's
observed hyperparameter-transfer benefit may be explained by the embedding layer
learning rate staying large instead of being scaled down by width. The paper
reports that standard parameterization with a width-corrected embedding LR can
match muP transfer behavior, while downscaling the embedding LR in muP damages
training. The direct contest relevance is not score authority; it is a parameter
group LR policy hypothesis for representation-training substrates.

## Pact Interpretation

The useful abstraction is not "use muP everywhere." It is a typed
parameter-group LR policy:

- embedding-like parameters: AdamW, Theta(1)-style LR, no inverse-width
  downscale by default;
- hidden matrix parameters: Muon or the selected hidden-layer optimizer;
- heads, scalars, norms, and fragile boundary params: AdamW unless a substrate
  gives a stronger reason;
- policy must declare width basis and param-group fingerprint before any MLX or
  PyTorch telemetry can be used for spend triage.

Pact "embedding-like" includes vocabulary embeddings only when present, but also
latent grids, codebooks, frame/pair/position embeddings, learned coordinate
tables, and first-layer/stem parameters in NeRV/HNeRV/INR-style substrates.

## Landed Wiring

Commit `3320bbf9d` adds a typed planning-only policy to
`src/tac/optimization/optimizer_scheduler_registry.py`:

- `PARAMETER_GROUP_LR_POLICY_SCHEMA`
- `DEFAULT_PARAMETER_GROUP_LR_POLICY`
- `EMBEDDING_THETA1_PARAMETER_GROUP_LR_POLICY`
- descriptor-level `parameter_group_lr_policy`
- planner candidate fields `parameter_group_lr_policy` and
  `parameter_group_lr_policy_id`

The Muon+AdamW representation recipe now declares
`embedding_theta1_hidden_muon_adamw`. The learned sweep plan surface now exposes
this policy through `optimizer_scheduler_candidates`, still with
`score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`,
and `ready_for_exact_eval_dispatch=false`.

## Remaining Integration

Next engineering steps:

- Add a param-group fingerprint helper that classifies named parameters for
  HNeRV, PR95 variants, PR101 clone, block/ff NeRV, VQ, and self-compression
  substrates.
- Wire `embedding_lr_scaling_policy`, `width_basis`,
  `embedding_param_patterns`, `param_group_fingerprint`, `state_bytes`, and
  `seconds_per_epoch` through the optimizer-training signal bridge,
  PR95 local-training integration, representation-training probe integration,
  MLX dynamic learned sweep, and candidate queue.
- Add same-seed local smoke variants: baseline, embedding-Theta1, inverse-width
  embedding ablation, high-to-low switch, and low-to-high switch.
- Score the archive-aware objective, not just loss: distortion proxy, encoded
  state bytes, archive/export readiness, exact-eval blockers, and runtime cost.
- Require PyTorch/MLX param-group fingerprint parity before MLX rows can guide
  exact-eval spend triage.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/optimizer_scheduler_registry.py src/tac/tests/test_optimizer_scheduler_registry.py src/tac/optimization/mlx_dynamic_learned_sweep.py src/tac/tests/test_mlx_dynamic_learned_sweep.py src/tac/optimization/__init__.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_optimizer_scheduler_registry.py src/tac/tests/test_mlx_dynamic_learned_sweep.py src/tac/tests/test_proxy_candidate_contract.py src/tac/tests/test_optimizer_guided_candidate_generation.py`
- `git diff --check`

## Verdict

Adopt as a bounded optimizer/scheduler policy axis for local/proxy sweeps. Do
not treat it as contest evidence until a byte-closed candidate passes the normal
local custody gates and exact auth eval.
