# Kaggle PR101 bias-refine harvest and Vast blocker - 2026-05-11

## Scope

Continue score-lowering work without duplicating the active T1 Modal claim.
The active claim remains `t1_balle_128k_endtoend`; no duplicate T1 dispatch was
started.

## Vast PR106 latent sidecar

Attempted the highest-ranked non-duplicate score-lowering lane:

```bash
.venv/bin/python scripts/launch_lane_on_vastai.py phase1 \
  --lane-script scripts/remote_lane_pr106_latent_sidecar.sh \
  --label lane_pr106_latent_sidecar \
  --predicted-band 0.205 0.208 \
  --estimated-cost 0.60 \
  --council-priority 1 \
  --max-dph 0.30 \
  --env PR106_LATENT_MODE=score_table \
  --env PR106_LATENT_SCORE_TABLE_RESUME=1
```

Result: refused before instance creation because no RTX 4090 offer was
available below `$0.30/hr`.

Retried with the canonical launcher ceiling:

```bash
--max-dph 0.50
```

Result: offer `36197114` found, but instance creation failed with:

```text
Your account lacks credit; see the billing page.
```

No `instance_id` was returned, no Vast billing instance was created, and no
active claim was added. Vast is blocked on account credit for this lane.

## Kaggle PR101 proxy harvest

Kaggle credentials exist at `~/.kaggle/kaggle.json`; local venv lacks the
`kaggle` package, so the repo's `uv run --with kaggle` fallback was used.

`scripts/kaggle_check.py` showed:

- `adpena/pr101-bias-refine`: COMPLETE
- `adpena/pr101-proxy-sweep`: COMPLETE
- older asym/constrained-gen kernels: ERROR or NOT FOUND

Downloaded completed outputs into ignored custody:

- `experiments/results/kaggle_pr101_bias_refine_harvest_20260511_codex/`
- `experiments/results/kaggle_pr101_proxy_sweep_harvest_20260511_codex/`

Best harvested candidates:

- bias refine: `bias_refine_cmaes_0050`
  - params: `bias_b=-1.0027525485325404`, `bias_g=-0.9922764812932092`,
    `bias_r=-1.0055585926234436`
  - proxy objective: `0.19285462481263735`
  - authority: `score_claim=false`, `proxy_only=true`,
    `ready_for_exact_eval_dispatch=false`
- broad proxy sweep: `proxy_cmaes_0037`
  - proxy objective: `0.19287550335547282`
  - authority: `score_claim=false`, `proxy_only=true`,
    `ready_for_exact_eval_dispatch=false`

The best bias-refine proxy is slightly worse than the current A1
`[contest-CPU]` anchor (`0.19284758`) and should not be promoted as a score
claim or exact-dispatch priority without a separate reason.

## Materialized local handoff

Materialized only the bias-refine row because it uses the runtime-consumed
three-bias schema:

- handoff:
  `experiments/results/kaggle_pr101_bias_refine_harvest_20260511_codex/pr101_bias_refine/local_materialization/archive_builder_handoff.json`
- materialization manifest:
  `experiments/results/kaggle_pr101_bias_refine_harvest_20260511_codex/pr101_bias_refine/local_materialization/materialization_manifest.json`
- runtime packet manifest:
  `experiments/results/kaggle_pr101_bias_refine_harvest_20260511_codex/proxy_runtime_packet/runtime_packet_manifest.json`
- runtime-consumption proof:
  `experiments/results/kaggle_pr101_bias_refine_harvest_20260511_codex/proxy_runtime_packet/runtime_consumption_proof.json`

Runtime proof:

- `runtime_consumption_proven_for_supported_bias_params=true`
- `inflate_sh_routes_to_packet_inflate_py=true`
- `ready_for_exact_eval_dispatch=false`
- blockers:
  - `proxy_substrate_not_contest_exact_eval`
  - `no_contest_cuda_auth_eval`
  - `no_scorer_runtime_probe_not_contest_auth_eval`
  - `active_level2_lane_dispatch_claim_required_before_exact_eval`

## Classification

This is useful optimizer/proxy evidence and a valid byte-closed local handoff,
not a score result. It lowers no claimed score. It should inform future
CMAES/Optuna priors, but the immediate score-lowering priority remains:

1. harvest the active T1 Modal run when it terminates;
2. rerun PR103-on-PR106 CPU/CUDA raw-output-manifest pair for mechanism
   closure;
3. use Modal/Kaggle/free-provider paths for training/proxy work while Vast is
   credit-blocked.
