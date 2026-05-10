# Kaggle proxy-sweep substrate (2026-05-10)

## Summary

Worker A built a private Kaggle script-kernel substrate for cheap PR101/A1-style
configuration search. The substrate is intentionally proxy-only and cannot be
used as exact auth eval, contest CUDA evidence, or a score claim.

## Artifacts

- Builder: `tools/build_kaggle_proxy_sweep_kernel.py`
- Kernel directory: `experiments/kaggle_kernels/pr101_proxy_sweep/`
- Tests: `tests/test_build_kaggle_proxy_sweep_kernel.py`

## Operator-controlled launch command

```bash
uv run --with kaggle kaggle kernels push -p experiments/kaggle_kernels/pr101_proxy_sweep
```

The builder prints this command but does not execute it. No Kaggle push,
provider launch, dispatch claim, or remote job is performed by default.

## Evidence contract

The builder manifest and generated kernel outputs always declare:

- `score_claim=false`
- `score_claim_valid=false`
- `ready_for_exact_eval_dispatch=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `dispatch_attempted=false`
- `proxy_only=true`
- `exact_auth_eval_performed=false`
- `contest_cuda_auth_eval=false`
- `mps_auth_eval=false`
- `archive_zip_emitted=false`
- `inflate_runtime_emitted=false`
- `evidence_semantics=kaggle_gpu_proxy_config_search_only_not_exact_auth_eval`

Dispatch blockers:

- `kaggle_proxy_substrate_not_contest_exact_eval`
- `no_archive_zip_emitted`
- `no_inflate_runtime_emitted`
- `no_contest_cuda_auth_eval`
- `operator_must_promote_candidate_manually`

## Research-only integration

`research_only=true` by construction. This landing does not create a new score
lane or mutate `.omx/state/lane_registry.json`; it is a proxy substrate for
candidate generation. Any winning `best_proxy_candidate.json` must be copied
into a real archive-builder or training dispatch and then pass a claimed exact
CUDA archive/eval path before it can influence lane status.

Six-hook disposition:

- Sensitivity-map contribution: N/A, proxy substrate emits config candidates only.
- Pareto constraint: N/A until a candidate becomes a byte-closed archive.
- Bit-allocator hook: N/A until a promoted candidate changes charged bytes.
- Cathedral autopilot dispatch hook: blocked by proxy-only evidence semantics.
- Continual-learning posterior: blocked; no empirical anchor is emitted.
- Probe-disambiguator: N/A; optimizer choice is exposed as `random`, `optuna`, or
  `cmaes` mode in the generated script.

## Score-lowering role

Kaggle is useful here as free GPU/proxy wall-clock for optimizer plumbing,
warm-start curves, and config selection. It is not a promotion substrate. The
promotion path is:

1. Run the private Kaggle proxy kernel.
2. Harvest `best_proxy_candidate.json`.
3. Feed the candidate into a real archive-builder or training dispatch.
4. Claim a lane before remote exact-eval dispatch.
5. Promote only after exact CUDA archive/runtime custody is complete.
