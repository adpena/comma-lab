# Codex Findings - V8 Blocker Burndown

**UTC:** 2026-05-20T03:44:13Z  
**Owner:** codex  
**Lane:** `lane_v8_learned_compression_faiss_scaffold_codex_20260520`  
**Scope:** V8 learned-compression/Faiss implementation, runtime, recipe, and tests  
**Score claim:** none  
**Promotion eligible:** false  
**Ready for dispatch:** false

## Landed

1. **Byte-closed runtime grammar:** `src/tac/substrates/v8_learned_compression_faiss/archive.py` defines fixed-header `V8FAISS1` raw-frame fixture archives with version, dimensions, payload length, and payload SHA-256. `submissions/v8_learned_compression_faiss/inflate.py` consumes the validated bytes and writes raw output without scorer imports or network.

2. **Local learned export smoke:** `src/tac/substrates/v8_learned_compression_faiss/smoke.py` runs a deterministic CPU-only categorical-posterior plus scale-hyperprior proxy export. The export uses distinct magic `V8FEXP01` so learned-export payloads cannot be mistaken for runtime raw-frame fixture archives.

3. **Trainer unblocked locally:** `experiments/train_substrate_v8_learned_compression_faiss.py` no longer dead-ends in `_full_main`. Both smoke and full-local modes write a non-promotional manifest, learned export payload, runtime fixture archive, and decoded raw fixture. It sets the macOS Faiss/Torch OpenMP guard before importing the torch-backed smoke path.

4. **Recipe stays fail-closed:** `.omx/operator_authorize_recipes/substrate_v8_learned_compression_faiss_modal_a100_smoke.yaml` remains `dispatch_enabled: false`. Blockers are now the real remaining promotion blockers: contest-video scorer training, exact CUDA auth eval, Catalog #324 Tier-C validation, and recipe authorization.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q src/tac/tests/test_v8_learned_compression_faiss_scaffold.py -p no:cacheprovider` -> `15 passed`
- `PYTHONDONTWRITEBYTECODE=1 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q src/tac/tests/test_atw_v2_1_faiss_ivf_pq.py src/tac/tests/test_probe_atw_v2_1_faiss_pq_disambiguator.py src/tac/tests/test_v8_learned_compression_faiss_scaffold.py -p no:cacheprovider` -> `53 passed`
- `.venv/bin/python experiments/train_substrate_v8_learned_compression_faiss.py --smoke --output-dir /tmp/v8_smoke_check --num-pairs 4 --categorical-groups 4 --codebook-size 8 --fixture-frames 3 --fixture-height 2 --fixture-width 2` -> wrote manifest, learned export, fixture archive, and raw fixture with `score_claim=false`
- `.venv/bin/python tools/operator_authorize.py --recipe substrate_v8_learned_compression_faiss_modal_a100_smoke --dry-run --target none` -> refused as expected because `dispatch_enabled=false`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/local_pre_deploy_check.py --trainer experiments/train_substrate_v8_learned_compression_faiss.py --recipe substrate_v8_learned_compression_faiss_modal_a100_smoke --strict` -> expected exit 1; dispatch protocol now passes, remaining failures are auth-eval reachability and disabled recipe status
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py --candidates v8_learned_compression_faiss --json` -> expected exit 1 with `NEEDS_FIX`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_predicted_band_provenance.py --recipe-glob '.omx/operator_authorize_recipes/substrate_v8_learned_compression_faiss_modal_a100_smoke.yaml' --strict` -> PASS, `research_only`
- `.venv/bin/python tools/lane_maturity.py validate` -> clean
- `.venv/bin/python tools/canonical_task_status.py --validate` -> valid
- `.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --strict` -> clean

## Findings

- The prior dead-scaffold blockers are locally burned down: categorical codewords exist, scale-hyperprior stats exist, byte-closed export exists, and eval-roundtrip contract is recorded in the learned export.
- Recipe/dispatch false-authority blockers are now explicit and machine-consumed: readiness assessment sees the disabled V8 recipe and local pre-deploy refuses only on the real auth-eval/recipe-status blockers.
- This is still not a submission or leaderboard artifact. No contest-video scorer training was run, no exact CUDA eval was launched, and no score can be promoted.
- The next promotion step is not another local scaffold patch. It is either a real scorer-trained V8 run with dispatch claim + operator authorization, or an explicit decision to keep V8 as a research-only substrate until a stronger frontier target clears.
