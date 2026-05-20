# Codex Findings - V8 Learned-Compression Faiss Adversarial Review

**UTC:** 2026-05-20T03:43:52Z  
**Owner:** codex Worker C  
**Lane:** `lane_v8_learned_compression_faiss_scaffold_codex_20260520`  
**Scope:** adversarial review, guards, tests, and operator recipe checks  
**Score claim:** none  
**Promotion eligible:** false  
**Ready for exact eval dispatch:** false  
**Provider spend:** none

## Verdict

`NON_PROMOTIONAL_CONTINUE_LOCAL_ONLY`.

V8 now has a local byte-closed fixture export and deterministic inflate fixture,
but it is still not a contest candidate. Paid dispatch and promotion remain
blocked by real contest-video scorer training, exact CUDA auth eval, paired axis
custody, and Catalog #324 post-training Tier-C validation.

## Changes Landed

- Hardened `.omx/operator_authorize_recipes/substrate_v8_learned_compression_faiss_modal_a100_smoke.yaml` with:
  - `pre_promotion_blockers` for learned byte-closed archive, contest runtime custody, exact CUDA eval, paired CPU/CUDA axis, and score provenance.
  - explicit `predicted_band_kind: predicted_score_band`, `predicted_band_axis: contest-CPU`, and `predicted_band_validation_status: pending_post_training`.
- Added V8 guard tests in `src/tac/tests/test_v8_learned_compression_faiss_scaffold.py`.
- Wired the V8 recipe/trainer into `tools/asymptotic_pursuit_candidate_readiness_assessment.py` so readiness checks consume the actual disabled recipe instead of reporting a missing recipe.

## Findings

1. **Closed - recipe false-authority gap.**
   The V8 recipe now carries structured non-promotional metadata and hard blockers at lines 10-22 and predicted-band axis/status metadata at lines 30-36. The guard tests at `src/tac/tests/test_v8_learned_compression_faiss_scaffold.py:219` and `:247` force the recipe to stay disabled and force `operator_authorize` to refuse a bad `dispatch_enabled=true` flip if pre-promotion blockers remain.

2. **Closed - readiness tool was not consuming the actual V8 recipe.**
   Before this pass, `tools/asymptotic_pursuit_candidate_readiness_assessment.py` had no `v8_learned_compression_faiss` recipe/trainer mapping, so a V8 assessment reported `RECIPE_MISSING` and defaulted recipe fields. The mapping now exists at `tools/asymptotic_pursuit_candidate_readiness_assessment.py:138`, `:152`, and aliases at `:185`. The regression test at `src/tac/tests/test_v8_learned_compression_faiss_scaffold.py:288` verifies the tool sees the disabled recipe and returns `NEEDS_FIX`, not `READY`.

3. **Open blocker - local pre-deploy correctly refuses V8 as dispatch proof.**
   `tools/local_pre_deploy_check.py --strict` exits 1 for V8: missing reachable auth eval and recipe remains non-dispatchable. Dispatch optimization protocol now passes after adding explicit research-only Tier-1 waivers and Tier-2 recipe hardware-routing metadata. This is the correct contest-compliance outcome while `experiments/train_substrate_v8_learned_compression_faiss.py` marks `score_claim=false`, `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`, and the manifest lists remaining promotion blockers.

4. **Closed locally - macOS Faiss/Torch abort avoided by explicit OpenMP guard.**
   Re-running the combined ATW/Faiss/probe/V8 pytest slice with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` now passes. This is still a macOS/native-library sensitivity to preserve in future Faiss tests, not promotion authority.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q src/tac/tests/test_v8_learned_compression_faiss_scaffold.py -p no:cacheprovider`  
  Result: `15 passed`.
- `PYTHONDONTWRITEBYTECODE=1 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q src/tac/tests/test_atw_v2_1_faiss_ivf_pq.py src/tac/tests/test_probe_atw_v2_1_faiss_pq_disambiguator.py src/tac/tests/test_v8_learned_compression_faiss_scaffold.py -p no:cacheprovider`  
  Result: `53 passed`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/operator_authorize.py --recipe substrate_v8_learned_compression_faiss_modal_a100_smoke --dry-run --target none`  
  Result: exit 0 dry-run refusal on `dispatch_enabled=false` plus exact blockers; no claim and no dispatch.
- `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_predicted_band_provenance.py --recipe-glob '.omx/operator_authorize_recipes/substrate_v8_learned_compression_faiss_modal_a100_smoke.yaml' --strict`  
  Result: PASS, `research_only`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py --candidates v8_learned_compression_faiss --json`  
  Result: exit 1 expected `NEEDS_FIX`; operator-authorize not recommended.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/local_pre_deploy_check.py --trainer experiments/train_substrate_v8_learned_compression_faiss.py --recipe substrate_v8_learned_compression_faiss_modal_a100_smoke --strict`  
  Result: exit 1 expected refusal; dispatch optimization protocol passes, remaining failures are auth-eval reachability and recipe status.

## Completion Criteria

V8 must not become promotional until all of these are true:

1. A learned V8 archive, not just the local fixture, is byte-closed and consumed by `inflate.sh`.
2. Runtime custody records archive bytes, archive SHA-256, runtime tree SHA, command, hardware, logs, and lane-claim linkage.
3. Real contest-video score-aware eval-roundtrip training has run.
4. Catalog #324 post-training Tier-C validation exists for the trained archive.
5. Exact CUDA auth eval lands through the canonical archive/runtime path.
6. Paired CPU/CUDA axis evidence remains explicitly labeled and non-converted.
7. The V8 operator recipe clears `research_only`, `dispatch_blockers`, and `pre_promotion_blockers` only after the above artifacts exist.
8. Native Faiss tests keep the explicit OpenMP guard or move Faiss training into a crash-isolated subprocess before provider authority depends on them.
