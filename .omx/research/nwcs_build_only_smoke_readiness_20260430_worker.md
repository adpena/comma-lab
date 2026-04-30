# NWCS Build-Only Smoke Readiness - 2026-04-30 Worker

Evidence grade: engineering hardening / empirical build-only smoke. No score
claim, ranking claim, promotion claim, or method-retirement claim is made here.
No `experiments/contest_auth_eval.py` command was run.

## Scope

Owned surface: J-NWC/J-NWCS build-only smoke and provenance readiness only.

Reviewed and touched:

- `src/tac/neural_weight_corpus.py`
- `src/tac/neural_weight_codec_sensitivity.py`
- `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh`
- `scripts/remote_lane_j_nwcs_ec_stack.sh`
- `src/tac/tests/test_neural_weight_codec_corpus.py`
- `src/tac/tests/test_neural_weight_codec_sensitivity.py`
- `src/tac/tests/test_remote_lane_j_nwc_hardening.py`

Read-only review context:

- `.omx/research/j_nwc_j_nwcs_manifest_fake_sensitivity_hardening_20260430_codex.md`
- `.omx/research/nwcs1_build_smoke_and_sensitivity_plan_20260430_agent.md`
- `src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py`
- `scripts/remote_lane_j_nwc_neural_weight_compression.sh`

## Changes Landed

1. Corpus manifest generation now excludes unsafe hidden or macOS-sidecar
   relative paths under a declared `corpus_dir`.
   - Hidden path parts such as `.cache/hidden.pt` and `__MACOSX/resource.pt`
     are recorded as `selected=false` with
     `exclusion_reason="unsafe_relative_path"`.
   - This extends the existing relocated replay guard to generation time, so a
     manifest cannot train on hidden sidecars and only fail later when replayed.

2. NWCS sensitivities now fail closed on negative or non-finite values.
   - Core `encode_with_variable_codebook` / quantile bucketing reject NaN/Inf
     and negative block sensitivities.
   - Both NWCS remote scripts reject negative anchor, corpus, and per-parameter
     sensitivities before encoding.

3. NWCS export no longer silently emits `tensor_only` metadata on
   promotion-eligible export when architecture inference fails.
   - `_infer_asymmetric_config(model)` failures now raise a fatal error while
     `promotion_eligible` is true.
   - Non-promotable/debug paths may still use `tensor_only`, but the warning is
     explicit.

4. Exact-path NWCS provenance heredocs now emit valid Python booleans.
   - Replaced shell-expanded `"promotion_eligible": $PROMOTION_ELIGIBLE` with
     `"promotion_eligible": "$PROMOTION_ELIGIBLE" == "true"`.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile \
  src/tac/neural_weight_corpus.py \
  src/tac/neural_weight_codec_sensitivity.py \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_neural_weight_codec_sensitivity.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py

bash -n scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh
bash -n scripts/remote_lane_j_nwcs_ec_stack.sh
bash -n scripts/remote_lane_j_nwc_neural_weight_compression.sh

.venv/bin/python -m pytest \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py \
  src/tac/tests/test_neural_weight_codec_sensitivity.py \
  -q
```

Observed:

```text
36 passed in 1.48s
```

## Local Build-Only Smoke

Ran a CPU-only synthetic NWCS smoke that:

- built a tiny `NWCS1` renderer container,
- detected it as `neural_weight_compression_sensitivity_v1`,
- loaded it through `load_nwcs_sensitivity_compressed_checkpoint`,
- packed a deterministic contest-shaped ZIP with `renderer.bin`,
  `masks.mkv`, and `optimized_poses.pt`,
- wrote explicit non-promotable provenance.

Artifact packet:

```text
experiments/results/nwcs_build_only_smoke_20260430_worker/
```

Summary:

```text
archive: experiments/results/nwcs_build_only_smoke_20260430_worker/archive_nwcs_build_only_smoke.zip
archive_bytes: 3895
archive_sha256: 9339fed08deffb25b73803b2e311ec34a93508256e5aff993758d23ec0e9c6fd
renderer_bytes: 7210
renderer_sha256: 2eaefccf8bf321a4334bc29f46ffc2816dc15fb04605437ef53abb5300731db5
archive_members: ["renderer.bin", "masks.mkv", "optimized_poses.pt"]
device: cpu
auth_eval_skipped: true
promotion_eligible: false
score_claim: false
```

This smoke is not a contest runtime validation. It is a local format,
container-load, deterministic-archive, and provenance check only.

## Residual Readiness Gaps

- No promotable `ANCHOR_SENSITIVITY_PT` or `CORPUS_SENSITIVITY_PT` was present
  in this pass. Exact J-NWCS promotion still requires CUDA scorer-derived,
  nonnegative finite sensitivity with anchor/corpus hashes and component
  sensitivity custody.
- The remote NWCS scripts intentionally require `AUTH_EVAL_DEVICE=cuda` and
  NVDEC preflight, so a no-GPU local build-only run should use a dedicated
  local smoke like the one above rather than weakening promotion guards.
- The J-NWCS-EC remote script still requires CUDA for the EC correction search
  before its build-only stop; this is correct for that stack but makes local
  no-GPU stack smoke infeasible through the remote script.
- Promotion remains blocked until a real `component_sensitivity_v1` packet and
  NWCS anchor/corpus sensitivity artifacts are available and reviewed.
