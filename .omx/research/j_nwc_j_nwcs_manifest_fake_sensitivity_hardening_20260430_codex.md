# J-NWC/J-NWCS Manifest And Fake-Sensitivity Hardening - 2026-04-30

Evidence grade: engineering hardening / empirical tests only. No score claim,
ranking claim, lane promotion, or method retirement claim is made here.

## Scope

Owned stream: J-NWC/J-NWCS corpus manifest replay and fake/debug sensitivity
promotion hardening.

Reviewed surfaces:

- `scripts/remote_lane_j_nwc_neural_weight_compression.sh`
- `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh`
- `scripts/remote_lane_j_nwcs_ec_stack.sh`
- `src/tac/neural_weight_codec.py`
- `src/tac/neural_weight_codec_sensitivity.py`
- `src/tac/neural_weight_corpus.py`
- `src/tac/tests/test_neural_weight_codec_corpus.py`
- `src/tac/tests/test_remote_lane_j_nwc_hardening.py`
- `src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py`
- `.omx/research/nwcs1_build_smoke_and_sensitivity_plan_20260430_agent.md`

## Changes Landed

1. Corpus manifest replay now rejects unsafe relocated paths.
   - `build_corpus_from_manifest(..., replay_root=...)` rejects absolute
     `relative_path`, backslashes, parent traversal, empty/current-dir parts,
     hidden path parts, and `__MACOSX`.
   - Replayed checkpoint paths are resolved and verified to remain under the
     replay root before loading.
   - Direct in-memory manifest dicts now reject unsupported `schema_version`
     the same way JSON-loaded manifests do.
   - Manifest generation excludes checkpoint paths outside the declared
     `corpus_dir` instead of silently recording selected files with absolute
     non-relocatable `relative_path` values.

2. Debug/fake J-NWCS sensitivity can no longer produce auth-eval-shaped score
   JSON by default.
   - Added `NWCS_BUILD_ONLY="${NWCS_BUILD_ONLY:-0}"` to both J-NWCS scripts.
   - If `NWCS_ALLOW_DEBUG_SENSITIVITY=1` or `NWCS_BUILD_ONLY=1`, the scripts
     stop after archive construction and before `experiments/contest_auth_eval.py`.
   - The build-only exit writes `provenance.json` and `final_record.json` with
     `build_only=true`, `score_claim=false`, `promotion_eligible=false`,
     `auth_eval_skipped=true`, and `result_json=null`.
   - This preserves smoke/build artifacts while preventing synthetic uniform
     sensitivity from entering the normal score-harvest path.

3. Tests now pin the hardened behavior.
   - Added corpus replay tests for traversal/absolute path rejection, direct
     dict schema rejection, and outside-`corpus_dir` exclusion.
   - Added remote-script static guards requiring the debug/build-only stop to
     appear before the auth eval command and to mark no score claim.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile src/tac/neural_weight_corpus.py
bash -n scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh
bash -n scripts/remote_lane_j_nwcs_ec_stack.sh
bash -n scripts/remote_lane_j_nwc_neural_weight_compression.sh
.venv/bin/python -m pytest \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  -q
.venv/bin/python -m py_compile \
  src/tac/neural_weight_corpus.py \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py
.venv/bin/python -m pytest \
  src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  -q
git diff --check -- \
  scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh \
  scripts/remote_lane_j_nwcs_ec_stack.sh \
  src/tac/neural_weight_corpus.py \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py
git diff --no-index --check /dev/null src/tac/neural_weight_corpus.py
git diff --no-index --check /dev/null src/tac/tests/test_neural_weight_codec_corpus.py
git diff --no-index --check /dev/null src/tac/tests/test_remote_lane_j_nwc_hardening.py
git diff --no-index --check /dev/null \
  .omx/research/j_nwc_j_nwcs_manifest_fake_sensitivity_hardening_20260430_codex.md
```

Observed:

```text
18 passed in 0.49s
25 passed in 0.49s
```

All listed syntax, compile, pytest, path-limited diff, and no-index diff
checks passed.

## Changed Files

- `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh`
- `scripts/remote_lane_j_nwcs_ec_stack.sh`
- `src/tac/neural_weight_corpus.py`
- `src/tac/tests/test_neural_weight_codec_corpus.py`
- `src/tac/tests/test_remote_lane_j_nwc_hardening.py`
- `.omx/research/j_nwc_j_nwcs_manifest_fake_sensitivity_hardening_20260430_codex.md`

## Residual Findings

- No promotable `ANCHOR_SENSITIVITY_PT` or `CORPUS_SENSITIVITY_PT` was present
  in this pass. Exact J-NWCS promotion still requires real CUDA scorer-derived
  sensitivity with anchor/corpus hashes, parameter metadata, positive finite
  signal, and component sensitivity provenance.
- The scripts still rely on the existing inline sensitivity validators. A
  follow-up cleanup could move those validators into a shared Python module so
  the J-NWCS and J-NWCS-EC scripts cannot drift.
- No CUDA auth eval was run in this hardening pass because no candidate
  promotable archive was produced.

## Worker 4 Addendum - 2026-04-30T22:12Z

Scope: J-NWC/J-NWCS exact-eval provenance hardening only. No training dispatch,
CUDA auth eval, score claim, promotion, ranking, or method-retirement claim is
made here.

Patch landed:

- `scripts/remote_lane_j_nwc_neural_weight_compression.sh`
- `scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh`
- `scripts/remote_lane_j_nwcs_ec_stack.sh`
- `src/tac/tests/test_remote_lane_j_nwc_hardening.py`

The three exact-eval paths already ran `scripts/adjudicate_contest_auth_eval.py`
with CUDA/sample/component gates. This addendum closes the remaining custody
gap: final `provenance.json` and `final_record.json` now also surface
`contest_auth_eval.adjudicated.json`, `adjudication_provenance.json`,
`score_source`, and explicit `adjudication_required=true` /
`component_gates_required=true` fields. Artifact custody now hashes both
adjudication files.

Focused verification:

```bash
.venv/bin/python -m py_compile src/tac/tests/test_remote_lane_j_nwc_hardening.py
bash -n scripts/remote_lane_j_nwc_neural_weight_compression.sh \
  scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh \
  scripts/remote_lane_j_nwcs_ec_stack.sh
.venv/bin/python -m pytest \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py \
  -q
.venv/bin/python -m pytest src/tac/tests/test_remote_auth_eval_hardening.py -q
git diff --check -- \
  scripts/remote_lane_j_nwc_neural_weight_compression.sh \
  scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh \
  scripts/remote_lane_j_nwcs_ec_stack.sh \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py
```

Observed:

```text
31 passed in 0.52s
26 passed in 1.48s
```

Exact eval command template for any future promotable candidate remains:

```bash
AUTH_EVAL_DEVICE=cuda \
COMPONENT_SENSITIVITY_MANIFEST=<component_sensitivity_v1.json> \
ANCHOR_SENSITIVITY_PT=<anchor_sensitivity.pt> \
CORPUS_SENSITIVITY_PT=<corpus_sensitivity.pt> \
bash scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh
```

The script must produce and preserve `RESULT_JSON`,
`contest_auth_eval.adjudicated.json`, `adjudication_provenance.json`,
`provenance.json`, `final_record.json`, archive SHA/bytes, manifest hashes, and
logs before any score claim is considered.
