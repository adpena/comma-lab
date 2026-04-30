# J-NWC Build-Only Manifest Provenance Hardening - 2026-04-30

Evidence grade: engineering hardening / static and focused tests only. No
training dispatch, CUDA auth eval, score claim, promotion, ranking, or method
retirement claim is made here.

## Scope

Owned stream: corpus codec / J-NWC amortization build-only path.

Reviewed surfaces:

- `scripts/remote_lane_j_nwc_neural_weight_compression.sh`
- `src/tac/tests/test_remote_lane_j_nwc_hardening.py`
- Existing J-NWC/NWCS corpus and build-only hardening ledgers.

## Change Landed

Plain J-NWC now has an explicit `NWC_BUILD_ONLY=1` path after deterministic
archive construction and before `experiments/contest_auth_eval.py`.

The build-only path writes:

- `provenance.json`
- `final_record.json`

Both records mark the artifact non-promotable with:

- `build_only=true`
- `score_claim=false`
- `promotion_eligible=false`
- `auth_eval_skipped=true`
- `result_json=null`

The provenance also records SHA-256 and byte custody for the anchor archive,
extracted anchor payloads, corpus manifest, codec checkpoint, NWC renderer,
and candidate archive.

## Verification

Commands run:

```bash
bash -n scripts/remote_lane_j_nwc_neural_weight_compression.sh \
  scripts/remote_lane_j_nwcs_sensitivity_aware_codec.sh \
  scripts/remote_lane_j_nwcs_ec_stack.sh
.venv/bin/python -m py_compile src/tac/tests/test_remote_lane_j_nwc_hardening.py
.venv/bin/python -m pytest \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py \
  src/tac/tests/test_neural_weight_codec_corpus.py \
  src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py \
  -q
git diff --check -- \
  scripts/remote_lane_j_nwc_neural_weight_compression.sh \
  src/tac/tests/test_remote_lane_j_nwc_hardening.py
```

Observed:

```text
29 passed in 0.51s
```

## Residual Blockers

- No J-NWC build-only run was dispatched in this pass.
- No CUDA auth eval was run.
- J-NWC/NWCS promotion still requires validated sensitivity/corpus provenance,
  exact archive custody, CUDA auth eval JSON, adjudication provenance, and
  component gates.
