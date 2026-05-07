# HNeRV HDM3 Runtime Adapter Parity - 2026-05-07

## Classification

Runtime adapter integrated; exact CUDA score not attempted.

HDM3 remains a tiny rate-only HNeRV decoder-section candidate, but it now has a
strict runtime bridge instead of only a byte-level archive manifest. The bridge
does not fork public PR106 `inflate.py` or `src/codec.py`; it normalizes the
candidate payload before public inflate.

## Candidate

- source archive:
  `experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/pr106x_hnerv_brotli_repack_candidate.zip`
- source archive bytes: `186080`
- source archive SHA-256:
  `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`
- HDM3 candidate archive:
  `experiments/results/hnerv_hdm3_archive_candidate_20260507_codex/pr106x_lowlevel_brotli_hdm3_archive_candidate.zip`
- HDM3 candidate archive bytes: `186066`
- HDM3 candidate archive SHA-256:
  `5b5619628b54ccec44d51360ecb258dfe61742a581c7605c74d1ddaa5c025771`
- archive byte delta: `-14`
- rate-only score delta if components are identical:
  `-0.000009322025`

## Runtime Bridge

Files:

- `src/tac/hnerv_hdm3_runtime_adapter.py`
- `experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/hdm3_normalize.py`
- `experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/inflate.sh`

The adapter:

1. Parses the `0xff + len24 + decoder + latents` HNeRV payload.
2. Detects `HDM3` only by exact magic `b"HDM3"`.
3. Fails closed on malformed HDM3, unknown non-Brotli sections, bad length
   prefixes, q-stream length mismatch, and trailing bytes.
4. Decodes HDM3 to the fixed-schema raw decoder bytes.
5. Recompresses those raw decoder bytes with Brotli quality `10`.
6. Writes a temporary legacy packed payload for the unmodified public PR106
   inflater.

For this PR106x-lowlevel candidate, Brotli quality `10` is not arbitrary: it
recreates the exact source decoder section SHA:

- source decoder section SHA-256:
  `07725c39ff436195e319f258b1e033290de30e259bc3f103b1b487f21a698c5c`
- restored decoder section SHA-256:
  `07725c39ff436195e319f258b1e033290de30e259bc3f103b1b487f21a698c5c`
- raw decoder SHA-256:
  `f22eb6be56499fa5785f47f85d2bef7f71246f29674691fd3e06af733c8c0703`
- latents and sidecar SHA-256:
  `94257b33cf3083c5daa0f3b1e127cb7c51bee42a6416b19763eea7bf9ecc3c32`

The runtime proof shows the normalized candidate payload is byte-identical to
the exact source payload before public PR106 inflate:

- source payload SHA-256:
  `b6ba493aa37446143b003235eaeb3a49c0748a6d64392fb9a666a6c872629171`
- restored payload SHA-256:
  `b6ba493aa37446143b003235eaeb3a49c0748a6d64392fb9a666a6c872629171`
- `restored_payload_matches_source=true`
- `restored_decoder_section_matches_source=true`
- `inflate_output_parity_proven_by_payload_identity=true`

## Evidence

- candidate build manifest:
  `experiments/results/hnerv_hdm3_archive_candidate_20260507_codex/manifest.with_tool_run.json`
- runtime proof:
  `experiments/results/hnerv_hdm3_archive_candidate_20260507_codex/runtime_adapter_proof.with_tool_run.json`
- focused tests:
  `src/tac/tests/test_hnerv_hdm3_runtime_adapter.py`
  and `src/tac/tests/test_hnerv_hdm3_archive_candidate.py`

Commands:

```bash
.venv/bin/python -m pytest src/tac/tests/test_hnerv_hdm3_runtime_adapter.py src/tac/tests/test_hnerv_hdm3_archive_candidate.py -q
.venv/bin/ruff check src/tac/hnerv_hdm3_runtime_adapter.py src/tac/hnerv_hdm3_archive_candidate.py tools/prove_hnerv_hdm3_runtime_adapter.py src/tac/tests/test_hnerv_hdm3_runtime_adapter.py src/tac/tests/test_hnerv_hdm3_archive_candidate.py experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/hdm3_normalize.py
.venv/bin/python tools/prove_hnerv_hdm3_runtime_adapter.py --source-archive experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/pr106x_hnerv_brotli_repack_candidate.zip --candidate-archive experiments/results/hnerv_hdm3_archive_candidate_20260507_codex/pr106x_lowlevel_brotli_hdm3_archive_candidate.zip --output-dir experiments/results/hnerv_hdm3_archive_candidate_20260507_codex --json-out experiments/results/hnerv_hdm3_archive_candidate_20260507_codex/runtime_adapter_proof.with_tool_run.json
```

## Remaining Blockers

No score claim is made. Remaining dispatch blockers are:

- `strict_pre_submission_compliance_json_missing`
- `lane_dispatch_claim_missing`
- `exact_cuda_auth_eval_missing`

The runtime tree hash will change because the adapter now includes a normalizer
and a repo-local `tac` import closure. That is expected; the rigorous claim is
not "runtime tree unchanged." The rigorous claim is "the candidate is normalized
to the exact source payload before the unmodified public PR106 inflater."
