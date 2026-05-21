# Codex Findings: MLX Cache Canonical Equation Identity Hardening

UTC: 2026-05-21T21:25:38Z

## Verdict

PROCEED for MLX local-acceleration transfer calibration, with a stricter
authority boundary.

An adversarial review found the previous cache audit could pass a cache with
bogus scorer-input tensor hashes when archive SHA, inflated-output aggregate
SHA, and pair count matched. This landing hardens
`scorer_input_cache_hash_identity_v1` so transfer eligibility now requires all
of:

- archive SHA match;
- inflated-output aggregate SHA match;
- raw file SHA match;
- pair/sample count match;
- scorer-input hash-domain match;
- SegNet last-RGB, PoseNet YUV6-pair, and pair-index array SHA-256 match;
- the same three tensor shapes match.

If an auth-eval JSON predates `scorer_input_cache_hash_manifest` provenance,
`tools/audit_mlx_scorer_input_cache.py` must receive an independent
`--reference-cache-manifest` from the target raw surface. Without auth-side or
reference scorer-input hashes, the audit fails closed and is debug-only.

## What Landed

- Added canonical equation implementation:
  `tac.canonical_equations.scorer_input_cache_hash_identity`.
- Exported the equation from `tac.canonical_equations`.
- Wired `tac.local_acceleration.mlx_cache_audit` to emit the canonical equation
  result and to fail closed on missing or mismatched scorer-input hashes,
  shapes, raw SHA, or hash domain.
- Added `--reference-cache-manifest` to
  `tools/audit_mlx_scorer_input_cache.py`.
- Added `hash_domain` and `producer_environment` to new full-cache and
  hash-only scorer-input manifests.
- Re-registered `scorer_input_cache_hash_identity_v1` in
  `.omx/state/canonical_equations_registry.jsonl`.

## Empirical Anchor

Local FEC6/PR101 full cache audited against the independent hash-only manifest
and local macOS advisory auth-eval surface:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_mlx_scorer_input_cache.py \
  --cache-manifest experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs/manifest.json \
  --auth-eval experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/local_macos_cpu_advisory_smoke_20260519T143700Z_workdir/contest_auth_eval.json \
  --reference-cache-manifest experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T211200Z_hash_only_macos_full600/manifest.json \
  --output experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs/cache_vs_macos_cpu_advisory_audit.v3.json
```

Result:

- audit verdict: `PASS_CACHE_AUTH_EVAL_IDENTITY`
- canonical equation verdict: `PASS_SCORER_INPUT_CACHE_IDENTITY`
- identity residual: `0`
- compared hashes: SegNet last RGB, PoseNet YUV6 pair, and pair indices all
  matched.
- compared shapes: all matched.

The same local cache remains blocked for the existing Modal contest-CPU auth
surface:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_mlx_scorer_input_cache.py \
  --cache-manifest experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs/manifest.json \
  --auth-eval experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json \
  --reference-cache-manifest experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T211200Z_hash_only_macos_full600/manifest.json \
  --output experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs/cache_vs_modal_contest_cpu_audit.v3.json
```

Result:

- audit verdict: `FAIL_CACHE_AUTH_EVAL_IDENTITY`
- canonical equation verdict: `FAIL_SCORER_INPUT_CACHE_IDENTITY`
- identity residual: `2`
- blockers: inflated-output aggregate mismatch and raw SHA mismatch.

This preserves the critical axis boundary: macOS-local MLX cache work can be
used for local transfer calibration only against the matching raw/scorer-input
surface, not as a contest-CPU or contest-CUDA score proxy.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/canonical_equations/tests/test_scorer_input_cache_hash_identity.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_scorer_fidelity.py -q
```

Result: `23 passed in 2.22s`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m compileall -q \
  src/tac/canonical_equations/scorer_input_cache_hash_identity.py \
  src/tac/canonical_equations/__init__.py \
  src/tac/local_acceleration/mlx_cache_audit.py \
  src/tac/local_acceleration/mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/canonical_equations/tests/test_scorer_input_cache_hash_identity.py \
  tools/audit_mlx_scorer_input_cache.py
```

Result: pass.

## Recommended Next Action

Mirror the scorer-input hash bridge into the CUDA Modal auth-eval wrapper and
run a fresh contest-CPU or contest-CUDA auth eval with
`scorer_input_cache_hashes=True`. That produces the first remote auth-side
tensor-hash identity artifact needed before MLX surrogate training signals can
be trusted within a small axis-specific margin.
