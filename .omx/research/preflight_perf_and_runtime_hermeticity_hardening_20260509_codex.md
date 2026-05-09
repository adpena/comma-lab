# Preflight performance + runtime hermeticity hardening (2026-05-09)

<!-- generated_at: 2026-05-09T22:45:00Z -->
<!-- evidence_grade: dev_guard_hardening; no score promotion; no remote dispatch -->

## Scope

This landing responds to the operator's request that preflight stay fast
enough for development velocity, use shared indexed scans instead of reopening
every file per check, and fail closed on non-hermetic Phase 1 runtime packets.

No remote, GPU, or exact-eval dispatch was launched.

## Preflight performance changes

- `check_test_files_imports_resolve` now scans only test files under
  `src/tac/tests` and top-level `tests`, then indexes only the project modules
  those tests import. The native Rust AST indexer no longer parses the whole
  repo for this check.
- The same check now covers bare `import tac.foo` / `import experiments.foo`
  statements, not only `from ... import ...`, preserving correctness while
  narrowing the indexed module set.
- `check_no_bare_round_in_eval_roundtrip` now uses the shared `SourceIndex`
  substring index for `.round(` + `F.interpolate` candidate narrowing before
  its precise AST scan. It intentionally does not prefilter on `roundtrip`, so
  case-insensitive `RoundTrip` spellings remain covered.
- `check_no_raw_zip_extractall` now uses the shared substring index for
  `extractall(` candidates before line-level validation.
- `SourceIndex` persistent text-fact schema was bumped to v6 so newly indexed
  hot substrings cannot false-miss from older cached rows.

## Runtime hermeticity changes

- `tac.phase1_packet_compiler.FORBIDDEN_NETWORK_TOKENS` now blocks
  `uv run --with`, `uv pip install`, `--extra-index-url`, `--index-url`,
  `--find-links`, `http://`, and `https://` in `inflate.sh`, in addition to
  the older curl/wget/pip/git tokens.
- Catalog #146 in `src/tac/preflight.py` now catches the same runtime-fetch
  bug at the Phase 1 trainer source level before packet compilation.
- `experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py`
  now emits `inflate.sh` using `"${PYTHON:-python3}" "$HERE/inflate.py" ...`
  with explicit `$DATA_DIR`, `$OUTPUT_DIR`, and `$FILE_LIST`, not `uv run`.

## Verification

Focused tests:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_source_index.py \
  src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py \
  src/tac/tests/test_preflight_meta_bugs.py::TestTestFilesImportsResolve \
  src/tac/tests/test_preflight_meta_bugs.py::TestArchiveBuildersUseDeterministicZip::test_check_catches_raw_zip_extractall \
  src/tac/tests/test_preflight_meta_bugs.py::TestArchiveBuildersUseDeterministicZip::test_check_allows_canonical_safe_zip_extractor \
  -q
```

Result: `40 passed`.

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_build_phase1_packet_compiler.py::test_fail_closed_on_network_token_in_sh \
  src/tac/tests/test_build_phase1_packet_compiler.py::test_fail_closed_on_uv_runtime_dependency_fetch_in_sh \
  src/tac/tests/test_build_phase1_packet_compiler.py::test_forbidden_network_tokens_includes_curl_wget \
  src/tac/tests/test_build_phase1_packet_compiler.py::test_compiler_rejects_trainer_scaffold_inflate_sh_signature \
  tests/paradigm_delta_epsilon_zeta/test_phase1_trainer_write_runtime_fix.py \
  -q
```

Result: `37 passed`, with pre-existing torch/torch-geometric deprecation
warnings only.

Default operator preflight:

```bash
/usr/bin/time -p .venv/bin/python -m tac.preflight
```

Result: `PREFLIGHT PASSED`, `real 10.49`.

Exhaustive latency profile:

```bash
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-all-codebase \
  --json-out experiments/results/preflight_latency_profile_20260509_codex_after2.json \
  --top 30 --max-step-records 80 --preflight-timeout-s 30
```

Result: the exhaustive all-codebase surface still intentionally times out at
`30.370s`, but every recorded step passed. The previous worst offender
`check_test_files_imports_resolve` dropped out of the top 30. Remaining
highest targets are `preflight_filename_contract`, `check_no_proxy_metric_drives_decision`,
`check_profile_keys_have_resolvers`, and `preflight_loader_format_safety`.

## Next optimization targets

1. Move `preflight_filename_contract` and `preflight_loader_format_safety`
   onto a single shared parsed-AST/query pass rather than per-check AST walks.
2. Add lower-level Rust/Rayon text-fact extraction for the `SourceIndex`
   persistent facts cache so cold all-codebase preflight is not Python I/O
   bound.
3. Keep the default operator preflight below 30s; if it regresses above that,
   treat it as a developer-velocity blocker rather than normal variance.
