# HDM5 Self-Describing Decoder Search Negative (2026-05-13, Codex)

## Scope

Planning-only PacketIR/decoder-section work on the PR106 R2 HDM4 exact-eval
release surface. This ledger records a byte-accounting result only. It is not a
score claim and is not an archive promotion.

## Input Custody

- Source archive:
  `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/archive.zip`
- Source archive SHA-256:
  `218ae16f3f13b722e9752d698667ed8770151e40d44b5756c0ebbccb7682825f`
- Source archive bytes: `186492`
- Source decoder section codec: `hdm4_q_brotli_split`
- Source decoder section bytes: `169990`
- Existing exact axes for this same archive/runtime surface:
  - `[contest-CUDA]` `0.20642625334307507`
  - `[contest-CPU]` `0.22787475059700513`

## Command

```bash
PYTHONPATH=src .venv/bin/python tools/profile_hnerv_decoder_structural_recode.py \
  --source-archive experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/archive.zip \
  --source-label pr106_r2_hdm4_release_surface \
  --include-hdm5-search \
  --hdm5-max-parts 8 \
  --hdm5-workers 8 \
  --hdm5-top-k 16 \
  --json-out experiments/results/pr106_r2_hdm5_decoder_search_20260513_codex/profile.json
```

- Profile artifact SHA-256:
  `1883ae88d16281857438445027930def1ab57781d9cd8e8e29ad12babf8be8f4`
- Profile artifact is ignored under `experiments/results/`; this ledger is the
  durable tracked summary.

## Result

HDM5 self-describing order/split search does **not** beat HDM4.

- Search family count: `7`
- Candidate count: `56`
- Best HDM5 self-describing bytes: `170024`
- Best HDM5 delta vs consumed HDM4 decoder section: `+34` bytes
- Best family: `hdm4_role_order`
- Best split points: `[6, 9, 26, 28]`
- Best fixed-recipe runtime projection: `169990` bytes, delta `0` vs HDM4
- Verdict:
  `hdm5_self_describing_search_does_not_beat_hdm4`

The best HDM5 candidate recovers the same order and split points as HDM4, but
loses by 34 bytes because HDM5 stores its record order and split metadata
explicitly. That is the intended conservative accounting: do not rely on a
future hard-coded recipe to claim a planning win.

The contest-compliant fixed-recipe projection was also checked. Moving the
order/split recipe into runtime code, as HDM4 already does, would reduce the
best HDM5 candidate back to exactly `169990` bytes. It ties HDM4; it does not
beat it. The next fixed-recipe projection is `169991` bytes, so this search
does not justify a new runtime/eval packet.

## Engineering Consequence

Do not add HDM5 to archive builder/runtime/eval dispatch from this
self-describing planner. Runtime implementation would be negative-EV for this
specific fixture because the byte-closed planning candidate is already worse
than the exact-evaluated HDM4 section it would replace.

Valid reactivation criteria:

1. A fixed-recipe HDM5 planner that proves a byte win after accounting for the
   recipe ID and runtime code budget. The current 7-family/8-part planner only
   ties HDM4 under this projection.
2. A non-Brotli entropy transform that beats `169990` consumed decoder-section
   bytes with an explicit decoder and no-op proof.
3. A latent/sidecar section transform that changes consumed bytes and has a
   packet/runtime path.

## Guardrails Added

- `parse_decoder_section_for_recode()` now parses legacy Brotli, HDM3, and HDM4
  decoder sections so profiling compares against the consumed archive section.
- `search_hdm5_q_brotli_split_recipes()` is planning-only:
  `score_claim=false`, `archive_ready=false`,
  `ready_for_exact_eval_dispatch=false`.
- Tests cover HDM5 roundtrip, malformed fixture rejection,
  serial/parallel determinism, HDM4-source profiling, and CLI
  `--include-hdm5-search`.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q src/tac/tests/test_hnerv_decoder_recode.py
# 17 passed

ruff check src/tac/hnerv_decoder_recode.py \
  src/tac/tests/test_hnerv_decoder_recode.py \
  tools/profile_hnerv_decoder_structural_recode.py
# All checks passed
```
