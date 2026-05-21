# Codex Findings — ATW2 CDF Materialized Smoke Archive Compaction

**Timestamp (UTC)**: 2026-05-21T05:53:48Z  
**Scope**: Convert the ATW2 CPU smoke path from raw `0.bin` only into a deterministic `archive.zip` producer, then verify compact-CDF removal against that materialized ZIP surface.  
**Verdict**: MATERIALIZED_SMOKE_PROOF_PASSED

## Summary

The prior ATW2 CDF real-archive scan found no parseable local ATW2 `archive.zip` artifacts. Codex closed that tooling gap by making the ATW2 smoke trainer build the same deterministic `archive.zip` and `submission/` runtime surface as the full trainer unless `--skip-archive-build` is passed.

The compact-CDF archive.zip rewriter then ran against the materialized smoke archive and preserved current inflate raw output exactly while reducing the ZIP by **2,398 bytes**.

This is a smoke/materialization proof only:

- `score_claim`: false
- `promotion_eligible`: false
- `ready_for_exact_eval_dispatch`: false

## Code Change

- `experiments/train_substrate_atw_codec_v2.py`
  - Smoke mode now writes deterministic `archive.zip` and `submission/` by default.
  - `--skip-archive-build` remains the opt-out.
  - `smoke_stats.json` now records `archive_zip_*`, `payload_0bin_sha256`, and `submission_dir` custody fields.
- `src/tac/substrates/atw_codec_v2/tests/test_atw_codec_v2.py`
  - The smoke trainer test now asserts the ZIP member is exactly `0.bin`, the ZIP payload matches the raw payload, and the runtime tree is emitted.

## Empirical Commands

Focused tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/substrates/atw_codec_v2/tests/test_atw_codec_v2.py \
  src/tac/substrates/atw_codec_v2/tests/test_cdf_dead_section.py
```

Result:

- `39 passed in 10.58s`

Materialize ATW2 smoke archive:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/train_substrate_atw_codec_v2.py \
  --output-dir experiments/results/atw2_cdf_materialized_smoke_20260521T055348Z/source \
  --epochs 1 --device cpu --smoke --variant B
```

Compact materialized `archive.zip`:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_atw2_cdf_dead_section.py \
  experiments/results/atw2_cdf_materialized_smoke_20260521T055348Z/source/archive.zip \
  --compact-cdf \
  --output-archive-zip experiments/results/atw2_cdf_materialized_smoke_20260521T055348Z/compact/archive.zip \
  --json-output experiments/results/atw2_cdf_materialized_smoke_20260521T055348Z/compact_proof.json \
  --device cpu
```

Scan materialized source root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/scan_atw2_cdf_compaction_candidates.py \
  experiments/results/atw2_cdf_materialized_smoke_20260521T055348Z/source \
  --json-output experiments/results/atw2_cdf_materialized_smoke_20260521T055348Z/scan_source.json \
  --md-output experiments/results/atw2_cdf_materialized_smoke_20260521T055348Z/scan_source.md
```

## Empirical Results

Smoke archive stats:

- `0.bin` bytes: `90,850`
- `0.bin` SHA-256: `83410a0a9962327bdc401112a94a897b461fe46b7a82e0644fa63220042d787b`
- Source `archive.zip` bytes: `90,664`
- Source `archive.zip` SHA-256: `dbde1713566a06538a6e718fca8d855b190cb3581d3ddf05d4bfa205c94a4f0f`
- Source ZIP compression method: deflated
- Source ZIP member: `0.bin`

Compaction proof:

- Compact `archive.zip` bytes: `88,266`
- Compact `archive.zip` SHA-256: `dff8e3f2ecfb06fcb787bfcc2954ceb314971469639a4d188b6cb5c8c65fb57f`
- ZIP bytes saved: `2,398`
- Rate-only delta estimate: `-0.001596729769586967`
- Inner ATW2 bytes saved: `2,552`
- Inner compact CDF bytes: `8`
- Raw output bytes compared: `48,832,128`
- Raw output equality: `true`
- Max absolute raw byte delta: `0`
- Source raw SHA-256: `c2c6e18c3ca3706d437126a5fd632be9d8e36eb88b8259353db17ddafc4c88fd`
- Compact raw SHA-256: `c2c6e18c3ca3706d437126a5fd632be9d8e36eb88b8259353db17ddafc4c88fd`

Scanner result:

- `archives_seen`: `1`
- `candidates_found`: `1`
- `zip_errors`: `0`
- Candidate `cdf_bytes`: `2,560`
- Candidate conservative bytes saved: `2,528`
- Candidate conservative rate-only delta: `-0.0016832914334928492`

Ignored proof artifact hashes:

- `experiments/results/atw2_cdf_materialized_smoke_20260521T055348Z/compact_proof.json`: `ff199c1c51bc8cf0ee9a1130ffe3a6c5f2f8c2e564c251fd05c5e3a0c7461495`
- `experiments/results/atw2_cdf_materialized_smoke_20260521T055348Z/source/archive.zip`: `dbde1713566a06538a6e718fca8d855b190cb3581d3ddf05d4bfa205c94a4f0f`
- `experiments/results/atw2_cdf_materialized_smoke_20260521T055348Z/compact/archive.zip`: `dff8e3f2ecfb06fcb787bfcc2954ceb314971469639a4d188b6cb5c8c65fb57f`

## Interpretation

The removal-paradigm CDF compactor is now proven across the full local toolchain surface:

1. trainer emits raw ATW2 bytes,
2. trainer emits deterministic `archive.zip`,
3. scanner discovers the ZIP as an ATW2 CDF candidate,
4. compactor rewrites the ZIP member,
5. current inflate raw output remains bit-identical.

This does not prove contest score movement because the archive is a tiny CPU smoke artifact, not a full 600-pair contest candidate. The concrete next frontier action is to run the same compactor against the first full ATW2 candidate archive, or route the removal-paradigm scanner to another substrate with parser-visible, decode-opaque sections.
