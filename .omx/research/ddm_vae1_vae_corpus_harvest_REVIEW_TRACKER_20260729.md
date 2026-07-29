---
schema: pact.ddm_vae1_vae_corpus_harvest.review_tracker.v1
utc: 2026-07-29
main_landing_review_required: true
---

# Review tracker — DDM VAE1 corpus harvest

## Scope

Owned code:

- `experiments/ddm_vae1_ar_prior_probe.py`
- `experiments/test_ddm_vae1_ar_prior_probe.py`

Owned evidence:

- `.omx/research/ddm_vae1_ar_prior_probe_20260729/receipt.json`, SHA-256
  `f81b49528413a374883e21b9cb7c17a34621bd6ee753197a231cd3181f293844`
- `.omx/research/ddm_vae1_ar_prior_probe_20260729/resume_replay_receipt.json`,
  SHA-256 `648af5c368c1e1d96bcebf1ec94a7d63e3a74908fb464b2f655f626340015bf0`
- `.omx/research/ddm_vae1_ar_prior_probe_20260729/progress_v2.json`, SHA-256
  `9ccb546df7662459e030e4053deae6aee81b9605f1b93c7c7ead56dfdf67cbaf`
- `.omx/research/ddm_vae1_ar_prior_probe_20260729/artifact_custody.json`, SHA-256
  `37d602009a3b01325a05ba0e6bf376247bfa1b04b7563c2fd9274fae7a38204a`
- immutable stage-001 model and stage-002 frame on the primary SSD tier, with logical
  repo-relative paths and explicit cold-store fallbacks in the progress record
- corpus memo, DAG feed, and papers-checked ledger

External/current dependencies were read-only:
`experiments/ddm_r7_token_coder.py`, `tac.optimization.ddm_tr1_runtime`, and the SSD endpoint
checkpoint.

## Pre-review failure caught and repaired

The first full endpoint execution failed closed during fitted-frequency normalization because a
heavily occupied context could floor a rare symbol to zero. The implementation was changed to
enforce positive frequencies while removing normalization overage only from reducible mass. A
skewed-row regression test was added. The abandoned progress file contains no completed stage and
did not record the exception, so it is deliberately excluded from landing; the regression test is
the durable failure/fix evidence.

## Review pass 1 — implementation and receiver closure

Status: `PASS_AFTER_PRE-REVIEW_FIX`.

Checks:

- Followed the actual encode/decode loop symbol by symbol.
- Verified the context available at decode is only channel, counted temporal mode, and the
  co-located previously decoded residual.
- Verified every fitted `uint16` frequency is serialized inside the counted model section.
- Verified positive frequency rows sum exactly to `2^15`.
- Verified header identity, version, model kind, levels, rank, shape bounds, section lengths, and
  semantic SHA fail closed.
- Verified LZMA and zlib streams require exact length/termination and reject trailing bytes.
- Verified canonical re-encode refits the model and rejects alternate valid-but-inert model bytes.
- Verified frame corruption, truncation, and trailers are covered by focused tests.
- Verified output/progress stay under `.omx/research`; checkpoint input remains read-only.
- Verified stage-001 fitted model and stage-002 complete frame are atomically written, separately
  named, and hash-checked on resume.
- Verified logical stage/progress references remain repo-relative under `.omx/research`; when the
  intentionally gitignored local stage is absent, only an absolute artifact inside the two approved
  Pact SSD roots is accepted, and its recorded bytes/SHA are rechecked before load.
- Re-ran from the completed progress ledger without recreating either stage. All scientific fields
  in the replay receipt are identical to the first receipt after excluding runtime timing and the
  deliberately different output-receipt path.
- Verified the receipt carries reconstructed argv, repo-relative cwd, `PYTHONPATH`, measurement-time
  git HEAD, exact source hashes, and explicit `seed=null` / `rng=none`.
- Verified the runner invokes no scorer/evaluator/training path.

Finding disposition:

- `FIXED`: positive normalization under skewed counts.
- `FIXED`: absolute temporary-worktree stage references replaced by fail-closed repo-relative
  logical references plus approved-SSD cold-store fallbacks; the scientific frame SHA remained
  byte-identical.
- `OPEN-NONBLOCKING`: cross-runtime byte identity is not certified; receipt is correctly
  `[macOS-CPU advisory, rate-only]`, and the candidate loses before production integration.
- `PASS`: no uncounted learned state or receiver-inert model section found.

## Review pass 2 — scientific claim, accounting, and negative scope

Status: `PASS`.

Checks:

- Recomputed component closure:
  `60 + 1,361 + 21,868 + 558,482 = 581,771 B`.
- Confirmed same-object SMEVR is independently encoded, decoded, and semantically identical.
- Recomputed `581,771 - 557,238 = 24,533 B`.
- Recomputed pure rate-score delta:
  `25*24,533/37,545,489 = 0.0163355176969462`.
- Confirmed the receipt labels the row non-promotable, rate-only, and macOS advisory.
- Confirmed the negative says
  `FORMULATION x STATIC_POOLED_MODE_DELTA_PREV1_COUNTED_CONFIG x config_sha256`, with all zero-init,
  smoothing/CDF/traversal/range-state/base-tie/compressor knobs machine-closed, not first-order or
  learned-prior family. The exact config SHA is
  `4f86dd2101c7e6b992f797255917010297fcaab768fbff29f442ca5c8e6ffd62`.
- Confirmed the memo does not call 562,174 B or 586,707 B an archive row.
- Confirmed the local 0.1910828242 contest-CPU custody pointer is distinct from the external
  effective 0.172141 contest-CUDA frontier.
- Confirmed row-7, posterior-collapse, amortization-gap, IWAE-K, and bits-back distinctions against
  current code/object preconditions.
- Confirmed #614 historical object verdicts are retained rather than rewritten.
- Confirmed no task closure, launch authority, or score promotion is implied.

## Verification commands

```text
ruff check experiments/ddm_vae1_ar_prior_probe.py experiments/test_ddm_vae1_ar_prior_probe.py
uv run --no-project --with pytest --with numpy --with brotli \
  python -m pytest -q experiments/test_ddm_vae1_ar_prior_probe.py
uv run --no-project --with pytest --with numpy --with brotli \
  python -m pytest -q experiments/test_ddm_vae1_ar_prior_probe.py \
  experiments/test_ddm_r7_token_coder.py
```

Observed final result: ruff clean; 11 focused tests pass; the combined probe plus inherited R7
dependency suite passes 37 tests.

The full measurement command and exact runtime/source hashes live in the receipt. Its input
checkpoint is on `/Volumes/VertigoDataTier/pact`; no bulky checkpoint was copied into git.
The two ignored stage artifacts were copied byte-identically to
`/Volumes/VertigoDataTier/pact/ddm_vae1_20260729/ar_prior_probe_static_pooled_4f86dd21`, verified,
then the local rebuildable copies were moved to Trash. The custody manifest is committed.

## Independent adversarial reviews

- VQ/collapse/prior review: `PASS_WITH_REVISIONS`; all required revisions were applied. It narrowed
  the negative to the config-hashed formulation, strengthened #417 from named parsing to actual
  decode/application, registered intervention granularity, qualified the FSQ analogy, and added
  counted bits-back support/initial-state preconditions and row prices.
- Ballé/Concrete/code review: final `PASS`. Its findings were fixed before the final receipt:
  repo-relative resumability, exact argv/cwd/git/seed provenance, fixed contest coefficient versus
  optional budget dual, machine formulation config, t0 law, true unused-context inertness coverage,
  bounded model allocation/no unbounded zlib flush, precise V4 schedule evidence, and removal of the
  stale empty failure progress file.
- Post-cleanup custody delta: both independent reviewers returned `PASS`. They verified the logical
  paths remain repo-relative, absent local stages resolve only below the two approved Pact SSD
  roots, symlink escapes are closed, bytes/SHA are rechecked before load, the cold replay is
  scientifically identical, and rule-118 accounting and verdict scope did not change.
- Fresh receipt/source/stage hashes match; the resume replay is scientifically identical; no
  remaining blocker was reported. These reviews do not replace MAIN landing review.

## Required MAIN review

MAIN must independently inspect:

1. the arithmetic coder and canonical re-encode contract;
2. rule-118 accounting of fitted frequencies and exact narrow formulation scope;
3. the 557,238 B same-object SMEVR control;
4. the exact `STATIC_POOLED_MODE_DELTA_PREV1_COUNTED_CONFIG x config_sha256` negative scope;
5. the mechanism corrections in the harvest table;
6. the proposed consumer routing and scorer-slot ownership; and
7. the approved-SSD fallback and custody/cleanup manifest; and
8. the exact files in the serializer commit(s).

This tracker records two worktree review passes. It does not satisfy or waive MAIN's landing review.
