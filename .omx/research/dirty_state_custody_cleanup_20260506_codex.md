# Dirty-State Custody Cleanup - 2026-05-06

Scope: custody refresh on `main` at `2adc4f81` after the HNeRV wavelet
residual planning commit. This ledger preserves signal from dirty nested
gitlinks, raw Kaggle ingest state, and orphan-pyc recovery snapshots without
promoting score claims or deleting raw custody.

No score-work implementation files were edited. The active HNeRV wavelet
sidechannel files remain operator-owned WIP and are only dispositioned in the
untracked-source manifest so cleanup audits can distinguish them from orphan
custody debt:

- `src/tac/hnerv_wavelet_sidechannel.py`
- `src/tac/tests/test_hnerv_wavelet_sidechannel.py`
- `tools/build_hnerv_wavelet_sidechannel_candidate.py`

## Nested Public-PR Gitlinks

Eight public-PR intake gitlinks remain dirty because external public source
trees contain local comment markers on `F.kl_div(..., reduction="batchmean")`
calls. These are not archive/runtime payload edits and should remain local
forensic custody unless a future public-intake cleanup resets or re-clones the
detached trees after confirming no unique annotations are needed.

Current nested heads and dirty source paths:

| Gitlink | Head | Dirty paths |
| --- | --- | --- |
| `experiments/results/public_pr100_intake_20260504_codex/source` | `0a8d343` | `submissions/fp4_mask_gen/compress.py`, `submissions/neural_inflate/train_ren.py`, `submissions/ph4ntom_drv/compress.py`, `submissions/quantizr/compress.py`, `submissions/svtav1_dilated_ren/svtav1_dilated_ren_training.py` |
| `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source` | `ec7e366` | same five paths as PR100 |
| `experiments/results/public_pr103_intake_20260504_codex/source` | `d202707` | same five paths as PR100 |
| `experiments/results/public_pr105_kitchen_sink_intake_20260504_codex/source` | `9376a6f` | same five paths as PR100 |
| `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source` | `339910e` | same five paths as PR100 |
| `experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/repo` | `854b397` | same five paths as PR100 |
| `experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/repo` | `4f46081` | `submissions/fp4_mask_gen/compress.py`, `submissions/neural_inflate/train_ren.py`, `submissions/quantizr/compress.py`, `submissions/svtav1_dilated_ren/svtav1_dilated_ren_training.py` |
| `experiments/results/public_pr91_intake_20260504_worker/pr91_src/repo` | `77f958d` | same four paths as PR82 |

Disposition: preserve as local forensic snapshots covered by
`public_pr_intake_gitlinks_forensic` in
`.omx/research/local_custody_release_manifest_20260505_codex.json`. Do not
commit nested gitlink pointer changes from these local annotations.

## Kaggle Raw Ingest `gt_passthrough`

The raw Kaggle ingest gitlink at
`reports/raw/kaggle_ingest/kaggle-dilated-h64-long1000-retry-v6-20260410T234220Z/comma_video_compression_challenge`
is dirty only from an untracked `submissions/gt_passthrough/` probe:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `submissions/gt_passthrough/inflate.py` | `3671` | `f6d80b68265015f91e23da78eeda3438d8c4f25a051316bcf5ceb031cc8b3d1e` |
| `submissions/gt_passthrough/inflate.sh` | `132` | `d2c3b491a52d09241e0660c0893e173b2411f4e87d8a3e880f91cb33c1bca36d` |
| `submissions/gt_passthrough/report_pyav.txt` | `1162` | `e657e39a444574aeff20c3defa5a3ecdf7cdbb783cf38dc01a01541051d40235` |

The probe reconstructs raw frames from repo-side videos, so it is useful as a
local scorer sanity/baseline check only. It is non-compliant for contest
custody, makes no score claim, and must not be used as promotion evidence.
Disposition: preserve under `kaggle_raw_ingest_externalized`; delete only after
a future cleanup records a rebuildable manifest or intentionally discards this
sanity probe.

## Orphan-Pyc Recovery Mirror

Thirty-four tracked files under
`reverse_engineering/orphan_pyc_recovery_20260505_codex/` remain modified. The
inspection found these are pass-2 rehydrations from recovered git blobs or
public raw blob sources replacing pycdc stubs. They are source-like and useful,
but this mirror is still private recovery custody; promotion must happen through
canonical `src/tac`, `experiments`, `tools`, `scripts`, `submissions`, `docs`,
or curated `reverse_engineering` paths with focused tests.

High-signal buckets:

- PR85/QH0 and PR91 replay tools:
  `experiments/build_pr85_qh0_serializer_candidates.py`,
  `experiments/replay_pr91_hpm1_mask.py`.
- Public runtime/adapters and codecs for PR65, PR67, PR85, PR86, PR90, PR95,
  PR96, PR98, and PR99 under `experiments/results/...`.
- Recovery of packet/snapshot/site tooling:
  `scripts/build_contest_submission_packet.py`,
  `scripts/q_faithful_snapshot_loop.py`,
  `reports/graphs/build_public_site_bundle.py`,
  `reports/graphs/test_build_public_site_bundle.py`.
- Regression candidates for already rehydrated codec surfaces:
  `src/tac/tests/test_endgame_archive_decision.py`,
  `src/tac/tests/test_pr85_bundle.py`,
  `src/tac/tests/test_quantizr_torch_fp4_codec.py`,
  `src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py`.
- Submission runtime candidate:
  `submissions/robust_current/apply_qzs3_postprocess.py`.

The release-index split guard documents all 34 modified recovery paths through
`orphan_recovery_snapshots_private`. The next cleanup tranche should not delete
or reset these files. Instead:

1. Promote the PR85/QH0 serializer candidate or retire it against live
   `src/tac/qh0_record_serializer.py` and PR85 bundle tests.
2. Compare the PR91 replay tool against live `src/tac/pr91_hpm1_codec.py` and
   existing PR91 ledgers before deciding whether a canonical CLI is still
   needed.
3. Fold any still-missing public-runtime insight into curated
   `reverse_engineering/` notes, not raw `experiments/results` source copies.
4. Only after canonicalization, stage deletions through
   `tools/audit_orphan_recovery_canonicalization.py`; no deletion is authorized
   by this ledger.

## Audit Evidence

- `.venv/bin/python tools/audit_nested_gitlink_custody.py --repo-root . --strict --local-custody-manifest .omx/research/local_custody_release_manifest_20260505_codex.json --format json`
  passed with `dirty_gitlink_count=9`, `documented_count=9`,
  `warning_count=0`.
- `.venv/bin/python tools/audit_release_index_split.py --repo-root . --strict --local-custody-manifest .omx/research/local_custody_release_manifest_20260505_codex.json --format json`
  passed with `record_count=43`, `documented_count=43`, `blocker_count=0`,
  `warning_count=0`.
- `.venv/bin/python tools/audit_orphan_recovery_canonicalization.py --repo-root . --strict --format json`
  passed with `source_like_delete_count=0`, `unstaged_delete_count=0`,
  `missing_canonical_count=0`.
- Initial untracked-source audit without the disposition manifest failed on
  only the three active HNeRV wavelet WIP files listed above. After updating
  `.omx/research/untracked_source_dispositions_20260505_codex.json`, the
  manifest-backed audit passed with `untracked_source_like_count=1`,
  `dispositioned_count=1`, `undispositioned_count=0`, and
  `invalid_disposition_count=0`; the one current untracked source-like path was
  this ledger before commit.
- `.venv/bin/python tools/audit_reverse_engineering_tree.py --repo-root . --summary`
  passed with `files=716` and `blockers=0`.
- `.venv/bin/python tools/audit_reverse_engineering_tree.py --repo-root . --release-strict --release-manifest .omx/research/reverse_engineering_release_manifest_20260505_codex.json --summary`
  passed with `files=716` and `blockers=0`.
- AST parsing of the 34 modified orphan-pyc recovery Python files passed.
