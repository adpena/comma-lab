---
license: mit
task_categories:
  - image-to-image
  - other
tags:
  - video-compression
  - neural-codec
  - rate-distortion
  - hnerv
  - comma-ai
  - benchmark
  - reproducibility
size_categories:
  - n<1K
pretty_name: comma video compression challenge - PR archive corpus
---

# comma video compression challenge - PR archive corpus

This dataset captures every scored Pull Request submitted to
[commaai/comma_video_compression_challenge](https://github.com/commaai/comma_video_compression_challenge),
the public 2026 contest to compress comma's `0.mkv` reference dashcam video
under perceptual + temporal scorer constraints.

For each scored PR we publish:

- **`archive.zip`** - the exact compressed-archive bytes that were scored
  by the contest evaluation pipeline (sha256-pinned, byte-identical to what
  upstream `evaluate.py` would receive).
- **`pr_metadata.json`** - leaderboard score, name, author, GitHub
  timestamps, head SHA, head repo, additions/deletions/changed-files counts.
- **`pr_body.md`** - full PR description as authored by the contestant.
- **`archive_provenance.json`** - discovery method (PR body, head_repo
  release, comma release, in-tree LFS), per-attempt log, sha256, size.
- **`source/`** (where the contestant's fork was downloadable as a shallow
  clone at the head SHA) - a filtered source mirror of contest-runtime code.
  The mirror intentionally omits repeated fixed contest assets and local
  checkout state.
- **`INTAKE_LOG.md`** - per-PR fetch log (what succeeded, what failed,
  what's missing).
- **`OMITTED_SHARED_ASSETS.json`** - manifest of files omitted from the
  upload view, with reason codes and byte counts.

## Why this exists

The contest closed 2026-05-04 12:00 UTC after a 4-hour 8-minute race
window from PR #95 (HNeRV root) publishing through the final top-3 lock-in.
Reverse-engineering, methodological audits, post-contest research, and any
paper figure comparing approaches require **byte-exact** access to the
scored archives - not just numeric scores. Several PRs link to release
asset URLs that may rot or be deleted; this dataset preserves the
contest-CUDA-evaluated bytes.

The dataset is primarily maintained for the community, continued iteration,
and the historical record of the challenge. It gives researchers and future
contestants byte-exact archives, PR metadata, and provenance without requiring
them to reconstruct a fast-moving public leaderboard after the fact.

## Final top 3 (for context)

| Rank | PR | Author | Score | LOC | Files | Submitted (UTC) |
|---:|---|---|---:|---:|---:|---|
| 1 | #101 | SajayR | 0.193 | 660 | 5 | 2026-05-04 11:50:13 |
| 2 | #103 | rem2 | 0.195 | **241** | **2** | 2026-05-04 11:55:56 |
| 3 | #102 | EthanYangTW | 0.195 | 367 | 7 | 2026-05-04 11:54:32 |

All three medalists are bolt-ons on top of PR #95 (`hnerv_muon`,
AaronLeslie138, score 0.20, published 2026-05-04 07:47:15 UTC).

## Fetcher methodology

The raw custody corpus was built by `tools/fetch_all_public_pr_archives.py`,
which walks the live leaderboard and for each scored PR attempts archive
recovery via four passes in priority order:

1. Direct `.zip` URL extracted from the PR body
2. GitHub release assets on the contestant's `head_repo`
3. GitHub release assets on `commaai/comma_video_compression_challenge`
4. In-tree `archive.zip` LFS pointer at the PR's head SHA

The release process follows the repository policy against silent-skip
cascades: every download attempt is logged with an explicit reason code in
`archive_provenance.json`. PRs with no recoverable archive are flagged in
`FETCH_SUMMARY.json` under `needs_manual_triage` for manual review.

## Canonical release view and deduplication

This dataset is the **canonical release view**, not the raw forensic intake
directory. It is materialized by
`tools/materialize_pr_archive_release_view.py`, which keeps all byte-exact
scored archives and provenance but omits files that are duplicated fixed
contest assets or reconstructable checkout state:

- Git metadata and Git LFS object stores (`source/.git/**`)
- Python caches (`__pycache__`, `*.pyc`)
- Hugging Face upload cache metadata (`.cache/**`)
- The fixed contest input video (`source/videos/0.mkv`)
- Fixed evaluator weights (`source/models/posenet.safetensors`,
  `source/models/segnet.safetensors`)
- Vendored ffmpeg/SVT-AV1 binaries such as `ffmpeg-new` and
  `libSvtAv1Enc.so*`
- Any other unexpectedly large non-archive file under `source/`, pending
  manual review

The omitted files are not score submissions; the scored bytes live in
`archive.zip`. The omitted fixed assets can be reconstructed from the upstream
contest repository and the recorded PR `head_repo`/`head_sha` metadata. The
complete omission ledger is stored in `OMITTED_SHARED_ASSETS.json`.

This keeps the dataset useful for research and reproducibility while avoiding
N copies of the same 37.5 MB input video, PoseNet/SegNet weights, git object
stores, and local runtime binaries.

The materializers also run static public-link hygiene on the exact publish
surface. Private repo links, local operator paths, provider job URLs, cache
metadata, and fixed contest assets are either omitted or replaced with
placeholders before upload; the manifests record `public_link_count` and
`public_link_violation_count`.

## Consumer API

```python
from datasets import load_dataset

ds = load_dataset("adpena/comma_video_compression_challenge_pr_archive")
# Each row has the full PR metadata + archive bytes
for row in ds["train"]:
    print(row["pr_number"], row["score"], row["name"])
    archive_bytes = row["archive_zip"]  # raw bytes, sha256-pinned
```

For just the metadata (small, no LFS bandwidth):

```python
import json, urllib.request
url = "https://huggingface.co/datasets/adpena/comma_video_compression_challenge_pr_archive/resolve/main/FETCH_SUMMARY.json"
summary = json.loads(urllib.request.urlopen(url).read())
print(f"{summary['n_with_archive']} of {summary['total_attempted']} PRs have archives")
```

## License

MIT - same as the upstream contest repo and the
[`tac` library](https://github.com/adpena/tac) the analysis tooling lives in.

## Citation

If you use this dataset in published work, please cite:

- This dataset (Hugging Face dataset DOI to be assigned on first release)

## Acknowledgments

- comma.ai for running the contest and the open public-PR submission
  format that made this dataset possible
- All 54 contestants whose scored PRs are included
- AaronLeslie138 (PR #95) for publishing the seminal HNeRV-class
  submission that defined the final winning paradigm
- SajayR (PR #101), rem2 (PR #103), EthanYangTW (PR #102)
  for the medal-winning bolt-ons

## Mirrors

Hugging Face is the canonical host because the corpus is file/provenance heavy
and benefits from resumable large-folder upload plus Git/LFS-backed history.
Kaggle is a good secondary mirror for discoverability and notebooks after the
canonical release view is finalized. A Kaggle mirror should upload the same
deduplicated release view plus a `dataset-metadata.json`, not the raw forensic
intake tree.

The canonical mirror tooling is:

- `tools/materialize_pr_archive_release_view.py` for the HF release view.
- `tools/upload_pr_archive_to_hf.sh` for canonical upload.
- `tools/materialize_pr_archive_kaggle_mirror.py` for the Kaggle mirror view.
- `tools/upload_pr_archive_to_kaggle.sh` for Kaggle create/version operations.
