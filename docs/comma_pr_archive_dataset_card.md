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

_Card last refreshed: 2026-05-11 (companion research artifacts section
added)._

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

## Evidence boundary

This corpus is a public archive and provenance record, not a claim that every
row has been independently replayed by this repository. The intended evidence
uses are:

| Item | Evidence grade | Allowed use |
|---|---|---|
| Public leaderboard score and PR metadata | `external` | Historical context and discovery index |
| Recovered `archive.zip` bytes plus SHA-256 | `empirical` / custody | Byte-exact deconstruction, replay setup, reproducibility checks |
| Local `contest_auth_eval.json` on the exact archive | `A++` / `A` when complete | Ranked score evidence after component recomputation and runtime custody |
| Failed replay, malformed archive, or sidecar dependence | `invalid` / `external_quarantine` | Compliance and methodology lessons only |

Any paper, report, or benchmark table should cite the exact replay artifact
when ranking scores. Public PR prose or rounded leaderboard metadata should
stay labeled as `external`.

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

- This dataset's Hugging Face repository, commit or revision, and any exact
  archive SHA-256 values used in your analysis. No DOI is promised until one
  exists in the release metadata.

## Acknowledgments

- comma.ai for running the contest and the open public-PR submission
  format that made this dataset possible
- All 54 contestants whose scored PRs are included
- AaronLeslie138 (PR #95) for publishing the seminal HNeRV-class
  submission that defined the final winning paradigm
- SajayR (PR #101), rem2 (PR #103), EthanYangTW (PR #102)
  for the medal-winning bolt-ons

## Companion research artifacts (refreshed 2026-05-11)

This section indexes companion artifacts that this repository's research
tooling has produced from the dataset. None of the numbers below are
contest-final scores — every empirical value is tagged with its provenance
axis. Promotion-eligible scores are gated on dual `[contest-CUDA]` AND
`[contest-CPU]` (Linux x86_64) runs of `upstream/evaluate.py` against the
exact archive bytes.

### Empirical anchors (today, public artifact-grounded)

| Anchor | Axis | Score | Provenance |
|---|---|---:|---|
| PR101 grammar paired runtime | `[contest-CUDA T4]` | 0.20662 | exact replay; archive sha checked against this dataset row |
| Residual basis L1 (5 scaffolds) | `[contest-CUDA T4]` | 0.20663 | bolt-ons on PR101 grammar |
| Family paired CPU (5 anchors) | `[contest-CPU GHA Linux x86_64]` | 0.22810 | sister CPU eval on the CUDA-anchored archive set |
| A1 (this repo's apogee submission) | `[contest-CPU GHA Linux x86_64]` | 0.19284 | dual-eval pair landed 2026-05-09 |
| A1 (this repo's apogee submission) | `[contest-CUDA Tesla T4]` | 0.22635 | dual-eval pair landed 2026-05-09 |

The CPU score for A1 rounds to 0.19, matching the gold-tier display band
of PR #101 (0.193 CPU).

### Substrate composition matrix v1

The repository's analysis tooling now classifies every public-PR substrate
plus 16 in-house substrate scaffolds against an 8-class composability
taxonomy (orthogonal / redundant / antagonistic / replacement /
stackable_serial / stackable_parallel / stackable_cascade / incompatible).
The matrix surfaces format-ID collisions, Pareto-dominated rows, and
expected composition coefficients for stacked dispatches.

The classifier output is serialized to a typed JSON schema
(`tac_substrate_composition_matrix_v1`) and consumed by the cathedral
autopilot's dispatch ranker (`tac_autopilot_dispatch_ranking_v1`). Both
schemas are committed to the analysis tooling and are regenerated on
demand from the in-repo lane registry.

### Theoretical floor v2 (refreshed 2026-05-11)

| Quantity | v2 (2026-05-09) | Refreshed (2026-05-11) |
|---|---:|---:|
| median estimate | 0.140 | 0.13867 |
| CI95 | [0.128, 0.152] | [0.12847, 0.14887] |
| std | 0.012 | 0.0052 |

The refresh integrates a Hinton-distilled scorer-saliency surrogate with
24 in-repo substrate priors plus a 19-primitive packet-compiler savings
cap. Three substrates predict per-class floors below v2 (`hessian_block_fp`
≈ 0.133, `mdl_fp4_tto` ≈ 0.134, `scpp_substrate` ≈ 0.135) — the
self-compression family dominates the predicted low end because it shifts
the rate-distortion curve at the parameter level rather than the
representation level.

### Pose-axis dominance at the PR106 frontier

At PR106's operating point (pose_avg ≈ 3.4e-5) the marginal-value-per-byte
of pose-axis improvements is approximately 2.71x SegNet's. This flips the
older 77x SegNet-greater-than-pose heuristic that was true at the 1.x
operating point. Operationally, dispatches targeting the pose axis rank
above SegNet-axis dispatches at the current frontier — but every numeric
prediction must still be confirmed by exact CUDA + exact CPU eval before
any contest claim.

### Provenance tags

Every numeric in this section carries one of:

- `[contest-CUDA <hardware>]` — exact `upstream/evaluate.py --device cuda`
  on the archive bytes, on contest-compliant NVIDIA hardware.
- `[contest-CPU GHA Linux x86_64]` — exact `upstream/evaluate.py --device cpu`
  on the contest's GitHub Actions CI runner family.
- `[predicted; <source>]` — derived from a model, surrogate, or
  composition rule; never a measurement.
- `[empirical:<artifact>]` — measurement against a checked-in artifact
  whose bytes are reproducible.

Other axis labels (`[macOS-CPU advisory only]`, `[MPS-PROXY]`,
`[MPS-research-signal]`) appear only in this repository's internal research
ledgers and are NOT promotion-eligible.

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
