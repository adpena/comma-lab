# PR102 Zero-Byte Inference Tuning Custody Unblock - 2026-05-07

## Classification

Custody unblocked. The 2026-05-05 auto-intake archive for PR102 was wrong: it
downloaded EthanYangTW's stale `qpose14-r55-segactions-minp` release asset
instead of the HNeRV LC-v2 archive associated with PR102.

Correct PR102 archive:

- PR: https://github.com/commaai/comma_video_compression_challenge/pull/102
- title: `hnerv_lc_v2_scale095_rplus1 submission (0.19538 CPU)`
- author: `EthanYangTW`
- head repo: `EthanYangTW/comma_video_compression_challenge`
- head sha: `1e330ec5633539c48278ce3cc96d2b15ea7a9eac`
- runtime source path:
  `submissions/hnerv_lc_v2_scale095_rplus1/`
- canonical fetched archive URL:
  `https://github.com/user-attachments/files/27369164/archive.zip`
- archive bytes: `178981`
- archive sha256:
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- zip member: `0.bin`
- member bytes: `178873`
- member sha256:
  `3234f0689164cfc95b7ee9f9cdf38ecf4d082cfb7048058e2b3ff0f54f864e43`

## Evidence

Fresh custody root:

- `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/`
- standard intake:
  `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/`
- archive provenance:
  `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/archive_provenance.json`
- release identity check:
  `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/release_identity_checks/archive_identity_check.json`
- HNeRV payload profile:
  `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/hnerv_payload_profile.json`

The identity check shows the correct PR102 comment attachment is byte-identical
to both:

- EthanYangTW release `v2-hnerv-lc-scale095/archive.zip`
- BradyMeighan release `hnerv-lc-v2-archive/archive.zip`, the URL hard-coded
  in PR102 `compress.sh`

All three have SHA
`afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`.

The disputed old auto-intake path
`experiments/results/public_pr_archive_release_view/public_pr102_intake_20260505_auto/archive.zip`
has SHA `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`,
bytes `276481`, and a single member `p`; this matches the stale qpose release
asset and is not PR102 HNeRV custody.

## Source-Path Note

The PR body says the archive is included under
`submissions/hnerv_lc_v2_scale095_rplus1/archive.zip`, but the PR head tree at
`1e330ec5633539c48278ce3cc96d2b15ea7a9eac` contains only the seven runtime
source files under that directory:

- `README.md`
- `compress.sh`
- `hnerv_model.py`
- `inflate.py`
- `inflate.sh`
- `schema.py`
- `sidecar.py`

No `archive.zip` blob exists in the PR head tree. The maintainer comment
attachment is therefore the exact archive source for custody; the PR source
path above is the exact runtime source path.

## Tooling Fix

Patched `tools/fetch_all_public_pr_archives.py` so future auto-intake:

- discovers PR comment attachment archive URLs;
- considers PR body archive path hints before broad release sweeps;
- ranks release assets by PR/name/body relevance;
- handles relative `--output-dir` display paths without crashing.

Focused regression coverage lives in
`src/tac/tests/test_fetch_all_public_pr_archives.py`.

## Commands

- `.venv/bin/python -m pytest src/tac/tests/test_fetch_all_public_pr_archives.py`
  - result: `4 passed`
- `.venv/bin/python tools/fetch_all_public_pr_archives.py --output-dir experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex --only-prs 102 --max-prs 1`
  - result: complete; archive downloaded; source present; no manual triage
- `.venv/bin/python experiments/profile_hnerv_frontier_payloads.py .../archive.zip --json-out .../hnerv_payload_profile.json --md-out .../hnerv_payload_profile.md`
  - result: single-member HNeRV payload profile emitted
- `.venv/bin/python tools/audit_archive.py .../archive.zip --strict`
  - result: failed legacy strict audit because it expects
    `renderer.bin`, `masks.mkv`, and `optimized_poses.pt`; not a PR102
    blocker because PR102 is a public single-member HNeRV archive.

## Exact Next Blockers

- No custody blocker remains for identifying the PR102 archive and runtime
  source.
- No A/A++ score claim is made here. Next score-grade step is exact local CUDA
  replay of `archive.zip -> inflate.sh -> upstream/evaluate.py` using the PR102
  runtime source at head SHA `1e330ec5633539c48278ce3cc96d2b15ea7a9eac`, then
  classify the public CPU/CUDA divergence recorded on the PR.
