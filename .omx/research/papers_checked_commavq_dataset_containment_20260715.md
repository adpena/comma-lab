# Datasets-checked: commaai/commavq — contest segment NOT CONTAINED (measured 2026-07-15)

Date: 2026-07-15 · operator hypothesis "the contest video is likely in the Comma VQ dataset" ·
anti-re-research ledger entry (sister of the papers_checked_* family; L55 discipline).

## STORES CONSULTED (first)
MEMORY.md CURRENT-STATE + L55 · memory
`project_contest_source_is_known_comma2k19_rav4_segment_pose_gt_downloadable_20260619.md`
(contest identity = comma2k19 `b0c9d2329ad1606b|2018-07-27--06-03-57/10`, GT pose npz local) ·
DAG tail. No prior commaVQ containment entry.

## What was checked (VERIFIED, $0, ~135 MB bounded transfer, scratch-only)
- HF `commaai/commavq` (rev `795c839a` webdataset + pre-webdataset rev `610fb2854c` ZIPs) +
  `github.com/commaai/commavq` (encode pipeline, VQ-VAE config, compression challenge).
- Full 100,000-member name index extracted via ZIP central-directory range requests (no bulk
  download); 50,220 unique anonymized 32-hex route hashes.
- All ~3,203 segment-index-10 `pose.npy` members range-extracted and speed-profile-correlated
  against our local comma2k19 GT trajectory for the exact contest segment.

## Verdict — NOT CONTAINED (two independent legs; full detail in
`.omx/research/commavq_containment_and_exploit_menu_20260715.md`)
- **Provenance/camera leg (decisive):** commaVQ tokens are comma-three-era **ecamera** (wide road
  cam, fl=567; `transform_img` hard-codes SCALE=567/455; example file `sample_video_ecamera.hevc`)
  cropped to 256×128. The contest clip is 2018 EON (comma2k19) road camera fl=910 — a device
  generation with NO ecamera. comma2k19 is NOT a subset of commaVQ's source corpus.
- **Empirical pose leg (measured):** across all 3,203 index-10 segments: corr max 0.9431 /
  median 0.0275; 0 pass corr≥0.99; 0 pass joint speed-band+corr≥0.95; best in-band candidate
  rejected adversarially (high-frequency detrended corr 0.1008 — wiggles don't align).
  Name-index MD5 test (6 encodings): 0/100,000 matches. Artifacts:
  `.omx/research/commavq_containment_probe_20260715/`.
- verdict_scope: FORMULATION (index-10 slice + corpus provenance); a hypothetical re-cut under a
  different index is untested but mooted by the camera leg.
- Useful residue: comma's own VQ spends ~192 KB raw / ~48 KB entropy-coded per driving minute at
  far-below-scorer quality — a corroborating benchmark that generic-reconstruction token codecs
  do not undercut the task-space witness. The corpus lever that DOES survive is comma2k19
  temporal-neighbor segments (`/9`,`/11` same route, same rig, downloadable) as train-time side
  information — noted, not actioned (d_seg is the binding axis; pose solved).
- Pointer 0.19108 UNMOVED (anti-re-research banking; MEANS).
