# C091 Mask/Packer Big-Move Screen - 2026-05-03

Scope: local-only mask/packer byte screen around C091/PR75/PR77/C089. No remote
GPU dispatch was performed and `.omx/state` was not edited.

## Anchor

- Frontier: C091 PR75 public replay exact T4.
- Artifact: `experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/contest_auth_eval.adjudicated.json`.
- Score: `0.31516575028285976`.
- Bytes/SHA-256: `276481`,
  `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`.
- Unchanged-component rate-only gap to `<0.314`: `1751` bytes.

## Tool And Artifacts

- Tool: `experiments/plan_c091_mask_packer_bigmove.py`.
- Result root: `experiments/results/c091_mask_packer_bigmove_20260503_codex/`.
- Primary artifact: `plan.json`.
- Derived artifacts: `candidate_matrix.json`, `mask_lossless_probe.json`,
  `lossy_transcode_probe.json`.
- Evidence grade: `empirical_deterministic_byte_screen`.
- Score claim: `false`.

## Findings

Exact-lossless mask stream:

- Current charged `masks.mkv` Brotli segment: `219472` bytes,
  SHA-256 `d1ae0d39c848e5715b74bb0122269066a4a1ab60ba9e12f34e70fd62ac136d87`.
- Best focused exact-lossless Brotli probe: `219465` bytes,
  SHA-256 `1ddf03dad8466797397ead26a0d481766e22e26cbc936251c1e3ed5d2f6eda99`,
  params `{quality=11, mode=0, lgwin=19, lgblock=17}`.
- Savings: `7` bytes. This is real but cannot create a big move. The public
  fixed-slice parser assumes a `219472`-byte mask segment; the shorter stream
  is only usable in self-describing P3/P6 payloads where existing header
  overhead dominates the fixed-slice comparison.

Lossy mask-transcode sample:

- Probe: first `60` frames via local `aomenc` OBU transcodes.
- Byte-saving sample rows were not geometry-safe:
  - CQ63: `6924` bytes for first 60 frames, but `64245` class flips.
  - CQ50: `22305` bytes for first 60 frames, but `11516` class flips.
- The only class-parity row in the sample was CQ0: `199275` bytes for first
  60 frames, `0` class flips, but `+176742` bytes versus the source first-60
  OBU slice. This is not a byte candidate.

Candidate matrix:

- Best existing byte-screen row ingested:
  `public_renderer_c089_p6_lossless_stream_resweep`, `276124` bytes,
  SHA-256 `337b04040bee1316375bae8b2cfc2f08acddc235e4ee02d37ecfd62c0c831d95`.
  It still needs `1394` byte-equivalent component improvement to reach
  `<0.314` and has a renderer-transplant safety gate, so this plan does not
  justify a new exact eval.
- Active row recognized but not touched:
  `pr77_actions_pr75mask_renderer_c089pose_fixedslice`, `276329` bytes,
  SHA-256 `27866172e76d27113e86a30f722588fd668f81a949be3acbe1e92cddc9a6a1d8`.
  The planner marks it `already_active_do_not_touch_or_duplicate` and computes
  `1599` byte-equivalent component improvement still needed for `<0.314`.

## Decision

No dispatchable new mask/packer candidate is justified from this screen. Exact
lossless packer space around the current mask stream is exhausted to tiny-byte
scale, and the local lossy AV1 byte-saving rows are not geometry-safe because
they change decoded mask classes.

Next local command, if deeper mask transcode evidence is needed:

```bash
.venv/bin/python experiments/plan_c091_mask_packer_bigmove.py \
  --frontier-archive experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip \
  --frontier-eval-json experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/contest_auth_eval.adjudicated.json \
  --output-dir experiments/results/c091_mask_packer_bigmove_20260503_codex \
  --lossy-probe-frames 600
```

Remote exact eval should wait for the active PR77 action + C089 pose fixedslice
T4 job. Only consider a follow-up mask-packer exact eval if that active result
lands within the `3-7` byte-rate window of the target or proves a component
positive basin that changes the break-even calculation.
