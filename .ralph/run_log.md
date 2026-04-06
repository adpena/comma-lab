# run log

## 2026-04-06 — AV1 Lanczos-upscale probe

- Candidate: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / unsharp=0.35`
- Hypothesis: swapping bicubic for lanczos on the upscale axis might improve task fidelity at identical bytes.
- Estimate before run: `2.16`–`2.19`.
- Smoke gate: PASS.
- Result: **`2.18`** at `864,455` bytes.
- Reflection: hypothesis held; bytes stayed flat while both pose and seg improved slightly.
- Decision: promote Lanczos-upscale as the new Track B floor.

## 2026-04-06 — comprehensive bug / rigor pass

- Scope: package/install/eval contract, rule-faithful accounting, ROI path correctness, repo hygiene.
- Fixed:
  - upstream-root packaging
  - package-without-sync rejection
  - stale `inflated/` cleanup before eval
  - installed-payload rule-faithful accounting
  - AV1+ROI fail-fast guard
  - `FFMPEG_BIN` / `FFPROBE_BIN` propagation in ROI metadata analysis
  - live `ROI_X_FRAC`
  - ROI-side `INFLATE_POSTFILTER`
  - root `.gitignore` + transient scratch cleanup
- Verified:
  - ROI guard fails deliberately on AV1
  - wrapper audit logs both `ffprobe` and `ffmpeg`
  - `ROI_X_FRAC` changes the produced ROI artifact
  - ROI inflate output changes under `INFLATE_POSTFILTER=hflip`
  - smoke gate passed on the historical 2.18 floor
- Note: git history could not be cleaned because this workspace is not a git repository.

## 2026-04-06 — colorspace/range hardening promotion

- Candidate: same live AV1 floor with explicit `tv / bt709 / bt709 / bt709` encode tags and explicit `rgb24(pc)` decode conversion.
- Hypothesis: explicit color handling would reduce evaluator mismatch on the flat path.
- Smoke gate: PASS.
- Encoded ffprobe tags: PASS.
- Result: **`2.12`** at `864,486` bytes.
- Reflection: bytes moved by only `+31`, SegNet worsened slightly, but PoseNet improved sharply enough to win by `0.06`.
- Decision: promote the hardened explicit-color path as the new Track B floor.

## 2026-04-06 — speculative lane capture

- Recorded the AV1 + ROI parity lane as explicitly speculative only.
- Required plan captured in durable state:
  1. codec-agnostic ROI encode abstraction
  2. AV1 params for base/ROI/ROI2 streams
  3. matching AV1-aware metadata ROI path
  4. matching inflate/smoke/scorer parity checks
  5. fresh scorer-backed evidence that it actually helps
- Rule: do not promote until scorer evidence justifies it.

## 2026-04-06 — writeup system / frontend pass

- Added generated artifacts for reproducibility and reuse:
  - `reports/graphs/build_experiment_manifest.py`
  - `reports/graphs/build_code_callouts.py`
  - `reports/graphs/build_comparison_media.py`
  - `reports/graphs/refresh_site.py`
  - `docs/repro_checklist.md`
  - `just rebuild-site`
- Added site-level context so the landing page now states the contest, the repo identity, the GitHub source, and when the page was last rebuilt.
- Added browser-preview comparison media with synced play/pause + scrubber and crop zoom.
- Added posters for the comparison videos and horizontal-scroll handling for the dense frontier table on mobile.
- Verified with local desktop and iPhone screenshots generated via Playwright.

## 2026-04-06 — player / scatter coherence pass

- Reworked the comparison-player state model so mode switches preserve playhead and pause hidden videos.
- Replaced the brittle 2.18→2.12 SVG metric rows with semantic HTML layout.
- Added published-baseline comparison to the summary strip.
- Flipped the bug detour downward in the lineage graph so failures read as failures.
- Added a focused operating-range scatter view plus lighter markers and hover/focus details.
- Verified via Chrome DevTools browser automation and local desktop/iPhone screenshots.
