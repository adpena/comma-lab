# challenge snapshot

This starter pack is shaped around the challenge as it was last manually verified on 2026-04-03.

## Facts this repo assumes

- The upstream repo is `commaai/comma_video_compression_challenge`.
- The README states the public input is `./videos/0.mkv`, a 1-minute 37.5 MB dashcam clip.
- The published score is `100 * segnet_distortion + 25 * rate + sqrt(10 * posenet_distortion)`.
- The public test list contains only `0.mkv`.
- The GitHub Action downloads a submission `archive.zip`, checks out the repo, installs `git-lfs`, pulls LFS files, installs dependencies, installs `ffmpeg`, and runs `bash evaluate.sh`.
- `evaluate.sh` unzips `archive.zip`, then calls `inflate.sh`.
- `evaluate.py` computes rate from `archive.zip` only.

## Why the starter pack is dual-track

Because the current evaluator and the written rules are not perfectly aligned, the lab should keep:

- a literal current-workflow submission
- a robust patched-world submission

Do not collapse these into one narrative.

## References

Use the following upstream files as the main source of truth:

- `README.md`
- `.github/workflows/eval.yml`
- `evaluate.sh`
- `evaluate.py`
- `frame_utils.py`
- `modules.py`
- `public_test_video_names.txt`
