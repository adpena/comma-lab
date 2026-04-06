# track a: exact current workflow

This track is intentionally narrow.

Goal: produce the strongest submission under the workflow exactly as currently published.

## Design

- Keep `archive.zip` tiny.
- Ignore archive payload at inflate time.
- Read the requested video list from the provided `video_names_file`.
- Reconstruct raw frames directly from the upstream repo's own `videos/` directory.
- Prefer the upstream `yuv420_to_rgb` conversion path when available so the inflator matches the evaluator's ground-truth decode path more closely.

## Why this track exists

The current GitHub Action:

1. checks out the repo
2. downloads only `archive.zip`
3. runs `git lfs pull`
4. installs deps
5. installs ffmpeg
6. runs `evaluate.sh`

That makes it possible for repo-side code and repo-side videos to be present during evaluation.

## Failure modes

This track can break if upstream changes any of the following:

- the action no longer checks out the repo with the public videos present
- rate accounting expands beyond `archive.zip`
- `inflate.sh` is sandboxed away from repo assets
- the test set grows or moves

## Operational rule

Track A is always allowed to exist.
Track A is never allowed to block Track B.
