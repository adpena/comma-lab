# ffmpeg path review

## scope

- `submissions/robust_current/compress.sh`
- `submissions/robust_current/inflate.sh`
- `submissions/robust_current/analyze_roi.py`

## standard

Senior engineer / ffmpeg-contributor-grade challenge review focused on evaluator-facing correctness, corner cases, and silent failure risk.

## findings

### no blocker for the promoted AV1 floor

For the live promoted AV1 path (`524x394 / svtav1-p0 / crf34 / film-grain22 / lanczos / unsharp` plus explicit `tv/bt709` encode tags and explicit `rgb24(pc)` decode), no obvious blocker was found in the encode/inflate path after the hardening pass.

### confirmed critical correctness point

- The AV1 path must emit rawvideo with explicit `-pix_fmt rgb24`.
- Without that, the evaluator can effectively read the wrong byte layout and the score can collapse catastrophically.
- This is the most important confirmed implementation lesson so far.

### fixed in the comprehensive rigor pass

1. **ROI branches now fail fast instead of silently mixing codec assumptions**
   - AV1 + ROI is now explicitly rejected because the ROI path is still x265-only.

2. **Metadata ROI analysis now honors the configured ffmpeg and ffprobe binaries**
   - This removes a hidden path-bypass bug for wrapped or pinned toolchains.

3. **`ROI_X_FRAC` now actually affects static ROI placement**
   - The previous centered calculation made this knob misleading.

4. **`INFLATE_POSTFILTER` now applies on ROI inflate paths too**
   - ROI and non-ROI decode paths are now more comparable.

5. **Static inflate dimensions are now explicit config values, not scattered literals**
   - `SOURCE_W` / `SOURCE_H` default to the current corpus dimensions.

6. **Flat-path colorspace/range handling is now explicit**
   - the encoded AV1 stream is tagged `tv / bt709 / bt709 / bt709`
   - the inflator explicitly converts that tagged stream to `rgb24(pc)`
   - smoke reference extraction now uses the same explicit interpretation instead of implicit defaults

7. **Packaging now survives invalid `TMPDIR`**
   - the archive scratch directory now falls back to `/tmp` if `TMPDIR` points at a missing path

### medium-risk follow-up items

1. **ROI path is not symmetry-tested against AV1**
   - ROI metadata / two-pass branches are x265-oriented.
   - That is not a blocker for the live AV1 floor, but it means AV1 + ROI should be treated as a separate correctness problem if reopened.

2. **Branch-specific parity should keep being checked**
   - x265 simple path
   - AV1 simple path
   - ROI branch
   - ROI metadata branch
   These should be treated as separate evaluator-facing pipelines, not assumed equivalent.

### low-risk observations

- even-dimension handling is consistent and intentionally floors odd values
- the public test video names file is used as the packaging source list
- archive packaging now cleans transient scratch instead of leaving `archive/` behind
- the new smoke gate passed on the live `2.12` AV1 floor:
  - exact raw file cardinality
  - exact frame count
  - exact geometry-derived byte size
  - sampled semantic sanity

## smoke gate

Use before future promotion decisions:

```bash
python3 -m src.comma_lab.cli smoke-submission robust_current --package
```

This reduces the chance of wasting scorer time on subtle raw-output mismatches.

## conclusion

- **APPROVE for current AV1 canonicalization**
- keep a bug-audit mindset for byte layout, colorspace/range, timing/parity, and branch-path asymmetry on every future promotion candidate
