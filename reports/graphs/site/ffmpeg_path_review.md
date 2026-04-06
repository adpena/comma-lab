# ffmpeg path review

## scope

- `submissions/robust_current/compress.sh`
- `submissions/robust_current/inflate.sh`
- `submissions/robust_current/analyze_roi.py`

## standard

Senior engineer / ffmpeg-contributor-grade challenge review focused on evaluator-facing correctness, corner cases, and silent failure risk.

## findings

### no blocker for the promoted AV1 floor

For the live promoted AV1 path (`524x394 / svtav1-p0 / crf34 / film-grain22 / bicubic / unsharp`), no obvious blocker was found in the encode/inflate path after the explicit `rgb24` fix.

### confirmed critical correctness point

- The AV1 path must emit rawvideo with explicit `-pix_fmt rgb24`.
- Without that, the evaluator can effectively read the wrong byte layout and the score can collapse catastrophically.
- This is the most important confirmed implementation lesson so far.

### medium-risk follow-up items

1. **Colorspace / range audit remains worth doing**
   - The current path relies on ffmpeg filter-chain format control (`format=yuv420p`, `format=rgb24`) rather than explicit matrix/range flags everywhere.
   - No blocker is currently evidenced for the promoted floor, but this is still an audit surface.

2. **ROI path is not symmetry-tested against AV1**
   - ROI metadata / two-pass branches are x265-oriented.
   - That is not a blocker for the live AV1 floor, but it means AV1 + ROI should be treated as a separate correctness problem if reopened.

3. **Branch-specific parity should keep being checked**
   - x265 simple path
   - AV1 simple path
   - ROI branch
   - ROI metadata branch
   These should be treated as separate evaluator-facing pipelines, not assumed equivalent.

### low-risk observations

- even-dimension handling is consistent and intentionally floors odd values
- the public test video names file is used as the packaging source list
- archive packaging and inflate directory handling are straightforward and reproducible
- the new smoke gate passed on the live `2.19` AV1 floor:
  - exact raw file cardinality
  - exact frame count
  - exact geometry-derived byte size

## smoke gate

Use before future promotion decisions:

```bash
python3 -m src.comma_lab.cli smoke-submission robust_current --package
```

This reduces the chance of wasting scorer time on subtle raw-output mismatches.

## conclusion

- **APPROVE for current AV1 canonicalization**
- keep a bug-audit mindset for byte layout, colorspace/range, timing/parity, and branch-path asymmetry on every future promotion candidate
