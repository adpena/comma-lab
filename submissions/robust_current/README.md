# robust_current submission

An honest patched-world submission lane built around stock AV1 compression plus a tiny learned inflate-side post-filter.

## current promoted floor

- `522x392 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / sharpness=1 / learned post-filter`
- `current_workflow`: `2.05`
- local `rule_faithful` estimate: `2.078`
- current_workflow bytes: `861,986`
- rule_faithful bytes: `896,432`

## design

- standard codec path using SVT-AV1
- no GPU requirement in the promoted inflate path
- decode/inflate path uses a tiny shipped int8 post-filter after upscale
- installed runtime payload is explicit:
  - `archive.zip`
  - `inflate.sh`
  - `inflate.py`
  - `inflate_postfilter.py`
  - `inflate_grain_mask.py`
  - `postfilter_int8.pt`
  - `config.env`
  - `analyze_roi.py`

## current findings

- the learned post-filter is the first decode-side lane in this repo to beat the `2.08` sharpness-only floor
- grain-mask synthesis recovered much of the `film-grain=0` catastrophe but still only scored `2.30`
- broad preprocessing remains a losing family because PoseNet is too sensitive
- encoder-side sharpness and tiny task-aware decode correction are currently the highest-signal combination

## pre-scorer smoke gate

Run before relying on a candidate:

```bash
python3 -m src.comma_lab.cli smoke-submission robust_current --package
```

This checks raw output existence, file cardinality, exact frame count, exact geometry-derived byte size, and sampled semantic sanity before a full scorer run.

## preserved comparison configs

- `config.env` — live promoted floor
- `config.av1-2.05-postfilter.env` — named current AV1 snapshot
- `config.av1-2.12.env` — previous honest floor
- `config.av1-2.18.env` — previous AV1 floor
- `config.x265-3.25.env` — preserved x265 floor
