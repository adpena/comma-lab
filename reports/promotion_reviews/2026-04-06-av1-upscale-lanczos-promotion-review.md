# AV1 Lanczos-upscale promotion review

## candidate

- Run id: `robust_current-av1-524x394-upscale-lanczos-promoted-cpu-2026-04-06`
- Config: `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / unsharp`
- Prior honest floor: `2.19` at `864,455` bytes with bicubic upscale

## hypothesis before the run

- Change only one axis: `UPSCALE_FLAGS bicubic -> lanczos`.
- Expected effect: slightly sharper reconstruction at identical bytes.
- Expected upside: around `0.00` to `0.03` score improvement if pose and seg both improved modestly.

## measured result

- Candidate run: `2.18` at `864,455` bytes
- Canonical default-config regression: `2.18` at `864,455` bytes
- Byte delta vs prior floor: `0` bytes (`0.000%`)
- Pose delta vs prior floor: `-0.00247114`
- Seg delta vs prior floor: `-0.00000886`
- Score delta vs prior floor: `-0.0100`

## smoke gate

- Candidate passed pre-scorer smoke gate:
  - raw file cardinality
  - exact frame count
  - exact geometry-derived byte size
- Evidence: `reports/raw/2026-04-06-av1-upscale-lanczos/robust_current-av1-upscale-lanczos-smoke.json`

## official-formula check

Using the contest formula

`100 * segnet_distortion + 25 * rate + sqrt(10 * posenet_distortion)`

with:
- seg = `0.00569617`
- pose = `0.106568`
- rate = `0.0230242`

gives `2.177539780530782` which is consistent with the reported `2.18` after normal rounding.

## current_workflow vs rule_faithful

- current_workflow: `2.18` at `864,455` bytes
- corrected installed-runtime-payload estimate after the bug pass: `2.196195252141633` at `892472` bytes
- separation remains explicit and preserved

## reflection

The hypothesis held. This is a particularly clean win because the bytes stayed fixed while both distortion terms improved. That suggests the reconstruction kernel itself was slightly better aligned with the evaluator than bicubic at this operating point.

## decision

**PROMOTE**

Reason:
- scorer-backed candidate beat the prior floor
- canonical default-config regression reproduced the win
- smoke gate passed
- packaging views remained explicit
- no new evaluator-path blocker was introduced
