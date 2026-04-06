# submission packet

## score target

- Best honest Track B **current_workflow** score: **`2.12`**
- Best honest Track B **current_workflow** bytes: `864,486`
- Promoted run id: `robust_current-av1-524x394-colorspace-hardening-promoted-cpu-2026-04-06`

## evidence

- scorer-backed candidate beat the prior floor
- canonical default-config regression matched it
- smoke gate passed
- written promotion review exists
- current_workflow vs rule_faithful separation is explicit
- Track A remains the only intentionally non-rule-faithful lane

## path summary

- x265 honest floor reached `3.25`
- AV1 byte-layout bug caused a false `97.45`
- repaired AV1 path reached `2.20`
- `crf34` reached `2.19`
- `bicubic -> lanczos` upscale reached **`2.18`** at unchanged bytes
- explicit `tv/bt709` encode tagging + explicit `rgb24(pc)` decode reached **`2.12`**
