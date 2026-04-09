# lab notebook

Last refreshed: `2026-04-09 10:07:00 -0500`

## executive summary

- current honest Track B floor: **`1.73`**
- current honest Track B bytes: `864,167`
- local rule-faithful estimate: `1.7947470454539947` at `966,071` bytes
- current gap to the public leaderboard snapshot checked at `2026-04-09 06:47:39 -0500`:
  - `0.16` better than first (`1.89`, `neural_inflate`)
  - `0.21` better than second (`1.94`, `roi_v2`)
  - `0.22` better than third (`1.95`, `av1_roi_lanczos_unsharp`)

## technique impact snapshots

### promoted `1.73` floor

- branch: `long1000_h64`
- idea: keep the same long-horizon QAT+EMA regime, scale width to `h64`, and promote the deployed int8 checkpoint that actually survives the official scorer
- result: scorer-backed promotion from `1.84` to `1.73`

### strongest unresolved follow-ons

- bat00 WSL quantization-parity rerun:
  - saved best `3.9258` at epoch `199`
  - latest printed checkpoint: epoch `200`
- strongest packaged local side lanes:
  - `alpha30 h32`: best `3.9355` at epoch `231`
  - `dilated h64`: best `4.0163` at epoch `58`
  - `h96`: best `4.0309` at epoch `66`
- SegNet side lanes:
  - `segnet_attack_fixed_v2`: latest printed epoch `630`, best observed local scorer `0.9819` at epoch `590`
  - `segnet_attack_h64`: latest printed epoch `150`, local scorer `1.1657`
  - blocker: neither run has written a rankable saved artifact yet

## methodology and evidence

- methodology: [../../docs/lab_methodology.md](/Users/adpena/Projects/pact/docs/lab_methodology.md)
- evidence index: [evidence_index.md](/Users/adpena/Projects/pact/reports/graphs/evidence_index.md)
- glossary: [glossary.md](/Users/adpena/Projects/pact/reports/graphs/glossary.md)
- current live status: [../../reports/latest.md](/Users/adpena/Projects/pact/reports/latest.md)
