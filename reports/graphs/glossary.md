# glossary

Last refreshed: `2026-04-08 18:38:04 -0500`

## core terms

- `current_workflow`
  - what the published workflow counts right now

- `rule_faithful`
  - stricter local estimate that charges the shipped runtime payload under test

- `promotion`
  - a branch that cleared packaging, smoke, and authoritative scoring

- `close miss`
  - a branch that produced real evidence but did not beat the floor

- `faithful proxy`
  - local ranking path that drives the official upstream evaluator end-to-end instead of a homemade scorer loop

- `authoritative scorer`
  - the local CPU scorer path used for real promotion claims on the current non-GPU shipped lane

## model / training terms

- `QAT`
  - quantization-aware training; train with fake quantization so the deployed int8 artifact matches training more closely

- `EMA`
  - exponential moving average of weights used for stabler checkpoint selection

- `saliency`
  - a map of which pixels matter more to the scorer models

- `per-channel quantization`
  - use one scale per output channel instead of one scale per tensor for conv weights

## operations terms

- `side lane`
  - non-authoritative remote or proxy work used for throughput, ranking, or debugging

- `artifact contract`
  - the rule that serious branches must end in a saved deployed artifact, not just a claim or fp32 checkpoint

- `remote manifest`
  - timestamped JSON record of a remote job’s host, command, root, and log path
