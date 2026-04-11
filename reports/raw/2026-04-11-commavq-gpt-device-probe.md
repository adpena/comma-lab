# commavq GPT Device Probe — 2026-04-11

## setup

- source: `reports/raw/2026-04-11-commavq-gpt-arithmetic/small/train_split0.bin`
- profile: `gpt_arithmetic_small`
- scorer: `tac lossless gpt-score`
- scored tokens: `64`
- context: `2580`

## results

### cpu / float32

- output: `reports/raw/2026-04-11-commavq-gpt-arithmetic/small/train_split0_gpt_score_probe_64.json`
- bits/token: `9.263728540522601`
- perplexity: `614.695682007668`
- wall time: `4.006473999936134s`

### mps / float32

- output: `reports/raw/2026-04-11-commavq-gpt-arithmetic/small/train_split0_gpt_score_probe_64.json`
- bits/token: `9.263728540522601`
- wall time: `2.3550468330504373s`

## takeaways

- numerical agreement is exact to the reported precision for this `64`-token probe
- on this scorer shape, `mps` is faster than CPU on the real local benchmark
- immediate backend guidance for the official-GPT scorer is now:
  - prefer `mps` on this machine for local scoring
  - keep CPU as the parity/reference backend
