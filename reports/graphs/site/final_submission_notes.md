# final submission notes

## submission posture

- submit Track B only
- cite `1.73` as the best honest **current_workflow** score and `1.795` at `966,071` bytes as the local **rule_faithful** estimate
- frame Track A as transparency-only, never as a submission candidate
- emphasize that every honest promotion is scorer-backed and review-gated

## milestone sequence

- `4.06` - honest Track B baseline
- `3.25` - best x265 floor
- `2.20` - repaired AV1 path became competitive again
- `2.12` - colorspace hardening reduced evaluator mismatch
- `2.08` - encoder-side `sharpness=1` became the clean AV1 floor
- `2.05` - tiny learned int8 post-filter became the new honest floor
- `2.01` - saliency-weighted learned int8 post-filter became the new honest floor
- `1.99` - long-500 QAT+EMA learned int8 post-filter became the new honest floor
- `1.95` - long-500 QAT+EMA h32 learned int8 post-filter became the new honest floor
- `1.92` - long1000 QAT+EMA h16 learned int8 post-filter became the new honest floor
- `1.85` - long1000 QAT+EMA h32 learned int8 post-filter became the new honest floor
- `1.84` - weighted ensemble of the `1.85` floor and the best Monte Carlo refinement became the new honest floor
- `1.73` - long1000 QAT+EMA h64 learned int8 post-filter became the new honest floor

## concise submission framing

This submission should be framed as the result of a measured progression, not a single trick. The repo moved from baseline compression improvements, through AV1 path repair and disciplined one-axis tuning, to a tiny shipped learned post-filter, then to a saliency-weighted version that reached `2.01`, then to a long-horizon QAT+EMA h16 run that reached `1.99`, then to an h32 follow-on in that same regime that reached `1.95`, then to a 1000-epoch h16 extension that reached `1.92`, then to a 1000-epoch h32 extension that reached `1.85`, then to a bounded weighted ensemble that reached `1.84`, and finally to the `h64` long-horizon QAT+EMA branch that reached `1.73`.

## current non-promoted research follow-ons

- The strongest non-promoted family is still bounded Monte Carlo / layer-scale search:
  - first transferred artifact: official proxy `1.93`
  - tighter refinements: official proxy `1.86`
- The SegNet-native branch transferred honestly but weaker, at `1.90`.
- The FiLM-conditioned smoke is now cut: its best saved checkpoint only reached local score `4.0287`.
