# write-up playbook

Treat the write-up prize as a first-class lane.

## Public Framing

- Frame public docs as a community and historical record plus a reproducible
  engineering artifact.
- Do not imply an arXiv/preprint submission is planned or guaranteed. Use
  "optional long-form note" or "future publication surface" until the human
  explicitly chooses a venue.
- Rank only `A++`/`A` rows. Everything else belongs in roadmap, external
  context, negative-results, or compliance sections.

| Evidence grade | Writeup placement |
|---|---|
| `A++` / `A` | Ranked score tables |
| `A-negative` | Scoped negative results |
| `empirical` | Roadmap and engineering signal |
| `derivation` / `prediction` | Hypothesis or next-wave plan |
| `external` | Public PR/historical context |
| `invalid` | Compliance lesson or quarantine |

## Capture from day one

- score-over-time plots
- archive-bytes vs score scatterplots
- representative frame comparisons
- failure galleries
- patch-sensitivity notes
- ablations

## Strong narrative shape (post-2026-04-29 update)

The original four-act outline (codec → repair → tune → tiny CNN) is now act 1 of a longer story. The full arc is:

1. what the evaluator measures (the scoring formula and SegNet/PoseNet leverage)
2. the codec/post-filter era (the 4.06 → 1.73 arc — story unchanged but no longer the headline)
3. why we abandoned the codec — the renderer paradigm (1.73 → 1.05)
4. compress-time intelligence — pose TTO and KL distill (Lane G v3)
5. the Selfcomp paradigm shift — block-FP self-compression, grayscale-LUT mask, 6-DOF affine duality (live work, only land if measured [contest-CUDA])
6. what the engineering rigor under the hood made possible (78 strict preflight checks, eval_roundtrip non-negotiable, MPS-vs-CUDA drift discovery)
7. negative results: 30+ counts, including the catastrophic 53.61 mask-resolution disaster, Lane GP Runge phenomenon, UNIWARD encoder no-op, etc.

## Minimum visuals to keep

- baseline vs robust vs exact-current
- score breakdown decomposition (the 100·seg + sqrt(10·pose) + 25·rate decomposition)
- one chart for x265 sweep
- one chart for residual ROI ablation
- NEW: leaderboard snapshot (Quantizr 0.33, Selfcomp 0.38, Mask2mask 0.60, Lane G v3 1.05)
- NEW: Lane G v3 component breakdown (PoseNet 0.0034, SegNet 0.0040, rate 0.0185)
- NEW: 78-check preflight diagram (or a simple list grouped by bug class)

## Strategic-secrecy guardrails (non-negotiable)

- Public-facing surfaces: only [contest-CUDA] tagged scores. No proxy, no MPS, no [advisory only].
- Do NOT publicize the Cloudflare site URL until the human explicitly says it is time.
- Do NOT publish the specific Lane W (hard-pair self-compress), Lane Ω (Hessian-aware quantization), or Lane DARTS-S architecture-search recipes on public-facing surfaces.
- Do publish the engineering-rigor story (preflight checks, MPS drift discovery, eval_roundtrip discipline) — that is differentiated and not competitively load-bearing.
- For any external paper, preprint, or public supplement: full disclosure can
  be acceptable, but coordinate with the human before naming a venue, URL, or
  release date.
