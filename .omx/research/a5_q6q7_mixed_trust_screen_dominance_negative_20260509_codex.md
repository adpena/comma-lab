# A5 Mixed q6/q7 Trust Screen Dominance Negative - 2026-05-09

## Verdict

No eval dispatched. Mixed `q6/q7` A5 trust-region schedules are dominated by
the measured q7-all result unless a future implementation changes the local
distortion geometry.

The q7-all packet already scored `0.2026389105740624` on macOS CPU advisory,
about `0.00979` worse than the A1 Linux CPU anchor while saving only `334 B`.
Adding `q6` to a subset of pairs can only reduce bytes and increase or preserve
distortion relative to q7-all. The largest screened byte delta versus q7-all is
`630 B`, worth only about:

```text
25 * 630 / 37,545,489 = 0.00041949 score points
```

That is more than `23x` smaller than the existing q7-all deficit to A1, before
any additional q6 distortion. The measured configuration family is therefore
not worth exact CPU/CUDA spend.

## Screen

Output root: `/tmp/pact_a5_q6q7_screen_20260509`

All rows use `base_q_bits=7`, `low_q_bits=6`, `latent_dim=28`, and the
advisory per-pair marginal manifest:

`experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json`

| marginal source | low fraction | low pairs | archive bytes | archive SHA-256 prefix | q mean | selected Pearson |
|---|---:|---:|---:|---|---:|---:|
| score | `0.05` | `30` | `177823` | `b6eeb9172b63` | `6.950` | `0.302` |
| score | `0.10` | `60` | `177718` | `688f443847e9` | `6.900` | `0.402` |
| score | `0.15` | `90` | `177613` | `2f670caabc31` | `6.850` | `0.474` |
| score | `0.20` | `120` | `177508` | `b370ac54635f` | `6.800` | `0.530` |
| score | `0.30` | `180` | `177298` | `271a0c9436dc` | `6.700` | `0.620` |
| seg | `0.05` | `30` | `177823` | `735cfa9c0061` | `6.950` | `0.202` |
| seg | `0.10` | `60` | `177718` | `f153e3bb40be` | `6.900` | `0.266` |
| seg | `0.15` | `90` | `177613` | `efcbe5e4ee6d` | `6.850` | `0.311` |
| seg | `0.20` | `120` | `177508` | `7436e05e684d` | `6.800` | `0.348` |
| seg | `0.30` | `180` | `177298` | `0ec9026efad0` | `6.700` | `0.407` |
| pose | `0.05` | `30` | `177823` | `3d488475acf5` | `6.950` | `0.287` |
| pose | `0.10` | `60` | `177718` | `57f421af145e` | `6.900` | `0.397` |
| pose | `0.15` | `90` | `177613` | `37b28f59682f` | `6.850` | `0.478` |
| pose | `0.20` | `120` | `177508` | `f8af881d2bf5` | `6.800` | `0.543` |
| pose | `0.30` | `180` | `177298` | `3ff2194b31db` | `6.700` | `0.647` |

## Command

```bash
base=/tmp/pact_a5_q6q7_screen_20260509
rm -rf "$base"
mkdir -p "$base"
for source in score seg pose; do
  for frac in 0.05 0.10 0.15 0.20 0.30; do
    tag="q6q7_${source}_${frac//./p}"
    sched="$base/${tag}.json"
    out="$base/${tag}_packet"
    .venv/bin/python tools/build_a5_score_marginal_qbits_schedule.py \
      --score-marginal-manifest experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json \
      --json-out "$sched" \
      --candidate-id "a5_screen_${tag}_20260509_codex" \
      --base-q-bits 7 \
      --low-q-bits 6 \
      --low-fraction "$frac" \
      --marginal-source "$source" \
      --latent-dim 28 >/dev/null
    .venv/bin/python tools/build_pr101_frame_conditional_runtime_packet.py \
      --q-bits-json "$sched" \
      --recompute-wire-contract-for-q-bits \
      --output-dir "$out" \
      --candidate-id "pr101_a5_screen_${tag}_20260509_codex" \
      --force >/dev/null
    .venv/bin/python - <<'PY' "$sched" "$out/candidate_archive_manifest.json"
import json, sys
sched=json.load(open(sys.argv[1])); manifest=json.load(open(sys.argv[2]))
archive=manifest['candidate_archive']
print(f"{sched['marginal_source']} frac={sched['low_fraction']:.2f} low={sched['low_pair_count']} bytes={archive['bytes']} sha={archive['sha256'][:12]} qmean={sched['q_bits_summary']['mean']:.3f} align_selected={sched['alignment']['q_bits_vs_selected_marginal_pearson']:.3f}")
PY
  done
done
```

## Classification

Measured-config family negative by dominance proof, not a family kill.

Retired scope:

- post-hoc two-level `q6/q7` schedules using only scalar per-pair marginals;
- low fractions up to `30%`;
- exact-eval spend for this geometry unless a future local advisory result
  breaks the monotonic distortion assumption.

## Reactivation Criteria

- A local SegNet-boundary-aware allocation that can improve distortion at the
  same byte count, not merely lower bytes.
- A packet compiler change that removes the 225 B q-bit side-info overhead or
  makes q-level changes local within a frame/pixel/latent channel.
- A training-loop variant where q6/q7 noise is present during optimization, so
  q6 is no longer a post-hoc perturbation of q7-all.
