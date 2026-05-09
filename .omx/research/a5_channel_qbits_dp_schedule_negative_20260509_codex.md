# A5 Channel Q-Bits DP Schedule Review - 2026-05-09

Generated: `2026-05-09T09:30:00Z`
Owner: `codex`
Score claim: `false`
Promotion eligible: `false`
Family killed: `false`

## Verdict

The A5 per-channel q-bit compiler primitive is now runtime-consumed and
tested, but the first latent-domain MSE schedule is a measured-config negative.

The `qsum=200` schedule cut the packet to `178,014 B` and proved that
`channel_raw3` side-info plus per-channel latent packing can be consumed by
`inflate.sh`. The macOS CPU advisory score was `0.2016517950045961`, worse
than the current A5 compact binary side-info point (`0.20101191630821824`) and
far worse than the A1/PR101 public-axis anchors. This retires only the
`latent_scale_weighted_truncation_mse @ qsum=200` schedule.

The retained value is architectural: future learned or score-domain schedules
now have a byte-closed channel-wise deployment target instead of being trapped
as proxy tables.

## Implemented Contract

Reusable codec support:

- `tac.codec.frame_conditional_bit_budget.pack_frame_conditional_channel_q_bits`
- `tac.codec.frame_conditional_bit_budget.unpack_frame_conditional_channel_q_bits`
- `tac.codec.frame_conditional_bit_budget.pack_frame_conditional_channel_latent_codes`
- `tac.codec.frame_conditional_bit_budget.unpack_frame_conditional_channel_latent_codes`
- `tac.codec.frame_conditional_bit_budget.build_frame_conditional_channel_wire_contract`
- `tools/build_pr101_frame_conditional_runtime_packet.py --q-bits-sideinfo-encoding channel_raw3`
- `tools/build_a5_channel_qbits_schedule.py`

The schedule builder solves the exact separable dynamic program over one q-bit
value per latent dimension for a fixed total qsum. It is deliberately
`score_claim=false`: the objective is latent-domain truncation MSE, not a
SegNet/PoseNet component model.

## CPU-Prep Schedules

`reports/a5_channel_qbits_dp_qsum210_20260509.json`

- Target qsum: `210` versus all-8 qsum `224`
- q-bit split: `14` channels at q7, `14` at q8
- Side-info: `11 B`
- Latent wire payload: `15,750 B`
- Estimated delta versus original PR101 LZMA latent blob: `+486 B`
- Classification: byte-dominated CPU-prep schedule; no runtime packet built

`reports/a5_channel_qbits_dp_qsum200_20260509.json`

- Target qsum: `200` versus all-8 qsum `224`
- q-bit split: `24` channels at q7, `4` at q8
- Side-info: `11 B`
- Side-info SHA-256:
  `0ab65860eb68d676a7c01841b0c2fc7e4f21a66c7deb9f83303abea9ac372790`
- Latent wire payload: `15,000 B`
- Latent wire SHA-256:
  `e41d8c218687d88ba1a25c3d3fb8bff0bd0bb6b3bbe9e143bfba4a08bc7d5d1d`
- Estimated delta versus original PR101 LZMA latent blob: `-264 B`
- Classification: byte-positive CPU-prep schedule; runtime packet built and
  advisory evaluated

## Runtime Packet

- Candidate id: `pr101_a5_channel_qbits_dp_qsum200`
- Packet archive:
  `experiments/results/pr101_a5_channel_qbits_dp_qsum200_20260509_codex/packet/archive.zip`
- Archive bytes: `178,014`
- Archive SHA-256:
  `efc0466bc38edb9cb57193d5f43e4d4fbf2d993cd41069f2c3700aa4bedbfeae`
- Member `x` bytes: `177,914`
- Member `x` SHA-256:
  `a6a34e66a280eae053fb481b3007d2fc7d1fa17defe43f6163b0ecfc5f528106`
- Runtime tree SHA-256:
  `f4c262b2fe134771a232a5e452d4f73d3236cf1b326a34351972b8d4dc857533`
- Runtime consumption proof:
  `experiments/results/pr101_a5_channel_qbits_dp_qsum200_20260509_codex/runtime_consumption_proof.json`
- Packet closure:
  `runtime_consumes_changed_archive_bytes=true`
- Wire contract:
  `tac_frame_conditional_latent_wire.v1`, `channel_raw3`

## macOS CPU Advisory

Evidence grade: `[macOS-CPU advisory negative]`

- Auth eval JSON:
  `experiments/results/pr101_a5_channel_qbits_dp_qsum200_macos_cpu_advisory_20260509_codex/contest_auth_eval.macos_cpu_advisory.json`
- Result review packet:
  `.omx/research/artifacts/a5_channel_qbits_dp_qsum200_macos_cpu_advisory_review_20260509_codex.json`
- Dispatch claim terminal status:
  `completed_macos_cpu_advisory_negative`
- Score: `0.2016517950045961`
- SegNet distortion: `0.00064302`
- PoseNet distortion: `0.00003541`
- Rate term: `0.11853221568109021`
- Samples: `600`
- Inflate elapsed: `37.83934537495952 s`
- Evaluate elapsed: `428.723048832966 s`

This result is not `[contest-CPU]` because it was run on Apple Silicon macOS.
It is a collapse screen only and is not rank, promotion, or kill evidence.

## Adversarial Interpretation

The schedule is mathematically optimal for its stated proxy, but the proxy is
not task-aligned. Cutting all selected channels uniformly to q7 saves bytes
while moving decoded frames into the same SegNet-limited basin seen in earlier
A5 scalar schedules. The advisory score worsens because the extra SegNet term
dominates the `81 B` byte win versus the compact q7/q8 scalar packet.

Do not infer that channel allocation is bad. The negative result says only
that latent-domain MSE over channels is the wrong allocator objective at this
trust point. The next channel schedule must be learned or score-domain:
boundary logits, component marginals, Jacobian/Fisher pullback, or q-bit noise
inside training.

## Commands

```bash
.venv/bin/python tools/build_a5_channel_qbits_schedule.py \
  --target-qsum 200 \
  --json-out reports/a5_channel_qbits_dp_qsum200_20260509.json \
  --candidate-id a5_channel_qbits_dp_qsum200

.venv/bin/python tools/build_pr101_frame_conditional_runtime_packet.py \
  --q-bits-sideinfo-encoding channel_raw3 \
  --channel-q-bits-json reports/a5_channel_qbits_dp_qsum200_20260509.json \
  --recompute-wire-contract-for-q-bits \
  --candidate-id pr101_a5_channel_qbits_dp_qsum200 \
  --output-dir experiments/results/pr101_a5_channel_qbits_dp_qsum200_20260509_codex \
  --force

.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_a5_channel_qbits_dp_qsum200_20260509_codex/packet/archive.zip \
  --inflate-sh experiments/results/pr101_a5_channel_qbits_dp_qsum200_20260509_codex/packet/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/pr101_a5_channel_qbits_dp_qsum200_macos_cpu_advisory_20260509_codex/work \
  --json-out experiments/results/pr101_a5_channel_qbits_dp_qsum200_macos_cpu_advisory_20260509_codex/contest_auth_eval.macos_cpu_advisory.json \
  --keep-work-dir
```

## Reactivation Criteria

- Replace latent-domain MSE DP with score-domain or learned channel schedules.
- Keep `channel_raw3` as the runtime compiler primitive for future schedules.
- Reopen exact CUDA or Linux x86_64 contest-CPU spend only if local advisory is
  within `0.001` of A1/PR101 or if this stacks with a byte-different learned
  schedule that lowers measured SegNet distortion.
- Do not relaunch the exact `qsum=200` MSE schedule.
