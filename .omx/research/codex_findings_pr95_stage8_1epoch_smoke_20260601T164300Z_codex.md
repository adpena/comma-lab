# Codex Findings: PR95 Stage-8 One-Epoch Smoke

UTC: 2026-06-01T16:43:00Z

## Artifact

Output root:

`/Volumes/VertigoDataTier/pact/compact_pr95_stage8_source_1ep_bridge_codex_v1`

Primary runner report:

`/Volumes/VertigoDataTier/pact/compact_pr95_stage8_source_1ep_bridge_codex_v1/compact_renderer_mlx_spine_runner_report.json`

Stage-8 source report:

`/Volumes/VertigoDataTier/pact/compact_pr95_stage8_source_1ep_bridge_codex_v1/pr95_stage8_source_lane/pr95_stage8_from_public_archive_report.json`

Receiver proof:

`/Volumes/VertigoDataTier/pact/compact_pr95_stage8_source_1ep_bridge_codex_v1/pr95_stage8_source_lane/archive_bound_candidate/receiver_proof/pr95_mlx_pytorch_package_receiver_proof.json`

## Result

Ran the real PR95 Stage-8 public-archive continuation path for one epoch on CPU
with a cached full-video target bundle.

Training report:

- Epochs: `1`
- Device: `cpu`
- Train elapsed: `1220.5032739639282` seconds
- Best advisory score: `0.1986848301858683`
- Best advisory SegNet distortion: `0.0006129031678816924`
- Best advisory PoseNet distortion: `0.00003497582838463131`
- Best inner `0.bin` bytes: `178255`

Byte-closed runner report:

- `archive.zip` bytes: `178363`
- `archive.zip` SHA-256:
  `61fae2691fc674e11307307e1a87e8f0aef75ebcad4f34fd780980ed68e87f74`
- Declared pairs: `600`
- Receiver proof observed: `true`
- Receiver proof passed: `true`
- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`

Receiver proof:

- Contract satisfied: `true`
- Runtime consumption proof passed: `true`
- Inflate return code: `0`
- Inflate wall seconds: `132.145235`
- Receiver output bytes: `3662409600`
- Receiver output SHA-256:
  `574374f2c5e29dbcb0d77db6d71afc89415a2d849c6fd0a97d972e9f80cd11a5`
- Runtime tree SHA-256:
  `db9968fc3095871c9280c404ddbf97733ca999f6c2a5dc72af8489004ee4c9bd`
- Receiver output retained: `false`

Disk hygiene:

- Artifact root size after cleanup: `6.1M`
- No `.raw`, `.yuv`, `.mp4`, or `.mkv` files remain under the output root.
- Bulky rebuildable artifacts are on `/Volumes/VertigoDataTier/pact`, not local
  source disk.

## Interpretation

This is a useful timing and custody smoke, not a score claim. It proves the
Stage-8 continuation path can run end-to-end from the public archive, emit a
byte-closed archive, and satisfy deterministic receiver consumption while
cleaning the huge raw output after proof.

It is not dispatchable yet because the runner correctly reports:

- `contest_cpu_cuda_exact_eval_missing`
- `contest_cpu_cuda_exact_eval_not_executed`
- `full_video_mlx_scorer_replay_not_attached`
- `some_sections_missing_value_per_byte_measurement`

The score-lowering implication is direct: continue Stage-8/compact-base work,
but the next promotion gate is not more prose. It is full-video MLX scorer replay
and section value-per-byte pricing on the byte-closed survivor, then exact
CPU/CUDA dispatch only if the full-video evidence says the candidate can move
the frontier.
