# Codex Findings: PR95 MLX Parallel Campaign And Dynamic Acquisition

UTC: 2026-05-25T11:54:10Z
Status: integrated evidence; no score authority

## PR95 MLX Campaign Status

The PR95/HNeRV MLX lane is active and queue-owned, but it is not yet a full
source-faithful PR95 reproduction. The implemented surface now includes native
MLX PR95 decoder/timing primitives, PR95 public archive parse/export smoke,
source-video pair loading, native MLX eval-roundtrip/YUV6 preprocessing,
RGB+YUV6 source-video timing loss, stage 1/5/8 optimizer descriptors, and
experiment_queue.v1 execution plans.

Fresh queue-owned source-video smoke:

- Queue:
  `experiments/results/pr95_mlx_source_video_rgb_yuv6_parallel_campaign_20260525T115210Z/experiment_queue.json`
- Matrix manifest:
  `experiments/results/pr95_mlx_source_video_rgb_yuv6_parallel_campaign_20260525T115210Z/matrix_manifest.json`
- Stages executed: 1, 5, 8
- Target: real `upstream/videos/0.mkv` pair index 0
- Loss surface: `rgb_yuv6_mse`
- Smoke scale: `base_channels=4`, `latent_dim=8`, one step, one pair
- Queue result: 3 succeeded, 0 failed, 0 orphaned
- Stage 1 timing: 0.03866720799123868 seconds/step
- Stage 5 timing: 0.02803820901317522 seconds/step
- Stage 8 timing: 0.031920666981022805 seconds/step
- Each stage wrote manifest, representation-training manifest,
  source-video training target, source-video preprocess smoke, public archive
  smoke, and archive export metadata.

This is useful local MLX training/substrate signal. It is not a score claim:
all manifests keep `score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Remaining PR95 Blockers

The current exact-readiness blockers are correct and should stay fail-closed:

- `pr95_stage_hparams_and_cosine_schedules_not_all_source_matched`
- `pr95_qat_c1a_and_resume_semantics_not_ported_to_mlx`
- `pr95_export_forward_parity_not_established`
- `byte_closed_smoke_archive_not_consumed_by_pr95_runtime`
- `runtime_consumption_proof_missing`
- `receiver_proof_missing`
- `requires_pytorch_export_forward_parity_on_source_checkpoint`
- `requires_byte_closed_contest_archive_export`
- `requires_exact_cpu_cuda_auth_eval_before_score_claim`
- `pr95_source_video_targets_ready_but_scorer_loss_not_wired_to_mlx`
- `pr95_segnet_posenet_network_loss_not_wired_to_mlx`
- `full_frame_inflate_parity_against_source_runtime_not_run`
- `pr95_source_video_rgb_yuv6_preprocess_loss_is_not_full_scorer_loss`

Immediate next implementation step: make the PR95 MLX campaign runner scale from
tiny source-video smokes to measured c36/latent28 timing smokes, then wire full
scorer-loss or a calibrated scorer-loss bridge before export-forward parity.

## Dynamic Acquisition Signal From Research Subagent

The MUDDFormer/Muon research pass was read-only and produced no code, but it
has actionable signal. The high-EV transfer is not to put a full MUDDFormer in
the final inflate runtime. The first useful bridge is a deterministic,
planning-only dynamic gate over existing acquisition/replan inputs:

- family and operation-set identity;
- pair/frame/window/region/unit features;
- queue cache identity and resource kind;
- observed materializer delta and receiver-contract outcome;
- uncertainty, negative observations, and exact-readiness blockers.

This belongs in the inverse-steganalysis acquisition/replan surface and should
consume `materializer_feedback` rows plus queue observations. It must remain
false-authority: dynamic coefficients may reorder local candidates, demote
blocked families, widen exploration, or select follow-up MLX/local CPU probes;
they must not imply score, rank/kill, promotion, or exact dispatch authority.

For PR95/HNeRV optimizer work, the actionable transfer is optimizer-routing
discipline: Muon only for eligible hidden 2D+ matrices; AdamW/Adam for heads,
embeddings, biases, norms/gains, and any small dynamic-gate MLP parameters.
The existing PR95 optimizer matrix queue is the right place to add explicit
descriptor variants.

## Wiring Direction

- Rate attack: materializer leaves emit canonical
  `serialized_archive_delta_contract.v1`; queue observation turns them into
  feedback; inverse-steganalysis acquisition learns grouped follow-up plans.
- PR95/local substrate training: queue-owned source-video MLX smokes graduate
  to source-faithful scorer-loss training, export parity, byte-closed archive
  replay, and exact auth anchors.
- Shared abstraction: both paths consume false-authority local evidence and
  only promote via byte-closed runtime proof plus exact contest CPU/CUDA eval.
