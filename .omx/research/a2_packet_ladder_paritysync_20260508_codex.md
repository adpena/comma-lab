# A2 Packet Ladder Parity-Sync Artifact - Codex - 2026-05-08

scope: local A2 packet-ladder parity/certification advancement only
remote_dispatch: false
scorer_run: false
score_claim: false

## Candidate

- lane: `track1_phase_a2_sensitivity_quant_packet_ladder`
- variant: `weighted_k_00_rms_0p0386`
- packet root: `experiments/results/track1_phase_a2_packet_ladder_codex_paritysync_20260508T211430Z/variants/weighted_k_00_rms_0p0386/packet`
- archive: `experiments/results/track1_phase_a2_packet_ladder_codex_paritysync_20260508T211430Z/variants/weighted_k_00_rms_0p0386/packet/archive.zip`
- archive bytes: `159491`
- archive SHA-256: `bfb912ff7dbbd843b3bf6e5d12ff876eeab359e38113204d0ccae4277fd35d27`
- ZIP member: `x`, bytes `159391`, SHA-256 `3906da037c6e6604669a863cacd6be88efb3453ea1acbf1b885bf22bb5771a78`
- evidence grade: `empirical`

## Local Closure

- `tools/build_a2_sensitivity_weighted_pr101_packet.py --run-inflate-parity` built the packet and ran source-vs-candidate inflate parity through the packet-local runtime.
- ladder manifest: `experiments/results/track1_phase_a2_packet_ladder_codex_paritysync_20260508T211430Z/a2_packet_ladder_manifest.json`, SHA-256 `b966517380713b63ab7cd2d918ee0c87af5d00bea87f374d0e4454c15fd5ade1`
- candidate manifest: `experiments/results/track1_phase_a2_packet_ladder_codex_paritysync_20260508T211430Z/variants/weighted_k_00_rms_0p0386/candidate_manifest.json`, SHA-256 `b4fb8b467cdff441c13fb62418bc235bb991ab553cbdfdaaad5e1381c0bc909a`
- runtime probe: `experiments/results/track1_phase_a2_packet_ladder_codex_paritysync_20260508T211430Z/variants/weighted_k_00_rms_0p0386/a2_runtime_closure_probe.json`, SHA-256 `3dcbc1a96a432a343ebfb737f04d15cd33f76a7281c2678ae0465998d7f674d3`
- nonfinal compliance: `experiments/results/track1_phase_a2_packet_ladder_codex_paritysync_20260508T211430Z/variants/weighted_k_00_rms_0p0386/pre_submission_compliance.nonfinal.json`, SHA-256 `c8442e2d212de46a4ab61933e2be65566871949d95c7b275814a0403fafdde51`
- A2 local audit: `experiments/results/track1_phase_a2_packet_ladder_codex_paritysync_20260508T211430Z/a2_packet_ladder_closure_audit.local.json`, SHA-256 `e3c6cecda7dd3c5d44b8ed105dbcec18af0e919e9704ca05cf906536225270c7`

## Remaining Blockers

- `cpu_local_allocator_proxy_only`
- `diagnostic_or_stub_sensitivity_map_not_score_authority`
- `is_stub=true`
- `score_sensitivity_artifact_must_be_certified_before_promotion`
- `tag contains 'stub'`
- `no_active_level2_lane_dispatch_claim`
- `no_contest_cpu_auth_eval`
- `no_exact_cuda_auth_eval`
- `operator_score_claim_review_not_done`

This artifact is exact-evaluable as a local packet candidate, but it is not score,
rank, promotion, or kill evidence until a certified sensitivity source replaces
the stub/proxy input and both exact contest-CPU and exact CUDA auth evals land
on 1:1 compliant hardware with a lane claim.
