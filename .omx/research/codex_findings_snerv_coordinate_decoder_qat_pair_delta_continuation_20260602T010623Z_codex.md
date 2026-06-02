# Codex Findings: SNeRV Coordinate Decoder QAT Pair-Delta Continuation

UTC: 2026-06-02T01:06:23Z
Agent: codex
Axis: [macOS-CPU advisory]
Authority: false-authority local continuation only

## Verdict

SNeRV coordinate decoder perturbation is NO-GO for promotion or exact eval.
The 1-pair improvement survived a 2-pair adjacent check but contracted sharply;
the new strided 2-pair window still improved aggregate score, but per-pair
metrics showed cancellation; the 4-pair strided window produced no
pose-guarded accepted coordinate row.

Next SNeRV implementation should pivot from linear top-weight coordinate
perturbation to learned/nonlinear or scorer-loop decoder QAT, with PoseNet as a
hard guard and per-pair component deltas preserved.

## Artifacts

- `.omx/research/snerv_scorer_loop_decoder_qat_coordinate_2pair_strided_smoke_20260601T2359Z.json`
  - sha256: `c983b7303177c58a8707f6ed829e3961ec5f8599f7f3ddd0c7403d859134c540`
  - baseline score: `1.0234573726510832`
  - best score: `1.014656504108282`
  - delta score: `-0.008800868542801155`
  - delta d_seg: `-7.62939453125e-06`
  - delta d_pose: `-0.0005776598118245602`
- `.omx/research/snerv_scorer_loop_decoder_qat_coordinate_2pair_strided_pose_gate_20260602T0001Z.json`
  - sha256: `fe7516ba190f47015683b31d2b0b40fd64e11a0fa0ac3427ed0cad8cb124d664`
  - verdict: `GO_LOCAL_CONTINUATION_ONLY`
- `.omx/research/snerv_scorer_loop_decoder_qat_coordinate_4pair_strided_smoke_20260602T0003Z.json`
  - sha256: `68a62581a7f5e7ff8cced44bdcd389301273928c872b9784d75bfa67b8a5a5eb`
  - baseline score: `1.3672043248166048`
  - best score: `1.3672043248166048`
  - accepted improvement: `false`
- `.omx/research/snerv_scorer_loop_decoder_qat_coordinate_4pair_strided_pose_gate_20260602T0007Z.json`
  - sha256: `977e46385c7db86ffc42abee9c13acf1d254bcd999b78a6ca5e749b7af816857`
  - verdict: `NO_GO_FOR_PROMOTION_OR_EXACT_EVAL`
  - blocker: `no_candidate_passes_pose_guarded_local_continuation_gate`

## Pair-Local Diagnosis

The 2-pair strided best row was `coord_039_plus`. Aggregate score improved, but
the component deltas were not uniformly good:

- pair 0: `d_pose_linf_delta=+0.0010519837960600853`,
  `d_seg_linf_delta=0.0`, no-rate score delta `+0.01858282580470616`
- pair 1: `d_pose_linf_delta=-0.0022073034197092056`,
  `d_seg_linf_delta=-1.52587890625e-05`, no-rate score delta
  `-0.027692473097189252`

So the aggregate 2-pair win was carried by one pair while another pair worsened.
That is not robust enough for full-600 escalation.

In the 4-pair strided window, the lowest-score coordinate candidate was
`coord_040_minus`, but it failed the pose guard:

- candidate score: `1.367112023164475`
- candidate d_seg: `0.0027821858529932797`
- candidate d_pose: `0.007283248327439651`
- blockers: `pose_guard_failed`

## Code/Schema Landing

`SnervDecoderEval` now preserves `per_pair` rows with:

- `pair_index`
- `d_seg_linf`
- `d_pose_linf`
- `score_linf_without_rate`

`SnervScorerLoopDecoderQatSmokeResult` now preserves
`best_pair_deltas` against the baseline. This prevents pair-local detector
response from being flattened away before the pose gate, memo layer, or next
decoder optimizer consumes the signal.

## Recovery/Dispatch State

PR101 storage-order CPU auth eval remains pending as of recovery poll
`2026-06-02T01:04:33Z`:

- call id: `fc-01KT2BZT54G6CXPMD94SY43MMH`
- output dir:
  `/Users/adpena/Projects/pact/experiments/results/modal_auth_eval_cpu/pr101_storage_order_len24_cpu_20260601T1955Z`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

No new full-video, exact, CUDA, or paired eval work should launch while this
lane is pending.

The completed local PACT-VQ QAT/int2 full-600 run already has terminal
false-authority claim evidence in `.omx/state/active_lane_dispatch_claims.md`;
an attempted duplicate mirror row was removed during review:

- lane id: `lane_compact_pact_vq_qat4_int2_full600_foreground_mlx_20260601`
- job id:
  `compact_pact_vq_qat4_int2_full600_2000ep_foreground_20260601T224126Z`
- archive bytes: `35550`
- archive sha256:
  `d1de439bed2ac68a453a943c0d83bb7cbce72b115e04e108ba1eaf57ae727fe4`
- receiver proof: terminal row says receiver-proven full-video MLX demoted
- `score_claim=false`
- exact status: not exact-gate plausible until stronger SegNet boundary/logit
  decoder-weight fitting

## Blockers

- SNeRV coordinate perturbation has contraction/cancellation across broader
  local windows.
- SNeRV still lacks full-600 byte-closed receiver proof for this QAT path.
- SNeRV still lacks paired contest CPU/CUDA pass.
- Current decoder QAT still emits fp32 receiver payload after fake quant; rate
  work remains open via mixed-precision decoder grammar or decoder-delta
  packing.
- PR101 CPU auth eval remains pending; exact/CUDA/full-video launch remains
  blocked.

## Next Actions

1. Implement learned/nonlinear or scorer-loop decoder QAT for SNeRV, keeping
   PoseNet as a hard guard and preserving per-pair deltas.
2. Add byte-work for SNeRV decoder payloads: mixed-precision grammar,
   decoder-delta packing, or both.
3. Continue polling PR101 CPU through `tools/recover_modal_auth_eval.py`; append
   a terminal claim row when it resolves.
4. Harvest/adjudicate the PACT-VQ QAT/int2 archive only as false-authority local
   evidence until scorer replay and exact gates are satisfied.
5. Do not promote SNeRV/PACT-VQ/HiNeRV work until the archive/runtime/eval axis
   is byte-closed and paired CPU/CUDA contest evidence exists.
