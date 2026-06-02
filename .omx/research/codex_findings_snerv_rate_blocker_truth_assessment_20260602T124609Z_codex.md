# Codex Findings: SNeRV Rate-Blocker Truth Assessment

UTC: 2026-06-02T12:46:09Z

## Question

Is the statement true?

> SNeRV: still rate-blocked on stored LF unless representation changes. Next
> useful work is learned/scorer-preserving LF/HF generation, SR low-res carrier,
> symbolic/shared residual grammar, score-aware decoder fit, and wavelet-group
> saliency binding. Human visual fidelity is irrelevant unless proven
> scorer-causal.

## Verdict

Mostly true for the current local SNeRV implementation, but not proven as a
fundamental limitation of SNeRV as a design.

The current full-600 packet is rate-blocked on explicit LF storage: the LF
payload is `9,996,235` bytes, and simple lossless LF predictors improved it by
only `355` bytes at best. That closes the easy deterministic LF-predictor branch
for the current packet. It does not prove SNeRV is intrinsically capped, because
the current local SNeRV stack is still not source-faithful and not fully
optimized.

## Evidence

- `.omx/research/snerv_full600_lf_predictor_profile_20260602T111906Z.json`
  - `current_lf_payload_bytes`: `9,996,235`
  - best simple predictor: `raster_delta`, `9,995,880` bytes
  - verdict: `simple_lossless_lf_predictors_do_not_collapse_full600_lf_payload`
- `.omx/research/snerv_rate_sweep_charged_adjudication_20260601T210518Z.json`
  - representative charged row: `273,300` archive bytes, `150,260` LF bytes,
    `122,624` step-map bytes, `score_linf_advisory=2.0789`
  - component terms from the row: rate term about `0.18198`, SegNet term about
    `0.98648`, PoseNet term about `0.91046`
  - classification: `distortion_promising_step_map_rate_blocked`
- `.omx/research/snerv_scorer_loop_decoder_qat_smoke_nes1pair_segslack_20260602T113400Z.json`
  - local one-pair score-aware continuation improved `score_linf` by
    `-0.009854`
  - PoseNet improved while SegNet worsened slightly within explicit slack
  - still blocked by `local_smoke_only_not_full_600_pairs`,
    `paired_contest_cpu_cuda_pass_missing`, and
    `mixed_precision_decoder_payload_grammar_not_byte_optimized`
- `.omx/research/nerv_source_parity_contract_20260602T124016Z_codex.json`
  - SNeRV long-training ready: no
  - blockers include missing MFU/HFR/stride-stack closure, scorer-loop decoder
    QAT, QAT receiver codec pricing, official Haar mode, dependency custody,
    and fc_dim/modelsize control.

## Interpretation

The phrase "stored LF unless representation changes" is accurate for the current
full-600 SNAR1-style packet and simple lossless coding attempts. The required
representation changes are exactly the right next direction:

- learned/scorer-preserving LF generation;
- learned/scorer-preserving HF/HFR receiver path;
- low-res/SR carrier when scorer downsample makes high-frequency output
  irrelevant;
- symbolic/shared residual grammar when residuals repeat across pairs/planes;
- score-aware decoder fit with receiver-visible quantization during training;
- wavelet-group saliency binding so protected bits go to scorer-causal groups.

The phrase "fundamental limitations" should not be used yet. Official SNeRV is
designed to store LF and generate HF through MFU/HFR structure; our local stack
still lacks source-faithful MFU/HFR/stride parity and full receiver-priced QAT.
Bad current scores are therefore implementation/config/proof blockers, not a
method-negative verdict.

Human visual fidelity is not an optimization objective in this contest unless a
specific visual feature is proven scorer-causal through SegNet/PoseNet response,
saliency, or paired component deltas.

## Current Profiling Status

Partial binary and score contribution profiling exists:

- LF payload byte profile across full 600 pairs;
- charged SNeRV rate sweep with LF bytes, step-map bytes, metadata/decoder
  bytes, and advisory `d_seg`/`d_pose`/score rows;
- receiver-visible step-map waterfill and LF codec portfolio;
- local score-aware decoder-QAT smokes with per-pair component deltas.

Missing before any production claim:

- full600 byte-closed receiver proof for the current SNeRV packet;
- same-axis PR95 control replay;
- paired contest CPU/CUDA component profile;
- per-section/atom saliency bound into SNeRV wavelet/LF/HFR groups;
- measured modelsize/fc_dim ladder under receiver archive bytes.

## Actionable Next Step

Continue SNeRV as a top-priority local optimization stack, but stop spending
time on simple lossless LF predictors as promotion evidence. Push the next work
into representation and scorer-aware training: receiver-visible learned LF/HFR
grammar, low-res/SR scorer-preserving carrier, and wavelet-group saliency
allocation inside decoder-weight QAT.

Authority: false. This memo is planning/control only; it is not a score claim,
promotion claim, rank/kill claim, or exact dispatch authorization.
