# PACT/VQ QAT4 rate solved, fit blocked - Codex findings

UTC: 2026-06-02T00:55:00Z

## Scope

Completed the full 600-pair PACT-NeRV-VQ ch48 coder-aware QAT4 run launched
from the foreground-managed lane. All evidence here is `[macOS-MLX
research-signal]` and false-authority. No contest CPU/CUDA exact score was
claimed or dispatched.

## Training Result

Foreground-managed long run:

- Output:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_qat4_int2_full600_2000ep_foreground_codex_20260601T224126Z`
- Report:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_qat4_int2_full600_2000ep_foreground_codex_20260601T224126Z/compact_renderer_mlx_spine_runner_report.json`
- Trained archive:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_qat4_int2_full600_2000ep_foreground_codex_20260601T224126Z/pact_nerv_vq_mlx_training/archive.zip`
- Trained archive bytes: 35,550
- Trained archive SHA-256:
  `d1de439bed2ac68a453a943c0d83bb7cbce72b115e04e108ba1eaf57ae727fe4`
- Receiver proof: passed.
- Epochs: 2,000.
- Wall clock: 5,816.98 s.

Final training telemetry:

- Loss: 0.046624571084976196
- Training SegNet proxy: 0.7535800337791443
- Training PoseNet proxy: 3.996178388595581
- Recon proxy: 0.006964808329939842

Interpretation: long training plus coder-aware QAT materially improved
pose/reconstruction versus startup, but SegNet stayed high. This is a fit
blocker, not a rate blocker.

## Codec Sweep

Sweep report:

`/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_qat4_int2_full600_2000ep_foreground_codec_sweep_codex_20260602T002200Z/compact_decoder_codec_sweep_report.json`

Best receiver-proven variant:

- Codec: `int2_mixed`
- Archive bytes: 33,657
- Archive SHA-256:
  `a87e340cbc037db2deabd5f8f413a5003ccf49763239c4663b6918ccefd9419d`
- Receiver proof: passed.
- Blockers:
  `full_video_mlx_scorer_replay_not_attached`,
  `contest_cpu_cuda_exact_eval_not_executed`.

The sweep auto-cleaned receiver raw proof output after completion.

## Full-Video MLX Replay

Section-value profile:

`/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_qat4_int2_mixed_full600_section_value_codex_20260602T002500Z/pact_nerv_vq_mlx_section_value_profile.json`

Baseline `int2_mixed` full-video MLX advisory:

- Archive bytes: 33,596 in the profiler-rebuilt baseline ZIP
- `avg_segnet_dist`: 0.5062692515552044
- `avg_posenet_dist`: 163.0826654179891
- Advisory `canonical_score`: 91.03279017666013
- Rate contribution: 0.022370197389092468

Section removals all improved the advisory objective:

- `decoder_qw`: removal delta total -0.41582863482919663
- `codebooks_q`: removal delta total -0.15453552598232534
- `selectors_rc`: removal delta total -0.21666897096754667

This means the trained bytes are not carrying enough score value in this
artifact. They are charged, receiver-consumed, but locally harmful.

## Materializer Gap Closed

The bounded runner selected PVQ section cuts, but only Selector-V4 had a
section-cut materializer. Landed:

- `tools/materialize_pact_nerv_vq_section_cut_candidate.py`
- VQ registration in `tac.substrates.hprc.spine_bounded_runner`
- Tests covering VQ section-cut work-order creation and combined materializing.

Focused verification:

- `ruff check tools/materialize_pact_nerv_vq_section_cut_candidate.py src/tac/substrates/hprc/spine_bounded_runner.py src/tac/tests/test_profile_pact_nerv_vq_mlx_section_value.py`
- `pytest src/tac/tests/test_profile_pact_nerv_vq_mlx_section_value.py src/tac/substrates/hprc/tests/test_spine_bounded_runner.py -q`

## Combined Cut Candidate

Materialized the combined measured cut:

- Sections cut: `decoder_qw`, `codebooks_q`, `selectors_rc`
- Candidate:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_qat4_int2_mixed_combined_section_cut_codex_20260602T004400Z/candidate/archive.zip`
- Candidate bytes: 22,176
- Candidate SHA-256:
  `061d434a0081d43b3194838fd6888ed29b989ce2aa51673480854cb42510eca4`
- Receiver proof: passed.

Combined-cut full-video MLX replay:

- Profile:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_qat4_int2_mixed_combined_cut_full600_replay_codex_20260602T004800Z/pact_nerv_vq_mlx_section_value_profile.json`
- `avg_segnet_dist`: 0.5048259229958058
- `avg_posenet_dist`: 160.95555227915446
- Advisory `canonical_score`: 90.61662461720066
- Rate contribution: 0.014766088144437271

Verdict: the combined cut is smaller and locally better than the trained
QAT/int2 packet, but still roughly 90 advisory score and not exact-gate
plausible.

## Durable Verdict

Rate axis for this compact PACT/VQ grammar is no longer the immediate blocker:
we can produce receiver-proven full-coverage archives at 22-34 KB. The blocker
is score fit, especially SegNet decision-boundary fit. The next run should not
be another generic QAT rerun. It should train decoder weights directly against
stronger full-video SegNet boundary/logit surfaces:

- raise boundary-argmax hinge authority relative to pose/recon;
- add class-region/logit cache surfaces so the profile can route by class and
boundary, not only section/full-video;
- consider PR95-style decoder-weight curriculum or HiNeRV/SNeRV carriers under
the same 22-100 KB grammar;
- admit residual tokens only if measured `delta_nonrate + rate_cost < 0`;
- keep exact CPU/CUDA blocked until local full-video evidence is plausibly
frontier-adjacent.
