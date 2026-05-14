# PR101 / PR106 Local-Minimum Escape Sweep - 2026-05-14

## Scope

Operator request: pursue local/proxy score-lowering for PR101/PR106 HNeRV artifacts, especially film-grain, postfilter, and component-bearing changes that can beat `<0.192` on CPU or produce a CUDA-improving candidate. No paid remote dispatch was launched in this pass.

Mandatory preflight was completed before the sweep: `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, git status, lane registry, active dispatch claims, and recent directive files. Protected dirty files named by the operator were not edited.

## Existing Best PR101 CPU Candidate

Current exact CPU near-miss:

- Archive: `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_overlay_20260514_codex/archive.zip`
- SHA-256: `8866ebb655e96ccf0ffcd84feae08c131734cba8c402bfb8c661a29f289ce409`
- Bytes: `178517`
- Selector payload: `249` bytes
- Selector palette: `none`, `frame0_blue_chroma_amp_3`, `frame0_luma_bias_-1`, `frame0_luma_bias_-4`, `frame0_rgb_bias_m2_p1_p1`, `frame0_rgb_bias_p0_m1_p1`, `frame0_rgb_bias_p0_p1_m1`, `frame0_rgb_bias_p2_m1_m1`
- Proxy charged score: `0.1921079648509577`
- Proxy uncharged score: `0.19193550738209908`
- Contest-CPU score: `0.19209788683213053`
- Contest-CPU components: avg PoseNet `0.00002959`, avg SegNet `0.00056029`
- Exact CPU artifact: `experiments/results/modal_auth_eval_cpu/archive_8866ebb655e9/modal_cpu_auth_eval_result.json`

Byte-only gap to `<0.192` at unchanged components is about `147.01` bytes. The uncharged proxy already clears `<0.192`, so the remaining bottleneck is selector/archive overhead or a small real component gain.

## New PR101 Top-64 CPU Overlay

Command:

```bash
.venv/bin/python tools/frame_exploit_segnet_posenet_sweep.py --device cpu --candidate-raw experiments/results/lane_macos_cpu_proxy_empirical_validation_20260513_20260513T154314Z/pr101/work/inflated/0.raw --baseline-json experiments/results/lane_macos_cpu_proxy_empirical_validation_20260513_20260513T154314Z/pr101/contest_auth_eval.json --output-dir experiments/results/frame_exploit_segnet_posenet_20260514_pr101_cpu_top64_topmodes_codex --mode-catalog existing --mode-ids frame0_blue_chroma_amp_3,frame0_rgb_bias_p2_m1_m1,frame0_rgb_bias_m2_p1_p1,frame1_rgb_bias_p2_m1_m1,frame1_rgb_bias_m2_p1_p1,frame0_luma_bias_-1 --pair-indices 2,502,546,515,540,379,535,372,558,507,581,504,320,237,394,79,409,410,533,188,282,126,293,80,213,184,273,37,395,355,397,166,24,531,518,309,19,508,163,64,337,457,269,16,380,365,545,305,248,526,270,528,512,481,162,537,562,557,43,53,40,36,291,61 --batch-size 16
```

Artifacts:

- `experiments/results/frame_exploit_segnet_posenet_20260514_pr101_cpu_top64_topmodes_codex/sweep_manifest.json`
- `experiments/results/frame_exploit_segnet_posenet_20260514_pr101_cpu_top64_topmodes_codex/selector_policy_sample.json`
- `experiments/results/frame_exploit_segnet_posenet_20260514_pr101_cpu_top64_topmodes_codex/pair_component_rows.jsonl`
- `experiments/results/frame_exploit_segnet_posenet_20260514_pr101_cpu_top64_topmodes_codex/observations.json`

Result:

- Runtime: `320.71347284317017` seconds
- Axis: local CPU advisory / proxy, not contest score
- Selected non-none pairs: `63/64`
- Mean no-rate component delta on sampled pairs: `-0.00479551442384246`
- The existing sweep tool still reported `best_safe none`, so this was treated as an overlay signal only.

## New PR101 Exact-Evaluable Packet Build

Command:

```bash
.venv/bin/python tools/build_pr101_frame_exploit_selector_packet.py --artifact-dir experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex --archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip --source-runtime experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec --output-dir experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_top64_overlay_20260514_codex --selector-policy-mode compact_exact_k8 --lane-id lane_pr101_fec3_compact_exact_k8_cpu_top64_overlay_20260514 --overlay-artifact-dir experiments/results/frame_exploit_segnet_posenet_20260514_pr101_cpu_top64_topmodes_codex
```

Artifacts:

- `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_top64_overlay_20260514_codex/archive.zip`
- `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_top64_overlay_20260514_codex/packet_manifest.json`
- `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_top64_overlay_20260514_codex/archive_manifest.json`
- `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_top64_overlay_20260514_codex/submission_dir/`

Result:

- Archive SHA-256: `927169a6f7ad75b9a36999c130645eb65848c213978f471311253548ef4baa43`
- Bytes: `178517`
- Selector payload: `249` bytes
- Selected non-none pairs after merge: `435/600`
- Proxy charged score: `0.19211534792326052`
- Proxy uncharged score: `0.19194289045440188`
- Classification: not better than the existing exact CPU near-miss. No exact CPU/CUDA dispatch recommended from this packet without another reason.

## PR101 Guard Relaxation Boundary

The first guard sweep attempt exposed a wall-clock issue in the builder: repeated `compact_exact_k8` searches were pure-Python exhaustive loops. This pass hardened the builder with a deterministic NumPy exact-search path plus a manifest-visible `--selector-seg-guard-delta` knob, then built one byte-closed packet from the best observed guard boundary.

Command:

```bash
.venv/bin/python tools/build_pr101_frame_exploit_selector_packet.py --artifact-dir experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex --archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip --source-runtime experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec --output-dir experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_top64_guard1e7_20260514_codex --selector-policy-mode compact_exact_k8 --selector-seg-guard-delta 1e-7 --lane-id lane_pr101_fec3_compact_exact_k8_top64_guard1e7_20260514 --overlay-artifact-dir experiments/results/frame_exploit_segnet_posenet_20260514_pr101_cpu_top64_topmodes_codex --allow-nonpositive-charged-proxy
```

Artifacts:

- `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_top64_guard1e7_20260514_codex/archive.zip`
- `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_top64_guard1e7_20260514_codex/packet_manifest.json`
- `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k8_top64_guard1e7_20260514_codex/archive_manifest.json`

Result:

- Archive SHA-256: `40ddab6b553f93ed51631ff13bd6ca0b5ea38c2931b627b0600f48027b1ac0a3`
- Bytes: `178517`
- Selector payload: `249` bytes
- Selected non-none pairs after guard: `436/600`
- Search engine: `numpy_exact_exhaustive`
- Proxy charged score: `0.1921051475342287`
- Proxy uncharged score: `0.19193269006537006`
- Classification: not dispatchable. The guard helps the proxy by about `1.02e-5` versus the top-64 no-guard packet, but it still does not clear the existing exact CPU near-miss or the `<0.192` target.

## PR106 / HDM8 Postfilter Check

Existing exact CUDA artifacts reviewed:

- HDM8 fixed-length packet: `experiments/results/modal_auth_eval/hnerv_hdm8_fixed_lengths_modal_t4_retry2_20260514T095000Z/modal_cuda_auth_eval_result.json`
  - Archive SHA-256: `8a30730e863a2f846d7ca3a707b3191ad64312f5270976dc5f9322ba4228e8c2`
  - Bytes: `186395`
  - Contest-CUDA score: `0.20636166502462222`
  - Components: avg PoseNet `0.00003236`, avg SegNet `0.0006426`
- PR106 rank-elided format04: `experiments/results/modal_auth_eval/pr106_r2_rank_elided_format04_exact_cuda_20260514T165500Z/modal_cuda_auth_eval_result.json`
  - Archive SHA-256: `bf83c2ffc559dd42eec131e283bf106b789c0debfe7c3323e10ce1b5d8aa9a70`
  - Bytes: `186776`
  - Contest-CUDA score: `0.20661535728576175`
- HDM8 even-frame selector: `experiments/results/modal_auth_eval/hdm8_even_frame_selector_exact_cuda_20260514T111813Z/modal_cuda_auth_eval_result.json`
  - Archive SHA-256: `f44cabbdcbac50e49f1d67e5075bb6912919877f6566e34053ab6423eeb165fe`
  - Bytes: `188756`
  - Contest-CUDA score: `0.22816528594942062`
- HDM8 charged film-grain selector: `experiments/results/modal_auth_eval/exact_eval_hdm8_selector_charged_mps_aggressive_v2_20260514T211907Z/modal_cuda_auth_eval_result.json`
  - Archive SHA-256: `793747837bb1d71987e4a7055f35e25620f8eb530e6f297cc2020e5e00f1d798`
  - Bytes: `187366`
  - Contest-CUDA score: `0.2161099173824375`

The local HDM8 postfilter summary showed MPS/proxy positives, including charged selector proxy `0.21674300805153718` from `experiments/results/hdm8_selector_charged_mps_aggressive_v2_20260514_codex/packet_manifest.json`, but exact CUDA did not transfer into a frontier-improving packet.

## Decision

No new exact-eval dispatch was launched. The new PR101 top-64 overlay and the guarded exact-K8 packet produced valid archives but did not beat the already evaluated `8866ebb...e409` near-miss. The PR106/HDM8 postfilter and film-grain exact CUDA results are not `<0.192` and do not provide a better CUDA candidate in this slice.

Next exact-eval candidate from this pass: none. Best actionable existing packet remains the PR101 FEC3 compact exact-K8 CPU near-miss (`8866ebb655e96ccf0ffcd84feae08c131734cba8c402bfb8c661a29f289ce409`) for selector-overhead reduction work, not another paid eval of the top-64 overlay.
