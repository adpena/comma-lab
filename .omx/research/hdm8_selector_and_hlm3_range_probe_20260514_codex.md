# HDM8 Selector + HLM3 Range Probe - 2026-05-14

## Scope

Operator directive: continue score-lowering work, preserve all signal, avoid
memo-only churn, and keep public/contest axes separate.

This ledger records three concrete artifacts from the current frontier pass:

1. Exact entropy floor probe for the active HDM8 archive.
2. Charged MPS-derived HDM8 postfilter selector packet.
3. HLM3 PR103-style range-coded high-byte probe for PR106 fixed latents.

No artifact in this ledger is a score claim. Local CPU/MPS numbers are proxy
signals only. Promotion still requires claimed contest-CUDA exact auth eval.

## Active Source Archive

- label: `PR106-R2-HDM8-HLM2-XMEMBER`
- archive: `experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/exact_eval_static_release_surface/archive.zip`
- bytes: `186395`
- sha256: `8a30730e863a2f846d7ca3a707b3191ad64312f5270976dc5f9322ba4228e8c2`
- exact-CUDA reference artifact: `experiments/results/modal_auth_eval/hnerv_hdm8_fixed_lengths_modal_t4_retry2_20260514T095000Z/contest_auth_eval.json`
- exact-CUDA reference score: `0.20636166502462222`

## Entropy Floor Probe

Command:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/pr106_entropy_floor_probe.py \
  --archive experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/exact_eval_static_release_surface/archive.zip \
  --json-out experiments/results/pr106_hdm8_entropy_floor_probe_20260514_codex/entropy_floor.json \
  --md-out experiments/results/pr106_hdm8_entropy_floor_probe_20260514_codex/entropy_floor.md \
  --pr101-reference-archive-bytes 178517 \
  --active-floor-archive-bytes 186395 \
  --active-floor-label PR106-R2-HDM8-HLM2-XMEMBER
```

Artifacts:

- JSON: `experiments/results/pr106_hdm8_entropy_floor_probe_20260514_codex/entropy_floor.json`
- JSON sha256: `8ba9c9ebbce1a60881e537914668d3438ef1c193d80141fca41f64266acc27cd`
- Markdown: `experiments/results/pr106_hdm8_entropy_floor_probe_20260514_codex/entropy_floor.md`
- Markdown sha256: `ca97d0712563b39d0997deec0f7a58b63d1f90c1212293558c4ffb7ed1a24352`

Key result:

- decoder section: `169974` bytes
- latent section: `15776` bytes
- oracle best decoder markov2 floor is large but unpriced model complexity is
  prohibitive for a near-term packet.
- fixed-latent markov floors suggested possible byte mass, so the next action
  was a charged HLM3/transform probe.

## Charged Latent Probe Outcome

Measured on exact HDM8 fixed latents:

- current HLM2 latent section: `15776` bytes
- low-byte transform + Brotli variants: best was `+1` byte vs current.
- Huffman, prev-symbol Huffman, and static range variants were all worse.
- HLM3 range-coded high-byte probe is byte-closed but not a rate win on this
  binary high-byte stream because the charged model/header cost exceeds HLM2's
  sparse delta-position stream.

Engineering outcome:

- HLM3 lands as a fail-closed reusable probe for dense/non-binary high-byte
  latent streams, not as a promotion candidate for current HDM8.
- Dispatch blocker is explicit: `hlm3_inflate_runtime_decoder_missing`.
- This prevents future agents from treating the PR103 high-byte arithmetic idea
  as untested on the current fixed-latent stream.

Focused test:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_pr106_fixed_latent_recode.py -q
```

Result: `20 passed`.

## Charged HDM8 Selector Packet

Command:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/build_hdm8_film_grain_sidecar_packet.py \
  --archive experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/exact_eval_static_release_surface/archive.zip \
  --proxy-json experiments/results/hdm8_local_first_postfilter_sweep_20260514_codex/aggressive_rss_v1/mps_full_sweep.json \
  --selector-from-proxy-json \
  --pack-selector-into-archive \
  --selector-codec brotli \
  --require-positive-proxy \
  --output-dir experiments/results/hdm8_selector_charged_mps_aggressive_v1_20260514_codex \
  --json-out experiments/results/hdm8_selector_charged_mps_aggressive_v1_20260514_codex/packet_manifest.json
```

Packet artifact:

- archive: `experiments/results/hdm8_selector_charged_mps_aggressive_v1_20260514_codex/archive.zip`
- archive bytes: `187366`
- archive sha256: `793747837bb1d71987e4a7055f35e25620f8eb530e6f297cc2020e5e00f1d798`
- runtime: `experiments/results/hdm8_selector_charged_mps_aggressive_v1_20260514_codex/submission_dir`
- modal-uploaded runtime tree sha256: `3753e2d66b880cbda3742b924a3f268f351e03a06a8b4cd33d083fce57619b9f`
- selector payload codec: `brotli`
- selector payload charged bytes: `969`
- archive byte delta vs source: `+971`

Proxy-only MPS signal:

- axis: `local-mps-proxy-prefix`
- baseline proxy score: `0.22780881106535603`
- charged selector proxy score: `0.21674300805153718`
- charged proxy delta: `-0.011065803013818848`
- selector packed in archive: `true`
- score_claim: `false`
- promotion_eligible: `false`

Focused test:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_hdm8_film_grain_sidecar.py \
  src/tac/tests/test_probe_hdm8_postfilter_frame_parity_and_no_op.py -q
```

Result: `28 passed`.

## Next Actions

1. Treat the charged selector exact-CUDA result below as a negative transfer
   result from MPS proxy to contest CUDA.
2. Do not promote MPS-derived HDM8 selectors without a CUDA-gated PoseNet
   constraint or a cheap component predictor calibrated on exact CUDA anchors.
3. Do not spend more wall-clock on HLM2 fixed-latent transforms for this exact
   archive unless a new coder charges model cost below the current sparse
   delta-position stream.

## Exact CUDA Selector Result Review

Clean-source v2 packet:

- source head: `a2112a81736a48d02949b515d48a1d72b5423faa`
- source dirty: `false`
- packet artifact: `experiments/results/hdm8_selector_charged_mps_aggressive_v2_20260514_codex/archive.zip`
- packet bytes: `187366`
- packet sha256: `793747837bb1d71987e4a7055f35e25620f8eb530e6f297cc2020e5e00f1d798`
- runtime: `experiments/results/hdm8_selector_charged_mps_aggressive_v2_20260514_codex/submission_dir`
- modal-uploaded runtime tree sha256: `3753e2d66b880cbda3742b924a3f268f351e03a06a8b4cd33d083fce57619b9f`

Exact CUDA command:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/modal_auth_eval.py \
  --archive experiments/results/hdm8_selector_charged_mps_aggressive_v2_20260514_codex/archive.zip \
  --submission-dir experiments/results/hdm8_selector_charged_mps_aggressive_v2_20260514_codex/submission_dir \
  --inflate-sh inflate.sh \
  --gpu T4 \
  --output-dir experiments/results/modal_auth_eval/exact_eval_hdm8_selector_charged_mps_aggressive_v2_20260514T211907Z \
  --expected-runtime-tree-sha256 3753e2d66b880cbda3742b924a3f268f351e03a06a8b4cd33d083fce57619b9f \
  --detach \
  --provider-detach-ack \
  --lane-id hnerv_hdm8_film_grain_sidecar_exact_eval \
  --instance-job-id exact_eval_hdm8_selector_charged_mps_aggressive_v2_20260514T211907Z \
  --claim-agent codex:gpt-5.5 \
  --claim-notes "HDM8 charged selector exact CUDA eval; archive_sha256=793747837bb1d71987e4a7055f35e25620f8eb530e6f297cc2020e5e00f1d798; source_head=a2112a81736a48d02949b515d48a1d72b5423faa; proxy_axis=local-mps; charged_proxy_delta=-0.011065803013818848"
```

Recovered artifact:

- output dir: `experiments/results/modal_auth_eval/exact_eval_hdm8_selector_charged_mps_aggressive_v2_20260514T211907Z`
- call id: `fc-01KRM5QFCKKF543ETAQANF9TZN`
- result: `modal_cuda_auth_eval_result.json`
- contest artifact: `contest_auth_eval.json`
- inflated output aggregate sha256: `0968f37ba36498e8c5efcfbe5dc6702910cfd400fdf5c3b13bcb1d7808cc98bb`
- GPU: `Tesla T4`
- sample count: `600`
- evidence grade: `[contest-CUDA]`

Exact result:

- score: `0.2161099173824375`
- rounded score: `0.22`
- avg SegNet distance: `0.0006426`
- avg PoseNet distance: `0.00007339`
- archive bytes: `187366`
- promotion eligible: `false`

Compared to the current HDM8/HLM2 CUDA floor:

- reference score: `0.20636166502462222`
- reference avg SegNet distance: `0.0006426`
- reference avg PoseNet distance: `0.00003236`
- reference bytes: `186395`
- score delta: `+0.00974825235781529`
- byte delta: `+971`
- byte-term delta: `+0.0006465490434816284`
- SegNet-term delta: `0`
- PoseNet-term delta: `+0.009101703314333673`

Classification:

- `legitimate_exact_cuda_negative_for_this_selector_packet`
- `mps_proxy_to_cuda_transfer_failure`
- `posenet_regression_dominates`
- `segnet_not_improved_on_exact_cuda`
- `not_a_global_postfilter_lane_retirement`

Reactivation criteria:

1. Selector search must optimize a CUDA-calibrated objective or include a
   PoseNet-preservation constraint before exact-CUDA dispatch.
2. Any future MPS-only candidate must pass a cheap CUDA component predictor, a
   small exact-CUDA prefix probe, or a same-runtime component sensitivity check.
3. Postfilter packets remain worth exploring only if charged selector bytes are
   amortized by a component improvement larger than the `+0.00065` byte-term
   cost observed here.

## Fail-Closed CUDA Selector Gate Landing

Implementation status: code and tests landed locally; no GPU spend and no score
claim.

New deterministic gate:

- `src/tac/hdm8_selector_cuda_gate.py` defines
  `hdm8_selector_cuda_component_risk_gate_v1`.
- MPS/local-only HDM8 selector packets fail closed with
  `mps_or_local_proxy_axis_requires_cuda_component_probe`.
- CUDA-prefix selector evidence can pass only when PoseNet, SegNet, and charged
  score deltas are all non-positive against the CUDA component baseline.
- Candidate exact-CUDA JSON can also be consumed after eval; the known charged
  MPS selector result fails this gate because PoseNet and score regress.

Wire-in:

- `tools/build_hdm8_film_grain_sidecar_packet.py` now writes
  `cuda_component_risk_gate_required=true` and the gate payload into both the
  packet manifest and archive manifest for selector packets.
- `src/tac/optimizer/exact_readiness.py` refuses exact-ready promotion when an
  HDM8 selector manifest lacks a passing gate.
- `src/tac/optimizer/exact_ready_audit.py` re-checks live archive manifests so a
  stale exact-ready queue cannot dispatch after the selector gate regresses.
- `tools/parallel_dispatch_top_k.py` refuses ready rows that directly carry a
  required but non-passing selector gate.

Focused verification:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_hdm8_selector_cuda_gate.py \
  src/tac/tests/test_hdm8_film_grain_sidecar.py \
  src/tac/tests/test_optimizer_exact_readiness.py -q
# 46 passed

PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_modal_hdm8_postfilter_sweep.py -q
# 11 passed
```

Remaining blocker: no new CUDA-prefix selector payload has passed the component
gate yet. MPS-only selector positives are now research signals only until a
Modal/CUDA prefix sweep or candidate exact-CUDA component result clears the
gate.
