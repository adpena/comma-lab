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

1. Rebuild the selector packet after committing current HLM3 source changes so
   transparency records a clean source tree.
2. If Modal CUDA infrastructure is healthy, claim lane
   `hnerv_hdm8_film_grain_sidecar_exact_eval` and run the exact-CUDA command
   from the packet manifest.
3. If CUDA selector regresses, classify it as a transfer failure from MPS proxy
   to contest CUDA, not as evidence against postfilter selectors generally.
4. Do not spend more wall-clock on HLM2 fixed-latent transforms for this exact
   archive unless a new coder charges model cost below the current sparse
   delta-position stream.

