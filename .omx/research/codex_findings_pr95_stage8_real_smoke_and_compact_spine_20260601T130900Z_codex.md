# PR95 Stage-8 Real Smoke And Compact Spine Findings

- generated_utc: 2026-06-01T13:09:00Z
- author: Codex
- status: implemented_and_smoked
- score_claim: false
- promotion_eligible: false
- rank_or_kill_eligible: false
- ready_for_exact_eval_dispatch: false

## What Landed

The PR95 Stage-8-from-public-archive lane now has a reusable full-video target
cache path. The public PR95 trainer already supports `shared_state`; the lane
uses that hook so Stage-8 continuation can reuse full-video SegNet hard labels,
PoseNet targets, and the frozen DistortionNet instead of recomputing targets
for every smoke.

The same slice added a compact renderer spine adapter for RNeRV, SR-NeRV,
BoostNeRV, PVQ-NeRV, RT-VQ-NeRV, and VQ-NeRV. It only accepts charged trained
decoder/program bytes plus trained latent/token bytes, then emits the shared
HPRC representation spine. Promotion remains blocked until archive/runtime
receiver proof, full-video scorer value-per-byte, and contest CPU/CUDA exact
gate exist.

## Real Smoke

Command:

```bash
.venv/bin/python tools/run_pr95_stage8_from_public_archive.py \
  --execute \
  --epochs 1 \
  --eval-every 1 \
  --batch-size 8 \
  --device cpu \
  --output-dir /Volumes/VertigoDataTier/pact/pr95_stage8_from_public_archive_eval_ep1 \
  --overwrite
```

Evidence:

- Report: `/Volumes/VertigoDataTier/pact/pr95_stage8_from_public_archive_eval_ep1/pr95_stage8_from_public_archive_report.json`
- Target cache: `/Volumes/VertigoDataTier/pact/pr95_stage8_target_cache/pr95_stage8_targets_257c38195c3a854e.pt`
- Target cache bytes: `943735565`
- Target-cache build time in the prior first pass: `424.1275s`
- Cached target load time: `0.2344s`
- One public Stage-8 epoch plus full local eval: `856.1742s`
- Public PR95 local scorer result: score `0.19864657917772877`, SegNet `0.0006122928205877543`, PoseNet `3.495146761755071e-05`, Stage-8 member bytes `178299`
- Packaged archive.zip bytes: `178407`
- Packaged archive.zip sha256: `5f079f0e0db3369b675f64389cf9aa4d1dd6180aceafc743058d7c191e894b52`
- Receiver proof: ready, no blockers
- Exact blocker: `contest_cpu_cuda_exact_eval_missing`

## Verdict

The lane is real and scorer-faithful, but full local CPU Stage-8 is not a
healthy long-run path. At roughly `856s/epoch` for cached one-epoch train plus
full local eval, a 5,000-epoch continuation would be weeks on local CPU. This
should not be ground through CPU. The next high-EV action is MLX/Metal-first:
keep the public PR95 PyTorch path as calibration/control, but move long
training, compact renderer sweeps, and byte-value search through MLX-native
or NumPy-portable kernels with archive custody preserved.

## Compact Spine Adapter Smoke

Command:

```bash
.venv/bin/python tools/emit_compact_renderer_spine_adapter.py \
  --family rt_vq_nerv \
  --output-dir /Volumes/VertigoDataTier/pact/compact_spine_adapter_smoke_rt_vq \
  --decoder-blob /Volumes/VertigoDataTier/pact/compact_spine_adapter_smoke_inputs/decoder.bin \
  --latents-blob /Volumes/VertigoDataTier/pact/compact_spine_adapter_smoke_inputs/tokens.bin \
  --codebooks-blob /Volumes/VertigoDataTier/pact/compact_spine_adapter_smoke_inputs/codebooks.bin \
  --selectors-blob /Volumes/VertigoDataTier/pact/compact_spine_adapter_smoke_inputs/selectors.bin \
  --trained-weights-provenance smoke_trained_weights_fixture \
  --trained-latents-provenance smoke_trained_tokens_fixture
```

Evidence:

- Report: `/Volumes/VertigoDataTier/pact/compact_spine_adapter_smoke_rt_vq/rt_vq_nerv_spine_adapter_report.json`
- Spine projection bytes: `4080`
- Promotion blockers: archive/runtime receiver proof missing, full-video
  scorer value-per-byte missing, contest CPU/CUDA exact eval missing

## Next Implementation Step

Build the MLX/Metal-first compact-base runner that trains or resumes
PR95-scale HNeRV/RNeRV/SR-NeRV/BoostNeRV and PVQ/RT-VQ-NeRV candidates under
hard byte ceilings, emits the new spine adapter rows directly from trained
artifacts, then runs full-video scorer-value replay and receiver proof before
any exact gate.
