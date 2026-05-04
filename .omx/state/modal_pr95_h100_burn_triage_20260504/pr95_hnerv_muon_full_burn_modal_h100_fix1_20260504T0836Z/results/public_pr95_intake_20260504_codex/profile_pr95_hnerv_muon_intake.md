# PR95 HNeRV/Muon Static Intake

## Archive Anatomy

- archive: `experiments/results/public_pr95_intake_20260504_codex/archive.zip`
- bytes: `178417`
- sha256: `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`
- member: `0.bin` stored bytes `178309`
- rate component at contest denominator: `0.118800556839`

## Blob Sections

- `meta_json_brotli`: compressed `80`, uncompressed `80`, sha256 `49385dd99c02228dd0ed7649586fe5026940573a5675911ee7cb24ab2e8bdca7`
- `decoder_state_int8_brotli`: compressed `162349`, uncompressed `230048`, sha256 `fbeaf706e9d8c38cf39fd36b7f6c6312b66eae6c9a4f25a99708d8fbb3cdaf2e`
- `latents_delta_uint8_brotli`: compressed `15868`, uncompressed `33720`, sha256 `863634aba9a956d13eab8708133239ee861bb797cbad4ff2f39a46939987c709`

## Parameter And Latent Counts

- decoder tensors: `28`
- decoder params: `228958`
- Muon partition params: `177156`
- AdamW decoder partition params: `51802`
- latent matrix: `600 x 28`
- latent hi-byte nonzero fraction: `0.004345238095`

## Score-Term Math

- provided seg: `0.00061212`
- provided pose: `3.494e-05`
- recomputed seg component: `0.061212000000`
- recomputed pose component: `0.018692244381`
- recomputed rate component: `0.118800556839`
- recomputed total: `0.198704801220`
- evidence status: external/static only, not a score claim

## Training And Optimizer Stages

- `stage1_v328_ce`: epochs `3000`, loss `ce`, AdamW lr `0.001`, Muon `False`, QAT `False`, C1a lambda `0.0`, sigma `0.2`
- `stage2_v331_softplus`: epochs `5650`, loss `tau_softplus`, AdamW lr `0.001`, Muon `False`, QAT `False`, C1a lambda `0.0`, sigma `0.2`
- `stage3_v332_smooth`: epochs `1500`, loss `smooth_disagreement`, AdamW lr `0.0001`, Muon `False`, QAT `False`, C1a lambda `0.0`, sigma `0.2`
- `stage4_v332_qat`: epochs `500`, loss `smooth_disagreement`, AdamW lr `0.0001`, Muon `False`, QAT `True`, C1a lambda `0.0`, sigma `0.2`
- `stage5_c1a_l7`: epochs `9000`, loss `l7_softplus`, AdamW lr `3e-05`, Muon `False`, QAT `True`, C1a lambda `0.01`, sigma `0.2`
- `stage6_lambda_sweep`: epochs `2000`, loss `l7_softplus`, AdamW lr `3e-05`, Muon `False`, QAT `True`, C1a lambda `0.02`, sigma `0.2`
- `stage7_sigma_sweep`: epochs `3000`, loss `l7_softplus`, AdamW lr `3e-05`, Muon `False`, QAT `True`, C1a lambda `0.02`, sigma `0.1`
- `stage8_muon_finetune`: epochs `5000`, loss `l7_softplus`, AdamW lr `1e-05`, Muon `True`, QAT `True`, C1a lambda `0.02`, sigma `0.1`

## Dispatch Readiness

- ready_for_dispatch: `False`
- fail_closed: `True`
- required: Replay exact eval through archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA before any PR95 score claim.
- required: Record contest_auth_eval.json, archive SHA-256, archive bytes, runtime tree hash, hardware, sample count, and recomputed score.
- required: Owned retraining needs explicit manifest/checkpoint custody for every stage: source SHA, seed, stage config, checkpoint path, checkpoint SHA-256, optimizer state policy, and final archive builder provenance.
- blocked: Static PR95 public intake is not promotable score evidence.
- blocked: README/body score inputs remain external until replayed under our exact CUDA auth eval custody.
- blocked: Any HNeRV retrain without checkpoint and manifest custody is non-promotable replay work.

## Immediate Improvement Hypotheses

1. **RAFT/ego-motion/foveation latent bases**: PR95 stores 600x28 free per-frame-pair latents. Replace part of that table with charged coefficients over existing RAFT-like flow, ego-motion, and foveation bases so retraining searches a lower-dimensional, camera-aware latent manifold instead of replaying unconstrained latent memorization.
2. **Cool-Chic/C3/wavelet residual bases**: HNeRV owns the coarse neural reconstruction, but the current archive has no explicit residual basis. Attach a tiny charged residual program over Cool-Chic, C3, or wavelet atoms for systematic SegNet/PoseNet residuals that are too local for global latent movement.
3. **Fridrich/Lagrangian hard-pair weighting**: The eight-stage curriculum already changes seg losses, C1a strength, sigma, and Muon scheduling. Use exact component traces to drive hard-pair Lagrangian weights, selecting which frame pairs deserve latent movement, residual atoms, or entropy budget instead of uniform stage replay.
4. **engineered corrections and pixel-diff sparse atoms**: HNeRV frame outputs can expose sparse pixel/class disagreements after inflate. Encode deterministic sparse corrections only where pixel-diff atoms clear byte-normalized component benefit, using the HNeRV output as the base representation.
5. **self-compression entropy objectives**: Decoder section is 162,349 compressed bytes and the latent hi-byte nonzero fraction is 0.004345. Extend C1a from a weight-shaping regularizer into a measured coder-aware objective over decoder tensors, latent deltas, and optional residual atoms.
6. **decoder entropy and section recode**: Decoder section is 162,349 compressed bytes; source notes a prior hybrid categorical coder beat pure brotli by about 217 bytes. Generalize that into a deterministic per-tensor coder decision table and keep the inflate dependency surface auditable.
