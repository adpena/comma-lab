# A2 certified sensitivity source discovery (2026-05-10)

## Verdict

A2 remains blocked. I found local sensitivity maps that are useful diagnostic
signals, but none are promotion-grade `component_sensitivity_v1` inputs. Do
not synthesize a certified manifest from these artifacts.

## Current hard blocker

`reports/a2_certified_sensitivity_blocker_20260510_codex.json` is the current
machine-checkable blocker artifact for
`track1_phase_a2_sensitivity_quant_packet_ladder`.

Stable blocker codes:

- `a2_certified_sensitivity_binding_invalid`
- `a2_component_sensitivity_manifest_reference_missing`
- `a2_sensitivity_artifact_diagnostic_allowed`
- `a2_sensitivity_artifact_metadata_blockers_present`

The current A2 manifest references a stub map SHA
`ac259e314a23477d10bebaa0b89b0dc2fed0c5534c7201b05ded9239340caafb`
with `allow_diagnostic_sensitivity=true`, `metadata_blockers=["is_stub=true",
"tag contains 'stub'"]`, and no `component_sensitivity_v1` manifest reference.

## Tempting local artifacts reviewed

1. `experiments/results/lightning_batch/component_sensitivity_pfp16_a_plus_plus_cuda_fisher_20260501_r2/combined_sensitivity_map.pt`
   - SHA-256:
     `20348c00714bddc930a693bd2efc1dc74cd4a2bc87e4f48317df53fb284b5eb8`
   - `format=tac_score_sensitivity_map_v1`
   - `metadata.evidence_grade=diagnostic_cuda_fisher_proxy`
   - `metadata.sensitivity_source=fisher_proxy`
   - `metadata.official_component_response=false`
   - `metadata.canonical_scorer_path=false`
   - `metadata.promotion_eligible=false`
   - `metadata.promotion_blockers[0].code=fisher_proxy_not_official_component_response`
   - Decision: not certifiable for A2 promotion.

2. `experiments/results/modal_component_sensitivity/pfp16_direct_fd_modal_a10g_20260501_r3_complete_merge_20260501T0929Z/combined_sensitivity_map.pt`
   - SHA-256:
     `99a9cd55b031cdae2c8cb3e7c8444483837104bc45ac63603205e7feee88c456`
   - `format=tac_score_sensitivity_map_v1`
   - `metadata.evidence_grade=diagnostic_cuda_direct_renderer_finite_difference`
   - `metadata.sensitivity_source=direct_renderer_cuda_finite_difference_component_response`
   - `metadata.component_response_path=direct_renderer_tensor_inprocess_scorer`
   - `metadata.official_component_response=false`
   - `metadata.canonical_scorer_path=false`
   - `metadata.promotion_eligible=false`
   - `metadata.promotion_blockers[0].code=not_canonical_inflate_eval_path`
   - Decision: not certifiable for A2 promotion even though it is a full
     finite-difference merge; it did not perturb exact archive bytes through
     `archive.zip -> inflate.sh -> upstream/evaluate.py`.

## What would unblock A2

A promotion-grade A2 binding needs a real `component_sensitivity_v1` manifest
whose component maps and response curves satisfy all of the following:

- `official_component_response=true`
- `canonical_scorer_path=true`
- `promotion_eligible=true`
- no non-empty `promotion_blockers`
- map SHA matches the A2 `inputs.sensitivity_map_sha256`
- `contest_eval` archive bytes and SHA match the exact CUDA baseline archive
- `n_samples=600`
- CUDA scorer evidence and response curves are produced by the canonical
  archive/inflate/evaluate path, not direct renderer tensors or Fisher proxies.

Until that exists, A2 packetization stays `blocked_fail_closed` and
`score_claim=false`.

## Commands run

```bash
.venv/bin/python tools/build_a2_sensitivity_weighted_pr101_packet.py \
  --a2-manifest experiments/results/track1_phase_a2_sensitivity_quant_20260508T154125Z/A2_result.json \
  --output-dir experiments/results/a2_certified_probe_20260510_codex \
  --json-out reports/a2_certified_sensitivity_blocker_20260510_codex.json \
  --variant-limit 1 \
  --require-certified-sensitivity \
  --fail-if-blocked \
  --force

.venv/bin/python - <<'PY'
from pathlib import Path
import torch, hashlib
for p in [
    Path("experiments/results/lightning_batch/component_sensitivity_pfp16_a_plus_plus_cuda_fisher_20260501_r2/combined_sensitivity_map.pt"),
    Path("experiments/results/modal_component_sensitivity/pfp16_direct_fd_modal_a10g_20260501_r3_complete_merge_20260501T0929Z/combined_sensitivity_map.pt"),
]:
    obj = torch.load(p, map_location="cpu", weights_only=False)
    print(p, hashlib.sha256(p.read_bytes()).hexdigest(), obj["metadata"])
PY
```
