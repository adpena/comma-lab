# PR106 latent sidecar exact CUDA positive — Codex

Date: 2026-05-11

## Result

The PR106 latent sidecar materialized from the Kaggle score table produced a
valid A++ Modal T4 exact CUDA improvement.

- Lane: `lane_pr106_latent_sidecar`
- Job: `pr106_latent_sidecar_modal_exact_cuda_20260511T150517Z`
- Modal call id: `fc-01KRBS58Z6EW2A7A3FXJXB3N6S`
- Archive:
  `experiments/results/pr106_latent_sidecar_from_kaggle_table_20260511_codex/sidecar_archive.zip`
- Archive bytes: `186808`
- Archive SHA-256:
  `947b85e8a69db295d4dcf80b0b528639c47839f40f289a2c05b70a2064658b48`
- Runtime content tree SHA-256:
  `55989d263d4eb36e29c59998f87e9bbd432e69fab80ea1cc704c442dd083012e`
- Inflated raw aggregate SHA-256:
  `235752e9c7e4d4cc447ec1973f48f3be5135b8db36cc278ce2efaddc07badbe4`
- Evidence grade: `A++ contest T4`
- Promotion eligible after adjudication: `true`
- Lane status: `IN_PREDICTED_BAND`

Exact score:

```text
score_recomputed_from_components = 0.20739428085403283
avg_segnet_dist = 0.00064893
avg_posenet_dist = 0.00003281
archive_size_bytes = 186808
```

Against the previous HNeRV CUDA floor
`0.20898105277982337` at `185578` bytes:

```text
score_delta = -0.00158677192579054
seg_term_delta = -0.0021889999999999965
pose_term_delta = -0.00021677192579050777
rate_term_delta = +0.0008190065123402646
```

Interpretation: the sidecar pays `+1230` bytes but more than pays for itself
through SegNet and PoseNet distortion reduction. This validates the per-pair
latent sidecar as a real score-lowering axis, not just a Kaggle/proxy artifact.

## Custody

Raw recovered artifact directory:

```text
experiments/results/modal_auth_eval/pr106_latent_sidecar_20260511T150517Z
```

Key artifact SHA-256s:

```text
contest_auth_eval.json: 26b94f092e18a35f056cde6b8a66a4da189b9f43fc7e35637f2d10e4aca0829b
contest_auth_eval.adjudicated.json: see result copy in artifact directory
inflated_outputs_manifest.json: 0a876e0ee6fd78aa9d9e870ccaf2d40cfcbac7cbeb52813c134df9c5682a5eb9
modal_cuda_auth_eval_result.json: ffa364b7b48e9147c57da00f24577f0f3a59d51b80b4ce81c09c74e4bb214904
provenance.json: d8b5c1885eca2904e51a51a21efcdad320098ba066dcf32f2e1f775b7d436cb9
```

Canonical eval path:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda
```

Adjudication command:

```bash
.venv/bin/python scripts/adjudicate_contest_auth_eval.py \
  --contest-json experiments/results/modal_auth_eval/pr106_latent_sidecar_20260511T150517Z/contest_auth_eval.json \
  --provenance experiments/results/modal_auth_eval/pr106_latent_sidecar_20260511T150517Z/provenance.json \
  --archive experiments/results/pr106_latent_sidecar_from_kaggle_table_20260511_codex/sidecar_archive.zip \
  --result-copy experiments/results/modal_auth_eval/pr106_latent_sidecar_20260511T150517Z/contest_auth_eval.adjudicated.json \
  --baseline-score 0.2089810755823297 \
  --baseline-archive-bytes 185578 \
  --predicted-band 0.205 0.208 \
  --regression-threshold 0.25 \
  --required-device cuda \
  --required-samples 600 \
  --max-sane-score 1.0
```

Adjudicator returned:

```text
PROMOTION_ELIGIBLE=1
EVIDENCE_GRADE=A++ contest T4
REGRESSION_TRIGGERED=0
SANE_SCORE_GATE_TRIGGERED=0
COMPONENT_GATE_TRIGGERED=0
SOURCE_MANIFEST_CLOSURE_GATE_TRIGGERED=0
```

## Next score-lowering moves

1. Re-run/recover the active T1 Ballé Modal job; do not duplicate the active
   claim.
2. Stack PR106 latent sidecar with yshift/LRL1 only after each sidechannel has
   its own score-table builder and exact CUDA gate; the latent result validates
   the class, not arbitrary stacked sidecars.
3. Use the sidecar's per-pair improvements as a saliency prior for the next
   PR106 sidechannel search: allocate bytes to pairs/classes where the exact
   score table showed strict improvement.
4. Run the device-axis raw-output matrix for PR106-sidecar versus PR103-on-PR106
   to understand whether CPU/CUDA drift can be exploited without breaking the
   now-positive CUDA path.
5. Update frontier ranking surfaces so the new A++ T4 candidate is visible to
   operator briefing and exact-ready planning without relying on chat state.
