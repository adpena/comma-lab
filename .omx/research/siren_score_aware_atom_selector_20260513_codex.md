# SIREN score-aware residual atom selector hardening - 2026-05-13

## Scope

Lane family: PR106/HLM1 + SIREN residual sidecar.

This tranche pivots SIREN from a standalone full-frame coordinate-MLP dispatch
candidate to a byte-closed residual atom family that can compete inside the
existing PacketIR/sidecar compiler by score movement per charged byte.

## Why

The exact frontier gap is distortion-dominated, not a small ZIP grammar issue.
At the contest rate slope `25 / 37,545,489`, a 2048 byte sidecar costs about
`0.00136` score. A SIREN residual sidecar must therefore buy real SegNet/PoseNet
movement, not just reduce RGB L2 residual energy.

Independent research agents converged on the same route:

- standalone full-frame SIREN is rate-expensive and not credible as the next
  sub-0.19 spend without a byte-closed empirical anchor;
- SIREN/FINER/WIRE-style functions remain valuable as sparse, local residual
  atom families if selected by scorer saliency or finite-difference response;
- the next implementation surface is the existing PR106 SIREN materializer and
  sparse PacketIR runtime, not a new provider script or duplicate trainer.

## Literature Hooks

- SIREN: periodic activations for implicit neural representations and signal
  derivatives. Source: https://arxiv.org/abs/2006.09661
- Fourier features: coordinate maps tune spectral bias. Source:
  https://arxiv.org/abs/2006.10739
- FINER: variable-periodic activations tune supported frequency sets. Source:
  https://arxiv.org/abs/2312.02434
- WIRE: Gabor/wavelet INR activation, space-frequency localized. Source:
  https://arxiv.org/abs/2301.05187
- NeRV/HNeRV/HiNeRV/FFNeRV: video INR evidence favors frame-wise/hybrid and
  flow/hierarchical structures over pure per-pixel coordinate MLP for video
  compression. Sources:
  https://arxiv.org/abs/2110.13903,
  https://arxiv.org/abs/2304.02633,
  https://arxiv.org/abs/2306.09818,
  https://arxiv.org/abs/2212.12294
- COIN++: compressed modulations over a shared INR base, relevant to future
  codebook/modulation work but not a standalone immediate dispatch. Source:
  https://arxiv.org/abs/2201.12904

## Landed Change

`tools/materialize_siren_residual_pr106_sidecar.py` now accepts
`--saliency-map-npy` in `--residual-mode l2_encoded`. The saliency map is a
precomputed scorer/surrogate guidance tensor with one of these shapes:

- `(T, 874, 1164)`
- `(T, 874, 1164, 1)`
- `(T, 874, 1164, 3)`

The encoder normalizes saliency per frame, applies optional
`--saliency-power` and `--saliency-floor`, weights `gt - decoded` before the
FFT, and then reuses the existing dense/sparse SIREN residual wire formats.

Inflate remains scorer-free. The manifest records saliency SHA, shape, dtype,
and proxy status fields. It still emits `score_claim=false`,
`promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`; exact
CUDA/CPU eval is the only score authority.

## Tests Added

`test_siren_l2_encoded_saliency_map_steers_score_relevant_frame` proves a
two-frame equal-L2 residual chooses the saliency-marked frame when only one
15-byte coefficient can fit. This guards the core anti-local-minimum behavior:
byte allocation follows scorer relevance, not raw energy.

Existing dense/sparse no-op and runtime-consumption tests still cover the
wire-format path.

## Immediate Build Path

1. Produce or locate matched raw custody for HLM1/PR106 decoded outputs and
   ground truth under the same axis/runtime contract.
2. Generate a scorer/surrogate saliency `.npy` for the same frame count. The
   first acceptable source is a committed distilled-scorer saliency map tagged
   proxy-only; exact scorer finite-difference response is stronger.
3. Build candidate ladders:

   ```bash
   .venv/bin/python tools/materialize_siren_residual_pr106_sidecar.py \
     --pr106-archive <archive.zip> \
     --output-dir experiments/results/siren_saliency_<budget> \
     --residual-mode l2_encoded \
     --decoded-raw <decoded.raw> \
     --gt-raw <gt.raw> \
     --saliency-map-npy <saliency.npy> \
     --n-frames 1200 \
     --byte-budget <512|1024|2048|4096|8192> \
     --max-k <k> \
     --encoding sparse \
     --decoded-axis contest_cuda \
     --decoded-inflate-device cuda \
     --decoded-runtime-tree-sha256 <sha>
   ```

4. For every candidate: prove residual bytes are runtime-consumed, record
   archive SHA/bytes, and run exact CUDA only after lane claim and readiness
   gate.

## Kill Criteria

- If a 2 KB saliency-guided ladder cannot beat its rate cost by at least
  `0.00136` on exact CUDA, the measured saliency source is not promotable.
- If SegNet improves while PoseNet collapses, narrow to SegNet-boundary atoms
  and rerun; do not kill SIREN/WIRE/FINER as a family.
- If L2 and saliency rankings choose the same atoms across budgets, the current
  saliency map is not adding signal; upgrade to finite-difference response.

## Next Tranche

The highest-EV next tranche is not a pure SIREN Modal dispatch. It is:

1. scorer/surrogate saliency map production for HLM1/PR106 raw outputs;
2. SIREN/FINER/WIRE atom-family disambiguator over the same byte budget;
3. exact CUDA ladder only for byte-closed packets whose manifest proves charged
   residual bytes are consumed by inflate.
