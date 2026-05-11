# PR103-on-PR106 Modal CPU/CUDA raw-manifest verdict (2026-05-11)

## Result

The paired Modal rerun is now valid for exact CPU/CUDA mechanism analysis.

- Archive SHA-256: `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- Archive bytes: `185578`
- Runtime content tree SHA-256:
  `f2ebe56a408a55b39070f9f86ba77fb11a9b43d83c0e02692f0acc0bf1ff28bb`
- Samples: `600` on both axes
- Analysis artifact:
  `.omx/research/artifacts/cpu_cuda_drift_exact_pr103_pr106_raw_manifest_20260511_codex/analysis.json`

| Axis | Score | Pose | Seg | Raw aggregate SHA-256 |
| --- | ---: | ---: | ---: | --- |
| contest-CPU | `0.229665686265` | `0.000164` | `0.000656` | `1e5fdfa06090922af9d55a54dbce1972709885fa77001278aae564439e14583b` |
| contest-CUDA | `0.208983075582` | `0.0000336` | `0.00067084` | `99141b32678d60bb736fce21eac1f68fce402e627ed44e7fc9a0c9a76b44c1e7` |

CUDA minus CPU score gap: `-0.0206826106828098`.

## Verdict

This exact pair falsifies any broad rule that HNeRV-family packets always score
better on CPU than CUDA. This packet scores much better on CUDA.

The raw inflated outputs differ even though the archive and runtime tree match.
Therefore the dominant gap is already upstream of scorer evaluation:
device-dependent inflate math in the runtime is part of the mechanism.

The immediate suspect surface is
`submissions/pr103_pr106_final_runtime/inflate.py`:

- `select_inflate_device()` chooses CUDA when available;
- the HNeRV decoder runs `torch.sin`, `torch.sigmoid`, and convolution on the
  selected device;
- final resizing uses `torch.nn.functional.interpolate(..., mode="bicubic")`
  on the selected device before rounding to `uint8`.

This does not prove scorer kernels are irrelevant. It proves the first observed
CPU/CUDA divergence is before the scorer and must be isolated before training
or loss-design conclusions are drawn.

## Score-lowering implication

Do not compress CPU/CUDA behavior into one scalar. Every serious HNeRV-style
candidate needs axis-separated custody:

1. default contest-CUDA inflate/eval;
2. default Linux x86_64 contest-CPU inflate/eval;
3. forced-CPU-inflate then CUDA scorer, if contest-compliant as a diagnostic;
4. forced-CUDA-inflate then CPU scorer, if available as a diagnostic;
5. raw-output aggregate SHA and per-video raw SHA for every run.

The score-lowering opportunity is not to assume CPU or CUDA is better. It is to
make device-dependent inflate an explicit optimization axis and then search for
runtime math choices whose raw frames land in a better SegNet/PoseNet basin.

## Claim status

This is a mechanism diagnostic, not a promotable score claim. The CPU claim was
closed as `completed_modal_cpu_auth_eval_recovered`; the CUDA claim was closed
as `completed_modal_auth_eval_recovered`.

Remaining live dispatch at this checkpoint: T1 Ballé Modal Phase 1
`t1_balle_modal_phase1_ab2d0f6_20260510T1437Z` is still pending.
