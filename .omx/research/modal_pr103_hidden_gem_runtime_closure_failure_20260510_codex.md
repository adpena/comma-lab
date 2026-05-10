# Modal PR103 Hidden-Gem Runtime Closure Failure (2026-05-10)

## Scope

This ledger records the failed Modal T4 exact auth-eval attempt for
`pr103_ac_merged_range_drop_u32_at_160824_20260510_agent`.

This is not a score result and not a lane negative. The candidate failed before
`contest_auth_eval.json` was produced because the provider image was missing a
runtime entropy-decoder dependency.

## Custody

- Lane claim: `pr103_ac_hidden_gem`
- Dispatch job: `pr103_ac_hidden_gem_exact_cuda_modal_20260510T170420Z`
- Modal run: `ap-hIqP29g1Qx1jPRWapcERLW`
- Archive:
  `experiments/results/hnerv_hidden_gem_pr103_ac_candidate_20260510_agent/release_surface/archive.zip`
- Archive bytes: `185574`
- Archive SHA-256:
  `8274e88c0ab1d26a06470a0730d17fe004556afa564460cf1c05624ff6060278`
- Runtime tree SHA-256:
  `9f0602f18ed1c71d2e3b2f8fd38e8b992c0e8fff9ec1aef2520b6597b87a6308`
- Modal artifacts:
  `experiments/results/modal_auth_eval/pr103_ac_hidden_gem_exact_cuda_modal_20260510T170420Z/`

## Failure

Inflate failed after `3.3s` with:

```text
RuntimeClosureError: missing runtime dependencies: constriction: No module named 'constriction'
```

`contest_auth_eval.json` was not produced, so there is no score, no component
distortion, and no rank/promotion authority.

## Bit/Byte/Layer Classification

- Mutation: remove one 4-byte word at payload offset `160824`.
- Mutated section: runtime-consumed `merged_ac` range stream.
- Header update: PR106 packed decoder length changed from `169617` to
  `169613`.
- Charged archive delta: `-4` bytes.
- Rate-only score delta if distortion were unchanged:
  `-0.0000026634358124886856`.
- Decode proof: generated runtime and TAC closure decode the candidate; latents
  are exact; three decoded tensors change.
- Changed tensors: `blocks.4.weight`, `blocks.5.weight`, `refine.0.bias`.

Adversarial classification: useful byte/layer deconstruction signal, blocked
by provider/runtime dependency closure. The effect size is too small to justify
promotion unless CUDA distortion is neutral or positive.

## Fix Landed

The repo already declares `constriction>=0.4,<0.5` and `pyppmd>=1.3,<2.0` as
hard runtime dependencies in `pyproject.toml`. The Modal CUDA/CPU auth-eval
images were missing those hard entropy-runtime dependencies. The provider image
contract was updated so Modal exact eval mirrors the repo runtime dependency
surface.

## Next Action

Rerun `pr103_ac_hidden_gem` exact CUDA only after the dependency fix is
committed and gates pass. If it scores worse, classify it as a measured
4-byte range-stream perturbation negative, not as a broad arithmetic-coder
negative. If it scores equal or better, use it as a local proof that bitstream
word-level range-stream edits can move the score surface and launch a
CUDA-calibrated nearby offset search.
