# HDM4 HLM1 Exact CUDA Runtime-Enforced Result (2026-05-13)

## Summary

HNeRV PR106 HDM4 + HLM1 fixed-latent recode was rerun through canonical Modal
T4 exact auth eval after the Modal wrapper was hardened to pass
`--expected-runtime-tree-sha256` into `experiments/contest_auth_eval.py`.

This result is a byte-rate-only score lowering on the PR106 HDM4 substrate. It
does not claim global frontier or submission readiness by itself.

## Exact Result

- axis: `[contest-CUDA]`
- evidence grade from auth eval: `contest-CUDA`
- adjudication grade: `A++ contest T4`
- archive:
  `experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/pr106_r2_hdm4_exact_cuda_hlm1_latent_candidate.zip`
- archive bytes: `186423`
- archive SHA-256:
  `8801845d5099b957898fb6c6e58625bfb4cc065085ed2e3154c2cbc702dc91e0`
- source HDM4 bytes: `186492`
- source HDM4 SHA-256:
  `218ae16f3f13b722e9752d698667ed8770151e40d44b5756c0ebbccb7682825f`
- score recomputed from components: `0.20638030907530963`
- displayed rounded score: `0.21`
- HDM4 baseline score used for adjudication: `0.20642625334307507`
- exact score delta vs HDM4 baseline: `-0.000045944267765440916`
- PoseNet distance: `0.00003236`
- SegNet distance: `0.0006426`
- samples: `600`

Manual recompute:

```text
100 * 0.0006426
+ sqrt(10 * 0.00003236)
+ 25 * 186423 / 37545489
= 0.20638030907530963
```

## Runtime Custody

- Modal call id: `fc-01KRGPGJNJ0BYHKW80GZPARMW8`
- Modal app: `https://modal.com/apps/adpena/main/ap-2OAqQAFgHdHRcf7c68gQCh`
- output directory:
  `experiments/results/modal_auth_eval/hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513`
- wrapper commit recorded by remote provenance:
  `2b250d7abaa78a54742d5cd2b511e723eb403df7`
- expected runtime tree SHA-256:
  `de844bdf4170b40e7ca2c94cc546c039b49704f6d99c730d4318803eed81aab9`
- observed runtime tree SHA-256:
  `de844bdf4170b40e7ca2c94cc546c039b49704f6d99c730d4318803eed81aab9`
- observed runtime content tree SHA-256:
  `e05e79416feed12996efce6e23cc6d21cbc276ff6221c7a2a4eea8dd2d4d2161`
- inflated raw aggregate SHA-256:
  `5f65c70f59c78e5a4394dc062fe750cf721619f6d67790c4844d52f14d248993`
- inflated raw total bytes: `3662409600`
- terminal dispatch claim:
  `completed_contest_cuda_modal_auth_eval_recovered`

The enforced command in `modal_cuda_auth_eval_validation.json` includes:

```text
--expected-runtime-tree-sha256 de844bdf4170b40e7ca2c94cc546c039b49704f6d99c730d4318803eed81aab9
```

## Adversarial Review Closure

- Runtime hash was previously advisory in Modal claim notes only; fixed in
  `experiments/modal_auth_eval.py` and `experiments/modal_auth_eval_cpu.py`.
- HLM1 packet refresh no longer self-injects `--operator-approved-exact-cuda`.
- HLM1 packet source-tree custody now records `provider_dispatch_source_tree_custody_v2`,
  including untracked content hashes when present.
- Static release wording no longer claims SegNet/PoseNet invariance from parser
  parity; exact auth eval is the score authority.
- Full preflight after the hardening patch: `ALL 31 PREFLIGHT CHECKS PASSED`,
  `real 2.39`.

## Classification

Legitimate exact `[contest-CUDA]` byte-rate score lowering on the measured HDM4
configuration. The result is in predicted band and adjudicates as A++ contest
T4, but remains a candidate packet pending any operator submission decision and
does not resolve the separate `[contest-CPU]` axis.

## Durable Review Packet

- machine-readable result review:
  `.omx/research/hdm4_hlm1_exact_cuda_result_review_20260513_codex.json`
- autopilot evidence row:
  `reports/hdm4_hlm1_exact_cuda_evidence_row_20260513_codex.json`
- continual-learning append:
  `reports/cathedral_autopilot_evidence.jsonl`
- review status: `exact_cuda_result_reviewed`
- failure class: `not_negative_against_supplied_baseline`
- dispatch-claim status:
  `completed_contest_cuda_modal_auth_eval_recovered`
- promotion flags remain fail-closed:
  `score_claim=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`

This closes the signal-preservation loop for the exact HLM1 CUDA result without
turning the candidate into a submission or score-promotion claim.

## Next Work

1. Run true `[contest-CPU]` closure on the GHA Linux x86_64 path if public
   leaderboard reproduction is needed. Modal CPU already measured the archive,
   but the posterior rejected it as a non-1:1 `[contest-CPU]` substrate.
2. Continue HDM5 semantic/structured recode work on the remaining high-byte
   sections; generic Brotli is near saturation.
3. Generalize the enforced-runtime-hash pattern to every provider exact-eval
   packet and keep packet refresh commands approval-free by default.
