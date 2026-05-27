# Codex Finding: FECa selector reparameterization scale64 alpha1 exact anchor

Date: 2026-05-27T23:16Z
Agent: codex

## Summary

The queue-owned FECa selector-stream context recode found a larger P11
selector entropy win than the earlier scale256/alpha2 probe. A wide bounded
sweep selected scale=64, alpha=1 and reduced the FP11 FECa selector payload
from 236 bytes to 220 bytes. The byte-closed archive is 16 bytes smaller than
the source archive while preserving decoded selector codes, source payload,
DQS1 tail bytes, and full-frame shell inflate output.

This is a rate-only archive win at entropy position P11
(`selector_stream`, `at_entropy_coder_symbol_coding`). It is same-position
subadditive with other selector-codec transforms and should be composed after
upstream distribution shaping and before post-coder container cleanup.

## Source And Candidate

- source submission dir: `experiments/results/v14_v2_dqs1_plus_fec10_substituted_20260526T023000Z/submission_dir`
- source archive sha256: `0a3abfe645c4fac0df9ea89237f25dd9bfc6b2471b897c36d7437795d27d1403`
- source archive bytes: `178546`
- candidate artifact root: `/Volumes/VertigoDataTier/experiments/results/feca_selector_reparameterized_wide_sweep_20260527Tlocal`
- candidate archive: `/Volumes/VertigoDataTier/experiments/results/feca_selector_reparameterized_wide_sweep_20260527Tlocal/submission_dir/archive.zip`
- candidate archive sha256: `18e3155fbbbe9ab23e1c21bc0d99ba8d18657a71c3129fc5ff9e0405b67d1669`
- candidate archive bytes: `178530`
- saved bytes: `16`

## Proofs

- manifest: `/Volumes/VertigoDataTier/experiments/results/feca_selector_reparameterized_wide_sweep_20260527Tlocal/feca_selector_reparameterization_manifest.json`
- full-frame shell inflate parity proof: `/Volumes/VertigoDataTier/experiments/results/feca_selector_reparameterized_wide_sweep_20260527Tlocal/feca_selector_full_frame_inflate_parity_proof.json`
- full-frame parity raw sha256: `00b479229c97ede3e776846297269f7785285702b8dbf3e5dccc733557da605a`
- strict proof blockers: none
- readiness blocker after local proof: `candidate_requires_exact_auth_eval_before_promotion`

## Exact CPU Anchor

- axis: `[contest-CPU]`
- final runtime-stable Modal call id: `fc-01KSNW4ZWK52405FW0DR31DQKW`
- final runtime-stable Modal output dir: `experiments/results/modal_auth_eval_cpu/feca_selector_reparameterized_runtime_stable_scale64_alpha1_20260527T2326Z_cpu`
- final runtime-stable Modal run URL: `https://modal.com/apps/adpena/main/ap-lyxTOVOcQTJuGkG4iSy8Cm`
- final runtime-stable score: `0.1920089730474962`
- final runtime-stable avg_segnet_dist: `0.00055978`
- final runtime-stable avg_posenet_dist: `0.00002943`
- final runtime-stable archive bytes charged by eval: `178530`
- final runtime-stable auto-bound Modal runtime tree sha256: `f2044ce5956b928ec2322b490d4df5d0cfd334831d24508cffcaa4c858151da9`
- final runtime-stable auto-bound Modal runtime content tree sha256: `a75ea061d06979e12a2b842b9dbc234f7fd09e04c0877211538f7409ee87077d`

The initial exact CPU anchor on the pre-runtime-stabilized candidate also
passed and is retained as a diagnostic:

- Modal call id: `fc-01KSNV5NYCEA7C5ZPV6205DV0D`
- Modal output dir: `experiments/results/modal_auth_eval_cpu/feca_selector_reparameterized_scale64_alpha1_20260527T2310Z_cpu`
- Modal run URL: `https://modal.com/apps/adpena/main/ap-46Ud21Rj7VzvU7Orx2qfVD`
- score: `0.1920099730474962`
- avg_segnet_dist: `0.00055979`
- avg_posenet_dist: `0.00002943`
- archive bytes charged by eval: `178530`

The earlier 5-byte scale256/alpha2 exact CPU anchor is superseded by this
16-byte scale64/alpha1 anchor for the same source archive. The 5-byte result
remains useful as a consistency check but is not the best known point in this
operator family.

## Engineering Follow-Through

Landed code hardening:

- invalid parameter cells are recorded as `roundtrip_failed` instead of aborting
  the sweep;
- FECa manifests/proofs now carry the shared canonical entropy-position
  descriptor;
- candidate runtime trees no longer copy or generate Python bytecode caches;
- Modal CPU/CUDA dispatchers accept `--expected-runtime-tree-sha256 auto`,
  binding the Modal-projected runtime tree before claim/spawn and avoiding the
  fail-copy-rerun loop.

Runtime-stable follow-up artifact:

- artifact root: `/Volumes/VertigoDataTier/experiments/results/feca_selector_reparameterized_runtime_stable_20260527Tlocal`
- archive sha256: `18e3155fbbbe9ab23e1c21bc0d99ba8d18657a71c3129fc5ff9e0405b67d1669`
- local runtime tree sha256: `95069c6536545598012211d6968c8f3616e33104c47547a8d9969af8d8039013`
- strict full-frame parity proof: `/Volumes/VertigoDataTier/experiments/results/feca_selector_reparameterized_runtime_stable_20260527Tlocal/feca_selector_full_frame_inflate_parity_proof.json`
- right-side proof runtime file count after inflate: `10`
- right-side proof runtime tree sha256: `3e0db9c1705d6a8ec94fab477f0ae2f133e8d4bb351df4d65adfe636f991e31d`
- right-side patched `inflate.sh` sha256: `0878a7730e5caa2d6e0044ef3378c5772c67da0499958629ebca5285cfd503ab`

Runtime-stable exact CPU anchor recovered:

- axis: `[contest-CPU]`
- Modal call id: `fc-01KSNW4ZWK52405FW0DR31DQKW`
- Modal output dir: `experiments/results/modal_auth_eval_cpu/feca_selector_reparameterized_runtime_stable_scale64_alpha1_20260527T2326Z_cpu`
- Modal run URL: `https://modal.com/apps/adpena/main/ap-lyxTOVOcQTJuGkG4iSy8Cm`
- auto-bound Modal runtime tree sha256: `f2044ce5956b928ec2322b490d4df5d0cfd334831d24508cffcaa4c858151da9`
- auto-bound Modal runtime content tree sha256: `a75ea061d06979e12a2b842b9dbc234f7fd09e04c0877211538f7409ee87077d`
- status: recovered
- score: `0.1920089730474962`
- avg_segnet_dist: `0.00055978`
- avg_posenet_dist: `0.00002943`

## Next Integration

Queue/autopilot consumers should treat this as a confirmed P11 same-position
selector-codec win and use it as a parent for:

- upstream P18/P19 scorer-response waterfill followed by P11 selector recode;
- selector-context families beyond scale/alpha, especially Markov/order models;
- post-coder rebrotli/repack cleanup after selector payload stabilization;
- runtime-stable candidate artifact promotion using the recovered exact CPU
  anchor and strict full-frame parity proof.
