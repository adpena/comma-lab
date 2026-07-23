---
schema: codex_findings.v1
date_utc: 2026-07-23
lane_id: regmax_family_probes
axis: "[macOS-CPU advisory]"
research_only: true
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU Linux x86_64] UNMOVED"
main_landing_review_required: true
---

# #581 stale-gate rerun against #583 prerequisite surfaces

## Outcome

The unchanged full-n600 probe runner consumed the SHA-bound #583 manifest and
confirmed all three advertised surfaces are present. One probe reached its
pre-registered terminal comparison; two still fail closed before their
receiver endpoint because the #583 adapter preserves a supplied continuous RGB
or scorer-plane proposal but does not derive RGB from class probabilities or
rank-4 SegNet-head features.

| probe | result against its own pre-registration | exact scope | receipt SHA-256 |
|---|---|---|---|
| sparsemax A/B versus entropy/Cole-Hopf | `BLOCKED_NOT_MEASURED`; the target-side `0.9733309173583984` exact-one-hot fraction reproduced, but `hard_accepts`, exact calls, and same-coder bytes remain null | `INSTANCE:MISSING_TYPED_CLASS_PROBABILITY_TO_RGB_PREIMAGE_PULLBACK` | `04746a083dd47abaa822c2febb507c808107fb1d4eade0bf68158ac5845e0c25` |
| tropical principal representative versus Aurenhammer LP | `FALSIFIED_FORMULATION`; all five frozen prototype cells retain exact identity, but tropical is 137 Brotli-q11 bytes versus Aurenhammer 134 bytes (`+3`), and zero-sum/min-norm is 131 bytes | `FORMULATION:TROPICAL_PRINCIPAL_AS_MINIMUM_BYTE_REPRESENTATIVE_FOR_FIVE_FROZEN_HEAD_STRICT_PROTOTYPE_CELLS_PDW2_BROTLIQ11` | `68af761bbb52e2b72e990e8fc2adb4a4b79ed559cd2aa87c8e2f3e3787b3fd9b` |
| frozen-prototype Hopfield memory-prox pre-step | `BLOCKED_NOT_MEASURED`; the frozen rank-4 bank resolves (rank 4, prototype SHA `5ce0458949acb1cde21022aef7bf642b4491ddb03d1ab66838866b20cb7b162f`), but no typed rank-4-feature-to-RGB pullback exists, so frozen SegNet was not invoked | `INSTANCE:MISSING_TYPED_RANK4_FEATURE_TO_RGB_PREIMAGE_PULLBACK` | `6b08c5da856b390dddeaa3fdddf679528b131ecc34d09c22abc28816a0bb2b52` |

Manifest SHA-256:
`436c255ee15f89309710c828f9e0814d7fd146ef3496037ba2eb3cbdfcc16660`.
Input custody was rechecked for the real n600 fp16 logits
(`41d3ef535f5b5855fe17aab678580114a50309dc48d04948af62c2f563ed3b52`),
hard labels
(`36c6be718916de9b0a62fec0c1229c94e38f84c3313a1fad1357c9a24eef8b68`),
frozen SegNet weights
(`68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`),
and #583 manifest
(`4a8330e79353cfd45dd3c7580c55013a878b6e72fb3e242223c82a0dc03f766f`).

## Premise correction

The a1 row-7 statement that #583 supplied every missing executable input is
only true surface-by-surface. It is false for the two end-to-end compositions:
`matched_continuous_to_uint8_hard_accept` is typed over continuous RGB/scorer
planes, while sparsemax emits five-class probabilities and the memory-prox arm
emits rank-4 quotient features. Treating either as RGB would be a type-invalid
proxy and would violate the pre-registered realized-through-R plus frozen
SegNet endpoint.

This does not invalidate #583 surfaces 1 or 2. It narrows the remaining debt to
the explicit typed pullbacks above. No redesign, fallback, score claim, or
family/paradigm negative is authorized.

## Re-derivation

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/Users/adpena/Projects/pact/.venv/bin/python \
tools/probe_regmax_family.py \
  --logits /Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/teacher_logits_n600/gt_segnet_logits.f16 \
  --labels /Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/targets_n600/gt_segnet_argmax.u8 \
  --prerequisite-dir .omx/research/prereq_surfaces_flush_20260720 \
  --weights /Users/adpena/Projects/pact/upstream/models/segnet.safetensors \
  --output-dir .omx/research/ddm_p581r_stale_gate_probe_rerun_20260723T030100Z \
  --timestamp 2026-07-23T03:01:00Z
```

```bash
PYTHONDONTWRITEBYTECODE=1 \
/Users/adpena/Projects/pact/.venv/bin/python -m pytest -q \
  src/tac/tests/test_probe_regmax_family.py \
  src/tac/boundary_math/tests/test_prereq_surfaces.py
```

Verification: `17 passed`; Ruff clean on the probe/prerequisite source and
tests; `git diff --check` clean.

## Triality and system intelligence

- **DAG:** the sibling FEED records one terminal formulation negative and two
  exact typed-composition blockers.
- **DSL/control:** unchanged and N/A. No probe treatment or launch config was
  changed.
- **Equations:** consumes the existing sparsemax unit-margin law,
  `segnet_head_rank4_linear_flipdist_v1`, and
  `bounded_uint8_resize_preimage_cell_feasibility_v1`; no new law is claimed
  from blocked compositions.
- **Pointer:** exact frontier `0.1910828242` is unchanged. These are local
  advisory receipts, not contest scores.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; a1 audit row
7; the ERM crosswalk and original #581 receipts; #583 prerequisite memo,
manifest, code, and tests; current lane/subagent state; per-arm and broadcast
inboxes; current probe source and focused regression tests.

## MAIN landing requirement

This worktree is not authority. MAIN must review the complete base-to-head
diff, independently verify the #583 adapter input type against both blocked
treatments, rerun the focused tests and command above, and only then merge.
