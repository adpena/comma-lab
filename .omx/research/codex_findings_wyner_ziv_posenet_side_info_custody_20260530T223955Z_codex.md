# Codex Findings: Wyner-Ziv PoseNet Side-Info Custody

- UTC: 2026-05-30T22:39:55Z
- Agent: Codex
- Scope: adversarial review of `.omx/research/wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_design_20260530.md`, the registered canonical equation, and the cathedral consumer.
- Score authority: none. This is a premise/custody hardening; all outputs remain `score_claim=false`, `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Finding

The design memo and implementation treated `PoseNet(frame_pair)` as free decoder-side Wyner-Ziv side information because `upstream/evaluate.py` loads PoseNet after inflate. That violates the strict scorer boundary for the submission pipeline: inflate/runtime code must not load or depend on PoseNet/SegNet, and any decoder side information must be charged inside `archive.zip` or derived from fixed contest inputs without scorer state.

## Fix Landed

- `tac.canonical_equations.wyner_ziv_decoder_side_posenet_side_information` now models PoseNet-conditioned coding as legal only when side information is `archive_charged` or `fixed_contest_input`.
- The prediction helper subtracts `side_info_charged_bytes` from gross savings and emits blockers/zero net savings for `scorer_runtime_free` and `compress_time_advisory_only`.
- `tac.cathedral_consumers.wyner_ziv_posenet_side_information_consumer` now fails closed on generic Wyner-Ziv/PoseNet tokens unless the candidate proves archive-bound or fixed-input side-info custody.
- Regression tests cover custody-required matching, free-scorer-runtime rejection, charged-byte subtraction, and strict-scorer-compatible domain metadata.

## Required Follow-Up

- Treat the historical design memo as superseded at the custody premise only; the Wyner-Ziv theorem and encoder-side PoseNet conditioning remain valid.
- Any substrate adopting this equation must emit `side_info_delivery_mode` plus charged-byte or fixed-input proof before acquisition can treat it as a matching Wyner-Ziv PoseNet side-info candidate.
- Do not promote, rank, kill, or exact-dispatch from this equation alone; exact axis promotion still requires byte-closed archive, receiver proof, and contest CPU/CUDA authority.
