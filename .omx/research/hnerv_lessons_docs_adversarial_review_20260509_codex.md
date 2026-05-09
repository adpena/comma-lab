# HNeRV lessons docs adversarial review (2026-05-09 Codex)

<!-- generated_at: 2026-05-09T11:20:00Z, author: codex -->
<!-- evidence_grade: documentation_review + source_forensics + contest_cpu_negative; no new score claim -->

## Scope

Review the latest `.omx` / Claude-authored HNeRV and public-PR retrospective
documents, especially:

- `.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md`
- `.omx/research/hnerv_forensics_critical_findings_for_a1a9359d_20260509.md`
- `.omx/research/CLAUDE_md_addition_HNeRV_parity_discipline_20260509.md`
- `.omx/research/t1_hnerv_parity_adversarial_hardening_20260509_codex.md`
- `.omx/research/lane_12_v2_nerv_as_renderer_phase_a_design_20260509.md`
- `.omx/research/a1_inflate_bias_sweep_exact_cpu_review_20260509_codex.md`

This review answers the operator's question: what did the other PR submissions
do that we did not, and why did we reach HNeRV-like ideas earlier without
realizing the score wins?

## Findings

### 1. The main HNeRV lesson is correctly identified but was overcompressed

The docs are right that leaderboard HNeRV did not win from a mysterious new
architecture alone. The win was the full packet discipline:

- RGB renderer, not mask-only slot replacement.
- Score-aware single-video training on `upstream/videos/0.mkv`.
- Eval-roundtrip inside the training step.
- Exact archive/export/runtime contract in the loop.
- Small inflate runtime and byte-consumed no-op proof.
- PR100/101-style bolt-ons applied only after the substrate was exact-evaluable.

The phrase "ONE meta-bug explains 100%" was too strong. I patched
`CLAUDE.md` and the paste-in source ledger to say this is the dominant
representation-lane integration meta-bug, not the whole miss. The full miss
also includes PR #95 source-ingestion failure, CPU-axis blindness, and the
`rgb_to_yuv6` autograd break.

### 2. The missing catastrophic technical detail was scorer-preprocess autograd

The forensics dossier says PR #95/#106 monkey-patched both
`frame_utils.rgb_to_yuv6` and `modules.rgb_to_yuv6` because the challenge
helper is `@torch.no_grad()` / in-place. Without the patch, PoseNet loss does
not reach the decoder. This was present in older research notes as a generic
gradient bug, but it was not elevated in the HNeRV parity section.

I patched the durable CLAUDE lesson from "eval-roundtrip-aware training" to
"eval-roundtrip-aware and differentiable scorer-preprocess training" and added
an explicit requirement for PoseNet/SegNet gradient-reachability checks before
new renderer GPU dispatches.

### 3. Why we got to HNeRV faster but did not realize the wins

We had the representation ingredients early, but not the closed loop that
turns a representation into a contest score:

- Lane 12 targeted masks, while the scorer derives masks from RGB frames.
- Cool-Chic/C3-style experiments reached export/compliance after training,
  not before.
- Training artifacts lived separately from archive builders and inflate
  runtimes, so promising representations did not become packet bytes.
- We optimized and reasoned on the CUDA axis while the public medal positions
  were decided by CPU-axis replay; PR107/A1 were much closer on CPU than CUDA.
- We did not consume PR #95's open training stack during the final race window.
- We over-weighted broad meta-solvers when the public frontier was moved by
  small exact-evaluable code deltas on a verified HNeRV substrate.

The accurate lesson is not "we lacked HNeRV." It is "we lacked HNeRV as a
closed submission-packet compiler with score-gradient, differentiable
preprocess, exact export, and dual-axis eval from byte zero."

### 4. Public PRs show runtime arithmetic is high-leverage but not arbitrary

The binary-forensics claim that PR100 and PR102 share archive bytes is a strong
signal that runtime constants can move the medal band. The A1 bias-correction
sweep adds a necessary adversarial correction:

- A1 baseline / V1 exact Linux x86_64 CPU: `0.192847577437`
- V2 half-magnitude PR101 bias exact Linux x86_64 CPU: `0.194295755690`
- same archive SHA: `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- classification: measured-config regression; PR101's inherited full bias is
  load-bearing for A1, but naive retuning worsens.

Therefore the correct next action is not "try arbitrary channel constants."
It is a constrained same-archive coordinate search around the verified PR101
bias and sidecar scale, with exact CPU/CUDA custody and runtime-smoke closure.

### 5. Lane 12-v2 is aligned with the lesson but still blocked before promotion

The Lane 12-v2 design memo correctly moves from mask logits to full RGB
renderer and declares archive grammar early. The remaining Phase B blockers
are real:

- self-contained contest runtime, not `tac` oracle import;
- differentiable scorer-preprocess gradcheck;
- PR95/PR100 training-stack parity or a documented deliberate deviation;
- exact packet builder and no-op proof before eval dispatch;
- paired `[contest-CUDA]` and `[contest-CPU]` once shippable.

## Required follow-up guards

1. Add a strict or warn-only preflight guard for renderer trainers that load
   SegNet/PoseNet but never verify nonzero gradient from `d_pose` and `d_seg`
   back to renderer output.
2. Extend lane maturity evidence for representation lanes with
   `scorer_preprocess_gradcheck` / `eval_roundtrip_inner_loop` fields.
3. Run a PR95 reproduction or exact source-parity smoke before treating new
   HNeRV-derived substrates as comparable to PR100/101.
4. Keep the A1 runtime-bias line as exact negative evidence: it narrows the
   coordinate-search trust region instead of killing runtime arithmetic.
