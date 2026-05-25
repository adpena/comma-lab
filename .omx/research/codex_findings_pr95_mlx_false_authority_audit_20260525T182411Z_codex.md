# PR95 MLX False-Authority Audit

UTC: 2026-05-25T18:24:11Z
Agent: codex
Evidence axes: `[macOS-CPU advisory]`, `[macOS-MLX research-signal]`

## Verdict

There are similar fake-shaped implementations. The issue is not malicious fake
code; it is partial/proxy work with names or fields that can be over-consumed as
1:1 PR95/HNeRV reproduction, full-frame parity, scorer parity, or score
authority.

This pass landed guards for the highest-risk active case:

- `tools/prove_shell_inflate_parity.py` now records `parity_scope_kind`,
  `contest_full_sample_claim`, and `contest_full_sample_parity_claim`.
- DFL1 queue postconditions now assert the scoped parity proof is
  `declared_file_list`, not complete contest-sample authority.
- The shell parity verifier refuses missing scope metadata for DFL1 exact
  readiness.
- `score_claim`, `promotion_eligible`, `rank_or_kill_eligible`,
  `ready_for_exact_eval_dispatch`, and `promotable` are all false in the proof.
- The proof runner now invokes non-executable `inflate.sh` through `bash`, passes
  absolute archive/output/file-list arguments, and preserves venv Python shims
  when adding `python_bin` to `PATH`.

## Current PR95 Proof

Artifact:

`.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/pr95_declared_file_list_shell_parity_proof_v4/shell_inflate_parity.json`

What it proves:

- source PR95 runtime vs packaged PR95 MLX-derived runtime emit byte-identical
  raw output for the declared file list `0.mkv`;
- output bytes match: `6104016`;
- output SHA-256 match:
  `db7e778f4a846a0417e630031b6bfeb9852d78642f1a26d9a52ffde17d4279ee`;
- archive SHA-256 matches on both sides:
  `6414614bd8f1ecbeb4c12b6f92ad670ea5a138941053ca7fa7d543c8e400e5f2`;
- blockers: `[]`;
- `full_frame_inflate_output_parity_claim=true` only for
  `parity_scope_kind=declared_file_list`;
- `contest_full_sample_parity_claim=false`.

What it does not prove:

- no contest CPU/CUDA score;
- no full contest-sample parity beyond the declared PR95 single-pair artifact;
- no training equivalence claim;
- no rank/kill or promotion authority.

## Similar Fake-Shaped Surfaces

1. `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`
   - Current status: sibling WIP being shaped as queue-visible local MVP
     infrastructure with
     `training_fidelity_class=rgb_frame_mse_local_mlx_research_mvp`.
   - Risk handled: earlier wording read like canonical PR95 scorer-faithful
     long training, but the implemented loss is RGB-frame MSE and still lacks
     source-schedule/scorer coupling.
   - Required before authority: paired targets, source-faithful
     optimizer/scheduler/QAT/C1a semantics, SegNet/PoseNet loss or calibrated
     exact scorer surrogate, export parity, shell parity, and paired contest
     CPU/CUDA auth eval.

2. Runtime-consumption proofs
   - Prove a runtime reads bytes and emits expected raw byte counts.
   - Do not prove source-vs-candidate raw equality, scorer equality, or score.

3. Selected-latent/export-forward parity
   - Proves selected MLX/PyTorch forward agreement inside an attested tolerance.
   - Does not prove shell inflate parity, full packet parity, or scorer parity.

4. Byte-closed package reports
   - Prove archive/runtime packaging structure.
   - Do not prove training equivalence, output parity, or score. Latent
     provenance must stay explicit, especially when source latents are reused.

5. PoseNet vector-MSE helpers
   - A first-six-vector MSE helper is not PoseNet/SegNet scorer-loss wiring.
   - This pass removed the proxy helper/tests from the PR95 MLX training diff
     rather than adding another authority-shaped affordance.

## Next Integration Required

- Promote PR95 package reports and queue observations to consume the new parity
  scope fields.
- Rename or gate any long-training plan that is not yet 1:1 as
  `pr95_mlx_rgb_frame_only_mvp` or equivalent.
- Add a real PR95 scorer-loss lane only when the full SegNet/PoseNet forward
  probes, pair targets, and export/archive/parity loop are all wired.
- Keep MLX rows `[macOS-MLX research-signal]` until paired contest CPU/CUDA auth
  eval anchors exist.
