# Pose Manifold Water-Fill Candidates - 2026-05-02 Codex

## Evidence Boundary

- Evidence grade: `diagnostic_planning_non_promotable`.
- Score claim: `False`.
- Required score truth: `archive.zip -> inflate.sh -> upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda`.
- H100/L40S/A100 diagnostics remain diagnostic until identical bytes pass T4/equivalent CUDA auth eval.

## Frontier Anchor

- Label: `C088_PR75_top40_p3_T4`.
- Archive bytes: `276386`.
- Archive SHA-256: `9feef7ffaa254f9e5408996a122682757a054144a3000539553786c5292b7d0a`.
- Score recomputed from components: `0.3155226919767294`.
- Avg PoseNet: `0.00049633`.
- Avg SegNet: `0.00061038`.

## Output Artifacts

- Plan JSON: `experiments/results/pr65_pose_transfer_sub314_20260503_codex/c088_pose_waterfill_plan/pose_manifold_waterfill_plan.json`.
- Dispatch recommendations: `experiments/results/pr65_pose_transfer_sub314_20260503_codex/c088_pose_waterfill_plan/exact_eval_recommendations.json`.

## Top Recommendation

- No dispatch recommendation was produced.

## Build-Only Macro Specs

- Macro specs emitted: `4`.
- These specs require a closed archive builder before exact eval; their expected utility is formula-only planning signal.

## Sub314 Local Candidate Addendum

Scope: local-only PR65/PR75/C088 pose-transfer planning and archive build.
No remote GPU job was dispatched and no dispatch claim was opened.

Artifacts:

- Summary JSON: `experiments/results/pr65_pose_transfer_sub314_20260503_codex/candidate_summary.json`.
- PR65/PR67 transfer policy: `experiments/results/pr65_pose_transfer_sub314_20260503_codex/pose_manifold_transfer_policy.json`.
- C088 waterfill plan: `experiments/results/pr65_pose_transfer_sub314_20260503_codex/c088_pose_waterfill_plan/pose_manifold_waterfill_plan.json`.
- Built candidate archive: `experiments/results/pr65_pose_transfer_sub314_20260503_codex/pr75_lag_eval_pose4_top67_p6_rebuild/c067_pr75_actions_lag_eval_pose4_top67_p6/archive.zip`.

Frontier anchor:

- C088 / PR75 top40 P3 exact T4: archive bytes `276386`, SHA-256
  `9feef7ffaa254f9e5408996a122682757a054144a3000539553786c5292b7d0a`,
  recomputed score `0.3155226919767294`, PoseNet `0.00049633`,
  SegNet `0.00061038`.

PR65 signal:

- PR65 has external/public global PoseNet advantage versus C-059:
  pose distance delta `-0.00014354`.
- The same public summary shows PR65 is `8078` bytes larger and has worse
  SegNet by `0.00009652`; it is not a direct transplant candidate.
- PR65 anatomy does not expose a compatible pair-local pose stream, so it is
  used only to motivate sparse/manifold pose repair, not as a copied payload.

PR75/C088 precision status:

- The local raw-output parity report is
  `experiments/results/pr75_raw_output_parity_20260503_codex/pr75_raw_output_parity.json`.
- The report isolates the PR75 robust-runtime mismatch to QP1 pose precision:
  robust output matches public output on selected pairs when public QP1
  float32 poses are fed directly.
- This makes QP1-float32 preservation the required runtime contract for future
  pose-manifold builds.

Built candidate:

- Candidate: `c067_pr75_actions_lag_eval_pose4_top67_p6`.
- Archive bytes: `276338`.
- Archive SHA-256:
  `de7d549ec4437b3ed9508c1f62f8a79bef88440ba3db0f4a7de5a2c83fe160ef`.
- Wire format: `P6`; selected records: `38`; no-op: `false`;
  source-preserving: `false`.
- Trace-estimated score: `0.3154606501519654`.
- Expected delta versus C088: `-0.00006204182476397995`.
- Rate delta versus C088: `-48` bytes / `-0.00003196122974986422` score.
- Selected trace sums: PoseNet `0.00006510502935348066`, SegNet
  `0.0001737806839325155`, combined `0.00023888571328599617`.
- Evidence grade: `byte_and_trace_planning_only_until_exact_cuda`.

Exact eval command if a local CUDA runner is available:

```bash
.venv/bin/python -u experiments/contest_auth_eval.py --archive experiments/results/pr65_pose_transfer_sub314_20260503_codex/pr75_lag_eval_pose4_top67_p6_rebuild/c067_pr75_actions_lag_eval_pose4_top67_p6/archive.zip --inflate-sh submissions/robust_current/inflate.sh --upstream-dir upstream --device cuda --keep-work-dir --work-dir experiments/results/pr65_pose_transfer_sub314_20260503_codex/exact_eval_work_c067_pr75_actions_lag_eval_pose4_top67_p6
```

Pose macro follow-up:

- `c_059_pose_atoms_top016_macro_build_spec`: 16 pairs, estimated charged
  bytes `32.0`, expected score saved `0.00029661178784829876`, net utility
  `0.00027530430134838925`, pairs
  `[164,64,130,112,97,153,70,198,420,289,166,435,78,418,87,159]`.
- `public_pose_manifold_transfer_top016`: 16 pairs, estimated charged bytes
  `32.0`, expected score saved `0.0002977666888432403`, net utility
  `0.00027099915892772895`, pairs
  `[164,64,130,112,97,153,70,198,420,289,166,435,78,156,418,87]`.

Failure risks:

- Direct public PR75 pose-only replay already regressed PoseNet on L40S
  diagnostic evidence; do not wholesale copy public pose bytes.
- The P6 archive is built and byte-closed but not exact CUDA scored.
- Trace deltas are not guaranteed to compose with exact PoseNet/SegNet.
- PR65 global pose advantage is external and non-local; it cannot promote or
  anchor stack math without a compatible charged archive and exact CUDA eval.
- Any remote eval must first claim the lane with
  `tools/claim_lane_dispatch.py claim ...`; this pass intentionally did not.
