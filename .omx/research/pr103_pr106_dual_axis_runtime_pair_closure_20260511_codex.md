# PR103-on-PR106 dual-axis runtime pair closure (Codex, 2026-05-11)

## Purpose

Close the PR103-on-PR106 dual-axis replay with explicit CPU/CUDA custody,
content-runtime pairing, and no promotion beyond the measured evidence.

## Paired artifacts

Both artifacts use archive SHA-256
`ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
with `185578` archive bytes and `n=600`.

- CPU axis:
  `experiments/results/dual_device_auth_eval/pr103_pr106_dual_runtime_cpu_v2_20260511T022553Z/contest_auth_eval.adjudicated.json`
  - score: `0.2296576634626332`
  - hardware: `github-actions-ubuntu-latest-x86_64`
  - device: `cpu`
  - GitHub Actions run: `25647084172`
  - runtime content tree SHA-256:
    `f2ebe56a408a55b39070f9f86ba77fb11a9b43d83c0e02692f0acc0bf1ff28bb`
  - runtime path tree SHA-256:
    `318a5aa0282e9f09c62148f24b46a65d79c7c0a3bf8bd641efd2fd7f44e6a8d7`

- CUDA axis:
  `experiments/results/modal_auth_eval/pr103_pr106_dual_runtime_cuda_v2_20260511T022553Z/contest_auth_eval.json`
  - score: `0.20898305277982338`
  - hardware: `Tesla T4`
  - device: `cuda`
  - Modal app: `ap-XtVKhB4XGNPJDfdrJBc3KZ`
  - runtime content tree SHA-256:
    `f2ebe56a408a55b39070f9f86ba77fb11a9b43d83c0e02692f0acc0bf1ff28bb`
  - runtime path tree SHA-256:
    `b652b24a21b232da6b4648eb580da4472f9db87b9ad5e2dfa74e5c7f76d8ed00`

The provider path tree hashes differ because the same runtime content is
mounted under different provider-local roots. The cross-provider pairing key is
the `runtime_content_tree_sha256`, which is basename-independent.

## Planner closure

Pair plan:
`experiments/results/dual_device_auth_eval/pr103_pr106_dual_runtime_pair_plan_20260511T0235Z.json`

Planner facts:

- `dual_axis_completion.blockers=[]`
- `paired_score_artifacts_complete=true`
- `same_archive_sha256=true`
- `same_archive_bytes=true`
- `same_runtime_tree_sha256=true`
- `rank_or_kill_eligible=true`
- plan-level `score_claim=false`
- plan-level `promotion_eligible=false`

## Interpretation

This closes the apples-to-apples custody blocker for this exact PR103-on-PR106
packet: same scored archive bytes, same runtime content, full-sample CPU and
T4 CUDA artifacts.

It does not prove that the CPU value is the public leaderboard value, and it
does not promote the CPU axis as a CUDA score. The active score-lowering
evidence for this packet is the CUDA score `0.20898305277982338`; the CPU score
is a separate public-replay diagnostic axis. The CPU/CUDA gap is therefore a
measured property requiring mechanism analysis, not a license to substitute
CPU replay values for contest-CUDA results.

Follow-up adversarial note after component inspection: this packet is
CUDA-better, not CPU-better. The component deltas are:

- CPU: SegNet `0.00065592`, PoseNet `0.00016400`, rate contribution
  `0.123568773`, canonical score `0.2296576634626332`.
- CUDA: SegNet `0.00067084`, PoseNet `0.00003360`, rate contribution
  `0.123568773`, canonical score `0.20898305277982338`.
- CUDA - CPU: SegNet contribution `+0.001492000` worse, Pose contribution
  `-0.022166611` better, net score `-0.020674611` better.

This does not contradict the A1/PR101 CPU-better trend. The PR103/PR106 runtime
auto-selects CUDA in `submissions/pr103_pr106_final_runtime/inflate.py`, so the
CUDA run can produce different raw decoded frames before `upstream/evaluate.py`
scores them. That makes this a full-pipeline device-drift finding
(`archive bytes + runtime source + inflate device + evaluate device`), not a
shared-decoded-output scorer-only drift finding. Future dual-axis artifacts must
record inflated raw-output hashes so comparisons can distinguish same archive
and same runtime source from same decoded frames.

Operator correction + xhigh review: do not invert the public HNeRV trend into
any universal CPU-better rule. The observed PR100/101/102/103/107 CPU-better
cluster and the PR103-on-PR106 CUDA-better pair are both legitimate measured
full-pipeline facts under their own runtime contracts. The operational model is
per-submission and per-runtime:

`archive bytes + runtime content + inflate device + inflated raw-output SHA +
evaluate device + PoseNet/SegNet components`.

The xray follow-up is therefore not "choose CPU" or "choose CUDA"; it is to
measure both axes, compare inflated raw-output aggregate hashes, then localize
the delta through PoseNet/SegNet/Hydra/dual-head diagnostics before treating a
gap as an exploitable mechanism.

## Dispatch claims

The CPU and CUDA lane claims for this v2 pair were closed with terminal
`completed_contest_cpu` and `completed_contest_cuda` rows. The remaining active
claim after closure should be the pre-existing T1 Ballé Modal Phase 1 job:

- lane: `t1_balle_128k_endtoend`
- job: `t1_balle_modal_phase1_ab2d0f6_20260510T1437Z`
- Modal call: `fc-01KR955JSYQAVTTYZA48VAV7WJ`

## Next score-lowering consequence

Do not continue treating PR103/PR106 CPU replay as an inferred submission
frontier. The next score-lowering implementation work is archive-in-loop,
rate-capped exact-CUDA candidate selection for T1/A1, so proxy improvements are
forced through a contest packet before lane status changes.
