# Shannon-Floor Large-Move State Checkpoint - 2026-05-01

Current operating correction: millipoint perturbation work is telemetry only.
The required path is a large score move toward the Shannon floor. Prioritize
Alpha representation collapse, learned/corpus codecs, and any route capable of
removing hundreds of KB or materially improving PoseNet/SegNet distortion.

Important current artifacts:

- PFP16 A++ promoted frontier remains `1.043987524793892`, archive SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  archive `686635` bytes.
- Exact CUDA perturbation candidate bundle:
  `experiments/results/frontier_candidate_pfp16_r7_eps_p2_20260501/`.
  Score `1.0368667951088641`, archive SHA
  `c00393dda0736edb2b2a25e3108624b7571042c63e6c5b138b5168c1a28f1193`,
  `686636` bytes, CUDA/T4, `600` samples. It is component-gate review only,
  not the breakthrough.
- Independent Lightning re-eval queued:
  `exact_eval_r7_p2_canonical_t4_20260501T1020Z`, status initially `Pending`,
  cost `0.0`.

Large-move Alpha/Lane 12 state:

- Existing Lane 12 `jsonfix40` is exact-CUDA negative for the measured
  implementation only: score `26.03719330455429`, PoseNet `49.7784996`,
  archive `296478` bytes, `masks.nrv` compressed about `22127` bytes.
- `scripts/remote_lane_12_alpha_geo0_pose_regen.sh` was added. It does not
  retrain NeRV. It keeps the exact measured `masks.nrv`, regenerates
  `optimized_poses.bin` against decoded candidate masks, rebuilds the archive,
  and runs CUDA auth eval/adjudication.
- Vast dispatch was started with label
  `lane_12_nerv_alpha_geo0_pose_regen`. Interpret as stale-pose isolation:
  PoseNet recovery means Alpha remains high-EV; continued collapse means
  `jsonfix40` mask geometry is incompatible and Alpha needs decoded-baseline
  retrain plus sparse residual repair.
- New NeRV retraining remains blocked by
  `.omx/state/lane12_nerv_l2_clearance.json` until the required fields and
  three Grand Council clean passes exist.

Modal state:

- `experiments/modal_auth_eval.py` was hardened to call canonical
  `experiments/contest_auth_eval.py --device cuda`, with CUDA/DALI preflight,
  archive SHA/byte custody, `n_samples=600` validation, and no CPU/direct
  inflate fallback. Modal artifacts remain non-promotable until adjudicated.

Alpha probe state:

- `experiments/paradigm_alpha_real_archive_eval.py` now uses the PFP16 A++
  archive by default, loads a safe named mask member, records custody, rejects
  hidden/unsafe ZIP members, and labels outputs empirical/no-score.
- Runbook: `docs/runbooks/alpha_lane12_large_move_next_actions.md`.

Verification in this loop:

- `bash -n scripts/remote_lane_12_alpha_geo0_pose_regen.sh` passed.
- `py_compile` passed for touched Alpha/Modal/eval files.
- Modal focused tests: `9 passed`.
- Alpha mask probe tests: `35 passed`.
