# Next Experiments -- 2026-04-29 Reinit Queue

## P0: Stop Modal Runtime Drift Before More Dispatch

- Status: required before any new Modal lane launch.
- Evidence: `experiments/results/lane_lane_mae_v_modal/modal_lane_lane_mae_v.log` failed on `ModuleNotFoundError: No module named 'pydantic'`.
- Hypothesis: Modal launcher/runtime is executing lane scripts without installing the project dependency set that `pyproject.toml` declares.
- Work: fix the Modal bootstrap/install step so `pydantic`, `brotli`, `cryptography`, and runtime extras are present before `scripts/remote_lane_*.sh` runs.
- Gate: run a one-command Modal dry run that imports `tac.training`, prints dependency versions, and exits before GPU work.
- Expected payoff: prevents immediate no-measurement failures across MAE-V, Q-faithful, self-compress, Uniward, SZ, and related lanes.
- Cost/risk: low local code/test cost; high value because multiple active T4 calls share this failure mode.

## P1: Repair Anchor Path Contracts in Remote Lanes

- Status: needed before relaunching Uniward or any lane referencing local-only submission paths.
- Evidence: `lane_uniward_texture` failed because `submissions/baseline_dilated_h64_0_90/renderer.bin` was absent on Modal.
- Work: make lane scripts anchor on committed/recovered experiment artifacts such as `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip` or `experiments/results/lane_a_landed/iter_0/*`, and add preflight hard errors with actionable paths.
- Gate: local static preflight plus canonical E2E smoke proof for the edited lane.
- Expected payoff: converts no-measurement remote failures into either local failures or measured auth results.
- Cost/risk: low; be careful not to rewrite unrelated lane scripts.

## P2: Diagnose Omega Hessian Device-Side Assert

- Status: blocked by remote CUDA crash.
- Evidence: `experiments/results/lane_lane_omega_hessian_modal/modal_lane_lane_omega_hessian.log` crashed in `profile_hessian_per_weight.py` while calling the renderer.
- Work: reproduce with `CUDA_LAUNCH_BLOCKING=1` on a CUDA host or add stronger CPU/MPS shape/dtype/range validation around masks, pose_dim, and renderer kwargs before CUDA launch.
- Gate: tiny Hessian profile completes on 1-2 batches and writes a profile artifact; no auth run before that.
- Expected payoff: preserves the high-EV per-weight bit-allocation lane without burning another T4/4090 attempt blind.
- Cost/risk: medium; avoid changing renderer semantics.

## P3: Decide Whether SZ No-Mask Paradigm Is Compatible

- Status: blocked by compliance/inflator contract.
- Evidence: recovered SZ phase2 archive is 3.3KB, but canonical smoke fails: no `masks.mkv`, and current `inflate_renderer.py` would fall back to non-compliant scorer-time extraction.
- Work: either implement an explicit rule-faithful no-mask inflator path for `SZv1` or demote SZ until a compliant mask/latent representation exists.
- Gate: `canonical_local_auth_eval_smoke.py --fixture-archive <sz archive>` passes before any auth eval.
- Expected payoff: potentially huge rate win only if the compliance gap is real; otherwise cut quickly.
- Cost/risk: high; do not claim the 3.3KB archive as a valid score candidate yet.

## P4: Rebootstrap Upstream Snapshot Deliberately

- Status: state hygiene, not score work.
- Evidence: snapshot file reports `ec82c291...` from 2026-04-03 while live workspace upstream is at `cd64c68...`; root `upstream/` is a separate dirty checkout at `11ad728...`.
- Work: choose the intended upstream root, run the bootstrap/snapshot path, and record the resulting commit and file hashes.
- Gate: `comma-lab status` and direct `git -C <upstream> rev-parse HEAD` agree, with installed submission changes understood.
- Expected payoff: removes ambiguity before final submission or reproduction.
- Cost/risk: medium because upstream dirs are dirty; review before overwriting anything.

## Hold / Do Not Relaunch Blind

- Pose-only TTO on h64: historical proxy-auth gap and worse auth result.
- Half-frame CRF56: measured 3.20 [contest-CUDA], negative vs Lane G v3.
- Lane M v2 radial zoom: measured 1.84 [contest-CUDA], negative vs Lane G v3.
- Any lane that uses KL distill as a primary/large-weight loss: only the corrected small-weight Lane G v3 result is current positive evidence.
