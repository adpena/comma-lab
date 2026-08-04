# NEXT_IF_RESUMED

TJ1 landed the scorer-free stopping law, SQ1 replay receipt, NG1 class map, and
canonical equation registration. No exact score moved.

Next steps:

1. When SQ2 completes all `32/32` rows, rerun:
   `.venv/bin/python tools/replay_tj1_trajectory_stopping.py --out-dir .omx/research/ddm_tj1_20260805 --register`
2. Compare complete SQ2 realized `flips_after` and eta against
   `trajectory_replay.json -> sq2_validation_target`.
3. If SQ2 remains `safety_bound_REPORTED`, do not call it converged. Use
   `allocate_adaptive_depths` to route additional depth by projected remaining
   S gain instead of uniform caps.
4. For pose GN, use the shared law only when `marginal_value_floor > 0.0`.
   `marginal_value_floor=0.0` is the legacy no-gate limit.
5. Do not run scorer or exact-eval work from this TJ1 resume unless a new
   charter explicitly releases the scorer-free constraint.
