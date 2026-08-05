# ddm_pg1 Next If Resumed

Status: build landed in the working tree; serializer commit is blocked by sandbox Git-object writes. No scorer and no launch performed.

Do next:

1. Commit the exact intent patch once Git writes are available:
   - `.omx/research/ddm_pg1_20260805/COMMIT_INTENT.patch`
   - Use serializer with `[no-triality] [p0-ledger-ok]`, `--no-co-author`, and post-edit shas.
2. Rerun the focused pg1 MLX tests on an MLX-capable host:
   - `.venv/bin/python -m pytest src/tac/tests/test_ddm_tb1_tr1_renderer.py -k 'pg1_q3_mlx or pg1_q3_custom_vjp or pg1_q3_off_path or pg1_q3_pose_grad' -q`
3. If those pass, queue the Q3-constrained window A/B at the next owned boundary slot.
4. Measure both arms under matched seed, schedule, parent checkpoint, and boundary conditions.
5. Bank only typed outcomes:
   - Q3 slower: Q4 spend was load-bearing.
   - Q3 comparable with zero/measured pose-null residual damage: pg1 wins.

Do not:

- Do not run scorer from this build-only lane.
- Do not touch jd5 live run dirs.
- Do not claim pose nullity after uint8/realization without measuring the integer residual.
- Do not treat this as a pointer-moving result; the frontier remains `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.
