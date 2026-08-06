# ddm_rw1 Checkpoints

Initial HEAD before rw1 edits: `a5146af5eb56c1e2c17d45d82fd62c15673e9d9d`.

## Checks Run

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/optimization/tests/test_rw1_true_domain_instruments.py \
  src/tac/optimization/tests/test_fd_integer_near_margin_proposals.py -q
```

Result: 7 passed.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  experiments/ddm_q3x_q3_convergence_measurement.py \
  tools/smoke_ddm_rw1_fd_integer_near_margin.py \
  src/tac/optimization/rw1_true_domain_instruments.py \
  src/tac/optimization/fd_integer_near_margin_proposals.py
```

Result: passed.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_q3x_q3_convergence_measurement.py \
  --selection-mode strided --n 1 --stride 20 --offset 0 \
  --out .omx/research/ddm_rw1_20260806/q3x_old_naive_n1.json \
  --realizer naive-round --solver-form project-after \
  --cap-ladder 2 --steps 2 --eval-every 1 --threads 4 --max-chunk-pairs 8
```

Result: retained fraction 0.30612244898, FOLDED.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_q3x_q3_convergence_measurement.py \
  --selection-mode strided --n 1 --stride 20 --offset 0 \
  --out .omx/research/ddm_rw1_20260806/q3x_dk1_cvp_n1_block64.json \
  --realizer dk1-cvp --solver-form project-after \
  --cap-ladder 2 --steps 2 --eval-every 1 --threads 4 --max-chunk-pairs 8 \
  --dk1-cvp-tap-radius 0 --dk1-max-blocks 64
```

Result: retained fraction -0.00680272109, FOLDED, 64/3117 snapped blocks realized.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/smoke_ddm_rw1_fd_integer_near_margin.py \
  --limit 1 --stride 20 --offset 0 --max-proposals 6 --threads 4 \
  --method cvp --cvp-tap-radius 0 \
  --out .omx/research/ddm_rw1_20260806/fd_integer_near_margin_smoke.json
```

Result: 1/6 accepted proposals.

## Boundaries

No `/tmp` evidence path is cited as durable. No full n600 scorer job was launched. No exact eval was launched. No forbidden common-contract files were edited.
