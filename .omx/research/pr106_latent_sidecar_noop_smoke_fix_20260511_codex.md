# PR106 latent-sidecar no-op smoke fix (2026-05-11)

## Scope

This is a score-lowering blocker fix for the PR106 sidechannel ladder. The
latent sidecar is the first gate before y-shift, LRL1, and stacked sidechannel
dispatches, so its local smoke artifacts must prove nontrivial charged runtime
behavior before any CUDA spend.

## Finding

`experiments/build_pr106_latent_sidecar.py` selected a latent dimension per
pair but computed:

```text
round((-0.005 * sign(abs(latent))) / 0.01)
```

Because `sign(abs(latent))` is nonnegative and NumPy rounds half steps to even,
the emitted `delta_q` values rounded to zero. The metadata could report many
selected dimensions while the encoded sidecar was effectively no-op.

## Fix

- Use the signed latent value at the selected dimension.
- Emit a one-quantum `delta_q = -sign(latent)` nudge toward zero for selected
  smoke pairs.
- Count corrections with both `dim != 255` and `delta_q != 0`.
- Record `nonzero_delta_count` in diagnostics.
- Keep PR100-derived sidecar gain as a `planning_target_*` field, not a
  predicted score for heuristic smoke artifacts.
- Update the latent-sidecar runbook comments so heuristic smoke is not described
  as scorer-backed search.
- Update the PCC9 live-registry test to assert the current strict state: no
  unresolved live smoke-promotion violations.

## Evidence boundary

The fixed smoke artifact remains `score_claim=false` and
`ready_for_exact_eval_dispatch=false`. It proves nontrivial sidecar wire/runtime
behavior; it does not prove score movement. Real score lowering still requires a
scorer-backed selector or an exact CUDA result from a claimed byte-closed packet.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_pr106_latent_sidecar.py src/tac/tests/test_check_lane_smoke_signal_nontrivial.py`
- `.venv/bin/python tools/check_lane_smoke_signal_nontrivial.py --strict`
- `.venv/bin/python experiments/build_pr106_latent_sidecar.py --source-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip --output-dir /tmp/pr106_latent_sidecar_smoke_check_20260511 --device cpu --smoke --top-k 3`

Smoke rebuild result: `n_corrections=3`, `nonzero_delta_count=3`,
`delta_q_min=-1`, `delta_q_max=1`, `score_claim=false`.
