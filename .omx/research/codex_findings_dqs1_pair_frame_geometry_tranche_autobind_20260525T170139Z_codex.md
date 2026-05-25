# Codex Findings: DQS1 Pair-Frame Geometry Tranche Autobind

- timestamp_utc: 2026-05-25T17:01:39Z
- agent: codex
- lane_id: lane_dqs1_pair_frame_geometry_tranche_autobind_20260525
- authority: local planning and queue preparation only; no score, promotion, rank/kill, paid dispatch, or exact-eval authority

## Finding

The DQS1 tranche runner was still too manual at the pair/frame/scorer geometry boundary: harvest observations could refresh the pairset acquisition plan, and the pair-frame lattice could independently produce low-impact geometry drop requests, but the tranche loop did not automatically build and bind that lattice during refresh.

I wired `tools/run_dqs1_local_first_tranche.py` so each acquisition refresh can run a two-pass local planner:

1. Build a base DQS1 pairset acquisition from the selector Pareto and current/prior DQS1 observations.
2. Build `pair_frame_scorer_geometry_lattice.v1` from that base plan plus frame-pair curriculum and pair-component xray evidence.
3. Rebuild the DQS1 pairset acquisition with `--pair-frame-geometry-lattice-json`, turning geometry-derived low-impact full-pair drops into normal acquisition rows.

The feature defaults on, can be disabled with `--no-refresh-pairset-geometry-lattice`, and preserves all false-authority fields.

## Concrete Proof

Artifact root:

`.omx/research/codex_dqs1_pair_frame_geometry_auto_refresh_20260525T170139Z/`

Inputs:

- selector Pareto: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/dqs1_gap_uleb_selector_pareto.json`
- frame-pair curriculum: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_frame_pair_curriculum_pair4_guarded_20260521_codex.json`
- pair-component xray: `experiments/results/d1_pair_component_xray_pair_mask_cpu600p235pn_cpu64_20260515_codex/pair_component_xray.json`

Outputs:

- base acquisition: `dqs1_pairset_acquisition_base.json`
- geometry lattice: `pair_frame_scorer_geometry_lattice.json`
- rebound acquisition: `dqs1_pairset_acquisition_with_geometry.json`

Observed local-planning result:

- base candidate count: 116
- lattice rows: 32
- lattice queue-executable requests: 6
- lattice geometry coverage: 0.0625
- rebound candidate count: 120
- rebound pair-frame geometry candidates: 4
- rebound geometry binding active: true
- score_claim: false
- ready_for_exact_eval_dispatch: false

## Integration

Changed code:

- `tools/run_dqs1_local_first_tranche.py`
- `src/tac/tests/test_dqs1_local_first_tranche.py`

The tranche result now records a `pair_frame_geometry_lattice` block inside each `pairset_acquisition_refresh`, including base plan paths, lattice paths, commands, selected evidence files, geometry request count, and coverage. This makes the geometry signal visible to the queue/DAG loop rather than requiring an operator to hand-run the lattice tool.

## Verification

- `.venv/bin/python -m ruff check tools/run_dqs1_local_first_tranche.py src/tac/tests/test_dqs1_local_first_tranche.py`
- `.venv/bin/python -m pytest src/tac/tests/test_dqs1_local_first_tranche.py -q` -> 12 passed
- `.venv/bin/python -m pytest src/tac/tests/test_dqs1_local_first_tranche.py src/tac/tests/test_pair_frame_scorer_geometry_lattice.py src/tac/tests/test_decoder_q_pairset_acquisition.py -q` -> 25 passed

## Remaining Gap

This closes the manual pair-frame lattice binding in the tranche loop, but masked/feathered/repair receiver starts remain blocked until receiver/materializer support exists for non-pair-drop mask semantics. The next non-leaf step is to turn the same refresh block into receiver-runtime-specific acquisition rows once a byte-closed masked or feathered receiver materializer is registered.
