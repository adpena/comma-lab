# Codex Findings: Pair/Frame Scorer-Geometry Lattice Bound Into DQS1

Generated: 2026-05-25T15:12Z
Agent: Codex
Topic: final rate attack automation / inverse scorer starts

## Result

Landed the pair/frame scorer-geometry lattice as a planning-only bridge between
existing frame/pair signal and queue-executable DQS1 pairset drops.

The point is to stop treating pair drops as isolated leaf tweaks. The lattice
turns SegNet/PoseNet topology, frame-pair curriculum signal, pair-component
X-ray signal, DQS1 rank order, and rate-repair budget semantics into grouped
queue starts that the existing local materializer path can already consume.

## Code Surface

- `src/tac/optimization/pair_frame_scorer_geometry_lattice.py`
- `tools/build_pair_frame_scorer_geometry_lattice.py`
- `src/tac/optimization/decoder_q_pairset_acquisition.py`
- `tools/plan_decoder_q_pairset_acquisition.py`
- `src/tac/tests/test_pair_frame_scorer_geometry_lattice.py`

The lattice remains false-authority throughout:

- no score claim
- no rank/kill authority
- no promotion authority
- no paid dispatch authority
- local queue start selection only

## Real Planning Artifact

Artifact root:

- `experiments/results/pair_frame_scorer_geometry_lattice_dqs1_20260525T151233Z/`

Inputs:

- DQS1 acquisition: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_acquisition/dqs1_pairset_acquisition_full_drop_two_20260523T121036Z.json`
- DQS1 selector Pareto: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/dqs1_gap_uleb_selector_pareto.json`
- frame-pair curriculum: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_frame_pair_curriculum_20260520_codex.json`
- pair-component X-ray: `experiments/results/d1_pair_component_xray_a1_baseline_cpu64_20260515_codex/pair_component_xray.json`

Outputs:

- `pair_frame_scorer_geometry_lattice.json`
- `pair_frame_scorer_geometry_lattice.md`
- `dqs1_pairset_acquisition_with_geometry.json`
- `dqs1_pairset_acquisition_with_geometry.md`

Observed artifact summary:

- lattice rows: 32
- geometry coverage: 0.0625
- queue-executable geometry requests: 6
- rebound DQS1 acquisition candidates: 39
- rebound geometry candidates: 6
- recommended rebound acquisition id: `pairset_drop_one_rank013_pair0327`

Low coverage is expected from the current available X-ray/curriculum inputs;
the important point is that sparse high-quality geometry signal now flows into
queue-executable grouped starts instead of being stranded in research artifacts.

## Tests

- `ruff check` on touched lattice, planner, CLI, and tests: pass
- `pytest src/tac/tests/test_pair_frame_scorer_geometry_lattice.py src/tac/tests/test_decoder_q_pairset_acquisition.py -q`: 13 passed

## Next Patch Set

1. Add a queue materialization handoff that turns
   `pair_frame_geometry_low_impact_drop_many` acquisition rows into local DQS1
   materializer queue entries.
2. Expand geometry coverage by canonicalizing more frame-pair curriculum and
   pair-component X-ray ledgers into the lattice input contract.
3. Add receiver/materializer families for masked/feathered within-selected-set
   variants, currently blocked by missing non-pair-drop semantics.
4. Bind inverse-scorer action cells into this same request schema so null-space
   starts become queue-executable rather than advisory.
5. Teach the acquisition planner to rank geometry starts with observed DQS1
   local CPU/MLX calibration once the first real geometry materializations land.
