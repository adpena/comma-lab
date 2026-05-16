# L5 v2 TT5L Contest Full-Frame Side-Info Proof

Date: 2026-05-16
Author: codex
Lane: time_traveler_l5_autonomy / L5 v2 staircase
Evidence grade: contest_full_frame_inflate_consumption_proof_committed_custody
Score claim: false
Promotion eligible: false
Ready for exact eval dispatch: false

## Purpose

Materialize the L5 v2 `byte_closed_temporal_sideinfo_consumption` gate against a
contest-shaped TT5L archive using full local CPU inflate outputs, not a
parser-only or toy-frame proof. This advances the staircase past the Dykstra
feasibility-control gate into probe-disambiguator planning without asserting
score movement.

## Source Archive

- Path:
  `experiments/results/lane_substrate_time_traveler_l5_autonomy_modal_a100_dispatch_20260514T100758Z__smoke__25ep_modal/lane_substrate_time_traveler_l5_autonomy_results/output/submission_dir/0.bin`
- Archive bytes: 34480
- Archive SHA-256:
  `f113598c4afd7fba4e75bb414bd0c1a331d07f8db6fdeb8e545ca85392f6733b`
- Parsed shape: `num_pairs=600`, `output_height=384`, `output_width=512`,
  `per_pair_bytes=45`, side-info bytes `(600, 45)`.

## Mutation

- Work dir:
  `experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_proof_20260516_codex`
- Mutated pair index: 17
- Mutated temporal-sideinfo range: bytes `[36, 45)` for that pair
- Mutated values: fixed int8 byte values at LSB-scale side-info positions
- Mutated archive bytes: 34486
- Mutated archive SHA-256:
  `e72d15c2b2567c7407d037b1059bbb1a4b0e988466f84674ad9a1a69f7313541`

## Commands

```bash
PACT_INFLATE_DEVICE=cpu .venv/bin/python -m tac.substrates.time_traveler_l5_autonomy.inflate \
  experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_proof_20260516_codex/baseline_archive_dir \
  experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_proof_20260516_codex/baseline_outputs \
  experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_proof_20260516_codex/file_list.txt

PACT_INFLATE_DEVICE=cpu .venv/bin/python -m tac.substrates.time_traveler_l5_autonomy.inflate \
  experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_proof_20260516_codex/mutated_archive_dir \
  experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_proof_20260516_codex/mutated_outputs \
  experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_proof_20260516_codex/file_list.txt

.venv/bin/python tools/build_tt5l_contest_sideinfo_consumption_proof.py \
  --baseline-archive experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_proof_20260516_codex/baseline_archive_dir/0.bin \
  --mutated-archive experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_proof_20260516_codex/mutated_archive_dir/0.bin \
  --baseline-output-dir experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_proof_20260516_codex/baseline_outputs \
  --mutated-output-dir experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_proof_20260516_codex/mutated_outputs \
  --file-list experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_proof_20260516_codex/file_list.txt \
  --artifact-out experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_consumption_proof.json \
  --manifest-out experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_outputs_manifest.json
```

## Durable Committed Custody

- Proof artifact:
  `.omx/research/tt5l_contest_sideinfo_consumption_proof_20260516_codex.json`
- Proof artifact SHA-256:
  `7efdac0ed0ac026d7adb7eb706b1a8f844ec6e15422b7defbb34114db5ef775a`
- Output manifest:
  `.omx/research/tt5l_contest_sideinfo_outputs_manifest_20260516_codex.json`
- Output manifest SHA-256:
  `00e121969d8f03876dfc84c31f742b720d95be93b60e3dd93170a672ad3ffcb2`
- Runtime tree SHA-256:
  `4f4f5d2e090386d90962145727ea3bfc74f417e3d034ecea4a81d43de3b81ff4`
- Inflated raw-output aggregate SHA-256:
  `15a9b2b197bec1393417368be98e891d79b656124bc452c3983a47096ed26206`

The large raw outputs remain ignored under `experiments/results/`; the durable
proof and manifest are committed under `.omx/research/`.

## Result

`predicate_passed=true` with:

- `parser_consumed_bytes=true`
- `output_changed=true`
- `raw_output_shape_compatible=true`
- `non_target_sections_identical=true`
- `n_pairs_hashed=600`
- `total_frames=1200`
- `raw_output_frame_nbytes=3052008`

The proof records per-section baseline/mutated hashes and requires all
non-target payload sections (`world_model_blob`, `ac_state_blob`, `meta_blob`)
to remain identical. This closes the false-causality hole where a side-info
proof could pass while a simultaneous decoder/meta/AC-state mutation actually
caused the output change.

After this artifact is present, `l5_v2_dispatch_readiness()` reports:

- `dykstra_valid=true`
- `sideinfo_valid=true`
- `timing_allowed=true`
- next non-PR106 action:
  `emit_c1_z5_tt5l_probe_template`

Remaining blockers are intentionally still active: probe disambiguator, paired
CPU/CUDA axis plan, empirical anchor, prediction-band baseline/custody/artifact,
and score-authority promotion evidence.
