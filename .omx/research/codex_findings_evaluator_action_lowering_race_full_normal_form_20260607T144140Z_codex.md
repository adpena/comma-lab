# Codex Findings: Evaluator-Action Lowering Race Full Normal Form

UTC: 2026-06-07T14:41:40Z

## Landed

- Expanded `tac.evaluator_action_lowering_race.v1` target accounting from the old four-arm surface to the measured evaluator-action normal form:
  - backend realization
  - pair-local latent action
  - frame0 pose compensation
  - frame1 Seg wall crossing
  - byte-priced sidecar
  - pose-compensated composite
  - SNeRV source-state action
  - semantic/pose primitive
  - byte/entropy rewrite
- Kept `discard` as a fail-closed verdict, not as a fake ActionEffect target.
- Added exact winner sections to `lowering_verdict.json`: `bytes_by_section`, `exact_score_terms`, `target_statuses`, and `next_blocker`.
- Embedded the existing measured-only ActionEffect commutator ledger in the lowering-race report.
- Hardened `nerv_long_run_launch_gate` so stale four-target accounting fails closed and parseback/inflate/byte/codec/authority blockers are emitted by name.

## Real Artifact

Path: `/Volumes/VertigoDataTier/pact/experiments/results/evaluator_action_lowering_race_20260607Tcodex_v2_full_normal_form`

Input support codec report: `/Volumes/VertigoDataTier/pact/experiments/results/support_codec_router_20260607T110540Z_codex/support_codec_router_report.json`

Action: `path_tube:p0:c0:a0f24babfd13cfe4`

Selected support codec remained `rle` at `2378` bytes, but the lowering verdict stayed fail-closed:

- `best_lowering=discard`
- `promotion_eligible=false`
- `score_claim=false`
- `first_failing_surface=path_action_support_without_wrong_to_target_is_not_birth`
- present target: `byte_priced_sidecar`
- missing measured targets: backend realization, pair-local latent action, frame0 pose compensation, frame1 Seg wall crossing, pose-compensated composite, SNeRV source-state action, semantic/pose primitive, byte/entropy rewrite

## Next Blocker

The cheapest support codec is not yet a scorer wall-crossing action. The next concrete burn-down is to bind a real direct-live frame1 Seg wall action or backend/latent realization to the same support/action identity, then re-run this race with parseback and inflate survival on the same `action_id`.

## Addendum: Receiver-Bound HiNeRV Sidecar Repack

The same normal form now has one real receiver-bound HiNeRV target-region sidecar
row after split-stream Brotli repack:

- report: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_witness_readiness_short_smoke_current_main_20260607Tcodex_v3_export_wall_normal_support/hi_nerv_mlx_training/target_region_action_split_brotli_repack/hinerv_target_region_action_sidecar_backend_comparison.json`
- action: `df6f7301995ee1ac60f84637beed9b390826b32c62521f1a5446680b8785c7a2`
- support: `d56a75511244d1cb71bfaca9ddff67513ab80c452e41e088b0a204897a2ced0c`
- archive: `788a3ca93cb340a7114b053ea18d03d237facb21b515023c52d89fab6ba818ae`
- archive bytes: `332848`
- payload codec: `split_brotli_v1`
- action payload bytes: `123159`
- verdict: `best_lowering=byte_priced_sidecar`, `first_failing_surface=none`, `promotion_eligible=false`

The support-identity summary still records one missing-support backend diagnostic
row, but this no longer poisons the receiver-bound sidecar candidate. The
current blocker is therefore precise: backend realization still has
`wrong_to_target=0`, while the sidecar survives but is too expensive to promote
without either better sidecar grammar or a better backend actuator basis.

## Validation

- `uv run pytest src/tac/tests/test_evaluator_action_lowering_race.py src/tac/tests/test_hinerv_target_region_action_comparison.py src/tac/tests/test_nerv_long_run_launch_gate.py -q` -> 78 passed
- `uv run ruff check src/tac/analysis/evaluator_action_lowering_race.py src/tac/analysis/nerv_long_run_launch_gate.py src/tac/tests/test_evaluator_action_lowering_race.py src/tac/tests/test_hinerv_target_region_action_comparison.py src/tac/tests/test_nerv_long_run_launch_gate.py tools/run_evaluator_action_lowering_race.py` -> clean
