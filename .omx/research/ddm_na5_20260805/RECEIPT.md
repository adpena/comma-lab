# ddm_na5 2026-08-05 Receipt

Charter: `.omx/tmp/codex_runs/na5_prompt.md`  
Common contract: `.omx/tmp/codex_runs/_common_contract.md`  
Axis: `[macOS-CPU advisory]` and `[macOS-CPU frozen-scorer advisory]` only.  
Score claim: false. Contest pointer: borrowed/unmoved.

## Status

NA5 completed the bounded custody, registration, and fire-order work, but did
not produce a new n120 pose-family scorer verdict. The route-changing
post-hoc/stored pose-family verdict therefore remains the NA2 weak-evidence
standing verdict pending a successful seeded-random/stratified n>=120 rerun.

## Recall Evidence + SHA Table

| Artifact | SHA-256 | Use |
|---|---:|---|
| `.omx/tmp/codex_runs/na5_prompt.md` | `615cc9c2d405a5575fbc0407f52d8b4e232259ebdfa9ace9b94f5972e5dfcb90` | NA5 charter |
| `.omx/tmp/codex_runs/_common_contract.md` | `eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771` | common operating contract |
| `PROGRAM.md` | `a6d5f79f3241ca1ae17b2587afd9940e1a4ea598804fd9efa152f2330e15db82` | governing program read |
| `CLAUDE.md` | `65da6dd8dcf6b11c0ecdd352938570fd5589c5e5e014d97acd63297f82a8c47c` | governing instructions |
| `AGENTS.md` | `65da6dd8dcf6b11c0ecdd352938570fd5589c5e5e014d97acd63297f82a8c47c` | governing instructions mirror |
| `docs/operating_manual_craft_handoff.md` | `40d157a039d4dd242bfb189d53e6b82abcc5d037adceb0a52c9bb2956903f212` | handoff contract |
| `.omx/state/main_hot_state.md` | `326cfbc9ab72f49cdb6ec836ffea80795d94b234a8d884d59d692b6e31a3cd8a` | current pointer/focus read during receipt |
| `.omx/research/ddm_na2_negative_audit_20260803.md` | `d7d7b503b461c0754226d9e0b18866bce095ae94d296019b03c46f24b634dbf2` | source negative audit |
| `.omx/research/ddm_na3_20260805/ddm_na3_receipt.md` | `5b80bbe4807823059791063d069f554b729d73d6d730ed045b9fb29e05dcde6b` | prior #931/#923 custody |
| `.omx/research/ddm_ub1_20260805/UB1_RECEIPT.md` | `8033fa6dcaf63558ab30b25edd671e42141f487938dd271b579f140c92325fa2` | recovered pose harness/fire-order custody |
| `.omx/research/ddm_ng1_20260805/ng1_negative_results_audit.md` | `857f32168b798995d62d21ee6f7e9e1a0c51f576626c64714ea4702d941eec81` | negative audit / sigma_eff fire-order custody |
| `.omx/research/operator_directive_per_edge_optimality_criteria_20260805.md` | `8a42292e6565089b927dd3c747df6b91b4ec92a6bd6a18dab48f9c9732d4383b` | addendum-8 native-coordinate law |
| `src/tac/canonical_anti_patterns/na3_subset_bias_builders.py` | `3a990684ede1d21a17727c375f0017c1a25a7a5020f5b289bd354910c8dda4bf` | two NA2-demanded classes already registered |
| `src/tac/canonical_anti_patterns/na5_native_coordinate_builders.py` | `ae77f2c88097ff07d60fe795b54f216d40e7a66ee05573bbaa213fa7fcefe34b` | NA5 addendum-8 class builder |

## NA5 Artifacts

| Artifact | SHA-256 | Status |
|---|---:|---|
| `.omx/research/ddm_na5_20260805/prefix_pose_bias_rederive_931.json` | `0342aa52af7ccbae7cc5ac5d0a2da4283571b002b0612026766d867fd624d6ae` | reproduced from raw D2 JSONL |
| `.omx/research/ddm_na5_20260805/pose_family_923_fire_orders.json` | `9da76b2fde4f340e48b8012d994d06e18c4c8ac5fd7a4eb8fda452b43cc2c1db` | fire orders preserved |
| `.omx/research/ddm_na5_20260805/pose923_render_cache_attempt.json` | `6340ceff2e5d3ba3725a59b99fd923c197827e3e7a3a198a82aa469fa229a315` | attempted, interrupted, no cache |
| `.omx/research/ddm_na5_20260805/pose_family_rerun_status_923.json` | `f81c62c9ae181a2849037a0752ac1c98cd8777d968961b9999eec8731cc39030` | no new verdict |
| `.omx/research/ddm_na5_20260805/sigma_eff_n600_fire_order.json` | `f33eab75def70933f6738913107c366cdfbcfca978601be2247eb8299a01a71c` | queued, not run |

## #931 Ratio Re-Derivation

NA5 reproduced the pose prefix hardness ratios from the raw D2 JSONL:

- Source: `/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl`
- Source SHA-256: `d2853c92090c28ebe558ece4a21b2847b55e25c9d768bef167bcba9dc67b72e5`
- Rows: 600
- Field: `d_pose_shipped_f16`
- Population mean: `0.15950891917937635`
- n24 prefix ratio: `2.535475579649216`
- n48 prefix ratio: `2.640181689154513`
- n64 prefix ratio: `2.6477688499984713`
- n96 prefix ratio: `4.206770932037034`
- hardest/easiest 60-pair block ratio: `79.43661398538532`

Interpretation: video-order pose prefixes are harder than population, so prefix
pose NO-GO verdicts are false-negative shaped and are not conservative
population-family walls.

## #923 Pose-Family Rerun Status

Selection custody:

- Mode: `stratified_blocks`
- Seed: `20260805`
- Denominator: `120/600`
- Governing ratio: `d_pose_shipped_f16` subset/population `1.0057539935665503`
- Selection verdict: `MATCHED`

NA5 attempted the scorer-free render-cache prerequisite:

```
.venv/bin/python experiments/ddm_ub1_pose_family_923_harness.py build-render-cache --selection .omx/research/ddm_na3_20260805/stratified_pose_selection_923.json --checkpoint experiments/results/levelset_n600_crucible_v6_run1_20260708T095730Z/levelset_witness_ema_mlx.npz --out-cache /Volumes/VertigoDataTier/pact/ddm_na5_20260805/pose923_run1_stratified_n120_oracle_render_cache.npz
```

The run remained compute-bound inside `numpy_oracle_reference_frames` and was
interrupted before any durable cache was materialized. Therefore no scorer
rerun was started.

| Pose family item | NA5 classification | Reason |
|---|---|---|
| `pose_l2_truedepth` | `NOT_RERUN_NO_VERDICT` | blocked by absent depth cache / true-depth harness in searched UB1 scope |
| `pose_carrier_arms` | `NOT_RERUN_NO_VERDICT` | recovered scorer command exists but render cache did not materialize |
| `pose_mladder_depthwarp` | `NOT_RERUN_NO_VERDICT` | A0 command exists but render cache did not materialize; A2/A2+ logs absent |
| `pose_stratified_texture` | `NOT_RERUN_NO_VERDICT` | blocked by absent texture grid / texture harness in searched UB1 scope |

Route status: no NA5 route-changing verdict. The post-hoc/stored family remains
weak-evidence standing guidance, not a measured population closure.

## Sigma_eff n600

NA5 did not consume a full n600 scorer slot. The exact queued command is
preserved in `sigma_eff_n600_fire_order.json`:

```
.venv/bin/python experiments/probe_lever_d_selective_fullstack.py --state-ckpt experiments/results/torch_vehicle_full_mps_basin_bc20_n600/torch_vehicle_checkpoint_state.pt --video upstream/videos/0.mkv --which ema --n-pairs 600 --tau 0.5 --batch 6 --targets-cache /Volumes/VertigoDataTier/pact/ddm_na5_20260805/lever_d_selective_targets_n600 --out-json /Volumes/VertigoDataTier/pact/ddm_na5_20260805/lever_d_selective_probe_n600_sigma_eff.json
```

## Anti-Pattern Registration

The two NA2-demanded subset-bias classes were already registered by NA3 and
verified in the registry:

- `prefix_bias_sign_inversion_pose_axis_v1`
- `subset_default_silent_under_sampling_v1`

NA5 added and registered:

- `lossy_projection_shipped_expecting_decode_realization_v1`

The new class is implemented in
`src/tac/canonical_anti_patterns/na5_native_coordinate_builders.py`, exported
through `src/tac/canonical_anti_patterns/__init__.py`, covered by
`src/tac/canonical_anti_patterns/tests/test_na5_native_coordinate_builders.py`,
and appended to `.omx/state/canonical_anti_patterns_registry.jsonl` at
`2026-08-05T18:47:09Z`.

## Verification

Commands run:

```
.venv/bin/python -m pytest src/tac/canonical_anti_patterns/tests/test_na5_native_coordinate_builders.py src/tac/canonical_anti_patterns/tests/test_na3_subset_bias_builders.py
.venv/bin/python -m py_compile src/tac/canonical_anti_patterns/na5_native_coordinate_builders.py src/tac/canonical_anti_patterns/__init__.py src/tac/canonical_anti_patterns/tests/test_na5_native_coordinate_builders.py
.venv/bin/python tools/review_tracker.py mark-file src/tac/canonical_anti_patterns/na5_native_coordinate_builders.py --status reviewed
.venv/bin/python tools/review_tracker.py mark-file src/tac/canonical_anti_patterns/tests/test_na5_native_coordinate_builders.py --status reviewed
```

Results:

- `7 passed in 0.14s`
- `py_compile` passed
- review tracker marked the builder file twice and the test file twice
- review tracker refused `src/tac/canonical_anti_patterns/__init__.py` because it is not an ingested tracked entity; no override was used

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
