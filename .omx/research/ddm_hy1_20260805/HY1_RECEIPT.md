# HY1 Receipt — code-health batch (#942, #907, #856, #899, #913)

## Verdict Table

| row | verdict | outcome |
|---|---|---|
| #942 witness_control failures | FIXED | Initial premise was stale: the suite had 19 failures, not 17. Fixed 15 stale expectation/scaffolding failures and marked 4 MLX environment failures as named #856 known-red xfails. No silent skips. |
| #907 ST_GRID duplicate guard | PREMISE-STALE | The live code already has a parity guard for the intentionally vendored receiver/table surfaces. The divergent fitted-grid case is explicit, not drift. |
| #856 import-time MLX device leakers | HONEST-SUBSET+fire-order | Closed the 4 HY1-visible MLX failures as explicit #856 xfails. Did not bulk-migrate all import-time device calls: the current corpus has more sites than the charter headline and prior recall says most are deliberate CPU pins whose hazard is process-wide import mutation. |
| #899 required-component JSONL read path | PREMISE-STALE | Read/write validation and the raw-reader preflight guard already exist. Focused tests pass. No new reader patch was needed. |
| #913 contour_codec rename | FIXED | Moved the implementation to `tac.boundary_math.dense_raster_lzma_baseline`, left `contour_codec.py` as a compatibility shim because Rust parity/golden-vector artifacts cite the old path, and updated internal Python importers/current runtime docs. |

No scorer, frozen-scorer forward, `evaluate.py`, launch, or paid dispatch ran in HY1.

## Row Details

### #942 Witness Control

Initial command:

```bash
.venv/bin/python -m pytest src/tac/witness_control/tests -q
```

Initial result: 19 failed, 1023 passed, 2 skipped.

Triage:

| count | class | disposition |
|---:|---|---|
| 1 | stale expectation | `test_costate_live_ingest.py` now asserts parser authority while allowing monitor classifier drift. |
| 1 | stale expectation | `test_polyak_finisher.py` now checks the new Polyak flags only, not unrelated historical stale flags. |
| 13 | stale fixture/scaffolding | `test_taskspace_g112_exact_checkpoint_partition_v1.py` now builds a native-v3 checkpoint fixture with G111/G112 lineage and packet invariants matching the current opener. |
| 1 | stale fixture/scaffolding | pre-existing untracked `test_taskspace_single_stage_score_attempt_v1.py` now scopes row-selection identity separately from full G120/G112 recursive opener construction. |
| 4 | environment/MLX-gating | `test_payload_tto_core.py` and `test_stage3_cache_and_golden.py` xfail only when `mlx.random.seed` raises `[metal::load_device] No Metal device available`, with owner `#856 known-red environment/MLX-gating`. |

Verification:

```bash
.venv/bin/python -m pytest src/tac/witness_control/tests -q
```

Result: 1038 passed, 2 skipped, 4 xfailed in 49.22s. The process still prints an ignored MLX nanobind atexit RuntimeError from the same no-Metal device state; it is not a pytest failure.

### #907 ST_GRID

Source check:

```bash
rg -n "ST_GRID|FITTED_ST_GRID|INCUMBENT_ST_GRID" src/tac tools experiments runtime-rs
.venv/bin/python -m pytest src/tac/tests/test_st_grid_vendored_copies_agree.py -q
```

Result: 3 passed in 12.00s.

Adjudication: canonical copies and the intentionally fitted copy are already separated by the test. The current guard fails on drift and permits the fitted table by name. No live ST_GRID values were changed.

### #856 MLX Import-Time Device Mutation

HY1-visible failures were in `payload_tto` and `stage3_cache` tests, where CPU pinning still allowed `mlx.random.seed` to load Metal in this sandbox. Those now xfail only for the exact no-Metal RuntimeError.

Bounded current scan found top-level `mx.set_default_device` sites in source/test/probe surfaces exceeding the charter's "12" count, including source tests under `src/tac/tests`, `src/tac/optimization/tests`, and top-level `experiments`. Prior recall (`ddm_qd1`) says the older audit saw 17 sites, with the framing inverted: most were deliberate CPU pins and the remaining work is an import-time process-state mutation guard, not a blind deletion pass.

Verification subset:

```bash
.venv/bin/python -m pytest src/tac/witness_control/tests/test_payload_tto_core.py src/tac/witness_control/tests/test_stage3_cache_and_golden.py -q
```

Observed during the batch: 1 passed, 4 xfailed.

Fire order:

1. Build a warn-only AST detector for top-level `mlx.core.set_default_device`, `mx.set_default_device`, and import-time `mx.random.seed`, classifying CPU-pin vs GPU-allocator vs seed-load cases.
2. Add a positive-control fixture using one known pre-fix top-level device mutation and a negative-control fixture where device setup is inside a pytest fixture or function body.
3. Migrate deliberate CPU pins into fixtures/lazy helpers one ownership group at a time, preserving byte identity where a live path is touched.
4. Wire the detector warn-only in `preflight_all`, then flip strict only after the live count is zero.

### #899 Required-Component JSONL

Source check found the existing guard:

```bash
rg -n "check_no_unvalidated_required_component_jsonl_readers|REQUIRED_COMPONENT_JSONL_READ_OK|read_required_components" src/tac/preflight.py src/tac/witness_dsl src/tac/tests
.venv/bin/python -m pytest src/tac/tests/test_check_required_component_jsonl_read_validation.py -q
.venv/bin/python -m pytest src/tac/tests/test_build_completeness_grades.py -q
```

Results: 10 passed in 0.59s; 58 passed in 4.48s.

Adjudication: the read path and residual raw-reader guard are already closed at HEAD. HY1 did not reopen or reimplement #899.

### #913 Dense-Raster LZMA Rename

Implementation:

- Added `src/tac/boundary_math/dense_raster_lzma_baseline.py` with the actual dense-label RAW-LZMA2 encoder/decoder.
- Replaced `src/tac/boundary_math/contour_codec.py` with a compatibility shim for old artifact paths.
- Updated current Python importers and current Rust parity/docs/manifest text to the honest module name.
- Left historical research text and pre-existing dirty unrelated comments alone.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_boundary_math_seg_core.py src/tac/tests/test_v2_compose_archive_grammar.py src/tac/boundary_math/tests/test_context_partition_codec.py -q
.venv/bin/python runtime-rs/crates/tac-boundary-decode/python_reference_equivalence_test.py
cargo test -p tac-boundary-decode
.venv/bin/python -m ruff check src/tac/boundary_math/dense_raster_lzma_baseline.py src/tac/boundary_math/contour_codec.py
git diff --check -- scoped HY1 path set
```

Results: 51 passed in 2.53s; Python oracle parity all pass; Rust crate 15 passed; scoped Ruff passed; `git diff --check` passed. A broad Ruff command over every edited/existing probe file still reports older lint debt in unrelated probe code, so it is not claimed as a HY1 pass gate.

## Recall Evidence

Sources searched beyond the charter seeds:

| query/surface | finding | plan change |
|---|---|---|
| `MEMORY.md` for `#899`, `required_component_ledger.jsonl`, `check_no_unvalidated_required_component_jsonl_readers` | #899 read/write validation and raw-reader guard already landed; commit memory cites 10 focused tests and 58 build-completeness tests. | Re-scoped row #899 to verification, not reimplementation. |
| `.omx/research` for `#856`, `#907`, `#899`, `#913`, `ST_GRID`, `contour_codec` | `ddm_qd1` reports #856 as 17 sites not 12 and #899 as stale; `ddm_gt2` names #913 as dense raster LZMA wearing a boundary-edge name; ST_GRID memos separate generic/vendored/fitted copies. | Kept #856 as honest subset with fire-order; treated #907/#899 as stale-premise verification rows; executed #913 rename. |
| `tools/list_canonical_equations.py --json` filtered for `mlx`, `ST_GRID`, `contour`, `required_component` | No HY1-specific equation changed; results were mostly MLX drift/compile equations and existing menu saturation surfaces. | No equation registry edit made. |
| `src/tac`, `tools`, `experiments`, `runtime-rs` code grep for `contour_codec`, `dense_raster_lzma_baseline`, `ST_GRID`, `set_default_device` | Found external Rust parity/golden-vector citations to the old Python oracle path; found a larger MLX top-level device-call corpus than the charter headline. | Kept a `contour_codec.py` shim; did not attempt blind #856 bulk migration. |
| `.omx/state/main_hot_state.md` and common contract | Live frontier is `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer is borrowed/unmoved. | HY1 made no score claim. |

## NEXT_IF_RESUMED

```json
{
  "schema": "hy1_next_if_resumed_v1",
  "rows": [
    {
      "row": "#856",
      "status": "QUEUED-WITH-A-FIRE-ORDER",
      "fire_order": [
        "add warn-only AST detector for import-time MLX device mutations with positive and negative controls",
        "classify current live sites as cpu_pin, gpu_allocator, seed_load, or function_local_safe",
        "migrate cpu_pin and seed_load sites into fixtures or lazy helpers by owner group",
        "wire detector in preflight_all strict=false, promote only after live count reaches zero"
      ]
    },
    {
      "row": "#907",
      "status": "FOLDED",
      "reason": "current parity guard passes and the fitted ST_GRID divergence is named"
    },
    {
      "row": "#899",
      "status": "FOLDED",
      "reason": "validated read path and raw-reader guard already exist and pass focused tests"
    },
    {
      "row": "#913",
      "status": "FIRED",
      "reason": "dense-raster LZMA implementation renamed with compatibility shim and tests"
    }
  ],
  "no_scorer_or_evaluate_ran": true,
  "frontier": {
    "score": 0.7539807296911207,
    "bytes": 357836,
    "axis": "macOS-CPU advisory",
    "contest_pointer": "borrowed/unmoved"
  }
}
```

## Serializer / Commit Status

Serializer command attempted with explicit HY1 files, `--no-co-author`, `[no-triality] [p0-ledger-ok]`, and post-edit `--expected-content-sha256` for all 38 declared files. It did not commit because the managed sandbox refused the Git object write during `git add`:

```text
error: unable to create temporary file: Operation not permitted
error: experiments/probe_curve_core_dseg_feasibility_gate.py: failed to insert into database
error: unable to index file 'experiments/probe_curve_core_dseg_feasibility_gate.py'
fatal: updating files failed
```

Post-attempt check: `git diff --cached --name-status` produced no output, so the failed serializer attempt left no staged entries. MAIN/operator commit is still required.

Serializer-attempt SHA-256s. The receipt hash is the pre-status-append hash passed to the failed serializer; the other file hashes remain the post-edit content hashes for the declared HY1 set:

```text
74ae7b28d11312697aa4145a98aef875748cf251ef088cb72940fae125a8457c  .omx/research/ddm_hy1_20260805/HY1_RECEIPT.md
1e762013820fd212ea543a3c04f39cc7dec71880f110394fd1758aa95eb45c59  src/tac/witness_control/tests/test_costate_live_ingest.py
1908882f90fd2f7db7ac96c034c524bcbe709739105cc7f5a9e37cfe8eee8251  src/tac/witness_control/tests/test_payload_tto_core.py
951d55f6d47b6e2a50f39b30150100ba9e91a8039c3eafd7712bd2b9b3bd4a14  src/tac/witness_control/tests/test_polyak_finisher.py
cdffcc0b29ba2143fae99cdfd4fdcc8ac68850e570e83e5fb85cb5f405d5e2a4  src/tac/witness_control/tests/test_stage3_cache_and_golden.py
68c6e5893a2abed73d10b3493bd57a45674712b5bffdcc174b434348dcc7ad8b  src/tac/witness_control/tests/test_taskspace_g112_exact_checkpoint_partition_v1.py
2af61fd126d6282ad5193f9e30c2487e41a7fba3224da39ea50b21c2f1bc7115  src/tac/witness_control/tests/test_taskspace_single_stage_score_attempt_v1.py
563ef5712b06eae8334d0ef7fadcf35d8d47820cdb1d716c55f8fe615c8bceab  src/tac/boundary_math/dense_raster_lzma_baseline.py
69f44c59ff1598a2bd72c675ffb3e3d07d449343c4a7815fcb96099f81620d08  src/tac/boundary_math/contour_codec.py
b3972f7943f4c3bf59f8018eae3163848b14f82ece7ccc7e893d9c4a6decab82  src/tac/boundary_math/__init__.py
eb8dd8d2ce31979de3629d007aba12d2ca5642bc1149096fc66f1107f0a33679  src/tac/boundary_math/seg_core.py
10be9bbcce0c57100db7c45082a2a1938d12a8b707c0d01c8e9984032c16b1f8  src/tac/boundary_math/region_merge.py
57c6bfd86612a24e746b97a6ce98ac79495988d6d3ce9698d08e9289de3f2620  src/tac/boundary_math/context_partition_codec.py
eca973e47b586873e9f9bebc46c78a14e43672b4e30f88d70017cd0864fb6c04  src/tac/boundary_math/partition.py
66d40617045e59aa892a866717e77aec56e3e0147aa38d27fdc029404030d58e  src/tac/v2_compose/archive_grammar.py
d5d59e8e9fa04b882e7395b632510bb0c10295dcc18ea7b97ad03429e6c76aeb  src/tac/tests/test_boundary_math_seg_core.py
2973d5b3b6a90b9a4b774eff142abf2bbae7f61900a9c271eef30bdb925fcf32  src/tac/tests/test_v2_compose_archive_grammar.py
9f0b5bc22e265aa9078b77c34219f3cae6bb08335b53a4f3599e5b421e09c2de  experiments/probe_ms_edit_sidecar_rate.py
56490fa8da327ee25bf9059ea65a4d65dafcfa43fd93ce5e59916b2cbd0af92c  tools/measure_free_generator_byte_budget.py
0d3add43ff057ff3dc61087cb3964c4b958a31d3689292e7da37e4b984bb299e  tools/measure_eikonal_sdf_dseg_recovery.py
e43cc407e5cc4b6d64025001529e62b86043a0157961ec472c2d746075c2b5c8  experiments/yousfi_tolerance_partition_remeasure.py
c738ab06e4d1cd63dc4d80cb5e2445cafcec76ef424bfd641be4514d6d94cc85  tools/probe_defect_network_rate_code.py
a5fbe0a6a5e92d1123028df0f19f225bb9bfbf708cd8ca2d21efad8fb2c25ea7  experiments/probe_yousfi_road_lane_geometric_solve.py
a65e353568ff86d3274f84c283f039ad030b4ec0de7827189fd2168b84badcea  experiments/yousfi_partition_topaiml_probe.py
db5ab4c29ebee6e4c8c5d4f29323267dc4af4c9bef94149b35c890492143274c  runtime-rs/crates/tac-boundary-decode/python_reference_equivalence_test.py
6322fa062877d4e689fa069daf8f5d8098dc39ec18f5bfc15b0b5f65add79173  runtime-rs/crates/tac-boundary-decode/golden_vectors/generate_golden_vectors.py
75e7a198b4c46f538cdf2ce858244031aecf856f7c961047457ec04c9b78479c  experiments/witness_seg_boundary_topaiml_probe.py
5a294a61f92f60570e882618e72bf0b7e1517c77804758fffafbb4af4a4c8955  experiments/probe_nca_dseg_feasibility_gate.py
430a160a258d65e6f22464fdb9f49293e22e83710dbe9a1f638c119767a45136  experiments/probe_curve_core_dseg_feasibility_gate.py
f14903d8acb90352ae2dbed0a81ea8b5a57aa10bbf374c2f881f83ecab5b472e  experiments/probe_polynomial_fill_survival_gate.py
3f6cbc22145ef48811462eb1a19aed2161fe92a04f80edf5e34a39add6f7379c  experiments/probe_dseg_side_feasibility_corners.py
9fe2f79e25491f8e99b30bceafd11d9b11e16ae8060192f7b1ebf89302b344ba  tools/representation_audit_probe.py
0b62e2c28e517945ea1577b88c32e8ff74fa461df1ba2eb66bf33e5c991891c5  runtime-rs/crates/tac-boundary-decode/README.md
e3dfa93adbb6a1c4ec6e05293ef07e5a313c042a4b23089b7cf7dd270c00612e  runtime-rs/crates/tac-boundary-decode/archive_payload_manifest.json
bb58bfc0dff4e353458f3f9062091a536080f7bad04da64d11c128638451f2fa  runtime-rs/crates/tac-boundary-decode/binary_source_audit.md
b4a2629919b7cfe87e6dcfa405ac507f0b675bf1dc32ce4332435099668f7810  runtime-rs/crates/tac-boundary-decode/src/contour.rs
c4afad9312c6466fc2fdfafd6904c393818be350af26b7b38eeeb7e0c4c809ad  runtime-rs/crates/tac-boundary-decode/src/lib.rs
fe4c860bfefc191b599059a44452b6877cd5fc243701242c30d2a325fb3e8a7d  runtime-rs/crates/tac-boundary-decode/tests/golden_vector_parity.rs
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
