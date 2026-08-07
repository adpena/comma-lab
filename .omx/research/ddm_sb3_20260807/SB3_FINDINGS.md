# SB3 Findings - #983 CI-Blind MLX Runtime Triage

## Verdict

Outcome: FIXED for the local CI-blind failure class in
`src/tac/tests/test_compact_renderer_mlx_spine_runner.py`.

The original SIGBUS did not reproduce on this host. The reproduced failure is the
#856 environment class: `mlx.core` is importable, but array allocation and
`mx.random.seed` both raise `[metal::load_device] No Metal device available`.
The targeted HiNeRV execute smoke now xfails with a named owner before entering
the real MLX adapter path, instead of failing the suite or reaching lazy MLX
evaluation in a no-Metal sandbox.

No scorer, `upstream/evaluate.py`, remote dispatch, Metal training, or score
authority ran.

## Reproduction

Direct one-pair/one-epoch subprocess reproduction of
`execute_hi_nerv_mlx_scoreaware_and_adapt` returned:

```text
mode = hi_nerv_mlx_scoreaware_failed
failure = RuntimeError('[metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not accessible.')
blockers = ["hi_nerv_mlx_scoreaware_or_export_failed"]
```

Receipt:
`repro_hinerv_execute_failure_summary_20260807.json`
sha256 `89870402e3c2853e94655803c1be4c4778762fa16924b893526a4641ad511f5f`,
bytes `1143`.

Minimal MLX runtime probe:

```text
import_only: returncode=0, but atexit reports [metal::load_device]
array_eval_no_set: returncode=1 at mx.array([1.0])
seed: returncode=1 at mx.random.seed(0)
```

Receipt:
`probe_mlx_runtime_no_metal_20260807.txt`
sha256 `62ac023ab728bc11f7fad45e4aa5abd682834f1f9251186c23250f7812e24c9a`,
bytes `1448`.

Raw debug payload custody:

- `.omx/research/ddm_sb3_20260807/repro_hinerv_execute_payload_20260807.json`
  was externalized to
  `/Volumes/VertigoDataTier/pact/ddm_sb3_20260807/repro_hinerv_execute_payload_20260807.json`;
  bytes `20202982`, sha256
  `0650ae0755764c41b7fb7c2db069355e49f1b07f10ee6ff9db1972e315e97bd8`.
- `.omx/research/ddm_sb3_20260807/repro_run_hinerv_execute/compact_renderer_mlx_spine_runner_report.json`
  was externalized to
  `/Volumes/VertigoDataTier/pact/ddm_sb3_20260807/compact_renderer_mlx_spine_runner_report.json`;
  bytes `20202832`, sha256
  `5f83a343a8007551f3281116273192b968fd730faa9ebba633bafc9beb76653d`.

Reason: both files are raw generated debug payloads from the same direct
`execute_hi_nerv_mlx_scoreaware_and_adapt` one-pair/one-epoch reproduction.
The committed summary receipt above preserves the failure, blockers, upstream
snapshot hashes, and no-score-claim flag; the raw payloads were moved to the
SSD tier rather than deleted.

## Diagnosis

`src/tac/substrates/_shared/mlx_score_aware/adapter.py:5516` is not an mmap,
NPZ, or fp16 alignment dereference. It is telemetry scalar materialization:
the adapter iterates loss parts, calls `mx.eval(value)`, then `value.item()`;
line 5516 participates in the weighted `pose_distill_train_loss` calculation.

On this host the test does not reach that exact adapter line after a clean MLX
allocation. It fails earlier in the same runtime class: `tools/run_compact_renderer_mlx_spine_runner.py:17056`
calls `mx.arange(...)` during output-head contrast init, and the minimal probe
shows any `mx.array(...)` allocation raises the same no-Metal RuntimeError. The
SIGBUS report is therefore not reproduced here; the local mechanism is
importable-MLX / unusable-Metal, matching HY1's #856 precedent.

Verdict scope: FORMULATION / ENVIRONMENT for this host and this test module.
This is not a family kill of the MLX adapter and not evidence about a real
Metal-backed host.

## Cure

Changed `src/tac/tests/test_compact_renderer_mlx_spine_runner.py` only:

- Replaced the import-time `import mlx.core` availability probe with
  `importlib.util.find_spec("mlx.core")`, so merely collecting the test module
  does not load MLX.
- Added `_require_mlx_runtime_or_xfail()`, which imports MLX lazily and performs
  one real allocation. If allocation raises the exact no-Metal RuntimeError, it
  calls `pytest.xfail` with owner `#856 known-red environment/MLX-gating`.
- Added a pure recognizer regression,
  `test_mlx_runtime_gate_recognizes_no_metal_error`.
- Applied the runtime gate to the real-runtime sibling tests in this module,
  including the #983 HiNeRV execute smoke.

No adapter or production runner behavior changed. A real operator launch still
fails closed if MLX cannot allocate.

## Sibling Sweep

Targeted sweep in `test_compact_renderer_mlx_spine_runner.py` found the real
MLX runtime test surface:

- `test_hinerv_execute_runs_training_archive_and_receiver_proof`
- `test_hinerv_live_birth_hysteresis_probe_restores_model_state`
- `test_hinerv_live_birth_survival_writes_four_arm_rows_when_birth_not_accepted`
- `test_hinerv_private_smoke_defaults_to_full_target_hydration_for_hard_pairs`
- `test_hinerv_private_smoke_forwards_explicit_pr95_curriculum_total_epochs`
- `test_hinerv_private_smoke_generates_startup_section_telemetry_for_qat_terms`

All six runtime tests now share the same #856 xfail gate on this no-Metal host.
Post-edit `rg` found no remaining `import mlx.core` or
`pytest.importorskip("mlx.core")` in the module.

Broader corpus sweep was bounded to this charter's named surface. HY1 already
queued the wider import-time MLX device mutation audit with a fire order; SB3
does not claim that broader corpus is closed.

## Recall Evidence

| query / surface | finding beyond charter seeds | plan change |
|---|---|---|
| `MEMORY.md` for `#856`, `metal::load_device`, MLX no-device | Memory records the same stop rule: no usable Metal device means keep work partial and move GPU proof to a real Metal-backed surface. | Treated no-Metal as environment, not an adapter code fix. |
| `.omx/research/ddm_hy1_20260805/HY1_RECEIPT.md` | HY1 closed visible MLX failures as named #856 xfails; it observed ignored MLX nanobind atexit RuntimeError after xfail. | Reused named xfail pattern and did not hide the environment class as a silent skip. |
| `tools/list_canonical_equations.py --json` filtered for MLX/Metal/SIGBUS/adapter | Found MLX/torch parity and MLX determinism equations, but no SB3-specific equation governing this failure. | No equation registry edit. |
| `rg` in target test + runner + adapter | Test importability gate was weaker than runtime usability; runner has intended internal MLX imports; adapter line 5516 is lazy scalar telemetry. | Patched tests, not production adapter/runner. |
| `.omx/state/main_hot_state.md` | Live own-vehicle frontier is `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer is borrowed/unmoved. | No score claim. |

## Verification

```bash
PYTHONFAULTHANDLER=1 .venv/bin/python -X faulthandler -m pytest --basetemp=.omx/research/ddm_sb3_20260807/pytest_basetemp_verify \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_mlx_runtime_gate_recognizes_no_metal_error \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_live_birth_hysteresis_probe_restores_model_state \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_live_birth_survival_writes_four_arm_rows_when_birth_not_accepted \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_private_smoke_defaults_to_full_target_hydration_for_hard_pairs \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_private_smoke_forwards_explicit_pr95_curriculum_total_epochs \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_private_smoke_generates_startup_section_telemetry_for_qat_terms \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_execute_runs_training_archive_and_receiver_proof -q
```

Result: `1 passed, 6 xfailed in 2.22s`.
Receipt:
`verify_affected_mlx_runtime_gate_20260807.txt`
sha256 `cbb2263ef7670e483c28c1f58df601e7b00308830484791572190b65efda5608`,
bytes `359`.

```bash
.venv/bin/python -m ruff check src/tac/tests/test_compact_renderer_mlx_spine_runner.py
```

Result: `All checks passed!`.
Receipt:
`ruff_test_compact_renderer_mlx_spine_runner_20260807.txt`
sha256 `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18`,
bytes `19`.

```bash
git diff --check -- src/tac/tests/test_compact_renderer_mlx_spine_runner.py
```

Result: clean.

Review tracker: `src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
marked reviewed twice after edit.

## Follow-Ons

FIRED: #983 local failure class in the target module is gated with a named #856
xfail and focused verification.

QUEUED-WITH-A-FIRE-ORDER: wider #856 corpus audit remains HY1's queue item:
warn-only AST detector for import-time MLX device mutations, classify live
sites, migrate owner groups, then wire strict only after count reaches zero.

## Frontier Boundary

No score row moved. Own-vehicle frontier remains
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer
`0.19108` remains borrowed/unmoved.
