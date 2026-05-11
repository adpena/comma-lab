# Wiring + integration check — 2026-05-11

**Scope:** comprehensive wiring + integration check across today's 40+ new
files (per the license-audit file enumeration). Per operator directive
2026-05-11 ("keep pushing the compiler and wiring and integration and
everything"). Sister to U's full-stack integration audit; this check focuses
on the wiring layer specifically — module imports, --help parsing, shell
parse, format_id declarations, and `__init__.py` re-export coverage.

## TL;DR

| Check | Status | Count | Notes |
|---|---|---|---|
| **1. tac.* module imports** | PASS | 10/10 | All 10 new modules import cleanly + carry `__all__` |
| **2. trainer --help parses** | PASS | 4/4 | All new + the wired-up T1 trainer parse cleanly |
| **3. inflate.sh `bash -n`** | PASS | 5/5 | All 5 new submission inflate scripts parse |
| **4. inflate.py compile** | PASS | 8/8 | All 8 new submission Python files compile |
| **5. format_id declarations** | PARTIAL — see §5 | 3 of 5 | 2 substrates declare format_id only in lane registry, not in submission code |
| **6. tac/__init__.py re-export coverage** | INTENTIONALLY NARROW | 0/10 | New scaffold modules accessible by direct import; per CLAUDE.md "narrow public API" discipline |

## Check 1 — `tac.*` module imports (10/10 PASS)

| Module | Import OK | Has `__all__` |
|---|---|---|
| `tac.anr_token_renderer` | YES | YES |
| `tac.categorical_substrate` | YES | YES |
| `tac.foveation_field` | YES | YES |
| `tac.hessian_block_fp` | YES | YES |
| `tac.lapose_motion_atom_allocator` | YES | YES |
| `tac.mdl_fp4_tto` | YES | YES |
| `tac.raft_pose_stream` | YES | YES |
| `tac.scpp_substrate` | YES | YES |
| `tac.kl_pose_distill` | YES | YES |
| `tac.temporal_consistency_regularizer` | YES | YES |

```
$ .venv/bin/python -c "import tac.anr_token_renderer; ..."
# 10/10 succeed
```

## Check 2 — trainer `--help` parses (4/4 PASS)

| Trainer | rc | Notes |
|---|---|---|
| `experiments/train_anr_token_renderer.py` | 0 | OK |
| `experiments/train_categorical_renderer.py` | 0 | OK |
| `experiments/train_scpp_self_compression.py` | 0 | OK |
| `experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py` | 0 | OK; new flags (`--enable-t20-kl-pose-distill`, `--enable-t22-temporal-consistency`, etc.) parse correctly |

The Phase 1 T1 trainer's new T20/T22 flags were wired in this same landing
batch (Deliverable 3); the `--help` parse confirms `parse_args` accepts the
4 new T20 flags and 3 new T22 flags without error. Verified via
introspection:

```
parse_args OK
  enable_t13_sqrt_n_budget: False
  enable_t19_adaptive_rho: False
  enable_t20_kl_pose_distill: False
  t20_temperature: 2.0
  t20_weight_pose: 1.0
  t20_mode: distill_softmax_full
  enable_t22_temporal_consistency: False
  t22_lambda_weight: 0.1
  t22_boundary_handling: border
```

## Check 3 — `inflate.sh bash -n` (5/5 PASS)

All 5 new submission inflate.sh scripts pass bash syntax-only parse:

```
bash -n submissions/anr_substrate/inflate.sh                    # rc=0
bash -n submissions/categorical_substrate/inflate.sh            # rc=0
bash -n submissions/pr106_foveation_field_sidecar/inflate.sh    # rc=0
bash -n submissions/pr106_lapose_atom_sidecar/inflate.sh        # rc=0
bash -n submissions/scpp_substrate/inflate.sh                   # rc=0
```

## Check 4 — `inflate.py` Python compile (8/8 PASS)

All 8 new submission Python files compile cleanly:

```
python -m py_compile submissions/anr_substrate/inflate.py                     # rc=0
python -m py_compile submissions/categorical_substrate/inflate.py             # rc=0
python -m py_compile submissions/pr106_foveation_field_sidecar/inflate.py     # rc=0
python -m py_compile submissions/pr106_foveation_field_sidecar/src/codec.py   # rc=0
python -m py_compile submissions/pr106_foveation_field_sidecar/src/model.py   # rc=0
python -m py_compile submissions/pr106_foveation_field_sidecar/src/pr106_inner_sidecar.py  # rc=0
python -m py_compile submissions/pr106_lapose_atom_sidecar/inflate.py         # rc=0
python -m py_compile submissions/scpp_substrate/inflate.py                    # rc=0
```

## Check 5 — format_id declarations (PARTIAL)

Per the operator directive's wiring spec, every new format_id should be in
the registry. Inspection of the 5 new submission directories:

| Submission | format_ids found in code | Lane-registry declaration | Status |
|---|---|---|---|
| `submissions/anr_substrate` | NONE_DECLARED | (lane registry: format_id not declared in lane notes) | DEFER — scaffold-only; will declare on first dispatch |
| `submissions/categorical_substrate` | NONE_DECLARED | (lane registry: format_id not declared in lane notes) | DEFER — scaffold-only |
| `submissions/pr106_foveation_field_sidecar` | `0x01`, `0x02`, `0x30` | lane registry says `format_id 0x30` | OK — registry matches code |
| `submissions/pr106_lapose_atom_sidecar` | `0x01`, `0x02`, `0x32` | lane registry says `format_id 0x32` | OK |
| `submissions/scpp_substrate` | `0x02`, `0x40` | lane registry says (none yet) | OK — code declares; lane should be updated |

**Action items (not blocking; surface for follow-up):**
- 2 substrates (`anr_substrate`, `categorical_substrate`) lack format_id in
  both code and lane registry. These are SKETCH-state scaffolds; format_id
  assignment can happen on first L2 INTEGRATION promotion. Per CLAUDE.md
  HNeRV parity discipline, every Level 1+ representation lane MUST declare
  `archive_grammar / parser_section_manifest / ...` (Catalog #124 strict
  preflight); the format_id falls under `parser_section_manifest`.
- `scpp_substrate` has `format_id 0x40` in code; lane registry's notes
  string should be updated to declare it explicitly (low-priority cleanup).

## Check 6 — `tac/__init__.py` re-export coverage (INTENTIONALLY NARROW)

`tac/__init__.py` exposes only the **canonical lazy public API** (8 symbols:
`Trainer`, `TrainConfig`, `build_postfilter`, `build_renderer`,
`ScoreResult`, `CheckpointMeta`, `AveragedCheckpoint`, `SensitivityResult`).
The 10 new scaffold modules are accessible via direct module imports
(`from tac.anr_token_renderer import ...`) but are NOT re-exported.

**This is intentional per CLAUDE.md "Beauty, simplicity, and developer
experience" non-negotiable**: "make the interface narrow and explicit."
Adding 10+ scaffold-only modules to the top-level `__init__.py` would:
- bloat `import tac` time (heavy torch deps eagerly imported)
- pollute the public API with research-state surfaces
- violate the established lazy-loading pattern (`_LAZY_PUBLIC_API` dict)

**No retrofit needed.** Verified-working access pattern:

```python
# Each new module is direct-importable
from tac.anr_token_renderer import TokenRendererV62
from tac.kl_pose_distill import apply_kl_pose_distill
from tac.temporal_consistency_regularizer import apply_temporal_consistency
```

The `Trainer` / `TrainConfig` / `build_postfilter` / `build_renderer`
canonical entry points remain the public API per the existing convention.

## Verification command transcript

```
$ .venv/bin/python tools/lane_maturity.py validate
OK — 348 lane(s) validated cleanly.

$ .venv/bin/python -m pytest tests/paradigm_delta_epsilon_zeta/test_train_renderer_t13_sqrt_n_budget_integration.py tests/paradigm_delta_epsilon_zeta/test_train_renderer_t19_adaptive_rho_integration.py -q
42 passed in 0.53s
```

## Cross-references

- License audit: `.omx/research/license_audit_20260511.md`
- Lane registry sweep: `.omx/research/lane_registry_stale_sweep_20260511.md`
- T13/T19 prior wire-in landing: `feedback_t13_t19_phase1_trainer_integration_landed_20260509.md`
- T20/T22 module landings: `feedback_t20_t22_pose_axis_temporal_lateral_leaps_landed_20260509.md`
- Sister U integration audit: `feedback_full_stack_integration_audit_landed_20260511.md`
