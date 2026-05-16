# Submission inflate.py PYTHONPATH-shim audit (Catalog #295 Part A inventory)

**Date:** 2026-05-16
**Lane:** `lane_check_submission_inflate_empty_pythonpath_strict_gate_20260516`
**Anchor:** NSCS06 v5 Modal dispatch `fc-01KRQMAQ7V41AFYMJH5HRK9P10` failed at runtime because `submissions/nscs06_carmack_hotz_strip_everything/inflate.py` used PYTHONPATH-shim that worked locally but the `src/tac/...` tree was never vendored into the submission.
**Canonical fix reference:** commit `90bca47ff` (NSCS06 v6) vendored codec package alongside as `submissions/<id>/_nscs06_codec/`.

## Methodology

Scanned every `submissions/*/inflate.py` (excluding `exact_current/` per CLAUDE.md mutation frontier; excluding `_intake_` vendored clones). For each match:

1. Detect `sys.path.insert(0, ...)` pattern (AST-aware).
2. Detect `from tac.*` / `import tac.*` imports (AST-aware).
3. Verify whether the imported `tac` package is vendored alongside as `submissions/<id>/tac/` OR `submissions/<id>/src/tac/`.
4. Trace one-level variable indirection (e.g. `SRC_DIR = HERE / "src"`).
5. Detect parent-directory traversal (`HERE.parent / "sibling" / "src"`).

## 3-column inventory (39 submission inflate.py files scanned)

### Column A: AFFECTED — NSCS06 v5 bug class active (6 substrates)

These submissions WILL FAIL on a Modal worker with empty PYTHONPATH unless either (a) the codec is vendored alongside, OR (b) the sibling submission they depend on also ships in the same dispatch packet, OR (c) the operator's repo tree is bind-mounted (which is the bug — defeats self-containment).

| # | Submission id | Failure mechanism | Recommended fix |
|---|---|---|---|
| 1 | `apogee_v2` | `sys.path.insert(0, str(SRC_DIR))` (HERE / "src") + `from tac.water_filling_codec_v2 import decode_omega_w_v2` — `tac.*` not vendored alongside | Vendor `src/tac/water_filling_codec_v2.py` OR copy the helper into `src/codec.py` and remove the tac import |
| 2 | `magic_codec_pr106_r2` | Bare `from tac.*` import at line 120 with NO sys.path manipulation — silent dependency on operator's working tree | Vendor as a sibling package OR add explicit sys.path.insert + vendored src/tac/ |
| 3 | `nscs01_nullspace_split_renderer` | `sys.path.insert(0, str(vendored_src))` + `from tac.substrates.nscs01_nullspace_split_renderer.inflate import main_cli` — has fail-closed `raise RuntimeError` guard if `src/` missing (GOOD) but the trainer never vendors the package alongside (BAD; this is the EXACT NSCS06 v5 bug class anchor) | The trainer's `_write_runtime` must copy `src/tac/substrates/nscs01_nullspace_split_renderer/` AND `src/tac/substrates/_shared/inflate_runtime.py` into `submission_dir/src/tac/...` |
| 4 | `pr106_lrl1_sidechannel` | `sys.path.insert(0, str(APOGEE_SRC))` where `APOGEE_SRC = HERE.parent / "apogee_intN" / "src"` — depends on sibling submission shipping in the same dispatch packet | Either vendor PR106 codec into own `src/` (duplication) OR add `# SUBMISSION_PYTHONPATH_SHIM_OK:<rationale>` waiver declaring sibling-submission shipping requirement |
| 5 | `pr106_stacked` | `sys.path.insert(0, str(PR106_SRC))` where `PR106_SRC = HERE.parent / "pr106_latent_sidecar" / "src"` — same sibling-dep pattern | Same as #4 |
| 6 | `pr106_yshift_sidechannel` | `sys.path.insert(0, str(APOGEE_SRC))` where `APOGEE_SRC = HERE.parent / "apogee_intN" / "src"` — same sibling-dep pattern | Same as #4 |

### Column B: FIXED — Canonical self-contained pattern (16 substrates)

These vendor `codec.py` + `model.py` (+ optional helpers) directly under `submissions/<id>/src/` with NO `from tac.*` imports. The `sys.path.insert(0, str(HERE / "src"))` then `from codec import ...` pattern is the canonical NSCS06-v6-style self-contained pattern.

| # | Submission id | Vendored under | Notes |
|---|---|---|---|
| 1 | `apogee_intN` | `src/codec.py` + `src/intn_codec.py` + `src/model.py` | Canonical |
| 2 | `factorized_hnerv_v1` | `src/codec.py` + `src/model.py` | Canonical |
| 3 | `frame_exploit_selector_sidecar` | `src/codec.py` + `src/frame_selector.py` + `src/model.py` + `src/pr106_runtime.py` | Canonical |
| 4 | `hdm8_film_grain_sidecar` | `src/codec.py` + `src/model.py` + `src/pr101_grammar.py` | Canonical |
| 5 | `nscs02_downsampled_renderer` | `src/codec.py` + `src/model.py` | Canonical |
| 6 | `pr106_c3_residual_sidecar` | `src/codec.py` + `src/model.py` + `src/pr106_inner_sidecar.py` + `src/sparse_packet_ir_inline.py` | Canonical |
| 7 | `pr106_cool_chic_residual_sidecar` | Same shape as #6 | Canonical |
| 8 | `pr106_coord_mlp_residual_sidecar` | Same shape as #6 | Canonical |
| 9 | `pr106_foveation_field_sidecar` | `src/codec.py` + `src/model.py` + `src/pr106_inner_sidecar.py` | Canonical |
| 10 | `pr106_lapose_atom_sidecar` | Same shape as #9 | Canonical |
| 11 | `pr106_latent_sidecar` | `src/codec.py` + `src/model.py` | Canonical |
| 12 | `pr106_latent_sidecar_r2` | Same shape as #11 | Canonical |
| 13 | `pr106_latent_sidecar_r2_pr101_grammar` | `src/codec.py` + `src/model.py` + `src/pr101_grammar.py` | Canonical |
| 14 | `pr106_raft_pose_sidecar` | Same shape as #9 | Canonical |
| 15 | `pr106_siren_residual_sidecar` | Same shape as #6 | Canonical |
| 16 | `pr106_wavelet_residual_sidecar` | Same shape as #6 | Canonical |

### Column C: NOT_APPLICABLE — No PYTHONPATH manipulation OR pinned upstream (17 substrates)

Submissions with no `sys.path.insert(...)` pattern at all (canonical NSCS06-v6-style direct vendoring with sibling `_<id>_codec/` package OR no-codec inflate). These have no PYTHONPATH-shim bug surface.

| # | Submission id | Pattern |
|---|---|---|
| 1 | `a1` | Vendored `src/codec.py` + `src/model.py`; sys.path.insert exists but no `from tac.*` imports |
| 2 | `alpha_repair_*_*` (4 dirs) | No inflate.py (legacy archive) |
| 3 | `anr_substrate` | No PYTHONPATH-shim; direct vendoring |
| 4 | `apogee` | No PYTHONPATH-shim; legacy |
| 5 | `baseline_dilated_h64_0_90` | No PYTHONPATH-shim |
| 6 | `blocknerv_substrate` | No PYTHONPATH-shim; direct vendoring |
| 7 | `categorical_substrate` | No PYTHONPATH-shim; direct vendoring |
| 8 | `cnerv_substrate` | No PYTHONPATH-shim |
| 9 | `dsnerv_substrate` | No PYTHONPATH-shim |
| 10 | `e_nerv_substrate` | No PYTHONPATH-shim |
| 11 | `ego_nerv_substrate` | No PYTHONPATH-shim |
| 12 | `exact_current` | Pinned upstream (CLAUDE.md mutation frontier) — EXEMPT |
| 13 | `ffnerv_substrate` | No PYTHONPATH-shim |
| 14 | `hinerv_substrate` | No PYTHONPATH-shim |
| 15 | `hnerv_lc_ac` | No PYTHONPATH-shim |
| 16 | `nervdc_substrate` | No PYTHONPATH-shim |
| 17 | `nscs03_end_to_end_balle_joint_codec` | No PYTHONPATH-shim (per recent NSCS03 _full_main landing) |
| 18 | `nscs06_carmack_hotz_strip_everything` | v6 fix: `sys.path.insert(0, str(HERE))` + vendored `_nscs06_codec/` package alongside — CANONICAL REFERENCE PATTERN |
| 19 | `pr103_pr106_final_runtime` | No PYTHONPATH-shim |
| 20 | `pr106_c3_residual_sidecar` ... | (See Column B for vendored cases) |
| 21 | `pr106_hdm3_*` / `pr106_hdm4_*` | No PYTHONPATH-shim |
| 22 | `pr106_pr101grammar_*` | No PYTHONPATH-shim |
| 23 | `proven_037_unlimited_compute` | Legacy archive |
| 24 | `robust_current` | Track B mutation frontier |
| 25 | `scpp_substrate` | No PYTHONPATH-shim |
| 26 | `tcnerv_substrate` | No PYTHONPATH-shim |

## Sister-subagent backfill wave (op-routable)

The 6 AFFECTED substrates should be addressed by a follow-on subagent:

### Priority 1: NSCS01 (canonical bug-class anchor sister of NSCS06 v5)

Update `experiments/train_substrate_nscs01_nullspace_split_renderer.py::_write_runtime` to copy:
- `src/tac/substrates/nscs01_nullspace_split_renderer/__init__.py`
- `src/tac/substrates/nscs01_nullspace_split_renderer/inflate.py` (+ any deps)
- `src/tac/substrates/_shared/inflate_runtime.py`

into `submission_dir/src/tac/substrates/...` so the existing `sys.path.insert(0, str(vendored_src))` line resolves. The fail-closed `raise RuntimeError` guard already handles the missing case correctly; we just need to make sure the trainer ALWAYS produces the vendored tree.

### Priority 2: APOGEE_V2 (single helper missing)

`apogee_v2/inflate.py` imports `tac.water_filling_codec_v2.decode_omega_w_v2`. Either:
- (a) Copy the function into `submissions/apogee_v2/src/codec.py` and remove the tac import.
- (b) Vendor `src/tac/water_filling_codec_v2.py` into `submissions/apogee_v2/src/tac/water_filling_codec_v2.py` + add empty `__init__.py`.

Option (a) is simpler and more reviewable per HNeRV parity L4 (≤100 LOC inflate budget).

### Priority 3: MAGIC_CODEC_PR106_R2

`from tac.*` import at line 120 with NO sys.path manipulation = relies on operator's working tree. Investigate which `tac.*` symbol is imported and apply the same fix shape as APOGEE_V2.

### Priority 4: PR106 sister-dependent triplet

`pr106_lrl1_sidechannel` + `pr106_stacked` + `pr106_yshift_sidechannel` all `sys.path.insert(...)` into a SIBLING submission's `src/`. The recommendation is to either:
- (a) Duplicate the PR106 codec into each sibling's own `src/` (NSCS06-v6-style; preferred for OSS-release self-containment).
- (b) Add an explicit `# SUBMISSION_PYTHONPATH_SHIM_OK:<rationale>` waiver documenting "sibling-submission shipping requirement; dispatch packet manifest declares the dependency". This is acceptable for the LAB dispatch flow where both submissions are mounted; it is NOT acceptable for the OSS-release path where a single submission must be self-contained.

The 6-violation live count is fixed once the priorities above are addressed; strict-flip per CLAUDE.md "Strict-flip atomicity rule" follows the sister-backfill commit batch.

## Cross-references

- CLAUDE.md "Meta-bug class catalog" Catalog #295 row.
- Sister Catalog #205 (`check_inflate_py_uses_canonical_select_inflate_device`) — same `submissions/*/inflate.py` scan surface, different bug class (inline device-fork).
- HNeRV parity discipline lesson 9 (Runtime closure) — dependency closure failures are runtime blockers.
- 8th forbidden pattern (research-substrate trap) — applied at the runtime self-containment surface.
- NSCS06 v6 fix commit `90bca47ff` — canonical reference pattern for vendored sibling codec package.
