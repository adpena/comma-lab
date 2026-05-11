# License audit — 2026-05-11

**Scope:** every file added since 2026-05-08 (`git log --since="2026-05-08"
--diff-filter=A`). Per CLAUDE.md "Public Disclosure Hygiene" non-negotiable +
council Q5 verdict (b) (9/10 for tag-first, publish-on-operator-trigger;
SBOM-and-parity-results-in-release-notes per Yousfi/Ballé) + operator
directive 2026-05-11 ("production hardened OSS direction").

## TL;DR

- **40 new files** added since 2026-05-08 (30 non-test code files + 10 test
  files).
- **0 violations.** Every new file complies with the project's licensing
  posture.
- **Project licensing model:** repository-level LICENSE (MIT) + per-package
  metadata (`pyproject.toml license = "MIT"`, `Cargo.toml license = "MIT OR
  Apache-2.0"` for the Rust crate per dual-license Rust ecosystem
  convention). **No SPDX per-file headers in the repo by convention** (0
  files repo-wide carry `SPDX-License-Identifier`); this audit confirms the
  new files match.
- **No retrofit recommended.** SPDX per-file headers across 40 files would
  contradict the established repo convention without OSS benefit; LICENSE +
  package-metadata is the canonical OSS-distribution mechanism for this
  project class. Recommendation surfaces in §"Recommendations" below.

## Project licensing posture (canonical)

| Surface | License declaration | Source |
|---|---|---|
| Repository root | `MIT License` | `LICENSE` |
| Python package `tac` | `license = "MIT"` | `pyproject.toml` |
| Rust crate `tac-packet-compiler` | `license = "MIT OR Apache-2.0"` | `runtime-rs/crates/tac-packet-compiler/Cargo.toml` |
| Per-file SPDX headers | NONE BY CONVENTION (0 files repo-wide) | grep -rl `SPDX-License-Identifier` |

**Why dual MIT/Apache-2.0 for the Rust crate?** Rust ecosystem standard;
dual-licensing maximizes downstream compatibility (per crates.io guidance +
council Q5 Yousfi/Ballé SBOM hygiene). MIT alone is allowed; dual-license is
preferred for Rust crates intended for crates.io publish.

## Per-file audit table

### Production code (30 files)

| File | Kind | Per-file license header? | Project license applies? | Verdict |
|---|---|---|---|---|
| `experiments/train_anr_token_renderer.py` | Python | None | YES (LICENSE + pyproject) | OK |
| `experiments/train_categorical_renderer.py` | Python | None | YES | OK |
| `experiments/train_scpp_self_compression.py` | Python | None | YES | OK |
| `runtime-rs/crates/tac-packet-compiler/src/pr101_sidecar_grammar/split_brotli.rs` | Rust | None | YES (Cargo.toml MIT/Apache-2.0) | OK |
| `runtime-rs/crates/tac-packet-compiler/src/pr103_arithmetic_coding/latent_hi.rs` | Rust | None | YES | OK |
| `runtime-rs/crates/tac-packet-compiler/src/pr93_pose_codec/lowpass_luma.rs` | Rust | None | YES | OK |
| `runtime-rs/crates/tac-packet-compiler/src/pr97_h3_grammar/mod.rs` | Rust | None | YES | OK |
| `runtime-rs/crates/tac-packet-compiler/src/pr97_h3_grammar/tile_band_streams.rs` | Rust | None | YES | OK |
| `runtime-rs/crates/tac-packet-compiler/src/sparse_packet_ir/rle_of_zeros.rs` | Rust | None | YES | OK |
| `src/tac/anr_token_renderer.py` | Python | None | YES | OK |
| `src/tac/categorical_substrate.py` | Python | None | YES | OK |
| `src/tac/foveation_field.py` | Python | None | YES | OK |
| `src/tac/hessian_block_fp.py` | Python | None | YES | OK |
| `src/tac/lapose_motion_atom_allocator.py` | Python | None | YES | OK |
| `src/tac/mdl_fp4_tto.py` | Python | None | YES | OK |
| `src/tac/raft_pose_stream.py` | Python | None | YES | OK |
| `src/tac/scpp_substrate.py` | Python | None | YES | OK |
| `submissions/anr_substrate/inflate.py` | Python | None | YES | OK |
| `submissions/anr_substrate/inflate.sh` | shell | None | YES | OK |
| `submissions/categorical_substrate/inflate.py` | Python | None | YES | OK |
| `submissions/categorical_substrate/inflate.sh` | shell | None | YES | OK |
| `submissions/pr106_foveation_field_sidecar/inflate.py` | Python | None | YES | OK |
| `submissions/pr106_foveation_field_sidecar/inflate.sh` | shell | None | YES | OK |
| `submissions/pr106_foveation_field_sidecar/src/codec.py` | Python | None | YES | OK |
| `submissions/pr106_foveation_field_sidecar/src/model.py` | Python | None | YES | OK |
| `submissions/pr106_foveation_field_sidecar/src/pr106_inner_sidecar.py` | Python | None | YES | OK |
| `submissions/pr106_lapose_atom_sidecar/inflate.py` | Python | None | YES | OK |
| `submissions/pr106_lapose_atom_sidecar/inflate.sh` | shell | None | YES | OK |
| `submissions/scpp_substrate/inflate.py` | Python | None | YES | OK |
| `submissions/scpp_substrate/inflate.sh` | shell | None | YES | OK |

### Test files (10 files; non-code-distributable)

| File | Kind | Verdict |
|---|---|---|
| `src/tac/tests/test_anr_categorical_trainers_smoke.py` | pytest | OK (project license) |
| `src/tac/tests/test_anr_token_renderer.py` | pytest | OK |
| `src/tac/tests/test_categorical_substrate.py` | pytest | OK |
| `src/tac/tests/test_foveation_field.py` | pytest | OK |
| `src/tac/tests/test_hessian_block_fp.py` | pytest | OK |
| `src/tac/tests/test_lapose_motion_atom_allocator.py` | pytest | OK |
| `src/tac/tests/test_mdl_fp4_tto.py` | pytest | OK |
| `src/tac/tests/test_raft_pose_stream.py` | pytest | OK |
| `src/tac/tests/test_scpp_substrate.py` | pytest | OK |
| `src/tac/tests/test_train_scpp_self_compression_smoke.py` | pytest | OK |

## Verification methodology

```
# 1. Enumerate files added since 2026-05-08
git log --since="2026-05-08" --diff-filter=A --name-only --pretty=format: \
  | sort -u | grep -E '\.(py|rs|sh)$'
# Result: 40 files

# 2. Verify project LICENSE exists + is MIT
cat LICENSE  # MIT License (1.1K, 2026 copyright)

# 3. Verify pyproject license metadata
grep -E "(license|License)" pyproject.toml
# Result: license = "MIT"

# 4. Verify Cargo.toml license metadata for each Rust crate
grep "^license = " runtime-rs/crates/tac-packet-compiler/Cargo.toml
# Result: license = "MIT OR Apache-2.0"

# 5. Sample SPDX prevalence repo-wide
grep -rl "SPDX-License-Identifier" \
  src/tac/ runtime-rs/ experiments/ submissions/ 2>/dev/null | wc -l
# Result: 0 (no SPDX per-file headers exist anywhere)

# 6. Sample new-file headers to confirm convention match
head -3 src/tac/anr_token_renderer.py
# "ANR TokenRendererV62 + ShrinkSingleNeRV full substrate port — Phase A scaffold."
# (module docstring; matches existing convention; NO SPDX header)
```

## Findings

- **PASS**: All 40 new files comply with the project's established licensing
  posture (LICENSE-at-root + per-package metadata, no SPDX per-file headers
  by convention).
- **CONFIRMATION**: 0 files repo-wide carry SPDX headers; this is consistent
  project-wide convention, not a gap.
- **CONFIRMATION**: All new files are inside the project root, so the
  repository LICENSE applies transitively.
- **CONFIRMATION**: All new Rust files in `tac-packet-compiler/` are covered
  by the crate's `MIT OR Apache-2.0` Cargo.toml dual-license declaration.
- **CONFIRMATION**: All new Python files in `src/tac/`, `experiments/`, and
  `submissions/` are covered by the `pyproject.toml license = "MIT"`.
- **CONFIRMATION**: All new shell scripts (3 inflate.sh files) are covered
  by the project LICENSE; shell-script SPDX headers are uncommon and not
  required by Linux ecosystem convention.

## Recommendations

### Standing (no action required)

- **Maintain LICENSE + per-package-metadata convention.** Per-file SPDX
  headers across 1000+ existing repo files would contradict the established
  posture without measurable OSS-distribution benefit. The current model is
  standard for MIT-licensed projects (e.g., NumPy, PyTorch, scikit-learn all
  use LICENSE-at-root without per-file SPDX).

### Operator-decision items (NOT auto-changed; OSS-direction work)

These are surfaced for operator review per CLAUDE.md "Public Disclosure
Hygiene". Each is a $0 documentation change but requires explicit
authorization to modify:

1. **LICENSE copyright line (`Copyright (c) 2026 OpenAI artifact output for
   user-directed scaffold`)**: this string is unusual for an OSS project; a
   typical MIT project uses `Copyright (c) 2026 <author or organization
   name>`. NOT auto-changed; operator should confirm the canonical
   copyright-holder string before any crates.io publish (per Q5 verdict
   "License + repo URL operator-review per CLAUDE.md Public Disclosure
   Hygiene non-negotiable").
2. **Cargo.toml `repository` URL** (`https://github.com/commaai/commavq-comma-video-compression-challenge`):
   this points at the upstream contest repo, NOT a fork or our own remote.
   Per Q5 verdict, the operator should confirm this is the canonical OSS
   repository URL before crates.io publish.
3. **SBOM (per Yousfi/Ballé Q5 ADD)**: a software bill of materials listing
   every Rust dependency + version pin (`constriction = "0.4"`, `brotli =
   "8"`, `liblzma = "0.4"`, `serde = "1"`, `serde_json = "1"`, `sha2 =
   "0.10"`, `hex = "0.4"`, `ndarray = "0.16"`, `half = "2"`, `proptest = "1"`,
   `criterion = "0.6"`) is the publishable-grade artifact for crates.io
   release. NOT generated in this audit (operator decision-pending whether
   to publish); should land before any `cargo publish`.

## Loop status

PAUSED per 2026-05-09 directive. No `ScheduleWakeup` outstanding.

## Cross-references

- LICENSE: `LICENSE`
- Project pyproject: `pyproject.toml` (`license = "MIT"`)
- Rust crate metadata: `runtime-rs/crates/tac-packet-compiler/Cargo.toml`
  (`license = "MIT OR Apache-2.0"`)
- Council Q5 verdict (b): `feedback_grand_council_a_b_work_plan_review_20260511.md`
- Operator directive: 2026-05-11 ("production hardened OSS direction" +
  "don't submit the PR yet")
- 19/19 GREEN landing: `feedback_rust_packet_compiler_complete_19_19_native_parity_landed_20260511.md`
- v0.2.0-rc1 release-tag work: this audit's sister landing
  `feedback_github_release_tag_license_audit_phase1_wiring_lane_sweep_landed_20260511.md`
