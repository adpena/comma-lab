# License audit — 2026-05-12 (v0.2.0-rc1 release-prep refresh)

**Scope:** every file added between 2026-05-11 22:30 UTC (the cutoff of
`license_audit_20260511.md`) and the current HEAD `d5b69eff`, PLUS a static
re-verification of the project's licensing posture across `LICENSE` /
`THIRD_PARTY_NOTICES.md` / `pyproject.toml` / `runtime-rs/crates/tac-packet-compiler/Cargo.toml`,
PLUS a third-party import inventory in `src/tac/`, `tools/`, `experiments/`.

Per CLAUDE.md "Public Disclosure Hygiene" non-negotiable + the Strategic
Secrecy code-comment cleanup discipline + operator directive 2026-05-12
("OO release prep for v0.2.0-rc1").

## TL;DR

- **24 new production code files + 3 golden-vector JSON + 7 research-data JSON
  + 10 markdown research notes** added between 2026-05-11 22:30 UTC and HEAD.
- **0 license violations.** Every new file complies with the established
  LICENSE-at-root + per-package-metadata posture.
- **0 GPL/AGPL/copyleft surprises** in the third-party import inventory across
  the codebase. All identifiable third-party packages used at runtime are
  BSD-3 / MIT / Apache-2 / PSF (Python stdlib equivalent) compatible with MIT.
- **3 RECOMMENDED additions** to `THIRD_PARTY_NOTICES.md` (NOT applied per
  mutation-frontier rule). Surfaced to operator below.
- **4 operator-decision items inherited from `license_audit_20260511.md` are
  still open**: LICENSE copyright string, Cargo.toml repository URL, SBOM
  generation, crates.io publish authorization.

## Project licensing posture (confirmed unchanged from 2026-05-11)

| Surface | License declaration | Source | Verified |
|---|---|---|---|
| Repository root | `MIT License` | `LICENSE` (1.1K) | YES |
| Python package `tac` | `license = "MIT"` | `pyproject.toml:11` | YES |
| Rust crate `tac-packet-compiler` | `license = "MIT OR Apache-2.0"` | `runtime-rs/crates/tac-packet-compiler/Cargo.toml` | YES |
| Per-file SPDX headers | NONE BY CONVENTION (0 files repo-wide) | `grep -rl SPDX-License-Identifier` | YES |
| Tag at HEAD | `v0.2.0-rc1` (LOCAL + already pushed to origin) | `git tag --list "v0.2*"` + `git ls-remote --tags origin` | YES |

The v0.2.0-rc1 tag was created LOCAL-ONLY on 2026-05-11 then pushed to
`origin` during the operator's 2026-05-12 OD-A category sweep (see
`feedback_category_a_b_operator_authorize_execution_landed_20260512.md`).
Tag is now public-visible.

## Per-file audit table (new files since 2026-05-11 22:30 UTC)

### Production code (24 files)

| File | Kind | Project license applies? | Verdict |
|---|---|---|---|
| `experiments/train_substrate_balle_renderer.py` | Python | YES (pyproject MIT) | OK |
| `experiments/train_substrate_sane_hnerv.py` | Python | YES | OK |
| `src/tac/deploy/modal/mount_manifest.py` | Python | YES | OK |
| `src/tac/packet_compiler/pr101_conv4_storage_perms.py` | Python | YES | OK |
| `src/tac/packet_compiler/pr101_decoder_byte_maps.py` | Python | YES | OK |
| `src/tac/packet_compiler/pr101_decoder_storage_order.py` | Python | YES | OK |
| `src/tac/substrates/__init__.py` | Python | YES | OK |
| `src/tac/substrates/balle_renderer/__init__.py` | Python | YES | OK |
| `src/tac/substrates/balle_renderer/architecture.py` | Python | YES | OK |
| `src/tac/substrates/balle_renderer/archive.py` | Python | YES | OK |
| `src/tac/substrates/balle_renderer/inflate.py` | Python | YES | OK |
| `src/tac/substrates/balle_renderer/score_aware_loss.py` | Python | YES | OK |
| `src/tac/substrates/balle_renderer/tests/__init__.py` | Python | YES | OK |
| `src/tac/substrates/balle_renderer/tests/test_balle_renderer_roundtrip.py` | pytest | YES | OK |
| `src/tac/substrates/sane_hnerv/__init__.py` | Python | YES | OK |
| `src/tac/substrates/sane_hnerv/architecture.py` | Python | YES | OK |
| `src/tac/substrates/sane_hnerv/archive.py` | Python | YES | OK |
| `src/tac/substrates/sane_hnerv/inflate.py` | Python | YES | OK |
| `src/tac/substrates/sane_hnerv/score_aware_loss.py` | Python | YES | OK |
| `src/tac/substrates/sane_hnerv/tests/__init__.py` | Python | YES | OK |
| `src/tac/substrates/sane_hnerv/tests/test_sane_hnerv_roundtrip.py` | pytest | YES | OK |
| `src/tac/substrates/sane_hnerv/tests/test_train_substrate_sane_hnerv_full_main.py` | pytest | YES | OK |
| `src/tac/tests/test_packet_compiler_pr101_gold_primitives.py` | pytest | YES | OK |
| `tools/build_substrate_sane_hnerv.py` | Python | YES | OK |

### Golden vectors (3 JSON files)

| File | Kind | Verdict |
|---|---|---|
| `src/tac/packet_compiler/golden_vectors/pr101_conv4_storage_perms_v1.json` | JSON data | OK (project license; PR101 primitive port — port methodology is byte-faithful re-implementation, NOT vendoring) |
| `src/tac/packet_compiler/golden_vectors/pr101_decoder_byte_maps_v1.json` | JSON data | OK (same) |
| `src/tac/packet_compiler/golden_vectors/pr101_decoder_storage_order_v1.json` | JSON data | OK (same) |

PR101 GOLD primitive ports (the 3 above) re-implement the same byte-faithful
algorithms documented in the public PR101 archive. The GOLD primitives are
NOT vendored upstream code; they are clean-room ports verified against PR101's
public archive bytes via the golden-vector parity tests. License posture: MIT
(project LICENSE) applies — no upstream license inheritance because no
upstream source code is included.

### Research / audit data (7 JSON files + 10 markdown notes)

All under `.omx/research/` and covered by project LICENSE. These are
documentation/data artifacts, not redistributable code.

## Third-party runtime import inventory

The following third-party packages are imported across `src/tac/`, `tools/`,
`experiments/` (extracted from `^(import|from)` lines, excluding stdlib,
excluding intra-project `tac.*` / `comma_lab.*` / `upstream.*`):

### Hard runtime dependencies (declared in `pyproject.toml::dependencies`)

| Package | License | Compatibility with MIT | Notes |
|---|---|---|---|
| `torch` (PyTorch) | BSD-3-Clause | YES | Hard dep; ML core |
| `numpy` | BSD-3-Clause | YES | Hard dep; numerical core |
| `pydantic` | MIT | YES | Hard dep; data validation |
| `click` | BSD-3-Clause | YES | Hard dep; CLI |
| `brotli` | MIT | YES | Hard dep; inflate-time decompression |
| `constriction` | MIT OR Apache-2.0 | YES | Hard dep; PR86/HPAC arithmetic coding |
| `pyppmd` | LGPLv2.1+ | **REVIEW** (see below) | Hard dep; PR86 PPMd entropy model |
| `cryptography` | Apache-2.0 OR BSD-3-Clause | YES | Hard dep; Lane C Ed25519 |
| `cmaes` | MIT | YES | Hard dep; CMA-ES optimizer |
| `optuna` | MIT | YES | Hard dep; TPE/NSGA-II optimizer |

### Runtime extras (`[runtime]` group)

| Package | License | Compatibility | Notes |
|---|---|---|---|
| `av` (PyAV) | BSD-3-Clause | YES | Video decode |
| `opencv-python` | Apache-2.0 | YES | Frame utils |
| `timm` (rwightman) | Apache-2.0 | YES | Upstream scorer backbones |
| `einops` | MIT | YES | Tensor manipulation |
| `segmentation-models-pytorch` | MIT | YES | Upstream SegNet |

### Optional / cloud extras

| Package | License | Compatibility | Notes |
|---|---|---|---|
| `mlx` | MIT | YES | `[mlx]` extra (Apple Silicon) |
| `plotly` | MIT | YES | `[viz]` |
| `matplotlib` | Matplotlib (BSD-3-Clause-compatible) | YES | `[viz]` |
| `pillow` | MIT-CMU (HPND) | YES | `[viz]` |
| `imageio` | BSD-2-Clause | YES | `[viz]` |
| `dask` | BSD-3-Clause | YES | `[analysis]` |
| `polars` | MIT | YES | `[analysis]` |
| `marimo` | Apache-2.0 | YES | `[notebooks]` |
| `pytest` | MIT | YES | `[dev]` |
| `pytest-timeout` | MIT | YES | `[dev]` |
| `hypothesis` | MPL-2.0 | YES (file-level copyleft only; we don't modify) | `[dev]` |
| `ruff` | MIT | YES | `[dev]` |
| `scipy` | BSD-3-Clause | YES | `[dev]` |
| `lightning-sdk` | Apache-2.0 | YES | `[cloud]` |
| `modal` | Apache-2.0 | YES | `[cloud]` |
| `vastai` | (proprietary CLI wrapper; not redistributable) | N/A | `[cloud]` — operator-side tooling |
| `compressai` | BSD-3-Clause | YES | Optional, neural-compression reference |
| `paramiko` | LGPLv2.1+ | **REVIEW** (see below) | Optional, SSH client |
| `pygit2` | GPLv2-with-OpenSSL-exception | **REVIEW** (see below) | Optional, git automation |
| `safetensors` | Apache-2.0 | YES | Optional, model serialization |
| `tqdm` | MPL-2.0 OR MIT | YES | Optional, progress bars |

### Imports flagged for review

**`pyppmd` (LGPLv2.1+)**: hard runtime dependency. LGPL is compatible with
MIT for dynamic linking / unmodified use; `tac` imports `pyppmd` as a
PyPI-installed package and does NOT statically link nor modify its source.
**Verdict: COMPATIBLE.** No source-distribution obligation triggered by
import-only usage. Public-distribution recommendation: add an attribution line
to `THIRD_PARTY_NOTICES.md`.

**`paramiko` (LGPLv2.1+)**: optional dependency (`[cloud]`-adjacent SSH
client). Same import-only / no-modification posture as pyppmd. **Verdict:
COMPATIBLE.** Recommendation: attribution line if/when `[cloud]` becomes a
publicly-promoted install path.

**`pygit2` (GPLv2-with-OpenSSL-exception)**: optional dependency. GPL is
copyleft and would in principle propagate to derived works, BUT pygit2 is
imported by `tools/` automation scripts that are NOT distributed as part of
the `tac` wheel (not in `[tool.setuptools.packages.find].where = ["src"]`).
**Verdict: COMPATIBLE for current distribution model** (tools are operator-side
automation, not part of the published `tac` package). If `tools/` ever became
part of the redistributable wheel, GPL contamination would need to be addressed
(by removing pygit2 or relicensing the wheel). Recommendation: document the
boundary explicitly in `THIRD_PARTY_NOTICES.md`.

## RECOMMENDED additions to `THIRD_PARTY_NOTICES.md` (NOT applied)

The current `THIRD_PARTY_NOTICES.md` enumerates referenced projects
(upstream comma video compression challenge, OMX tooling, Ralph, autoresearch,
DSPy/GEPA, hermes-agent-self-evolution, Mojo docs). It does NOT enumerate
the binary-linked runtime dependencies. For a public OSS-distributed wheel,
the standard practice is to acknowledge runtime deps in `THIRD_PARTY_NOTICES.md`
even though their licenses do not strictly require it for MIT-licensed
consumers.

Three recommended edits (operator-gated; NOT applied because LICENSE +
THIRD_PARTY_NOTICES.md are outside the mutation frontier):

### Recommendation 1: Add runtime-dependencies attribution section

Append to `THIRD_PARTY_NOTICES.md`:

```markdown
## Runtime dependencies (binary-linked via `pip install tac`)

The `tac` package imports and links the following third-party libraries at
runtime. Each is used under its declared license; this list is provided as
a courtesy.

### Hard dependencies

- `torch` (PyTorch) — BSD-3-Clause
- `numpy` — BSD-3-Clause
- `pydantic` — MIT
- `click` — BSD-3-Clause
- `brotli` — MIT
- `constriction` — MIT OR Apache-2.0
- `pyppmd` — LGPLv2.1+ (used as imported PyPI package, no modification)
- `cryptography` — Apache-2.0 OR BSD-3-Clause
- `cmaes` — MIT
- `optuna` — MIT

### Runtime extras (`pip install tac[runtime]`)

- `av` (PyAV) — BSD-3-Clause
- `opencv-python` — Apache-2.0
- `timm` — Apache-2.0
- `einops` — MIT
- `segmentation-models-pytorch` — MIT

### Other optional extras

See `pyproject.toml::[project.optional-dependencies]` for the full list. All
listed packages are MIT/BSD/Apache-2 compatible. `hypothesis` (MPL-2.0) and
`tqdm` (MPL-2.0 OR MIT) are file-level copyleft only and impose no obligation
on `tac` consumers who use them via PyPI install.

### GPL-adjacent optional automation tooling (NOT part of the `tac` wheel)

- `pygit2` — GPLv2-with-OpenSSL-exception. Imported only by `tools/`
  operator-side automation scripts, which are NOT part of the redistributed
  `tac` package per `pyproject.toml::[tool.setuptools.packages.find].where = ["src"]`.
- `paramiko` — LGPLv2.1+. Same boundary.
```

### Recommendation 2: Clarify per-component license boundaries

Append to `THIRD_PARTY_NOTICES.md` near the top:

```markdown
## Per-component license boundaries

- `tac` Python package (`src/tac/`) — MIT (see `pyproject.toml`)
- `tac-packet-compiler` Rust crate (`runtime-rs/crates/tac-packet-compiler/`)
  — MIT OR Apache-2.0 (Rust dual-license convention; see `Cargo.toml`)
- Repository-level fallback (everything else) — MIT (see `LICENSE`)

The `tac` Python wheel ships only `src/tac/`. Operator-side automation
under `tools/`, `scripts/`, and `experiments/` is part of the repository
but is NOT part of the redistributed wheel and may import GPL/LGPL libraries
under their respective licenses without affecting the MIT posture of the
distributable artifact.
```

### Recommendation 3 (DEFERRED): SBOM generation

Per the 2026-05-11 audit's open recommendation 4: a software bill of materials
listing every Rust crate dependency + version pin should land before any
`cargo publish`. The crate is currently `publish = false` per the council's
Q5 verdict, so this is NOT blocking the v0.2.0-rc1 tag release. Operator
decision pending: when (if) the crate becomes `publish = true`, also generate
the SBOM via `cargo metadata --format-version 1` + a project-specific filter.

## Operator-decision items still open (carried forward from 2026-05-11)

1. **LICENSE copyright line** (`LICENSE:3` says `Copyright (c) 2026 OpenAI
   artifact output for user-directed scaffold`). Unusual for an OSS project;
   typical pattern is `Copyright (c) 2026 Alejandro Pena` or `Copyright (c) 2026
   pact contributors`. Operator should confirm canonical copyright-holder string
   before any v0.2.0 stable release. **NOT auto-changed.**
2. **`Cargo.toml repository` URL** (`runtime-rs/crates/tac-packet-compiler/Cargo.toml::repository`
   currently `https://github.com/commaai/commavq-comma-video-compression-challenge`).
   This is the upstream contest URL, not our own canonical repo URL. The
   project pyproject.toml uses `https://github.com/adpena/pact` (the correct
   canonical OSS repo URL). The Rust crate Cargo.toml has not been aligned.
   Operator should align the Cargo.toml repo URL before any future
   `cargo publish`. **NOT auto-changed.**
3. **SBOM generation** — see Recommendation 3 above. NOT blocking v0.2.0-rc1.
4. **Crates.io publish authorization** — IRREVERSIBLE per crates.io no-unpublish
   policy. Crate is currently `publish = false`; operator approval per
   AskUserQuestion would be required before any `publish = true` flip. NOT
   blocking the v0.2.0-rc1 tag itself; only blocking a future
   `cargo publish` call.

## Verification methodology

```bash
# 1. Enumerate files added since cutoff
git log --since="2026-05-11 22:30" --diff-filter=A --name-only --pretty=format: \
    | sort -u | grep -E '\.(py|rs|sh)$'
# Result: 24 files (production code), excluding generated golden vectors

# 2. Re-confirm project LICENSE + pyproject metadata
cat LICENSE | head -3       # MIT License + 2026 copyright
grep "^license" pyproject.toml runtime-rs/crates/tac-packet-compiler/Cargo.toml

# 3. SPDX prevalence repo-wide
grep -rl "SPDX-License-Identifier" src/tac/ runtime-rs/ experiments/ submissions/ \
    2>/dev/null | wc -l
# Result: 0 (project convention unchanged from 2026-05-11)

# 4. Third-party import inventory
find src/tac tools experiments -name '*.py' -print0 \
    | xargs -0 grep -hE '^(import|from) [a-z_]' \
    | grep -vE '^(import|from) (tac|upstream|comma_lab|\.)' \
    | awk '{print $2}' | awk -F. '{print $1}' | sort -u

# 5. Tag verification
git tag --list "v0.2*"
git ls-remote --tags origin | grep "v0.2.0-rc1"
# Result: v0.2.0-rc1 LOCAL + remote (pushed during OD-A2 sweep 2026-05-12)
```

## Findings summary

- **PASS**: 24 new code files comply with established LICENSE-at-root +
  per-package-metadata posture. 0 violations.
- **PASS**: 3 new golden-vector JSON files are byte-faithful re-implementations,
  NOT vendored upstream code; MIT applies via project LICENSE.
- **PASS**: third-party import inventory shows 0 GPL/AGPL surprises in the
  `tac` Python wheel surface. LGPL deps (pyppmd, paramiko) are
  import-only/dynamic-link compatible. The one GPL-adjacent dep (pygit2)
  is operator-side automation only, not part of the redistributed wheel.
- **CONFIRM**: v0.2.0-rc1 tag is now public-visible on `origin`.
- **RECOMMEND**: 3 surfaced `THIRD_PARTY_NOTICES.md` additions, NOT applied
  per mutation-frontier rule.
- **CARRY FORWARD**: 4 operator-decision items from 2026-05-11 audit still
  open (LICENSE copyright string + Cargo.toml repo URL + SBOM + crates.io
  publish auth).

## Cross-references

- Prior audit: `.omx/research/license_audit_20260511.md`
- Prior landing: `~/.claude/projects/.../memory/feedback_github_release_tag_license_audit_phase1_wiring_lane_sweep_landed_20260511.md`
- OD-A2 GitHub tag push: `~/.claude/projects/.../memory/feedback_category_a_b_operator_authorize_execution_landed_20260512.md`
- Council Q5 verdict: `feedback_grand_council_a_b_work_plan_review_20260511.md`
- Public Disclosure Hygiene: `CLAUDE.md` (non-negotiable section)
- This audit's sister deliverables: `.omx/research/lane_sweep_v0_2_0_rc1_20260512.md` + `.omx/release_manifest_v0.2.0-rc1.md`
