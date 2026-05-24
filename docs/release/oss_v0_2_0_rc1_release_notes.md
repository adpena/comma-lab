<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2026 Alejandro (Alex) Peña -->

# OSS v0.2.0-rc1 — GitHub Release notes (DRAFT)

> **DRAFT STATUS — public push HALTED 2026-05-14 pending Hygiene-Sweep-2.**
> The `oss-v0.2.0-rc1` annotated tag is LOCAL-ONLY at `82ecc2a0c`. Pushing
> to the public remote (`git@github.com:adpena/comma-lab.git`) would
> publish 1,285 tracked files containing local absolute paths
> (`/Users/adpena/...`) plus 4 files leaking private Tailscale IPs.
> See "Hygiene Sweep 2 — pre-publish blockers" below for the remediation
> plan. This notes file is committed as the artifact that lets the
> operator confirm intent **before** the public push.

## TL;DR

`tac` (the Task-Aware Compression library) ships its first
public-ready OSS release candidate aligned with the
[comma.ai / openpilot](https://github.com/commaai/openpilot) style:

- **MIT License**, single-author copyright (Alejandro (Alex) Peña).
- **THIRD\_PARTY\_NOTICES.md** in openpilot style (hard runtime / opt-in /
  upstream-referenced).
- **CONTRIBUTING.md** in openpilot style.
- **SPDX-License-Identifier: MIT** headers across 3,204 `.py` files.
- `pyproject.toml` aligned to PEP 621 (version `0.2.0rc1`, single author,
  keywords, maintainer field).
- `CHANGELOG.md` `[0.2.0-rc1]` entry.

This is a *provenance-only* release. Archive bytes, the contest scorer
contract, the pinned upstream snapshot under `upstream/`, and the
`inflate.sh` runtime contract are unchanged. The release candidate makes
the project publishable; it does not change scores.

The runtime-rs tag `v0.2.0-rc1` (a different tag, on commit `73ff0dba`,
already public) is the Rust-side `tac-packet-compiler` 19 committed primitive
golden-vector parity milestone (2026-05-11), not a complete native crate-parity
claim. The `oss-v0.2.0-rc1` tag is the broader OSS posture alignment milestone
(2026-05-14) and is the subject of these notes.

## Posture alignment

| Surface                       | comma.ai / openpilot reference                                                                                | This release                                                                                       |
|-------------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| License                       | MIT, `Copyright (c) 2024, comma.ai, Inc.`                                                                      | MIT, `Copyright (c) 2026 Alejandro (Alex) Peña`                                                    |
| SPDX headers                  | Per-source-file `# SPDX-License-Identifier: MIT`                                                               | 3,204 `.py` files carry the canonical header (idempotent injection via `tools/oss_inject_spdx_headers.py`) |
| Third-party notices           | `THIRD_PARTY_NOTICES.md` enumerating runtime deps + upstream references                                        | `THIRD_PARTY_NOTICES.md` with hard-runtime / opt-in / upstream-referenced sections                 |
| Contribution guide            | `CONTRIBUTING.md` with license-agreement opening                                                               | `CONTRIBUTING.md` mirroring the openpilot pattern                                                  |
| Pyproject metadata            | PEP 621, single source-of-truth                                                                                | `pyproject.toml` with `version = "0.2.0rc1"`, single author, keywords                              |
| Default install path          | Permissive-only (no LGPL/GPL by default)                                                                       | `pyppmd` (LGPL) → `constriction` (MIT) migration; `[pr86_replay]` opt-in kept for legacy replay    |

## What's inside the library at this rc1

These highlights describe the *capability surface* — the engineering
substrate that the OSS posture publishes. Score claims are out of scope
for this release (see the [empirical:provenance-only] tag below).

### Architecture

- **`tac/`** — reusable Task-Aware Compression runtime-contract library. Substrates (the
  per-architecture trainers + archive grammar + score-aware loss) live
  under `tac/substrates/`. There are 31 substrate trainers in
  `experiments/train_substrate_*.py` at rc1, each declaring a
  `TIER_<N>_OPERATOR_REQUIRED_FLAGS` manifest that propagates to the
  Modal/Vast.ai/Lightning dispatch wrappers via canonical helpers.
- **`tac/packet_compiler/`** — deterministic submission-packet compiler
  (Python oracle). Companion Rust port lives in `runtime-rs/` with 19 committed
  primitive golden-vector parity fixtures vs the Python oracle; selector,
  search, and decode meta surfaces remain Python-oracle or fail-closed Rust
  scaffolds (see the runtime-rs `v0.2.0-rc1` tag for that milestone).
- **`tac/cathedral_autopilot/`** — the candidate ranker that consumes
  cost-band posterior anchors + scorer-conditional MDL evidence + lane
  maturity registry rows. Updated 2026-05-14 with the Z1 empirical
  revision (within-class-saturation penalty + class-shift literature
  reward) — the autopilot now routes operator dispatch budget toward
  class-shift substrates (Wyner-Ziv / cooperative-receiver /
  predictive-coding / MDL-IBPS) once an architecture class exceeds
  density 0.90.
- **`tac/continual_learning/`** — posterior store for empirical anchors
  (`[contest-CUDA]` / `[contest-CPU]` / `[macOS-CPU advisory]`) with
  fcntl-locked transactional writes (Catalog #128 + #131) and typed
  custody verdicts (Catalog #127 + #130).
- **`experiments/pipeline.py`** — the canonical training entry point.
  All training surfaces flow through named profiles in
  `tac/profiles.py`; ad-hoc shell scripts are forbidden by `set -e` +
  preflight gates.

### Tests and STRICT preflight coverage

- 1,400 test files under `src/tac/tests/` + `tests/`.
- 140+ STRICT preflight gates enumerated in CLAUDE.md "Meta-bug class
  catalog". Each gate is paired with dedicated tests (typically 10-30
  per gate) covering positive / negative / waiver semantics / live-repo
  regression.
- A `Meta-bug class catalog (strict-mode preflight)` table inside
  `CLAUDE.md` (currently 219+ rows) documents every gate's bug class,
  fix-and-self-protect lineage, and live-violation count.
- Continuous self-discipline gates: Catalog #117 (subagent commits
  serializer use), #157 (commit-swap pre-pre-lock content hash), #176
  (catalog row matches preflight strict value), #185 (catalog Live
  count: 0 claims verified against gate function output), #206
  (subagent crash-resume checkpoint discipline).

### Cost-band posterior + autopilot

- `.omx/state/cost_band_posterior.jsonl` (fcntl-locked append-only) is
  the canonical store for per-platform-per-GPU dispatch cost anchors
  with explicit `outcome` field (Catalog #175 + #177).
- The autopilot ranker emits a re-ranked candidate queue with the Z1
  empirical revision applied (within-class-saturation penalty +
  class-shift reward). Re-ranked top-3 at this rc1: time_traveler L5
  / DARTS-SuperNet / C6 MDL-IBPS — all class-shift candidates ahead of
  within-A1-class sidecar bolt-ons.

### Six canonical operational directives

The `feedback_*.md` memory ledger encodes six standing directives that
shape every subagent prompt:

1. **NO-SIGNAL-LOSS** — every empirical result is harvested into the
   posterior store; no negative result is "thrown away."
2. **Recursive R1-R4 review** — every code landing receives 3 clean
   adversarial-review passes before merge.
3. **Journal-grade documentation** — every landing memo carries 11
   structural sections (Hypothesis / Math / Citations / Provenance /
   Empirical tag / Reproducibility / Sister-lane / 6-hook wire-in /
   Stop-continue thresholds / Reactivation criteria / Operator-routable
   decisions).
4. **Harness-rigor** — `experiments/pipeline.py` + canonical profiles +
   preflight gates form the deterministic experiment harness.
5. **7-factor lane maturity** — every lane is tracked across 7 gates
   (impl_complete / real_archive_empirical / contest_cuda /
   strict_preflight / three_clean_review / memory_entry /
   deploy_runbook).
6. **IBPS1 grammar (Information Bottleneck Predictive-Sufficient)** —
   the canonical substrate composition grammar used by the C6
   MDL-IBPS substrate and the cathedral autopilot ranker.

### Four critical pattern feedbacks

The memory ledger surfaces four recurring failure patterns that the
preflight gates structurally extinct:

- **Editor-vs-editor race** — pre-pre-lock content-hash check via
  `--expected-content-sha256` (Catalog #157 + #216).
- **Modal 3600s timeout** — smoke-before-full pattern + min-smoke-gpu
  recipe field (Catalog #167 + #215).
- **Prompt-premise verification** — subagent pre-flight read of
  CLAUDE.md + AGENTS.md + lane registry + sibling subagents.
- **Parallel-wave serialization** — canonical commit serializer +
  fcntl-locked state writes (Catalog #117 + #128 + #131).

## Compatibility

- Python 3.11+. Recommended toolchain: `uv` for package management.
- CUDA optional for inference / inflate; required only for
  `[contest-CUDA]` authoritative score evaluation. Modal A100 + Vast.ai
  4090 + Lightning T4 are the supported GPU substrates.
- Apple Silicon CPU is supported as a `[macOS-CPU advisory]` proxy
  (NOT promotable to `[contest-CPU]`; see Catalog #192).
- Linux x86_64 is the canonical `[contest-CPU]` axis (matches the
  contest GitHub Actions runner).

## Empirical tag

[empirical:provenance-only; no score claim]. OSS posture changes are
non-score-affecting. The pinned upstream snapshot under `upstream/` is
untouched (read-only per CLAUDE.md "Non-Negotiable Upstream Rule"). The
contest scorer contract is untouched. Archive bytes, inflate runtime,
and authoritative-eval behavior are unchanged.

## Hygiene Sweep 2 — pre-publish blockers

The release-prep audit
(`reports/release_v0_2_0_rc1_audit_20260514.md`) marked
"Public disclosure hygiene" as `NEEDS_SANITIZATION` at the time the
LOCAL `oss-v0.2.0-rc1` tag was created. Hygiene Sweep 1 sanitized the
`docs/superpowers/**` slice (Catalog #208, `docs/` violations: 0).
Hygiene Sweep 2 — covering `.omx/research/`, `src/tac/`, `CLAUDE.md`,
`tools/`, `experiments/`, `reports/raw/` — is **NOT yet complete**.

Live findings at the tagged commit (`82ecc2a0c`) on 2026-05-14:

| Check                                               | Files leaking | Lines |
|-----------------------------------------------------|---------------|-------|
| Local absolute paths `/Users/adpena/...`            | 1,285         | ~308+ |
| Tailscale IP `100.x.x.x` (private fleet)            | 4             | ~14   |
| Credentials (api key / password / bearer)           | 0             | 0     |
| Provider tokens (Vast.ai / Modal / Lightning)       | 0             | 0     |
| `Catalog #208` `check_docs_no_local_absolute_paths` | 0 (docs/ only)| 0     |

Files leaking Tailscale IPs (top of list):
- `CLAUDE.md` — operator memory; **not OSS-public**.
- `docs/tailscale-fleet-setup.md` — internal fleet guide.
- `scripts/bat00.py` + `scripts/bat00_wsl_setup.ps1` — operator helpers.
- `.omx/research/domain_exploitation_catalog_20260509.md`.

Files leaking `/Users/adpena/...` paths by top-level directory:

| Directory             | Count |
|-----------------------|-------|
| `.omx/research/`      | 67    |
| `src/tac/`            | 19    |
| `experiments/`        | ~11   |
| `tools/`              | ~3    |
| `reports/`            | ~5    |
| `tests/`              | 1     |
| `CLAUDE.md`           | 1     |
| Others                | balance |

Per CLAUDE.md "Public Disclosure Hygiene — non-negotiable":
> Keep credentials, private infrastructure URLs, **local absolute
> paths**, raw provider logs, unpublished operator state, and account
> metadata out of GitHub/docs/site/public supplement surfaces.

Pushing the `oss-v0.2.0-rc1` tag would make 142 commits reachable via
the tag ref on the public remote (`origin/main` is currently at
`99ca7859`, 142 commits behind the tag). Those 142 commits contain
every leak above.

### Remediation plan (operator-routable; choose one)

**Option A — sanitize-broader-scope-then-retag (recommended).**
Run a Hygiene-Sweep-2 lane that sanitizes the 1,285 files leaking
`/Users/adpena/...` and the 4 files leaking Tailscale IPs. Replace
absolute paths with `<repo-root>` placeholders or repo-relative paths
(per the Catalog #208 pattern). Once `git grep -E '/Users/[a-zA-Z]+/'`
returns 0 across the relevant scope, delete the LOCAL `oss-v0.2.0-rc1`
tag, re-create it on the new sanitized HEAD, and re-run this push
procedure. **Cost: $0 GPU; ~2-4h engineering; produces a clean public
repo.**

**Option B — public-mirror-with-sanitization (CI-rebuilt).**
Establish a separate `oss-public/` worktree (or a CI workflow) that
filters every commit pushed to the public remote through a
sanitization pass (`sed -E 's|/Users/adpena/Projects/pact/|<repo-root>/|g'`,
strip CLAUDE.md sections matching the fleet table, etc.). The public
remote becomes a derived-output of the private `main`. **Cost: 1-2 days
to set up; ongoing maintenance; preserves private operator state under
`adpena@` while still publishing the OSS posture.**

**Option C — narrow-publication (release-only-artifacts).**
Push ONLY the release artifacts that are already clean — `LICENSE`,
`THIRD_PARTY_NOTICES.md`, `CONTRIBUTING.md`, `pyproject.toml`,
`CHANGELOG.md`, `README.md`, `docs/release/` — to a fresh
`oss-publication/` branch on the public remote. The public remote does
NOT mirror `main`. The OSS posture is published; the operator state
stays private. **Cost: 30-60 min; loses the source-code-publish
benefit but unblocks the OSS license posture.**

**Option D — accept-and-push (NOT recommended).**
Per CLAUDE.md the operator may override the hygiene check, but
publishing local paths + Tailscale IPs makes the private fleet
topology + operator filesystem layout permanent on a public GitHub
repository. Reversal is not clean (force-pushing the tag still leaves
the tag-reachable commits in any fork or local clone made between
publish + reversal). **Not recommended.**

The OSS landing memo (`feedback_oss_release_v0_2_0_rc1_landed_20260514.md`)
operator-routable decision #1 explicitly flagged this:

> Push `oss-v0.2.0-rc1` to public remote? The tag is currently LOCAL-only.
> The CLAUDE.md "Public Disclosure Hygiene" rule says "Public release is
> intentional, not automatic."

This release-notes draft is the artifact that lets the operator pick
A / B / C / D explicitly. The default verdict per the existing CLAUDE.md
non-negotiable is HALT-pending-Hygiene-Sweep-2.

## Verifying release artifacts (clean side)

These artifacts are already clean and ready for public publication
under any of A / B / C:

```bash
# Tag points at the canonical OSS release commit
git show oss-v0.2.0-rc1 --stat --no-patch

# License is MIT, single-author copyright
head -3 LICENSE
# MIT License
# Copyright (c) 2026 Alejandro (Alex) Peña

# Third-party notices follow the openpilot pattern
head -20 THIRD_PARTY_NOTICES.md

# CONTRIBUTING.md mirrors the openpilot pattern
head -10 CONTRIBUTING.md

# pyproject.toml is PEP 621, version 0.2.0rc1
grep -E '^(version|name|authors)' pyproject.toml | head -5

# CHANGELOG has the rc1 entry
sed -n '/\[0.2.0-rc1\]/,/^## /p' CHANGELOG.md | head -40

# SPDX headers across in-scope .py files
head -3 src/tac/training.py
head -3 experiments/pipeline.py

# Catalog #208 (docs/ scope) returns 0 violations
.venv/bin/python -c "from tac.preflight import check_docs_no_local_absolute_paths; \
  print(len(check_docs_no_local_absolute_paths(strict=False, verbose=False)))"
# expect: 0
```

## Acknowledgments

- The comma.ai openpilot project for the canonical OSS posture
  template ([LICENSE](https://github.com/commaai/openpilot/blob/master/LICENSE),
  [THIRD_PARTY_NOTICES.md](https://github.com/commaai/openpilot/blob/master/third_party/README.md)).
- Yassine Yousfi + Jessica Fridrich (Binghamton DDE Lab) for the
  contest scorer design (SegNet + PoseNet from EfficientNet-B2 +
  FastViT-T12). The challenge is inverse steganalysis.
- The pinned `upstream/` snapshot at the contest reference SHA.
- All authors of the runtime dependencies enumerated in
  `THIRD_PARTY_NOTICES.md`.

## Provenance

- Release-candidate commit: `82ecc2a0ce89d1de3bcb7e6e9e8cfaf7e87302b9`
- LOCAL annotated tag: `oss-v0.2.0-rc1`
- Tag object: `ba9906069d0b19d65e0fb46b7828b62f9fc0c328`
- Public remote: `git@github.com:adpena/comma-lab.git` — TAG NOT YET PUSHED.
- OSS landing memo: `feedback_oss_release_v0_2_0_rc1_landed_20260514.md`
- Hygiene-sweep-2 blocker memo (this push attempt):
  `feedback_oss_public_push_v0_2_0_rc1_landed_20260514.md`
- Lane: `lane_oss_public_push_v0_2_0_rc1_20260514` (Phase 4.0, L1
  SKETCH — impl_complete + memory_entry; blocked on Hygiene Sweep 2).

[provenance-only; no score claim]
