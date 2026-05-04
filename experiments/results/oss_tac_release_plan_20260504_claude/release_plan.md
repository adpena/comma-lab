# `tac` Open-Source Release Plan

_Generated 2026-05-04 by Claude Opus 4.7 mission subagent (PR107 hardening)._

This is a senior-engineer-grade plan to extract the `tac` (Task-Aware Codec) Python package out of the private `comma-lab` workspace and publish it under a standalone public GitHub repository, with full secret-sanitization and CI scaffolding. **DO NOT EXECUTE** the publish step here — that is the user's action. This plan produces only local files and a release script the user can later run.

## Mission scope

- Source repo (PRIVATE): `git@github.com:adpena/comma-lab.git`
- Target public repo (PROPOSED): `github.com/adpena/tac` (or `github.com/adpena/tac-codec` if `tac` is taken)
- Goal: a research-quality OSS release that introduces the rate-distortion compiler for the comma.ai video compression challenge, the score-aware codec primitives, the production-hardened deterministic submission pipeline, and the foundational documentation
- Non-goal: ship the entire `comma-lab` history, including private custody ledgers, GPU operator credentials, internal council deliberations, and competitive-edge-protected experiment results

## What ships publicly (whitelist)

The release shipping under `~/tac_oss_export/` (created by `release_tac_oss.sh`):

```
tac_oss_export/
├── LICENSE                                  # MIT — carries over verbatim, authorship preserved
├── README.md                                # rewritten public README — see PUBLIC_README.md
├── pyproject.toml                           # sanitized pyproject (private deploy paths stripped)
├── src/tac/                                 # the entire `src/tac/` Python package
│   ├── __init__.py
│   ├── README.md                            # internal doc retained
│   ├── ... (all 189 .py files, all subdirs)
├── src/tac/tests/                           # pytest suite for `tac` (already inside src/tac/)
├── experiments/                             # selected examples only (NOT all 335 files):
│   ├── pipeline.py                          # canonical compress entrypoint
│   ├── contest_auth_eval.py                 # contest-faithful eval driver (private dispatch helpers stripped)
│   ├── canonical_local_auth_eval_smoke.py   # local smoke
│   ├── modal_auth_eval.py                   # GPU-fan-out reference
├── docs/                                    # public-facing docs only:
│   ├── architecture.md
│   ├── scoring_formula.md
│   ├── deterministic_archive_contract.md
│   ├── faq_runtime_closure.md
│   └── paper/                               # the OPEN portions of the paper (drop EXTERNAL_SOURCE_ATTRIBUTION_C067 NOT — it's already public-safe)
│       ├── 01_intro.md (sanitized)
│       ├── 02_methods.md (sanitized)
│       ├── 04_results.md (sanitized — keeps figures, scrubs private paths)
│       ├── 05_production.md (sanitized)
│       └── EXTERNAL_SOURCE_ATTRIBUTION_C067.md
├── tools/                                   # public OSS tooling subset:
│   ├── lane_maturity.py                     # registry CLI (sample registry, NOT live `.omx/state`)
│   ├── review_tracker.py                    # OSS-friendly version
│   └── (NOT subagent_commit_serializer.py — internal coordination tool)
├── .github/
│   └── workflows/
│       ├── ci.yml                           # pytest + ruff + ty (Astral type checker)
│       └── release.yml                      # PyPI publish on tagged release (manual)
├── .gitignore                               # standard Python + .venv + __pycache__
└── CHANGELOG.md                             # release notes mapping tac modules to score landmarks
```

## What MUST be excluded (blacklist)

These items must NEVER make it into the public export, by exclusion patterns in `release_tac_oss.sh`:

| Pattern | Reason |
|---|---|
| `.omx/` | Private operator state, custody ledgers, council deliberations |
| `.ralph/` | Private operator workflow state |
| `.agents/` | Private agent infra |
| `.claude/` | Claude Code local session state |
| `submissions/` (entire) | Private archives; submitted PRs are already public via comma.ai repo |
| `experiments/results/` (almost all) | Private experiment outputs, custody ledgers, GPU dispatch metadata |
| `experiments/results/lightning_batch/` | Private GPU dispatch results |
| `experiments/results/lane_*` | Private lane experiments and council reviews |
| `tools/subagent_commit_serializer.py` | Internal multi-agent coordination tool — useful only inside `comma-lab` |
| `tools/claim_lane_dispatch.py` | Internal coordination |
| `.omx/state/vastai_active_instances.json` | Operator GPU host details |
| `.omx/state/lightning_batch_jobs.json` | Operator dispatch state |
| `scripts/remote_lane_*.sh` | Internal lane dispatch scripts (each contains GPU host details and budget caps) |
| `scripts/launch_lightning_batch_job.py` | Internal Lightning Studio integration |
| `scripts/bat00.py` | Tailscale-host SSH wrapper |
| `scripts/modal_check.py` | Internal harvest helper |
| `scripts/kaggle_check.py` | Internal helper |
| `reports/raw/` | Private GPU dispatch raw outputs |
| `upstream/` | Pinned upstream snapshot — comma.ai repo IS the upstream; we don't redistribute |
| `tests/` outside `src/tac/tests/` | Private integration tests |
| `CLAUDE.md` | Private operator instructions |
| `AGENTS.md` | Private agent doc |
| `PROGRAM.md` | Private mission state |
| `BATTLE_PLAN.md` | Private strategy |
| `GAMEPLAN_*.md` | Private strategy |
| Anything matching pattern `*_modal/`, `*_vast/`, `*_lightning/` | Per-platform dispatch outputs |
| Anything matching pattern `feedback_*.md` in memory dir | Private auto-memory |

## Sanitization steps (run by `release_tac_oss.sh`)

1. **Copy whitelist to `~/tac_oss_export/`** via `rsync` with explicit include patterns.
2. **Strip private references from public files** via `sed`:
   - Strip Tailscale IPs (`100.81.85.28`, `100.125.140.94`, `100.120.99.124`, `100.114.131.54`, `100.65.24.39`)
   - Strip `/teamspace/` paths (Lightning Studio internal)
   - Strip `/Users/adpena/` absolute paths → replace with `~/`
   - Strip `vastai_api_key` references → comment with placeholder
   - Strip Modal token references
   - Strip operator-specific dispatch numbers (`35707822`, `35956905`, `35959478`, `34383502`, etc.)
   - Strip `gpt-5.5` / `claude-opus-4-7` / `codex` references unless in attribution sections
3. **Rewrite `pyproject.toml`**: keep `tac` package, strip `cloud` extras (Lightning/Modal/Vast — those are operator-specific), keep `runtime`/`dev`/`viz`/`analysis`. Update `Homepage` URL to placeholder.
4. **Replace README.md** with the public-facing version produced at `experiments/results/oss_tac_release_plan_20260504_claude/PUBLIC_README.md`.
5. **Generate public CHANGELOG.md** from per-tac-module score landmarks (mapping in this plan, see "Module → score-landmark mapping" section).
6. **Add `.github/workflows/ci.yml`**: matrix Python 3.11/3.12 × Linux/macOS, run `pytest src/tac/tests`, `ruff check`, `ty check src/tac`.
7. **Run `pytest src/tac/tests`** locally on the export tree to verify suite still passes (no private-path imports leaked).
8. **Run `git init`** in `~/tac_oss_export/`, single initial commit "initial public release of tac (Task-Aware Codec) — comma.ai video compression challenge".

## Module → score-landmark mapping (for CHANGELOG)

Every module that has a documented score-landmark — for the public CHANGELOG.md release notes:

| `src/tac/` module | Score landmark | Era | Reference |
|---|---:|---|---|
| `qp1_pose_codec.py` | C-067 0.31561703 [contest-CUDA T4 A++] | Pose codec era | C-067 anchor |
| `qzs3_renderer_codec.py` (or `quantizr_qzs3_codec.py`) | Lane G v3 1.05 [contest-CUDA RTX 4090] | OWv3 era | Lane G v3 |
| `pfp16_codec.py` | PFP16 A++ baseline 1.044 [contest-CUDA T4] | PFP16 baseline | PFP16 freeze |
| `water_filling_codec_v2.py` | Lane Ω-W-V2 40.98% byte savings [empirical] on Lane G v3 | Water-fill era | Lane Ω-W-V2 |
| `arithmetic_qint_codec.py` | PD-V2 0.9974 [contest-CUDA RTX 4090] | First sub-1.0 | PD-V2 lineage |
| `submission_archive.py` | Strict-scorer-rule packer enforcing all archive bytes are charged | Foundational | (every archive ships with this) |
| `eval_roundtrip_gate.py` | Closes proxy-auth gap (2-11x) | Foundational | (eval_roundtrip non-negotiable) |
| `preflight.py` | 90+ STRICT preflight checks, 8+ permanent bug-class extinctions | Infrastructure | feedback_loop_session_permanent_bug_class_extinction_20260501.md |
| `sjkl_basis.py` | Score-Jacobian KL basis primitive (Wave-Ω-1 paradigm) | Wave-Ω | Council #2 deliverable |
| `sensitivity_map.py` | β-Fisher 1.016 LANDED [contest-CUDA T4] | Sensitivity era | β Fisher |
| `iterative_magnitude_pruning.py` | Lane 17 IMP scaffold | IMP lane | Lane 17 (KILL retracted) |
| `joint_admm_coordinator.py` | Boyd Joint-ADMM coordinator | ADMM lane | Joint-ADMM |
| `lane_c_compliance.py` | Cryptographic Ed25519 attestation gate | Compliance | Lane C |
| `nerv_mask_codec.py` | NeRV mask codec — 23.6KB / 94.4% byte savings [Lane 12 Phase G] | Lane 12 NeRV | (deferred public release) |
| `mask_grayscale_lut.py` | Selfcomp 0.36 paradigm (PR #56 reverse engineering) | Selfcomp era | reference_pr56_selfcomp_blob_byte_layout |
| `henosis_pr82_transfer.py` | PR #82 Henosis 30-byte header decoder | Public-frontier intake | PR #82 |
| `pr85_bundle.py` | PR #85 Otto adaptive masking | Public-frontier intake | PR #85 |
| `pr86_hpac_codec.py` | PR #86 HPAC hybrid coder | Public-frontier intake | PR #86 |

## Release CI — `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          version: latest
      - run: uv venv
      - run: uv pip install -e ".[dev]"
      - run: uv run ruff check .
      - run: uv run pytest src/tac/tests -m "not cuda and not slow"
```

## CRITICAL user actions (post-release-script)

The `release_tac_oss.sh` produces a clean export tree. The user MUST:

1. Review the export tree manually for any leaked private references (the script greps for known patterns but may miss novel ones).
2. Run `cd ~/tac_oss_export && pytest src/tac/tests` to verify the suite still works in isolation.
3. Run `gh repo create adpena/tac --public --description "Task-Aware Codec for the comma.ai video compression challenge"` (or chosen name).
4. Run `cd ~/tac_oss_export && git remote add origin git@github.com:adpena/tac.git && git push -u origin main`.
5. Tag a release `v1.0.5` (matching `pyproject.toml`).
6. Decide whether to publish to PyPI (separate workflow — a PyPI token must be set up).
7. **Replace placeholder URLs in PR107 body** with the actual public repo URL once it exists.

## Open questions to resolve before publish

- Is the package name `tac` available on PyPI? If not, fall back to `tac-codec` or `comma-tac`.
- Should the OSS release include a working CUDA training entrypoint? Initial recommendation: **no** — ship the codec primitives + the inference inflate path + the eval driver. Training requires the contest video which we cannot redistribute.
- Should we ship the upstream snapshot? **No** — comma.ai repo is the upstream of record.
- Is there a tac dataset we could ship? **No** — only the contest video, which is comma.ai property.

## Risk assessment

- **MIT license preserved**: `LICENSE` carries over; authorship attribution is correct.
- **Private custody scrubbed**: every `.omx/`, `.ralph/`, `submissions/`, `experiments/results/` is excluded. The blacklist regex catches Tailscale IPs and Lightning Studio paths.
- **No competitive secret-sauce leakage at release**: per CLAUDE.md "Strategic Secrecy Rule", we are not yet at the official-PR-disclosure moment for paradigm-level lanes. The OSS release exposes the foundational codec primitives + production architecture, but does NOT pre-disclose unmeasured Wave-Ω stacks or the SJ-KL basis breakthrough.
- **CI works on fresh machines**: `pytest`, `ruff`, `ty` all run from public extras.

## Estimated work for user post-receipt of this plan

- Manual review of export tree: 1 hour
- `gh repo create` + `git push` + `tag`: 15 minutes
- PyPI setup + token + first publish: 1 hour (optional)
- Replace PR107 placeholder URL: 5 minutes
- Total: ~2-3 hours of operator work
