# TAC Naming And Documentation Hardening - Codex Findings 2026-05-19T041951Z

## Verdict

Canonical expansion: `tac` means **Task-Aware Compression**.

Rationale: the package scope is broader than a single codec. It owns codec
primitives, but also scorer geometry, exact-eval custody, deterministic packet
compilation, sensitivity maps, master-gradient consumers, and optimization
planners. `codec` should remain reserved for a concrete encoder/decoder, entropy
coder, archive grammar, or wire format inside the broader Task-Aware Compression
stack.

## Terminology Authority

- Industry and standards-language anchor: MPEG uses **Video Coding for Machines
  (VCM)** for video bitstreams and descriptors optimized for machine-task
  performance after decoding, and **Feature Coding for Machines (FCM)** for
  compressed intermediate feature tensors in machine vision networks.
- Academic-language anchor: CVPR 2023 AccelIR uses **Task-Aware Image
  Compression** for compression optimized against an end-to-end neural
  restoration task.
- Repo-language consequence: `Task-Aware Compression` is the right public
  package expansion; `video coding for machines`, `feature coding for machines`,
  `task-oriented compression`, and `neural/learned compression` are adjacent
  search terms and literature bridges.

Sources:

- https://www.mpeg.org/standards/MPEG-AI/2/
- https://www.mpeg.org/standards/MPEG-AI/4/
- https://www.mpeg.org/structure/video-coding/
- https://openaccess.thecvf.com/content/CVPR2023/html/Ye_AccelIR_Task-Aware_Image_Compression_for_Accelerating_Neural_Restoration_CVPR_2023_paper.html

## Local Actions

- Confirmed `pyproject.toml`, `README.md`, `src/tac/README.md`, and
  `src/comma_lab/README.md` already use the canonical expansion and package
  boundary.
- Fixed stale root README pipeline examples so `experiments/pipeline.py eval`
  includes the required `--checkpoint`, and profile compression examples include
  required `--video` and `--checkpoint`.
- Updated `THIRD_PARTY_NOTICES.md`, `SYSTEM_MAP.md`, and `CONTRIBUTING.md` so
  they no longer present `tac` as "Task-Aware Codec" or point contributors at
  the historical `pact` remote.
- Added a root README identity table: `comma-lab` = public repo/lab workspace,
  `tac` = Task-Aware Compression package, `comma_lab` = operations package, and
  `pact` = historical/internal checkout alias.
- Probed a stronger Python-docstring/test version, but did not land it because
  the review gate correctly treats broad legacy Python files as policy-scoped
  code entities. That follow-up should be a dedicated reviewed code-hygiene
  landing, not bundled into this docs pass.

## Verification

- `PYTHONPATH=src:upstream .venv/bin/python experiments/pipeline.py compress --help` and `eval --help` were consulted before editing README commands.
- `.venv/bin/python tools/audit_public_publish_links.py README.md CONTRIBUTING.md THIRD_PARTY_NOTICES.md SYSTEM_MAP.md --repo-root . --strict --format text` passed.
- `.venv/bin/python tools/scan_best_anchor_per_axis.py --repo-root . --format json` returned no frontier citation drift.
- `rg "Task-Aware Codec|canonical reusable codec library|github.com/adpena/pact"`
  no longer finds live documentation matches outside the regression test
  sentinel.
- `git diff --check` passed on the docs/memo patch.

## Residual Risk

Legacy Python docstrings still contain "task-aware codec" in a few early
post-filter modules. Those are lower-priority wording cleanup candidates; they
should not be touched without a dedicated review-gate plan because broad legacy
Python file edits pull many unrelated entities into the policy surface.

`src/tac/__init__.py` still reports `__version__ = "1.0.5"` while
`pyproject.toml` reports `0.2.0rc1`. The existing package-hygiene test catches
this, but fixing it is a Python-code commit and should be separated from this
docs-only naming pass.
