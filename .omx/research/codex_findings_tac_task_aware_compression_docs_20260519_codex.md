# Codex findings: TAC terminology and comma-lab boundary docs

research_only: true
date_utc: 2026-05-19
agent: codex
scope: docs_and_naming_standardization

## Verdict

`tac` should canonically mean **Task-Aware Compression**.

Reason: the package now covers the full task-conditioned compression stack:
archive grammars, entropy coders, deterministic packet compilation, scorer
contracts, master-gradient and sensitivity surfaces, procedural byte derivation,
planning, and exact-eval custody. A "codec" is one narrower concrete
encoder/decoder or wire format inside that stack.

This aligns the project with the broader academic and industry vocabulary:

- video coding for machines / coding for machines;
- feature coding for machines when compressed representations are machine
  features or tensors;
- task-aware or task-oriented compression for objectives conditioned on
  downstream task loss;
- neural or learned compression as the implementation family, not necessarily
  task-aware by itself.

Source anchors: MPEG-AI Part 2 Video coding for machines, MPEG-AI Part 4
Feature coding for machines, MPEG WG4 video-coding scope, and CVPR 2023
AccelIR task-aware image-compression terminology.

## Landing

Docs and public metadata now make the hierarchy explicit:

- `README.md` states `tac = Task-Aware Compression` and points to package
  boundary docs.
- `src/tac/README.md` has terminology, references, active workflows, package
  scope, and the `tac` / `codec` / `comma_lab` hierarchy.
- `src/comma_lab/README.md` documents the lab operations layer, active
  workflows, and explicit non-ownership of algorithmic compression logic.
- Public project metadata and docs now point at the active `adpena/comma-lab`
  repository rather than stale `adpena/pact` URLs.
- Generic "task-aware codec" docstrings in public `tac` modules were corrected
  to "task-aware compression"; "codec" remains reserved for concrete codec
  components.

## Approval-gated leftover

`THIRD_PARTY_NOTICES.md` still contains one "Task-Aware Codec" phrase. It is
explicitly protected by `PROGRAM.md`'s mutation frontier, so Codex did not edit
it without a targeted operator/legal-doc approval.

## Verification

- `rg "Task-Aware Codec|task-aware codec|Task Aware Codec|task aware codec" README.md pyproject.toml src docs AGENTS.md CLAUDE.md -g '!**/__pycache__/**'` returns no normal tracked-source matches outside the protected `THIRD_PARTY_NOTICES.md` notice.
- `ruff check` passed on the touched docs/code surfaces.
- `py_compile` passed on touched Python modules.
- Focused pytest passed: `test_tto.py`, `test_quantization.py`,
  `test_losses.py`, `test_training.py` (78 passed).

## Authority

No score claim, dispatch claim, or promotion authority is created by this docs
landing. It is `research_only=true` and only hardens naming, package metadata,
and boundary interpretation for future agents and public readers.
