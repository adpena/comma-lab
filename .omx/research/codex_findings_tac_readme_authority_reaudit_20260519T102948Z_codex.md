# Codex TAC / comma-lab README authority re-audit 2026-05-19

## Verdict

`tac` should remain canonically expanded as **Task-Aware Compression**, not
Task-Aware Codec.

The reason is structural, not cosmetic: this repository's `tac` package now
owns more than codec implementation. It owns scorer contracts, archive grammar,
packet compilation, procedural byte derivation, master-gradient and
sensitivity-map consumers, exact-eval custody helpers, and planning primitives.
A codec is one concrete encoder/decoder or wire format inside that broader
compression stack.

## External terminology authority

Current field vocabulary supports the broader name:

- Standards / industry: MPEG uses **Video coding for machines** and
  **Feature coding for machines** for compression optimized for machine
  analysis or feature/tensor consumption.
- Research literature: "task-aware compression" and "task-oriented
  compression" are used for compression objectives conditioned on downstream
  task loss, including image/video restoration and distributed source-coding
  settings.
- Adjacent theory: semantic or goal-oriented communication is useful only when
  the design explicitly models receiver utility or a channel protocol.

Canonical external anchors already cited in docs:

- MPEG-AI Part 2: https://www.mpeg.org/standards/MPEG-AI/2/
- MPEG-AI Part 4: https://www.mpeg.org/standards/MPEG-AI/4/
- MPEG WG 4: https://www.mpeg.org/structure/video-coding/
- CVPR 2023 AccelIR: https://openaccess.thecvf.com/content/CVPR2023/html/Ye_AccelIR_Task-Aware_Image_Compression_for_Accelerating_Neural_Restoration_CVPR_2023_paper.html
- CVPR 2025 NVIDIA task-aware video rate control: https://research.nvidia.com/labs/par/publication/realtime_rate_control_video_compression.html
- NeurIPS 2023 Task-aware Distributed Source Coding: https://proceedings.neurips.cc/paper_files/paper/2023/file/016c63403370d81c24c1ca0123de6cfa-Paper-Conference.pdf

## Local authority surfaces checked

- `README.md` defines `comma-lab`, `tac`, `comma_lab`, and the internal `pact`
  alias, and points readers to the package boundary docs.
- `src/tac/README.md` carries the strongest package-level explanation:
  Task-Aware Compression is the full package/research stack; codec is a
  narrower implementation artifact.
- `src/comma_lab/README.md` clearly keeps lab state, reports, public intake,
  and operations separate from algorithmic compression implementation.
- `docs/terminology_and_boundaries.md` is the canonical naming and contest
  compliance boundary document, including the explicit "never expand TAC as
  Task-Aware Codec" rule.
- `pyproject.toml` uses the Task-Aware Compression description and field-level
  keywords: task-aware-compression, task-oriented-compression,
  coding-for-machines, video-coding-for-machines, and
  feature-coding-for-machines.
- `HANDOFF.md`, `PROGRAM.md`, `SYSTEM_MAP.md`, `CONTRIBUTING.md`, and
  `docs/README.md` all route new readers back to the same authority chain.

## Subagent audit integration

Xhigh read-only audit `019e3fc3-ae63-7172-be56-f829eaba6473` independently
found the same result: no live normal-doc/code definition asserts
`tac = Task-Aware Codec`; remaining occurrences are intentional negative
examples or guard fixtures.

The audit also flagged three stale public-link candidates. This pass patched
the two public generated/readme surfaces and the still-usable bat00 bootstrap
script:

- `tools/build_comma_video_substrate_eval_600pairs_dataset.py`: generated HF
  dataset card now points at `github.com/adpena/comma-lab`, not the historical
  `adpena/pact` repo.
- `docs/comma_pr_archive_dataset_card.md`: dataset card now links `tac` to
  `comma-lab/src/tac` instead of assuming a separate `adpena/tac` public repo.
- `scripts/bat00_runner.sh`: clone URL now uses `adpena/comma-lab.git` while
  preserving the historical local directory alias.

## Verification

Commands run:

```bash
.venv/bin/python tools/check_tac_terminology.py --strict
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_tac_terminology_guard.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check tools/check_tac_terminology.py src/tac/tests/test_tac_terminology_guard.py
```

Results:

- TAC terminology check passed.
- `src/tac/tests/test_tac_terminology_guard.py`: 9 passed.
- Ruff touched-file check passed.

## Residual risk

No public-facing stale "Task-Aware Codec" definition was found outside
intentional negative examples and guard tests. Historical/internal `.omx`
research ledgers may still contain old phrasing; do not rewrite those unless a
new public artifact consumes them. Preserve them as provenance and route active
public wording through the canonical docs above.

## 6-hook wire-in declaration

- Sensitivity-map contribution: N/A; documentation authority only.
- Pareto constraint: N/A; no score-affecting candidate.
- Bit-allocator hook: N/A.
- Cathedral autopilot dispatch hook: N/A; no candidate rows emitted.
- Continual-learning posterior: N/A; no empirical score anchor.
- Probe-disambiguator: Active via `tools/check_tac_terminology.py --strict`,
  which is wired into `tools/all_lanes_preflight.py` as Gate #32.
