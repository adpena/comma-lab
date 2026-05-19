# comma-lab

Task-Aware Compression research and artifact tooling for the
[comma.ai video compression challenge](https://github.com/commaai/comma_video_compression_challenge).
This repository is maintained as a community and historical-record workspace:
public-archive intake, exact replay custody, writeup drafts, and OSS tooling.
It is not a live leaderboard page and it does not make an arXiv or preprint
commitment.

The local checkout may still be named `pact` because that was the original
internal workspace alias. Public docs, package metadata, and release surfaces
should use `comma-lab` for the repository and `tac` for the reusable
Task-Aware Compression library.

In this repository, `tac` means **Task-Aware Compression**: compression
optimized for a downstream task/scorer. A **codec** is a narrower concrete
encoder/decoder or wire format inside that stack. The `comma_lab` package is the
lab and operations layer around `tac`, not a second compression engine.
`TAC` is a repository/package acronym, not a standards-body initialism; public
docs should expand it on first use and map the work to the field terms
task-aware compression, task-oriented compression, coding for machines, video
coding for machines, and feature coding for machines.
The package-level boundary docs are [src/tac/README.md](src/tac/README.md) and
[src/comma_lab/README.md](src/comma_lab/README.md). The canonical terminology
and contest-compliance boundary is
[docs/terminology_and_boundaries.md](docs/terminology_and_boundaries.md), with
the upstream rule and public-PR precedent ladder in
[docs/contest_compliance_authority.md](docs/contest_compliance_authority.md).

| Name | Canonical role |
|---|---|
| `comma-lab` | Public repository and lab workspace for the challenge research system |
| `tac` | Python package: Task-Aware Compression library and algorithmic engine |
| `comma_lab` | Python package: lab operations, custody, state projection, and reporting |
| `pact` | Internal workspace alias retained in historical docs and local paths |

Score-bearing claims must be read through the repository evidence grades. The
ranked public rows live in `docs/paper/04_results.md`; roadmap and planning
rows are not score claims until exact CUDA auth eval lands on exact archive
bytes.

Scoring formula: `S = 100 * seg_distortion + sqrt(10 * pose_distortion) + 25 * rate`

## Evidence Grades

| Grade | Public/writeup use | Minimum requirement |
|---|---|---|
| `A++` / `A` | Ranked score row | Exact archive bytes and SHA-256, CUDA auth-eval JSON, component recomputation, runtime custody, full sample count |
| `A-negative` | Scoped negative result | Same custody standard as a score row, but used only for the measured implementation/config |
| `empirical` | Roadmap or engineering signal | Byte, smoke, loss, round-trip, or component evidence without full score custody |
| `derivation` / `prediction` | Roadmap only | Formula or model-based hypothesis awaiting archive evidence |
| `external` | Community/historical context | Public PR text, leaderboard metadata, or outside papers before local exact replay |
| `invalid` | Compliance lesson | Proxy, CPU/MPS, stale, sidecar, exploit, malformed, or otherwise non-ranking evidence |

## Current Workflows

This is an active research and engineering repo. The high-traffic surfaces are:

- public-frontier intake, archive byte anatomy, and exact replay custody;
- Task-Aware Compression primitives in `src/tac/`: packet compilers, entropy
  coders, scorer-aware losses, renderers, quantizers, sensitivity maps,
  master-gradient consumers, and deterministic/procedural byte derivation;
- lab-control tooling in `src/comma_lab/`: state projection, strict preflight
  adapters, public-intake hygiene, release hygiene, and operator reports;
- exact `[contest-CPU]` and `[contest-CUDA]` auth-eval separation;
- procedural generation from archive-contained seeds or weights, with
  compliance mode recorded before score-bearing use;
- release/paper/OSS hygiene for the reusable `tac` package and the surrounding
  `comma-lab` lab workspace.

## Authority And Gates

Score authority is byte-closed. A row is not ranked because a memo, local smoke
test, or proxy run looks good; it becomes ranked only when the exact archive and
runtime pair pass the required custody checks.

Useful gates:

```bash
.venv/bin/python tools/check_tac_terminology.py --strict
.venv/bin/python tools/all_lanes_preflight.py
.venv/bin/python scripts/pre_submission_compliance_check.py --contest-final --strict ...
```

Compliance authority for procedural generation, scorer-aware inflate designs,
and deterministic packet compilation is summarized in
[docs/contest_compliance_authority.md](docs/contest_compliance_authority.md).
The short rule: decoder code may be clever, but score-bearing information must
be charged through `archive.zip` unless a documented tiny-runtime contract
proves it is decoder logic rather than payload relocation.

## CUDA vs CPU auth eval split (2026-05-08)

The contest scorer at `upstream/evaluate.py` produces two distinct authoritative
score axes for the same archive bytes — `--device cuda` and `--device cpu` —
and the public leaderboard ranks by the **CPU** score, not the CUDA score.
Across the medal-band HNeRV cluster (PR100/101/102/103/105) we measured a
remarkably tight `R_pose = pose_cuda / pose_cpu = 5.04 ± 0.10` and
`R_seg = seg_cuda / seg_cpu = 1.17 ± 0.01`, producing a near-constant
score-axis gap of `Δscore = 0.0330 ± 0.0004`. PR #102's third-prize 0.195 was
the CPU score; the CUDA bot comment for the same archive bytes was 0.228.

Operational consequence: every shippable archive now gets dual-eval —
authoritative `[contest-CUDA]` and `[contest-CPU]` axes on Linux x86_64
hardware that is 1:1 contest-compliant with the GitHub Actions CI runner.
Apple Silicon CPU eval is `[macOS-CPU advisory only]`, never `[contest-CPU]`.

Full write-up: [`docs/findings/cuda_cpu_auth_eval_split_20260508.md`](docs/findings/cuda_cpu_auth_eval_split_20260508.md).
Methodology long-form: [`docs/writeup/cuda_cpu_drift_methodology.md`](docs/writeup/cuda_cpu_drift_methodology.md).

## Package Map

| Surface | Role |
|---|---|
| `src/tac/` | Reusable Task-Aware Compression library: archive grammars, packet compilers, scorer contracts, codecs, substrates, sensitivity maps, and optimization primitives. |
| `src/comma_lab/` | Lab operations package: state projection, strict preflight adapters, public-frontier hygiene, scheduler/reporting support, release hygiene. |
| `tools/`, `scripts/`, `experiments/` | Thin operator entry points. Durable logic should delegate to `tac` or `comma_lab`. |
| `docs/` | Public methodology, runbooks, compliance readings, writeups, and release notes. |
| `.omx/research/` | Dated research ledgers, negative results, directives, and adversarial reviews. |
| `reverse_engineering/` | Clean public-submission deconstruction runbooks and manifests. |
| `submissions/` | Candidate and historical contest runtime packets. |
| `upstream/` | Pinned upstream challenge snapshot. Treat scorer files as read-only. |

### Related: `adpena/tac` standalone OSS package

The reusable codec, predictor, search, and runtime-contract primitives developed
in this research environment are open-sourced as standalone Python package
**[`adpena/tac`](https://github.com/adpena/tac)** (MIT licensed). This
`comma-lab` repo contains the full research environment, experimental
scaffolding, council deliberations, dispatch ledgers, and state-of-development
artifacts; `adpena/tac` is the curated production extract suitable for OSS
adoption and integration by external users (comma.ai, openpilot, downstream
research). The library surface is import-compatible across both repos.

## Quick start

```bash
# Install
git clone https://github.com/adpena/comma-lab.git && cd comma-lab
uv venv && uv pip install -e ".[dev]"

# Inspect the canonical package CLI
.venv/bin/python -m tac.cli --help
.venv/bin/python -m tac.cli lossless profiles

# Run the public terminology / docs boundary guard
.venv/bin/python tools/check_tac_terminology.py --strict
```

The research pipeline remains available for historical lossy renderer
experiments:

```bash
PYTHONPATH=src:upstream .venv/bin/python experiments/pipeline.py compress --help
PYTHONPATH=src:upstream .venv/bin/python experiments/pipeline.py eval --help
```

## Historical Training Profiles

Early renderer/post-filter profiles are retained for reproducibility and
historical comparison. They are not the whole current system; newer lanes may
use HNeRV-family replay, packet compilation, procedural codebooks, Wyner-Ziv
side information, entropy-coder repacks, or scorer-aware deterministic
transducers.
Active readers should start with `reports/latest.md`, `SYSTEM_MAP.md`, and the
current `.omx/research/*_directive_*` files before treating a profile as live
work.

Three baseline profiles encode different training philosophies. All share the
same historical architecture for fair comparison.

| Profile | Strategy | Key idea |
|---------|----------|----------|
| **WILDE** | Empirical 5-phase schedule | Freeze/unfreeze phases with hard-mined error boosting (9x/49x). Quantizr-adapted anchor training. |
| **SHIRAZ** | Principled adaptive training | PCGrad gradient surgery + focal STE loss. No freeze/unfreeze. Score-contribution-proportional weighting. |
| **GREEN** | WILDE + radial zoom warp | Same as WILDE but MotionPredictor outputs only gate+residual (4ch). Flow from RadialZoomWarp. 14K fewer params. |

```bash
# Train with a named profile
PYTHONPATH=src:upstream python experiments/pipeline.py compress \
    --profile wilde \
    --video upstream/videos/0.mkv \
    --checkpoint path/to/checkpoint.pt \
    --device cuda --output-dir results/wilde
```

Profiles are defined in `src/tac/profiles.py` with full provenance for every hyperparameter choice.

## Project structure

```
src/tac/                    Task-Aware Compression library and reusable algorithms
src/comma_lab/              Lab operations, state projection, preflight adapters
experiments/                Training scripts, pipeline, analysis tools
experiments/pipeline.py     Historical/research lossy compress + eval pipeline
docs/paper/                 Technical paper (in progress)
submissions/                Submission packaging
upstream/                   Pinned upstream challenge snapshot (read-only)
```

## Historical Timeline

The early renderer/post-filter numbers in this repository are retained as
historical research context. They should not be copied into a public ranked
table unless the row has an `A++`/`A` evidence tag and a cited
`contest_auth_eval.json`.

| Thread | Public/writeup status |
|---|---|
| H.265 and CNN post-filter baselines | Historical context for the scorer-aware workflow |
| Asymmetric warp renderer and pose TTO | Methodology and negative-result context; only exact CUDA rows may rank |
| Gradient obstruction fix | Measurement-methodology contribution; see `docs/paper/03_gradient_bug.md` |
| Public PR replay/deconstruction | Community/historical-record corpus plus exact replay rows when CUDA custody exists |
| Post-deadline hidden-gem lanes | Roadmap until charged archives pass exact CUDA auth eval |

## Methodology

The repository treats compression as a small compiler for contest archives:
representations lower into quantized payloads, entropy-coded packets, runtime
decoders, custody manifests, and exact auth eval artifacts. Positive results,
negative results, harness bugs, and compliance blockers are all preserved as
research signal.

Research state is maintained in durable files (`.omx/research/`, selected
`.omx/state/` ledgers, `docs/`, and `reports/`) so work can resume across
sessions without relying on chat context. Raw provider logs, large artifacts,
and rebuildable experiment outputs stay ignored unless curated into a manifest
or compact ledger.

## Paper

The technical paper is in `docs/paper/`. It covers the asymmetric warp architecture, the gradient obstruction bug discovery, Fridrich-informed loss design, and the rank-1 radial zoom warp derivation.

## Requirements

- Python 3.11+
- PyTorch 2.0+
- ffmpeg (video decode)
- CUDA GPU recommended (MPS and CPU supported for development)

## License

MIT
