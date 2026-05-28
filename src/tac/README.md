# tac - Task-Aware Compression

`tac` is the reusable Task-Aware Compression library for this repository. It
contains the algorithmic compression stack: scorer-aware losses, renderers,
archive grammars, packet compilers, quantizers, exact-eval helpers, sensitivity
maps, and planning primitives that optimize compressed bytes for downstream
machine perception rather than generic human-facing fidelity.

Use **compression** for the project and research program. Use **codec** only for
a concrete encoder/decoder, archive grammar, entropy coder, or wire-format
component such as `tac.packet_compiler`, `tac.mask_codec`, or a substrate-local
`archive.py` / `inflate.py` pair.

## Terminology

Academic and industry-adjacent work uses several overlapping names for this
space:

- **Task-aware compression** and **task-oriented compression**: research terms
  for rate-distortion objectives conditioned on downstream task loss.
- **Video coding for machines (VCM)** and **coding for machines**: standard
  terminology for compression optimized for machine analysis tasks. MPEG lists
  VCM under MPEG-AI and frames it around bitstreams that preserve machine-task
  performance after decoding.
- **Feature coding for machines (FCM)**: related terminology when compressed
  representations are machine features rather than reconstructed pixels. MPEG
  WG 4 describes current scope that includes feature maps, tensors, neural
  aspects of video, and intelligent machine consumption.
- **Semantic communication** and **goal-oriented communication**: adjacent
  communications-theory framing when the receiver's task utility, not generic
  reconstruction fidelity, defines useful information. Use these as analogies
  unless a design actually models a channel/receiver protocol.
- **Neural compression** and **learned compression**: broad implementation
  families that may or may not be task-aware.

`tac` deliberately uses the broader **Task-Aware Compression** name because this
package now includes more than codecs: deterministic packet compilation,
procedural byte derivation, scorer geometry, Venn/sensitivity maps, Pareto
planning, archive custody, and exact scorer contracts.

`TAC` is a repository/package acronym, not a standards-body initialism. When
writing for standards or industry audiences, map `tac` to VCM/FCM/coding for
machines. When writing for ML, information theory, or learned-compression
audiences, map it to task-aware or task-oriented compression. Do not expand it
as "Task-Aware Codec"; codec is an implementation artifact inside the
compression stack, not the full research program.

References:

- MPEG-AI Part 2, [Video coding for machines](https://www.mpeg.org/standards/MPEG-AI/2/)
- MPEG-AI Part 4, [Feature coding for machines](https://www.mpeg.org/standards/MPEG-AI/4/)
- MPEG WG 4, [Video Coding Working Group](https://www.mpeg.org/structure/video-coding/)
- CVPR 2023, [AccelIR: Task-aware Image Compression for Accelerating Neural Restoration](https://openaccess.thecvf.com/content/CVPR2023/papers/Ye_AccelIR_Task-Aware_Image_Compression_for_Accelerating_Neural_Restoration_CVPR_2023_paper.pdf)
- CVPR 2025, [Real-Time Rate Control for Task-Aware Video Compression Using Reinforcement Learning](https://research.nvidia.com/labs/par/publication/realtime_rate_control_video_compression.html)
- NeurIPS 2023, [Task-aware Distributed Source Coding](https://proceedings.neurips.cc/paper_files/paper/2023/file/016c63403370d81c24c1ca0123de6cfa-Paper-Conference.pdf)

## Scope

`tac` owns reusable, contest-relevant compression implementation:

- representation substrates and renderers;
- score-aware training losses and gradient/sensitivity tools;
- quantization, entropy coding, packet compilation, and archive grammars;
- exact score formula helpers and scorer/runtime custody contracts;
- byte-level analyzers, master-gradient consumers, and optimization planners;
- deterministic inflate/runtime components that can be vendored into a
  self-contained submission packet.

`tac` should not own operator state, provider transcripts, hosted dashboards, or
research-council bookkeeping. Those belong in `comma_lab`, `tools/`, `docs/`,
or dated `.omx/research/` ledgers.

## Core Surfaces

| Surface | Role |
|---|---|
| `tac.packet_compiler` | Byte grammars, entropy-coder primitives, and deterministic packet compilation |
| `tac.substrates` | Reusable representation families, renderer substrates, and substrate contracts |
| `tac.master_gradient` | Canonical master-gradient anchors, custody checks, and axis authority |
| `tac.master_gradient_consumers` | Venn maps, per-pair atlases, Wyner-Ziv covariance, and treatment plans |
| `tac.scorer` | Contest score formula, scorer loading, and score-axis helpers |
| `tac.optimizer` | Exact-readiness, dispatch authority, proxy-candidate contracts, and planning gates |
| `tac.procedural_codebook_generator` | Archive-seeded and weight-derived deterministic codebook generation, including procedural-seed authority packets |
| `tac.preflight` | Reusable contest/runtime/package validity checks |
| `tac.reverse_engineering_curation` | Pure reverse-engineering tree curation rules consumed by `tac.preflight` and wrapped by `comma_lab.reverse_engineering` |

## Active Workflows

This package is active research software, not a frozen artifact. Current
high-traffic workflows are:

- exact archive and runtime custody for `[contest-CPU]` and `[contest-CUDA]`
  score claims;
- deterministic packet compilation and byte-closed archive grammar work;
- master-gradient, xray, Venn, and sensitivity-map surfaces for score-aware
  optimization;
- procedural generation from archive-contained seeds or charged archive
  weights, with explicit provenance, mutation-proof checks, and authority
  packets that keep archive-seeded and weight-derived promotion paths separate
  from runtime-constant probes;
- Cathedral autopilot candidate rows that stay false-authority until archive
  bytes, inflate/runtime consumption, full-frame parity, and exact eval land;
- contest-compliant replay and deconstruction of public submissions.

For naming, keep the hierarchy strict:

- `tac`: Task-Aware Compression, the whole package and research stack.
- `codec`: a concrete encoder/decoder, entropy coder, packet grammar, or
  archive/inflate pair inside that stack.
- `comma_lab`: the lab control plane that records, audits, and publishes
  `tac` outputs.

The repository-wide terminology authority is
[`docs/terminology_and_boundaries.md`](../../docs/terminology_and_boundaries.md).
Contest rule and public-PR precedent authority is
[`docs/contest_compliance_authority.md`](../../docs/contest_compliance_authority.md).

## Installation

```bash
pip install tac
```

For this repository checkout, the preferred development install is:

```bash
uv venv
uv pip install -e ".[dev]"
```

Optional extras:

```bash
pip install tac[runtime]      # scorer/runtime closure for GPU eval/training
pip install tac[analysis]     # planning/profiling table tooling
pip install tac[viz]          # plotting and image IO helpers
pip install tac[notebooks]    # marimo notebooks
pip install tac[pr86_replay]  # opt-in LGPL PPMd dependency for public PR replay
```

## Minimal Usage

```python
from tac.master_gradient import OperatingPoint, compute_marginal_coefficients

op = OperatingPoint(
    d_seg=0.0005,
    d_pose=0.00003,
    rate=178_517 / 37_545_489,
    score=0.192,
)
seg_marginal, pose_marginal, rate_marginal = compute_marginal_coefficients(op)
```

```python
from tac.packet_compiler import RankedSidecarSchema, encode_ranked_no_op_sidecar

schema = RankedSidecarSchema(n_pairs=600, n_dims=28)
payload = encode_ranked_no_op_sidecar(dims, delta_indices, schema=schema)
```

## Development Contract

- Keep APIs typed and deterministic.
- Preserve scorer-axis labels: `[contest-CPU]`, `[contest-CUDA]`, and advisory
  local/proxy axes are separate evidence spaces.
- Do not load contest scorers in inflate paths unless a specific sanctioned
  packet/compiler mode proves compliance.
- Emit archive bytes, SHA-256s, runtime custody, and exact-eval provenance for
  score-bearing artifacts.
- Keep new reusable algorithms in `tac`; keep operator orchestration in
  `comma_lab`, `tools/`, or `experiments/`.

## Package Boundary

`tac` is the Task-Aware Compression library/engine. `comma_lab` is the lab and
operations layer.
If a module manipulates archives, tensors, codecs, scorer contracts, or
optimization math, it likely belongs in `tac`. If it manages run state,
provider dispatch, ledgers, public-frontier intake, reports, or release hygiene,
it likely belongs in `comma_lab` or a thin CLI that delegates to `tac`.

## License

MIT
