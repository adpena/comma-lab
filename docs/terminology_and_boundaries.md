# Terminology And Boundaries

This document is the canonical naming and package-boundary reference for the
repository. It exists to keep public README text, package metadata, and operator
docs aligned while the research system is changing quickly.

## Canonical Names

| Term | Canonical meaning | Use for |
|---|---|---|
| `TAC` / `tac` | Task-Aware Compression | The reusable Python package and the broader optimization objective: compression optimized for a downstream task or scorer. |
| `codec` | Concrete encoder/decoder or wire format | A specific entropy coder, archive grammar, packet compiler, substrate-local `archive.py` / `inflate.py` pair, or submission decoder. |
| `comma-lab` | Repository and lab workspace | Public challenge research, exact replay custody, docs, release hygiene, and OSS-facing artifacts. |
| `comma_lab` | Python operations package | State projection, research-state tracking, preflight adapters, public-frontier intake hygiene, scheduler/reporting support. |
| `pact` | Historical/internal workspace alias | Local paths and older research ledgers only. Do not introduce it as the public project name. |

Never expand TAC as "Task-Aware Codec." That phrasing is too narrow: the package
contains codecs, but it also contains scorer contracts, master-gradient and
sensitivity-map consumers, deterministic packet compilation, procedural
generation, byte profilers, custody validators, and optimization planners.

## Academic And Industry Jargon

The most faithful field-level phrase for this repository is **task-aware
compression**: rate/distortion optimization where distortion is downstream task
loss rather than only human perceptual error.

Adjacent terms should be used precisely:

| Phrase | When to use it |
|---|---|
| Task-aware compression | General research program and package identity. |
| Task-oriented compression | Synonym in papers; useful when citing rate-distortion objectives conditioned on a task. |
| Video coding for machines (VCM) | Standards/industry framing for video bitstreams optimized for machine analysis. |
| Feature coding for machines (FCM) | Standards/industry framing when the compressed object is a machine feature, tensor, or neural representation rather than reconstructed RGB frames. |
| Neural compression / learned compression | Implementation family. It may be task-aware, but not all neural compression is task-aware. |
| Codec | Concrete implementation artifact inside the compression stack. |

Current public package metadata should therefore say "Task-Aware Compression"
and can carry keywords such as `task-aware-compression`,
`video-coding-for-machines`, `neural-compression`, and `perception`.

## Contest Compliance Boundary

The upstream contest contract is archive-centered: a submission provides an
`archive.zip` plus an `inflate.sh` runtime that converts extracted archive
contents into raw frames. Upstream also allows compression-time use of models,
the original video, and other assets, but large score-affecting artifacts such
as neural networks must be included in the archive and count toward rate.

Repo policy is stricter than the permissive edge of that rule:

- Score-bearing information must be byte-closed: archive bytes, archive
  SHA-256, runtime tree SHA, component distances, and exact eval logs are the
  authority.
- Procedural generation from an archive-contained seed, archive-contained
  weights, or a deterministic packet compiler is valid only when the archive is
  self-contained and exact auth eval validates it.
- Constants in `inflate.py` may describe how to decode a charged payload, but
  they must not hide a large uncharged model, lookup table, sidecar payload, or
  scorer-derived artifact.
- Generated code, tiny transducers, and fixed tables are valid design tools
  when they remain self-contained, documented, and cheaper than shipping the
  equivalent bytes directly.
- CPU, CUDA, and local/proxy axes must stay labeled. A local advisory result is
  not a public leaderboard score claim.

This keeps procedural generation and deterministic packet compilation available
without smuggling rate-bearing information outside the scored archive.

## Package Ownership

| Artifact | Owner |
|---|---|
| Archive grammar, entropy coder, packet compiler, scorer contract, quantizer, optimizer primitive | `src/tac/` |
| Reusable contest/runtime validity check | `src/tac/preflight.py`, optionally exposed through `src/comma_lab/preflight/` |
| State projection, provider state, research-state policy, public-intake hygiene, report formatting | `src/comma_lab/` |
| Thin operator CLI | `tools/`, `scripts/`, or `experiments/`, delegating to `tac` or `comma_lab` |
| Dated research decision, negative result, or design memo | `.omx/research/` |
| Public explanatory narrative | `README.md`, `docs/`, and package README files |

If a module computes or validates bytes, tensors, score formula fields, archive
layout, or runtime-safe decoding, it probably belongs in `tac`. If it records,
audits, routes, or publishes the work, it probably belongs in `comma_lab`.

## Documentation Contract

The root README, `src/tac/README.md`, `src/comma_lab/README.md`,
`src/tac/__init__.py`, `src/comma_lab/__init__.py`, and `pyproject.toml` must
agree on these points:

1. TAC means Task-Aware Compression.
2. A codec is a concrete implementation artifact, not the whole project.
3. `comma_lab` is the operations/control-plane package, not a second
   compression engine.
4. Score authority comes from byte-closed exact evaluation, not prose.
5. Procedural generation is contest-eligible only when self-contained and
   charged through archive bytes or a clearly sanctioned tiny runtime contract.

Run the terminology guard before public-facing documentation commits:

```bash
.venv/bin/python tools/check_tac_terminology.py --strict
```

## References

- Upstream contest rules: `upstream/README.md`
- MPEG-AI Part 2: Video coding for machines
- MPEG-AI Part 4: Feature coding for machines
- MPEG WG 4: Video Coding Working Group
- CVPR 2023 AccelIR: Task-aware Image Compression for Accelerating Neural Restoration
