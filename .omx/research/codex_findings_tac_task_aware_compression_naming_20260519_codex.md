# Codex Findings - TAC Naming Authority And Documentation Canonicalization - 2026-05-19

## Verdict

Standardize `tac` as **Task-Aware Compression**.

Use **codec** as a narrower implementation noun: a concrete encoder/decoder,
archive grammar, entropy coder, wire format, or substrate-local runtime pair.
Do not use "Task-Aware Codec" as the package expansion because the repository
now covers deterministic packet compilation, procedural byte derivation,
scorer geometry, sensitivity maps, Venn cells, optimizer authority, archive
custody, and exact-eval dispatch gates.

## External Terminology Anchors

- MPEG-AI Part 2 uses **Video coding for machines** and defines bitstream syntax,
  decoding, and descriptors optimized for bitrate and machine-task performance:
  https://www.mpeg.org/standards/MPEG-AI/2/
- MPEG WG 4 describes video-related compression for both human vision and
  intelligent machine consumption, including neural networks, feature maps, and
  tensors:
  https://www.mpeg.org/structure/video-coding/
- CVPR 2023 uses **Task-Aware Image Compression** for compression optimized
  around an end-to-end downstream restoration task:
  https://openaccess.thecvf.com/content/CVPR2023/html/Ye_AccelIR_Task-Aware_Image_Compression_for_Accelerating_Neural_Restoration_CVPR_2023_paper.html

## Canonical Repo Language

- `tac`: Task-Aware Compression library and reusable algorithmic engine.
- `comma_lab`: lab operations and state/custody layer around `tac`.
- `codec`: concrete encoder/decoder, entropy coder, archive grammar, or
  substrate wire format.
- `packet compiler`: deterministic lowering layer from structured candidate
  state to charged bytes.
- `exact-ready`: byte-closed archive/runtime packet with custody sufficient for
  exact auth eval dispatch.

## Landed Documentation Updates

- `pyproject.toml`: package description now expands TAC as Task-Aware
  Compression and adds `task-aware-compression` as a keyword.
- `src/tac/README.md`: rewritten as the canonical package README with scope,
  terminology, module map, references, install modes, and boundary rules.
- `src/comma_lab/README.md`: added canonical lab/operations README and module
  boundary rules.
- `src/tac/__init__.py`: package docstring now matches the canonical expansion.
- `src/comma_lab/__init__.py`: package docstring now names the operations role.
- `README.md`: root README now states the `tac`/`codec`/`comma_lab` boundary.

## Follow-Up

Historical `.omx/research` and dated design docs may keep old wording as
provenance. Future public docs, package metadata, README surfaces, and new code
docstrings should use Task-Aware Compression unless the object is specifically
a codec implementation.
