# Codex Findings: TAC Terminology Authority

Timestamp: 2026-05-19T06:45:00Z
Agent: codex
Scope: `tac` / `comma-lab` public naming, README rigor, and contest-compliance
authority language.

## Verdict

Use `TAC` / `tac` as **Task-Aware Compression**, not Task-Aware Codec.

The acronym is repository-local, so public text must expand it on first use and
then map the work to field-standard language:

- research / ML: task-aware compression, task-oriented compression,
  task-aware distributed source coding;
- standards / industry: coding for machines, video coding for machines (VCM),
  feature coding for machines (FCM);
- adjacent theory: semantic or goal-oriented communication only when the design
  explicitly models receiver utility or a communication channel.

## Rationale

`codec` is too narrow for the current `tac` package. The package owns concrete
codecs, but it also owns scorer contracts, sensitivity maps, master-gradient
consumers, deterministic packet compilation, procedural generation, authority
packets, custody validators, and optimization planners. Calling the whole
surface a codec makes non-codec abstractions look like implementation detail.

The upstream contest also centers authority on archive bytes and exact eval, so
the public docs should separate:

- `tac`: reusable Task-Aware Compression library and algorithmic engine;
- `comma_lab`: lab operations, state projection, public-frontier hygiene,
  release/reporting, and preflight adapters;
- `codec`: concrete encoder/decoder, entropy coder, archive grammar, packet
  compiler, or inflate/runtime pair.

## Sources Checked

- Upstream contest README: archive.zip is compressed data; `inflate.sh` converts
  extracted archive data to raw frames; large artifacts such as PoseNet and
  SegNet must be included in the archive and count toward compressed size.
- MPEG-AI Part 2: Video coding for machines defines bitrate-efficient video /
  descriptor bitstreams for machine-task performance after decoding.
- MPEG-AI Part 4 / ISO CD 23888-4: Feature coding for machines targets feature
  bitstreams efficient in bitrate, machine-task performance, and computational
  complexity.
- NeurIPS 2023 task-aware distributed source coding: formulates compression
  around minimizing task loss rather than reconstructing the source.
- Independent xhigh reviewer converged on the same terminology: `TAC` is
  repository shorthand for Task-Aware Compression; MPEG/ISO-facing prose should
  use VCM/FCM/coding-for-machines; semantic/goal-oriented communication remains
  adjacent framing unless a receiver/channel model is explicit.

## Landed Changes

- Root README now states that `TAC` is repository/package shorthand and maps the
  project to task-aware compression, task-oriented compression, VCM, FCM, and
  coding for machines.
- `src/tac/README.md` now distinguishes external-audience mappings from the
  local acronym and reserves codec for concrete implementation artifacts.
- `docs/terminology_and_boundaries.md` now includes an explicit authority model:
  upstream contest rules, standards/industry terminology, and research
  literature.
- `tools/check_tac_terminology.py` now guards the new authority-model language.

## Adversarial Notes

- Do not imply that `TAC` is an MPEG/ISO standards term. It is local shorthand.
- Do not use "semantic communication" as the primary label unless the design
  actually has receiver/channel semantics; otherwise it is only an analogy.
- Do not let procedural generation language imply free uncharged payloads.
  Score-bearing seeds, weights, tables, generated code, and distilled
  transducers must be archive-charged unless a stricter authority memo proves
  they are decoder logic.
