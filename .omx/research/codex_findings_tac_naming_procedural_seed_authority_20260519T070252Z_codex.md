# Codex Findings - TAC Naming And Procedural Seed Authority

Timestamp: 2026-05-19T07:02:52Z
Author: Codex
Scope: documentation and reusable TAC authority helper
Evidence axis: docs/code/test guard, no score claim

## Verdict

`tac` should canonically expand to **Task-Aware Compression**, not
Task-Aware Codec.

Reason: the repository and the field use compression/coding for the broader
rate-distortion program, while codec is narrower: a concrete encoder/decoder,
entropy coder, archive grammar, wire format, or inflate/archive pair. The repo
already encoded this in the root README, package README, terminology doc,
package metadata, and the `tools/check_tac_terminology.py` guard. This landing
hardens the boundary rather than renaming it.

## Authority Sources

- Internal authority: `README.md`, `src/tac/README.md`,
  `src/comma_lab/README.md`, `docs/terminology_and_boundaries.md`,
  `docs/contest_compliance_authority.md`, and `pyproject.toml` already use
  Task-Aware Compression.
- External terminology: MPEG uses Video Coding for Machines / Feature Coding
  for Machines; recent papers use Task-aware Image Compression, Task-aware
  Video Compression, and Task-aware Distributed Source Coding.
- Contest authority: upstream README and public PR precedents support
  scorer-aware compression, but block uncharged large artifacts or script-side
  payload relocation at inflate time.

## Procedural Generation Clarification

Procedural generation from seeds, weights, generated code, or tiny transducers
is a first-class TAC path when the information that determines scored frames is
byte-closed.

Canonical promotion path:

- `archive_seeded`: seed/weights/tables/transducer bytes live in
  `archive.zip`; exact eval can rank after custody and mutation proofs.
- `weight_derived`: seed/codebook derives from an already charged archive
  member; exact eval can rank after source-member SHA and no-new-bytes proofs.

Probe path:

- `runtime_constant`: generic decoder constants in `inflate.py` may be valid
  code, but per-video score-bearing literals remain research-only unless an
  explicit compliance ruling proves they are decoder logic rather than payload
  relocation.

## Landing

- Added `tac.procedural_codebook_generator.build_procedural_seed_authority_packet`
  to emit a fail-closed authority packet for archive-seeded, weight-derived,
  and runtime-constant variants.
- Exported the helper from `tac.procedural_codebook_generator`.
- Added tests proving archive-seeded promotion is not blocked by a simultaneous
  runtime-constant probe, while the packet itself remains `score_claim=false`.
- Added `CONTRIBUTING.md` naming guidance and extended
  `tools/check_tac_terminology.py` to guard it.
- Updated `src/tac/README.md`, `docs/terminology_and_boundaries.md`, and
  `docs/contest_compliance_authority.md` with the authority-packet protocol and
  additional terminology references.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_procedural_codebook_generator.py src/tac/tests/test_tac_terminology_guard.py`
  -> 28 passed.
- `.venv/bin/ruff check src/tac/procedural_codebook_generator/authority.py src/tac/procedural_codebook_generator/__init__.py src/tac/tests/test_procedural_codebook_generator.py src/tac/tests/test_tac_terminology_guard.py tools/check_tac_terminology.py`
  -> passed.
- `.venv/bin/python tools/check_tac_terminology.py --strict --json`
  -> `ok: true`.

## Next Consumer Hook

Cathedral autopilot should consume procedural candidates through the authority
packet before routing exact-eval dispatch. The packet makes the selected
promotion mode explicit and preserves any alternate runtime-constant probe as a
separate research branch.
