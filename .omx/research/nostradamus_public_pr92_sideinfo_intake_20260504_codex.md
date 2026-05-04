# Nostradamus Public PR92 Side-Info Intake - 2026-05-04

## Scope

Built one deterministic local intake artifact for likely PR95-PR100 frontier
moves: strict public-archive ZIP custody plus PR85-family bundle/side-info
byte diffing. No remote GPU, training, exact eval, scorer load, or dispatch was
performed.

## Public Signals Inspected

- PR92: `qzs3_range_joint_r258 (0.26)`
  <https://github.com/commaai/comma_video_compression_challenge/pull/92>
- PR92 report states CUDA, 600 samples, archive SHA
  `f0dedeb7ad3c019ab3f4e2dea317f3192408e47a573897adb208030709d01490`,
  archive bytes `236516`, PoseNet `0.00018963`, SegNet `0.00057675`.
- PR92 author comment: range-coded semantic masks plus compact joint-frame
  renderer and small correction stream, influenced by PR85.
- PR94 was inspected as a pose/MPS public signal, not a new local HPAC
  contract source. It reports Mac MPS, not promotable CUDA score evidence:
  <https://github.com/commaai/comma_video_compression_challenge/pull/94>

## New Artifact

- Tool: `src/tac/public_frontier_intake.py`
- CLI: `experiments/profile_public_frontier_intake.py`
- Test: `src/tac/tests/test_public_frontier_intake.py`
- PR92 JSON:
  `experiments/results/public_pr92_intake_20260504_codex/public_frontier_intake_profile.json`
- PR92 Markdown:
  `experiments/results/public_pr92_intake_20260504_codex/public_frontier_intake_profile.md`

The tool is byte-only. It validates central/local ZIP names, duplicate names,
zip-slip/hidden member blockers, PR85-family primary bundle slices, charged
side-info members, and segment deltas against named baselines. It writes
`score_claim=false`, `promotion_eligible=false`, and evidence grade
`external_archive_byte_intake_only`.

## PR92 Local Anatomy

- Archive bytes/SHA-256:
  `236516`,
  `f0dedeb7ad3c019ab3f4e2dea317f3192408e47a573897adb208030709d01490`
- Strict ZIP: passed.
- Members:
  - `x`: `235952` bytes, SHA
    `89f14d331063125c88db0b4e3e51a92f21d2edc175a64d5c9cb6f873130763d8`
  - `a`: `386` bytes, SHA
    `5422d47b4092e7304649cc49f4d4c8c7efa9c3d5a4fc7d39ab63cf2518e0897e`,
    magic `RSB1X...`
- Primary member `x` parses as PR85 v5 micro header.
- Versus PR85:
  - archive delta: `+188` bytes
  - changed PR85 segment: `randmulti`
  - `randmulti`: `16101 -> 15825` bytes, candidate magic `RMB1`
  - charged side-info member `a` consumes `386` bytes; the net archive is
    still larger after side-info and ZIP overhead.
- Versus PR85+STBM1BR A++ frontier:
  - archive delta: `+6760` bytes
  - changed segments: `mask` and `randmulti`
  - PR92 keeps PR85 QMA9 mask bytes (`159011`) while the frontier uses
    STBM1BR (`152439`), so the public side-info trick does not supersede the
    current mask recode.

## Anticipated Frontier Move

Competitors are likely to keep the PR85 single-member bundle grammar and add
small charged side-info members that shrink one in-bundle control stream
instead of replacing the dominant mask codec. This is stackable only if the
side-info runtime contract can be ported onto the STBM mask frontier without
undoing the `6572` mask-byte win.

## Next Action

Use the new intake diff to isolate and decode PR92's `RSB1` side-info and
`RMB1` randmulti contract, then build a local planning-only transplant sketch:
PR85+STBM mask plus PR92 randmulti/side-info, with byte closure and runtime
contract blockers recorded before any exact-eval dispatch plan.

## Verification

```bash
.venv/bin/python -m py_compile src/tac/public_frontier_intake.py experiments/profile_public_frontier_intake.py src/tac/tests/test_public_frontier_intake.py
.venv/bin/python -m pytest src/tac/tests/test_public_frontier_intake.py -q
.venv/bin/python experiments/profile_public_frontier_intake.py \
  --archive experiments/results/public_pr92_intake_20260504_codex/archive.zip \
  --label PR92_qzs3_range_joint_r258 \
  --baseline PR85=experiments/results/public_pr85_intake_20260503_codex/archive.zip \
  --baseline PR85_STBM1BR=experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/archive.zip \
  --json-out experiments/results/public_pr92_intake_20260504_codex/public_frontier_intake_profile.json \
  --markdown-out experiments/results/public_pr92_intake_20260504_codex/public_frontier_intake_profile.md
```

Focused pytest result: `3 passed in 0.08s`.

## Recursive Review

- OSS hygiene: additive module/CLI with docstrings, type hints, no hard-coded
  credentials, no machine-private URLs beyond local artifact paths.
- Determinism: fixed sorting for baseline reports and no network dependency in
  the tool.
- Contest compliance: no scorer load, no inflate, no GPU dispatch, no score
  promotion. Public PR scores remain external until exact CUDA replay.
- Paper-citability: artifact paths, SHA-256s, byte deltas, and evidence grade
  are recorded.
- Secret leakage: no tokens, SSH targets, provider job endpoints, or private
  account metadata added.
