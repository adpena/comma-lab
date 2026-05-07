# PR91 HPM1 Reference Teacher-Forcing Next-Row Diagnostic - 2026-05-07

Scope: PR91/HPM1 entropy-contract recovery only. Local CPU probe; no scorer
load, no GPU, no remote dispatch, no lane claim, and no score claim.

Artifact:

- `.omx/research/pr91_hpm1_reference_teacher_forcing_nextrow_20260507_codex.json`
- Artifact SHA-256:
  `1744dde126a0380c37cffff8c99f5730209317466deda6817458c2558eace9d7`
- Canonical payload without tool manifest SHA-256:
  `cd834d56eeb0c5281ea2fce5dc65a59600109056e7a85f610a4081a16f5c3b32`

Command:

```bash
.venv/bin/python tools/audit_pr91_hpm1_reference_teacher_forcing_probe.py \
  --archive experiments/results/public_pr91_intake_20260504_codex/archive.zip \
  --reference-tokens experiments/results/pr85_qma9_mode_sweep_20260504_codex/adaptive6pr.decoded.raw \
  --reference-layout legacy_assume_nhw \
  --spatial-order-candidates tile_major_row_major,phase_major_row_major \
  --reference-window-before 2 \
  --reference-window-after 5 \
  --json-out .omx/research/pr91_hpm1_reference_teacher_forcing_nextrow_20260507_codex.json
```

Rows:

| candidate | decoded-context failure | reference-forced failure | new diagnostic |
| --- | --- | --- | --- |
| `tile_major_row_major` | frame `0`, group `12`, symbol `210`, decoded-before `8274` | frame `0`, group `5`, symbol `305`, decoded-before `2033` | false lead; reference forcing regresses and `RangeDecoder.maybe_exhausted=false` before/after failure |
| `phase_major_row_major` | frame `0`, group `11`, symbol `14`, decoded-before `6926` | frame `0`, group `17`, symbol `437`, decoded-before `15989` | still live; reference window at failure records symbols `4,2,2,2,2,2,2,1` at relative rows `-2..5`; `RangeDecoder.maybe_exhausted=false` before/after failure |

Interpretation:

- The `phase_major_row_major` teacher-forced row remains the only advancing
  candidate in this probe.
- The failure is not explained by simple stream exhaustion/finalization at the
  failed decode call: public `RangeDecoder.maybe_exhausted()` is `false`
  immediately before and after the thrown decode error.
- The new reference window is a canonical PR85/QMA9 token diagnostic only. It
  does not prove PR91 encoder semantic tokens, full HPM1 decode, byte-exact
  re-encode, score validity, or dispatch readiness.

Remaining blockers:

- Recover the encoder-side probability/range grammar at the phase-major
  reference-forced row.
- Recover or derive true PR91 semantic tokens if they differ from the PR85/QMA9
  reference.
- Decode all `600` HPM1 frames and re-encode the exact token stream before any
  exact-eval dispatch can be considered.
