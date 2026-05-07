# PR91 HPM1 Reference Teacher-Forcing Probe - 2026-05-07

Scope: Worker HPM1-grammar local CPU-only grammar recovery. No scorer load, no
GPU or remote dispatch, no score claim.
Evidence grade: `empirical` local CPU hypothesis probe only. Not rankable, not
promotion eligible, not full standalone HPM1 decode proof, and not byte-exact
re-encode proof.

Artifact:

- `.omx/research/pr91_hpm1_reference_teacher_forcing_probe_20260507_worker_hpm1.json`

Command:

```bash
.venv/bin/python tools/audit_pr91_hpm1_reference_teacher_forcing_probe.py \
  --archive experiments/results/public_pr91_intake_20260504_codex/archive.zip \
  --reference-tokens experiments/results/pr85_qma9_mode_sweep_20260504_codex/adaptive6pr.decoded.raw \
  --reference-layout legacy_assume_nhw \
  --spatial-order-candidates tile_major_row_major,phase_major_row_major \
  --json-out .omx/research/pr91_hpm1_reference_teacher_forcing_probe_20260507_worker_hpm1.json
```

Result rows:

| candidate | decoded-context failure | PR85/QMA9 reference-forced failure | interpretation |
| --- | --- | --- | --- |
| `tile_major_row_major` | `frame=0 group=12 symbol=210 decoded_before=8274` | `frame=0 group=5 symbol=305 decoded_before=2033` | Reference teacher-forcing makes this candidate fail earlier; not the recovery path. |
| `phase_major_row_major` | `frame=0 group=11 symbol=14 decoded_before=6926` | `frame=0 group=17 symbol=437 decoded_before=15989` | Reference teacher-forcing advances this candidate; semantic-context hypothesis remains open for phase-major only. |

Reference token custody:

- Path: `experiments/results/pr85_qma9_mode_sweep_20260504_codex/adaptive6pr.decoded.raw`
- SHA-256: `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`
- Layout: `legacy_assume_nhw`

Remaining blockers:

- The advanced `phase_major_row_major` row still fails range decode at
  `decoded_before=15989`; it is not full frame decode.
- Candidate rows are requested-prefix entropy probes only; a row advancing does
  not imply full decode, score validity, or exact-eval dispatch readiness.
- The reference tensor is PR85/QMA9 semantic context, not proven PR91 encoder
  teacher tokens.
- Range-coder construction/finalization and probability numeric contract remain
  unresolved.
- No full 600-frame HPM1 decode or byte-exact range re-encode is proven.
