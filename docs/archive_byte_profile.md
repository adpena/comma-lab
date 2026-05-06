# Archive Byte Profile

`experiments/profile_archive_bytes.py` profiles one or more ZIP archives without
extracting payloads, importing scorer code, or dispatching jobs. The output is
for byte attribution and compression opportunity discovery only:
`score_claim=false` and `evidence_grade=byte_profile_only`.

Example:

```bash
.venv/bin/python experiments/profile_archive_bytes.py \
  experiments/results/submission_packet_c067_20260502/automated_packet/archive.zip \
  --json-out experiments/results/c067_archive_byte_profile.json \
  --markdown-out experiments/results/c067_archive_byte_profile.md
```

The JSON and markdown include:

- archive total bytes and contest rate term, `25 * bytes / 37545489`
- per-member compressed size, uncompressed size, ZIP method, CRC, SHA-256, and
  byte histogram statistics
- extension totals and top-level path group totals
- ZIP overhead estimate from non-payload bytes
- duplicate member-name detection and duplicate payload-hash detection
- top compressed-byte contributors
- cross-archive duplicate payload hashes when more than one archive is profiled

Unsafe ZIP member paths fail closed during introspection. The profiler rejects
absolute paths, parent traversal, Windows drive paths, backslashes, empty names,
and NUL bytes before reading member contents.

Histogram statistics are computed from member contents streamed through
`ZipFile.open()`. The tool does not write member contents to disk by default.
For compressed ZIP members, the compressed byte count is reported from the ZIP
directory and the histogram describes the logical member payload after ZIP
decompression.

Use this output to choose candidate byte targets for Lagrangian allocation,
water-filling, duplicate removal, and self-compression probes. Do not use it to
promote, rank, kill, or claim score for a lane; exact CUDA auth eval on the
identical archive bytes remains the score truth.

## Packed Stream And Atom Accounting

`experiments/profile_archive_byte_accounting.py` is the deeper local profiler
for single-blob contest archives (`p`, `x`, RPK/RP2, PR65/PR67-style packed
payloads). It remains local and non-promotable, but it can now normalize
sidecar evidence into a reusable rate-distortion table:

```bash
.venv/bin/python experiments/profile_archive_byte_accounting.py \
  --archive experiments/results/.../archive.zip \
  --eval-json experiments/results/.../contest_auth_eval.json \
  --component-trace experiments/results/.../component_trace.json \
  --action-json experiments/results/.../candidate_actions.json \
  --output-json experiments/results/.../byte_accounting.json \
  --output-csv experiments/results/.../rate_distortion_atom_table.csv \
  --output-md experiments/results/.../byte_accounting.md \
  --output-png experiments/results/.../byte_accounting.png
```

For each archive it emits:

- `byte_ledger` / `rate_distortion_atom_table`: normalized archive, ZIP member,
  packed stream, action-record, and component-trace-pair rows.
- `bytes`, SHA-256 where known, and contest rate cost
  `25 * bytes / 37545489`.
- known or estimated pose/seg/score deltas when supplied by action JSON.
- hard-pair component pressure from `component_trace.json`, including the
  rate-only break-even bytes for removing that component contribution.
- rate-only benefit estimates for local self-compression probes.

The CLI accepts repeated `--archive` values for deterministic collection mode.
When multiple archives are supplied, repeat `--eval-json` and
`--component-trace` in the same order if sidecars are available. `--action-json`
records are treated as local planning inputs and copied into each profiled
archive's atom table with their source JSON path and SHA-256.
