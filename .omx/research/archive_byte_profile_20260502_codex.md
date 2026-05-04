# Archive Byte Profile Tool Note - 2026-05-02

Implemented a narrow byte-attribution profiler for C067/Apogee and
public/top-submission-like ZIP archives.

Evidence grade: `byte_profile_only`.
Score claim: `false`.

The profiler records deterministic archive and member byte accounting,
per-member histogram statistics, extension/path-group totals, ZIP overhead,
duplicate member names, duplicate payload hashes, top contributors, and the
contest rate term `25 * bytes / 37545489`. It streams member contents through
the Python standard-library ZIP reader and rejects zip-slip names before
introspection.

Research use: the profile is compiler feedback for Lagrangian byte allocation,
water-filling, and self-compression opportunity discovery. Large contributors
identify streams where one byte saved has immediate rate-term value; extension
and group totals show where representation families are spending bytes; hash
duplicates expose no-op or reuse opportunities; histogram entropy gives a cheap
signal for whether a stream may still contain lossless structure.

Non-claim boundary: this tool does not inflate payloads, run CUDA auth eval,
dispatch jobs, or validate scorer behavior. Its output can prioritize archive
edits and exact-eval candidates, but cannot promote, rank, kill, retire, or
anchor paper score claims.

## Addendum - 2026-05-02T20:07Z

Public and C067 archive forensics are now reproducible:

- `experiments/results/archive_byte_profile_20260502/public_and_c067_profile.json`
- `experiments/results/archive_byte_profile_20260502/public_and_c067_profile.md`

The ZIP-level finding is strict: PR65, PR67, and C067/Apogee are all
single-member stored ZIP archives, so there is no meaningful outer ZIP overhead
left to attack. C067/Apogee is `276214` bytes with a single member `p` of
`276114` bytes and `100` bytes of ZIP overhead. PR67 is `276564` bytes with
the same single-member shape. PR70's local archive is malformed/nonstandard
for Python's ZIP reader; the profiler now records it as an invalid forensic
input instead of aborting the whole collection.

The deeper stream-level profiler also now understands the public PR65 compact
`x` bundle as analysis-only external grammar:

- `experiments/results/public_archive_byte_accounting_20260502/pr65/archive_byte_accounting.json`
- `experiments/results/public_archive_byte_accounting_20260502/pr65/archive_byte_accounting.md`
- `experiments/results/public_archive_byte_accounting_20260502/pr67/archive_byte_accounting.json`
- `experiments/results/public_archive_byte_accounting_20260502/pr67/archive_byte_accounting.md`

PR65 stream budget:

```text
masks.mkv             219472 bytes
renderer.bin           57074 bytes
optimized_poses.bin     1487 bytes
qpost.randmulti         3731 bytes
qpost.post              1400 bytes
qpost.region             273 bytes
qpost.shift              226 bytes
qpost.bias               223 bytes
qpost.frac3              154 bytes
qpost.frac2              149 bytes
qpost.frac               106 bytes
```

This confirms PR65's useful trick is not generic compression; it spends a
charged, structured postprocess/control budget on top of the shared mask
stream. Prior exact diagnostics already show global and pair-gated qpost
transfers are not standalone wins, so the next valid use is learned or
pair-local qpost-style repair after a larger mask-rate win, not blind PR65
sidecar stacking.

Verification:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_archive_byte_profile.py \
  src/tac/tests/test_profile_archive_byte_accounting.py \
  src/tac/tests/test_qzs3_postprocess_candidate.py \
  src/tac/tests/test_remote_lane_sjkl_c067_script.py -q

27 passed, 1 warning
```
