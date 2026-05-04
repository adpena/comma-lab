# PR92 QMA9 Native Decode Profile - 2026-05-04

## Scope

Local CPU-only native decode observability for the PR92 compact-bundle QMA9 mask
slice. This is not a score claim and did not dispatch remote GPU, eval, or
training work.

## Code Artifact

- Added `experiments/profile_qma9_native_decode.py`
- Added `src/tac/tests/test_profile_qma9_native_decode.py`

The profiler:

- reads a direct QMA9 stream or ZIP archive member;
- extracts compact v5 micro-bundle mask slices fail-closed;
- runs native `runtime-rs` `qma-codec metadata` and `qma-codec decode`;
- records archive/member custody, QMA9 header, decoder SHA, metadata timing,
  decode bytes/sec, decoded SHA-256, repeat determinism, and no-op checks.

## Verification

Focused unit/native smoke:

```text
.venv/bin/python -m pytest src/tac/tests/test_profile_qma9_native_decode.py -q
```

Result:

```text
2 passed in 0.29s
```

Full local PR92 profile command:

```text
.venv/bin/python experiments/profile_qma9_native_decode.py \
  --input experiments/results/public_pr92_intake_20260504_codex/archive.zip \
  --member x \
  --repeat 2 \
  --build-profile release \
  --output-json experiments/results/pr92_qma9_native_decode_profile_20260504_codex/timing_report.json
```

Result artifact:

- `experiments/results/pr92_qma9_native_decode_profile_20260504_codex/timing_report.json`
- schema: `qma9_native_decode_timing_report_v1`
- archive SHA-256: `f0dedeb7ad3c019ab3f4e2dea317f3192408e47a573897adb208030709d01490`
- member `x` SHA-256: `89f14d331063125c88db0b4e3e51a92f21d2edc175a64d5c9cb6f873130763d8`
- extracted QMA9 mask bytes: `159011`
- extracted QMA9 SHA-256: `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`
- QMA9 header: `600` frames, `512x384`, `158991` bitstream bytes,
  `117964800` decoded bytes
- native decoder: `runtime-rs/target/release/qma-codec`
- native decoder SHA-256: `6850150301ff3791a92c9434b3312ef21ce1e8dbd434308e06a5e6cd5a4ef5d7`
- metadata timing: `0.1241382909938693` seconds
- decode repeat 1: `1.3427478749654256` seconds,
  `87853276.254887` decoded bytes/sec
- decode repeat 2: `1.3373528749798425` seconds,
  `88207684.15499765` decoded bytes/sec
- repeated decoded SHA-256:
  `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`
- determinism: passed
- no-op check: decoded output nonempty, decoded SHA differs from encoded QMA9 SHA

## Immediate Use

The next native-lowering pass can use the decoded SHA above as the PR92 QMA9
storage-order parity target before attempting RMB1/STBM stack changes. The
decode timing gives a local baseline for deciding whether future native codec
variants are an observability win or a regression.
