# PR85 Non-Mask Self-Compression Audit - Worker G - 2026-05-04

Scope: local-only bit-by-bit audit of PR85 non-mask bundle segments and
single-blob container overhead. No training, remote dispatch, lane claim, exact
eval, or score claim was performed.

## Owned Artifacts

- profiler: `experiments/profile_pr85_nonmask_self_compression.py`
- tests: `src/tac/tests/test_profile_pr85_nonmask_self_compression.py`
- JSON audit:
  `experiments/results/pr85_nonmask_self_compression_audit_20260504_worker/pr85_nonmask_self_compression_audit.json`
- Markdown audit:
  `experiments/results/pr85_nonmask_self_compression_audit_20260504_worker/pr85_nonmask_self_compression_audit.md`

## Inputs

- PR85 archive:
  `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- PR85 archive bytes: `236328`
- PR85 archive SHA-256:
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- PR85 member: `x`, stored, `236228` bytes
- PR85 member SHA-256:
  `53bc78effa78cc7850d08a9ddc5488665b93136e9843549d917c17df729a1c50`
- PR90 comparison source:
  `experiments/results/public_pr90_intake_20260504_worker/payload_probe.json`
- PR91 comparison source:
  `experiments/results/public_pr91_intake_20260504_worker/archive.zip`

## Bundle Anatomy

The canonical parser accepted PR85 as
`pr85_v5_micro_24bit_lengths_fixed_bias_region`.

| segment | bytes | decoded bytes | best decoded Brotli bytes | best decoded delta |
| --- | ---: | ---: | ---: | ---: |
| `model` | `57074` | `61590` | `57227` | `+153` |
| `pose` | `1487` | `1806` | `1539` | `+52` |
| `post` | `1400` | `2400` | `1439` | `+39` |
| `shift` | `226` | `603` | `227` | `+1` |
| `frac` | `106` | `179` | `113` | `+7` |
| `frac2` | `149` | `603` | `150` | `+1` |
| `frac3` | `154` | `603` | `155` | `+1` |
| `bias` | `223` | `603` | `223` | `0` |
| `region` | `273` | `603` | `274` | `+1` |
| `randmulti` | `16101` | `27105` | `16454` | `+353` |

Generic zlib/lzma/Brotli wrapping of the encoded bytes was also non-improving
for every non-mask segment. The best encoded wrapper delta was `+4` bytes for
each segment where Brotli was the smallest wrapper. These are no-op or larger
recommendations and are not promoted.

## Single-Blob Overhead

- ZIP container overhead: `100` bytes
- ZIP structural floor for one member named `x`: `100` bytes
- Arbitrary extra ZIP overhead: `0` bytes
- PR85 bundle header: `24` bytes
- The v5 header saves `6` bytes relative to an explicit 30-byte ten-length
  header.

Deterministic outer ZIP baselines:

| container | archive bytes | delta vs PR85 | runtime risk |
| --- | ---: | ---: | --- |
| stored | `236328` | `0` | low |
| deflated | `236403` | `+75` | low |
| bzip2 | `237785` | `+1457` | medium |
| lzma | `239548` | `+3220` | medium |

Conclusion: there is no direct single-blob overhead candidate. The archive is
already at the structural floor, and outer ZIP compression worsens bytes.

## PR91 Non-Mask Identity

PR91 was parsed as the same PR85/PR91 v5 bundle family. Every non-mask segment
is byte-identical to PR85:

- `model`, `pose`, `post`, `shift`, `frac`, `frac2`, `frac3`, `bias`, `region`,
  and `randmulti` all have `byte_delta_vs_pr85 = 0` and matching SHA-256.

Conclusion: PR91 changes the mask only. Any future true PR85 non-mask byte
reduction should stack byte-for-byte with PR91, but this audit found no direct
lossless non-mask reduction.

## PR90 Non-Identity Size Signal

PR90 is not an identity-preserving PR85 recode, but its compact fixed-offset
layout gives two non-mask architecture signals:

| surface | PR85 bytes | PR90 bytes | formula-only delta |
| --- | ---: | ---: | ---: |
| model | `57074` | `56385` | `-689` |
| pose/post/motion/bias/region/randmulti controls | `20119` | `9164` | `-10955` |
| total non-mask | `77193` | `65549` | `-11644` |

These are not score evidence and not drop-in byte substitutions.

## Archive-Builder Candidates

No direct lossless PR85 non-mask self-compression archive-builder candidate is
available from this audit. All runtime-compatible recodes were non-improving or
blocked by fixed-length public-v5 constraints.

Planning-only, non-noop, high-risk architecture candidates:

1. `pr90_qrgb_control_stack_recode_probe`
   - Surface: PR85 pose/post/motion/bias/region/randmulti control stack.
   - Formula-only byte delta: `-10955`.
   - Formula-only rate-score delta: `-0.007294484831`.
   - Runtime risk: high.
   - Required next gate: build a local non-noop candidate with raw-output and
     runtime parity diagnostics; do not exact-eval without the Level-2 lane
     claim protocol.

2. `pr90_qfq4_style_pr85_model_serializer_probe`
   - Surface: PR85 model segment.
   - Formula-only byte delta: `-689`.
   - Formula-only rate-score delta: `-0.000458776819`.
   - Runtime risk: high.
   - Required next gate: local decoded-QH0 to QFQ4/grouped-FP serializer parity
     harness and paired runtime loader proof.

These candidates are implementable builder experiments, not dispatch-ready
candidates. They require state changes and parity proof, so they are not no-op
recommendations.

## Verification

```text
.venv/bin/python -m pytest src/tac/tests/test_profile_pr85_nonmask_self_compression.py -q
```

Result: `3 passed in 0.13s`.

```text
.venv/bin/python experiments/profile_pr85_nonmask_self_compression.py \
  --archive experiments/results/public_pr85_intake_20260503_codex/archive.zip \
  --pr90-probe-json experiments/results/public_pr90_intake_20260504_worker/payload_probe.json \
  --pr91-archive experiments/results/public_pr91_intake_20260504_worker/archive.zip \
  --json-out experiments/results/pr85_nonmask_self_compression_audit_20260504_worker/pr85_nonmask_self_compression_audit.json \
  --markdown-out experiments/results/pr85_nonmask_self_compression_audit_20260504_worker/pr85_nonmask_self_compression_audit.md
```

Result: completed locally with `score_claim=false`, `dispatch_performed=false`,
`direct_lossless_candidate_count=0`, and `no_op_recommendations_promoted=0`.
