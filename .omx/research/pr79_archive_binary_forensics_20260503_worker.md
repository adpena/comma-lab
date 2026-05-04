# PR79 Archive Binary Forensics Worker - 2026-05-03

Evidence grade: empirical reverse-engineering forensics. No GPU dispatch was
performed. No score claim is made here; exact CUDA auth eval remains the score
truth.

## Artifacts

- Analyzer: `experiments/analyze_public_qpose_payload_family.py`
- Tests: `src/tac/tests/test_analyze_public_qpose_payload_family.py`
- Output directory: `experiments/results/pr79_archive_binary_forensics_20260503_worker/`
- Main profile: `experiments/results/pr79_archive_binary_forensics_20260503_worker/family_profile.json`
- Human summary: `experiments/results/pr79_archive_binary_forensics_20260503_worker/family_profile.md`
- Full action diff: `experiments/results/pr79_archive_binary_forensics_20260503_worker/action_record_multiset_union_diff.csv`
- Full QP1 word diff: `experiments/results/pr79_archive_binary_forensics_20260503_worker/pose_qp1_q0_word_diff.csv`
- QP1 byte diff vs C-102: `experiments/results/pr79_archive_binary_forensics_20260503_worker/pose_qp1_byte_diff_vs_c102.csv`
- Stream identity matrix: `experiments/results/pr79_archive_binary_forensics_20260503_worker/stream_identity_matrix.csv`

## Archive Set

| label | archive bytes | payload bytes | ZIP overhead | payload format | archive SHA-256 |
|---|---:|---:|---:|---|---|
| C-102 | 276485 | 276385 | 100 | `public_pr75_qzs3_qp1_segactions_p3` | `79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8` |
| PR75 | 276481 | 276381 | 100 | `public_pr75_qzs3_qp1_segactions_fixed_slices` | `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746` |
| PR77 | 276551 | 276451 | 100 | `public_pr75_qzs3_qp1_segactions_fixed_slices` | `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af` |
| PR79 | 277388 | 277288 | 100 | `public_pr75_qzs3_qp1_segactions_fixed_slices` | `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446` |

ZIP overhead is already the 100-byte single stored member minimum for member
name `p`: 31 local header bytes, 47 central-directory bytes, and 22 EOCD bytes.
There is no standards-compliant ZIP-overhead win unless the member name becomes
empty, which violates the repo's nonempty-member custody rule.

## Stream Identity

Mask stream is byte-identical and decode-identical across all four archives:
charged SHA `d1ae0d39c848e5715b74bb0122269066a4a1ab60ba9e12f34e70fd62ac136d87`,
decoded SHA `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`.

Renderer stream is byte-identical and decode-identical across all four
archives: charged SHA
`e892539adec2406f87c824accc0effc80911f160ca8d324429c5d2bac175f2cf`,
decoded SHA `30159b6ace27a4013d1516c340d58f6d683e6847429fd3d6303a2c650aa2abef`.

C-102 and PR75 action streams are byte-identical and runtime-record-identical:
255 charged bytes, 108 runtime records, runtime SHA
`5af557cdf4c8c4c3747b06c1daabfe34581b62cb9f317d41593b836c6727427a`.
PR77 changes the action stream to 325 charged bytes and 147 runtime records.
PR79 changes it to 1162 charged bytes and 672 runtime records.

PR75, PR77, and PR79 share the same QP1 pose stream: 898 charged bytes,
raw QP1 SHA `8d77f6a39d1a84eca78fbe8fa5ddc31f6ada29814b49e32daeb800ea84a015cc`.
C-102 uses a different 892-byte Brotli QP1 stream with raw QP1 SHA
`80094e20f4f6cd29869e043eb6d224a6697ecd7fb0e77728ddcd6c7a05fccb9a`.
The C-102 vs public pose bitstream diff has 185 raw-byte differences and 192
quantized q0 word differences across the 600 QP1 rows.

## Action Diff

| comparison | common | left-only | right-only | sequence equal |
|---|---:|---:|---:|---|
| C-102 vs PR75 | 108 | 0 | 0 | true |
| C-102 vs PR77 | 56 | 52 | 91 | false |
| C-102 vs PR79 | 105 | 3 | 567 | false |
| PR77 vs PR79 | 62 | 85 | 610 | false |

The full multiset diff is in
`experiments/results/pr79_archive_binary_forensics_20260503_worker/action_record_multiset_union_diff.csv`.

## Entropy And Repacking

Whole-payload generic compression probes do not beat stored `p`: Brotli, zlib,
and LZMA all expand the already high-entropy nested payloads. Segment-level
charged-byte probes also do not beat current bytes.

Decoded-to-Brotli re-encode probes find no positive lossless stream win for
C-102: masks re-encode to the same 219472 bytes; actions re-encode to the same
255 bytes; renderer and QP1 pose re-encode one byte larger than current.

## Lossless / Semantics-Preserving Opportunities

1. Strip C-102 `P3` header and use existing fixed-slice parser fallback.
   - Saves exactly 10 archive bytes.
   - New archive-byte estimate: 276475.
   - Analyzer verified robust unpacker parses the stripped 276375-byte payload
     as fixed slices and all decoded member SHA-256 values remain identical.
   - This is implementable but not strategically sufficient by itself.

2. ZIP overhead squeeze.
   - Saves 0 bytes under strict standard ZIP custody.
   - Already at 100-byte minimum for one stored member named `p`.

3. Lossless nested recompression.
   - Saves 0 bytes on C-102 under tested Brotli q5/q9/q11, zlib-9, and LZMA-9
     probes.

## Break-Even To 0.31

C-102 input exact score from `contest_auth_eval.json`:
`0.31514430182167497`, with archive bytes `276485`.

Rate lambda: `25 / 37,545,489 = 6.658589531221714e-7` score per archive byte.
With components unchanged, reaching `<=0.31` requires at least 7726 bytes saved.
The best found semantics-preserving opportunity saves 10 bytes, reducing score
by only `0.000006658589531221714`; therefore no lossless byte-only path to
`<=0.31` was found.

| transform | byte delta vs C-102 | required total component score gain | seg-only dist reduction | pose-only posenet dist reduction | status |
|---|---:|---:|---:|---:|---|
| strip P3 header | -10 | 0.005137643 | 0.000051376 | 0.000069534 | lossless but insufficient |
| remove C-102 actions | -255 | 0.004974508 | 0.000049745 | 0.000067408 | risky, likely component harm |
| public PR75/77/79 pose transplant | +6 | 0.005148297 | 0.000051483 | 0.000069673 | risky, byte-worse |
| PR77 action transplant | +70 | 0.005190912 | 0.000051909 | 0.000070228 | risky component-affecting |
| PR79 action transplant | +907 | 0.005748236 | 0.000057482 | 0.000077447 | risky component-affecting |

## Ranking

1. Implementable lossless packer improvement: strip the C-102 `P3` header only
   if harvesting every confirmed byte matters. It is clean, verified locally,
   and semantics-preserving, but it is not a leaderboard-moving path.
2. Stop chasing ZIP/header micro-optimization beyond the `P3` header. Strict
   ZIP overhead is already minimal.
3. Stop chasing generic nested recompression for this family. Current streams
   are at or better than the tested local recompression probes.
4. If component-affecting work is allowed later, PR77 actions are the cheapest
   public action perturbation to study (+70 bytes, 147 records), but they need
   component-trace evidence before any dispatch planning.
5. PR79 actions are a much larger perturbation (+907 bytes, 672 records) and
   need a larger component win to break even; treat as scorer-basin research,
   not a packer improvement.
6. Public pose transplant is byte-worse versus C-102 and differs in 192 q0
   rows; do not treat it as a packer path.
