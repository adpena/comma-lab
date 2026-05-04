# Archive Byte Profile

- schema: `archive_byte_profile_collection_v1`
- evidence_grade: `byte_profile_only`
- score_claim: `False`
- rate formula: `25 * bytes / 37545489`
- archives: `3`
- invalid archives: `0`

This is byte attribution only. It does not inflate payloads, run scorers, dispatch jobs, promote methods, or claim contest score.

## Archives

| archive | total bytes | rate term | members | ZIP overhead est. |
|---|---:|---:|---:|---:|
| experiments/results/leaderboard_intel_20260504_codex/pr96_archive.zip | 186631 | 0.124269922 | 3 | 296 |
| experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive.pr96_drop_unused_repack.zip | 185593 | 0.123578761 | 2 | 218 |
| experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive.pr96_member_preserving_repack.zip | 185682 | 0.123638022 | 3 | 296 |

## pr96_archive.zip

- path: `experiments/results/leaderboard_intel_20260504_codex/pr96_archive.zip`
- sha256: `2ecbd2118bebdb5566f719ed538a89c4608ccab19c9edc7ae7a6de778bd42b46`
- duplicate member names: `False`
- duplicate payload hashes: `False`

### Top Contributors

| member | compressed | uncompressed | method | rate term |
|---|---:|---:|---|---:|
| decoder.bin | 169272 | 169242 | deflated | 0.112711277 |
| latents.bin | 16133 | 16920 | deflated | 0.010742302 |
| p | 930 | 930 | stored | 0.000619249 |

### Extension Totals

| group | members | compressed | uncompressed | rate term |
|---|---:|---:|---:|---:|
| .bin | 2 | 185405 | 186162 | 0.123453579 |
| (no_extension) | 1 | 930 | 930 | 0.000619249 |

## archive.pr96_drop_unused_repack.zip

- path: `experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive.pr96_drop_unused_repack.zip`
- sha256: `b1d8ee040f462d5623d77ed5941a01933c0d88c93a5f3664bf25a7363a85b779`
- duplicate member names: `False`
- duplicate payload hashes: `False`

### Top Contributors

| member | compressed | uncompressed | method | rate term |
|---|---:|---:|---|---:|
| decoder.bin | 169242 | 169242 | stored | 0.112691301 |
| latents.bin | 16133 | 16920 | deflated | 0.010742302 |

### Extension Totals

| group | members | compressed | uncompressed | rate term |
|---|---:|---:|---:|---:|
| .bin | 2 | 185375 | 186162 | 0.123433603 |

## archive.pr96_member_preserving_repack.zip

- path: `experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive.pr96_member_preserving_repack.zip`
- sha256: `615b5125314ee5c8c55f34a21766c7e6835767796ff490fd4c59a91586fb2f9b`
- duplicate member names: `False`
- duplicate payload hashes: `False`

### Top Contributors

| member | compressed | uncompressed | method | rate term |
|---|---:|---:|---|---:|
| decoder.bin | 169242 | 169242 | stored | 0.112691301 |
| latents.bin | 16133 | 16920 | deflated | 0.010742302 |
| p | 11 | 930 | deflated | 0.000007324 |

### Extension Totals

| group | members | compressed | uncompressed | rate term |
|---|---:|---:|---:|---:|
| .bin | 2 | 185375 | 186162 | 0.123433603 |
| (no_extension) | 1 | 11 | 930 | 0.000007324 |

## Cross-Archive Duplicate Payloads

| sha256 | size | archives | members |
|---|---:|---:|---|
| `940d2328ad5384c99b51872cc5f87b6153befa20c10c1c2564b08ba469c97868` | 169242 | 3 | pr96_archive.zip:decoder.bin#1, archive.pr96_drop_unused_repack.zip:decoder.bin#1, archive.pr96_member_preserving_repack.zip:decoder.bin#1 |
| `be4ed381db136baf114701fe6a9bba4f604dac18e79244c9f657383578a6ff71` | 16920 | 3 | pr96_archive.zip:latents.bin#1, archive.pr96_drop_unused_repack.zip:latents.bin#1, archive.pr96_member_preserving_repack.zip:latents.bin#1 |
| `bf04a4e2dd69ca32e3b1bd1a3c64481d7f6930096b552d49d175eec8768d1c43` | 930 | 2 | pr96_archive.zip:p#1, archive.pr96_member_preserving_repack.zip:p#1 |
