# Archive Byte Profile

- schema: `archive_byte_profile_collection_v1`
- evidence_grade: `byte_profile_only`
- score_claim: `False`
- rate formula: `25 * bytes / 37545489`
- archives: `4`
- invalid archives: `1`

This is byte attribution only. It does not inflate payloads, run scorers, dispatch jobs, promote methods, or claim contest score.

## Archives

| archive | total bytes | rate term | members | ZIP overhead est. |
|---|---:|---:|---:|---:|
| reports/raw/leaderboard_intel_20260501/pr65_archive.zip | 284425 | 0.189386933 | 1 | 100 |
| reports/raw/leaderboard_intel_20260501/pr67_archive.zip | 276564 | 0.184152616 | 1 | 100 |
| reports/raw/leaderboard_intel_20260502/pr70_archive.zip | 57329 | 0.038173028 | 0 |  |
| experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip | 276214 | 0.183919565 | 1 | 100 |

## Invalid Archives

| archive | bytes | error |
|---|---:|---|
| reports/raw/leaderboard_intel_20260502/pr70_archive.zip | 57329 | ArchiveByteProfileError: bad ZIP archive: reports/raw/leaderboard_intel_20260502/pr70_archive.zip |

## pr65_archive.zip

- path: `reports/raw/leaderboard_intel_20260501/pr65_archive.zip`
- sha256: `b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68`
- duplicate member names: `False`
- duplicate payload hashes: `False`

### Top Contributors

| member | compressed | uncompressed | method | rate term |
|---|---:|---:|---|---:|
| x | 284325 | 284325 | stored | 0.189320347 |

### Extension Totals

| group | members | compressed | uncompressed | rate term |
|---|---:|---:|---:|---:|
| (no_extension) | 1 | 284325 | 284325 | 0.189320347 |

## pr67_archive.zip

- path: `reports/raw/leaderboard_intel_20260501/pr67_archive.zip`
- sha256: `a5ed8da0d9988943c986b231b4cd33cea0ab878a8e1628134341db5f7f41c765`
- duplicate member names: `False`
- duplicate payload hashes: `False`

### Top Contributors

| member | compressed | uncompressed | method | rate term |
|---|---:|---:|---|---:|
| p | 276464 | 276464 | stored | 0.184086030 |

### Extension Totals

| group | members | compressed | uncompressed | rate term |
|---|---:|---:|---:|---:|
| (no_extension) | 1 | 276464 | 276464 | 0.184086030 |

## archive.zip

- path: `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`
- sha256: `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- duplicate member names: `False`
- duplicate payload hashes: `False`

### Top Contributors

| member | compressed | uncompressed | method | rate term |
|---|---:|---:|---|---:|
| p | 276114 | 276114 | stored | 0.183852979 |

### Extension Totals

| group | members | compressed | uncompressed | rate term |
|---|---:|---:|---:|---:|
| (no_extension) | 1 | 276114 | 276114 | 0.183852979 |
