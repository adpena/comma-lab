# PR97 H3 Static Intake

- Evidence grade: `external_archive_byte_intake_only_until_exact_cuda_replay`
- Archive: `experiments/results/leaderboard_intel_20260504_codex/pr97_archive.zip`
- Archive bytes/SHA: `197160` / `6785a84879d3e3395bbf990b980fe32182fca7255c5b8559dcdaac9da7516642`
- ZIP overhead: `100` bytes
- Payload `p`: `197060` bytes / `3ff68e3b463498a6e054cd69ff286e5098333a24c4e187c70acd585af794a32a`

## Payload Split

| part | bytes | sha256 |
|---|---:|---|
| mask | 135120 | `bdf2f7a54a245e8a11d5b1ca74e7a7099c4754b3f5f8faa57e71418208031642` |
| pose | 2310 | `c7a62b5a8feaa43ad2aa42dafec98774adea78a82c31cebb9530326302993f54` |
| model | 57238 | `0ac6ff812fe3cda523960d62ddca11e96703deb44b3c212110b16923322fc472` |
| sidecar | 2376 | `b8bb46e5f4e5966489ff7ea12f2b814c78548c98c886fab25818b8c71ad93045` |

## Parsed Subformats

- Mask: `22` chunks, `135120` bytes.
- Pose: bits per dim `[14, 4, 4, 4, 4, 4]`, raw `2612` bytes.
- Model: `102` schema entries, raw `61164` bytes.
- Sidecar: `455` pair records, raw `3482` bytes.

## Byte Candidates

| label | archive bytes | delta | runtime change |
|---|---:|---:|---|
| pr97_deflated_p | 196954 | -206 | False |
| pr97_pose_model_br10_deflated_p | 196914 | -246 | False |
| pr97_pose_model_br10_sidecar_br11_deflated_p_runtime_patch | 196803 | -357 | True |

## Risks

- inflate.sh attempts pip install brotli if missing; exact replay should pin dependency closure
- inflate.py compiles range_mask_codec.cpp at inflate time and requires c++/g++/clang++
