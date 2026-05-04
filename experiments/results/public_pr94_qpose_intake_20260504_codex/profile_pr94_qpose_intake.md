# PR94 Qpose Static Intake

- archive: `experiments/results/public_pr94_qpose_intake_20260504_codex/archive.zip`
- archive_bytes: `277087`
- archive_sha256: `280fc6555e684f2d5f56e0ef883a6d30d00ec4722bd1ce5d959e27525f06eac7`
- member: `p` stored=True bytes=276987
- payload_format: `pr94_fixed_range_qpose_tile_actions`
- evidence_grade: `empirical_static_archive_intake`
- score_claim: `False`

## Segments

| segment | charged bytes | charged sha256 | decoded bytes | decoded prefix |
| --- | ---: | --- | ---: | --- |
| `masks.mkv` | 219472 | `d1ae0d39c848e5715b74bb0122269066a4a1ab60ba9e12f34e70fd62ac136d87` | 223385 | `12000a0a0000000c` |
| `renderer.bin` | 55756 | `e892539adec2406f87c824accc0effc80911f160ca8d324429c5d2bac175f2cf` | 59288 | `515a533320001675` |
| `seg_tile_actions.bin` | 861 | `aa0c0971452386cf27d8d44779f2784e73f441c3c80cf480f69b35a4d15317fd` | 1033 | `5347325308100705` |
| `optimized_poses.qp1` | 898 | `7d7c35f4e7b0eb7022e56aaa76cad111b6c2e536b68080f10a536b2cb418a082` | 1140 | `5150316d1cd003c1` |

## Stackability

- `pose_side_qp1_velocity`: `blocked_not_isolated`, byte_delta `-589`. PR94 encodes only velocity col0 and relies on its own qpose runtime path; PR85/STBM uses a different pose contract, so this is not a drop-in pose-side stack without decode/reencode and runtime-output parity.
- `renderer_qzs3_model`: `blocked_coupled_model_mask_pose`, byte_delta `-1318`. The smaller QZS3 model is trained for PR94 masks/pose/actions; transplanting it onto PR85_STBM1BR/RMB1 would be a renderer replacement, not an isolated non-mask recode.
- `tile_actions_control`: `not_rate_stackable`, byte_delta `861`. Tile actions add charged bytes and only become useful if exact CUDA component gain exceeds their rate cost; PR94 only supplies MPS evidence.
- `mask_stream`: `do_not_stack_onto_stbm`, byte_delta `67033`. Replacing PR85_STBM1BR's lossless mask recode with PR94's full mask stream gives back the STBM byte win and changes the scorer-visible mask basin.
