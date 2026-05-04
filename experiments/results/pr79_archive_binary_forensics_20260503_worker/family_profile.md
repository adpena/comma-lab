# PR79 Archive Binary Forensics Worker

Local byte/archive reverse-engineering only. No GPU dispatch and no score claim.

## Archive And ZIP Overhead

| archive | archive bytes | payload bytes | zip overhead | payload format | header bytes | sha256 |
|---|---:|---:|---:|---|---:|---|
| c102 | 276485 | 276385 | 100 | public_pr75_qzs3_qp1_segactions_p3 | 10 | `79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8` |
| pr75 | 276481 | 276381 | 100 | public_pr75_qzs3_qp1_segactions_fixed_slices | 0 | `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746` |
| pr77 | 276551 | 276451 | 100 | public_pr75_qzs3_qp1_segactions_fixed_slices | 0 | `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af` |
| pr79 | 277388 | 277288 | 100 | public_pr75_qzs3_qp1_segactions_fixed_slices | 0 | `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446` |

## Charged Slice Layout

| archive | stream | offset | charged bytes | decoded bytes | charged sha equal group | decoded sha equal group |
|---|---:|---:|---:|---:|---|---|
| c102 | masks.mkv | 10 | 219472 | 223385 | `d1ae0d39c848` | `a5c2b89c110d` |
| c102 | renderer.bin | 219482 | 55756 | 59288 | `e892539adec2` | `30159b6ace27` |
| c102 | seg_tile_actions.bin | 275238 | 255 | 281 | `e2f8a113ee9e` | `b0a4d55d19a5` |
| c102 | optimized_poses.qp1 | 275493 | 892 | 1140 | `c4c6e22ef870` | `80094e20f4f6` |
| pr75 | masks.mkv | 0 | 219472 | 223385 | `d1ae0d39c848` | `a5c2b89c110d` |
| pr75 | renderer.bin | 219472 | 55756 | 59288 | `e892539adec2` | `30159b6ace27` |
| pr75 | seg_tile_actions.bin | 275228 | 255 | 281 | `e2f8a113ee9e` | `b0a4d55d19a5` |
| pr75 | optimized_poses.qp1 | 275483 | 898 | 1140 | `7d7c35f4e7b0` | `8d77f6a39d1a` |
| pr77 | masks.mkv | 0 | 219472 | 223385 | `d1ae0d39c848` | `a5c2b89c110d` |
| pr77 | renderer.bin | 219472 | 55756 | 59288 | `e892539adec2` | `30159b6ace27` |
| pr77 | seg_tile_actions.bin | 275228 | 325 | 371 | `d8c75e4f3725` | `f79727da7664` |
| pr77 | optimized_poses.qp1 | 275553 | 898 | 1140 | `7d7c35f4e7b0` | `8d77f6a39d1a` |
| pr79 | masks.mkv | 0 | 219472 | 223385 | `d1ae0d39c848` | `a5c2b89c110d` |
| pr79 | renderer.bin | 219472 | 55756 | 59288 | `e892539adec2` | `30159b6ace27` |
| pr79 | seg_tile_actions.bin | 275228 | 1162 | 1451 | `3fdf6320b776` | `700d1ba2e11c` |
| pr79 | optimized_poses.qp1 | 276390 | 898 | 1140 | `7d7c35f4e7b0` | `8d77f6a39d1a` |

## Action Records

| archive | wire kind | records | runtime bytes | unique pairs | unique tiles | unique actions |
|---|---|---:|---:|---:|---:|---:|
| c102 | SG2_grouped_tile_frame_delta_varint | 108 | 432 | 106 | 21 | 60 |
| pr75 | SG2_grouped_tile_frame_delta_varint | 108 | 432 | 106 | 21 | 60 |
| pr77 | SG2_grouped_tile_frame_delta_varint | 147 | 588 | 121 | 24 | 73 |
| pr79 | SG2_grouped_tile_frame_delta_varint | 672 | 2688 | 295 | 45 | 99 |

## QP1 Pose Diff

| archive | raw bytes | q0 row diffs vs C-102 | byte diffs vs C-102 | sha256 |
|---|---:|---:|---:|---|
| c102 | 1140 | 0 | 0 | `80094e20f4f6cd29869e043eb6d224a6697ecd7fb0e77728ddcd6c7a05fccb9a` |
| pr75 | 1140 | 192 | 185 | `8d77f6a39d1a84eca78fbe8fa5ddc31f6ada29814b49e32daeb800ea84a015cc` |
| pr77 | 1140 | 192 | 185 | `8d77f6a39d1a84eca78fbe8fa5ddc31f6ada29814b49e32daeb800ea84a015cc` |
| pr79 | 1140 | 192 | 185 | `8d77f6a39d1a84eca78fbe8fa5ddc31f6ada29814b49e32daeb800ea84a015cc` |

## Break-Even To 0.31

C-102 canonical score input: 0.315144301821675; gap to 0.31: 0.005144301821675. With components unchanged, required archive-byte savings: 7726 bytes.

## Lossless/Semantics-Preserving Opportunities

| rank | opportunity | byte savings | new archive bytes | target status |
|---:|---|---:|---:|---|
| 1 | strip_c102_p3_header_use_existing_fixed_slice_parser | 10 | 276475 | insufficient_alone |

## Risky Component-Affecting Transforms

| transform | byte delta | required score gain | seg-only dist reduction | pose-only dist reduction |
|---|---:|---:|---:|---:|
| remove_c102_seg_tile_actions_stream | -255 | 0.004974508 | 0.000049745 | 0.000067408 |
| replace_c102_qp1_pose_with_public_pr75_pr77_pr79_pose | 6 | 0.005148297 | 0.000051483 | 0.000069673 |
| transplant_pr77_action_records | 70 | 0.005190912 | 0.000051909 | 0.000070228 |
| transplant_pr79_action_records | 907 | 0.005748236 | 0.000057482 | 0.000077447 |

CSV companions: `action_record_multiset_union_diff.csv`, `pose_qp1_q0_word_diff.csv`, `pose_qp1_byte_diff_vs_c102.csv`, and `stream_identity_matrix.csv`.
