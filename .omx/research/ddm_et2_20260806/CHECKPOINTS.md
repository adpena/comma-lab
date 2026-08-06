# ddm_et2 checkpoints

All durable outputs are on SSD under `/Volumes/VertigoDataTier/pact/ddm_et2_20260806`.

## Parent decode

- Canonical inflate receipt: `parent_tq1c_decode/parent_tq1c_inflate_receipt.json`
- SHA-256: `9bc035cc42c78b338877611e96169e327375c0bffc979e989018221d103ba5ff`
- Inflated raw: `parent_tq1c_decode/submission/inflated/0.raw`
- Raw bytes: `3662409600`

## Parent score

- Aggregate: `parent_score/aggregate.json`
- SHA-256: `3b74376190d0661d87ad5f07b2bcdbb49d1302c5345bd1b0732ecddafbaf7724`
- Batch checkpoints: `parent_score/batch_0000_0016.json` through `parent_score/batch_0592_0600.json`
- Cached argmax: `parent_score/parent_tq1c_argmax_n600.npy`, `117964928` B
- Cached pose6: `parent_score/parent_tq1c_pose6_n600.npy`, `28928` B

## Phase field

- Summary: `phase_field/phase_field_rederive_summary.json`
- SHA-256: `34794a08ffdab56956f6cf3b4a1a030030c75dbbb28dd48ea3b22010c0dfc4b5`
- Current offsets: `phase_field/tq1c_block16_offsets.npy`
- Old comparator offsets: `phase_field/et1_base_block16_offsets.npy`

## Fire-order 1 Arm E

- Row JSONL: `fire_order_1_projected_rows.jsonl`
- Row JSONL SHA-256: `6b315fd0fa26f23ff8b72c077e4dd6e85ac9512ff91216adb91a810dd9dc6fbf`
- Rows: `32`
- Summary: `fire_order_1_projected_summary.json`
- Summary SHA-256: `146bf84f223f7f377d91d480b414865d41bd4bca13759a83fc74adf6adb855e9`

## Fire-order 1 Arm M

- Row JSONL: `fire_order_1_m_projected_rows.jsonl`
- Row JSONL SHA-256: `5443d9fc671420a9c18fe5b74a7b9b131a9866840980e5a5760d6ddf76f3561f`
- Rows: `32`
- Summary: `fire_order_1_m_projected_summary.json`
- Summary SHA-256: `d2bdd8d08382f855496dae6f61cdcaf75e187f7a84c96ff7df903dbb619264ac`

## Final

- Final JSON: `et2_projected_phase_field_final.json`
- Final SHA-256: `e5310cd1bb7f7f62584de342162f1d702b28f0737d3ff91e31c78c0622621929`
- Mirrored repo receipt JSON: `.omx/research/ddm_et2_20260806/et2_projected_phase_field_final.json`
- Runner SHA-256: `4f2b7e52e7461a3f438ed6b4a39f72f67c01b68892024fcccc71ac8e50445e93`

## Failed preflight replay

The first local replay attempt failed before scoring because `upstream/.venv` lacked `brotli`.

- Summary: `parent_tq1c_replay/local_submission_replay_summary.json`
- Stderr: `parent_tq1c_replay/local_replay_stderr.txt`
- Classification: environment preflight failure, not a scorer result.
