# ddm_tk1 2026-08-06 receipt

## Coded-Bytes Ladder

Scope: n600 5-class SegNet argmax label maps only. Axis is byte-only and scorer-free:
`upstream/evaluate.py` was not run and no scorer forwards were run.

| coder row | tq1c parent argmax bytes | tq1c S_rate | GT `lstars` bytes | GT S_rate | proof status |
|---|---:|---:|---:|---:|---|
| PP1 KT temporal context-arith | 142001 | 0.094552637 | 173617 | 0.115604434 | n600 closed-form KT over all cells; subset n=6 bit-exact range proof |
| bz2-9 raw uint8 | 285394 | 0.190032150 | 338593 | 0.225455181 | whole-stream decompress equality |
| LZMA1-x9e raw uint8 | 354900 | 0.236313342 | 409989 | 0.272994846 | whole-stream decompress equality |
| Brotli-q11 raw uint8 | 368760 | 0.245542148 | 424728 | 0.282808941 | whole-stream decompress equality |
| zlib-9 raw uint8 | 504452 | 0.335893881 | 581266 | 0.387041170 | whole-stream decompress equality |
| small counted learned prior, exact TK1 frame | 700111 | 0.466175178 | 713345 | 0.474987155 | whole-frame decode equality plus canonical re-encode equality |

Selected semantic stream price: **142001 B** for the tq1c shipped-vehicle
partition. GT `lstars` under the exact learned prior loses to PP1 KT by
**+539728 B**; the learned-prior family is folded at this small static
conditional-table form.

The estimate-only JSON rows for `learned_static_prior` understate the final
range stream and are not used in the ladder above. Exact learned-prior bytes are
recorded in `learned_prior_exact_addendum.json`.

## Source Proof

Parent row: tq1c `b35e756829` from the banked phase-B realized row. The argmax
source was read from the banked et2 parent-score cache only as a read-only
input:
`/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/parent_tq1c_argmax_n600.npy`.
No et2, rw1, or vo2 surface was edited.

Source digests:

| source | shape | raw sha256 | file sha256 | extra proof |
|---|---:|---|---|---|
| tq1c parent argmax | 600x384x512 uint8 | `a7dd6f4271eedfa877f6499348de5f9dae2d97311f9e98f4f534908eb66e044e` | `764a244c4890b22a67c4dbe95a959e970c29328778d41ffe4deb85f5b650eee6` | 38/38 tq1c batch `cells_sha256` checks passed |
| GT `lstars` | 600x384x512 uint8 | `f2c8be94774780bda718adf337900403a8533b6ffa1352b5aae19e200a005557` | `b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d` | matches `experiments/results/mlx_fleet_gt_cache/gt_n600.npz:lstars` raw sha |

Measured geometry:

| object | boundary px/frame mean | temporal disagreement |
|---|---:|---:|
| tq1c parent argmax | 2176.511667 | 0.011333026 |
| GT `lstars` | 2436.190000 | 0.012456377 |

## Route Table

Score formula: `100*d_seg + sqrt(10*d_pose) + 25*bytes/37545489`.
Rows below are projections except Route L, which is the live measured archive.

| route | bytes | d_seg | d_pose | projected S | status |
|---|---:|---:|---:|---:|---|
| L current latent tq1c | 357837 | 0.004305420 | 0.000716509 | 0.753457813 | measured live row |
| S semantic, flat-paint floor, current pose sections | 158670 | 0.008305000 | 0.000716509 | 1.020798695 | bad renderer endpoint |
| S semantic, PR130-class d_seg, current pose sections | 158670 | 0.000296600 | 0.000716509 | 0.219958695 | renderer-open projection |
| S semantic, PR130-class d_seg, PR130 pose, 3.3KB renderer | 168892 | 0.000296600 | 0.000023310 | 0.157385863 | renderer-open projection |
| S semantic, PR130-class d_seg, PR130 pose, 40KB renderer | 205878 | 0.000296600 | 0.000023310 | 0.182013322 | renderer-open projection |
| H semantic base plus latent residual R | 168892 + R | variable | variable | condition below | residual not priced here |

Route S arithmetic uses the measured tq1c semantic stream at 142001 B, the
banked tq1c fixed-section bracket for a 3.3KB renderer, the PR130 40252 B
renderer upper bracket, current pose sections from tq1c where stated, and PR130
pose anatomy where stated. No composed archive was built.

Hybrid crossover conditions using the 168892 B PR130-pose base:

| condition | max residual R |
|---|---:|
| beat live L at d_seg=0.0002966, d_pose=0.00002331 | 895193 B |
| beat live L at d_seg=0.0010, d_pose=0.00002331 | 789555 B |
| beat live L at d_seg=0.0010, current d_pose=0.000716509 | 685359 B |
| beat 0.19110 at d_seg=0.0002966, d_pose=0.00002331 | 50633 B |
| beat 0.19110 at d_seg=0.0005, d_pose=0.00002331 | 20086 B |
| beat 0.15 at d_seg=0.0002966, d_pose=0.00002331 | needs 11092 B less, or lower distortion |

Interpretation: the semantic stream is cheap enough to beat the current latent
row if the renderer can reach a PR130-like class-field distortion, but the
sub-0.15 target is still not crossed at the measured PR130-class/PR130-pose
projection unless bytes drop by about 11KB or d_seg improves below the PR130
anchor.

## Renderer Discriminator

Cheapest next discriminator, not fired here: **D1 semantic renderer source-forward
n600 SegNet-only closure**.

Fire order if claimed later: build RGB frames from the 142001 B semantic label
stream plus the candidate renderer, run the frozen CPU SegNet argmax only, and
measure `d_seg` against the GT `lstars` cache. This directly distinguishes the
flat-paint floor (`d_seg=0.008305`) from a PR130-like renderer cell
(`d_seg=0.0002966`) without PoseNet or `upstream/evaluate.py`. It must claim a
scorer lane before firing because this TK1 unit was scorer-free.

## Recall Evidence

Read before or during this unit: `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`,
`.omx/tmp/codex_runs/tk1_prompt.md`, `_common_contract.md`,
`.omx/state/main_hot_state.md`, `.omx/research/ddm_tq1_20260805/tq1c/RECEIPT.md`,
`.omx/research/ddm_pp1_direct_partition_pricing_20260728.md`,
`.omx/research/pr86_pr130_fullstack_intake_20260728.md`,
`.omx/research/ddm_fp1_class_field_projection_20260731.md`,
`.omx/research/ddm_pk1_20260805/PK1_RECEIPT.md`,
`.omx/research/ddm_rl1_roadlane_interface_price_20260803.md`, and
`.omx/research/ddm_hp1_20260806/RECEIPT.md`.

The memory quick pass found no TK1-specific prior entry; it only surfaced
#899/#904 review/serializer precedent, so no prior TK1 result was treated as
settled.

## Files

Primary receipt JSON: `.omx/research/ddm_tk1_20260806/semantic_stream_race.json`
with sha256 `e090ef422a0e13f434c2c8aa0f85a7a077edf3357602bd188cdda8eb9db68abc`.

Exact learned addendum: `.omx/research/ddm_tk1_20260806/learned_prior_exact_addendum.json`.

Exact learned TK1 frames:

| path | bytes | sha256 |
|---|---:|---|
| `/Volumes/VertigoDataTier/pact/ddm_tk1_20260806/tq1c_parent_argmax_learned_prior.tk1` | 700111 | `d6170de29851366bb4028acc4f30e80d7a450e44c13ed13437f6784ac974cae0` |
| `/Volumes/VertigoDataTier/pact/ddm_tk1_20260806/gt_lstars_learned_prior.tk1` | 713345 | `54a624ad55fb2b9b01d1cb0d62c7c81fb8fcf4071c87f8b8a17b686d08b9cbc2` |

