# ddm_sw1 receipt - null-basis phase solve

Axis: `[macOS-CPU frozen-scorer advisory]`. `score_claim=false`. No n600 scorer slot, no `upstream/evaluate.py`, no archive build, no pointer move.

## Answer First

Tax recovery on the fixed tq1c parent, bounded n=4 pairs `[0, 20, 32, 48]`:

| arm | net flips fixed | eta | eta/bar | clears bar | pose ratio mean |
|---|---:|---:|---:|---|---:|
| unconstrained | 590 / 1201 | 0.4912572856 | 2.8727677377 | yes | 5.3406042088 |
| project-after Euclidean | 113 / 1201 | 0.0940882598 | 0.5502080582 | no | 0.9930848914 |
| project-after diagonal metric | 118 / 1201 | 0.0982514571 | 0.5745535475 | no | 0.9917949851 |
| solve-within null basis | 374 / 1201 | 0.3114071607 | 1.8210425998 | yes | 1.0057338219 |

`eta_bar = 0.1710048742006269`. Solve-within recovered `+0.2173189009` eta over Euclidean project-after and `+0.2131557036` eta over diagonal-metric project-after, equal to `0.5471698113` of the unconstrained-minus-Euclidean gap. This is a small-n formulation result only.

## Per-Pair Rows

| pair | label ceiling | E eta | M eta | solve-within eta | unconstrained eta | solve-within pose ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 285 | 0.1087719298 | 0.1017543860 | 0.2596491228 | 0.5368421053 | 0.9983658757 |
| 20 | 323 | 0.1176470588 | 0.1238390093 | 0.4086687307 | 0.5882352941 | 0.9999144164 |
| 32 | 302 | 0.0761589404 | 0.0794701987 | 0.2284768212 | 0.4304635762 | 0.9997451174 |
| 48 | 291 | 0.0721649485 | 0.0859106529 | 0.3402061856 | 0.4020618557 | 1.0249098780 |

## Null-Basis Certificate

The runner constructs `A` as the 6x12 per-2x2 YUV6 constraint matrix and uses `N = Vh[rank:].T` from `numpy.linalg.svd(A)`.

| field | value |
|---|---:|
| `N.shape` | `[12, 6]` |
| `rank(A)` | 6 |
| `max_abs_A_times_N` | 1.1796119636642288e-16 |
| `max_abs_NtN_minus_I` | 1.5585082678759737e-16 |

The solve-within arm optimizes block coefficients `c` with `delta = N @ c` directly against the realized segmentation objective. The current realizer still uses naive `uint8` rounding, explicitly recorded in every row; dk1 owns the lattice-native replacement.

## Artifacts

| artifact | path | sha256 |
|---|---|---|
| runner | `experiments/ddm_sw1_null_basis_phase_solve.py` | see commit manifest / serializer |
| bulk rows | `/Volumes/VertigoDataTier/pact/ddm_sw1_20260806/sw1_null_basis_rows.jsonl` | `5994edf22d5af37a5cbfe17712d4ae9ad610aacb86453120c65a9ddd00d8026a` |
| bulk summary | `/Volumes/VertigoDataTier/pact/ddm_sw1_20260806/sw1_null_basis_summary.json` | `231c11e958731450c8821d70d494ff26e7d4959276dd085ac80bd4a8330f5ac1` |
| committed summary copy | `.omx/research/ddm_sw1_20260806/sw1_null_basis_summary.json` | `231c11e958731450c8821d70d494ff26e7d4959276dd085ac80bd4a8330f5ac1` |
| seam ledger | `.omx/research/ddm_sw1_20260806/SEAM_METRIC_LEDGER.jsonl` | computed at commit time |

Parent custody: tq1c archive `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes`, sha256 `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`, bytes `357837`. Parent own-vehicle advisory row is `S=0.7534578126155775`, `d_seg=0.004305419922`, `d_pose=0.000716508925`, bytes `357837`.

## Seam Ledger Counts

`SEAM_METRIC_LEDGER.jsonl` has 16 rows:

| group | count |
|---|---:|
| `PRICE_INTERFACE_LOSS` | 7 |
| `ALREADY_JOINT` / prefix / measured-no-archive variants | 4 |
| `CONVERT_TO_SOLVE_WITHIN` / if-reopened variants | 3 |
| `HOLD_*` receiver/coder variants | 2 |

By surface class: 7 project-after seams, 5 rate/coder/receiver sites, 2 Euclidean/metric sites, 2 solved-control surfaces.

## Boundaries

- `score_claim=false`; this is not a contest score and does not move the pointer.
- No n600 scorer slot was consumed; n=4 uses the fixed et1/et2 prefix selection.
- The runner validates decoded parent argmax against `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/parent_tq1c_argmax_n600.npy` and GT argmax against `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`.
- GT frames are decoded only through the existing `decode_gt_frames` path using `frame_utils.yuv420_to_rgb`.
- Full MS4D margin-Fisher is not claimed as used; SW1 uses diagonal margin-saliency weights and ledgers the MS4D substitution.
- No candidate archive or counted `archive.zip` bytes were produced.

## Recall Evidence

- Read the charter and contract: `.omx/tmp/codex_runs/sw1_prompt.md`, `.omx/tmp/codex_runs/_common_contract.md`.
- Read governing files: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- Reused/checked implementation precedents: `experiments/train_tr1_partition_renderer_mlx.py` for the trainer null projector matrix, `experiments/ddm_sq1_pose_null_constrained_paint.py`, `experiments/ddm_sq1_eta_seg_realization.py`, `experiments/ddm_et1_ph1_block16_on_our_vehicle.py`, and the untracked et2 runner without committing it.
- Consumed prior decision receipts in bounded scope: `.omx/research/ddm_et2_metric_amendment_20260806.md`, `.omx/research/ddm_et1_eta_on_the_priced_band_20260803.md`, `.omx/research/ddm_q31_20260804/Q31_Q3_CONSTRAINED_SOLVE_RECEIPT_20260804.md`, `.omx/research/ddm_tq1_20260805/tq1c/RECEIPT.md`, `.omx/research/ddm_wf2_waterfill_reprice_20260803.md`, `.omx/research/g2f_bidirectional_amplitude_ladder_20260721T145157Z.md`, `.omx/research/g2f_bidirectional_amplitude_ladder_chart_level_20260721T153318Z.md`, `.omx/research/ddm_od3_20260805/OD3_TERMINALITY_RECEIPT.md`, `.omx/research/ddm_en1_20260805/EN1_RECEIPT.md`, and MS4/MS4D receipts.
- Searched for extra seam precedents in `.omx/research` by `Euclidean`, `metric`, `oblique`, `project`, `MS4`, `margin`, `lattice`, `round`, `snap`, `receiver`, `context`, `coder`, `waterfill`, `g2f`, `OD2`, and `OD3`. I did not find in those scoped reads a receiver-closed SW1-compatible archive/coder row; the absence is scope-bounded.

## Verdict

`READY_TO_FIRE_N32_IF_MAIN_ACCEPTS_FORM` from the generated summary is accepted only as a bounded form verdict. The next move is n32 on the same parent/protocol after lane clearance, with dk1 lattice realization treated as the first interface-risk reducer if it can run before the scorer slot opens.
