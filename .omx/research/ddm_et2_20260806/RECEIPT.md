# ddm_et2 projected phase-field receipt

Axis: `[macOS-CPU frozen-scorer advisory]`. `score_claim=false`; no contest-CPU/CUDA row and no pointer promotion.

## Fire-order-1 verdict

`FOLDED_FORMULATION_PROJECTED_STATIC_ETA_NOT_ABOVE_BAR`.

Amendment 1 A/B completed on the re-solved tq1c block16 phase field:

| arm | projector | n | eta | bar | eta / bar | net flips | pose min / median / max | pose pass |
|---|---|---:|---:|---:|---:|---:|---|---|
| E | Euclidean rank-6 Q3 | 32 | 0.040337200870195794 | 0.1710048742006269 | 0.23588333992673946 | 445 / 11,032 | 0.9616827821 / 0.9999387225 / 1.0356058119 | PASS |
| M | diagonal seg-metric Q3 | 32 | 0.04396301667875272 | 0.1710048742006269 | 0.2570863367740868 | 485 / 11,032 | 0.9175391542 / 1.0010000374 / 1.3898756802 | FAIL |

Winner by the pre-registered selection rule is **Arm E**: Arm M recovered 40 more flips on the subset but failed the 1.04 pose-neutrality threshold, and both arms retained far too little eta to clear the priced phase-field bar. This folds the projected-static rank-6 Q3 formulation. It does not kill the phase-field family or a solve-inside-Q3/constrained-descent formulation.

Fire-order 2 was **not fired** because no arm was green. No composed candidate archive was produced.

## Amendment-1 metric notes

The located per-pixel gradient cache, `.omx/research/uniward_per_pixel_n_plus_1_artifacts_20260526/real_scorer_gradients_cache.npz`, is real but covers only the first 50 pairs, while the et1 n32 pairset contains many pair ids above 50. It was therefore not used as a hidden metric source.

Arm M used an on-the-fly current-pair SegNet CE gradient on the same phase-field target/band, with diagonal `M = grad_abs^2 + lambda`. Lambda used the recorded spectrum rule with floor `1e-12`; in this n32 run the floor bound was active. Algebraic checks across projected blocks:

- max `|A P_M|`: `6.467049118441537e-15`
- max `|P_M^2 - P_M|`: `9.992007221626409e-15`

Fallback per-pair joint acceptance was kept separate and not blended: Arm E accepted `32/32`; Arm M accepted `30/32` and rejected pairs `32` and `533`.

## Freshness gate

Verified the chartered tq1c parent archive:

- Archive: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes`
- SHA-256: `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`
- Bytes: `357837`

Canonical inflate through shipped `inflate.sh` succeeded with the repo venv on PATH:

- Receipt: `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_tq1c_decode/parent_tq1c_inflate_receipt.json`
- SHA-256: `9bc035cc42c78b338877611e96169e327375c0bffc979e989018221d103ba5ff`
- Raw: `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_tq1c_decode/submission/inflated/0.raw`, `3662409600` B

The parent scorer remeasure matched the named neighborhood:

- Measured: `S=0.753457812580`, `d_seg=0.004305419922`, `d_pose=0.000716508925`, bytes `357837`
- Delta vs charter: `delta_S=-3.601263731667359e-11`, `delta_d_seg=-1.2499983687019878e-13`, `delta_d_pose=-3.9805561190958683e-13`, `delta_bytes=0`

## Phase-field rederive

The block16 field was re-solved on tq1c, not reused from 2026-08-03:

| field | gross S | projected bytes | bar | reach |
|---|---:|---:|---:|---:|
| tq1c parent | 0.1799714830186632 | 46220 | 0.1710048742006269 | 0.4180114513436033 |
| et1 2026-08-03 base | 0.1803885565863715 | 46247 | 0.17070916020272411 | 0.41836072664359863 |

Field delta vs 2026-08-03:

- Parent argmax changed cells: `2746`
- Offset blocks changed: `1268 / 460800`
- Translated target changed cells: `5043`
- Net target-correct delta vs old: `+260`

## Artifacts

- Final JSON: `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/et2_projected_phase_field_final.json`, sha256 `e5310cd1bb7f7f62584de342162f1d702b28f0737d3ff91e31c78c0622621929`
- Arm E summary: `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/fire_order_1_projected_summary.json`, sha256 `146bf84f223f7f377d91d480b414865d41bd4bca13759a83fc74adf6adb855e9`
- Arm E rows: `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/fire_order_1_projected_rows.jsonl`, sha256 `6b315fd0fa26f23ff8b72c077e4dd6e85ac9512ff91216adb91a810dd9dc6fbf`
- Arm M summary: `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/fire_order_1_m_projected_summary.json`, sha256 `d2bdd8d08382f855496dae6f61cdcaf75e187f7a84c96ff7df903dbb619264ac`
- Arm M rows: `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/fire_order_1_m_projected_rows.jsonl`, sha256 `5443d9fc671420a9c18fe5b74a7b9b131a9866840980e5a5760d6ddf76f3561f`
- Phase-field summary: `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/phase_field/phase_field_rederive_summary.json`, sha256 `34794a08ffdab56956f6cf3b4a1a030030c75dbbb28dd48ea3b22010c0dfc4b5`
- Parent score: `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/aggregate.json`, sha256 `3b74376190d0661d87ad5f07b2bcdbb49d1302c5345bd1b0732ecddafbaf7724`
- Runner: `experiments/ddm_et2_projected_phase_field.py`, sha256 `4f2b7e52e7461a3f438ed6b4a39f72f67c01b68892024fcccc71ac8e50445e93`

## Recall Evidence

Sources searched/read: `.omx/tmp/codex_runs/et2_prompt.md`, `.omx/tmp/codex_runs/_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, `.omx/research/ddm_et1_eta_on_the_priced_band_20260803.md`, `.omx/research/ddm_et2_metric_amendment_20260806.md`, `.omx/research/ddm_ffm1_20260806/RECEIPT.md`, `.omx/research/ddm_ffm1_20260806/PREDICTIONS.md`, `.omx/research/ddm_tq1_20260805/tq1c/RECEIPT.md`, `.omx/research/ddm_tq1_20260805/tq1c/NEXT_IF_RESUMED.md`, `.omx/research/ddm_q31_20260804/q31_summary.json`, `.omx/research/ddm_sq1_eta_seg_and_hinge_ab_20260803.md`, `.omx/research/ddm_ph1_phase_mass_reach_ceiling_20260803.md`, canonical equations registry command (`.venv/bin/python tools/list_canonical_equations.py --json`, 424 entries, transient output sha256 `73452be589e6d8429e427418b8ca34801758d417fd792b768a9fca43b845eaac`), and targeted `rg` queries for `ddm_et1`, `ddm_et2`, `phase-field`, `pose-null`, `rank-6`, `Q3`, `block16`, `tq1c`, `q43a`, `UF1`, `metric`, `Fisher`, `saliency`, `ms3`, `ms4`, and `ffm1`.

Beyond-charter findings that changed the plan:

- Amendment 1 required Arm E and Arm M; the Euclidean-only result was not sufficient.
- `q31` is a Q3-first Road/Lane target solver on qo1, not the phase-field projected-static gate, so et2 needed a narrow runner rather than reusing q31 directly.
- A direct `tools/run_local_submission_replay.py` freshness run failed before scoring because the upstream venv lacked `brotli`; the shipped `inflate.sh` was kept unchanged and `tac.submission_chain.run_inflate` was used with the repo venv absolute python path to perform the canonical decode.
- The tq1c receipt confirmed the fresh parent path and that `tq1c_base/archive.zip` is the stale 75df9cc3 decoy, so all measurement used the b35e7568 receipt-bytes archive.
- The only located real per-pixel gradient cache was a 50-pair prefix cache, so Arm M measured current-pair saliency instead of applying a partial-population cache to the n32 gate.

Own-vehicle advisory frontier remains: `S = 0.7534578126155775 @ 357837 B [macOS-CPU advisory]`. Contest pointer remains borrowed/unmoved: `S = 0.1910828242`.
