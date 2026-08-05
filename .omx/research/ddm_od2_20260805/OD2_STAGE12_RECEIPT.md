# OD2 Stage 1+2 staged-composition receipt - 2026-08-05

Status: `MEASURED_N32_ADVISORY / STAGE2_PASS / STAGE1_TERMINALITY_NOT_CLOSED`.

Axis: `[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE`.
`score_claim=false`, `promotion_eligible=false`, `full_n600_scorer_job=false`.

## Answer First

OD2 measured the staged route on a seeded stratified-random n32 set:

| quantity | measured |
|---|---:|
| selected pairs | 32/600, seed `20260805`, 10 temporal blocks |
| seg ratio check | `1.0099888594483923x` population, `MATCHED` |
| pose ratio check | `0.42628664334579025x` population, `MATCHED` within the random null band |
| Stage-1 d_seg before -> after | `0.004331270853678386 -> 0.003349463144938151` |
| Stage-1 pooled eta | `0.554438560272866` |
| Stage-1 stop census | 29/32 `iteration_cap_best_at_cap`, 2/32 `converged_projected`, 1/32 `marginal_below_bar` |
| Stage-2 k=4 frame_0 carriage | 96 B/pair, projected 57,600 B n600, rate cost `0.03835347569983707` S |
| Stage-2 seg preservation | 32/32 exact |
| d_pose before -> Stage 1 -> Stage 2 k4 | `0.0008014285623403339 -> 0.0058411338650330435 -> 0.0007588698333620414` |
| staged subset projection vs current own vehicle | delta S `-0.062236702464336054`, projected `S = 0.6917440272267846` |

Interpretation: the k=4 frame_0 carriage passes the Stage-2 same-row gate. It repairs pose below the same-row baseline while retaining Stage-1 eta exactly. The Stage-1 cap-bound blocker is not closed: 29/32 rows still selected their best iterate at the safety cap, and 25/32 trajectory decisions report `safety_bound_REPORTED`.

## Commands

Selection:

```bash
.venv/bin/python - <<'PY'
from tac.subset_selection import MODE_STRATIFIED, select
# seed=20260805, n=32, population=600, block_count=10
PY
```

Smoke:

```bash
.venv/bin/python experiments/ddm_js1_staging_discriminator.py \
  --sub-dir /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2 \
  --gt-mkv upstream/videos/0.mkv \
  --pairs-npy .omx/research/ddm_od2_20260805/od2_pairs_n32_seed20260805.npy \
  --argmax-cache /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache \
  --out .omx/research/ddm_od2_20260805/od2_js1_smoke_pair8.json \
  --only-pairs 8 --block 16 --rmax 5 --seg-steps 5 --pose-steps 5 \
  --eval-every 5 --arms cheapdct --dct-k 4 --threads 4 --resume
```

Full n32 measurement:

```bash
.venv/bin/python experiments/ddm_js1_staging_discriminator.py \
  --sub-dir /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2 \
  --gt-mkv upstream/videos/0.mkv \
  --pairs-npy .omx/research/ddm_od2_20260805/od2_pairs_n32_seed20260805.npy \
  --argmax-cache /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache \
  --out .omx/research/ddm_od2_20260805/od2_js1_n32_cprime_k4.json \
  --block 16 --rmax 5 --seg-steps 25 --pose-steps 40 \
  --eval-every 5 --arms cheapdct --dct-k 4 --threads 4 --resume
```

The first attempt included the optional `poseonly` control and was interrupted after two durable rows to remove that non-charter overhead. Resume preserved those two rows. The second process reported `DONE 32 t=3738.4s`; total observed pair-compute including the first two rows was approximately 4085.4s.

## Artifacts

| path | bytes | sha256 |
|---|---:|---|
| `.omx/research/ddm_od2_20260805/PAIR_SELECTION.json` | 2,388 | `0a8ac26a1cd39c7dc425dbb4922d0dda6f71227b205241d3d771ea9791c2d4f9` |
| `.omx/research/ddm_od2_20260805/od2_pairs_n32_seed20260805.npy` | 384 | `8b4aa8d47787757ca9a29cb1d176670ad2f39c15b7daf97f25615006c98a3f94` |
| `.omx/research/ddm_od2_20260805/od2_js1_smoke_pair8.json` | 2,396 | `b4d65ff569c9e9dc68782ae80add85cd2835b2a4559999b778718da0bb529292` |
| `.omx/research/ddm_od2_20260805/od2_js1_n32_cprime_k4.json` | 103,690 | `fd1016751e4668ff786692f52f91d924be97081a70a20d11e470150aaf85c6af` |
| `.omx/research/ddm_od2_20260805/OD2_AGGREGATE.json` | 31,831 | `43c97e844c23c00b5ad7367e147735587e00dec21b2f274ebfef7770b32a3ace` |
| `.omx/research/ddm_od2_20260805/od2_js1_aggregate_legacy_constants.json` | 105,069 | `55558b2f9e0681120513d782353683cf396feb7451b1b1960f4bc50b20469af9` |

The `.npy` pair file is a gitignored convenience input, not a committed source
of truth. The committed authority for the selection is `PAIR_SELECTION.json`,
including the ordered pair list and seed.

## Boundaries

- No full-n600 scorer job was run.
- No contest-CPU or contest-CUDA authority row was run.
- No receiver-closed archive was produced.
- The staged `S = 0.6917440272267846` is a subset advisory projection, not a pointer move.
- Stage 1 remains cap-bound on this formulation and safety cap. The OD1 blocker `OD1_BLOCKER_SEG_BASE_CAP_BOUND` is not closed.
- Stage 2 passes on these same rows: k=4 frame_0 DCT carriage preserved seg on 32/32 rows and reduced mean d_pose versus both Stage 1 and the same-row baseline.

Own-vehicle frontier line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
