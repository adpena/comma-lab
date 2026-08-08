# DDM PK2 Candidate-A PR130 CPR1 Surface Fit Findings

Tags: [no-triality] [p0-ledger-ok]

## VERDICT

`score_claim=false`. Axis: `[macOS-CPU advisory]`.

Candidate A is a scoped negative on the measured instance: PR130-style neutral-gray CPR1 carrier fit on the ep854 `cell_drop50` frame1 surface, seeded stratified `n=120`, CPU-torch scorers.

The byte-closed carrier improved from the cold PR130-surface transfer scale, but did not enter the pose tube:

| item | value |
|---|---:|
| selected rows | 120 / 600 |
| selection seed | 20260728 |
| selection mode | stratified blocks, governing `pose_target_center_energy` |
| governing ratio | 1.1959591357433967, MATCHED |
| best quantized step | 750 |
| measured `d_pose` | 0.06136142462491989 |
| pose tube | 0.0025 |
| PR130 external reference `d_pose` | 0.00002331 |
| rows under tube | 17 / 120 |
| measured `d_seg` | 0.003967624623328447 |
| frame1 byte identity | 120 / 120 |
| CPR1 carrier bytes | 23672 |
| composed sample `S` | 1.385129085660972 |

Pose status: FAIL. The final `d_pose` is about 24.54x above the 0.0025 tube and about 2632.41x above the PR130 external reference. This does not compose into the banked ep854 x `cell_drop50` surface as a frontier move.

Seg collateral: the carrier writes frame0/slave only, and the candidate frame1 was byte-identical to the ep854 master for all 120 selected rows. Since the official SegNet scorer uses only the last frame, the measured carrier-induced d_seg collateral on this sample is zero. The sample d_seg value, 0.003967624623328447, should be read as the selected-sample ep854 surface d_seg against GT, not as a n600 replacement for the ep854 base d_seg 0.003943024 from #827.

## ARTIFACTS

SSD output root: `/Volumes/VertigoDataTier/pact/ddm_pk2_20260808`

| artifact | bytes | sha256 |
|---|---:|---|
| `pk2_result.json` | 17527 | `1271623e51b13fa721030c522c27b680a7f3cbde1da581e882a6bc6d85459ee6` |
| `receipts.jsonl` | 15519; 43 lines | `b6aafe846bc8147e4a6bdf0393ad6005c7c6f9c633f1664e41340a5b12301b93` |
| `pk2_fitted_cpr1.carrier` | 23672 | `ba5df6ef3fe1f311282c6a5baa3c92b2281331f63f28ff149b6b4230c63caed1` |

Checkpoints were written under `/Volumes/VertigoDataTier/pact/ddm_pk2_20260808/checkpoints/`, with stage-safe best checkpoints every quantized improvement through `pk2_best_step00750.pt` and resumable `pk2_latest.pt`.

Reviewed runner: `tools/run_ddm_pk2_pr130_surface_fit.py`.

Executed command, before moving the runner into the tracked `tools/` review surface:

```bash
.venv/bin/python .omx/research/ddm_pk2_20260808/run_pk2_pr130_surface_fit.py --out-dir /Volumes/VertigoDataTier/pact/ddm_pk2_20260808 --n 120 --seed 20260728 --steps 750 --stop-after-step 750 --batch-size 12 --eval-batch-size 12 --eval-every 75 --log-every 25 --torch-threads 8 --resume
```

Equivalent rerun command after review-surface move:

```bash
.venv/bin/python tools/run_ddm_pk2_pr130_surface_fit.py --out-dir /Volumes/VertigoDataTier/pact/ddm_pk2_20260808 --n 120 --seed 20260728 --steps 750 --stop-after-step 750 --batch-size 12 --eval-batch-size 12 --eval-every 75 --log-every 25 --torch-threads 8 --resume
```

## RECALL EVIDENCE

Consulted before and during execution:

- `.omx/research/ddm_pk2_20260808/CHARTER.md`: Candidate-A decisive surface-fit charter; candidate B already folded negative; n>=120 seeded stratified requirement; PR130 CPR1-style recipe; CPU-only axis; required artifacts.
- `.omx/tmp/codex_runs/_common_contract.md`: serializer, review-gate, storage, GT decode, scorer authority, no-prefix, and reporting constraints.
- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`: governing no-fake, frontier, operating, and pointer state.
- `.omx/research/ddm_pk2_20260808/PK1_FINDINGS.md`: PR130 CPR1 custody smoke, candidate-B terminal GN folded negative, and required candidate-A continuation.
- `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/README.md` and `recipe/TRAINING.md`: PR130 release custody, raw-video E2E recipe, carrier stage groups, and CPR1 serialization contract.
- `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/carrier_codec.py`, `learned_pose_carrier_oracle.py`, `pose_basis_oracle.py`, `train_pose_carrier_full.py`, `repack_carrier.py`, `inflate.py`: PR130 low-rank neutral-gray carrier and CPR1 codec implementation.
- `.omx/research/ddm_cr1_composition_row_827_20260801.md` and `.omx/research/ddm_cr2_composition_row_ep854_base_20260801.md`: ep854 bytes/d_seg composition arithmetic and #827 base context. These protected files were read-only.
- `.omx/research/ddm_cr2r_ep854_pose_resolve_refuted_matched_control_20260802.md`: prior post-hoc ep854 pose re-solve refutation; not treated as a kill for PR130-style standalone carrier.

## METHOD

The runner keeps PR130's neutral-gray low-rank carrier shape: 12 coefficients per pair, 12x3x24x32 basis fields, 5-bit basis quantization, 12-bit coefficients, and CPR1 compact carrier parse-back. It starts from PR130's `archive_carrier_int6_coefftail_s4k.pt` checkpoint, then fits the selected coefficient rows plus basis against ep854 frame1 masters and GT PoseNet-6 targets on CPU.

GT frame decode used `upstream/frame_utils.yuv420_to_rgb` from `upstream/videos/0.mkv`. Scoring used CPU `upstream/modules.py` PoseNet and SegNet with `upstream/models/*.safetensors`. MPS was not used. `upstream/` was not modified.

The local `score_selected` pass uses the official SegNet last-frame behavior and verified frame1 byte identity before reporting d_seg.

## BORROWED SUBSTRATE ACCOUNTING

Borrowed from PR130:

- neutral-gray low-rank semantic-pose carrier;
- 12D coefficient shape and 12x3x24x32 basis shape;
- 5-bit basis / 12-bit coefficient CPR1 carrier codec shape;
- PR130 carrier init checkpoint and pose-carrier training code structure.

New in PK2:

- ep854 frame1-master fit surface;
- canonical seeded stratified `n=120` selection with governing-ratio receipt;
- CPU-torch scorer pass on the same rows;
- CPR1 parse-back byte-close receipt for the fitted ep854 carrier;
- explicit composed-sample arithmetic against the banked ep854 x `cell_drop50` surface.

Mechanism deltas vs full PR130 recipe: scope-reduced CPU fit, no CUDA raw-video 49-stage E2E rerun, no n600 final scoring, and no contest archive submission. This is a toy-bracket/scoped-instance measurement, not full PR130-family promotion or rejection.

## NOT MEASURED

- No n600 score.
- No contest-CPU or contest-CUDA authority replay.
- No archive.zip exact score.
- No full PR130 raw-video E2E reproduction.
- No claim that PR130 candidate A is globally dead; this verdict is scoped to the selected n120 ep854-surface CPU recipe above.

## FRONTIER STATUS

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]` from `.omx/state/main_hot_state.md`.
