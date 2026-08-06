# ddm_tk2 D1 Harness Receipt

## Summary

Axis: `[macOS-CPU frozen-SegNet harness-smoke]`

`score_claim=false`. This is a D1 harness/candidate smoke, not a contest score and not a frontier move.

R path: `torch_cpu_bicubic_to_camera_uint8_then_segnet_preprocess_bilinear`.

| candidate | n | d_seg | errors/sites | wall_s | status |
|---|---:|---:|---:|---:|---|
| c0_flat_paint | 4 | 0.008454641 | 6649/786432 | 1.18 | measured_harness_smoke |
| c1_v15_template_paint | 4 | 0.010585785 | 8325/786432 | 1.27 | measured_harness_smoke |
| c2_boundary_aa | 4 | 0.010674795 | 8395/786432 | 1.25 | measured_harness_smoke |
| c3_tr1_onehot_retarget | 0 | NA | NA | NA | blocked_fail_closed |

## Artifact Boundary

tk1 selected a `142001` byte KT/context-arith semantic stream by receipt, but did not persist a standalone full stream file. This runner verified the persisted tq1c source-label `.npy` file and GT `.npy` file by SHA-256, then ran a real subset coder round-trip before rendering.

- tq1c labels: `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/parent_tq1c_argmax_n600.npy`
- tq1c file sha256: `764a244c4890b22a67c4dbe95a959e970c29328778d41ffe4deb85f5b650eee6` (expected match: `True`)
- GT labels: `/Volumes/VertigoDataTier/pact/ddm_ph1_lstars_u8.npy`
- GT file sha256: `b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d` (expected match: `True`)
- subset coder proof: `{"bit_exact_roundtrip": true, "closed_form_bytes_laplace": 2125.798826338975, "coded_bytes": 2126, "coded_over_closed": 1.0000946343833352, "subset_frames": 4}`

## Candidate Ladder

- `c0_flat_paint`: C0 flat-paint prototype colors. Provenance: fp1 measured palette; positive control target is the tk1-cited flat-paint floor near d_seg 0.008305 on n600 when driven from GT argmax
- `c1_v15_template_paint`: C1 v14/v15 analytic margin paint. Provenance: fp1 palette plus v15 lane-boundary scorer-solved template color (51,255,204) and v14 movable prototype (107,0,114); no training
- `c2_boundary_aa`: C2 C1 plus boundary coverage treatment. Provenance: #149 survival-wall lesson: all-class boundary band dominates through-R errors; this applies a deterministic 4-neighbor coverage blend before R
- `c3_tr1_onehot_retarget`: C3 TR1 one-hot retarget probe. Provenance: banked LOTTO checkpoint candidate /Volumes/VertigoDataTier/pact/ddm_tb1_20260728/t2_n600_lotto/checkpoints/stage_seg_trunk_tau_final.npz; fail-closed until a real one-hot adapter/checkpoint shape is declared

`c3_tr1_onehot_retarget` is fail-closed here. The banked TR1 LOTTO checkpoint consumes its own token/code surface; no compatible one-hot class-plane adapter was found or trained in this arm.

## D1 Fire Order Packet

Claim the scorer slot first. Run one candidate at a time, chunked at 120 with `--resume`; do not run this while et/rw scorer work owns the slot.

```bash
.venv/bin/python experiments/ddm_tk2_d1_runner.py --pair-start 0 --pair-count 600 --chunk-size 120 --resume --claim-scorer-slot --candidates c0_flat_paint
.venv/bin/python experiments/ddm_tk2_d1_runner.py --pair-start 0 --pair-count 600 --chunk-size 120 --resume --claim-scorer-slot --candidates c1_v15_template_paint
.venv/bin/python experiments/ddm_tk2_d1_runner.py --pair-start 0 --pair-count 600 --chunk-size 120 --resume --claim-scorer-slot --candidates c2_boundary_aa
# C3 is intentionally not fireable until a real one-hot TR1 adapter/checkpoint shape is declared.
```

Expected wall clock from smoke scales roughly linearly from the observed per-candidate smoke wall seconds in `harness_smoke.json`; the exact n600 fire must record actual wall clock per chunk.

## RECALL EVIDENCE

- Read `.omx/tmp/codex_runs/tk2_prompt.md` and `_common_contract.md`.
- Read `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, and `.omx/state/main_hot_state.md`.
- Read tk1 receipts under `.omx/research/ddm_tk1_20260806/` and confirmed the selected 142,001-byte stream is a receipt-backed closed-form coder row, not a persisted full stream file.
- Searched/read through-R and frozen SegNet harness code in `experiments/train_witness_realized_through_R_mlx.py` and `src/tac/boundary_math/seg_core.py`.
- Read fp1/v14/v15/#149/TR1 provenance surfaces for the candidate ladder; C3 remained scoped negative because no real one-hot TR1 adapter surface was found in this arm.

## Frontier Honesty

Own-vehicle frontier remains the current pointer from `.omx/state/main_hot_state.md`: `S=0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest borrowed pointer remains unmoved. This tk2 artifact is means, not goal progress.
