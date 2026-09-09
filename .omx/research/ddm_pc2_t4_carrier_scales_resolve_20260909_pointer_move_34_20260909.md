# THIRTY-FOURTH POINTER MOVE — S 0.13882326433317044 @ 181,373 B [contest-CUDA T4 n600]: pc2 carrier scales = 1.0 + zero-byte lattice re-solve: −41 B at held seg, pose within the print (34th pointer move) (2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M23BECK3ASVVNKZXC7Y5JGTY`. Lane `ddm_pc2_t4_carrier_scales_resolve_20260909`. Modal wall 573.1 s. Archive sha `e138ee097905902ad6e1d49841b2ff2f043736298079f66c0a1628031bcd8372`, 181,373 B. Runtime tree `32a74083c0dff8c3eb20305faba5b9ab3ae26c171b673372a070326862063cb0`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·181,373/37,545,489 | 0.12076883590462759 |
| seg 100·0.00010913 | 0.010912999999999999 |
| pose √(10·5.1e-06) | 0.0071414284285428505 |
| **S** | **0.13882326433317044** |

| | ddm_rc2_t4_hpac_semistatic_mixing_20260909 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13885056455024844 | 0.13882326433317044 | **-2.7300217078002342e-05** |
| d_seg | 0.00010913 | 0.00010913 | 0.0 |
| d_pose | 5.1e-06 | 5.1e-06 | 0.0 |
| bytes | 181,414 | 181,373 | -41 |

## Projection fidelity

Projected 0.13882141970714074; realized − projected = 1.8446260297011463e-06. projection −1.84e-6 optimistic: the 2-sig-fig pose print (5.1e-6) vs the instrument's 5.0901e-6; seg reproduced exactly

## The mechanism

pc2 (`.omx/research/ddm_pc2_carrier_rice_k_width_rank_cut_and_scales_20260908.md`, seal `SEAL_ddm_pc2_carrier_scales_resolve.json`, commits 783bef58a/64cf42609) changed the pose carrier section only: all twelve per-atom basis scales set to 1.0 (twelve identical float32 words compress where twelve distinct ones do not; −41 B at the container) plus a zero-byte 40-round `jg5.refine_pair` re-solve on the live int12 lattice that moved 17 of 7,200 coordinates (9/600 pairs improved; 581 stopped at no_improving_step, 19 at lattice_floor). The receiver ships unmodified (reserved 0x7a, no format change). The carrier renders frame 0 only, so d_seg is carried by construction (0.00010913 reproduced); d_pose printed 5.1e-6 at two significant figures on T4 (instrument 5.0928e-6 → 5.0901e-6). Rate 181,414 → 181,373 B. Identity control on rc2's archive first (181,414 B, delta 0); smoke pair on both trees with the frontier role pinned to rc2's archive.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.018823264333170442. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.01805442842854285: archive ≤ 153,103.9 B → **-28,269.1 B**.
- **DISTORTION corner** at held bytes 181,373: distortion ≤ -0.00076884 → **inf× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **-1,154.653 B over** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_pc2_t4_carrier_scales_resolve_20260909/MODAL_REMOTE_RESULT.json` (sha `535c83ca164d4d03c1d91d6d245e42b3b526d49d2a09479a452f9d0bd1ebf4c6`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_pc2_carrier_kwidth_rankcut/rebase_scales_resolve/candidate_runtime/archive.zip` (sha `e138ee097905902ad6e1d49841b2ff2f043736298079f66c0a1628031bcd8372`, 181,373 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_pc2_carrier_kwidth_rankcut/SEAL_ddm_pc2_carrier_scales_resolve.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_pc2_t4_carrier_scales_resolve_20260909/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_pc2_t4_carrier_scales_resolve_20260909/SEAL_ddm_pc2_carrier_scales_resolve.json` (sha verified: True).

## What this does NOT claim

Not claimed: the rank cut (r=8, −6,356 B) — REFUSED 48.9× past break-even on a population lower bound; every Jacobian column lies exactly in the span of the other eleven, so a rank cut is free at first order and the measured 4–2,733× pose cost is the int12 LATTICE; halving the lattice step fails paired. Not claimed: pc1's "more solving helps" — settled by this move's own re-solve (17/7,200 coordinates, 0.076%); the shipped codes were already a `refine_pair` fixed point and the lattice was doing the work. Not claimed: ITEM 2's 48-B block shrink (−7 B at the container once the scales are identical; needs the renderer's own carrier reader). Projection residual +1.84e-6 S (the 2-sig-fig pose print), inside the banked band. No CPU-axis number.

## Next from here

1. sj1's pass 4 (in flight: 423 cells repaired, seg −3.59e-4 S) runs its carrier chain from THIS tree's coefficients (its `load_carrier_state` reads the pointer tree's own archive) and seals against 181,373 B / 0.13882326433317044.
2. rc3's held seal (−201 B, model section only) re-bases onto whichever tree sj1's row produces and fires next; fe1 (FiLM codes) and rw1 (renderer edge fold-back) follow, rw1 last.
3. Carrier doors remaining: gb1 (generated basis priced on its lattice cost; live), ITEM 2 (−7 B, only with a reader change). The lattice-not-span law is registered on `pose_carrier_basis_rate_fidelity_exchange_v1`.

Equations leg (`tac.canonical_equations`): tac.canonical_equations: pose_carrier_basis_rate_fidelity_exchange_v1 (anchors: rank cut free in span / 4–2,733× on the lattice; re-solve fixed point 17/7,200) + model_section_edit_container_break_fee_v1 (0 B on whole-column Rice/CABAC rewrites)

Own-vehicle frontier: **S 0.13882326433317044 @ 181,373 B [contest-CUDA T4 n600]**, archive sha `e138ee09…d8372`.
