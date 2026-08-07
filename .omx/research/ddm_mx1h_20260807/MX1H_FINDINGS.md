# ddm_mx1h findings

## Verdict

ARM-CAP CPU-torch n32 verdict passed on the copied live checkpoint snapshot.

| field | value |
|---|---:|
| axis | [macOS-CPU advisory torch upstream SegNet] |
| verdict_scope | n32 arm-selection instrument |
| score_claim | false |
| checkpoint step | 1500 |
| checkpoint copy | `.omx/research/ddm_mx1h_20260807/receipts/arm_cap_mlx_latest_snapshot.npz` |
| checkpoint sha256 | `b01ee1e41018dda863c2eb6c3e708a71c512b002b765a1462b27ff40ed35986a` |
| checkpoint bytes | 983,954 |
| pair count | 32 |
| segnet batch size | 32 |
| authority d_seg | 0.0010689099629720051 |
| MLX in-training d_seg_batch at same step | 0.0010215441385904949 |
| authority - MLX proxy | +0.00004736582438151027 |
| authority / MLX proxy | 1.046366889684145 |
| fp1 flat-paint floor d_seg | 0.008305 |
| authority - fp1 floor | -0.007236090037027995 |
| authority / fp1 floor | 0.12870679867212584 |

Receipt: `.omx/research/ddm_mx1h_20260807/receipts/arm_cap_torch_verdict.json`

Receipt sha256: `393e8d9d27321a724826afc288da3d782445c6872cef91d7f458c9cc1249c71e`

This is not an n600, family, contest-CPU, or contest-CUDA verdict. It is the Row-1 n32 arm-selection instrument requested by the charter.

## What Landed

- Added `--mode torch-verdict` to `experiments/ddm_mx1_pr130_semantic_renderer.py`.
- Added a strict MLX NPZ -> torch state_dict loader for the saved `param::*` checkpoint format. It checks the complete tensor key set and mapped tensor shapes before loading.
- The verdict path reads `pair_ids` from checkpoint metadata and refuses to rederive them from seed.
- The verdict path runs CPU torch only, skips MLX/Metal probing, renders through `lifted.render_for_seg(..., exact_path=True)`, runs the real upstream SegNet, and records per-pair plus aggregate d_seg.
- Added tests for synthetic NPZ -> torch mapping, receipt schema, and the no-MLX-probe CPU-only contract.

## Live Command

```bash
.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode torch-verdict --init .omx/research/ddm_mx1h_20260807/receipts/arm_cap_mlx_latest_snapshot.npz --input-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt --target-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt --verdict-batch-size 32 --out .omx/research/ddm_mx1h_20260807/receipts/arm_cap_torch_verdict.json
```

The original live run directory was not mutated. The only operation against it was reading/copying `.omx/research/ddm_mx1e_20260807/regen2/launch_arm_cap/n32_metal/mlx.latest.npz`.

## Recall Evidence

Sources searched beyond the charter seeds:

- Memory registry query: `ddm_mx1h|mx1h|ddm_mx|20260807|common_contract|lane registry|main_hot_state|harness_tasklist_bridge`. Found the bridge/live-state caution that `main_hot_state.md` is live queue authority and task IDs should be resolved against the bridge before ownerless claims. Change to plan: used `.omx/state/main_hot_state.md` as live boundary authority and did not claim global queue ownership.
- Live board query: `mx1|PR130|CPU-torch verdict|0.008305|Row-1` in `.omx/state/main_hot_state.md`. Found ddm_mx1h named as the immediate burn-window endpoint: CPU-torch verdict mode plus live dress rehearsal. Change to plan: treated this as endpoint readiness, not a broad redesign.
- Corpus query: `torch-verdict|CPU-torch|authority verdict|fp1 floor|0.008305|d_seg_batch|batch_shape` across `.omx/research`, `experiments`, `src/tac`, `docs`, and `reports`. Found RR7-F3's prior finding that the Row-1 ticket had no executable CPU-torch post-train verifier, and the launch ticket's rule that n120 selection comes only from two n32 CPU-torch verdicts. Change to plan: added an executable mode and kept the scope n32-only.
- Canonical equations registry command: `.venv/bin/python tools/list_canonical_equations.py --json`. Relevant findings included `ddm_rr9_mem_probe_fire_protocol_v1` and `ddm_hb1_semantic_label_incumbent_transfer_v1`; no equation changed the torch-verdict implementation. Change to plan: preserve axis/score_claim boundaries and do not treat MLX telemetry or external PR130 rows as authority.
- PR130/mx1 receipt inspection: `.omx/research/ddm_mx1_20260806/RECEIPT.md`, `.omx/research/ddm_rr7_20260806/ROUND7_FINDINGS.md`, `.omx/research/ddm_rr9_20260807/ROUND9_FINDINGS.md`, and `experiments/ddm_mx1_pr130_semantic_renderer.py`. Found real-cache torch-smoke/parity surfaces, strict ticket source ownership, and batch-shape recording precedent. Change to plan: reuse those cache/model/scorer surfaces and record `segnet_batch_size`, chunk sizes, token shapes, and scorer batch shapes.

Scoped negative: in the searched corpus, I did not find an existing implemented `--mode torch-verdict` before this patch.

## Verification

```bash
.venv/bin/python -m pytest experiments/tests/test_ddm_mx1_memory_probe.py
.venv/bin/python -m py_compile experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py
.venv/bin/python -m ruff check experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py
```

Results: pytest `23 passed`, py_compile passed, ruff passed, and the persisted evidence path grep found no temp-path references in the receipt artifacts.

## Follow-Ons

- ARM-VEH CPU-torch n32 verdict: QUEUED-WITH-A-FIRE-ORDER after ARM-VEH produces its own `mlx.latest.npz`; copy that checkpoint first, then run `--mode torch-verdict` with `--input-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt` and `--target-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt`.
- n120 dispatch: QUEUED-WITH-A-FIRE-ORDER only after both n32 CPU-torch verdict receipts exist and explicitly select ARM-CAP or ARM-VEH. MLX telemetry alone is not a scale-up authority.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer `0.19108` remains borrowed and unmoved.
