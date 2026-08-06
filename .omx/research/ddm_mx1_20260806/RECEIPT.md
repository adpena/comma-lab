# ddm_mx1 receipt

ddm_mx1 lifted PR130's width-96 semantic-token renderer trainer into the repo and added a fail-closed MLX/Metal training driver for the Row-1 first measurement. The local host cannot execute MLX, so MLX parity and MLX smoke are queued to MAIN's Metal host; the lifted torch CPU reference and real-label n=2 smoke were verified locally.

## What Landed

- Borrowed PR130 source under `src/tac/pr130_lift/lifted/` with per-file `borrowed_substrate_accounting` headers and pinned source hashes.
- MLX substrate port in `src/tac/pr130_lift/mlx_semantic_renderer.py`: renderer architecture, torch weight import, QAT parameter transform, curriculum loss, exact checkpoint/resume helpers, and fail-closed MLX probe.
- Driver `experiments/ddm_mx1_pr130_semantic_renderer.py`: `probe`, `torch-smoke`, `mlx-parity`, and `mlx-train` modes; stratified sampling; SSD evidence outputs; MAIN launch-ticket generation.
- Focused tests in `src/tac/pr130_lift/tests/test_mx1_pr130_lift.py`.
- Durable evidence files: `PARITY.md`, `LAUNCH_TICKET.md`, `NEXT_IF_RESUMED.md`.

No protected file was edited. `upstream/` was not edited. No n600 scorer job was run. No persisted evidence uses a slash-tmp path.

## Borrowed Substrate Accounting

PR130 source root: `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo`

Source HEAD: `2f94596bb0136d342254022a5c9584756eae0468`

| file | source sha256 | accounting |
| --- | --- | --- |
| `semantic_renderer_oracle.py` | `2bf3a6a8621334723fec1c3e596665d1f049a55f311c5348dd5a4c588873f25b` | theirs: architecture/curriculum/R path; ours: header |
| `train_semantic_full.py` | `2d7a3575e422dc2b5823b97e52101ad5632e5fa04a98e0af8a83d85b7c2176b8` | theirs: full trainer; ours: header |
| `train_semantic_quantized.py` | `4bcaf8a5c581c1e5eb057ea0ef760f269e1eabfdd2fca926bcbf61f4163a248d` | theirs: QAT loop; ours: header |
| `evaluate_semantic_quantization.py` | `5bbd2136174bfa2c99219d73f45103d4293f60e3f4eced5ee188e38053923962` | theirs: evaluation utility; ours: header |

The port is not an originality claim on PR130's mechanism. The original contribution here is the MLX/Metal substrate adaptation and the Pact-specific fail-closed harness.

## Mechanism Extraction

- Architecture: semantic token embedding, pair/frame embedding, coordinate features, optional phase/temporal channels, 4 residual token blocks at width 96, and RGB sigmoid head. Local line anchor: `src/tac/pr130_lift/lifted/semantic_renderer_oracle.py:86-168`.
- Token conditioning: center token map plus optional temporal neighbor one-hot channels, then coordinate mixing and FiLM blocks. Anchor: `src/tac/pr130_lift/lifted/semantic_renderer_oracle.py:136-168`.
- Loss: CE, softplus margin, then expected-flip margin. Anchor: `src/tac/pr130_lift/lifted/semantic_renderer_oracle.py:178-191`.
- R/uint8 training path: optional camera-resolution interpolate, uint8 STE, scorer-resolution bilinear downsample. Anchor: `src/tac/pr130_lift/lifted/semantic_renderer_oracle.py:194-202`.
- QAT: per-output-channel int4-ish fake quant with fp16 scales; embeddings use last-axis scale. Anchor: `src/tac/pr130_lift/lifted/train_semantic_quantized.py:37-71`.
- Schedule and selected boundaries: `repro_repo/scripts/e2e.py` stages 02-08; `recipe/TRAINING.md` lists semantic renderer stages 02-08 and warns that selected boundaries preserve scheduler horizon.

## Harness Decision

Decision: standalone `src/tac/pr130_lift` module plus existing MLX scorer/R imports, not an edit to the TR1 trainer.

Reason: PR130's trainer is a distinct semantic-token -> RGB recipe. Reusing the TR1 trainer loop directly would risk swapping mechanisms. The driver binds only the existing drop-in substrate surfaces:

- `torch_segnet_to_mlx` from `src/tac/local_acceleration/mlx_scorer_adapters.py:1468-1471`.
- `apply_contest_faithful_roundtrip_nhwc` from `src/tac/local_acceleration/pr95_hnerv_mlx_training.py:126-155`.

## Local Measurements

| item | result |
| --- | --- |
| py_compile | PASS |
| focused pytest | PASS, `4 passed in 0.67s` |
| local MLX probe | BLOCKED: `[metal::load_device] No Metal device available` |
| MLX parity | BLOCKED-LOCAL-RUNTIME, no max-abs/argmax/loss value measured |
| MLX smoke | BLOCKED-LOCAL-RUNTIME, driver exits rc 2 and writes blocker JSON |
| lifted torch CPU n=2 smoke | PASS-SMOKE-ONLY; loss `0.004268820863217115 -> 0.004265481140464544`; d_seg batch `0.0042317709885537624`; `4.949445009231567` s/step; resume load OK |

Artifacts are on SSD under `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/`; hashes are in `PARITY.md`.

## RECALL EVIDENCE

Searches performed beyond the charter seeds:

- Memory registry query: `mx1|MX1|#899|required-component|REQUIRED_COMPONENT|declared-on-never-read|margin_targets|preflight`. Found only #899/#904 preflight context, not an mx1-specific prior. Change to plan: keep review/serializer discipline and no raw-ledger side trip.
- Repo/corpus query: `PR130|Row-1|semantic renderer|ddm_tb1|train_tr1_partition_renderer_mlx|apply_contest_faithful_roundtrip_nhwc|torch_segnet_to_mlx|ddm_hb1|ddm_eh1` across `.omx/research`, `src/tac/local_acceleration`, and `experiments/train_tr1_partition_renderer_mlx.py`. Found EH1 row-1 trained receiver priority, HB1 label payloads/blocker, TR1 trainer context, and the canonical MLX scorer/R surfaces. Change to plan: standalone PR130 port using existing scorer/R adapters; no n600 scorer; queue n32/n120 at et4 boundary.
- PR130 recipe inspection: `recipe/TRAINING.md` plus `scripts/e2e.py` stage 02-08 commands. Found that the retained init is the QAT12k semantic checkpoint, so the launch ticket is the 6k low-LR stage 08 tail (`lr=2e-7`, `ce_fraction=0`, `softplus_fraction=-999`), not a fresh 12k QAT run.

Scoped negative: within the searched memory registry scope, I did not find a prior mx1 implementation or parity receipt.

## Follow-ons

- Row-1 MAIN fire: QUEUED-WITH-A-FIRE-ORDER in `LAUNCH_TICKET.md`.
- MLX parity: QUEUED-WITH-A-FIRE-ORDER as `--mode mlx-parity` on MAIN before treating MLX training telemetry as admissible.
- n600/full scorer: NOT OWNED by mx1; et4 owns the slot. This arm produced no n600 row and did not move the pointer.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]` per `.omx/state/main_hot_state.md` pointer line; mx1 did not produce a new byte-closed score row.
