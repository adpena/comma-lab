# DDM MP2 findings — PR130 semantic stack on local Metal

Tags: `[no-triality] [p0-ledger-ok]`

**Verdict: NO on this managed M5 Max execution surface.** The exact PR130 semantic model and the pose leg's exact semantic-master model both fail at their first `.to(torch.device("mps"))` call because this session exposes zero MPS devices. This is an `INSTANCE` verdict about the current execution surface, not a `FORMULATION` or `FAMILY` verdict against PyTorch MPS, MLX, PR130, or Apple Metal.

Axis: `[macOS-MPS device-coverage; synthetic shape declaration; no scorer; no training]`.
`score_claim=false`, `promotion_eligible=false`, `pointer_moved=false`,
`full_n600_scorer_forwards_run=0`, `metal_training_steps_run=0`, `paid_dispatches=0`.

## What was measured

The host identifies as an Apple M5 Max MacBook Pro with 128 GB RAM. The active repo runtime is Python 3.13.12, PyTorch 2.12.1, and safetensors 0.8.0. PyTorch reports `mps_built=true`, `mps_available=false`; the alternate SSD project runtime reports `torch.mps.device_count() == 0`. MLX imports in that alternate runtime but its first `mx.eval` raises:

> `[metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not accessible.`

`sw_vers` reports macOS 26.4, so PyTorch's shorter error text about requiring macOS 14+ is not accepted as the root cause. The independent MLX result and MPS device count identify the blocker as Metal-device exposure to this managed session.

| leg | exact source surface | requested coverage | measured result | CPU fallback warnings | tax / timing |
|---|---|---|---|---:|---|
| semantic | `semantic_renderer_oracle.SemanticTokenRenderer`, width 96, four blocks, frame dim 8 | move real model to MPS; then shape-correct `[1,384,512]` token forward and backward | `BLOCKED_BEFORE_FORWARD` at `SemanticTokenRenderer(...).to(torch.device("mps"))` | 0 captured before failure | fallback tax **unmeasured**; s/step **unmeasured** |
| pose | `train_pose_carrier_full.py:246-249` exact semantic-master construction | move exact master to MPS; then exercise pose learnables and backward | `BLOCKED_BEFORE_POSE_FORWARD` at `.eval().to(torch.device("mps"))` | 0 captured before failure | fallback tax **unmeasured**; s/step **unmeasured** |

The empty warning list is not evidence that every op is MPS-native: execution never reached MPS kernel dispatch. It establishes neither zero fallbacks nor zero tax.

The coverage attempt is a declared `TOY-BRACKET / device-COVERAGE` mechanism reduction. It used the real model class and checkpoint configuration, but no synthetic or real tensor reached the device. It is not a training result.

## Source confirmation and refutation

- `[MEASURED source]` `train_semantic_full.py:58,65,73-76` makes `--device` real and transfers the model through the selected `torch.device`.
- `[MEASURED source]` `train_semantic_full.py:118-122` enables bf16 autocast only when `device.type == "cuda"`; `--amp` therefore remains a no-op on MPS as MAIN expected.
- `[MEASURED source]` `train_pose_carrier_full.py:220,232,246-249` likewise accepts `--device` and first transfers the exact semantic master through it.
- `[MEASURED source]` the semantic model uses standard PyTorch embeddings, convolutions, group norm, linear FiLM, one-hot, interpolation, GELU, sigmoid, meshgrid/linspace/arange, stack, and concat. The pose trainer adds sparse embedding/gradient operations, einsum, quantization primitives, Adam, and a custom sparse Adam.
- `[REFUTED only at whole-directory scope]` the charter's inherited statement that the whole `code/` directory has zero hardcoded CUDA device pins is too broad. There are zero `.cuda()` calls, but `build_gt_cache_official.py:28-30` and `inflate.py:665-669` explicitly select CUDA. Neither pin is in `train_semantic_full.py` or `train_pose_carrier_full.py`, so MAIN's narrower conclusion that these two trainers have a CUDA default rather than a CUDA pin remains source-confirmed.

The actual device-coverage result is stronger than the static inventory for this session: no listed semantic or pose operation was reached, so none can yet be classified as MPS-native or CPU-fallback.

## Cache label choice

No cache was trained against because no training step started. The selected cache for the blocked real-short-train stage was the retained PR130 official Ada DALI cache:

- durable materialization: `/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt`
- bytes: `117,981,301`
- SHA-256: `382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195`
- compressed intake source: `gt_cache_600_official_ada.pt.xz`, `526,820 B`, SHA-256 `233884c672eff22258376cf9532bb69a52017980000a2615bbd917ba7a8ec3dc`

The uncompressed SHA was re-measured both by streaming the intake XZ and by hashing the durable OP1R materialization. This cache was selected because OP1R established it as the exact shipped PR130 DALI label object. The Modal-T4 cache `a91d98252fe377c5...` and the local AV-like cache were not treated as interchangeable with it.

## Real short train and wall-clock envelope

The requested real short train did not fire. It was blocked before cache consumption, SegNet load, optimizer creation, forward, backward, or checkpoint output. Therefore:

- M5 Max Metal loss trajectory: **not measured**;
- M5 Max Metal seconds per step: **not measured**;
- CPU fallback op tax: **not measured**;
- semantic or pose full-stage local wall-clock: **not derivable from this receipt**;
- reference-comparable CUDA wall-clock: **did not find in the inspected semantic/pose recipe, scripts, checkpoint history, or evidence files**. The retained checkpoint histories contain steps and losses but not per-step elapsed time.

The source recipe specifies a selected 6,000-step semantic tail and a selected 4,000-step carrier tail, but multiplying those horizons by an invented rate would be a fake envelope. Full-stage-group feasibility remains `UNKNOWN` for a normal host Metal session and `NO` for this managed execution surface until it exposes a Metal device.

No heavy launch was attempted, so the governed-launch and passed Metal mem-probe gates were not bypassed. The intake clone remained read-only and clean. No cache was copied, no scratch bulk was created, and no cleanup action was needed.

## RECALL EVIDENCE

Sources and queries searched beyond the charter seeds:

- Memory registry: `ddm_mp2|PR130|semantic stack|semantic_renderer_oracle|train_semantic_full|train_pose_carrier_full|Metal|MPS`. This found the earlier Task #356 managed-sandbox Metal blocker and required a fresh device check rather than reuse of the old verdict.
- Full research corpus: `382d7d...|a91d982...|train_semantic_full.py|train_pose_carrier_full.py|semantic_renderer_oracle|pr130_eureka_intake|dali_vs_dali_hardware_gt_ambiguity`. This found OP1R's retained official DALI materialization and the MX1/RR9 launch apparatus.
- Canonical equations registry: `PR130|semantic|MPS|Metal|GT cache|DALI`. This found the RR9 mem-probe fire protocol and semantic-label transfer boundaries; nothing superseded the current device gate.
- Research index, DAG, hot state, and task/ledger surfaces: `PR130|semantic renderer|GT cache|DALI|MPS|Metal`. This found the compute-split directive, PR130-as-base routing, and the current own-vehicle pointer.
- Direct source: the intake `recipe/TRAINING.md`, `scripts/train.sh`, `scripts/verify.sh`, both trainers, semantic/pose oracles, checkpoint metadata, and CUDA-pin searches.

What changed: the plan selected the official DALI cache rather than either alternate cache; reused the existing Metal admission/mem-probe law instead of inventing a launcher; stopped before a non-device-backed training fire; and scoped the result to the managed execution instance. The MX1 prior does not replace this measurement: MP2 independently executed the original PyTorch model transfer and independently checked the alternate MLX runtime.

## Boundaries

- No real-input training, loss trajectory, fallback tax, or s/step was measured.
- No SegNet/PoseNet forward, n600 scorer, archive build, exact evaluation, Modal job, paid dispatch, or promotion occurred.
- MPS/MLX results are device-coverage evidence only and never score authority.
- The source intake, `upstream/`, protected common-contract files, and the shared staged index were not edited.
- No Python file was edited; review-tracker passes were not applicable.
- Verdict scope: `INSTANCE — current managed macOS execution surface exposes zero Metal devices`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN on a host process with real Metal access; consumer store: `.omx/research/ddm_mp2_20260809T105302Z/MP2_ROWS.jsonl`; fire trigger: `torch.backends.mps.is_available() == true`, `torch.mps.device_count() >= 1`, a passed governed Metal mem-probe, the official cache hash `382d7dfe...`, the PR130 challenge snapshot pin, and a P0 resumable/per-stage checkpoint wrapper are all present. Then rerun semantic and pose real-model coverage with fallback-warning capture, followed by a bounded real-cache short train with AMP off and measured loss plus synchronized s/step.

Own-vehicle frontier remains **S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]**. This arm produced no byte-closed candidate and did not move the borrowed contest pointer.
