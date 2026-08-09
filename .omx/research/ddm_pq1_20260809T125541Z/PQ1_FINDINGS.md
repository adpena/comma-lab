# PQ1 findings — PR130 pose-leg Metal port build

**Verdict: `BUILD-COMPLETE / METAL-RECEIPT-OWED`.** MAIN now has a push-button
pose-wrapper port and a decisive native-sparse MPS probe. This arm did not run
Metal, MPS, CUDA, the trainer, a scorer, an evaluator, or an archive. It measured
no score and moved no frontier.

Axis: `[scorer-free CPU/static build]`. Score claim: false. Promotion eligible:
false.

## What is now built

| surface | behavior | source proof |
|---|---|---|
| Device cache | CPU no-op; MPS calls `torch.mps.empty_cache`; CUDA calls `torch.cuda.empty_cache`; unknown devices refuse | `src/tac/pr130_lift/pose/mps_port.py:19-30`; wired at `train_pose_carrier_full_resumable.py:341-345` |
| PoseNet load | safetensors load occurs on CPU, then `load_state_dict`, then module placement | `mps_port.py:33-49`; wired at wrapper lines 347-351 |
| CPU/CUDA coefficients | preserves the borrowed `Embedding(..., sparse=True)` plus `RowLocalSparseAdam` path | `mps_port.py:159-179` |
| MPS coefficients | uses a dense gradient representation, but updates only explicitly selected rows with the borrowed row-local Adam equations and the same state keys/shapes | `mps_port.py:81-156,159-179,182-219`; wired at wrapper lines 356-374,429-469 |
| Fail-closed guards | refuses stale/missing row declarations, undeclared nonzero gradient rows, sparse/dense optimizer mismatches, and selected-row-set drift | `mps_port.py:52-78,99-125,182-219` |
| MAIN probe | subprocess-isolated CPU/MPS native-sparse comparison with fallback disabled, exact selected rows/clocks/untouched rows, and predeclared fp32 tolerances | `tools/probe_sparse_mps.py:21-28,31-123,126-164,176-310` |

The intake remains read-only. Its trainer SHA-256 is still
`684a4906edecb7653572db77c11a03a4e445eb256a8dc7b665e8fa0f78cab649`.
The tested lifted `RowLocalSparseAdam` block is text-identical to intake lines
117-177; the full lifted file differs for prior custody/lift reasons and has
SHA-256 `0695685d1e8d61556da64f482296660779228088a1a8d8acb8eb6347d8afd807`.

## CPU equivalence proof

The reference fixture loads the real lifted `RowLocalSparseAdam`. It initializes
identical sparse and dense coefficient tables, then runs two squared-loss steps
with repeated and overlapping IDs:

1. `[5, 1, 5, 3, 1]` → selected rows `{1, 3, 5}`.
2. `[3, 9, 3, 5, 9]` → selected rows `{3, 5, 9}`.

Both paths use a `0.75` selected-gradient norm cap. After each step the test
requires exact equality of the full coefficient table and the complete
`row_step`, `exp_avg`, and `exp_avg_sq` tensors. It separately requires every
row not selected in that step to remain bit-identical to its pre-step value.
All assertions passed. Therefore repeated IDs increment a row clock once per
coalesced row, not once per occurrence; stock dense Adam semantics are not
being substituted.

Verification:

- Focused port + resume tests: `10 passed`.
- Complete `src/tac/pr130_lift/tests`: `27 passed`; the process emitted an
  atexit `No Metal device available` message from the imported MLX binding, but
  pytest exited 0 and no Metal operation ran.
- CPU worker for `Embedding(600,12,sparse=True)` + two repeated-ID steps:
  exited 0 under local Torch 2.12.1, selected rows `[2,5,19]` then `[5,7,19]`,
  nonzero row-clock union `[2,5,7,19]`, and untouched rows bit-identical.
  This is a build check, not a receipt for pinned Torch 2.10.0.
- `py_compile`, Ruff, and `git diff --check`: passed.

## PP2 unknown denominator after this build

PP2 inventoried 60 distinct execution families and classified 3 UNKNOWN (5.0%):
rows 33, 36, and 37.

| PP2 row | post-build disposition | honest residual |
|---|---|---|
| 33 direct safetensors-to-MPS | **STATIC-REMOVED** from the wrapper path by CPU-first load | real PoseNet state/device parity still needs the full Metal graph receipt |
| 36 sparse embedding backward | **STATIC-BYPASSED on the active MPS wrapper path** by dense-gradient representation | native sparse reference form remains unmeasured on pinned Torch 2.10.0 |
| 37 sparse COO ingress | **STATIC-BYPASSED on the active MPS wrapper path** by the row-selected adapter | native COO/coalesce path remains unmeasured on pinned Torch 2.10.0 |

Thus the native-sparse residual is **2/60 = 3.333...%**. All **3/3** PP2
unknowns are removed or bypassed in the active MPS wrapper source, but this is
not a portability verdict: **0/60 execution families have a pinned-version
real-Metal receipt from PQ1**. The other 57 were statically covered by PP2, not
executed here.

## Optimal form and mechanism fidelity

REFERENCE form remains the borrowed sparse embedding plus `RowLocalSparseAdam`
on pinned Torch 2.10.0, with zero automatic CPU-fallback warnings and CPU/MPS
state/update parity. PQ1's only scope reduction is no accelerator execution.

The MPS adapter is not a TOY-BRACKET and not a mechanism reduction: only the
autograd gradient representation changes from sparse COO to a dense 600×D
table. The update set, coalesced repeated-ID behavior, row-local clocks,
moments, bias correction, clipping norm, and untouched-row identity are the
same and are covered by exact CPU-vs-CPU behavioral equality. If MAIN's native
sparse probe passes, the reference path remains available for direct use; if
it fails, the active wrapper already has the mechanism-equivalent fallback.

## Ordered MAIN runbook

Every step is sequential. MAIN owns the Metal device and any local training
process. Stop on the first failed verification and preserve its receipt.

### 1. Re-verify source and input custody

```bash
shasum -a 256 \
  /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/train_pose_carrier_full.py \
  /Volumes/VertigoDataTier/pact/ddm_mx2_20260806/inputs/gt_pose_cache_600.pt \
  /Volumes/VertigoDataTier/pact/ddm_mx2_20260806/master_cache/OUR_SURFACE_MASTERS.pt
```

Verification: require respectively
`684a4906edecb7653572db77c11a03a4e445eb256a8dc7b665e8fa0f78cab649`,
`0eae6dab35331bfacebd787548b901553bdcf373abe3d88371a723989fb65d68`,
and `3a9792136823046eb89d3b7d808d07e5a1186cbef6ec78f58d260a5472b709b4`.
The target is the prior selected PR130 official DALI-axis cache; do not mix it
with an AV target in the same receipt.

### 2. Materialize the pinned MPS runtime outside upstream

```bash
export PQ1_ENV=/Volumes/VertigoDataTier/pact/ddm_pq1_runtime_20260809/venv
UV_PROJECT_ENVIRONMENT="$PQ1_ENV" uv sync --project upstream --frozen --group mps
"$PQ1_ENV/bin/python" -c 'import torch; assert torch.__version__.split("+")[0] == "2.10.0"; assert torch.backends.mps.is_available(); print(torch.__version__)'
```

Verification: the command prints Torch `2.10.0` and exits 0. The environment
is outside the immutable `upstream/` tree.

### 3. Run the decisive native sparse probe

```bash
export PQ1_SSD=/Volumes/VertigoDataTier/pact/ddm_pq1_20260809
mkdir -p "$PQ1_SSD"
PYTHONPATH=src "$PQ1_ENV/bin/python" tools/probe_sparse_mps.py \
  --out "$PQ1_SSD/sparse_mps_probe.json"
```

Verification:

```bash
jq -e '.verdict == "PASS" and .zero_cpu_fallback_warnings == true and .required_torch_version == "2.10.0"' "$PQ1_SSD/sparse_mps_probe.json"
```

A `FAIL` receipt is a first-class native-sparse result, not permission to edit
the tolerance. The tolerances were fixed in source before execution:
`atol=2e-6`, `rtol=2e-5`.

### 4. Run the actual wrapper graph on both carrier-base branches

First confirm no other Metal arm is running; this is sequential
one-Metal-fire-at-a-time. Then run each branch separately:

```bash
export PYTHONPATH=src
export PYTORCH_ENABLE_MPS_FALLBACK=0
for PQ1_BASE in gray master; do
  PQ1_RUN="$PQ1_SSD/full_graph_${PQ1_BASE}"
  mkdir -p "$PQ1_RUN"
  "$PQ1_ENV/bin/python" tools/safe_run.py \
    --rss-mb 45000 --timeout 1800 --projected-gib 45 \
    --label "pq1-pose-${PQ1_BASE}" \
    --status-receipt "$PQ1_RUN/safe_run.json" \
    --child-pidfile "$PQ1_RUN/child.pid" -- \
    "$PQ1_ENV/bin/python" -m tac.pr130_lift.pose.train_pose_carrier_full_resumable \
      --challenge-root upstream \
      --target-cache /Volumes/VertigoDataTier/pact/ddm_mx2_20260806/inputs/gt_pose_cache_600.pt \
      --master-checkpoint /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt \
      --init-carrier /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/archive_carrier_int6_stable_s8k.pt \
      --master-cache /Volumes/VertigoDataTier/pact/ddm_mx2_20260806/master_cache/OUR_SURFACE_MASTERS.pt \
      --reuse-master-cache --cache-masters-on-device \
      --steps 2 --stop-after-step 2 --batch-size 4 --eval-batch-size 4 \
      --render-batch-size 4 --eval-every 1 --state-save-every 1 \
      --lr-basis 1e-6 --lr-coeff 3e-4 \
      --basis-freeze-fraction 0 --basis-train-until-fraction 1 \
      --qat-fraction 0 --coeff-qat-fraction 0 --always-metric-loss \
      --basis-bits 6 --coeff-bits 12 --amplitude 64 \
      --carrier-base "$PQ1_BASE" --seed 20260722 --device mps \
      --smoke-pairs 4 --out "$PQ1_RUN/result.json" \
      --save "$PQ1_RUN/carrier.pt" \
      >"$PQ1_RUN/run.log" 2>&1
  jq -e '.status == "ok" and .exit_code == 0' "$PQ1_RUN/safe_run.json"
  jq -e '.verdict == "SMOKE_ONLY" and .scope.full_scope == false' "$PQ1_RUN/result.json"
  test -s "$PQ1_RUN/carrier.step000002.full_state.pt"
  if rg -i 'fallback|fall back|not currently supported on the mps backend' "$PQ1_RUN/run.log"; then
    exit 1
  fi
done
```

Verification: both branches exit 0 under the governor, write step-1 and step-2
full-state checkpoints, have finite results, and emit zero fallback text. This
is a scope-reduced port receipt, not a family or score verdict.

### 5. Register the Metal result before any bounded training fire

Append the probe and full-graph receipts through
`tac.probe_outcomes_ledger.register_probe_outcome`; do not bare-write the
ledger. A full n600 training fire remains separately governed, resumable, and
stage-checkpointed. Do not claim all stages 09-32 portable until the three
non-trainer script families are audited.

## RECALL EVIDENCE

| scope searched | query / source | beyond charter seeds | plan change |
|---|---|---|---|
| Live board and ledgers | `ddm_pq1`, `ddm_pp2`, `pr130_pose`, `sparse MPS` in hot state, lane registry, task/bridge stores, and probe ledger | Hot state explicitly lists PQ1 live; PP2's PARTIAL row is present; no superseding pose-MPS receipt was found | Kept this arm as the implementation owner and registered a new PARTIAL build row rather than overwriting PP2 |
| Research corpus and indexes | content search for `RowLocalSparseAdam`, `SparseMPS`, `train_pose_carrier_full`, and `probe_sparse_mps` across research index, DAG, specs, and receipts | MX2B already built the resumable wrapper and 600-pair caches; the semantic MPS port did not cover the pose sparse path | Extended MX2B in place and reused its cache/run surfaces instead of creating a parallel trainer |
| Canonical equations | `tools/list_canonical_equations.py --json` filtered for row-local/sparse/MPS/pose-carrier port terms | No equation or empirical anchor settles pinned-version sparse MPS optimizer behavior | Kept portability as an execution probe, not an equation-derived claim |
| Primary source | intake trainer, lifted custody copy, `upstream/uv.lock`, and current wrapper/tests | The row-local optimizer block is text-identical between intake and lift; lock selects Torch 2.10.0 for MPS | Tested against the real borrowed equations and hard-pinned the MAIN probe version |

## Boundaries

- No MPS, MLX, CUDA, trainer, scorer, evaluator, archive, paid job, or remote
  dispatch was executed by PQ1.
- The local environment is Torch 2.12.1 with
  `torch.backends.mps.is_available() == False`; its CPU worker is not evidence
  for the governed 2.10.0 runtime.
- The other 9/24 pose stages using `learned_pose_carrier_oracle.py`,
  `search_pose_coeff_cpu.py`, and `refine_pose_coeff_codes.py` remain outside
  this trainer-port verdict.
- Own-vehicle frontier unchanged at
  `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.

## LIVE-HYPOTHESES

- Native sparse MPS may pass on Torch 2.10.0: current Torch exposes the needed
  sparse registrations, and the decisive script isolates exactly that path.
- The selected-row dense adapter should run without automatic CPU fallback:
  its update uses dense MPS indexing/arithmetic already present in the borrowed
  optimizer, while only tiny validation copies are explicitly moved to CPU.
- The full PoseNet backward is likely portable because PP2 found current MPS
  coverage for every dense constituent family; exact weights and shapes remain
  the necessary test.

## DEAD-ENDS

- Stock dense Adam is closed: it changes row clocks and can move untouched rows.
- Direct `load_file(..., device="mps")` is closed: CPU-first load removes an
  avoidable external-loader unknown.
- Unconditional or availability-global CUDA cache clearing is closed: dispatch
  must follow the selected device.
- A semantic-renderer MPS receipt cannot stand in for the pose port: the pose
  path adds full PoseNet backward and row-local coefficient optimization.
- Static coverage cannot produce a portability verdict: pinned real-Metal
  execution is still required.
