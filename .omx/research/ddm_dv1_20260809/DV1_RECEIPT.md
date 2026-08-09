# ddm_dv1 device-axis decomposition

`score_claim=false` · pointer unchanged · no scorer or Modal dispatch by this arm

The fixed PR130 semantic renderer is slightly better against the retained AV/PyAV-lineage targets than against the retained DALI-lineage targets. That is the only signed same-object score component currently available. The whole axis winner is **UNDECIDED** because the six candidate PoseNet outputs needed for the paired pose calculation were not retained, and this arm does not own the fleet scorer slot.

## Leg A — identical renderer and bytes, target path changed

Archive: 191,052 B, SHA-256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`. The fixed rate term is `0.12721368471189708`.

| Component | AV/PyAV lineage | DALI lineage | DALI − AV | Status |
|---|---:|---:|---:|---|
| `d_seg` | 0.0002764044867621528 (32,606 / 117,964,800 sites) | 0.0002857038709852431 (33,703 / 117,964,800 sites) | +0.000009299384223090295, or +0.0009299384223090295 score units | retained `[macOS-Metal advisory, n600]`, integer arithmetic re-derived |
| candidate `d_pose` | unavailable | unavailable | unknown | **UNMEASURED_SCORER_GATED** |

The retained target-to-target PoseNet6 MSE is `0.00014004340079290474`, giving a root-metric separation of `0.03742237309323191`. This is a bound, not candidate impact. Combining that bound with the signed Seg movement gives a DALI-minus-AV fixed-renderer interval of `[-0.03649243467092288, 0.03835231151554094]`. The interval crosses zero and is not a score row.

Disposition: **HOLD at INSTANCE scope**. AV reduces semantic mismatches by 1,097, or 3.254903% relative to DALI, but the unknown candidate pose residual can dominate the 0.00092994 score-unit Seg movement.

## Leg B — isolated CPU-capable receiver

The CPU-capable copy lives at `src/tac/pr130_runtime/dv1_cpu_runtime/`; the intake clone and shared runtime were not edited. It accepts `PR130_INFLATE_DEVICE=auto|cpu|cuda`, also honors the harness-standard `PACT_INFLATE_DEVICE`, defaults to CUDA when available and CPU otherwise, and fails closed on invalid or unavailable explicit CUDA requests.

Fresh scorer-free verification passed for the archive parse, dependency set, shell syntax, token stream, semantic model, basis tensor, coefficient tensor, and HPAC state. The durable machine result is `/Volumes/VertigoDataTier/pact/ddm_dv1_20260809/cpu_runtime_verification.json`, 6,619 B, SHA-256 `4c8e6c8b5998707405dcb2869d43c5300184b95fb7d60932faa9872901249dd0`, on `[macOS-CPU scorer-free build verification]`.

The copy also repairs an evaluator-contract defect found during full recall: the committed shared FX1 `inflate.sh` passes the contest's three arguments directly to a Python entry point that expects one video base at a time. The isolated copy restores the per-video loop. This is an INSTANCE defect; the shared runtime remains unchanged and must not be dispatched as-is.

The retained primary CPU artifact proves exact reconstruction of the 600×384×512 token tensor, SHA-256 `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`, but that proof predates this copy. Raw CPU frames, CPU/CUDA raw-frame equality, scorer numerics, Linux closure, and evaluator completion are unmeasured.

Disabling both matmul and cuDNN TF32 removes one reduced-mantissa CUDA source and plausibly narrows CPU/CUDA drift. It cannot guarantee equality for convolution, interpolation, GELU, normalization, or reductions.

## Leg C — priced and unfired

One properly governed Linux CPU dispatch would buy the missing composite row for the exact archive: Linux x86_64 receiver on CPU, AV ground truth, and CPU SegNet/PoseNet. It would still conflate three changes: ground-truth decode, receiver floating kernels, and scorer kernels. A separate fixed-tensor CPU/T4 scorer matrix is required to isolate scorer numerics.

At Modal's 2026-08-09 Function rates, 8 physical cores plus 16 GiB cost `$0.00014032/s`, `$0.505152/hour`, or at most `$0.252576` for an official 1,800-second envelope. The current local wrapper permits 9,000 seconds and would cap compute at `$1.26288`; that diagnostic envelope is not contest-shaped. These figures exclude credits, image build, network, storage, regional multipliers, and nonpreemptible multipliers.

No dispatch was made. Fire is blocked on the repaired entrypoint, pinned Linux dependencies, a full-job 1,800-second governor, exact digests, a non-conflicting lane claim, fleet scorer ownership, operator GO, and durable resumable harvest.

The concurrent `ddm_dt1` range/ANS curve is complete for scorer-free n={2,8,32,120}. Its linear n600 projections were 532.9190576922163 s for range decode and 525.4987576561052 s for ANS on `[macOS-CPU advisory, scorer-free]`. While this receipt was being sealed, its existing n600 range job completed all 600 decode frames in 536.960712665983 s and decode plus render in 792.7735011659388 s. That leaves 1007.2264988340612 s of inflate-only headroom against 1,800 s. The boundary excludes checkout, LFS, dependency fetch, evaluator data loading, and the scorer pass, and it is macOS rather than contest Linux. The 3,662,409,600-byte raw scratch was SHA-256 certified as `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353` and automatically removed.

## RECALL EVIDENCE

Stores consulted:

- `.omx/research/` and `.omx/state/` with content queries for `device axis`, `DALI`, `AV`, `PyAV`, `scorer numerics`, `CPU CUDA`, `PR130`, `ddm_dt1`, and retained pose/scorer outputs.
- `.omx/research/CANONICAL_RESEARCH_INDEX*`, `.omx/research/sub015_DAG_*`, probe/task ledgers, `main_hot_state.md`, and the current handoff/system map.
- the canonical equation registry via `tools/list_canonical_equations.py --json`.
- `upstream/evaluate.py`, `upstream/.github/workflows/eval.yml`, the shared PR130 FX1 runtime, the read-only intake runtime, retained PR130/FX4 receipts, historical device-axis matrices, and `ddm_dt1` machine receipts.

What was found beyond the charter seeds and what changed:

- Public evaluation is a runner-selected mixed axis: CPU binds AV while CUDA binds DALI. This prevents describing the public ranking as a uniform CPU table.
- Historical paired gaps change sign across packets, and the canonical CPU/CUDA-gap equation is packet-specific. PR102's approximately 0.033 gap was therefore rejected as a transferable prior.
- The shared FX1 shell wrapper violates the real evaluator entrypoint contract. The isolated build was repaired and any remote fire now has an explicit entrypoint gate.
- `ddm_dt1` already owned the end-to-end timing measurement. This arm consumed its completed n≤120 receipt and terminal n600 result instead of duplicating the job.

## Durable dispositions

- **QUEUED-WITH-A-FIRE-ORDER** — owner: fleet scorer-slot owner; consumer store: `.omx/state/probe_outcomes.jsonl`; fire when a scorer slot is assigned and the fixed PR130 candidate PoseNet6 outputs can be preserved once, then evaluate all 600 pairs against both pinned target caches.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN PR130 runtime owner; consumer store: `src/tac/pr130_runtime/fx1_runtime_tree` plus its dependency-closure receipt; fire before any shared-runtime exact eval or remote dispatch, porting the isolated contest-loop repair and replaying the real three-argument contract.
- **FIRED-AND-FOLDED** — owner: `ddm_dt1`; consumer store: `/Volumes/VertigoDataTier/pact/ddm_dt1_20260809` terminal machine receipt, with the `ddm_dt1` owner responsible for repository landing; its existing n600 range decode-plus-render result was harvested during sealing and its certified scratch was removed by the owner.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN paired exact-eval owner; consumer store: `.omx/state/probe_outcomes.jsonl` and paired contest CPU/CUDA result stores; fire only after pose, entrypoint, Linux closure, terminal timing, lane, scorer-slot, and operator-GO gates pass, with the fixed-tensor scorer matrix preceding interpretation of the composite CPU row.

## LIVE-HYPOTHESES

- AV may win the full fixed-renderer target-path comparison because its measured semantic component is lower, but only the missing candidate pose residual can establish the sign of the whole target-path change.
- TF32-off may leave a smaller CPU/T4 numeric gap than older HNeRV evidence because PR130 removes both matmul and cuDNN TF32 and quantizes logits before entropy coding.
- The refactored CPU rail may fit 30 minutes because full n600 decode plus render measured 792.774 seconds, leaving 1007.226 seconds for omitted evaluator stages; scorer, Linux transfer, and full-job overhead remain unmeasured.

## DEAD-ENDS

- Do not infer candidate pose movement from target-to-target MSE: the score term is nonlinear and the missing residual direction permits either sign.
- Do not transfer PR102's CPU/CUDA gap: historical paired rows reverse sign and PR130 uses a different receiver and operating point.
- Do not call parse/model equality contest readiness: raw frames, scorer numerics, Linux closure, terminal headroom, and a real 1,800-second evaluator result are absent.
- Do not use the shared FX1 `inflate.sh` unchanged: it passes the evaluator's output directory as the per-video base and violates the receiver contract.
