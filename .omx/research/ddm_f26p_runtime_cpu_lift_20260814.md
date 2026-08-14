# ddm_f26p runtime CPU lift

**Status:** COMPLETE for the chartered local CPU lift. The exact contest-CPU
row is QUEUED, not fired. The canonical frontier pointer did not move.

## Result first

The sealed MC36 Variant C archive now has a real CPU-capable lifted runtime.
One full n600 decode completed on the real archive and retained both its token
checkpoint and its 3.66 GB raw output.

| fact | result |
|---|---:|
| Archive | 186,269 B, SHA-256 `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de` |
| Axis | `[M5-CPU 4-thread LOWER BOUND on contest wall]` |
| Full subprocess wall | **646.416 s** |
| Charter budget verdict | **LIKELY-IN-BUDGET** (`<900 s`) |
| Token/HPAC+RC64 stage | 383.354 s, 59.30% of subprocess wall |
| Neural render and resize | 211.329 s, 32.69% |
| Frame-0 selector and I/O | 47.032 s, 7.28% |
| CPU raw | 3,662,409,600 B, SHA-256 `e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9` |
| Retained T4 raw pin | SHA-256 `a41ca69d2288d3edd8f009b03404ef070661297a8f962a067e663ff26f7c0e8b` |
| Cross-device identity | **MISMATCH_UNQUANTIFIED** |
| Exact evaluator score | **NOT MEASURED** |

The local four-thread wall is a lower bound on the contest wall, not a Linux
x86 authority measurement. It is enough to make an exact CPU fire reasonable;
it is not evidence that the Modal host will finish under 1,800 seconds.

## Identity verdict

The CPU aggregate SHA differs from the retained T4 aggregate SHA, so bytewise
identity is refuted: at least one byte and at least one of 1,200 frames differ.
The exact divergent-frame count, changed-byte count, and maximum absolute u8
delta remain unmeasured because the retained T4 raw bytes were not available
on the mounted local stores. A bounded search did not find them in the mounted
MC36/F26 scopes under `/Volumes/VertigoDataTier/pact` or
`/Volumes/APDataStore/pact`; the MC36 dual-axis receipt places the retained T4
raw on the Modal volume `comma-ddm-js1b-argmax-retained` at
`/ddm_js1b_retained/ddm_mc36_dual_axis_t4_r1/`.

The runner accepts `--t4-raw` only when that file itself hashes to the charter
pin. Once supplied, it computes all three owed quantities over 1,200/1,200
frames: divergent-frame count, changed-byte count, and maximum absolute u8
delta. Until then, no frame-level mismatch number is claimed.

## Lifted runtime

The sealed runtime stage was not edited. The runner copies it to a new tree and
applies three narrow transformations:

1. The F26 inflator accepts an explicit device. CUDA remains the default; CPU
   is an opt-in that requires exactly four Torch threads.
2. The residual decoder invokes CUDA reproducibility setup only on CUDA.
3. The lifted entrypoint pins `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
   `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
   `NUMEXPR_NUM_THREADS` to 4 before importing Torch work.

The archive parser, HPAC model, RC64 probabilities and native decoder,
semantic renderer, carrier renderer, interpolation modes, rounding, selector,
operation order, and dtypes are otherwise preserved. The copied archive is
byte-identical to the sealed archive.

The first implementation increment is therefore a working four-thread CPU
receiver, not a proxy: it decoded all 117,964,800 tokens and rendered all 1,200
frames. It also adds an atomic end-of-token-stage checkpoint and an atomic raw
completion rename. Interrupted render payloads are retained with a failure
receipt rather than deleted.

Runtime custody:

- repo lift module SHA-256: `2da706538755d55bade782f24558e1e61992f177c2e9cc9f06ab0d24f2574182`
- lifted runtime manifest SHA-256: `111d0f3f86063101a11db08e7e2998236f3dbedb2f34392c484d2c48ce969a4d`
- harness local runtime tree SHA-256: `c9d64197f1ec843b61c9443181bdac3e65dd275e3db9dba3b146c185bfdb033b`
- environment-free runtime FILES SHA-256: `b2294857a4fd2654b604de23ce53cdaf530a629baeeda53c9784ea48cb37b94a`
- runtime content-tree SHA-256: `d4259cf3143b7b61f15715dd868ea39a05077b20a2c1309f8acf12798ebd8e05`
- informational projected Modal tree SHA-256: `57a2c803df78e19a42e167062a79f138f57fd9191d91a2d2ecd480d27d5e7478`

The projected Modal tree hash is path-dependent. The exact fire must pass
`--expected-runtime-tree-sha256 auto`; the wrapper then enforces the portable
FILES digest remotely, as designed.

## Retained payloads and receipts

All durable artifacts are under
`/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/`:

- `output/.f26_cpu_checkpoints/tokens_cpu_stage_complete.u8`: 117,964,800 B,
  SHA-256 `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52`
- `output/0.raw`: 3,662,409,600 B, SHA-256
  `e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9`
- `receipts/cpu_frame_manifest.json`: 1,200/1,200 per-frame SHA-256 rows,
  receipt SHA-256 `968c69ea2b6de1242023e7eb9eb1b087e7053cd9aae34171e6ad7f50ec23bc4f`
- `receipts/runtime_analysis.json`: hot-path operation census, speed bounds,
  and reuse inventory
- `receipts/result.json`: terminal local result
- `receipts/decode_failed.json` and `logs/decode.log`: the two pre-decode uv
  PATH failures are retained and excluded from the measured wall; neither
  failure reached token decoding or materialized a measured payload

Vertigo storage was preflighted before launch with 15.73 GB free against a
5.66 GB fail-closed requirement. The retained raw later reduced free space to
about 11 GB. No payload from this arm was discarded, no scorer slot was used,
and no Modal, Metal, MPS, or CUDA job was run.

## CPU lowering plan

The current archive has one RC64 stream, not four independent streams. Each of
its 190 groups per frame depends on earlier groups in the same frame, and each
frame's HPAC context consumes the previous token frame. The existing wire
therefore cannot be split across four process workers without changing the
byte-closed model and stream. “Parallel per-stream decode” is closed for this
exact archive shape.

The measured/counted hot path is:

- 600 frames × 190 groups = 114,000 causal group calls
- 48 patches per group call, 64 hidden channels
- 117,964,800 output symbols
- about 15.141 trillion dense-equivalent selected-logit MACs
- conv-a accounts for about 14.662 trillion of those MACs

Ranked lowering plan:

1. **Direct gathered-one-hot conv-a in Torch, then native C++/Rust if it
   survives parity.** Conv-a currently multiplies a 161-wide vector made from
   five-class one-hot planes plus two coordinate planes across 23 active
   offsets. Selecting the one active class weight and precomputing the
   coordinate contribution reduces the conv-a arithmetic count from 14.662
   trillion dense MACs to 2.095 trillion gathered class terms, a 7.0×
   arithmetic-count opportunity. This is not a measured wall speedup. Admission
   requires equality of the corrected-logit digest, CDF-input digest, all token
   bytes, and final raw output.
2. **Fuse selected-logit generation in C++ or Rust.** The target is a measured
   2× token-stage speedup. By Amdahl's law on this run, that would project the
   local full wall from 646.416 s to 454.738 s, a 1.422× whole-run speedup.
   This is a projection, not a measured implementation result.
3. **Leave RC64 alone unless direct timing changes.** The decoder is already
   native C. Prior full-n600 receiver profiling measured entropy calls at only
   1.11–3.13 seconds while HPAC probability generation consumed more than
   99.5% of token-wall time. Porting RC64 again cannot buy the needed margin.
4. **Do not prioritize render lowering before HPAC.** Even ideal removal of the
   entire measured render stage has only a 1.486× whole-wall ceiling, versus a
   2.457× ceiling for the token stage. NumPy/BLAS render work is a fallback only
   after selected logits are lowered.

As a deliberately pessimistic sensitivity, multiplying this vehicle's full
M5 wall by the related lc2 receiver's observed 2.8× M5-to-Modal slowdown gives
1,809.96 s, ten seconds over budget. That factor is from a different archive
and host pair, so it is not a contest-wall prediction. Under that sensitivity,
the token stage needs about a 1.406× speedup to project the total to 1,500 s.

## Reuse inventory

| asset | exact role for F26 | named consumer |
|---|---|---|
| HB1 `tq1c/hpac.bin.xz` and `tokens.bin` | Reuse the model-pack and exact encode/decode proof pattern for a future retrained HPAC. The learned values and token stream are not MC36's and cannot be substituted. | future HPAC retrain/packer arm |
| HB2 deploy-bound repair and receipts | Reuse the `-weight_bound..weight_bound` pack gate and full logit/token equality checks if F26's model is retrained or repacked. It does not accelerate this decoder. | future HPAC retrain/packer arm |
| `mlx_score_aware/adapter.py` | Local scorer-aware training only. It can optimize a successor HPAC/carrier but has no contest decode role. | successor training arm |
| `mlx_score_aware/portability.py` | MLX-to-portable export gate for future learned payloads. | successor export arm |
| `mlx_scorer_adapters.py` | Advisory local Seg/Pose candidate screening. | local candidate-screening arm |
| Metal grouped-conv backward, SegNet conv, fused R, sparse adjoint, integer-R adjoint | Local encode/solve/screening acceleration. None maps to F26's contest-CPU HPAC forward decoder or to exact evaluator authority. | local optimizer and screening arms |

There is no existing MLX or Metal asset that can legally execute on contest
CPU. MLX/Metal must not appear in the CPU runtime fire.

## Exact CPU fire form for MAIN

This command was **not run**. The live lane registry still contains MAIN's
earlier MC36 contest-CPU claim for the CUDA-locked runtime. MAIN must reconcile
that record to its true terminal failure, confirm no Modal job is live, and
then claim the new lifted-runtime lane before using this form. The retry stays
in the existing MC36 pair group, creates a new result directory, and uploads
the new lifted submission tree; it does not mutate or reuse the sealed CUDA
directory.

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \
  experiments/modal_auth_eval_cpu.py::main \
  --archive /Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/lifted_submission_cpu/archive.zip \
  --submission-dir /Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/lifted_submission_cpu \
  --inflate-sh inflate.sh \
  --expected-archive-sha256 f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de \
  --expected-runtime-tree-sha256 auto \
  --output-dir experiments/results/ddm_f26p_mc36_contest_cpu_20260814 \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --detach \
  --provider-detach-ack \
  --lane-id lane_ddm_f26p_mc36_contest_cpu_20260814 \
  --instance-job-id ddm_f26p_mc36_contest_cpu_20260814 \
  --claim-agent MAIN \
  --claim-policy require_active \
  --pair-group-id ddm_mc36_promotion_paired_modal_auth_20260814T182512Z
```

The fire is scorer-light but not score-free: if inflate finishes, the wrapper
runs `upstream/evaluate.py --device cpu`. Any resulting number is
`[contest-CPU]` only. It cannot replace the CUDA pointer unless it is a
qualifying lower exact row and survives the canonical promotion gates.

## RECALL EVIDENCE

Before implementation, full-corpus recall ran with:

```bash
.venv/bin/python tools/corpus_query.py \
  --stores research,equations,memory,dag,council,tasks,docs \
  --top 30 --json \
  'MC36 F26 CPU runtime lift HPAC native device axis #998 retained T4 raw decode'
.venv/bin/python tools/list_canonical_equations.py --json
```

Stores consulted were research, equations, memory, DAG, council, tasks, and
docs. Direct content recall also covered the canonical research index/DAG,
task ledger, hot state, MC36 promotion and dual-axis memos, the F26 runtime,
HB1/HB2 receipts, the native RC64 study, the `mlx_score_aware` package, and the
Metal/local-acceleration inventory.

Findings beyond the original charter changed the work:

1. `000_MAIN_ADDENDUM_READ_FIRST.md` and the lc2/RC64 receipts showed that the
   lineage already had a contest-CPU timeout and a later 1,683 s success with
   1.45× Modal host variance. That prevented treating the M5 wall as a contest
   guarantee and made the lowering census mandatory even though this local run
   finished below 15 minutes.
2. `ddm_rc64p_native_cpu_decode_20260810.md` had already measured native entropy
   work at less than 0.5% of token wall. That closed a redundant RC64 rewrite
   and redirected the plan to selected-logit generation.
3. The actual F26 receiver has one causal RC64 stream and uses the previous
   token frame in every new frame's HPAC context. That refuted the addendum's
   provisional “independent streams” premise for this archive.
4. The MC36 dual-axis memo locates the retained T4 raw on a Modal volume rather
   than either mounted SSD. That forced a typed unquantified mismatch instead
   of inventing per-frame delta evidence.
5. HB1/HB2 supply exact pack/decode gates and different learned payloads; the
   MLX/Metal packages accelerate training and screening, not the contest-CPU
   forward receiver. None can be silently transplanted into F26 decode.

Relevant registered equations included
`pr95_family_l30_range_arithmetic_coding_categorical_v1`,
`modal_dispatch_runtime_tree_hash_local_vs_worker_parity_v1`,
`ddm_hb1_semantic_label_incumbent_transfer_v1`, and
`segnet_exact_forward_cpu_thread_control_v1`. The HNeRV-specific
`cpu_cuda_score_gap_v1` was recalled but not transferred to F26.

## Boundaries and dispositions

**MEASURED:** one uninterrupted full-n600 four-thread M5 decode; archive and
token hashes; all 1,200 CPU frame hashes; CPU aggregate raw hash; local stage
walls; runtime dependency hashes; HPAC operation counts.

**NOT MEASURED:** exact contest-CPU wall; any CPU score or Seg/Pose component;
frame-count/max-delta comparison to T4; timing repeat/noise floor; a direct
gathered-one-hot or Rust/C++ HPAC speedup; any Modal, Metal, MPS, or CUDA row.

The CPU lift is real, but it did not lower an exact score. The effective
frontier remains **MC36 Variant C S 0.1619344578804448 @ 186,269 B
`[contest-CUDA T4, n600]`**.

## NEXT_IF_RESUMED

- **QUEUED** — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/receipts/`; fire trigger: before making any cross-device identity claim, download retained T4 raw SHA `a41ca69d…` from `comma-ddm-js1b-argmax-retained` and run `experiments/ddm_f26p_f26_cpu_lift.py finalize --t4-raw <downloaded-path>`.
- **QUEUED WITH FIRE ORDER 2** — owner: MAIN; consumer store: `experiments/results/ddm_f26p_mc36_contest_cpu_20260814/`; fire trigger: T4 mismatch is quantified or explicitly accepted as non-blocking, the stale/failed earlier MC36 CPU claim is reconciled terminal, the new lifted-runtime lane is claimed, no Modal job is live, and MAIN authorizes the exact command above.
- **QUEUED CONDITIONAL** — owner: successor CPU-runtime arm; consumer store: `/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/receipts/runtime_analysis.json`; fire trigger: exact contest CPU exceeds 1,500 s, times out at 1,800 s, or MAIN explicitly buys robustness headroom; prototype direct gathered-one-hot conv-a first and require full corrected-logit, CDF, token, and raw parity.

## LIVE-HYPOTHESES

- The lifted F26 runtime may fit a favorable contest-CPU host because the full four-thread M5 wall is only 646.4 s. It may still miss on a slow host because the related lc2 receiver showed large M5-to-Modal drift and 1.45× variance between Modal hosts.
- The exact MC36 CPU score may differ materially from CUDA because the raw bytes already differ. The related lc2 CPU row was worse, which makes a worse MC36 CPU row plausible, but that sign and magnitude do not transfer across archives.
- Direct gathered-one-hot conv-a may preserve every rounded logit while cutting conv-a's arithmetic count by up to 7×. The inputs and weights are quantized and the coordinates are deterministic, but changed summation order could still alter float32 rounding, so full digest/token/raw parity is required.

## DEAD-ENDS

- Independent four-worker stream decode for the existing MC36 archive is closed: the archive contains one RC64 stream, groups are causal within a frame, and HPAC consumes the previous frame.
- Another RC64/ANS native rewrite as the CPU cure is closed for this receiver: RC64 is already native C and prior direct timing put entropy calls below 0.5% of token wall.
- MLX or Metal in the contest-CPU runtime is closed: those assets are local training/screening machinery and are unavailable on the contest CPU axis.
- CPU/T4 byte identity is closed for this lifted realization: the two aggregate SHA-256 values differ. Only the extent of the mismatch remains open.
- Any score or frontier claim from this arm is closed: no exact evaluator ran, and the canonical pointer is unchanged.
