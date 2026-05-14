# HDM8 Frame-0 Postfilter Local-First RSS Sweep

Date: 2026-05-14
Author: Codex
Status: active, exact-CUDA confirmation pending
Axis labels: `[local-CPU proxy]`, `[local-MPS proxy]`, `[modal-T4-CUDA proxy-prefix]`, `[contest-CUDA]`

## Question

Can deterministic first-frame-only postfilters exploit the scorer contract by changing PoseNet input while leaving SegNet's scored frame unchanged, then compress the per-pair policy as charged archive bytes?

This is a cooperative-receiver lane: the receiver/scorer is fixed, known, and deterministic. The correct workflow is therefore local broad search, CUDA confirmation on a short list, then exact auth eval only for a byte-closed archive/runtime packet.

## Scorer-Contract Basis

- SegNet scores the second/last frame of each pair. HDM8 emits pair frames as `(2p, 2p+1)`.
- Modes prefixed `even_` modify frame `2p` only; they are SegNet-null by construction under this scorer contract.
- PoseNet consumes both frames, so first-frame-only transforms can move the pose term.
- All postfilters are deterministic scalar/vector/coordinate functions. No randomness is used.

## Implementation Changes

- `tools/screen_hdm8_postfilter_sweep.py`
  - Decodes HNeRV frames once per batch instead of once per mode.
  - Batches mode chunks through PoseNet.
  - Reuses baseline SegNet values for first-frame-safe modes.
  - Emits deterministic metadata: `tool_version`, `archive_sha256`, `mode_count`, `mode_list_sha256`, batch sizes, source commit, and explicit `score_claim=false`.
- `tools/run_hdm8_local_first_postfilter_sweep.py`
  - Generates first-frame-safe palettes.
  - Runs CPU guard, MPS prefix, MPS full survivor pass.
  - Supports RSS-aware sharded parallel subprocesses.
  - Emits CUDA confirmation command only after local shortlist creation.
- `experiments/modal_hdm8_postfilter_sweep.py`
  - Runs CUDA proxy-prefix screen on Modal T4.
  - Requires lane id and instance job id before spend.
  - Emits `score_claim=false`, `promotion_eligible=false`.
- `submissions/hdm8_film_grain_sidecar/inflate.py`
  - Supports archive-packed selector format `0x03` JSON.
  - Supports archive-packed selector format `0x04` brotli-compressed JSON.
- `tools/build_hdm8_film_grain_sidecar_packet.py`
  - Packs selector bytes into archive payload.
  - Computes charged proxy fields from actual archive bytes.
  - Records selector codec, payload bytes, raw JSON bytes, archive byte delta, runtime SHA.

## Base Archive

- Path: `experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/exact_eval_static_release_surface/archive.zip`
- Bytes: `186395`
- SHA-256: `8a30730e863a2f846d7ca3a707b3191ad64312f5270976dc5f9322ba4228e8c2`
- Current HDM8 exact baseline: `[contest-CUDA] 0.20636166502462222`
- Baseline exact artifact: `experiments/results/modal_auth_eval/hnerv_hdm8_fixed_lengths_modal_t4_retry2_20260514T095000Z/contest_auth_eval.json`

## Local-First Sweep Results

Artifact root: `experiments/results/hdm8_local_first_postfilter_sweep_20260514_codex/aggressive_rss_v1`

| Stage | Axis | Pairs | Modes | Best mode | Delta vs none | Best proxy | Notes |
|---|---:|---:|---:|---|---:|---:|---|
| CPU guard | `[local-CPU proxy]` | 4 | 31 | `even_rgb_bias:-4,2,2` | `-0.009939265840281042` | `0.1995656930549582` | Representative deterministic sanity guard |
| MPS prefix | `[local-MPS proxy]` | 64 | 202 | `even_grain_chroma:3` | `-0.0011495541558772526` | `0.21742326575781934` | RSS-sharded, broad search |
| MPS full | `[local-MPS proxy]` | 600 | 65 | `even_rgb_bias:-1,0.5,0.5` | `-0.000359200303083701` | `0.22744961076227233` | Full survivor pass with per-pair arrays |

RSS evidence:

- CPU guard shards: `3`, max observed RSS `16.520843505859375 GB`
- MPS prefix shards: `7`, max observed RSS `4.0442962646484375 GB`
- MPS full shards: `4`, max observed RSS `10.858901977539062 GB`
- Configured local RSS cap: `48 GB`

## Charged Selector Packet

Local MPS per-pair selector artifact:

- Output root: `experiments/results/hdm8_even_frame_selector_mps_aggressive_rss_v1_brotli_20260514_codex`
- Archive: `experiments/results/hdm8_even_frame_selector_mps_aggressive_rss_v1_brotli_20260514_codex/archive.zip`
- Archive bytes: `187366`
- Archive SHA-256: `793747837bb1d71987e4a7055f35e25620f8eb530e6f297cc2020e5e00f1d798`
- Runtime tree SHA-256: `654fcf31045e40c76e87c973bb809bd8d533d59f040dadc80f25ad22ec69114f`
- Modal-uploaded runtime tree SHA-256: `c0aa84c41500b3f4c2f8be3036313379962fcafc7ce479c72dea666fe5370f23`
- Selector format: `0x04`
- Selector codec: `brotli`
- Selector payload bytes charged: `969`
- Raw selector JSON bytes: `3426`
- Archive byte delta vs source: `+971`
- Charged MPS selector proxy: `0.21674300805153718`
- Charged MPS selector delta vs none: `-0.011065803013818848`

Interpretation: the local selector economics are viable after byte charging. This is still not a promotion claim because earlier MPS-selected policies inverted on CUDA. CUDA confirmation must decide whether to build a CUDA-derived charged selector packet.

## CUDA Confirmation

Primary CUDA confirmation dispatched:

- Tool: `experiments/modal_hdm8_postfilter_sweep.py`
- Axis: `[modal-T4-CUDA proxy-prefix]`
- Output root: `experiments/results/modal_hdm8_postfilter_sweep/hdm8_local_first_cuda_confirm_20260514T115726Z`
- Call id: `fc-01KRK5KKSQYCCMAG95CN4FCE0K`
- Lane id: `hdm8_local_first_postfilter_cuda_confirm_20260514`
- Instance/job id: `hdm8_local_first_cuda_confirm_20260514T115726Z`
- Status at ledger write: pending

Legacy broad CUDA confirmation still pending:

- Output root: `experiments/results/modal_hdm8_postfilter_sweep/hdm8_modal_t4_policy_palette_v1_20260514T113221Z`
- Call id: `fc-01KRK453W4GT5A99XFTMQ3KXMF`
- Status at ledger write: pending

## Exact-Eval Gate

No archive from this lane is promotable until all are true:

1. CUDA proxy-prefix confirms a positive per-pair selector after charged bytes.
2. A selector is rebuilt from the CUDA proxy JSON, not MPS.
3. Selector bytes are packed in archive format `0x04`.
4. Runtime manifest uses the Modal-uploaded tree SHA.
5. `experiments/modal_auth_eval.py` exact `[contest-CUDA]` replay returns below the current exact baseline.
6. If the result is above `0.192`, this local HNeRV/HDM8 postfilter family is not allowed to keep consuming priority without a new mechanism.

## Reproduction Commands

Focused tests:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_hdm8_local_first_postfilter_sweep.py \
  src/tac/tests/test_modal_hdm8_postfilter_sweep.py \
  src/tac/tests/test_hdm8_film_grain_sidecar.py -q
```

Aggressive RSS-sharded local sweep:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/run_hdm8_local_first_postfilter_sweep.py \
  --output-dir experiments/results/hdm8_local_first_postfilter_sweep_20260514_codex/aggressive_rss_v1 \
  --profile aggressive \
  --cpu-guard-pairs 4 \
  --cpu-guard-max-modes 32 \
  --cpu-decode-batch-pairs 4 \
  --cpu-score-batch-pairs 4 \
  --cpu-mode-batch-size 8 \
  --cpu-parallel-workers 4 \
  --cpu-shard-size 12 \
  --mps-prefix-pairs 64 \
  --mps-prefix-parallel-workers 3 \
  --mps-prefix-shard-size 32 \
  --mps-decode-batch-pairs 8 \
  --mps-score-batch-pairs 4 \
  --mps-mode-batch-size 3 \
  --full-pairs 600 \
  --full-top-k 48 \
  --full-margin 0.0010 \
  --mps-full-parallel-workers 2 \
  --mps-full-shard-size 18 \
  --cuda-top-k 20 \
  --cuda-margin 0.00045 \
  --cuda-mode-batch-size 4 \
  --max-local-rss-gb 48
```

Compressed charged selector packet from local MPS arrays:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/build_hdm8_film_grain_sidecar_packet.py \
  --archive experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/exact_eval_static_release_surface/archive.zip \
  --runtime-template submissions/hdm8_film_grain_sidecar \
  --output-dir experiments/results/hdm8_even_frame_selector_mps_aggressive_rss_v1_brotli_20260514_codex \
  --mode ignored \
  --proxy-json experiments/results/hdm8_local_first_postfilter_sweep_20260514_codex/aggressive_rss_v1/mps_full_sweep.json \
  --selector-from-proxy-json \
  --pack-selector-into-archive \
  --selector-codec brotli \
  --require-positive-proxy
```

CUDA confirmation recovery:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/modal_hdm8_postfilter_sweep.py recover \
  --output-dir experiments/results/modal_hdm8_postfilter_sweep/hdm8_local_first_cuda_confirm_20260514T115726Z
```

## Current Decision

Continue polling CUDA confirmation. If CUDA per-pair selector is positive after charged bytes, build a CUDA-derived `0x04` packet and run exact `[contest-CUDA]` auth eval. If CUDA confirms the MPS inversion pattern, retire this HDM8 first-frame postfilter variant and move to a different non-local mechanism rather than spending more time on global film-grain tuning.

## 2026-05-14T12:11Z Production Transparency Hardening

The packet builder now emits source/release transparency by default instead of
requiring a manual reconstruction after candidate generation:

- Reusable module: `src/tac/reproducibility.py`
- Packet wire-in: `tools/build_hdm8_film_grain_sidecar_packet.py`
- Tests: `src/tac/tests/test_reproducibility_transparency.py` and
  `src/tac/tests/test_hdm8_film_grain_sidecar.py`
- Latest transparent local-MPS packet:
  `experiments/results/hdm8_even_frame_selector_mps_aggressive_rss_v1_brotli_transparent_v3_20260514_codex/packet_manifest.json`

The manifest records architecture, training/curriculum provenance, experiment
axis, deployment path, eval contract, repo URL, online commit links, dirty
status, working-tree fingerprint, artifact hashes, and build/claim/eval
commands. This is not a score claim. It is a default submission/writeup/report
custody surface so score-lowering packets are automatically paper/OSS/
production-ready when they become exact-eval-positive.

## 2026-05-14T12:15Z Multiplicative Postfilter Extension

Implemented a second HDM8 first-frame-safe transform family:

- `even_contrast:<delta>`
- `even_gamma:<gamma>`
- `even_rgb_scale:<r>,<g>,<b>`

Touched runtime and proxy paths stay matched:

- `submissions/hdm8_film_grain_sidecar/inflate.py`
- `tools/screen_hdm8_postfilter_sweep.py`
- `tools/run_hdm8_local_first_postfilter_sweep.py`

Focused MPS prefix artifact:
`experiments/results/hdm8_local_first_postfilter_sweep_20260514_codex/multiplicative_probe_v1/mps64.json`

- Baseline: `none score=0.218572503`, `pose=0.00010678204`, `seg=0.00061782203`
- Best prefix: `even_rgb_scale:1.02,0.99,0.99+even_grain_chroma:1`
- Best prefix score: `0.217830532`
- Prefix delta: `-0.000741971`

Focused 600-pair MPS artifact:
`experiments/results/hdm8_local_first_postfilter_sweep_20260514_codex/multiplicative_probe_v1/mps600_top.json`

- Baseline: `none score=0.227808811`, `pose=0.00016401862`, `seg=0.0006319682`
- Best 600-pair mode: `even_gamma:0.985`
- Best 600-pair score: `0.227660137`
- Full-MPS delta: `-0.000148674`

Because the signal is weak but local-first-positive, dispatched a small CUDA
proxy confirmation (not exact eval, no score claim):

- Lane: `hdm8_multiplicative_cuda_probe_20260514`
- Job: `hdm8_multiplicative_cuda_probe_20260514T121530Z`
- Call id: `fc-01KRK6M42CB2VGZCZASGHBSV72`
- Output dir:
  `experiments/results/modal_hdm8_postfilter_sweep/hdm8_multiplicative_cuda_probe_20260514T121530Z`
- Recover command:
  `PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/modal_hdm8_postfilter_sweep.py recover --output-dir experiments/results/modal_hdm8_postfilter_sweep/hdm8_multiplicative_cuda_probe_20260514T121530Z`

Gate: only a positive CUDA proxy result earns an exact-eval packet. If CUDA
does not preserve the tiny MPS gain, this transform family is retired as a
local proxy artifact rather than expanded.

## 2026-05-14T12:19Z Modal Probe Transparency Default

`experiments/modal_hdm8_postfilter_sweep.py` now embeds
`tac.reproducibility.collect_source_transparency` in the local request JSON
before any detached Modal spawn. Future CUDA proxy probes record source files,
archive path, mode source, repo/commit/dirty state, artifact placeholders, and
the reproduction command at dispatch time. This is production hardening only;
the Modal proxy artifacts remain `score_claim=false` and
`promotion_eligible=false`.

## 2026-05-14T12:35Z Frame-Exploit Tile-Chroma Transfer Probe

The frame-exploit subagent found that first-frame-only chroma perturbations can
lower PoseNet proxy loss without changing SegNet, because SegNet scores only the
last frame. I transferred the smallest reusable version into the HDM8
postfilter runtime as `even_tile_chroma:<amp>` and kept runtime/proxy behavior
matched in:

- `submissions/hdm8_film_grain_sidecar/inflate.py`
- `tools/screen_hdm8_postfilter_sweep.py`
- `tools/run_hdm8_local_first_postfilter_sweep.py`

Local MPS prefix artifact:
`experiments/results/hdm8_local_first_postfilter_sweep_20260514_codex/tile_chroma_probe_v1/mps64.json`

- Baseline: `none score_proxy=0.2185725033387697`,
  `avg_posenet_dist=0.00010678203511815809`,
  `avg_segnet_dist=0.0006178220319270622`
- Best prefix mode: `even_tile_chroma:3+even_rgb_bias:-2,1,1`
- Best prefix delta: `-0.0010632672988666325`
- SegNet delta: `0.0`

Full 600-pair MPS artifact:
`experiments/results/hdm8_local_first_postfilter_sweep_20260514_codex/tile_chroma_probe_v1/mps600.json`

- Baseline: `none score_proxy=0.22780881106535603`,
  `avg_posenet_dist=0.00016401861513562228`,
  `avg_segnet_dist=0.0006319681976068144`
- Best full mode: `even_tile_chroma:3+even_grain_chroma:1`
- Best full score proxy: `0.22745982466236628`
- Full-MPS delta: `-0.00034898640298974826`
- SegNet delta: `0.0`

CUDA proxy confirmation is detached, not exact eval, and not a score claim:

- Lane: `hdm8_tile_chroma_cuda_probe_20260514`
- Job: `hdm8_tile_chroma_cuda_probe_20260514T123320Z`
- Call id: `fc-01KRK7KYQVJH4BCPBBDF93BHRR`
- Output dir:
  `experiments/results/modal_hdm8_postfilter_sweep/hdm8_tile_chroma_cuda_probe_20260514T123320Z`
- Recover command:
  `PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/modal_hdm8_postfilter_sweep.py recover --output-dir experiments/results/modal_hdm8_postfilter_sweep/hdm8_tile_chroma_cuda_probe_20260514T123320Z`

Gate: only a CUDA-positive mode may become a charged packet. If CUDA rejects the
MPS tile-chroma signal, retire this transfer as local-proxy-only and shift to a
charged per-pair selector derived from the full frame-exploit sweep.

## 2026-05-14T12:39Z Compiler/Xray Stackability Hook

Registered the postdecode selector as a first-class cooperative-receiver packet
grammar row:

- Magic: `FGS1`
- Xray label: `frame0_grain_selector_sidecar_v1`
- Substrate class: `frame0_postdecode_selector_packet`
- Compiler stage: `postdecode_scorer_aware_selector_pack`
- Source module: `submissions.hdm8_film_grain_sidecar.inflate`

This does not create a score claim. It makes the sidecar visible to PacketIR,
xray, compiler manifests, and cooperative-receiver integration ledgers as a
stackable atom rather than a one-off film-grain runtime. Focused verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_cooperative_receiver_packet_grammars.py \
  src/tac/tests/test_xray_substrate_classifier.py \
  src/tac/tests/test_hdm8_film_grain_sidecar.py \
  src/tac/tests/test_hdm8_local_first_postfilter_sweep.py -q
# 74 passed
```

## 2026-05-14T12:45Z Meta-Lagrangian Water-Fill Integration

Added a canonical planner bridge so postdecode selector sweeps can flow into
the same Pareto/meta-Lagrangian/field-equation stack as other byte-priced atoms:

- Track registry row: `frame0_postdecode_selector`
- Phase: `postdecode_atom`
- Pareto axis: `multi`
- Planner visibility:
  `packet_compiler`, `xray_substrate_classifier`, `meta_lagrangian_search`,
  `pareto_3axis`, `cathedral_autopilot`, `continual_learning_posterior`,
  `field_equation_planner`
- Planner module:
  `src/tac/optimization/postdecode_selector_waterfill.py`
- CLI:
  `tools/build_postdecode_selector_atom_ledger.py`

First concrete atom ledger from the tile-chroma full-600 MPS sweep:

```bash
.venv/bin/python tools/build_postdecode_selector_atom_ledger.py \
  --sweep-json experiments/results/hdm8_local_first_postfilter_sweep_20260514_codex/tile_chroma_probe_v1/mps600.json \
  --output experiments/results/hdm8_local_first_postfilter_sweep_20260514_codex/tile_chroma_probe_v1/postdecode_selector_waterfill_plan.json \
  --atom-ledger-output experiments/results/hdm8_local_first_postfilter_sweep_20260514_codex/tile_chroma_probe_v1/postdecode_selector_atom_ledger.json \
  --selector-byte-delta 512 \
  --confidence 1.0
```

Planning-only result, with selector bytes charged:

- Top atom: `postdecode_selector:oracle_pairwise_waterfill`
- Charged byte delta: `512`
- MPS-proxy expected score delta: `-0.004205361464`
- Pose-dist delta: `-0.000034757295311464985`
- Seg-dist delta: `~0`
- Selected non-`none` pairs: `405/600`
- Dispatch blockers: `planning_only_lagrangian_atom`,
  `requires_exact_cuda_auth_eval`, `proxy_or_planning_only`,
  `requires_byte_closed_selector_archive`

This is the concrete stack answer: global film grain is weak, but a per-pair
selector is a water-fill atom. It can be optimized by CMA-ES/Optuna over palette
and amplitudes, lowered by PacketIR/compiler into charged bytes, inspected by
xray, and admitted to Pareto/autopilot only after CUDA proxy and exact-eval
custody.

## 2026-05-14T15:38Z Transfer-Failure Guard Hardening

Exact-CUDA replay invalidated the MPS-selected HDM8/FES1 selector transfer
path:

- HDM8 fixed-length baseline: `[contest-CUDA] 0.20636166502462222`
  (`186,395` bytes; `avg_posenet_dist=3.236e-05`,
  `avg_segnet_dist=0.0006426`).
- HDM8 even-frame selector built from MPS proxy: `[contest-CUDA]
  0.22816528594942062` (`188,756` bytes;
  `avg_posenet_dist=0.00014608`, `avg_segnet_dist=0.0006426`).
- FES1 selector: `[contest-CUDA] 0.2261263460995081`, while the paired
  `[contest-CPU]` run was `0.2088908495021802`; the CUDA failure is a
  scorer-device/PoseNet transfer failure, not a SegNet failure.

Classification: measured MPS/CPU-ranked selector configurations are retired
for CUDA ranking. The method family is not retired; future film-grain/
postdecode-selector work must rank on `modal-t4-cuda-proxy-prefix` rows first,
then build a byte-closed selector packet and run exact `[contest-CUDA]` auth
eval before any promotion or kill decision.

Builder hardening landed:

- `tools/build_hdm8_film_grain_sidecar_packet.py` now records
  `positive_proxy_candidate_for_cuda_probe=true` for positive local/MPS proxy
  signals but keeps `ready_for_exact_cuda_after_positive_proxy=false`.
- The manifest now includes `cuda_transfer_policy` with `rankable_on_cuda=false`,
  `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`, and the
  required CUDA confirmations before ranking.
- Positive proxy packets carry
  `positive_proxy_requires_cuda_transfer_confirmation` in `dispatch_blockers`.

Focused tests:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_hdm8_film_grain_sidecar.py \
  src/tac/tests/test_frame_exploit_cuda_transfer_audit.py
# 18 passed
```

Current Modal proxy-prefix harvest status remains nonterminal as of
`2026-05-14T15:37:57Z`: all four HDM8 sweeps
(`hdm8_local_first_cuda_confirm_20260514T115726Z`,
`hdm8_modal_t4_policy_palette_v1_20260514T113221Z`,
`hdm8_multiplicative_cuda_probe_20260514T121530Z`,
`hdm8_tile_chroma_cuda_probe_20260514T123320Z`) recovered as `pending` with
`score_claim=false` and `promotion_eligible=false`.

Local CPU advisory calibration:

- HDM8 baseline `[macOS-CPU advisory]`: `0.22782116227855223`,
  `avg_posenet_dist=0.00016402`, `avg_segnet_dist=0.00063209`,
  `archive_bytes=186395`.
- `even_grain_chroma:1.0` `[macOS-CPU advisory]`: `0.22748270585134556`,
  `avg_posenet_dist=0.00016129`, `avg_segnet_dist=0.00063209`,
  `archive_bytes=186395`.
- Advisory delta: `-0.0003384564272066737`; `score_claim=false`,
  `promotion_eligible=false`, `hardware_compliance_blocker=contest_cpu_requires_linux_x86_64`.

Interpretation: local CPU supports using small film-grain modes as hypothesis
generators, but it does not rescue the MPS-selected HDM8 selector because exact
CUDA already showed a large PoseNet regression. CUDA ranking remains mandatory.

## 2026-05-14T20:56Z Modal-Cancel Recovery Closure

The four pending Modal T4 proxy-prefix calls were recovered after hardening
`experiments/modal_hdm8_postfilter_sweep.py` to persist provider exceptions
instead of letting `FunctionCall.get()` failures escape before custody writes.
All four calls now have terminal recovery summaries and terminal lane-claim rows:

- `hdm8_local_first_cuda_confirm_20260514T115726Z`:
  `cancelled_provider_function_call`, Modal `RemoteError`,
  `Function call was cancelled by user.`
- `hdm8_modal_t4_policy_palette_v1_20260514T113221Z`:
  `cancelled_provider_function_call`, Modal `RemoteError`,
  `Function call was cancelled by user.`
- `hdm8_multiplicative_cuda_probe_20260514T121530Z`:
  `cancelled_provider_function_call`, Modal `RemoteError`,
  `Function call was cancelled by user.`
- `hdm8_tile_chroma_cuda_probe_20260514T123320Z`:
  `cancelled_provider_function_call`, Modal `RemoteError`,
  `Function call was cancelled by user.`

Classification: provider-cancelled infrastructure outcome, not a score result
and not method evidence. These artifacts carry `score_claim=false` and
`promotion_eligible=false`. The lane remains blocked on fresh CUDA-ranked proxy
evidence before any new byte-closed selector packet or exact `[contest-CUDA]`
auth eval can be justified.

Focused regression:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_modal_hdm8_postfilter_sweep.py -q
# 10 passed
```

## 2026-05-14T21:05Z Adversarial Recovery-Failclose Patch

Fresh-eye adversarial review of the recovery hardening found two custody bugs:

- `invalid_result` recovery returned no `passed=false` / `returncode`, so the
  CLI could exit `0`.
- Raw provider exception text could contain markdown table separators or
  control characters, causing terminal claim writes to fail after a terminal
  summary had already been persisted.

Fix landed in `experiments/modal_hdm8_postfilter_sweep.py`:

- invalid non-dict Modal results now produce `passed=false`, `returncode=5`,
  `axis=modal-t4-cuda-proxy-prefix`, and a terminal
  `failed_modal_hdm8_postfilter_sweep_invalid_result_no_score_claim` claim
  when claim identity is available.
- provider failure notes are sanitized before dispatch-claim writes.
- recovery summaries now include `lane_id`, `instance_job_id`, `claim_agent`,
  `archive_sha256`, `archive_size_bytes`, `terminal_claim_closed`,
  `terminal_claim_status`, and `terminal_claim_error`.
- claim-close failures fail closed with `returncode=6`.

The four cancelled HDM8 Modal proxy-prefix calls were re-recovered through the
patched path; each summary is now self-contained and still has
`score_claim=false` / `promotion_eligible=false`.

Focused regression:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
  src/tac/tests/test_modal_hdm8_postfilter_sweep.py -q
# 11 passed
```
