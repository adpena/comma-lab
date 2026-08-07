# ddm_mx1f Receipt - Load-Phase Allocator Bound

borrowed_substrate_accounting: PR130 semantic renderer mechanism unchanged. This landing is load-path instrumentation, static allocation bounding, and GPU microbatch memory safety for the MX1 trainer. No score claim.

## Verdict

DERIVED allocator: `experiments/ddm_mx1_pr130_semantic_renderer.py:1322`, the former n32 full-batch first `mx.value_and_grad(loss_for_params)(base_params)` graph. A conservative renderer-only live-tensor lower bound at ARM-CAP n32 is 40.59 GiB before SegNet U-Net internals, reverse-mode saved tensors, MLX kernel workspaces, and allocator cache. That is the only enumerated site large enough to explain the observed ~65 GiB system-free-memory collapse. The label caches, selected tensors, checkpoint, renderer weights, and SegNet weight file are all <2 GiB each in their worst static forms.

BOUND implemented: GPU mode now defaults to `microbatch_pairs=4`, converting tokens lazily per chunk and accumulating weighted gradients serially before the single AdamW update. The first chunked allocator is `experiments/ddm_mx1_pr130_semantic_renderer.py:1355`; the same renderer-only lower bound is 5.07 GiB per n4 chunk. CPU mode keeps full-batch behavior unless `--microbatch-pairs` is explicit.

MEASURED locally: no Metal run. Verification was CPU-safe: static file/shape introspection, lint, and unit tests.

## Recall Evidence

| Surface searched | Query or source | Found beyond charter | Plan effect |
|---|---|---|---|
| Governing files | `mx1f_prompt.md`, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | Live board keeps the own-vehicle pointer at `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; MAIN owns the Metal re-probe; commit must use serializer and Python review marks. | Kept this scorer-free, Metal-free, and pointer-honest. |
| Memory registry | `rg -n "mx1|mx1b|mx1e|mx1f|RR11|ARM-CAP|set_wired_limit|set_memory_limit|subagent_commit_serializer|review_tracker" /Users/adpena/.codex/memories/MEMORY.md` | No mx1f-specific memory hit. Registry reinforced serializer/review gate discipline and exact axis separation. | Planned review marks x2 and serializer hashes; no score promotion. |
| MX1 research corpus | `rg -n "mx1f|mx1e|mx1b|mem-probe|memory spike|subset-before|load-phase|software cap|set_wired_limit|RR11-F1|RR9-F1|custom_grouped_backward|semantic renderer" .omx/research ... .omx/tmp/codex_runs` | mx1b fixed full-cache retention and measured CPU retained peak 1.27 GiB; mx1e added software cap plus `set_wired_limit`; RR9 registered mem-probe-before-fire; RR11 fixed ticket immutability outside probe mode. | Treated cache loading as bounded-but-not-sufficient; preserved probe-only ticket writing; added log lines that survive a mid-stage kill. |
| Canonical equations | `.venv/bin/python tools/list_canonical_equations.py --json | rg -n "mx1|mem_probe|memory|RR9|software"` | Found `ddm_rr9_mem_probe_fire_protocol_v1`: a safe_run projection is not a load-stage receipt. | Kept mem-probe receipt semantics and did not treat static projections as Metal clearance. |
| Current git history | `git log --oneline -15` | HEAD includes `0d2b27413d` RR11 ticket immutability and `a70626bd5e` mx1e software cap. | Did not reopen those solved surfaces; only filled mx1f residual allocator/logging gap. |

## Static Bisection

Definitions: ARM-CAP uses `pairs=32`, `H=384`, `W=512`, renderer width `C=96`, fp32=4 bytes, int64=8 bytes, int32=4 bytes. File sizes were read from the live files with `stat`. Tensor dense-byte arithmetic was recomputed directly.

| Order | Allocation site | Current line(s) | ARM-CAP n32 derived bytes | n600/default derived bytes | Verdict |
|---:|---|---|---:|---:|---|
| 1 | PR130 init checkpoint torch load | 1137-1138 | 283,432 B file; 265,356 B state tensors | same | Not a 65 GiB allocator. |
| 2 | Input label cache deserialize, select, `.long().clone()` | 466-470, 497 | ARM-CAP input==target, so one transient full cache: 0.878906 GiB; selected int64 clone: 0.046875 GiB | full cache per file: 0.878906 GiB | Bounded by mx1b pattern: select, clone, delete full cache before MLX/scorer setup. |
| 3 | Target label cache deserialize, select, `.long().clone()` | 500-505 | ARM-CAP skips this because input==target; ARM-VEH sequential transient another 0.878906 GiB | two retained full caches would be 1.757812 GiB, but current code does not retain both full caches | Not the third OOM allocator. |
| 4 | Torch selected tensors to NumPy int32 | 509-515 | two n32 int32 arrays: 0.046875 GiB total | two n600 int32 arrays: 0.878906 GiB total if full population selected | Subset-before-MLX holds; not enough for 65 GiB. |
| 5 | MLX import/device and memory-limit setup | 1156-1177 | unknown runtime overhead; now logged before/after | same | Instrumented, not statically dominant from tensor shapes. |
| 6 | Renderer object + PR130 weight conversion | 1182-1201 | checkpoint state tensors 0.000247 GiB; model params same order | same | Not a 65 GiB allocator. |
| 7 | Selected token MLX conversion | 1203-1247 | old full n32 conversion: 0.046875 GiB for two token arrays; new GPU path: lazy n4 chunks, 0.005859 GiB per two token arrays | old full n600 conversion: 0.878906 GiB for two token arrays | Bounded by lazy per-batch conversion. |
| 8 | SegNet torch load + MLX adapter weights | 1249-1255 | frozen weight file 38,502,892 B = 0.035856 GiB, plus adapter arrays | same | Not enough alone; now has before/after emitted checkpoints. |
| 9 | First full-batch renderer/scorer reverse-mode graph | 1311-1323 | renderer lower bound: token embed 2.25 GiB + coord/cat/coord-mix and four 96-plane block intermediates = 40.59375 GiB before SegNet/backward/workspaces/cache | n600 impossible shape would put token embed alone at 42.1875 GiB | Culprit class. Explains ~65 GiB with SegNet and reverse-mode overhead. GPU no longer defaults to this branch. |
| 10 | First chunked renderer/scorer reverse-mode graph | 1324-1381 | n4 renderer lower bound: 5.07421875 GiB before SegNet/backward/workspaces/cache | n600 is processed as serial chunks if explicitly selected with GPU defaults | Bound path. Same selected rows, weighted serial gradient accumulation, one optimizer update. |
| 11 | Eval d_seg pass | 1394-1439 | old full n32 eval includes scorer activations; new GPU chunked eval walks n4 chunks and sums mismatch counts | n600 chunked if selected | Bounded to avoid a second full-batch scorer graph. |

Subset-before-materialize check: `pair_ids` is selected before cache load at `experiments/ddm_mx1_pr130_semantic_renderer.py:1136`; `_load_selected_token_arrays` runs before `require_mlx(device=args.device)` at `:1156`; `_load_selected_seg_tokens` clones only indexed rows and deletes the full cache at `:480-484`. I did not find a current path that converts the full 600-pair cache to MLX before subsetting.

## Instrumentation Landed

- `LoadPhaseMemoryProbe(emit_log_lines=True)` is default-on for `--mode mem-probe`.
- Every probe sample emits one flushed stderr line prefixed `[mx1-load-phase]` with schema `ddm_mx1_load_phase_checkpoint.v1`, stage, RSS, system available memory, and MLX active/cache/peak when available.
- `_mx_eval_setup_barrier` now emits/checks a `before_*` sample before `mx.eval`, so a crash inside an allocating barrier leaves the preceding allocator name in `run.log`.
- The setup path now samples before/after checkpoint load, cache select/clone, `require_mlx`, memory-limit setup, model init, optimizer init, weight conversion, token conversion plan, SegNet torch load, SegNet MLX conversion, and first train/eval chunk allocators.

## Code Changes

- `experiments/ddm_mx1_pr130_semantic_renderer.py`
  - Added flushed load-phase sample lines.
  - Added `_derive_train_microbatch_pairs`, `_iter_pair_chunks`, `_mlx_token_chunk`, and weighted gradient-tree accumulation.
  - GPU defaults to n4 serial gradient accumulation; CPU defaults to full batch; `--microbatch-pairs` can override.
  - Train and eval paths avoid constructing full n32 MLX token/scorer graphs on GPU.
- `experiments/tests/test_ddm_mx1_memory_probe.py`
  - Added CPU-safe emitted-line regression.
  - Added GPU default microbatch policy test.
  - Added chunk coverage/equivalence test for selected token arrays.

Note: concurrent/pre-existing same-file hunks around safe_run status receipts and the RR11/RR12 ticket immutability regression were present in the working tree during this arm. They were preserved and the focused tests pass with them. I did not revert or rewrite them.

## Verification

```
.venv/bin/python -m ruff check experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py
# All checks passed

.venv/bin/python -m pytest experiments/tests/test_ddm_mx1_memory_probe.py
# 17 passed in 0.56s

git diff --check -- experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py
# clean
```

Post-edit SHA-256:

```
ed070e47a9a1a4a9fcdd8c8d993c236619262c069ea35c293343f5d4af07f8ca  experiments/ddm_mx1_pr130_semantic_renderer.py
e735b9344e0ce975d3842e7e5ffb1c5c5f5eb98ebbae0481a1ecfdab909f8c00  experiments/tests/test_ddm_mx1_memory_probe.py
```

## Boundaries

- I did not run `--device gpu`, Metal, or any scorer job.
- I did not regenerate or touch an existing launch ticket artifact.
- This does not move the own-vehicle or contest frontier.
- The exact Metal peak after chunking remains owed to MAIN's re-probe.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer `0.19108` remains borrowed/unmoved.
