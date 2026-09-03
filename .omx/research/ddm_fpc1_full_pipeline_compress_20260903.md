# ddm_fpc1 full-pipeline compression handoff — 2026-09-03

## Verdict

**PARTIAL / BLOCKED_PORT_REQUIRED.** The implemented unit is a real, reusable pipeline substrate: explicit
device refusal, real-container auto-configuration through the contest YUV converter, atomic retained
stage boundaries, config-bound resume checks, and exact revalidation of the six retained AFR1 replay
payloads. It does **not** implement or claim the requested raw-video → trained fresh archive path.

The full path stopped before training because the named solve tools and shipped receiver do not expose
the contracts the charter requires. Chaining them as if they did would be a fake implementation. The
blocker is machine-readable at
`/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress/full/RESULT.json` (5,610 B, SHA-256
`e439784a2d4459c0c2abf952277173e0d03045330a313395e8ff124008cdace6`). No scorer ran,
no n600 job ran, no Modal job fired, and the exact frontier did not move.

Implementation:

- `experiments/semantic_joint_ctxmix_pipeline.py` — thin `--mode {replay,full}` / explicit-device CLI.
- `src/tac/semantic_pipeline/contracts.py` — storage preflight, exact device binding, real clip probe,
  atomic writes/copies, payload facts, and config/input/output-bound resume receipts.
- `src/tac/semantic_pipeline/pipeline.py` — typed replay graph, fail-closed full-mode boundary, and the
  conditional n600 fire order.
- `src/tac/tests/test_semantic_pipeline.py` — real 0.mkv probe, exact retained replay, resume identity,
  unavailable-device refusal, and the explicit full-mode refusal.

This follows the evidence-first, preserve-the-payload handoff discipline in
`docs/operating_manual_craft_handoff.md`. It deliberately reports an incomplete capstone rather than
turning a verified retained replay into a false fresh-rebuild claim.

## MEASURED rows

Axis for all rows below: **[macOS-CPU advisory / scorer-free exact byte measurement]**,
`score_claim=false`.

### Real input probe

The driver opened `upstream/videos/0.mkv`, counted the container through
`upstream/frame_utils.py::frame_count`, and decoded its sampled frame only through
`frame_utils.yuv420_to_rgb`:

| field | measured value |
|---|---:|
| frames | 1,200 |
| width × height | 1,164 × 874 |
| sequence length | 2, non-overlapping |
| pairs | 600 |
| trailing frames | 0 |
| first converted RGB frame SHA-256 | `3afa2e8f58a65805a1d2daacefd0af7781206fd97f00d3a4a5de6a2e8a3e0bff` |
| requested/tested device | CPU / available |
| CUDA availability here | false |
| MPS availability here | false |

### Replay payload revalidation

The retained G8S result was located at
`/Volumes/VertigoDataTier/pact/ddm_g8s_single_run_reproof/store_v2/RESULT.json`, SHA-256
`d958b9a7fdde554bef391551edad9964a7cc77d91469c35369b279240c10adf1`. The driver re-hashed each
source payload, retained a byte-identical copy under its own SSD store, and wrote one atomic receipt per
boundary. These seconds are copy/hash receipt times, **not** the lossless mechanism's compute time.
The composed FPC1 replay receipt is
`/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress/replay/RESULT.json` (11,090 B,
SHA-256 `6425ffaaa5d33838a6ac73e031e2b72d85a0acda46037e072fc8d852d6dbc55e`).

| stage | actual parent SHA-256 | retained output SHA-256 | bytes | device | seconds recorded | resumable | verified non-negotiables |
|---|---|---|---:|---|---:|---|---|
| FX5 | `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080` | `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841` | 180,386 | CPU | 0.005547 | yes | input/output pins, atomic retention, no score claim |
| DX2 | `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841` | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | 180,368 | CPU | 0.005374 | yes | same |
| GB1 pointer branch | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4` | 180,215 | CPU | 0.005570 | yes | same; non-consuming sibling typed |
| GB1 joint branch | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | `ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3` | 180,192 | CPU | 0.000564 | yes | same; actual DX2 parent restored |
| LB1 | `ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3` | `5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9` | 180,083 | CPU | 0.006487 | yes | same |
| AFR1 | `5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9` | `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` | 180,002 | CPU | 0.005213 | yes | same |

The graph is five sequential lossless transformations but six retained rows: both GB1 branches consume
DX2, and LB1 consumes `gb1_joint`; `gb1_pointer` is not LB1's parent. The delivered retained payload is
`/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress/replay/retained/archive.zip`, 180,002 B,
SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.

This run **revalidated and adopted the proven G8S payloads**. It did not spend another 4,140.9 seconds
re-executing `submissions/semantic_joint_ctxmix/compress.py`; the receipt records
`fresh_rebuild_executed=false` and `mechanism_reexecution=false`. Therefore it is an exact payload
regression gate, not a new independent lossless-tail reproduction.

## Full-mode boundary

The full-mode probe and CPU binding passed, then the driver wrote `BLOCKED_PORT_REQUIRED` and retained
no alleged candidate. The blockers are `INSTANCE` scope: these named scripts/current packet, not a
family-level claim.

1. **QS5 is an old-object executable, not a per-clip stage.** It binds `OUTPUT`, `QS4_RESULT`,
   `PAIR_COUNT=600`, CP135 bytes/pose, and QS4 fields at
   `experiments/ddm_qs5_resolve_compensation.py:44-79`; its only top-level arguments are `--output` and
   `--resume-from` at lines 1034-1037. There is no input archive, clip config, pair-scope, or device
   argument.
2. **The solve CLIs do not share the promised device contract.** FCD1 has runtime/output/publish
   arguments but no device argument (`experiments/ddm_fcd1_incompile_schur.py:409-434`); JG5 has a
   smoke `--limit` but hard n600 surfaces and no device argument (`...jg5...py:1402-1551`); UP2 exposes
   `--axis`, not `--device`, and maps into its existing fixed object (`...up2...py:1072-1091`).
3. **The shipped receiver rejects both halves of the required smoke.** Its top-level verifier pins only
   AFR1's exact SHA/size (`submissions/semantic_joint_ctxmix/inflate.py:18-39`), so a fresh archive is
   rejected, and it explicitly rejects a CPU host (`:56-60`).
4. **The lower F26 prefix route is internally unreachable.** It parses `F26_ADVISORY_PAIR_LIMIT`
   (`runtime/f26_inflate.py:279-289`), but lines 433-443 first reject every token decoder other than
   Python and then require native-hpac for a prefix. No value satisfies both predicates.
5. **The reusable trainer surfaces do not meet one three-device contract.** MX1 torch-smoke fixes
   `device=cpu` and saves only live weights/optimizer/scheduler (`ddm_mx1...py:1042-1133`). Its MLX
   trainer accepts MLX `cpu/gpu` (`src/tac/pr130_lift/mlx_semantic_renderer.py:92-109`); its EMA exists
   only when a controller policy is supplied (`ddm_mx1...py:2944-2958,3064-3083`).
6. **Fresh PR130 targets are a different formulation from the selected retained lineage.** The strict
   49-stage E2E graph uses one fresh DALI cache, while the historical selected semantic stage used
   AV-like targets and carrier/HPAC/token stages used DALI targets. The full-population fields differ.
   This is documented in `.omx/research/ddm_pr130_reproduce_20260809/RR1_PREPARE_VERIFY_AUDIT.md:140-188`.
   A future fresh run can be valid, but it cannot be called bit-identical reconstruction of the selected
   historical prefix.

No eval-roundtrip, EMA, differentiable YUV6, receiver identity, or advisory score was claimed as
verified for full mode because no training/scorer/receiver construction occurred. CUDA remains
**UNTESTED-HERE**; on this process both CUDA and MPS reported unavailable. `upstream/` and
`submissions/semantic_joint_ctxmix/` were not edited.

## Test transcript

Command:

```text
.venv/bin/python -m pytest src/tac/tests/test_semantic_pipeline.py -q
.....                                                                    [100%]
5 passed in 0.84s
```

Ruff passed on the new package, CLI, and test file. The five tests prove the substrate and its honest
refusal. They **do not satisfy the charter's requested full-mode n=2 receiver-closed smoke**: that test
is deliberately a fail-closed blocker assertion, not an xfail or a renamed end-to-end success.

## RECALL EVIDENCE

Searched beyond the charter's seed list:

- `rg -n -i "semantic_joint_ctxmix|full.pipeline|from.raw|QS5|AFR1|G8S"` over
  `.omx/research/CANONICAL_RESEARCH_INDEX*`, `sub015_DAG*`, task-ledger/state, memos, docs, experiments,
  and submissions.
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for `ema`, `cache`,
  `roundtrip`, `resume`, and `receiver`.
- `rg -n "add_argument|PAIR_COUNT|N_PAIRS|600"` over the actual FCD1, JG5, QS5, UP2, MX1, public
  compressor, and public receiver sources.
- Inventory of `/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809`,
  `/Volumes/VertigoDataTier/pact/ddm_pr130_encode_tokens_metal_20260809`, the PR130 credited repo,
  and G8S retained stage payloads.

Findings beyond the seeds, and what they changed:

- The credited repo is actually at
  `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo`; the charter's implied
  `ddm_pr130_train_20260809/repro_repo` location does not exist. The plan now pins the real source root.
- `experiments/ddm_pq2_compress_e2e.py` already types AFR1's GB1 fork and explicitly says
  content-redeciding candidates are outside a token-only replay grammar. This changed the replay from a
  false linear six-stage chain to two DX2 children and reinforced that a fresh content path needs real
  new adapters.
- The PR130 RR1 audit found the DALI/AV target-lineage confound above. This added a new blocker: choose
  and label the target formulation before any fresh training fire.
- Canonical equations `ema_decay_substrate_stage_aware_v1` / `ema_decay_run_geometry_v1` forbid a
  borrowed constant decay, and `scorer_input_cache_hash_identity_v1` requires archive/raw/pair/hash
  domain/tensor-shape binding before a cache can transfer. This changed the plan from wrapping MX1's
  optional EMA and retained cache to refusing until both become explicit pipeline contracts.
- The live board supersedes the stale common-contract frontier: AFR1 is our current exact
  contest-CUDA n600 frontier at 180,002 B and S 0.14797617125559104
  (`.omx/state/main_hot_state.md:6-18`).

## Conditional n600 fire order

Machine-readable ticket:
`/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress/full/governed_n600_launch_ticket.json`
(4,219 B, SHA-256 `880c4261ec5e778694feefc94a99f91b58ffdffd05ef81e641f15b94087c82ec`).
Disposition is **QUEUED-WITH-FIRE-ORDER**, owner `MAIN`, consumer store
`/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress`. Its exact argv is:

```text
.venv/bin/python tools/launch_detached_process.py \
  --output-dir /Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress/full/detached_n600 \
  --done-receipt ddm_fpc1_full_n600 -- \
  .venv/bin/python experiments/semantic_joint_ctxmix_pipeline.py \
  --mode full --device mps --video /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --store /Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress \
  --seed 20260903 --smoke-pairs 600 --smoke-steps 6000 \
  --verdict-batch-size 32 \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress/full
```

This order is **not fireable yet**. Fire only after all listed ports land, the real n=2 CPU
receiver-identity test passes, a config-matched memory-preflight receipt supplies the currently unknown
peak RSS projection, and MAIN owns the launch/scorer lane. The wall-clock field is a recalled bracket
(14,400–259,200 seconds) from the 49-stage PR130 recipe plus G8S's measured 4,140.9-second tail, not a
fresh FPC1 timing measurement. The full archive will not be bit-identical to AFR1 and needs a fresh T4
score.

## NEXT_IF_RESUMED

- **QUEUED-WITH-FIRE-ORDER** — owner: `fpc1 successor`; consumer store:
  `src/tac/semantic_pipeline/` plus `/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress`;
  action: extract per-clip, explicit-device stage adapters from FCD1/JG5/QS5/UP2 without changing their
  mechanisms, add derived EMA + exact-R/YUV6 training contracts, and add a fresh-archive CPU prefix
  receiver; fire trigger: the real 0.mkv n=2 end-to-end receiver-identity test passes with every payload
  retained.
- **QUEUED-WITH-FIRE-ORDER** — owner: `MAIN`; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_fpc1_full_pipeline_compress`; action: fire the detached n600 ticket;
  fire trigger: the prior port row passes, a fresh memory preflight supplies a numeric RSS projection,
  and MAIN holds the local Metal lane.
- **QUEUED-WITH-FIRE-ORDER** — owner: `MAIN`; consumer store: the exact fresh archive's governed T4
  evaluation store; action: run `upstream/evaluate.py` on the exact n600 archive and record components;
  fire trigger: the detached full run produces a receiver-closed archive and MAIN claims the unique
  full-scorer lane.

## LIVE-HYPOTHESES

- A faithful full pipeline remains buildable by extracting parameterized stage kernels from the named
  scripts, because their core functions already accept some runtime/codes/rows; the missing part is the
  orchestration boundary, not evidence that the mechanisms require hardcoded historical paths.
- A cheap real CPU prefix receiver remains plausible because F26 already carries a pair-limit parser,
  durable token checkpoints, and a parallel render path; the reachable implementation must either wire
  the exact free corrector into native-hpac or safely decode the full token field and render only the
  requested pairs.
- A fresh PR130-derived archive can still be valid even though it cannot reproduce the historical
  selected prefix bit-for-bit: the credited 49-stage graph begins at raw video and ends at an archive.
  It needs an explicit DALI-versus-AV formulation choice and fresh score, not a historical identity
  claim.
- One torch implementation can plausibly serve CPU/MPS/CUDA because the lifted semantic renderer is
  already PyTorch and the MPS scorer compatibility patch exists. The untested work is adding the
  config-derived EMA shadow, exact eval-roundtrip/YUV6 ordering, and device propagation through every
  solve adapter.

## DEAD-ENDS

- **Simple CLI chaining is closed for this instance.** QS5 and the other solve scripts do not expose the
  same archive/per-clip/device inputs; inventing flags would not execute their claimed work.
- **The shipped top-level receiver cannot validate a fresh CPU smoke.** It accepts only the exact AFR1
  SHA/size and requires CUDA.
- **Setting `F26_ADVISORY_PAIR_LIMIT=2` is closed on the current runtime.** Prefix mode requires
  native-hpac, while the runtime refuses native-hpac first.
- **Calling retained-payload adoption a fresh replay is closed.** FPC1 measured exact bytes and hashes,
  but did not re-execute the 4,140.9-second lossless mechanisms.
- **Calling a fresh single-DALI run the selected PR130 prefix reproduction is closed.** The retained
  semantic and carrier/HPAC/token stages used different target axes, and the axes differ at n600.

Own-vehicle frontier: **S 0.14797617125559104 @ 180,002 B [contest-CUDA T4, n600] — unchanged; this arm produced no new exact score.**
