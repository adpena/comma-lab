# DDM AI1 — ANS receiver integration and temporal closure

Date: 2026-08-09/10

Authority: `[macOS-CPU advisory; upstream AV GT; immutable evaluate.py; n600]`

Score authority: `score_claim=false`; promotion eligible: false

## Result

| candidate | archive B | d_seg | d_pose | rate contribution | S | evidence |
|---|---:|---:|---:|---:|---:|---|
| shipped Range comparator | 191,052 | 0.0004273478116374463 | 0.0001591133332112804 | 0.1272136847118971 | 0.20983747854972035 | DERIVED from the byte-identical scored raw; no separate evaluator invocation |
| ANS control | 188,932 | 0.0004273478116374463 | 0.0001591133332112804 | 0.12580206373127809 | **0.2084258575691013** | MEASURED exact evaluator row |
| ANS + `temporal_reversion` | **188,636** | 0.0004273478116374463 | 0.0001591133332112804 | **0.12560496948115393** | **0.20822876331897716** | MEASURED exact evaluator row |

The blocked bytes are now real shipping objects. The ANS control saves 2,120 B against the shipped Range archive. `temporal_reversion` saves another 296 B, for an exact total of 2,416 B. The temporal row improves the local same-raw ANS control by 0.00019709425012415238 score units.

This does **not** achieve the project goal. The best row remains above T1=0.19, does not move the borrowed contest pointer, and is not an own-vehicle result. The PR130 learned state and TM1 correction table are predecessor substrate; AI1's work is receiver, resumability, custody, deterministic runtime, and exact-evaluator closure.

## Shipping artifacts and deterministic decode

The final candidate store is `/Volumes/VertigoDataTier/pact/ddm_ai1_20260809/temporal_v3/`.

- Archive: `retained/temporal_reversion/archive.zip`, 188,636 B, SHA-256 `0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84`.
- Repeat archive: byte-identical, same size and SHA-256.
- Stored member: 188,536 B, SHA-256 `330c0a63e58002403e0a1d86bdcb13939b5768acc80c6cd599cf8e1dad7d5349`.
- ANS tokens: 114,528 B, SHA-256 `85d6c199ffb93ddab0fe1631448882a255e9fea1f6858bab5a04cea2310a7331`.
- Counted temporal sidecar: 39 B, SHA-256 `f920f7be8108b83831971a8d07c9ef522eadb18abed095cf395bf3a6f871e796`.
- Both independent cold decodes reconstructed 117,964,800 tokens, SHA-256 `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`, and both proved an empty ANS terminal state.
- Both raw outputs are 3,662,409,600 B, SHA-256 `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353`, exactly matching the ANS and Range controls.
- Cold `inflate.sh` wall times were 1,026.0432417920092 s and 824.6592473330093 s on `[macOS-CPU receiver decode; scorer-free]`; both are below 1,800 s.
- Determinism receipt: `/Volumes/VertigoDataTier/pact/ddm_ai1_20260809/temporal_v3/temporal_reversion_determinism_receipt.json`.
- Exact evaluator receipt: `/Volumes/VertigoDataTier/pact/ddm_ai1_20260809/temporal_v3/evaluation_temporal_reversion/evaluator_trace.json`.

The pure ANS control is independently closed under `/Volumes/VertigoDataTier/pact/ddm_ai1_20260809/resumable_v1/`: two full raw decodes were byte-identical, both were within 1,800 s, and the exact evaluator row is retained at `evaluation_ans_control/evaluator_trace.json`.

## Receiver and runtime closure

The receiver preserves the legacy Range path and uses the existing zero-byte outer ANS selector. The temporal extension parses only the exact counted `TM1P`/`TM1C`/`T1E1` raw sidecar, requires candidate id 6 and the measured 5x5 off-diagonal int8 geometry, and applies its causal t-1/t-2 correction to the exact int16 logit lattice before probability-table construction. Resume restores the last two decoded frames and binds checkpoints to the full model wire bytes, token bytes, codec, and archive member.

The initial temporal cold run materialized the correct full raw payload but timed out at 1,800 s inside the terminal mmap flush. Its bytes were not discarded: the raw was hashed, recorded, and moved to `temporal_v2/retained/temporal_reversion/failed_decode_a_timeout.raw`, with the same canonical raw SHA-256. The machine-readable incident receipt is `.omx/research/ddm_ai1_temporal_timeout_a_20260809.json`. The launcher now refuses to overwrite uncertified staging payloads and owns a process group so a timeout cannot leave an unlocked decoder grandchild.

The measured runtime cure streams CPU slave/master frame pairs in final raw order and fsyncs the sequential file. It preserves exact bytes and changed the two cold temporal runs from the I/O-fragile timeout path to 1,026.04 s and 824.66 s.

## Exact evaluator boundary

The temporal evaluator receipt binds pre/post-identical hashes for `archive.zip`, the 3.66 GB raw, immutable `upstream/evaluate.py`, its scorer dependencies, AV ground truth, the original video, and the names file. It used CPU, batch 16, seed 1234, two threads, and prefetch depth 4. The exact terms are:

- Seg contribution: `100 * d_seg = 0.04273478116374463`.
- Pose contribution: `sqrt(10 * d_pose) = 0.039889012674078614`.
- Compression ratio: `188636 / 37545489 = 0.005024198779246157`.
- Rate contribution: `25 * ratio = 0.12560496948115393`.
- Recomputed total: `0.20822876331897716`.

NOT MEASURED: Linux, contest CPU, contest CUDA, CUDA raw equality/runtime for this temporal receiver, or any promotion-eligible score. No Modal dispatch occurred. No contest or own-vehicle pointer moved.

## Verification and landing status

The focused regression suite passed 23/23. Ruff, Python compilation, `git diff --check`, and `tac.payload_retention_gate` passed. Every changed Python file received two review-tracker passes. The pure ANS receiver and duplicate-launch guard landed earlier as commits `46c7b85219` and `caa8eef4d8`.

The temporal source is complete but **UNCOMMITTED**. The required serializer failed before staging because this managed sandbox cannot write Git objects: `failed to insert into database` / `Operation not permitted`. The index remained empty; no direct-Git or review override bypass was used. Final working-tree SHA-256 values are:

- `experiments/ddm_ai1_ans_receiver_integration.py`: `788c55f61f4fb418fab991c5ba7847612675e75335eebb3597a8efda3acc4022`
- `experiments/tests/test_ddm_ai1_ans_receiver_integration.py`: `36d55026057518fbf1f12c652d0e2f649abfdb357dc83eed8c89290ff6cf7edc`
- `src/tac/pr130_runtime/dv1_cpu_runtime/inflate.py`: `e01325d65c42223d5e1ca8169f2bef0f62ae59bdcfeabf321e681fa2cd07d4e2`
- `src/tac/pr130_runtime/dv1_cpu_runtime/receiver.py`: `7dd29117a0cac30b32eb21bcc0e7ee6e1a45bf7f4af8f52ed5e94231945cc111`

## RECALL EVIDENCE

Sources searched included `.omx/research/`, `.omx/state/`, the canonical research index/DAG, task and lane stores, and the canonical-equation registry. Content queries included `ANS|ans_control|receiver_complete|token_codec|temporal_reversion`, `ddm_cx2|CX2|dv1_cpu_runtime|fx1_runtime_tree`, and `receiver_complete=false|research ANS control is not wired|ans_terminal_state_empty`.

Beyond the charter seeds, recall found:

- `ddm_dv1_20260809/DV1_RECEIPT.md` had already isolated the CPU-capable runtime and established the canonical raw SHA, so AI1 reused that receiver rather than forking another renderer.
- `ddm_fx2_20260809/FX2_RECEIPT.md` had already repaired the evaluator's three-argument shell contract, so the real wrapper—not a research decoder—became the integration surface.
- `ddm_cx2_20260809/CX2_FINDINGS.md` had already full-decoded ANS through this runtime, but only inside an SD1M composition; this changed the first action to a pure PR130 ANS control before temporal adjudication.
- `ddm_tm1_20260809/TM1_FINDINGS.md` supplied the exact causal correction grammar and retained sidecar/tokens, while explicitly leaving receiver closure and scoring blocked.
- `.omx/state/main_hot_state.md` still recorded ANS and temporal bytes as receiver-blocked; the current receipts close that bounded blocker but do not authorize a pointer edit.
- The canonical equation search found the generic static-packet byte-delta score law but no ANS-receiver-specific equation; exact archive and evaluator measurements remained mandatory.

This changed the plan from implementing a duplicate selector or another coder experiment to: reuse the landed selector/runtime, close the pure ANS control first, then integrate only the counted TM1 sidecar, prove two cold decodes, and score the exact archive.

## Follow-on dispositions

- **FIRED-AND-FOLDED** — owner: AI1; consumer store: `/Volumes/VertigoDataTier/pact/ddm_ai1_20260809/resumable_v1/`; pure ANS receiver closure, determinism, and local exact score are complete.
- **FIRED-AND-FOLDED** — owner: AI1; consumer store: `/Volumes/VertigoDataTier/pact/ddm_ai1_20260809/temporal_v3/`; temporal sidecar receiver closure, two cold decodes, and local exact score are complete.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: operator-authorized contest-row owner; consumer store: paired contest CPU/CUDA evaluator receipts keyed to archive SHA `0f5a797f...`; fire only after lane claim, Linux/runtime hash parity, and explicit operator GO.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: next Git-writable AI1 custodian; consumer store: repository history; fire when the mandated serializer can write `.git/objects`, then commit only the four listed source/test files plus this receipt and timeout receipt with post-edit hashes.
- **RETIRED at INSTANCE scope** — owner: MAIN rate routing; consumer store: this receipt; do not reopen memoryless coder swapping on the present HPAC model because measured remaining coder slack is only about 8 B. Route goal work to a new representation or token model instead.

Own-vehicle frontier remains `S=0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600`; this borrowed PR130-base receiver closure does not move it.
