# G29 specification — public LVPG2 decoder and auth-eval closure

Date: 2026-07-26  
Lane: `lane_g29_public_decoder_auth_closure_20260726`  
Status: implemented research-only compiler/preflight; official execution evidence open

## Objective

Build the smallest honest bridge from the counted `LVPG2` archive produced by
G25 to the frozen public contest evaluator.  The bridge has two deliberately
separate meanings:

1. a real, generic public decoder that converts the exact counted `LVPG2`
   member to the retained `LVLS1` state and invokes the generic LVLS1 renderer;
2. a typed authority-evidence protocol that can describe, but cannot itself
   manufacture, an exact public n600 evaluation.

The current landing completes the first meaning and a resumable static
compile/discovery/preflight path.  It does **not** complete the second meaning:
no official Linux evaluation, scorer observation mirror, double run, whole
GitHub test job, C0B seal, score, candidate, or frontier-pointer mutation was
performed.

## Frozen inputs and exact constants

- G25 archive SHA-256:
  `68351f57781d8fe60c05ab59fc250e48d6bb03e7cdf95b3d00987328d08d2a98`
  (`80,238` bytes).
- `LVPG2` member SHA-256:
  `cd2a55dc4725a9514abc9fd35dae45cce14b1ae9ea42c0c5c604271362e8defa`
  (`80,985` bytes).
- Approved upstream snapshot SHA-256:
  `d46d89155dbf0848e357858c8f62e12ef450a2914ef65814a4359ef6768d2d41`.
- Frozen official workflow SHA-256:
  `8a6cd6300b51a44f36b49774bc0c6100dbb37ef8290d42bf8e584f1dceddce56`.
- Official n600 geometry is derived from frozen `frame_utils.py`, not copied
  from prose: `1164 * 874 * (600 * 2) * 3 == 3,662,409,600` raw bytes.
- Official batch geometry is exactly 37 batches of 16 pairs plus one batch of
  8 pairs: 38 ordered batches and 600 ordered scorer cells.
- CPU authority means Linux x86_64, 4 vCPU, 16 GiB RAM.  CUDA authority means
  Linux x86_64 with T4 16 GiB VRAM and 26 GiB host RAM.  The axes remain
  separate; macOS CPU/MLX/MPS is advisory only.

## Public decoder contract

`taskspace_lvpg2_public_inverse.py` is a self-contained strict inverse for the
G25 wire.  The emitted public runtime has exactly three ordinary files:

```text
inflate.sh
inflate.py
lvls1_runtime.py
```

`inflate.py` installs its audit policy before opening the counted source or
running any inverse logic.  The policy permits the explicit counted source and
the exact generic runtime tree, but denies network, entropy, subprocess, and
ambient external reads during the inverse.  The emitted inverse performs a
strict parse, exact EOF/canonicality validation, deterministic inverse
transforms, and a `runpy.run_path` load of the retained generic LVLS1 renderer.
The shell disables Python bytecode writes and uses the public `${PYTHON:-python}`
selector that the execution receipt must bind to the interpreter actually
observed.

The compiler physically reopens the ZIP/member, parses `LVPG2`, materializes
`LVLS1`, reparses it, and compares the complete named base tensors,
`[600,2,32]` population code, and pose payload.  The bounded compile obtained:

- decoded-state SHA-256:
  `4b265f2e0e9280996d134a11335cf69b9eaaa0420671ff700f52ccc0f1801c85`;
- materialized-LVLS1 SHA-256:
  `ce3b2698aecb3f73602b6aae08576788b6e8574073ada40fe3a42e543065ec6f`;
- emitted runtime-tree SHA-256:
  `9ae4854b413bf0fd9047eb95a75831d4630763bb76cec1b9cb2402695ab52c45`.

This is real state inversion.  It is not yet proof that the full public render
produces the expected 3.66 GB raw output, nor proof of scorer equality.

## Counted/free ownership contract

Every declared video-derived weight, latent, selector, threshold, exception,
table, fitted coefficient, or other instance-specific operand is `COUNTED` and
must descend from the exact archive member.  Only typed generic VM facilities
and independently generic algorithms may live in free public code.  Static AST
audits reject obvious scorer/teacher/original-video access, dynamic loaders,
network facilities, subprocess facilities, oversized literals, hidden fitted
payloads, and unresolved third-party decoder dependencies.

The required `lineage_attested_generic` field is still testimony.  Syntax
inspection does not prove historical provenance or exclude data planted below
an otherwise hashed interpreter prefix.  Only governed trace custody plus
review can close that boundary.

## Typed authority evidence model

The model keeps declarative structure and measurement authority separate:

- `InterpreterDistributionABIClosureV1` binds the interpreter, installed
  distributions and files, native-linker evidence, and a complete prefix tree.
- `RuntimeDependencyDiscoveryReceiptV1` roots the real graph at
  `upstream/evaluate.sh`; the shell owns system unzip, the public decoder, and
  `evaluate.py`.  A fabricated `evaluate.py -> inflate.sh` process edge is
  forbidden.
- `PublicTraceClosureReceiptV1` retains raw trace paths, a reviewed
  run/scratch-root normalization, phase attribution, exact read ownership,
  exec ownership, network denial, and ABI-prefix custody.
- `OfficialWorkflowJobReceiptV1` describes the entire GitHub `test` job and
  its single 1,800-second envelope.  It is expressly an observation schema,
  not proof that checkout, LFS, setup, cache, apt, uv, evaluator, and upload
  steps were governed.
- `ScorerInputBatchLedgerV1` models all 38 ordered actual raw and preprocessed
  scorer input batches.
- `ScorerOutputCellLedgerV1` models all 600 ordered cells with Seg argmax
  hashes/mismatch counts/fp32 distance bits and Pose6 hashes/fp32 MSE bits.
- `ScorerOutputMirrorEquivalenceReceiptV1` keeps instrumented observation
  evidence separate from the unmodified official result and binds exact raw,
  preprocessed, and report equality.
- `OfficialEvaluationRunReceiptV1`, `PublicDecodeEqualityReceiptV1`, and
  `PublicEvaluatorExecutionReceiptV1` embed and reopen all parents, bind exact
  A/B observations, and carry both proof-context and semantic scorer IDs.
- `AuthEvalAdapterIngredientsV1` exposes reviewed ingredients for the sealed
  C0B adapter but never exposes or impersonates its private seal.

The semantic candidate-cell ID intentionally depends only on the versioned
domain and ordered `(pair_index, candidate Seg argmax hash, candidate Pose6
fp32 hash)` cells.  Archive, axis, run path, raw, mirror, and proof context
remain in the separate proof identity.  Thus G33 may compare the same
functional candidate quotient across execution contexts without laundering
those contexts into one authority receipt.

All caller-authored or canonical-JSON-reopened official-run, equality,
execution, and workflow observations are forced to `research_only=true`.
The mirror receipt instead requires
`instrumented_mirror_not_official_authority=true` and can feed only those
research-only parents.  The adapter factory rejects research-only execution
and equality.  External governed evidence collection plus independent sealed
C0B review are therefore mandatory; well-formed JSON cannot mint authority.

## Resumable runner and storage discipline

`run_taskspace_public_auth_eval_closure.py` is an append-only, stage-checkpointed
operator runner.  Its current stages are:

```text
COMPILE_PUBLIC_RUNTIME
DEPENDENCY_DISCOVERY
EXECUTION_PREFLIGHT
```

Each checkpoint binds its previous checkpoint, retained artifacts, cleanup
certification, blockers, stage ordinal, and research-only state.  Writes use
temporary-file plus hard-link publication and refuse overwrite drift.  Resume
reopens and validates the entire unique, contiguous parent chain and every
retained byte.  Completed compile/discovery stages are loaded, not rerun; a
blocked preflight may append a retry, and resume selects the latest blocked
attempt.  Compile artifacts are sufficient to resume when the original archive
and renderer paths are no longer available.

The bounded run lives on the preferred SSD tier at:

```text
/Volumes/VertigoDataTier/pact/g29_public_auth_closure_20260726_codex.pc0fLk
```

It contains only the small runtime, canonical receipts, checkpoints, and
lossless cleanup certifications.  No raw video tree or large scorer cache was
created.  The cleanup records certify source retention and no destructive
deletion.

## F0/F1 closure matrix

`CLOSED` below means only that the stated local/static proposition is proved.
It never upgrades an execution-dependent authority proposition.

| # | proposition | current status | exact remaining evidence |
|---:|---|---|---|
| 1 | frozen `evaluate.sh` ownership graph | **CLOSED (static)** | Governed execution of the graph remains row 17. |
| 2 | strict `LVPG2 -> LVLS1` parseback | **CLOSED (local)** | Full public raw render remains open. |
| 3 | declared counted/free placement | **CLOSED (schema/source audit)** | Historical/generic lineage and undeclared-read proof remain open. |
| 4 | exact three-file public runtime tree | **CLOSED (local)** | Same tree must be observed in each official run. |
| 5 | decoder-only interpreter ABI | **CLOSED (local)** | Axis-specific evaluator/package/native ABI is open. |
| 6 | approved upstream snapshot and no current bytecode | **CLOSED (static discovery)** | Isolated pre/post checks for A/B remain open. |
| 7 | complete Linux evaluator interpreter/package/native ABI | **OPEN** | Capture official CPU and CUDA lock-group environments separately. |
| 8 | complete phase-attributed read trace | **IMPLEMENTED TYPE; OPEN OBSERVATION** | Retained syscall/open/import/native trace for A/B. |
| 9 | exact process/exec ownership and network denial | **IMPLEMENTED TYPE; OPEN OBSERVATION** | Retained process tree and exec/network trace for A/B. |
| 10 | official whole GitHub `test` job under 1,800 seconds | **OPEN** | Governed workflow graph, log, trace, report, and monotonic timing. |
| 11 | actual 38-batch scorer-input mirror ledger | **OPEN** | Ordered frozen/mirror tensors from the executed axis. |
| 12 | actual n600 scorer-output-cell mirror ledger | **OPEN** | 600 ordered Seg/Pose cells and reviewed equivalence. |
| 13 | Linux CPU/CUDA contest hardware | **OPEN** | Independent CPU and T4 observations; no cross-axis inference. |
| 14 | fresh official run A | **OPEN** | Exact public evaluator execution and retained parents. |
| 15 | fresh official run B | **OPEN** | Independent cold workdir with the same governed inputs. |
| 16 | A/B raw, trace, scorer-cell, and report equality | **OPEN** | Canonical equality receipt from actual observations. |
| 17 | external governed evidence boundary | **OPEN** | Trusted collector must attest system facts; caller JSON is insufficient. |
| 18 | sealed C0B adapter review | **OPEN** | Independent review and private seal after all other rows close. |
| 19 | exact score and frontier promotion | **OPEN / UNMOVED** | Exact public bytes on authority hardware must produce a lower row. |

The five execution-level blockers remain the G31 classification: whole-job
timing, two fresh same-axis outputs, actual scorer ledgers, paired CPU/CUDA
evidence, and full-graph reuse identity.  The present schemas close none of
those observations.

## Acceptance for eventual authority

For each claimed axis, a governed producer must run two fresh official-semantic
jobs from empty directories under the frozen workflow/lock group, using the
exact archive and runtime.  It must retain the complete ABI and process/read
trace, prove every path has one legal owner, deny hidden targets and network,
bind the exact regular output set and 3,662,409,600-byte total, populate the 38
batch and 600 cell ledgers through a reviewed observation mirror, establish
mirror equivalence to the unmodified result, and finish the whole job within
1,800 seconds.  A/B raw, semantic-cell, component, report, graph, and normalized
trace equality must hold.  CPU and CUDA are then reviewed separately through
the sealed C0B adapter.

Until that sequence occurs, all artifacts remain `research_only`, no exact
score is named, and the canonical frontier pointer is unchanged.
