# G29 findings — public LVPG2 decoder and auth-eval closure

Date: 2026-07-26  
Lane: `lane_g29_public_decoder_auth_closure_20260726`  
Verdict: real public-state compiler; research-only auth preflight; no authority

## Outcome

G29 now has a real generic public decoder for the G25 `LVPG2` packet, a typed
recursive authority-evidence model, and an append-only resumable
compile/discovery/preflight runner.  The compiler reopened the real G25 archive,
ran the emitted inverse, reparsed the resulting `LVLS1`, and proved equality of
the complete quantized state.

The authority result is deliberately negative: the local host is macOS, the
official Linux evaluator ABI was not captured, and no external governed
execution occurred.  The runner exited `4` with
`ready_to_execute=false`, `auth_closure_proven=false`, and
`research_only=true`.  No raw n600 render, scorer, official job, CPU/CUDA
dispatch, exact score, candidate, C0B seal, or pointer mutation occurred.

## Durable bounded artifact

Run directory:

```text
/Volumes/VertigoDataTier/pact/g29_public_auth_closure_20260726_codex.pc0fLk
```

Key retained identities:

| artifact | SHA-256 |
|---|---|
| compile receipt | `521e5f8a23bd90b49d5bb2368c6e0b378c361e7612bb89d7910ecb0842f6265c` |
| counted/free placement | `8aea3f1120f944ef688ec642d29bef11259b1cc50f3c9438e0d1ab41ced75b98` |
| decoder ABI | `6aae1dece3bd3828b944ed57124c8312e24c5019ba801b5fa26879a20ff14532` |
| dependency discovery | `a2ed453130ad8d555d464e314601be5c29f6d357a8cc38ca2b3159b4dbc55bc3` |
| readiness | `4b1d4d78e9e6f27a78d6cb0a6a0b707adefa943735c9f16ebe7175f45557b531` |
| compile checkpoint | `4b61616046d807912cdd2b6cf2160055e34ee51dfe05d20dca92efe401b510f0` |
| discovery checkpoint | `05323abdd359b36840b3d16f5ddfb7f205647792d9a9e4c434289a442cc86814` |
| readiness checkpoint | `bb716de792639bda62b5e8a786c6ec433b02742570899b4e245298c4899d921d` |
| emitted `inflate.py` | `1f9aa79b692bd322df459d25a5b5eb5977c653bf1623df8b97095e629425f3d9` |
| emitted `inflate.sh` | `aa22ff47561501abf5f8228625794a8be119d7e98e097dbf71b208b4d3aa15ab` |
| retained `lvls1_runtime.py` | `4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224` |

The compiled source archive is
`68351f57781d8fe60c05ab59fc250e48d6bb03e7cdf95b3d00987328d08d2a98`
at 80,238 bytes.  Its member is
`cd2a55dc4725a9514abc9fd35dae45cce14b1ae9ea42c0c5c604271362e8defa`
at 80,985 bytes.  Exact inversion produced decoded-state identity
`4b265f2e0e9280996d134a11335cf69b9eaaa0420671ff700f52ccc0f1801c85`,
materialized LVLS1 identity
`ce3b2698aecb3f73602b6aae08576788b6e8574073ada40fe3a42e543065ec6f`,
and runtime-tree identity
`9ae4854b413bf0fd9047eb95a75831d4630763bb76cec1b9cb2402695ab52c45`.

## What is closed

1. The emitted inverse is real and self-contained; it does not delegate the
   `LVPG2` inverse back to repository source.
2. The audit policy is active before the counted member is read or inverted.
3. The runtime tree is exactly three regular, non-symlink files.
4. Compile parseback covers the complete named quantized state, not a proxy or
   section summary.
5. Declared video-derived weights, latents, selectors, thresholds, exceptions,
   tables, and fitted operands cannot be labeled free.
6. The static frozen evaluator graph has the correct owner:
   `evaluate.sh` executes both `inflate.sh` and `evaluate.py`; no false
   `evaluate.py -> inflate.sh` edge is admitted.
7. The approved upstream snapshot is pinned to
   `d46d89155dbf0848e357858c8f62e12ef450a2914ef65814a4359ef6768d2d41`.
8. The expected raw total is executable geometry,
   `1164 * 874 * 1200 * 3 == 3,662,409,600`, and the scorer ledger shapes are
   exactly 38 batches / 600 cells.
9. Caller-authored canonical receipt bytes cannot set execution authority;
   reopened run/equality/execution/workflow receipts remain research-only and
   cannot enter the C0B adapter.
10. Resume reopens retained completed parents without recomputing them.  A
    second invocation using the readiness checkpoint and nonexistent original
    archive/renderer paths exited with the same blocker state and left the
    entire artifact tree byte-identical.  Repeated blocked preflight retries
    append checkpoints and the latest attempt is selected.

## Exact current blockers

The readiness receipt records:

```text
AXIS_SPECIFIC_EVALUATOR_INTERPRETER_PACKAGE_NATIVE_ABI_CAPTURE_OWED
OFFICIAL_AUTHORITY_REQUIRES_LINUX_X86_64_CONTEST_HARDWARE
```

It separately retains these authority debts:

```text
ACTUAL_38_BATCH_SCORER_INPUT_LEDGER
ACTUAL_N600_SCORER_OUTPUT_CELL_LEDGER
DOUBLE_RUN_RAW_SCORER_CLOSURE_EQUALITY
EXACT_PUBLIC_N600_OFFICIAL_RUN_A
EXACT_PUBLIC_N600_OFFICIAL_RUN_B
EXTERNAL_GOVERNED_EXECUTION_EVIDENCE_BOUNDARY
OFFICIAL_GITHUB_TEST_JOB_30_MINUTE_WALL_RECEIPT
REVIEWED_SCORER_INPUT_OUTPUT_OBSERVATION_MIRROR_EQUIVALENCE
SEALED_C0B_AUTH_EVAL_ADAPTER_REVIEW
```

The complete disposition is:

| surface | disposition |
|---|---|
| static evaluator graph | closed locally |
| real LVPG2 inverse/full-state parseback | closed locally |
| declared counted/free placement | closed as schema/source audit; provenance still open |
| exact decoder runtime tree | closed locally |
| decoder ABI | closed locally |
| approved snapshot/current bytecode check | closed statically |
| official CPU/CUDA package/native ABI | open |
| actual read/import/native trace | open; type only |
| actual process/exec/network trace | open; type only |
| official whole-job 1,800-second receipt | open |
| actual 38-batch input ledger | open |
| actual n600 output-cell ledger | open |
| contest CPU and CUDA hardware | open independently |
| official run A | open |
| official run B | open |
| A/B raw/trace/scorer/report equality | open |
| external governed observation authority | open |
| sealed C0B review | open |
| exact score/frontier promotion | open; pointer unmoved |

The workflow packet type is only a research observation shape.  It explicitly
requires `external_governed_custody_verified=false`,
`whole_job_graph_closure_owed=true`, and `research_only=true`.  Checkout, git,
LFS, setup, cache, curl/apt, uv, ffmpeg, evaluator, and upload custody have not
been observed.  Similarly, the scorer input/output ledgers are types for a
reviewed instrumented mirror; no mirror patch or evidence bytes exist.

## Resumability and hygiene evidence

The runner publishes artifacts without overwrite, binds a unique contiguous
checkpoint chain, validates every retained byte on resume, loads completed
compile/discovery stages, and retains distinct cleanup certification at every
stage.  The real resume check intentionally supplied nonexistent original
inputs; success proved that the retained compile checkpoint, not ambient
sources or a silent rebuild, fed later stages.  A tree comparison before and
after resume was byte-identical.

The run remained on the preferred SSD tier and created no raw output or bulky
cache.  Cleanup certification records no destructive deletion.  Scratch hash
comparison data was removed after the proof; the durable evidence paths do not
depend on `/tmp`.

## Validation

The bounded suite is:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  src/tac/witness_dsl/tests/test_taskspace_public_auth_eval_closure.py \
  tools/tests/test_run_taskspace_public_auth_eval_closure.py \
  src/tac/witness_dsl/tests/test_taskspace_selected_solution_compiler.py
```

It covers the real retained G25 inverse/compile path, policy-before-inverse
denial, typed placement, exact raw geometry, frozen graph ownership, sealed
public constructors, schema/dataclass field alignment, scorer-ledger semantic
versus proof identity, canonical checkpoints, tamper/noncontiguous-chain
rejection, no-recompile resume without original inputs, repeated blocked retry,
and cleanup custody.  Ruff format and lint are required over the five G29-owned
source/test/tool files.

Final bounded result: `43 passed in 1.03s`; Ruff format check and Ruff lint both
passed over the five G29-owned files.

This validation does not execute the official evaluator and cannot close any
execution-dependent row above.

## Pointer honesty

G29 lowers no exact contest score.  It produces a real public compiler and a
strict authority barrier, both means toward an exact row.  The canonical
frontier pointer is unchanged, remains above the mission target, and no G29
artifact may be described as `AuthEvalClosure` until the governed Linux
double-run and C0B gates actually close.
