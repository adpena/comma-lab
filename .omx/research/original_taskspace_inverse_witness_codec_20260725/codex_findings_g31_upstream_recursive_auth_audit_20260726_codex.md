# G31 adversarial audit: frozen upstream recursive authority execution

Date: 2026-07-26  
Lane: `lane_g31_upstream_recursive_auth_audit_20260726`  
Mode: read-only/declarative; no decoder, scorer, exact evaluation, dispatch, archive
mutation, upstream mutation, or frontier-pointer mutation was performed.

## Verdict

**`experiments/contest_auth_eval.py` is not yet sufficient to confer exact
contest-CPU or contest-CUDA authority on a public task-space decoder.**

It has useful custody checks, but the present authority grade can be granted
without proving the official environment, official shell/extraction path,
recursive runtime/native dependency closure, a clean pinned evaluator, one
30-minute end-to-end envelope, or a byte-identical second decode. Its cache
reuse contract is materially weaker still: it does not bind the upstream
snapshot, evaluator support modules, model weights, video/name bytes, package
ABI, output manifest, or execution environment.

This is a **closure/verdict-scope finding**, not a negative on the task-space
inverse-witness family. No score was produced. The canonical pointer is
unchanged.

G29's final compiler/discovery/preflight source, runner, receipt schemas, tests,
spec, and findings landed during this audit and were independently re-read.
The acceptance contract remains the independent standard for that final delta.
A schema receipt is declarative evidence only; authority begins only when it is
populated from a real, isolated, same-axis double run and all blockers are
empty.

## 1. Frozen official execution graph

The actual public authority graph is:

```text
PR merge commit
  ├─ submissions/<name>/inflate.sh and any public PR runtime files
  └─ downloaded submissions/<name>/archive.zip
       │
       ▼
GitHub workflow (30-minute job)
  ├─ git-lfs pull
  ├─ uv sync --group cpu|cu128 from upstream/pyproject.toml + uv.lock
  ├─ apt ffmpeg and host shell/unzip/native runtime
  └─ uv run --group cpu|cu128 bash upstream/evaluate.sh
       ├─ unzip -o archive.zip -> submission/archive/
       ├─ bash public inflate.sh archive/ inflated/ video_names_file
       ├─ existence check for every expected .raw
       └─ python upstream/evaluate.py
            ├─ imports torch, tqdm, frame_utils, modules
            ├─ modules imports timm, einops, SMP, safetensors, Pillow
            ├─ loads posenet.safetensors + segnet.safetensors
            ├─ reads public_test_video_names.txt
            ├─ CPU: PyAV/FFmpeg GT decode + explicit BT.601 conversion
            ├─ CUDA: DALI/NVDEC GT decode and distributed env
            ├─ candidate: mmap exact .raw bytes
            ├─ zip(GT batches, candidate batches), normally 37x16 + 1x8
            └─ PoseNet/SegNet -> report.txt + stdout
```

Evidence:

- The workflow chooses `cpu` versus `cu128`, has a 30-minute whole-job timeout,
  performs `uv sync`, and invokes `uv run ... bash evaluate.sh`
  (`upstream/.github/workflows/eval.yml:27-33,56-89`).
- The shell uses system `unzip -o`, then the public `inflate.sh`, then
  `python evaluate.py` (`upstream/evaluate.sh:25-74`).
- `evaluate.py` imports its local support modules, selects DALI for CUDA and
  PyAV for CPU, loads both frozen scorer weights, pairs the two loaders with
  Python `zip`, accumulates sample count, and prints only rounded components
  (`upstream/evaluate.py:1-6,21-53,55-104`).
- The CPU and CUDA GT inputs are genuinely different code paths:
  DALI/NVDEC at `upstream/frame_utils.py:110-157`; PyAV plus explicit YUV420
  conversion at `upstream/frame_utils.py:159-216`. Candidate raws are mmaped
  at `upstream/frame_utils.py:218-253`.
- Pose preprocesses both frames to YUV6 while Seg uses only the last RGB frame;
  both resize through torch bilinear interpolation
  (`upstream/modules.py:61-84,103-113,130-158`).

The direct scored inputs are not merely `archive.zip`, `inflate.sh`, and
`evaluate.py`. They include every byte and execution choice reachable from the
graph above, including imported Python/native modules, loaded weights and
videos, system executables, environment, device, distributed topology, and
batch order.

## 2. Current frozen anchors and contamination state

Observed content anchors:

| Object | SHA-256 |
|---|---|
| upstream git commit | `11ad728f563d8970929e8947a1cf6124ee6303e4` |
| approved clean upstream snapshot | `d46d89155dbf0848e357858c8f62e12ef450a2914ef65814a4359ef6768d2d41` |
| `evaluate.py` | `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b` |
| `evaluate.sh` | `9612284ce6e9585aefcf636f3027808a56160ffd572edffdf4b8622a65fac917` |
| `frame_utils.py` | `d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90` |
| `modules.py` | `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa` |
| `models/posenet.safetensors` | `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576` |
| `models/segnet.safetensors` | `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6` |
| `pyproject.toml` | `8651cd684a38cbe5f477d6904ff10bf0c64a917c58dab8e14221e2cc5d879459` |
| `uv.lock` | `eca4542ad8d21354fd1f2bada74e8659329c0176b17f1ae808e04e023674231f` |
| `public_test_video_names.txt` | `7ff99d08c8351dd8167ec09213b758da5bbb705dedabe361ba881217374029a8` |
| `videos/0.mkv` | `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9` |

The nested upstream checkout is not git-clean: executable-bit changes are
reported on many tracked scripts/binaries and two historical submission
symlinks are missing. Current tree hashing ignores mode bits, so a content
digest does not prove executable-mode identity. The authority receipt must bind
both file content and relevant metadata, and it should pin the exact expected
digest rather than accepting whichever digest happens to be present.

### Bytecode contamination was moved correctly, then immediately regenerated

The durable manifest
`upstream_pycache_coldstore_manifest_20260726.json` has SHA-256
`681cd5310c22ce98195d335194f86c3951801dce6a892f6bfeaaca381f81b002`.
All ten listed cold-store files were independently re-read and matched both
recorded byte count and SHA-256. The manifest records a recoverable move and no
destructive deletion (lines 1-80).

That is valid cleanup custody, but **not prevention**. During this audit,
`upstream/__pycache__/frame_utils.cpython-313.pyc` and
`upstream/__pycache__/modules.cpython-313.pyc` reappeared at
2026-07-26 12:07:47 -0500. A source-only authority hash then failed exactly as
designed:

```text
ValueError: canonical authority snapshot cannot contain executable bytecode:
__pycache__/modules.cpython-313.pyc
```

The manifest's `python_bytecode_writes_disabled: true` field is therefore a
claim about its producing process, not a globally enforced invariant. The
canonical snapshot helper correctly treats bytecode as executable and rejects
it (`src/tac/contest_compliance.py:35-45,48-84,87-141`). Closure must:

1. operate on an isolated clean copy or immutable mount of the approved
   upstream tree;
2. set `PYTHONDONTWRITEBYTECODE=1`/`python -B` for every Python process,
   including subprocesses and tests;
3. assert no `.pyc`, `.pyo`, `__pycache__`, symlink, or unexpected executable
   exists immediately before and immediately after each run; and
4. fail closed on regeneration, never auto-delete it and continue.

## 3. Material defects in `experiments/contest_auth_eval.py`

### F0 — false official-environment authority

The official workflow installs the frozen upstream `cpu` or `cu128` lock group
and invokes the shell inside that environment
(`upstream/.github/workflows/eval.yml:27-33,73-89`;
`upstream/pyproject.toml:1-24`). The harness instead launches
`upstream/evaluate.py` directly with its own `sys.executable`
(`experiments/contest_auth_eval.py:1541-1582`).

The local harness interpreter observed here is CPython 3.13.12 with torch
2.12.1, torchvision 0.27.1, av 17.1.0, NumPy 1.26.4, timm 1.0.27,
safetensors 0.8.0, SMP 0.5.0, einops 0.8.2, Pillow 12.2.0, and tqdm 4.68.3.
The frozen project requires Python `~=3.11`; its lock selects materially
different packages for official axes. Linux plus device plus `n=600` is not
proof of ABI parity.

Nevertheless the authority gate grants `contest-CPU` from only
CPU + n600 + Linux x86_64, and `contest-CUDA` from device + n600 + Linux
x86_64 + an accepted GPU class
(`experiments/contest_auth_eval.py:1795-1914`). It does not require the official
Python, lock group, package wheel hashes, imported module origins, native
libraries, or system tool identities.

**Required:** run the frozen `uv.lock` group or fail closed; record Python
implementation/version/executable hash, `sys.path`, import origins and
distribution versions/hashes for the complete imported closure, and native
linkage/build identities for torch, CUDA/cuDNN, DALI/NVDEC, PyAV/FFmpeg,
NumPy/BLAS, torchvision, timm, SMP, safetensors, einops, Pillow, bash, unzip,
and every decoder/runtime binary.

### F0 — official shell and ZIP semantics are bypassed

The harness manually extracts with Python `zipfile`
(`experiments/contest_auth_eval.py:1098-1120`), separately calls public
`inflate.sh`, and directly invokes `evaluate.py`. The official route uses
system `unzip -o` inside `evaluate.sh` (`upstream/evaluate.sh:41-74`).

Python extraction does not preserve all Unix executable/symlink metadata the
same way as Info-ZIP and the harness's additional ZIP policy can reject a
contest-valid archive. Conversely, the official shell's behavior, its cwd,
PATH, umask, locale, shell version, and system utilities are not exercised.
Safety checks may be stricter, but they cannot be called a byte-for-byte
reproduction of a path they bypass.

**Required:** authority replay must invoke the frozen `evaluate.sh` under the
official locked environment, or prove a maintained equivalence theorem with
mutation tests for permissions, symlinks, filename encodings, duplicate
members, archive comments, and native executable dispatch. The safer and
smaller authority surface is the official shell.

### F0 found and corrected during audit — the sealed C0B graph required a false edge

The pre-audit `C0BAuthEvalClosureV1` required a `PROCESS_EXEC` edge
`upstream/evaluate.py -> inflate.sh` and roots its reachability walk at
`upstream/evaluate.py`.
That edge does not exist in the frozen evaluator. The actual process owner is
`upstream/evaluate.sh`: it executes `unzip`, executes `inflate.sh`, and only
after inflation executes `evaluate.py` (`upstream/evaluate.sh:41-74`).
`evaluate.py` never imports, opens, or executes the decoder.

A “logical” edge emitted solely to satisfy C0B would be a NO-FAKE graph claim:
the receipt says `PROCESS_EXEC` while no such process execution occurred.

Root corrected C0B during this audit. Current
`src/tac/witness_dsl/taskspace_selected_solution_compiler.py:1963-2079` now:

- owns `upstream/evaluate.sh` as the public root;
- requires real shell-to-inflate, shell-to-evaluator, inflate-to-runtime, and
  evaluator-to-support-module edges;
- roots reachability at `evaluate.sh`; and
- explicitly rejects a fabricated `evaluate.py -> inflate.sh` edge.

The regression at
`src/tac/witness_dsl/tests/test_taskspace_selected_solution_compiler.py:515-574`
constructs the real graph and verifies false-edge refusal. Root reports twelve
focused tests passed. This closes the false-edge blocker at the typed C0B
surface.

The complete observed receipt must still extend the typed minimum with:

```text
workflow/uv -> upstream/evaluate.sh
upstream/evaluate.sh -> system unzip
upstream/evaluate.sh -> public inflate.sh
public inflate.sh -> public inflate.py and recursive runtime
upstream/evaluate.sh -> upstream/evaluate.py
upstream/evaluate.py -> frame_utils.py/modules.py/weights/videos/packages
```

In particular, current C0B does not yet require `unzip`, bash, weights, videos,
packages, or native libraries as fixed minimum nodes; G29's discovery receipt
must supply them and the full closure remains blocked until it does.

### F0 — recursive decoder/runtime closure is neither complete nor fail-closed

The runtime tree includes only a short textual suffix allowlist
(`.c/.cc/.cpp/.env/.h/.hpp/.json/.py/.sh/.toml/.txt`) and skips caches
(`experiments/contest_auth_eval.py:74-97,188-210`). It therefore omits
extensionless executables, `.so`, `.dylib`, `.wasm`, `.safetensors`, `.pt`,
`.npy/.npz`, lockfiles such as `uv.lock`, archives, generated code, device
kernels, and executable metadata.

Its recursive import scanner recognizes only static `tac` imports and one
literal `importlib.import_module("tac...")` shape
(`experiments/contest_auth_eval.py:284-328`). It misses:

- non-`tac` repo-local imports and relative imports from the public runtime;
- aliased or computed `importlib`, `__import__`, plugin/entry-point loading;
- `exec`, `eval`, source loaders, `ctypes`/`dlopen`, JIT/kernel caches;
- `open`/mmap/Path reads and model/table/config/data loads;
- subprocesses, sourced shell files, PATH lookup, network downloads;
- dynamic branches selected by OS, CPU/GPU, environment, archive contents, or
  video names.

Parse errors and unresolved modules are recorded but not made blockers
(`experiments/contest_auth_eval.py:359-442`). Of frozen upstream inputs, the
runtime manifest explicitly adds only `evaluate.py`
(`experiments/contest_auth_eval.py:445-538`); it omits `frame_utils.py`,
`modules.py`, weights, videos, names, and the dependency lock from that
identity.

**Required:** combine fail-closed static discovery with observed Linux syscall
and process tracing across both fresh runs. Record every opened executable/file,
every import origin, every `execve`, shared object, environment-dependent path,
and network attempt. `unresolved_modules` and `parse_errors` must be empty.
Every observed path must map to exactly one declared owner:
`public_receiver_code`, `counted_archive_payload`, `frozen_upstream_scorer`,
`frozen_public_input`, `locked_package`, or `host_system_runtime`. Undeclared
paths and network access block authority.

### F0 — reuse accepts stale scorer/evaluator results

Reuse validates archive SHA/size, public `inflate.sh` SHA, device, the *path* of
the video-names file, and the incomplete live runtime manifest
(`experiments/contest_auth_eval.py:956-1082`). It does not compare:

- recorded versus current `upstream_snapshot_sha256`;
- expected approved upstream digest or upstream commit;
- `evaluate.sh`, `frame_utils.py`, `modules.py`, model weights, video bytes,
  or video-name content;
- lock group, Python/packages/native libraries/hardware/environment;
- output raw manifest, report/stdout hashes, execution timings, or workdir
  extraction manifest.

The unit test constructs a reusable result with no upstream snapshot field and
expects no blockers (`src/tac/tests/test_contest_auth_eval.py:704-780`), making
the omission executable behavior rather than merely absent documentation.

**Required:** disable reuse for authority until the full run identity and all
produced artifacts are bound. Any mutation to any graph node must invalidate
reuse.

### F0 — no two-run deterministic decode proof

The harness hashes one set of raws
(`experiments/contest_auth_eval.py:719-778`). It does not decode twice in fresh
directories, prove the exact output set, compare per-file/aggregate hashes, or
prove the decoder did not use an external target through a symlink/FIFO.
Existence and size alone do not prove ordinary owned regular files.

This is especially important for a decode-time optimizer/network: identical
generic code can still drift through RNG, thread scheduling, device kernels,
package ABI, caches, current time, network, or ambient files.

**Required:** two isolated cold starts on each claimed axis, same inputs and
controlled environment, exact expected regular-file set, no extras/symlinks/
FIFOs, exact size `3,662,409,600` bytes for the one n600 video, equal per-file
and aggregate raw hashes, and equal decoder discovery/observed-path manifests.

The value must be derived, never copied from prose:

```python
assert 1164 * 874 * 1200 * 3 == 3_662_409_600
```

`experiments/contest_auth_eval.py:1443,1453` currently has the wrong
`3,663,237,120` value in comments while the executable multiplication on line
1453 computes the correct `3,662,409,600`. There is no direct regression for
this constant in `src/tac/tests/test_contest_auth_eval.py`. The closure test
must derive width/height from frozen `frame_utils.camera_size`, derive frame
count from the frozen n600/`seq_len=2` contract, and assert the resulting
per-file size. A copied decimal constant is not acceptable.

### F0 — the official 30-minute budget is not enforced end-to-end

The official workflow has one 30-minute job envelope. The harness permits a
default 1800 seconds for inflate and another 1800 seconds for evaluation
(`experiments/contest_auth_eval.py:1391-1439,1541-1587`), excluding extraction,
environment setup, provenance, and hashing. It also permits longer overrides
without demoting authority.

**Required:** record and gate one end-to-end monotonic duration under 1800
seconds for the official evaluation job, with no authority escape hatch.

### F1 — provenance is selective, not the execution identity

The provenance records useful archive/script/upstream digests, device,
platform, a selected environment subset, torch/CUDA, GPU model/driver, ffmpeg,
and commits (`experiments/contest_auth_eval.py:602-692`). It omits the complete
environment and dependency/native closure described above. It also records the
current upstream digest but does not compare it with an approved expected
digest.

`inflate.sh` inherits the ambient environment and the harness defaults four
Python-selector variables to its own interpreter
(`experiments/contest_auth_eval.py:1391-1429`), which can make local behavior
different from the public workflow.

### F1 — report parsing manufactures false numeric precision

Upstream computes `score` from the full-precision in-process accumulators, but
exports Pose/Seg/rate to eight decimals and the final score to two decimals
(`upstream/evaluate.py:89-104`). The two-decimal final is the **official
displayed score**. The harness recomputes and labels a `canonical_score` from
the already-rounded eight-decimal components
(`experiments/contest_auth_eval.py:1700-1763`). That recomputation can be closer
to the hidden in-process score than the two-decimal display, but it is still not
the exact internal float and may differ slightly in either direction.

**Required:** call it a recomputation from rounded report components, not an
exact canonical score. Preserve the upstream two-decimal value as the official
displayed score. Exact machine-readable internal custody needs an upstream-owned
unrounded output surface or a byte-identical instrumented evaluator whose patch
is itself declared non-official/advisory until accepted.

### F1 — batch coverage and actual scorer inputs are inferred, not proved

The upstream `zip(dl_gt, dl_comp)` truncates at the shorter iterator
(`upstream/evaluate.py:67-83`). n600 plus raw size is good evidence but does not
prove ordered 38-batch identity or hash the actual tensors consumed after
CPU/CUDA GT decoding and preprocessing. The optional MLX cache artifact is a
custom compressed-raw preprocessing surface, not the actual upstream
GT-plus-candidate scorer input path
(`experiments/contest_auth_eval.py:781-810`).

**Required:** an authority receipt must carry the ordered per-batch ledger
(37 batches of 16 and one of 8 for this frozen evaluator), video/pair indices,
actual post-decode GT and candidate tensor hashes, and actual Pose/Seg
preprocessing hashes. This instrumentation cannot be confused with the
unmodified official result.

## 4. Decoder latitude, rate ownership, and forbidden hidden data

The contest explicitly permits external libraries/tools without charging their
generic code, but requires large artifacts—including neural networks and the
PoseNet/SegNet case—to be inside `archive.zip` and counted
(`upstream/README.md:96-120`).

The correct ownership split is:

| Surface | Potentially legal/rate treatment |
|---|---|
| Generic deterministic postfilter, optimizer, rasterizer, solver, network architecture, or interpreter in public `inflate.sh`/PR code | Potentially free code, subject to time and recursive closure |
| Generic constants derived without this video, scorer, GT, teacher, or hidden fitted state | Potentially free, with provenance |
| Video-specific learned weights, fitted coefficients, tables, masks, trajectories, latents, codebooks, or programs | Counted; must be inside exact `archive.zip` |
| State deterministically regenerated at decode time using only counted archive payload plus generic public algorithm | Potentially legal; output must be double-run deterministic |
| Fitting against the original video, SegNet/PoseNet weights or outputs, GT masks, teacher caches, upstream model files, or another video-specific external artifact during inflate | Forbidden/unpriced leakage; no authority |
| Hiding video-specific state as literals/generated source in the public PR receiver | Forbidden hide-data-in-code fake |

The official shell happens to run inside the upstream checkout, so a malicious
or accidental decoder can technically reach `upstream/videos`,
`upstream/models`, evaluator modules, and cwd-relative assets. Technical
reachability is not authorization. Rule 118 makes scorer neural artifacts
counted when used, and rule 119 permits original-video use for **compression**,
not free inflation-time access.

For a decode-time fitter to be admissible, the observed-path graph must prove:

1. all training/fitting targets descend only from counted archive members and
   generic public code;
2. there is no read/import/mmap/dlopen/network access to original video,
   scorer/teacher weights or outputs, GT-derived tables, caches, prior
   candidate outputs, or undeclared host state;
3. every video-specific bit is in `archive.zip` and included in its exact byte
   count;
4. RNG, iteration order, threading, device selection, package/native ABI, and
   stop condition are fixed and recorded; and
5. two fresh same-axis runs produce byte-identical raws within the single
   official time envelope.

The closure should explicitly deny reads from `upstream/videos`,
`upstream/models`, `upstream/evaluate.py`, `upstream/modules.py`, and scorer
artifacts while `inflate.sh` is running, except the video-names file passed as
an argument. It should treat the names as filenames only, hash their bytes, and
prove they are not used as a covert lookup key into hidden video-specific
state.

## 5. CPU/CUDA authority must remain two independent axes

The same archive does not imply the same evaluator inputs:

- CPU GT is decoded by PyAV/software FFmpeg and explicitly converted from
  YUV420 with bilinear chroma and BT.601 limited-range math.
- CUDA GT is decoded by DALI/NVDEC and can be affected by DALI, driver,
  distributed environment, GPU, CUDA/cuDNN, and torch kernels.
- Model interpolation/inference itself can diverge by device.
- A public decoder can also branch on CPU/CUDA or ambient hardware.

Therefore:

1. perform two fresh runs on Linux x86_64 official CPU and two fresh runs on
   T4 official CUDA;
2. bind the same archive, public receiver commit/code, frozen upstream identity,
   names, and intended algorithm on both axes;
3. require identical raw hashes across CPU/CUDA only if the decoder claims
   platform-independent output; otherwise record two output identities and
   never infer one score from the other;
4. record actual CPU/PyAV and CUDA/DALI GT tensor hashes and per-component
   score drift independently; and
5. never use macOS/MPS or one axis to fill missing custody on the other.

## 6. Exact double-run proof receipt

No public decoder is exact-authority-ready until one receipt contains all of
the following.

### A. Immutable input and ownership identity

- exact archive bytes, size, SHA-256, ZIP central/local member metadata and
  extracted member hashes;
- public PR/merge commit, `inflate.sh` hash/mode, and every recursively
  reachable public receiver byte with ownership class;
- approved upstream commit and expected snapshot digest, plus the graph-specific
  hashes in section 2;
- `pyproject.toml`, `uv.lock`, selected `cpu`/`cu128` group, Python executable,
  imports/distributions/native libraries/system tools and hashes/versions;
- video-name and source-video bytes, scorer weights, device/hardware/topology;
- complete controlled environment, argv, cwd, PATH, umask, locale, timezone,
  thread counts, seed/determinism flags, and network policy.

### B. Two fresh official-semantic decode executions per axis

- separate durable empty workdirs; no cached reuse;
- official system `unzip -o` and frozen `evaluate.sh`, or an independently
  proved exact equivalent;
- full process tree, exec/file/import/native-load/network trace with no
  undeclared path;
- pre/post clean-upstream guard including no regenerated bytecode;
- exit codes, stdout/stderr hashes, start/end/elapsed times;
- exact regular output file set, no extras/symlinks/FIFOs, exact byte counts
  derived by executable assertion
  `1164 * 874 * (600 * 2) * 3 == 3_662_409_600`;
- equal run-1/run-2 raw per-file and aggregate hashes;
- whole official job under 1800 seconds.

### C. Scorer consumption and result

- ordered 38-batch ledger totaling exactly 600 pairs;
- actual GT/candidate tensor and post-preprocess hashes at each batch;
- CPU/PyAV or CUDA/DALI path identity;
- exact report/stdout bytes and parsed components;
- same-axis repeated component/result equality, or an explicitly bounded
  non-authority floating tolerance if upstream itself is nondeterministic;
- score provenance must not claim more precision than upstream exposes.

### D. Cross-axis paired receipt

- links the two CPU runs and two CUDA runs to one archive/public receiver;
- compares decoder raw outputs, actual GT/scorer-input hashes, Seg/Pose/rate,
  runtime/package/native identities, and timing;
- preserves `[contest-CPU]` and `[contest-CUDA]` as separate claims.

### E. Mandatory mutation tests

Each of the following must invalidate closure/reuse or block execution:
mutation of any public runtime file, loaded data/model, evaluator/support file,
video/name byte, lock/package/native binary, Python/system executable,
environment-controlled branch, output raw/report, or ownership class;
unresolved/dynamic import; undeclared subprocess or network; symlink/FIFO/extra
output; wrong raw size/order/count; ZIP permission/symlink divergence; and
regenerated upstream bytecode.

## 7. Minimum acceptance blockers for G29/C0B

### Historical source-level disposition of the bounded G29 checkpoint

This subsection records the adversarial findings at the intermediate source
checkpoint and is retained as review history.  Section 8 is the final
classification after G29's source and documents stabilized.

Reviewed:
`src/tac/witness_dsl/taskspace_public_auth_eval_closure.py` through line 1938
and `src/tac/witness_dsl/taskspace_lvpg2_public_inverse.py`.

What is materially real and useful:

- The placement types force every video-derived object into counted archive
  bytes and allow only typed generic algorithms in public code
  (`taskspace_public_auth_eval_closure.py:294-452`).
- The emitted inverse is a real self-contained LVPG2 parser/decoder, invokes
  the retained LVLS1 renderer as a subprocess, and keeps fitted state in the
  counted packet (`taskspace_lvpg2_public_inverse.py:1-29,85-127,431-483`).
- Compilation actually reopens the exact ZIP/member, parses the population
  member, writes the public runtime atomically, runs the emitted inverse in a
  subprocess, reparses the materialized LVLS1, and compares full quantized
  state (`taskspace_public_auth_eval_closure.py:1307-1338,1398-1514`).
- Public `inflate.sh` disables Python bytecode writes and the compiler checks
  runtime bytecode before and after parseback
  (`taskspace_public_auth_eval_closure.py:1225-1262,1440-1471`).
- Static public-code audit blocks known dynamic/model loaders, scorer/teacher
  names, oversized literals, forbidden network/scorer imports, and unresolved
  third-party distributions
  (`taskspace_public_auth_eval_closure.py:668-847,1006-1039`).
- Discovery now uses the corrected `evaluate.sh` graph and binds hashes for the
  three public runtime files plus evaluator shell/source/support sources and
  bash/unzip/python executable/version identities
  (`taskspace_public_auth_eval_closure.py:1781-1938`).
- Compile receipts remain explicitly `research_only=true` with
  `public_n600_output_equality_owed=true`
  (`taskspace_public_auth_eval_closure.py:1082-1214`). This is honest scope.

Remaining source-level blockers:

1. `discover_public_runtime_dependencies` is static, not an observed recursive
   process trace. Its `observed_runtime_paths` is populated by copying the
   seven declared file paths, and its method string explicitly says a public
   process trace is still required (`:1797-1938`).
2. Frozen runtime files omit `posenet.safetensors`,
   `segnet.safetensors`, `public_test_video_names.txt`, `videos/0.mkv`,
   `pyproject.toml`, `uv.lock`, workflow/group identity, package modules,
   native libraries, and the archive itself. System tools are separate rows,
   not reachable byte-owned graph nodes.
3. The upstream snapshot is whatever clean digest is observed; it is not
   compared to the approved expected
   `d46d89155dbf0848e357858c8f62e12ef450a2914ef65814a4359ef6768d2d41`.
4. Distribution closure hashes `METADATA`, `RECORD` text, and only the top-level
   import origin. It does not verify every file named by RECORD, imported
   submodules, shared libraries, transitive native linkage, or the official
   upstream lock group (`:850-1039`). A modified installed package file can
   leave METADATA/RECORD hashes unchanged.
5. Static syntax/name auditing plus caller-provided
   `lineage_attested_generic` cannot prove absence of GT/scorer/teacher or
   hidden-data reads. Only the pending syscall/process/open-file trace and
   explicit deny policy can close that boundary.
6. The public shell resolves `${PYTHON:-python3}`, discovery records the PATH
   executable named `python`, and the compiler records `sys.executable`.
   These can be three different interpreters; execution must bind the one
   actually used.
7. The compile receipt's bytecode proof covers the public runtime, and
   discovery checks current upstream/runtime pre/post, but it cannot prevent a
   concurrent process from regenerating upstream bytecode after discovery.
   Execution must isolate the tree and check again after both runs.
8. Resource observation, official total 1800-second timing, exact regular
   output set, actual batch/scorer-input ledger, CPU/CUDA separation, sealed
   authority construction, and double-run equality were not yet implemented
   at this checkpoint.

At that intermediate checkpoint, all of these blockers remained true:

```text
g31_g29_final_files_not_yet_reviewed
official_uv_lock_group_not_executed
official_evaluate_sh_unzip_path_not_executed_or_proved_equivalent
recursive_observed_runtime_dependency_closure_not_proved
runtime_parse_or_unresolved_imports_not_fail_closed
native_package_and_system_executable_identity_not_closed
approved_upstream_digest_not_pinned_at_execution
upstream_bytecode_regeneration_not_prevented_and_postchecked
decoder_no_gt_scorer_teacher_hidden_data_access_not_proved
video_specific_state_archive_ownership_not_proved
single_1800s_end_to_end_budget_not_enforced
two_fresh_same_axis_raw_hashes_not_proved
actual_n600_batch_and_scorer_input_ledger_not_proved
cpu_cuda_paired_double_run_not_proved
cache_reuse_full_graph_identity_not_proved
```

Pointer delta: **none**. This audit produced a falsifiable authority contract,
not a score or promotion.

## 8. Final G29 adversarial classification (supersedes the checkpoint wording above)

This table classifies the fourteen substantive blockers from section 7.  A
`PARTIAL` row means useful static/compiler evidence landed but the authority
statement remains false until the named execution evidence exists.  No row is
promoted merely because a receipt dataclass can deserialize authority-shaped
JSON.

| # | blocker | status | exact disposition |
|---:|---|---|---|
| 1 | official uv lock group not executed | **PARTIAL** | Discovery hashes `.python-version`, `pyproject.toml`, `uv.lock`, and `uv`; the research-only whole-job packet names the frozen workflow.  No retained GitHub job proves `uv sync` and `uv run` selected `cpu` or `cu128`, the cache/install result, or the environment actually consumed. |
| 2 | official `evaluate.sh`/system-unzip path not executed or proved equivalent | **PARTIAL** | The corrected graph roots at `upstream/evaluate.sh`, includes bash/unzip plus its real dirname/mkdir/rm helpers, and rejects the false `evaluate.py -> inflate.sh` edge.  The emitted public decoder is real, but no official command has run. |
| 3 | recursive observed runtime dependency closure not proved | **PARTIAL** | Static rows cover evaluator sources, scorer inputs, locks, and the exact three-file decoder tree.  Trace schemas retain raw and normalized read sets, phase attribution, accounted external reads, and exact exec allowlists.  No production tracer has created those packets; the C0B-facing `observed_runtime_paths` remains the seven-file declared graph rather than the whole absolute read set. |
| 4 | runtime parse or unresolved imports not fail-closed | **PARTIAL** | The LVPG2 inverse really parses, materializes LVLS1, reparses, and compares the complete quantized state.  Source audits reject dangerous loaders and unresolved decoder distributions; evaluator ABI types require axis-specific roots.  Dynamic evaluator/native resolution remains unexecuted. |
| 5 | native package and system executable identity not closed | **PARTIAL** | Distribution rows hash installed files and native-linker output; evaluator ABI now requires CPU/CUDA roots and a complete interpreter-prefix tree; bash, unzip, uv, dirname, mkdir, rm, and Python are identified.  No official Linux axis ABI was captured.  Treating every byte below a captured interpreter prefix as ABI also does not independently prove that an attacker did not place video-derived data there before capture. |
| 6 | approved upstream digest not pinned at execution | **PARTIAL** | Static discovery pins `d46d89155dbf0848e357858c8f62e12ef450a2914ef65814a4359ef6768d2d41`, and the workflow schema pins the frozen `eval.yml` hash.  No governed run proves either identity was consumed for the full job. |
| 7 | upstream bytecode regeneration not prevented and postchecked | **PARTIAL** | Public shell exports `PYTHONDONTWRITEBYTECODE=1`; compile/discovery/readiness reject contamination.  There is no isolated official worktree plus post-run-A/post-run-B check or retained trace proving no bytecode was executed. |
| 8 | decoder no-GT/scorer/teacher/hidden-data access not proved | **PARTIAL** | Review moved the Python deny policy before archive read/inversion, requires the exact three-file runtime tree, and added phase-attributed OS-trace schemas.  No OS trace ran.  Reads anywhere under the fully hashed ABI prefix remain ownership-ambiguous: byte identity is not generic-versus-video lineage. |
| 9 | video-specific state archive ownership not proved | **PARTIAL** | Placement charges every declared derived weight/latent/selector/threshold/exception to the exact archive; source audits reject obvious hidden state; parseback is real.  `lineage_attested_generic` is still caller testimony, and no governed observed run proves absence of copied or undeclared state. |
| 10 | single 1800-second end-to-end budget not enforced | **OPEN** | A separate whole-GitHub-`test`-job schema now binds workflow/log/report/trace identities and must remain `research_only=true`, with `external_governed_custody_verified=false` and `whole_job_graph_closure_owed=true`.  No official job receipt or measured job exists. |
| 11 | two fresh same-axis raw hashes not proved | **OPEN** | Run, equality, and execution schemas now embed their strict parent bytes, normalize run-specific scratch roots, and carry input/output-ledger proof IDs.  They explicitly remain research-only.  Neither official A nor B exists. |
| 12 | actual n600 batch and scorer-input ledger not proved | **OPEN** | The source now models all 38 ordered input batches and all 600 ordered scorer output cells.  A reviewed observation mirror is kept distinct from the unmodified authority run; the output ledger exposes a context-free ordered candidate-cell semantic ID and a separate context-rich proof/content identity, both wired through run/equality/execution/adapter packets.  No mirror patch, raw capture, equivalence run, ledger bytes, or governed proof exists. |
| 13 | CPU/CUDA paired double-run not proved | **OPEN** | There is only a single-axis equality object.  No two CPU plus two CUDA runs, no paired cross-axis receipt, and no GT-path comparison exist.  Review caused official-run receipts to require Linux x86_64 and axis-specific contest-hardware profile/memory fields, while macOS readiness now emits an explicit blocker; those are schema gates, not measurements. |
| 14 | cache reuse full-graph identity not proved | **OPEN** | The new path does not reuse the old result cache, and the stale in-process distribution-tree cache was removed.  No durable authority-result reuse predicate binds archive, upstream/workflow, authority inputs, lock group, evaluator ABI, system tools, trace policy, hardware, scorer cells, outputs, and report. |

### Important source corrections incorporated during review

- The old C0B graph bug was found and fixed: `evaluate.sh` is now the common
  parent of the decoder and evaluator, and the fabricated
  `evaluate.py -> inflate.sh` edge is explicitly rejected.
- `EXPECTED_RAW_NBYTES` is now the executable product
  `1164 * 874 * 1200 * 3 == 3_662_409_600`; do not revive the older incorrect
  comment value.
- The source now pins the approved upstream snapshot rather than accepting any
  currently clean digest.
- Installed-distribution closure improved from metadata-only to hashes over
  installed files plus native-linker output.  The evaluator path now requires
  the axis-specific import roots and a complete interpreter-prefix hash; that is
  byte custody, not independent generic-lineage proof for every file under the
  prefix.
- The earlier adjacent-cent score check was corrected to intersect the
  two-decimal display interval with intervals induced by the evaluator's
  eight-decimal component reports.
- The inverse now installs its deny policy before reading/inverting the counted
  member, uses `runpy.run_path` for the renderer, and reports the dependency as
  `DYNAMIC_LOAD`, not process exec.
- Exact system-exec closure now includes dirname, mkdir, and rm as used by the
  frozen shell path.
- A/B trace comparison retains raw absolute paths but compares a reviewed
  normalization over run/scratch roots.
- The scorer evidence model separates a clean frozen authority run from an
  instrumented observation mirror and separates pure candidate-cell semantic
  identity from context-rich proof-ledger identity.
- Most importantly, caller-authored or reopened official-run, A/B-equality,
  public-execution, and whole-workflow packets are forced to remain
  `research_only=true`.  The readiness receipt explicitly owes an
  `EXTERNAL_GOVERNED_EXECUTION_EVIDENCE_BOUNDARY`; G29 cannot mint authority.
- The runner now retains source-audit, placement, ABI, runtime, discovery, and
  readiness parents and reopens completed stages rather than recomputing them.
  Tests cover source-free compile resume and repeated blocked-readiness retry
  history selection.  This is still not proof of crash recovery through an
  actual official execution stage.

### Why the final execution-shaped types are not evidence yet

`OfficialEvaluationRunReceiptV1`,
`PublicTraceClosureReceiptV1`,
`PublicDecodeEqualityReceiptV1`, and
`PublicEvaluatorExecutionReceiptV1` are useful schemas.  They do not constitute
a run:

1. no production runner constructs them from retained system observations;
2. strict reopen proves canonical parse/re-emit identity, not that JSON fields
   came from the system;
3. the improved execution receipt embeds run A/B, equality, trace,
   evaluator-ABI, scorer-input, scorer-output, observation-mirror, and
   whole-workflow parents and re-derives their internal identities, but none
   came from a production runner;
4. all caller/reopen derivations now correctly remain research-only, so the
   external governed custody boundary and private C0B review remain owed; and
5. C0B receives a static graph subset as `observed_runtime_paths`; the full
   absolute read set is available only inside the still-unexecuted embedded
   trace packets.

The honest artifact is therefore a **research-only compiler/readiness
checkpoint**, not an AuthEvalClosure, contest row, or promotion receipt.

Final pointer delta: **none**.

## 9. Bounded validation of the landed source checkpoint

Independent validation deliberately did not run `upstream/evaluate.py`, a
scorer, full raw rendering, CUDA, or a paid job.

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider \
  src/tac/witness_dsl/tests/test_taskspace_public_auth_eval_closure.py \
  tools/tests/test_run_taskspace_public_auth_eval_closure.py \
  src/tac/witness_dsl/tests/test_taskspace_selected_solution_compiler.py

43 passed in 0.99s
```

The first independent collection attempt correctly caught a nonexistent parser
import in the new test; after repair, the test also corrected its score-rounding
fixture from a guessed decimal to the actually rederived `0.08`/`0.09`
boundary.  Focused Ruff lint passed on all seven reviewed source/test/tool files.
Ruff format check passed on the five G29-owned files; the concurrently owned
selected-solution compiler remains outside this audit's formatting ownership.

The durable preflight run was independently reopened from
`/Volumes/VertigoDataTier/pact/g29_public_auth_closure_20260726_codex.pc0fLk`.
Its compile, discovery, and readiness receipt hashes are respectively
`521e5f8a23bd90b49d5bb2368c6e0b378c361e7612bb89d7910ecb0842f6265c`,
`a2ed453130ad8d555d464e314601be5c29f6d357a8cc38ca2b3159b4dbc55bc3`,
and
`4b1d4d78e9e6f27a78d6cb0a6a0b707adefa943735c9f16ebe7175f45557b531`.
The retained resume chain reverified successfully.  Its terminal readiness
state is still exactly `ready_to_execute=false`,
`auth_closure_proven=false`, and `research_only=true`.

What these tests prove:

- exact raw byte count is derived from the frozen `frame_utils.py` dimensions;
- typed placement refuses uncounted declared instance-derived state;
- the real retained LVPG2 member inverts to byte-equivalent full quantized
  LVLS1 state;
- the emitted public inverse is actually invoked for the bounded LVLS1
  parseback conformance path;
- corrected evaluator graph and false-edge rejection work;
- public authority-shaped receipt constructors are sealed;
- adjacent-cent score text, decoder-only evaluator ABI, and macOS authority
  readiness are rejected; and
- checkpoint packets canonicalize and bind artifact hashes;
- completed compile state can be reopened without the source archive/renderer
  and without rerunning the compiler; and
- repeated blocked-readiness histories select the latest retained attempt; and
- scorer input/output mirror packets round-trip, keep semantic candidate cells
  separate from proof context, and do not expose a misleading aggregate
  re-derivation.

What they do **not** prove:

- the monkeypatched minimal ABI in the real compile test is not package/native
  authority;
- no test executes strict dependency discovery against the full frozen
  weights/video/lock graph;
- no test interrupts a real official execution stage;
- no test constructs the complete official-run -> A/B equality -> public
  execution chain from real system observations;
- no test runs the reviewed scorer observation mirror or proves it equivalent
  to an unmodified official run;
- no mutation matrix covers loaded models, packages/native libraries, system
  tools, environment branches, trace paths, or outputs; and
- none of the five OPEN execution blockers in section 8 moved.

## 10. Final authority boundary and pointer honesty

The final G29 source is a materially better **research-only compiler,
dependency model, mirror/ledger schema, and resumable preflight checkpoint**.
It is not execution authority.  The source itself now says so:

- readiness owes the actual 38-batch ledger, n600 output-cell ledger, two
  official runs, governed execution evidence, official whole-job timing,
  reviewed mirror equivalence, and sealed C0B review
  (`taskspace_public_auth_eval_closure.py:4800-4808`,
  `taskspace_public_auth_eval_closure.py:4904-4912`);
- reopened observations, equality, execution, and whole-workflow packets cannot
  set `research_only=false`
  (`taskspace_public_auth_eval_closure.py:3565-3572`,
  `taskspace_public_auth_eval_closure.py:3956-3959`,
  `taskspace_public_auth_eval_closure.py:4244-4247`,
  `taskspace_public_auth_eval_closure.py:5338-5341`);
- the pure candidate semantic-cell identity is isolated from proof context
  (`taskspace_public_auth_eval_closure.py:3256-3274`) and its proof binding is
  carried separately through the official-run/equality/execution chain; and
- the operator runner stops at preflight and never runs the evaluator
  (`run_taskspace_public_auth_eval_closure.py:602`).

No `upstream/evaluate.py` invocation, scorer run, GitHub test job, CPU/CUDA
dispatch, C0B seal, exact score, or pointer mutation occurred in G31.

**Final blocker count: 0 CLOSED / 9 PARTIAL / 5 OPEN.**

**Final authority: NO-AUTHORITY.  Final pointer delta: NONE.**
