# G111 macro release-path adversarial review — 2026-07-27

## Verdict

The fresh G111 → G121 → G119 → G110 implementation chain now has a
receiver-packaging transducer, but it does **not** yet have a candidate row.
The current fresh v6 G111 full-n600 cold-boot/resume run is still active under
`/Volumes/VertigoDataTier/pact`; therefore no v6 terminal checkpoint, G121
retained-population manifest, G119 joint ledger, materialized submission,
clean public double-decode, or `upstream/evaluate.py` CPU/CUDA score exists.
The exact frontier pointer remains unmoved at the external competitive target
`0.172`. This unit is apparatus progress, not goal progress.

## Landed release closure

Two commits close concrete receiver/release bugs:

- `9a63e5bc23` (`G110: close clean public XIP2 runtime`)
  - removed the undeclared public Brotli dependency from the fresh path;
  - added the stdlib-only `delta_ar_zlib` XIP2 coder while preserving the
    historical Brotli coder ID without admitting it to fresh G119/G110 rows;
  - made public `inflate.py` create the official missing output directory and
    atomically replace the one exact existing raw output on a second decode;
  - retained strict runtime member, symlink, geometry, and storage checks.
- `2efce907226fe018324ff5b5f24baf2f259bc6b5`
  (`G110: atomically materialize receipt-closed releases`)
  - added
    `src/tac/witness_dsl/taskspace_g110_release_materializer_v1.py`;
  - added `tools/materialize_taskspace_g110_release_v1.py`;
  - added adversarial regression coverage in
    `tools/tests/test_taskspace_g110_release_materializer_v1.py`.

The materializer accepts only a caller-named nondominated G119 row. It does
not infer `BEST` or a cross-stage winner. Before publication it:

1. checks the physical G119 file SHA and body self-hash;
2. opens every row's post-G105 audit, candidate receipt/state, run receipt,
   checkpoint, config, G112 receipt, and target-capsule binding;
3. proves exact one-to-one coverage of the physically reopened G121 retained
   stages;
4. locally recomputes the Pareto set using rational Seg coordinates,
   decimalized Pose coordinates, and integer archive bytes;
5. recompiles the selected G110 archive and checks the actual archive payload
   length/SHA—not only compiler metadata—before any rename;
6. captures the six-file public runtime and a 531-file transitive in-repo
   source closure before and after compilation, requiring stable HEAD and
   byte-identical source/runtime snapshots;
7. requires an existing non-symlink SSD root with enough room for the full
   3.66 GB raw output plus headroom;
8. writes files with post-mode-change fsync, fsyncs nested directories, and
   atomically renames the complete submission directory.

The sealed receipt is deliberately false-authority fenced:

- `receiver_files_packaged=true`;
- `receiver_packaging_closed=false`;
- `clean_public_entrypoint_double_decode_run=false`;
- `g119_pose_axis_remeasured_during_release=false`;
- `upstream_evaluate_py_run=false`;
- `candidate_claim=false`, `score_claim=false`, and `pointer_moved=false`.

It also does not claim that filename allowlisting proves the runtime is generic
or free: charged/free source-boundary audit and generic-free eligibility remain
explicitly unclaimed.

## Adversarial findings closed

The independent review found and verified fixes for:

- missing Brotli in the clean upstream interpreter;
- the official evaluator not creating `inflated/`;
- in-memory archive bytes being discarded without an atomic writer;
- publication before checking the actual compiler payload SHA/length;
- a resealed truncated G119 ledger falsely asserting exhaustive G121 coverage;
- a forged Pose/rate axis changing the claimed Pareto set;
- source, Git HEAD, runtime, ledger, or custody drift during compilation;
- omitted package `__init__.py` execution and lazy G120-v2 source custody;
- a caller-selected arbitrary checkout differing from the executing checkout;
- creation of a fake local `/Volumes/...` tree when the SSD root is absent;
- false `receiver_packaging_closed` and `video_derived=false` claims;
- chmod and nested-directory durability gaps.

Independent terminal review result: PASS, with 8/8 focused tests, 70/70
relevant implementation tests, Ruff green, 531 source paths, zero omissions
from the live loaded in-repo module census, and the forged-axis exploit
failing closed. The combined G110/G119/refit/XIP2 suite passed 93 tests.

## Exact owed execution sequence

First identify and pin the runtime:

```bash
uv run python tools/materialize_taskspace_g110_release_v1.py runtime-id
```

After v6 terminates successfully, G121 physically opens every retained stage,
and G119 writes its exhaustive joint ledger, select one row explicitly from
the ledger's recomputed nondominated list. Do not infer a winner:

```bash
G119_LEDGER=/Volumes/VertigoDataTier/pact/<fresh-g119-run>/g119_post_g105_joint_axes.json
G119_LEDGER_FILE_SHA256="$(shasum -a 256 "$G119_LEDGER" | awk '{print $1}')"
G119_ROW_SHA256=<explicit-nondominated-joint-row-sha256>
RUNTIME_TREE_SHA256=21e8288e7ea9bf46527e8c68db7b08886f396edae6b3d510b2ba43127a9ec686
SUBMISSION_DIR=/Volumes/VertigoDataTier/pact/<fresh-g110-release>/submission

uv run python tools/materialize_taskspace_g110_release_v1.py materialize \
  --joint-ledger "$G119_LEDGER" \
  --expected-joint-ledger-file-sha256 "$G119_LEDGER_FILE_SHA256" \
  --joint-row-sha256 "$G119_ROW_SHA256" \
  --expected-runtime-tree-sha256 "$RUNTIME_TREE_SHA256" \
  --resume-from "$SUBMISSION_DIR"
```

On contest-equivalent hardware, run the exact public root twice. Each
invocation clean-extracts `archive.zip`; the second invocation regenerates and
atomically replaces the existing exact raw output. Record the archive SHA and
the raw SHA after each run:

```bash
VIDEO_NAMES=upstream/public_test_video_names.txt

bash upstream/evaluate.sh \
  --submission-dir "$SUBMISSION_DIR" \
  --video-names-file "$VIDEO_NAMES" \
  --device cpu
CPU_RAW_SHA256="$(shasum -a 256 "$SUBMISSION_DIR"/inflated/*.raw | awk '{print $1}')"

bash upstream/evaluate.sh \
  --submission-dir "$SUBMISSION_DIR" \
  --video-names-file "$VIDEO_NAMES" \
  --device cuda
CUDA_RAW_SHA256="$(shasum -a 256 "$SUBMISSION_DIR"/inflated/*.raw | awk '{print $1}')"

test "$CPU_RAW_SHA256" = "$CUDA_RAW_SHA256"
```

Only those exact report files, archive bytes/SHA, raw double-decode identity,
hardware axes, and scorer outputs may support a candidate or pointer update.

## Current blocker

Fresh producer evidence is absent, not failed: v6 is actively running after
commit `318c5bf698`. Until its terminal checkpoint and receipt exist, G121,
G119, materialization, double-decode, and CPU/CUDA evaluation must remain
unrun and unclaimed. No historical archive or payload may substitute.
