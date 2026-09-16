# mrs2: identities pass; the one-C range decoder misses the timing screen

Local charter work is complete. The five-file receiver preserves move 53's raw output, including a working Python fallback, but the requested C range decoder is not the dominant runtime lever. The cold public host process took **1,342.013139 s [macOS-CPU advisory]**. Both prescribed T4 projections are about **1,699.1 s**, above the **1,200 s handoff screen** and **1,260 s risk ceiling**. **Do not dispatch this receiver to T4 under this charter.** This is an INSTANCE negative for this receiver/archive and projection screen, not a family verdict against native execution.

No new score was measured. The archive remains **179,286 B**, SHA-256 `aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957`. The exact frontier is unchanged by this arm. This receiver is research-only and non-promotable until MAIN closes its own gates.

## Measured profile and identities

Seed **20260916** selected two random pairs from each of 24 consecutive 25-pair strata: **48/600 pairs**, spread across the full population. The two implementations used the same retained pre-pair states and archive. Timers cover the selected pair work only; initialization, causal warm replay, checkpoints and raw writes are excluded. Each cell has one timing observation; no noise floor was measured.

| Stage | Python seconds | C-path seconds | Python share of pair work |
|---|---:|---:|---:|
| Arithmetic decoder + frequency table | 5.031967 | 0.371585 | 4.352% |
| Prior context statistics | 52.385812 | 53.905538 | 45.306% |
| Prior network | 42.773065 | 42.730146 | 36.992% |
| Renderer | 13.464085 | 13.329563 | 11.644% |
| Pose carrier + selector | 0.372326 | 0.369977 | 0.322% |
| Other token work | 1.599391 | 1.617714 | 1.383% |
| Total selected pair work | 115.626646 | 112.324523 | 100% |

All times are **[macOS-CPU advisory]**. Arithmetic became about 13.54 times faster in this observation, but it was only 4.35% of baseline pair work. The profile does not establish a stable end-to-end speedup. Context statistics and the prior network account for about 82.30% of baseline pair work. Source-bound evidence: `PROFILE_AND_PARITY.json`, plus the full retained profile results under `/Volumes/APDataStore/pact/ddm_mrs2/profile_python/` and `profile_native/`.

- **C versus Python: 48/48 pairs, 9,437,184 tokens, and 292,992,768 raw bytes identical.** The reusable `experiments/ddm_mrs2_audit/python_reference_equivalence_test.py` also reverified all retained artifacts.
- **Cold public C decode: 600/600 raw pair hashes identical** to the SHA-bound move-53/mrs1 reference chain. Full raw length **3,662,409,600 B**; SHA-256 `8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b`. The full raw was removed only after its identity/rebuild certificate; all sampled raw/token payloads and pre-pair states remain retained.
- **Compiler-present bare-venv smoke passed:** real six-line public shell, cold n600 decode, `--help` success, and expected default-CUDA refusal on this CPU host. The full process timing includes shell compilation, imports and raw writing, without receiver instrumentation.
- **Compiler-absent bare-venv smoke passed: 48/48 pairs and all 9,437,184 tokens identical.** Compiler absence means a controlled PATH with no compiler, not uninstalling the system compiler. The real shell failed compilation, removed the deliberately planted stale `.so`, and Python printed `Native range decoder unavailable; using the identical Python decoder.` This was a 48-pair fallback smoke, not a second cold n600 run.

Bare environment: Python **3.13.12**, NumPy **1.26.4**, Torch **2.12.1**, Brotli **1.2.0**; system/user site packages disabled. Python 3.11 grammar was checked, but execution on Python 3.11 and CUDA was not measured. APFS venv cleanup completed with a full installed-file/symlink certificate and retained rebuild wheels; `bootstrap/VENV_CLEANUP_DONE.json` confirms the environment is absent. Inherited cached wheel repacks are not claimed to have original upstream wheel hashes.

## Timing decision

| Basis | Ratio | Projected T4 seconds |
|---|---:|---:|
| Required mrs1 rounded ratio, 1,760.9 / 1,390.8 | 1.2661058383664079 | 1,699.130670 |
| Original measured receiver ratio, 1,185.899645171 / 936.6589231670368 | 1.2660954973462792 | 1,699.116793 |

Both multiply the measured cold mrs2 host process time **1,342.013138875016 s**. These are the same transfer assumption with different rounding, not independent calibrations or measured T4 results. `TIMING_PROJECTION_PLAN.json` binds the input receipts; `PUBLIC_NATIVE_VERIFICATION.json` binds the actual output and decision. MAIN owns any future T4 timing and fresh exact n600 row, after a successor passes the screen. Raw equality on macOS does not authorize transfer of the old contest score to the new receiver.

## Public source and reviewer budget

Exactly five public files exist in `submissions/mrs2/`:

| File | Physical LOC | Bytes |
|---|---:|---:|
| `inflate.py` | 2,715 | 138,152 |
| `inflate.sh` | 6 | 197 |
| `range_decoder.c` | 100 | 4,018 |
| `README.md` | 24 | 1,559 |
| `archive.zip` | binary | 179,286 |

`SOURCE_MANIFEST.json` supplies each SHA. The Python source SHA is `0689a77e061fe8a7d39953adb099c0503b8b9f2d983aacb7a98fc03d304e1425`. Relative to landed mrs1, 135 top-level definitions are AST-identical; the only changed class method is `ArithmeticDecoder.decode`, and `load_range_library` is the only added top-level definition. The remaining deltas are the ctypes import/selection, import-block deduplication, six-line compile shell, and the one README sentence. No prior, renderer, pose or payload change was made.

The native code implements only five-symbol frequency conversion and 63-bit range arithmetic. Its recurrence is reused from the sealed PR135-lineage decoder; no originality claim is made. A split multiply computes `floor(W*C/2^31)` with standard 64-bit unsigned integers, avoiding non-C11 wide-integer extensions. Strict `-std=c11 -Wall -Wextra -Werror -pedantic-errors` compilation passed with Apple clang 21.0.0 on arm64. Shipping flags remain exactly `-O2 -std=c11 -shared -fPIC`; no OpenMP, native-architecture flag or fast-math flag. Native code has no learned arrays, scorer data, or video-derived constants.

No public `.so`, `.pyc`, lab imports, internal identifiers, environment switches or dependency pins are present. `binary_source_audit.md`, `embedded_constants_audit.txt`, `archive_payload_manifest.json`, `SOURCE_SCOPE_AUDIT.json`, and `STATIC_PUBLIC_AUDIT.json` record the bounded audits. The development library was certified and externalized to `retained_build/range_decoder.strict.so`. All five changed/new Python sources received two actual review-tracker self-review passes without overrides. This is not an independent-reader claim.

Fresh-reader prompt: `.omx/research/ddm_mrs2_20260916/FRESH_READER_PROMPT.md`; SHA-bound request: `FRESH_READER_REQUEST.json`. **MAIN must spawn the reader and retain its reply.** The PR body draft is `PR_BODY_DRAFT.md`, exactly the inherited mrs1 draft, **20 lines**; no new timing claim was added and nothing was published.

## Custody, retention and boundaries

Durable bulk root: `/Volumes/APDataStore/pact/ddm_mrs2/`. `RETENTION_INDEX.json` there lists per-file bytes/SHA-256; `RETENTION_SUMMARY.json` records the final logical bytes, filesystem sidecar bytes and the **2 GiB** cap. The index explicitly excludes its own hash and the summary to avoid a recursive manifest. Source delivery is `/Volumes/APDataStore/pact/ddm_mrs2/mrs2_source_delivery.tar`, verified against the per-member manifest; `SOURCE_DELIVERY.json` binds its SHA and bytes. No proof depends on `/tmp`.

The serializer's exact argv, per-file post-edit hashes, log, return code and commit outcome are retained under `/Volumes/APDataStore/pact/ddm_mrs2/serializer/`; **`RESULT.json` is the landing authority, not the existence of the working-tree files.** The source tar also carries the serializer result. If the managed Git boundary produces rc 17/19, the verified tar is MAIN's explicit charter fallback; this arm does not call that a landed commit. The ignored archive is included in the delivery tar, not passed as a Git staging target. Shared ledger/state files are not swept into the owned-source commit; `CANONICAL_TASK_ROWS.json` preserves this arm's exact follow-on rows.

The original mrs1 tree and sealed runtime remain byte-identical across **55 source files**, and the closed PR tree has the same Git tree hash; see `BOUNDARY_VERIFICATION.json`. `upstream/`, the closed PR tree, sealed trees and the common contract's three protected paths were not edited. No Modal, fire, packet tool, PR, push, `authorize_*`, scorer, GT decode, MPS authority, stash, direct staged-index manipulation, or co-author/AI trailer was used. No scorer slot was claimed. Heavy jobs used detached launches and durable completion receipts. The source runners bind/resume saved stages, retain all sampled payloads, and certify cleanup; an interrupted public cold decode restarts that full-decode stage. APFS was used for the disposable venv; no SSD permission denial occurred. Unrelated dirty work was preserved.

## RECALL EVIDENCE

`RECALL_EVIDENCE.md` gives the queries and retained full-corpus results: research memo content, canonical equations (490 records), research index/DAG, design/SPEC files and task ledger. Beyond the charter seeds, `ddm_rc64p_native_cpu_decode_20260810.md` showed an ancestor's arithmetic loop was a tiny fraction of total time, requiring the actual mrs1 profile instead of assuming a native cure. `ddm_rr8_t4_wallclock_verdict_20260820.md` withdrew transfer of local stage seconds to T4 and distinguished entropy from corrector work; projections remain ranking evidence only. `ddm_rih1_receiver_import_hygiene_decision_packet_20260912.md` required a fresh exact row after receiver edits. Direct PR103 source inspection found a compiled `constriction` dependency, not a one-file shell-vendored C precedent. None of these ancestor measurements was substituted for this arm's measurements.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer `.omx/research/ddm_mrs2_20260916/MAIN_FIRE_ORDER.json`; trigger: harvest this completed handoff.** Verify serializer/source-tar custody and land only this arm's owned files, preserving shared work.
- **QUEUED-WITH-A-FIRE-ORDER; owner MAIN; same consumer; trigger: harvest local identity proofs and source custody.** Spawn the fresh reader from `FRESH_READER_PROMPT.md`, retaining the response verbatim.
- **QUEUED-WITH-A-FIRE-ORDER; owner MAIN; same consumer; trigger: harvest the failed timing screen.** Commission a bounded context-statistics runtime successor using the retained 48 states, unchanged archive, and same reviewer/identity budget.
- **FOLDED into that successor; owner MAIN; same consumer; trigger: a successor passes both <=1,200 s projections, full raw identity and reader review, then obtains the exclusive lane and required contract authorization.** Measure T4 decode and a fresh exact n600 row. This trigger is false for mrs2.

## LIVE-HYPOTHESES

- Prior-context computation is the most plausible next local runtime lever: it consumes 45.31% of Python pair work and remains about 53.91 s in the native profile. A bounded optimization might matter while preserving bytes and reviewer limits; that implementation and T4 benefit are untested.

## DEAD-ENDS

- **INSTANCE: one native range decoder as the complete timing cure.** Arithmetic falls from 5.03 to 0.37 s over the sampled work, but both full-process projections remain about 1,699 s. Further range-loop tuning alone cannot remove the dominant measured cost.
- **INSTANCE: dispatching this receiver on projections.** It fails the predeclared screen; no T4 fire is queued for the current receiver.
- **INFERENCE rejected: raw identity grants the old contest score to the edited receiver.** The new receiver has no fresh T4/exact row; its score remains unmeasured.

composition S 0.1361014714463198 @ 179,286 B [contest-CUDA T4 n600] (move 53)
