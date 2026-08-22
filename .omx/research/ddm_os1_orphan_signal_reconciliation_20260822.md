# ddm_os1 orphan-signal reconciliation — the live orphan class is arm-lifecycle Git custody, not graph ingestion

## Outcome

At the frozen pre-change snapshot `2026-08-22T15:07:14Z`, the inherited count of 21 was stale.
The live ordinary-untracked population was **26 / 9,119 present Markdown files** under
`.omx/research/`: **11** byte-for-byte `arm_final_messages` captures and **15** charters. A
separate **35 / 9,119** Markdown files were explicitly ignored artifact/extraction payloads, so
the full not-tracked population was **61 / 9,119**. I did not fold ignored raw payloads into the
ordinary `git status` count.

The other two invisibility legs were different populations:

| class | measured count | denominator | scoped conclusion |
|---|---:|---:|---|
| ordinary untracked in Git | **26** | **9,119** present `.md` | All 26 were arm-lifecycle births: 11 final captures + 15 charters. |
| ignored and not tracked | **35** | **9,119** present `.md` | Explicit raw artifact/extraction class; certified out of Git, not silently omitted. |
| tracked but absent from the frozen graph index | **4** | **9,047** graph-eligible tracked `.md` | Four newly tracked 2026-08-22 memos were newer than the cache. |
| present but absent from the frozen graph index | **14** | **9,108** graph-eligible present `.md` | The four tracked misses plus ten recent ordinary-untracked births. |
| recall-reachable by exact first-heading query | **28** | **32** seeded-random tracked top-level memos | **87.5%** reachability; four ambiguous-title misses. |

This is apparatus-only `[read-only scorer-free Git and frozen graph-cache audit]`. No scorer,
training, Metal, Modal, local advisory launch, archive build, or graph rebuild ran. The exact
frontier did not move.

Full machine-readable receipt:
`/Volumes/APDataStore/pact/ddm_os1_orphan_signal_reconciliation/os1_census_reachability_receipt.json`,
26,597 B, SHA-256 `93bef57a9db16c8b94c4296e29bd826c167f4aa6b7586df290e2d92c6c70885f`.

## Population definitions and frozen authority

The ordinary-untracked count is the output population of
`git ls-files --others --exclude-standard -- ':(glob).omx/research/**/*.md'`. The ignored count
uses the same query with `--ignored`; it is separate because `.gitignore` deliberately excludes
raw `.omx/research/artifacts/`, extracted trees, runtime snapshots, and packaged-submission
payloads while requiring their durable signal to be promoted to compact memos/manifests.

The graph-index comparison used the already-existing cache without invoking
`tools/graph_memory_recall.py`, because that entry point auto-rebuilds a stale cache. Frozen pins:

- nodes: 17,917,691 B, SHA-256
  `ca4e4576c1babf1e572e4f866a4ad530c4399594561212b21600f34fc7e61a8a`, mtime
  `2026-08-22T09:36:15-0500`;
- edges: 43,214,752 B, SHA-256
  `e95cfb5d13601e647f535c3c98a6696ca66da6891ceb161a86e771970e1b1457`, same mtime.

The four tracked graph misses were
`ddm_db1_decode_boundary_families_20260822.md`,
`ddm_jx1_joint_exchange_envelope_20260822.md`,
`ddm_rc1_rate_crush_20260822.md`, and
`ddm_vf1_evaluator_visible_floor_20260822.md`. All were newer than the cache. The graph builder
indexes every `.omx/research/**/*.md` regardless of Git tracking, and its normal loader compares
corpus mtime with cache mtime before serving recall. Therefore **Git-untracked is not equivalent
to graph-absent**: 16 / 26 ordinary-untracked files were already in the frozen graph; 10 / 26
were born after it. The four tracked misses are folded into existing auto-invalidation; OS1 did
not manually rebuild and self-confound the baseline.

## Intent adjudication

| class | count | intent from producer/source | disposition | owner · consumer store · fire trigger |
|---|---:|---|---|---|
| `arm_final_messages/*.md` | 11 | `tools/codex_arm_queue.py::persist_final_message` copies each `.last.txt` byte-for-byte, hashes it, and indexes it. NP1 calls this research custody. Prior commits track 372 peers; commit `14ab436cb6` landed 343 at once and `955b7e4266` landed another capture backlog. The files are untracked by omission, not policy. | **BLOCKED_PENDING_SERIALIZER**; do not mutate. | MAIN · `git:main` + graph-memory corpus · Git writes available, then serialize the unchanged captures. |
| top-level and nested `charters/*.md` | 15 | `codex_arm_queue.py::cmd_scaffold` defaults to the top-level charter directory; queue intake consumes charter files. The three nested VP1 charters are explicit fire-order records, and `COMMIT_BLOCKED.md` names repository history as their consumer. | **BLOCKED_PENDING_SERIALIZER**. | MAIN · `git:main` + graph-memory corpus · Git writes available, then serialize exact charter bytes. |
| ignored artifact/extraction `.md` | 35 | `.gitignore` explicitly treats these paths as raw artifact, extraction, runtime, or packaged-submission custody; canonical signal is promoted to a tracked memo/manifest. | **CERTIFIED_EXCLUDED**; no deletion and no force-add. | Historical artifact/intake owner · local artifact custody · promote only if unique decision signal is absent from tracked corpus. |
| tracked but frozen-graph-absent memos | 4 | Recent primary memos landed after the frozen cache. | **FOLDED_AUTO_INVALIDATION**. | `tac.graph_memory.load_or_build` · graph cache · first ordinary recall after source mtime. |

There are zero `UNKNOWN` rows in this population classification.

## Recall reachability

Sampling was seeded random without replacement, not a prefix:

- population: 7,525 tracked top-level `.omx/research/*.md` files, excluding
  `sub015_DAG_*` and `CANONICAL_RESEARCH_INDEX*`, with a `research:` node in the frozen cache;
- seed: `20260822`;
- sample: `n=32`;
- query: the memo's exact first Markdown heading, which the memo itself contains;
- default reconstruction: `max_seeds=4`, `max_nodes=18`, `max_depth=2`.

Result: **28 / 32 reachable = 87.5%**. The four misses were not graph absence. They had generic,
colliding titles: `HiNeRV archive-size ladder`, `NeRV control inventory`, and two copies of
`NeRV Long-Training Campaign Plan`. A diagnostic query made distinctive with each memo's filename
tokens recovered all four at rank 1. The scoped negative is therefore: **the default title-only
query did not surface 4 / 32 sampled memos**. It is not a claim that the memos are globally
unreachable. The live defect is ambiguous-title seed truncation, not missing graph ingestion.

## RECALL EVIDENCE

| source / query | finding beyond charter seeds | change to plan |
|---|---|---|
| `#878`, `record-censor`, `arm_final_messages`, `untracked research` over full `.omx/research` content | NP1 established byte-for-byte persistence; HV2 later measured that 343 captures were merely untracked and 454 finals still lived only in ignored run state. OC2 then committed the 343. | Classified final captures as commit-required forensic evidence, never deliberate scratch and never mutable. |
| Git history for `arm_final_messages` and untracked research | `14ab436cb6` tracks 343 captures; `955b7e4266` tracks 42 later research records and explicitly says omission, not policy. | Rejected a new exception policy; kept the existing commit intent and added a producer-side fail-closed debt gate. |
| VP1 report, three nested charters, queue rows, and `COMMIT_BLOCKED.md` | VP1 landed as an arm but its three fire-order charters remained from a failed serializer attempt; their named consumer is repository history. | Nested charter paths are included in the cure, not just the top-level default. |
| `src/tac/graph_memory/{build.py,__init__.py,recall.py}` | The builder recursively indexes all research Markdown independent of Git, loader auto-invalidates on source mtime, and recall seeds from node id + title + summary with a four-seed cap. | Separated Git custody, graph presence, and query reachability; did not rebuild during measurement. |
| canonical equation registry, 449 rows | `activation_ledger_not_run_truth_v1` says “fired but never recorded” is orphaned signal. | Treated an unrecorded Git-custody debt as a birth-time state that must block later actuation. |
| `CANONICAL_RESEARCH_INDEX*`, DAG `sub015_DAG_*`, canonical task status, and arm queue; queries `arm_final_messages`, `orphaned signal`, `graph_memory_recall`, `recall-before-decide` | No direct canonical-index row for this current orphan class was found. The DAG does carry the graph-memory parser/reconstruction contract and the default-off orphan law; the queue holds NP1 provenance. | Extended the existing arm queue rather than creating a second research producer or graph index. |
| prior week/two-week signal-loss memory hooks | Earlier orphan sweeps were partial and left no final memo/ledger commit. | Required typed denominators and an executable guard rather than another prose-only inventory. |

## Producer-side cure

The prior population was entirely controlled by the Codex arm lifecycle. The working-tree cure is
implemented in `tools/codex_arm_queue.py` with tests in
`src/tac/tests/test_codex_arm_queue.py`:

1. every final-message index row now records
   `git_custody_at_birth=untracked`, `custody_disposition=BLOCKED_PENDING_SERIALIZER`, owner MAIN,
   the Git/graph consumer, and the exact pre-saturation fire trigger;
2. `cmd_add` refuses a managed charter whose exact current bytes do not match `HEAD`;
3. `cmd_saturate` refuses if any final capture or top-level/nested charter is untracked, ignored,
   staged-only, deleted, or modified relative to `HEAD`;
4. queue status exposes `CLEAN` versus `BLOCKED` custody; and
5. the path check is lexical, so a symlink under `charters/` cannot escape the gate.

This is a class fix, not a sweep: a future capture may still be created as an untracked file because
the shared watcher cannot safely author a Git commit while other agents own the index, but the birth
is machine-recorded immediately and the actuator cannot fire another arm until MAIN commits the
exact bytes. Auto-committing inside the watcher was rejected because it would touch the shared Git
index asynchronously and could absorb someone else's staged state.

### Verification

- focused queue suite: **70 passed**;
- queue + completion-watcher integration: **91 passed**;
- compileall: pass;
- Ruff fatal/static subset `E9,F63,F7,F82`: pass;
- `git diff --check`: pass;
- real working-tree positive fire: the new saturation guard reported the frozen **26** debts = 11
  finals + 15 charters and returned `rc=4` before reading the queue or spawning. A concurrent XT1
  final capture and then an NI1 charter born after the snapshot raised the guarded set during
  landing retries without changing the frozen denominator;
- two genuine review-tracker passes marked both Python files reviewed after the last fix.

The first broader Ruff run reported three pre-existing style findings at unrelated unchanged lines;
none is introduced by OS1.

## Landing boundary

The producer cure and this memo are **implemented and verified but not committed**. Required
serializer attempts covered the frozen 26-file custody population and, after the concurrent XT1
and NI1 births, each then-live population returned by `uncustodied_research_births()` plus the two
Python files and this memo. Every file had a post-edit SHA-256, and the message carried
`[no-triality] [p0-ledger-ok]`. The serializer refused before staging:

`unable to create temporary file: Operation not permitted` / `failed to insert into database`.

`git diff --cached --name-status` remained empty. I did not bypass the serializer, write Git objects
through another path, mutate a forensic capture, or touch RC1/NR1/VF1 primary memos or retained
trees. This is the precise reason the producer cure is not claimed landed.

## Prior-law prediction verdict

**Verdict scope: INSTANCE — ordinary-untracked Markdown snapshot at
`2026-08-22T15:07:14Z`.** The prediction is **supported at lifecycle-producer level and refuted at
literal one-location level**: 26 / 26 ordinary-untracked files came from the Codex arm lifecycle,
but they split across two birth surfaces (11 final captures, 15 charter paths). The explicit
falsifier, at least four distinct producers with no shared default, did not fire. One queue-side
cure therefore covers the measured population without pretending that one directory held it all.

## NEXT_IF_RESUMED

- **BLOCKED_PENDING_SERIALIZER** — owner: MAIN or the next Git-writable operator; consumer store:
  `git:main` plus the graph-memory research corpus; fire trigger: Git index/object writes become
  available. Re-run the canonical serializer with post-edit SHA-256 values for every live birth
  record returned by `uncustodied_research_births()` and for `tools/codex_arm_queue.py`,
  `src/tac/tests/test_codex_arm_queue.py`, and this memo; verify the staged index is empty before
  and after any refusal.

## LIVE-HYPOTHESES

- Unique retrieval-card headings or exact-title collision expansion could raise default-query recall
  above 28 / 32, because every miss was a title collision and every distinctive query recovered at
  rank 1.
- The new fail-closed custody gate should keep the ordinary arm-lifecycle orphan population from
  accumulating, because all 26 measured rows pass through the guarded queue surfaces.

## DEAD-ENDS

- Treating “untracked” as “absent from graph” is closed: 16 / 26 ordinary-untracked files were
  already indexed, while four newly tracked memos were cache-absent.
- Blanket-committing ignored raw artifact/extraction Markdown is closed: `.gitignore` explicitly
  routes that class to local payload custody plus promoted compact signal.
- Auto-committing from the completion watcher is closed for this shared worktree: asynchronous Git
  index mutation can absorb unrelated staged state.
- Manual graph rebuild during baseline measurement is closed: it would self-confound the before
  population, and normal recall already has source-mtime auto-invalidation.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600; OS1 moved no score or archive bytes.
