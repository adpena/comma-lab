# ddm_pr15 — ratification of the fire-tool repair and first-real-dispatch findings

Date: 2026-09-10  
Axis: `[review; scorer-free; read-only code and retained-artifact inspection]`  
`research_only=true; score_claim=false; promotion_eligible=false`  
Tokens: `[no-triality] [p0-ledger-ok]`

## Verdicts

1. **Dispatch-path ordering — RATIFY.** Excluding exactly the context file that the tool itself creates after validation is the narrow cure; every other argv and known spawn path remains under `verify_dispatch_paths`. Freeze amendment `[1]` is a valid append-only implementation re-pin. Provenance correction: `21c9b17ba6957ecf0661288d0d7b4aaaac55b2c4` is the causal code-change commit; `808da43c02a32ae328fd73c707331b9aaee2ac36` is a later reachable implementation snapshot containing those bytes, not the change commit.
2. **Replay guard versus refusal receipts — RATIFY AS IS.** The guard already ignores direct `PREFIRE_REFUSAL_*.json` entries and must continue to refuse every other entry. The real refusal directory also contained a launcher directory, so the refusal was correct; placing launcher custody beside, rather than inside, the authorized output is the correct operational cure. No contract or code amendment is authorized.
3. **Authorization-tool import path — AMEND.** Add the repository root after `src`, add a foreign-working-directory regression, regenerate the full implementation manifest, append the next immutable freeze row, and re-emit any later intent. This is a pinned-consumer change. Because the axis-tag repair has since occupied amendment `[2]`, this repair is amendment `[3]`, not the charter's now-stale `[2]` forecast.

## Controlling boundary

The pr12 contract requires all pre-dispatch uncertainty to refuse before a subprocess, one durable
reservation per authorization nonce, immutable intent/authorization bytes, and only forward
`RESERVED -> SPAWNED -> HARVESTED` transitions. Pr13 ratified raw-file SHA/bytes plus canonical digest
and `arm produces -> MAIN commits -> MAIN authorizes`; pr14 allowed only the scoped
`candidate_prefire_timing_risk.v1` normalization and required append-only implementation re-pins.
Nothing reviewed here relaxes a dispatch, claim, result, exact-archive, axis, or timing check.

The first real producer pass is still mechanism evidence only: the retained candidate is 180,406 B,
SHA-256 `04758c0d...`, and raw-identical to the move-43 witness, but its score remains conditional until
the exact retained first-measurement result is harvested and accepted by the unchanged completion
consumer. This review ran no scorer, Modal command, fire, reservation, or pointer mutation.

## Clause -> code -> test

| Finding | Contract clause | Code evidence | Executed test/evidence | Ruling |
|---|---|---|---|---|
| Dispatch-path ordering | A locally produced future path is not pre-existing evidence; every actual argv/spawn dependency must resolve before spend. `_pf_write_new` must still refuse replacement. | Historical change `21c9b17b` filters only arguments exactly equal to `str(context_path)` before `verify_dispatch_paths`; `_pf_write_new(context_path, context)` remains later. The amendment `[1]` manifest pins the resulting fire-tool blob `e30105d9...` at snapshot `808da43c...`. | `test_dispatch_path_guard_catches_the_ps2_failure` proves unresolved known spawn paths still fail. The retained run3 directory contains both `FIRST_MEASUREMENT_CONTEXT.json` and an axis-tagged `FIRE_MANIFEST.json`, a real positive control that the formerly impossible pass path was reached. | **RATIFY** |
| Refusal receipts and output reuse | An unconsumed authorization may ignore its own diagnostic receipts, but any dispatch/claim/result or ambiguous non-receipt entry must refuse. A reserved nonce never resets. | `check_first_measurement_unconsumed` at `candidate_seal.py:2496-2507` excludes only names beginning `PREFIRE_REFUSAL_`; every other direct child refuses. Consumption and prior job/output checks remain earlier and unchanged. | Host probe: receipt-only directory `PASS`; adding empty `launch_fire/` produced `FIRST_MEASUREMENT_REPLAY_REFUSED`; replacing it with `modal_auth_eval_spawn.json` produced the same refusal. | **RATIFY AS IS** |
| Authorization-tool import closure | A committed consumer must import all of its transitive runtime-digest dependencies from any working directory; pytest's implicit repo root is not a runtime guarantee. Any source change in `PREFIRE_IMPLEMENTATION_PATHS` requires a fresh manifest/freeze append before a new intent. | `authorize_candidate_first_measurement.py:9-10` adds only `REPO/src`; `decode_wall_clock.py:189-196` imports `experiments.contest_auth_eval` inside `measure_t4_runtime_digest`. `quiesced_decode_timing.py:34-36` and fire-tool commit `6c74c56f...` provide the matching precedent. | From `/private/tmp` with `PYTHONPATH` removed, `src`-only reproduction raised `ModuleNotFoundError: No module named 'experiments'`; inserting the repo root at index 1 made the import pass. | **AMEND** |

Host validation also passed:

- `.venv/bin/python -m pytest src/tac/tests/test_candidate_prefire_intent.py src/tac/tests/test_candidate_seal.py src/tac/tests/test_decode_wall_clock_t4_direct.py -q` — **136 passed**.
- `.venv/bin/python -m pytest src/tac/tests/test_modal_source_snapshot.py src/tac/tests/test_fire_modal_lane_id_promotable.py -q` — **15 passed**.
- `.venv/bin/python -m ruff check tools/fire_modal_auth_eval.py tools/authorize_candidate_first_measurement.py src/tac/candidate_seal.py src/tac/decode_wall_clock.py src/tac/tests/test_candidate_prefire_intent.py src/tac/tests/test_modal_source_snapshot.py` — **passed**.

## Finding 1 — dispatch-path repair and amendment `[1]`

The old order gave `verify_dispatch_paths` an argv containing
`<authorized-output>/FIRST_MEASUREMENT_CONTEXT.json` before the same function created that file.
Pre-creating it could not be a workaround because `_pf_write_new` deliberately refuses replacement.
That was a gate with no door.

The repair is exact, not categorical: list comprehension equality removes only the generated context
path. It does not exempt the output directory, archive, runtime, Python executable, worker entrypoint,
claim tool, or any other local spawn dependency. Dry-run remains non-consuming; the non-dry path still
revalidates the authorization, rechecks the cloud/lane, reserves before dispatch, writes context with
create-new semantics, snapshots sources, writes the fire manifest, and only then launches.

Amendment `[1]` is internally valid:

- freeze bytes at `ff75f2f9f2dae555f25f04b9cc4ebd48106708fa` hash to
  `57f613af8c82c7d7ffaf40df5aafeea90bc6e33b59a64c02fed178d640ef2899`;
- its manifest is 1,982 B, SHA-256
  `77cd2db3a2624c0976933a5dcfb27116f5732565d3f62ec167721e107677b39c`;
- every manifest row matched `git show 808da43c:<path>` byte-for-byte in this review;
- `21c9b17b` is an ancestor of `808da43c`; and
- `808da43c` itself changed an unrelated gdc2 memo, so it must be described as the implementation
  snapshot, not as the causal repair commit.

The snapshot wording correction does not invalidate the row. Pr12/pr14 require the pinned commit to
contain the exact consumer set, be in the accepted ancestry, and precede the later intent. They do not
require that snapshot's own diff to introduce every pinned byte.

The live freeze later acquired amendment `[2]` for the separate axis-tag repair and now hashes to
`f2d94a58cd1abbe29bf089703a79d66c6f20253637da336ff4f8c58bf51ac85b` at this review snapshot. That
later row does not rewrite or erase amendment `[1]`. Its separate semantics are outside the three
chartered verdicts except that it determines the next available append index.

## Finding 2 — the replay guard did not mistake receipts for dispatch

The charter's phrase “only the tool's own refusal receipts and an empty launch dir” joins two different
classes. A direct `PREFIRE_REFUSAL_*.json` is diagnostic evidence and is explicitly ignored. An empty
directory named `launch_fire` is neither a refusal receipt nor safely distinguishable from partially
created launcher custody, so it is an ambiguous dispatch artifact and must refuse.

The real history agrees with the code. Authorization v3 targeted run1, whose launcher log directory was
created inside the authorized output before the replay guard. MAIN's later operational correction moved
the launcher directory to a sibling path. Weakening the guard to ignore arbitrary empty directories,
launcher names, or all tool-created content would reopen ambiguous-dispatch replay. No patch is allowed
for this finding.

## Superseded authorization ruling

Do **not** edit v1/v2 or add an in-object `SUPERSEDED` state. Intent and authorization objects are
immutable signed/committed bytes; mutating them would destroy the very identities the lifecycle protects.
No new lifecycle marker is required for safety:

- v1 remains a retained historical `AUTHORIZED_ONCE` artifact but its pre-amendment intent is not the
  latest frozen contract object;
- v2, v3, and v4 share nonce `1ad3a6b1...`; v4's durable `RESERVED` receipt now consumes that nonce for
  the whole equivalence class, so none can dispatch;
- v5 is retained but superseded; v6 uses the fresh job/nonce pair and is the run3 authorization; and
- prior job/output reuse plus the latest-freeze validation independently fail closed.

Commit prose and this adjudication are sufficient historical supersession records. If machine-readable
inventory is later desired, it must be a separate immutable receipt that cannot alter authorization
validity; it is not needed to close this dispatch.

## Finding 3 — literal AMEND patch

This patch is authorized only after the already-fired run3 lifecycle has reached a terminal retained
receipt and its completion/pointer adjudication is finished. Applying it sooner would make the live
implementation differ from intent v3's latest frozen pins during completion validation.

```diff
diff --git a/tools/authorize_candidate_first_measurement.py b/tools/authorize_candidate_first_measurement.py
--- a/tools/authorize_candidate_first_measurement.py
+++ b/tools/authorize_candidate_first_measurement.py
@@
 REPO = Path(__file__).resolve().parents[1]
 sys.path.insert(0, str(REPO / "src"))
+sys.path.insert(1, str(REPO))  # tac.decode_wall_clock imports experiments.contest_auth_eval for the T4 digest
 from tac.candidate_seal import (  # noqa: E402
```

Add this regression to `src/tac/tests/test_candidate_prefire_intent.py`:

```python
def test_authorize_tool_imports_experiments_from_a_foreign_cwd(tmp_path):
    """The standalone consumer must close its transitive import graph without PYTHONPATH."""
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[3]
    code = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('authorize_tool', "
        f"{str(repo / 'tools' / 'authorize_candidate_first_measurement.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "from experiments.contest_auth_eval import _runtime_dependency_manifest\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
```

Then run the relevant suites, commit the two source files, regenerate the complete
`PREFIRE_IMPLEMENTATION_PATHS` manifest from that committed snapshot, and append freeze amendment `[3]`
with the same pr14 digest definition and the new implementation pins. Any subsequent producer must emit
a new intent against amendment `[3]`; MAIN must create a new authorization only from that committed
intent. Never edit amendments `[0]` through `[2]`.

## Provenance pins

| Object | SHA-256 / commit | Review meaning |
|---|---|---|
| pr12 memo | `50d00e3956dc7ae5d3b15379d2ae6f8704119817b0b58f50aa30413b97eacadc` | Base lifecycle contract. |
| pr13 memo | `3156992449ea4f71966dfe96adea1471aa3b735dae3bac4a9f49c4a16ce4eba3` | Ratified dual identity, reference scope, and producer/MAIN ordering. |
| pr14 memo | `6e1732baea1912731fef6a53abca0cb94faf2e9440accc304dd9b34e593bade9` | Only allowed definition amendment. |
| rlc5 memo | `59f8fed003e03c05d5bf3f736a52aaadcc7cd2a2005c8604885e8d19a373d793` | First real producer-pass lineage; not an exact score. |
| dispatch repair | `21c9b17ba6957ecf0661288d0d7b4aaaac55b2c4` | Causal code change. |
| amendment `[1]` snapshot | `808da43c02a32ae328fd73c707331b9aaee2ac36` | Exact implementation snapshot; unrelated commit subject. |
| amendment `[1]` freeze append | `ff75f2f9f2dae555f25f04b9cc4ebd48106708fa` | Append-only re-pin. |
| amendment `[1]` freeze bytes | `57f613af8c82c7d7ffaf40df5aafeea90bc6e33b59a64c02fed178d640ef2899` | Historical freeze after append `[1]`. |
| amendment `[1]` manifest | `77cd2db3a2624c0976933a5dcfb27116f5732565d3f62ec167721e107677b39c` | All 15 committed source rows matched. |
| later axis-tag implementation | `7a837a9de38f1be1d6a023e0a6caab0e7696a1a2` | Separate current amendment `[2]` input. |
| later amendment `[2]` append | `815a6c9dcd160ee3969ede7a4868201852163cae` | Occupies index `[2]`; not rewritten. |
| intent v1/v2 raw files | `88204b883d622a6a...` / `8370bcf6401a40b9...` | Historical committed intent identities named by v1/v2 authorizations. |
| move-43 archive | `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e` | Current owned exact anchor, not changed here. |

## RECALL EVIDENCE

I searched the repository-wide research corpus, canonical research indexes, every `sub015_DAG_*`,
`docs/`, `SPEC.md`, task/ledger/state files, the implementation and test trees, and Git history for these
query families: `candidate_prefire`, `first_measurement`, `PREFIRE`, `AUTHORIZED_ONCE`, `replay`,
`refusal`, `verify_dispatch_paths`, `FIRST_MEASUREMENT_CONTEXT`, `measure_t4_runtime_digest`,
`experiments.contest_auth_eval`, `sys.path`, `gate with no door`, `manifest re-pin`, and the pr12-pr14
memo identities. I also ran the canonical-equations listing and searched its JSON for the same contract,
dispatch, replay, and import terms; no governing equation matched.

Material recalled evidence changed the adjudication in four ways:

- Git history proved `21c9b17b`, not `808da43c`, is the causal ordering repair, while a row-by-row blob
  check proved `808da43c` remains a valid snapshot pin.
- The exact guard and a host probe disproved the claim that refusal receipts themselves blocked replay;
  the launcher directory was the extra entry that correctly triggered refusal.
- Commit `6c74c56f...` and `quiesced_decode_timing.py` supplied a same-import precedent for the authorize
  fix and showed why a repo-root pytest can hide the defect.
- Live state showed amendment `[2]` is already occupied by the later FIRE_MANIFEST axis-tag repair and
  that run3 is already fired, so the import cure must wait for terminal completion and use `[3]`.

I did not find, in those searched scopes, a competing lifecycle rule that permits resetting a reserved
nonce, mutating a committed authorization to mark supersession, ignoring an arbitrary output-directory
entry, or treating the conditional rlc5 arithmetic as a measured exact row.

## Frontier and review boundary

At memo authoring, the canonical own-vehicle/effective frontier remains **move 43**:
`S = 0.1372848557085275`, `180,466 B`, `[contest-CUDA T4 n600]`, archive
`7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`.
The first-measurement run3 directory has a context, consumption record, and axis-tagged fire manifest,
but no retained `MODAL_REMOTE_RESULT.json` was present at the review snapshot. Therefore move 44 is not
established here, the pointer is unmoved by this unit, and the sub-0.12 goal remains unmet.

## NEXT_IF_RESUMED

- **MEASURE / owner MAIN / consumer store** `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run3/MODAL_REMOTE_RESULT.json` plus the committed `candidate_seal.v3` and pointer packet: **fire trigger** none—the v6 nonce/job is already reserved and fired; harvest only after the retained result appears, then complete and move the pointer iff the unchanged exact score and `t4_direct <= 1260 s` gates pass.
- **AMEND / owner MAIN implementation-and-freeze landing / consumer store** `tools/authorize_candidate_first_measurement.py`, its foreign-cwd regression, a complete new implementation manifest, and freeze amendment `[3]`: **fire trigger** only after run3 is terminal and its completion/pointer adjudication is committed; then re-emit any later intent under `[3]` and authorize once from those exact bytes.

## LIVE-HYPOTHESES

- Run3 may convert the 60-byte byte-closed rider into move 44, but this remains a hypothesis until the retained exact result closes score, timing, axis, and completion lineage.
- The one-line repo-root insertion should close the authorize consumer's foreign-cwd import graph; the mandatory regression and post-commit manifest re-pin are the proof gates.

## DEAD-ENDS

- Calling `808da43c` the causal repair commit is closed; it is a valid snapshot whose parent already contains the repair.
- Weakening replay protection to ignore an empty launcher directory is closed; only direct refusal receipts are safely ignorable, and the code already does that.
- Mutating v1/v2 with an in-object `SUPERSEDED` state is closed; immutable bytes plus latest-freeze, nonce, job, and output checks already fail closed.
- Numbering the authorize import cure amendment `[2]` is closed by the landed axis-tag amendment; the next append-only index is `[3]`.
- Treating rlc5's conditional arithmetic or a present fire manifest as an exact score is closed; only the harvested evaluator receipt and completion consumer can establish move 44.

<!-- # FORMALIZATION_PENDING: review/adjudication memo (contract consumer rules), no measured row of its own; pre-waiver sha 193a69241e748d08 -->
