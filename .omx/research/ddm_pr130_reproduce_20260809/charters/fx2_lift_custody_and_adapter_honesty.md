# ddm_fx2 — CURE 2+3: silent lift drift + the silently-selected dense adapter

**Operator 2026-08-09: "Continue with all."** Two rr4 HIGH findings, same territory
(`src/tac/pr130_lift/`), one arm. Both are custody-honesty defects: code that says one thing and does
another. Neither is a score claim; both poison every claim built on them.

## THE FINDINGS (rr4, `f31f818ba9`, do not re-derive)

**F1 — silent drift.** Of 13 Python files in the two lifted trees: 0/13 byte-identical to intake,
10/13 declared-adaptation-only, **3/13 silently drifted**. `lifted/train_semantic_full.py`,
`lifted/train_semantic_quantized.py`, `pose/lifted/train_pose_carrier_full.py` each gained an
`assert_governed_admission(...)` call from commit `f06c8493f2` **after** the lift, while their headers
still read `ours: This accounting header only` / `vendored custody copy only`. All 12 files citing a
`source_sha256` name the correct ORIGINAL bytes — **but those hashes do not authenticate the current
bodies.** Note also each header's `source_head:` carries `2f94596bb0…` (our LIFTED_AT_HEAD), while the
intake pin is `e34f31bc…` (SOURCE_REPO_HEAD) — the `__init__.py` split exists precisely because these
two were conflated once already.

**F2 — silent mechanism substitution.** `pose/mps_port.py`: `use_sparse = device.type != "mps"` makes
`RowLocalDenseAdam` **mandatory on every MPS run** — no flag, no warning, no fallback event, no result
field recording optimizer class, mode, runtime, or fallback policy. The adapter body IS
mechanism-preserving in the tested envelope (uniquifies row IDs, one clock increment per selected row
per step, per-row `exp_avg`/`exp_avg_sq`, borrowed bias corrections, refuses undeclared rows, requires
fresh declaration each step; 2-step repeated-ID CPU control matches `RowLocalSparseAdam` exactly).
**The defect is the selection, not the body** — and it contradicts our own stronger receipt
`/Volumes/VertigoDataTier/pact/ddm_pq1_probe_20260809/probe_torch2100_pinned.json` (sha
`32ce0585d070fd578bea563f94b33fffe6e000b8cc608f827d4fcb5319893ec3`): native sparse `nn.Embedding` +
coalesced COO + borrowed `RowLocalSparseAdam` PASSED on real MPS at pinned torch 2.10.0, row clocks
`[1,2,1,2]`, untouched rows bit-identical, CPU/MPS parity `atol=2e-6 rtol=2e-5`, zero CPU-fallback.

## YOUR SCOPE

1. **Cure F1 per rr4's own falsifier:** either restore body identity, or declare each admission guard
   explicitly in its accounting header — then add a test that **reconstructs the expected body from the
   pinned intake original and permits only an explicit, enumerated adaptation patch**. rr4 states
   plainly: *a manifest-literal test is not a falsifier.* The test must fail if a future edit lands
   undeclared. Reconcile the `source_head`/`source_sha256` semantics so a reader cannot mistake which
   head authenticates what.
2. **Cure F2:** make native sparse the explicit REFERENCE mode on the pinned runtime; any adapter use
   becomes a named opt-in or a LOUD fallback (never silent); and persist optimizer class, sparse/dense
   mode, fallback event, torch version, git SHA, argv, and the native-probe receipt identity into every
   result AND checkpoint. rr4's cure names the follow-on explicitly: compare uninterrupted reference vs
   adapter trajectories **beyond the present two-step control** — build the harness, and run it if a
   CPU-only run can carry it; if it needs Metal, leave it READY with the exact command.
3. **Sweep for siblings.** Both defects are genus, not instance: any other place where a header, a
   manifest, or a result claims a mechanism the code does not select, or a hash that does not
   authenticate current bytes. Report the denominator you swept.

## OPTIMAL FORM

- **Reference form:** all 3 drifted files reconciled with a reconstruct-from-intake test that fails on
  undeclared change; sparse-as-reference with loud opt-in and full provenance persisted; a sibling
  sweep with its denominator.
- **SCOPE reductions (legal):** CPU-only execution (arms have no Metal device); the >2-step trajectory
  comparison may land READY-with-command if it requires Metal.
- **MECHANISM reductions (declare TOY-BRACKET):** a test that asserts the manifest instead of
  reconstructing the body; declaring the adapter equivalent from its docstring; "fixing" F2 by adding a
  log line while selection stays unconditional.
- **Provenance pins:** intake `e34f31bc4969042c0051ac81aa3c56884419a231`; lift `LIFTED_AT_HEAD
  2f94596bb0136d342254022a5c9584756eae0468`; drift-introducing commit `f06c8493f2`; rr4 audit
  `f31f818ba9`; native-sparse receipt sha `32ce0585d070fd578bea563f94b33fffe6e000b8cc608f827d4fcb5319893ec3`.

## NON-NEGOTIABLES

- Intake READ-ONLY; never edit in place, never `git add` inside it.
- Borrowed-substrate honesty (NO-FAKE #7): borrowed code stays labelled borrowed WITH accurate
  accounting. Declaring an adaptation is the cure; hiding one is the defect.
- MPS/MLX never score authority. `score_claim=false` on every row.
- **Never consume a background job's output without asserting terminal status.**
- verdict_scope on every negative. Denominators on every count.
- Commit via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`. NO Claude/AI attribution, no `Co-Authored-By`.
- **`REVIEW_GATE_OVERRIDE=1` is FORBIDDEN here — this arm edits `.py`.** Use
  `tools/review_tracker.py mark-file <file> --status reviewed` (two passes per file).

## DELIVERABLE

`.omx/research/ddm_pr130_reproduce_20260809/FX2_LIFT_CUSTODY_CURES.md` — **§1 = the per-file custody
table after cure** (declared vs restored vs still-drifted, with the test that now guards each), then
the adapter selection cure with its persisted-provenance schema, the sibling sweep with denominator,
ranked residuals with falsifiers, and "could not check / why."
