# ddm_rr4 — recursive adversarial review, LAYER 4: the CARRIER leg + OUR lift's fidelity

**Operator directive 2026-08-09:** *"The Recursive adversarial Review must check every step of every
stage of everything, recursive fractal of our port to upstream PR one thirty."*
Operator frame: *"distinguish between everything we can use off the shelf... and that which must be
ported. We already have a bunch of stuff that we ported in the lift directory."*

Fresh eyes. This arm audits **the one stage that genuinely needed a port** — and **our own port code**.

## THE OBJECT

`BASE = PR130 CPR1 S = 0.172141297491896447` `[contest-CUDA, DALI GT, n600]`, archive 191,052 B.
Intake (READ-ONLY): `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo`,
`SOURCE_REPO_HEAD = e34f31bc4969042c0051ac81aa3c56884419a231`.
Our lift: `src/tac/pr130_lift/` — `LIFTED_AT_HEAD = 2f94596bb0136d342254022a5c9584756eae0468`.
Ledger under audit: `.omx/research/ddm_pr130_reproduce_20260809/OFF_THE_SHELF_VS_PORTED.md` (3856788c96).

Pose is 12.24% of archive bytes (23,384 B → marginal 0.0155704 S).

## YOUR SCOPE

1. **DRIFT AUDIT — the load-bearing one.** `pr130_lift/__init__.py` deliberately separates
   `LIFTED_AT_HEAD` (our copies) from `SOURCE_REPO_HEAD` (files we execute directly), introduced after
   hb2's fix advanced the repo while the pin stayed stale — *"Same value, two meanings, one of them
   silently wrong."* **Now verify it empirically:** for every file under `pr130_lift/lifted/` and
   `pr130_lift/pose/lifted/`, diff against its intake original at `SOURCE_REPO_HEAD`. Report per-file
   {byte-identical · drifted-with-declared-adaptation · drifted-SILENTLY}. Silent drift is the finding.
   Check each file's `borrowed_substrate_accounting` header + `source_sha256` against reality.
2. **`code/train_pose_carrier_full.py`** — the real port target. The 2 device sites MAIN named
   (`torch.cuda.empty_cache()` :264 · `load_file(..., device='mps')`). Are those the ONLY device
   sites? Sweep the whole file and its imports for CUDA assumptions, pinned memory, device-typed
   dtypes, `.cuda()`, stream/event use, anything that silently degrades on MPS.
3. **`RowLocalSparseAdam` mechanism fidelity** — per-coefficient-row bias-correction clocks. A dense-
   Adam substitution is a MECHANISM reduction (pp2's named trap: repeated ids within a step must
   increment the clock ONCE per coalesced row, not once per occurrence). Our `pr130_lift/pose/mps_port.py`
   keeps a dense-gradient ADAPTER as a fallback. **Audit: is the adapter mechanism-preserving, and can
   it be silently selected when the reference sparse path was available?** A fallback that engages
   without a loud signal is the inert-flag class.
4. **THE OURS COLUMN — unraced.** `pr130_lift/mlx_semantic_renderer.py`, `pose/mlx_pose_carrier.py`,
   `pose/repack_race.py` are OURS (optimizations), not PORT (required), because the OTS path runs
   directly. None has won an A/B. For each: does it claim, in code or docstring, to be equivalent to
   or better than the OTS path? Any such claim without a measured row is a NO-FAKE #8 surrogate claim.
   Report the honest status of each.
5. **`pr130_lift/pose/{source_loader,train_pose_carrier_full_resumable,repack_race}.py`** and the
   three test modules — do the tests verify BEHAVIOR or CONSTANTS? (NO-FAKE class 2: if every test
   would still pass with the body replaced by canonical markers, the suite verifies constants.)

## ROUND-1 FINDINGS — build past, do not re-derive

- **F2 (the serious one, YOUR territory):** the ledger cited `mean d_pose 2.4437744286842644e-05` as a
  measured n600 carrier result. **That figure has NO locatable receipt** — absent from the intake,
  from `/Volumes/VertigoDataTier/pact`, and from `.omx/research`. It is WITHDRAWN. The real receipt is
  `.omx/research/ddm_pq1_20260809T125541Z/` (`verdict: PARTIAL`, `metal_executed: false`) plus
  `/Volumes/VertigoDataTier/pact/ddm_pq1_probe_20260809/probe_torch2100_pinned.json` — scope **2 steps
  × 4 rows**, CPU/MPS parity within predeclared fp32 tol (atol 2e-6, rtol 2e-5), row clocks `[1,2,1,2]`
  over rows `[2,5,7,19]`, zero cpu-fallback, at PINNED torch 2.10.0. **The 4,000-step n600 carrier
  training has NOT been run here.** If you find the missing receipt, that is a real finding — say where.
- The pinned runtime is `/Volumes/VertigoDataTier/pact/ddm_pq1_runtime_20260809/venv` (torch 2.10.0).
  The probe FAIL-CLOSES on torch != 2.10.0 — it correctly REFUSED at the repo venv's 2.12.1. Check
  whether any OTHER consumer of the lift silently runs at 2.12.1 and would answer a different question.

## OPTIMAL FORM

- **Reference form:** a complete file-by-file drift diff of every lifted copy against its intake
  original at the pinned SOURCE_REPO_HEAD, plus a full device-site sweep of the carrier trainer, plus
  behavior-vs-constants classification of every test in the lift.
- **SCOPE reductions (legal):** static diff/read instead of execution — arms have no Metal device
  (pp2 lesson); CPU-only test reading and pure-python checks are in scope.
- **MECHANISM reductions (declare TOY-BRACKET):** spot-checking a few lifted files instead of all;
  trusting the `source_sha256` header instead of recomputing it; declaring the adapter
  mechanism-preserving from its docstring rather than its body.
- **Provenance pins:** intake `e34f31bc4969042c0051ac81aa3c56884419a231`;
  lift `LIFTED_AT_HEAD 2f94596bb0136d342254022a5c9584756eae0468`; ledger 3856788c96.

## NON-NEGOTIABLES

- Intake READ-ONLY; never edit in place, never `git add` inside it.
- MPS/MLX never score authority; no Metal device available to you.
- No number without a locatable receipt; ABSENT is honest, restating is not (F2 is the cautionary case).
- verdict_scope on every negative. Denominators on every count.
- Borrowed-substrate honesty (NO-FAKE #7): borrowed code stays labelled borrowed with accounting.
- Commit via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`. NO Claude/AI attribution, no `Co-Authored-By`.
- `REVIEW_GATE_OVERRIDE=1` FORBIDDEN with `.py`; fine for `.md`/`.json`.

## DELIVERABLE

`.omx/research/ddm_pr130_reproduce_20260809/RR4_CARRIER_LIFT_AUDIT.md` — **the per-file drift table as
§1** (every lifted file, byte-identical vs drifted vs silently-drifted), then the device-site sweep,
the adapter-fallback verdict, the OURS-column honest status, the tests behavior-vs-constants
classification, ranked findings with falsifiers, and "could not check / why."
