# ddm_fx2 — Our shared FX1 inflate.sh violates the evaluator's three-argument contract.

**Owner:** codex arm · scorer-free · shipping-blocker · no Modal

## THE DEFECT, FOUND BY RECALL

`ddm_dv1` (commit `9a1b483b22`) found that the shared FX1 `inflate.sh` in
`src/tac/pr130_runtime/fx1_runtime_tree/` **violates the evaluator's three-argument entrypoint
contract**. dv1's isolated CPU runtime at `src/tac/pr130_runtime/dv1_cpu_runtime/inflate.py` repairs
it; **the shared runtime is untouched.** The finding blocked an unsafe dispatch — it was caught
before spending, not after.

dv1 filed it as QUEUED-WITH-A-FIRE-ORDER, owner "MAIN PR130 runtime owner", fire trigger: *before
any shared-runtime evaluation or dispatch.* `cx2` is composing an archive right now, so that trigger
is imminent.

## YOUR JOB

1. **Re-derive the defect at source** — do not trust dv1's summary. Read the evaluator's actual
   invocation of `inflate.sh` in the pinned upstream snapshot, read our shared FX1 `inflate.sh`, and
   state the exact contract and the exact violation. If dv1 was wrong, say so with the source lines.
2. **Port the repaired per-video loop** from the dv1 isolated runtime into the shared FX1 tree.
   Internal-leverage authority applies: dv1's repair is ours — use it off the shelf, adapt it, or
   improve it, but cite the path and commit you took it from.
3. **Replay the REAL three-argument contract** — run the actual `inflate.sh <archive_dir>
   <output_dir> <file_list>` form the evaluator uses, end to end, and show it works. A code
   inspection is not the proof here; the contract is behavioral.
4. **Guard it.** This is a two-landing class: the repair, plus something that refuses the
   re-introduction. Our own runtime tree shipping a wrong entrypoint signature is exactly the
   silently-wrong-instrument genus. Prefer extending an existing gate over adding a new one
   (Catalog #299 consolidation discipline); a positive control that actually fires is required —
   a gate that cannot fail is not a gate.

Sister hazard already recorded and worth checking in the same pass: **bare `python`** in emitted
inflate scripts is a python3-only-host shipping hazard (task #929/#937), fail-closed in three
emitters, and `pr103` adapter line 727 currently ENFORCES the bug as a contract. If FX1 has the same
shape, say so; fix it only if it is in scope and you can prove the fix.

## OPTIMAL FORM

Reference form: the shared runtime passes the evaluator's real entrypoint contract, proven by
executing it, with a re-introduction guard carrying an executed positive control. Declared
reductions: SCOPE only — you may prove on a small file_list rather than all 600 pairs, since the
contract is per-invocation, not per-sample; say which you used. MECHANISM reductions (a mocked
evaluator, a signature-only inspection, a guard whose positive control was never run) are TOY
BRACKET and cannot close this row.

Provenance pins:
- dv1 receipts: `.omx/research/ddm_dv1_20260809/DV1_RECEIPT.md` + `.json`, commit `9a1b483b22`
- the isolated repaired runtime: `src/tac/pr130_runtime/dv1_cpu_runtime/inflate.py`
- the shared tree under repair: `src/tac/pr130_runtime/fx1_runtime_tree/`

## HARD RULES

- `upstream/` is IMMUTABLE and READ-ONLY — read the contract there, never edit it.
- The intake clone `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/` is READ-ONLY.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256`, tags
  `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py` files: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never
  `REVIEW_GATE_OVERRIDE=1` with a `.py`. `.sh` may use the override.
- No Modal. No scorer. Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_fx2_20260809/`.

## DELIVERABLE

The re-derived contract and violation stated from source, the ported repair, the executed
three-argument replay, and the guard with its executed positive control. If any leg did not close,
name it.
