# ddm_vd1b — FIX the vd1 worker's archive loader: decode via the ADAPTED runtime, then hand back READY_TO_FIRE

**Charter date:** 2026-08-12 · **Owner:** codex arm (xhigh) · **Consumer:** MAIN fires the repaired K=200 Modal T4 validator job under #381.
**Parent:** ddm_vd1 (commit `5afda1c9a7272b5deace39b45a723a0b0326acc5`, memo `.omx/research/ddm_vd1_modal_batch_event_validator_20260812.md`) — READY_TO_FIRE was claimed, but the FIRST REAL DISPATCH refuted the worker's loader. This arm fixes exactly that defect. Read the parent memo + `docs/operating_manual_craft_handoff.md` + CLAUDE.md/AGENTS.md first.

## The measured failure (exact, from Modal fc-01KZWHW948TQD3EAW2WY7WR2Z8, T4, 2026-08-13 03:15Z)

```
File ".../runtime_bundle/runtime/residual_archive.py", line 314, in read_residual_archive
    models = decoder.decompress(outer)
_lzma.LZMAError: Corrupt input data
→ runtime.residual_archive.ResidualArchiveError: invalid F24S model section
  at experiments/ddm_vd1_batch_event_validator_worker.py:202 load_receiver_state → read_residual_archive(archive_path)
```

**MAIN's hypothesis (verify, do not assume):** the worker's `load_receiver_state` re-implements receiver-state parsing and assumes the PR135-BASE coding of the model section (LZMA). The dispatched archive is the cp135 COMPOSED archive
`/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip`
(sha256 `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`, 186,252 B) whose model/token sections carry the **HP3/RC64 probability-object recode** (that is the −216 B compose leg). The ADAPTED runtime in the same directory decodes it exactly (proof: the cp135 contest-CUDA row S 0.16195513827824176 was bought with these very bytes). So the defect is loader-vs-archive coding mismatch inside OUR worker, not archive corruption.

## The job

1. **$0 local repro first** (no Modal): run the worker's `load_receiver_state` against the real archive + adapted runtime on this host; reproduce the exact LZMAError. Then decode the SAME bytes through the ADAPTED runtime's own reader (the code path `adapted_runtime/runtime/…` actually uses at inflate time) and show it succeeds. That pair of receipts pins the hypothesis or refutes it.
2. **Fix = reuse, never re-implement:** make the worker obtain renderer/semantic/basis/coefficients/tokens/selector state **through the adapted runtime's canonical decode functions** (import from the staged runtime_bundle, same modules inflate uses), not through a hand-rolled `read_residual_archive` call with base-era assumptions. If the adapted runtime lacks an importable seam, add the *thinnest* seam in the worker (not in the shipped runtime files — those are custody).
3. **Byte-equivalence gate:** after the fix, the worker's reconstructed token plane for the UNMODIFIED base must byte-match the adapted runtime's own decode output (sha over the 600-frame token plane). This is the same identity gate jo1 used — reuse its checker if convenient.
4. Rerun the worker's local self-tests + charter checks; keep P0 payload retention untouched.
5. **Do NOT dispatch Modal.** MAIN owns dispatch. Land the fix on main via the serializer (post-edit working-tree shas), then final message = `READY_TO_FIRE` + the pinned K=200 command (unchanged from the parent memo except any flag your fix requires) + the local repro/fix receipts.

## Custody / constraints
- The archive + adapted_runtime are READ-ONLY custody. Do not edit files inside `adapted_runtime/` — the fix lives in `experiments/ddm_vd1_batch_event_validator_worker.py` (and dispatcher if a flag is needed).
- Also carry forward the three dispatcher fixes MAIN already landed this session (lazy mounts · top-level module name · dual-path `modal_auth_eval` import · `base_archive_sha256` kwarg) — do not revert them.
- K arithmetic + budget from the parent memo stand (935.916 s vs 1,800 s; all 200 events fit).
- NO scorer runs, NO MPS. Advisory only; the exact row stays MAIN's.

## Falsifier
If the local repro shows the adapted runtime's OWN reader ALSO fails on these bytes, the hypothesis is wrong — stop, do not patch blindly; report the divergence (which bytes, which section, both tracebacks) as the finding.
