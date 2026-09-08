# ddm_ane3 — ane2 residue: re-measure the g13/g13stem finalists through the FIXED coremltools conversion, localize the tail-split worsening, gate the ANE screening backend (charter, 2026-09-08)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm · Spawned by MAIN 2026-09-08 under the operator's standing GO. Source directive: `.omx/research/ddm_ane2_residue_directive_20260905.md` (ITEMs 1–3). MEANS (screening throughput): `score_claim=false`; the ANE is a porting item (operator 2026-09-05), never a score axis.

## MANDATE

Operator 20260908: *"recover and respawn and continue with all, also codex is back"*. ane2 (`.omx/research/ddm_ane2_engineer_the_precision_drift_20260905.md`) found its own instrument defect: `FP16ComputePrecision(op_selector=…)` makes coremltools declare the model OUTPUT fp16 even when it transforms zero ops (MIL dtype 65552 vs 65568), so every SegNet flip rate in its split-point ladder and selective-fp32 table is an UPPER BOUND. Its `g13stem` pass (≤ 3.3e-05 flip rate at ≥ 75% ANE) is therefore provisional. Re-measure through the fixed path, explain the tail-split worsening with a named op, and only then admit the backend to `tac.ane_screening`.

## SCOPE

1. ITEM 1: fix the conversion (force the fp32 output dtype; verify at source in the ane2 conversion code the exact call), re-run n600 generated-decode flip rates for `all fp16`, `g13`, `g13stem`, tail k=64; record the n120→n600 agreement; update anchors on `scorer_fp16_drift_by_axis_v1` + `BACKEND_AXIS_VERDICTS` through `update_equation_with_empirical_anchor` (`tac.canonical_equations`). Acceptance: fixed-path g13stem ≤ 3.3e-05 at ≥ 75% ANE, else the pass is WITHDRAWN in the memo.
2. ITEM 2: per-op activation diff (fp16 vs fp32) at each boundary op 0:18 of PoseNet; one named op whose fp16 activation error explains ≥ 80% of the pinned dim-0 delta (0.150–0.154), or a scoped negative.
3. ITEM 3: only after ITEM 1 passes, register `g13stem` in `tac.ane_screening` as a SEG-ONLY screening axis with ITEM 1's measurement attached; pose stays refused on the ANE (hd128 self-MSE 7.02e-5 = 11× d_pose).
4. Memo `.omx/research/ddm_ane3_residue_fixed_conversion_remeasure_20260908.md`.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. Never edit `submissions/semantic_joint_ctxmix/`, the live pointer tree, or `experiments/ddm_sj1_*`.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29). This arm's n600 flip-rate runs use the coreml backends and the frozen cpu_torch reference; if a run needs the scorer lane, emit a typed fire order and let MAIN fire it.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes. Tokens `[no-triality] [p0-ledger-ok]`. NEVER a Co-Authored-By or AI-attribution trailer.
- ALWAYS KEEP THE PAYLOAD: converted models + per-pair flip ledgers to `/Volumes/APDataStore/pact/ddm_ane3_residue/` (ExFAT — payload blobs only; `.mlpackage` is a DIRECTORY tree → keep those on Vertigo only if ≥ 10 GiB free, else in `experiments/results/ddm_ane3_*` with a certify record). Both SSDs are near full: check `df -h` before every launch.
- VERIFIED-AT-SOURCE LAW: the MIL dtype ids, the 3.3e-05 bar, 75% ANE share, hd128 7.02e-5 — mark `verified-at-source:` in ane2's memo/code or re-measure.
- EQUATIONS-LEG LAW: anchors ONLY through `update_equation_with_empirical_anchor` on `scorer_fp16_drift_by_axis_v1`; run the Catalog #344 check before the final message.
- DETACHED >30-MIN COMPUTE: n600 decode + segment runs exceed 30 min — `tools/launch_detached_process.py --output-dir <run_dir> --done-receipt ddm_ane3_<stage>.done --nice 10 --nice-best-effort -- <cmd>`; ≤ 3 processes (the frontier arm's CPU shards have priority; ANE runs are light on CPU); artifact-bound waits ≤ 780 s; never `nohup`/`&`/clock waiters. No Metal.
- Checkpoint discipline: `tools/subagent_checkpoint.py --subagent-id ddm_ane3 …` every ~10 tool uses; `read` first (ane2's checkpoints are the predecessor context).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- Uniform fp16 on the ANE FAILS the scorer (flip rate above bar; PoseNet dim-0 error pinned 0.150–0.154 for every fp32-tail split) — `.omx/research/ddm_ane2_engineer_the_precision_drift_20260905.md`; memory `fp16_scorer_drift_inverts_the_intuition_ane_closed_20260905`. Do not re-run uniform fp16 as a candidate; it is the control.
- Both of ane2's candidate mechanisms for the tail worsening died to its own instruments (cast census 1–8 casts; exactly 1 ANE segment) — directive ITEM 2. Start from the per-op activation diff, not from those two hypotheses.
- Pose on the ANE is REFUSED (hd128 head cure 7.02e-5 self-MSE = 11× d_pose) — directive ITEM 3.

## OPTIMAL FORM

- Family exemplar: ane2's selective-fp32 ladder and n600 flip-rate instrument (memo `.omx/research/ddm_ane2_engineer_the_precision_drift_20260905.md`, its receipts under the ane2 tree) — the reference form: n600 generated-decode, frozen cpu_torch argmax as authority, per-pair ledgers. Pin: the ane2 memo's landing commit `769f4e0cd` (receipt path `.omx/research/ddm_ane2_engineer_the_precision_drift_20260905.md`).
- SCOPE reductions declared per row: n120 may SIZE a run; every verdict row is n600 (SCOPE, per the charter law "n600 or it is not evidence"). MECHANISM reductions FORBIDDEN: no proxy flip metric, no fp16 output dtype left declared.
- **PRIOR-LAW PREDICTION (falsifiable):** the fixed-path g13stem flip rate is ≤ the provisional upper bound (≤ 3.3e-05) and ≥ 0.5× it (the output-dtype defect inflated, not deflated, the rate); ITEM 2 localizes ≥ 80% of the dim-0 delta to one op in the FastViT-T12 stem/patch-embed region (0:6). FALSIFIER: the fixed rate EXCEEDS 3.3e-05 (pass withdrawn), or no single op reaches 50% of the delta (scoped negative: the error is distributed).

## DELIVERABLE

`.omx/research/ddm_ane3_residue_fixed_conversion_remeasure_20260908.md` — rows: backend · ANE share · n120 rate · n600 rate · agreement · verdict; the ITEM 2 op table; registration status. Commit via the serializer; register owed items with `tools/extract_canonical_tasks_from_directive.py --directive <memo> --register-all --owner ddm_ane3`. End with `sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]` (unchanged by this arm).
