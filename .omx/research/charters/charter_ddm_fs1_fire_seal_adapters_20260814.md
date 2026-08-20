# CHARTER — ddm_fs1_fire_seal_adapters (2026-08-14, the two MAIN-blocked fire gates made fireable — scorer-free, no Modal)

CONTEXT (recall, do not re-derive). TWO sealed T4 gates await MAIN fire and
BOTH are blocked by build defects, not physics:
(1) **mc36 Variant C** (THE nameable-row candidate: 37 seg flips · +17 B ·
pose −1.463e-10, ALL local gates pass, projected net ≈ −2.0e-5 — above the
1e-5 naming bar). Its SEALED_FIRE_ORDER names the proven QS1/RE1T dual-axis
worker, but its SEALED_REQUEST.json is order-metadata, NOT the dispatcher's
`ddm_qs1_t4_dual_axis_request.v1` schema (`experiments/ddm_qs1_modal_t4_dual_axis.py::load_sealed_inputs`
:56-81 requires: schema string · resume_from==run_id · retain_pose_vectors
true · score_claim false · promotion_eligible false · inputs census EXACTLY
{candidate_archive.zip, candidate_runtime.zip, POSE_SCREEN_RESULT.json}).
(2) **mt1 sign gate** — its arm-written dispatcher
`experiments/ddm_mt1_modal_multitoken_sign_gate.py` fails at Modal image
build: "image tried to run a build step after using image.add_local_*"
(MAIN reproduced it twice; the arm could never execute Modal so it never
surfaced — dt1-census genus: worker chain never executed against real
provider).

## THE WORK (both legs scorer-free; MAIN fires after landing)

1. **Recall first**: `experiments/ddm_re1_pose_leg_seal.py` (the BANKED seal
   builder — mirror its seal() structure EXACTLY: request fields :100-140,
   canonical bytes, atomic writes, the dispatcher-validation self-check at
   the end via `dispatcher.load_sealed_inputs`) + the retained qs5/qs2
   SEALED_REQUEST.json files under /Volumes/VertigoDataTier/pact/ (the
   dual-axis single-shot precedent — mc36 is dual-axis like qs2/qs5, NOT
   two-stage like re1; check how their POSE_SCREEN_RESULT/local placeholders
   were shaped) + `experiments/ddm_js1b_modal_cuda_argmax_field_materializer.py`
   record helpers + the mc36 fire order + pinned inputs at
   /Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/fire_order/
   (archive f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de
   186,269 B · runtime 64e4642d30b436e6393d5573efcb579a13f922726566790efad40bc2ca117545
   238,713 B).
2. **LEG A — mc36 conformant seal**: build
   `experiments/ddm_mc36_dual_axis_seal.py` (preparation-only, no Modal, no
   claims — the re1 builder's exact pattern with mc36's identity: run_id
   `ddm_mc36_dual_axis_t4_r1`, lane `ddm_mc36_dual_axis_t4_n600_20260814`,
   the mc36 candidate/runtime records, local-evidence payload carrying the
   arm's measured advisory triple (37 flips · +17 B · −1.463e-10) under the
   worker placeholder law (local_pose_delta literal 0.0 + pose_unmeasured
   true if that is what the qs5 dual-axis precedent used — MATCH the
   precedent, do not invent). SELF-CHECK MANDATORY: call
   `dispatcher.load_sealed_inputs(request_path, input_root, sha)` at the end
   — the seal is DONE only when the real dispatcher accepts it. Emit the
   exact `modal run` command block for MAIN.
3. **LEG B — mt1 dispatcher image fix**: minimal edit to
   `experiments/ddm_mt1_modal_multitoken_sign_gate.py` — reorder the image
   chain so all `add_local_*` calls come LAST (preferred, per Modal's own
   advisory) or add `copy=True` where a build step must follow; NO other
   behavior change (the sealed request/SHA gate must remain valid —
   dispatcher edits don't need reseal per the dt1 law, but VERIFY the
   `_load_sealed`-equivalent gate in that file only checks request+payload
   SHAs, not dispatcher source). Static-validate by constructing the Image
   object in a local dry test if importable without Modal auth; else a
   focused unit test on the chain order.
4. **Tests**: seal round-trip (load_sealed_inputs acceptance = the real
   gate) · placeholder-law conformance · mt1 image-chain order regression.
5. **Two-landing (dt1 genus)**: add the image-order defect to the
   worker-dependency-closure census seeds — a static check that flags
   `add_local_*` followed by build steps in `experiments/*_modal_*.py`
   (warn-only, same-line waiver) IF cheap; else record it as a typed
   worklist row for the dt1 recurring census.

## OPTIMAL FORM

Family reference PINS (receipts): re1 seal builder 9207d5eac0 ·
qs1 dispatcher load gate :56-81 · mc36 archive f0ba4bb4… @186,269 B ·
mc36 commit 2e4abc6210 · mt1 commit af56d51c48 · base instrument (34,970
flips · d_pose 6.885642960696714e-6 · 186,252 B). MECHANISM reductions =
TOY-BRACKET: a seal that skips the dispatcher self-check · inventing
placeholder semantics instead of matching the qs5 precedent · editing the
mt1 SEALED_REQUEST (only the dispatcher image chain may change). Payload
law DEF CON 1000. Arms cannot RUN Modal — build + validate locally; MAIN
fires. Git-blocked → declare memo SHA for MAIN handoff.

## OUTPUT

Both build files + tests + `.omx/research/ddm_fs1_fire_seal_adapters_20260814.md`
(exact fire commands for BOTH gates, serially ordered mc36-first). Commit via
`tools/subagent_commit_serializer.py` (post-edit shas, `[no-triality]
[p0-ledger-ok]`, no co-author trailer). End with NEXT_IF_RESUMED +
LIVE-HYPOTHESES + DEAD-ENDS.
