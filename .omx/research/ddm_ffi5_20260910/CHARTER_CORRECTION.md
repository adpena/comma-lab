# ddm_ffi5 CHARTER CORRECTION (MAIN, 2026-09-10 ~20:50Z) — fix 1 re-stated from the retained worker manifest

The dispatch ALREADY passes the worker's definition: `measure_t4_runtime_digest(rlc5 tree)` = e3d2371917920ce3… = the spawn's
`expected_runtime_tree_sha256`. The worker computed baeb53afc8d1bb0c… because the first-measurement worker extracts the runtime
under a Modal VOLUME root (`/__modal/volumes/vo-…/first_measurements/<authorization sha>/out/submission_dir/`, retained in
`/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run3/provenance.json` → `inflate_runtime_manifest`) while the local
predictor `tac.deploy.modal.auth_eval.modal_uploaded_submission_dir_runtime_manifest` rewrites paths to the NORMAL upload root
`/tmp/modal_auth_eval/submission_dir`. The dependency manifest's `runtime_tree_sha256` includes the root path → path-coupled
(r9m genus). MEASURED on rlc5's tree: `runtime_content_tree_sha256` LOCAL e1e6d1252b56b66b… = WORKER e1e6d1252b56b66b…;
`runtime_files_sha256` LOCAL fb1f6295a3527fdc… = WORKER fb1f6295a3527fdc…; file count 48 = 48. Only the path-coupled tree
digest differs.

**Fix 1 (replaces the charter's fix 1):** for first-measurement dispatches, the worker's runtime validation
(`experiments/contest_auth_eval.py::_validate_expected_runtime_tree`, reached from `experiments/modal_auth_eval.py`'s
first-measurement branch) must compare the CONTENT-ONLY digest (`runtime_content_tree_sha256`, which both sides compute
identically) and RECORD the path-coupled tree digest for custody; the local half passes `expected_runtime_content_tree_sha256`
(= local `runtime_content_tree_sha256`) and records both. Never loosen: a single content byte difference must still refuse.
Test: a manifest with a different root and identical content passes; a manifest with one changed file byte refuses. If
`contest_auth_eval.py` is not in `PREFIRE_IMPLEMENTATION_PATHS`, say so and pin it in the amendment manifest anyway.
Fixes 2 and 3 stand as chartered.
