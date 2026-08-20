# r9m runtime-tree hash mismatch — VERIFIED path projection, not drift (2026-08-04)

Attempt 3 (call fc-01KZ732JZ9BZJKGA43SET8V3C9) failed fail-closed:
expected=3ea13f96… (prepared pre-compaction, matches NEITHER current local nor remote) vs
remote actual=9982203ba6d00b2467b77f3719219e6030ab23221270a743920158810358700c.

Verification performed (MEASURED, this session):
1. Remote tree sha SELF-CONSISTENT: rebuilt tree_payload from the harvested
   provenance.json inflate_runtime_manifest fields → sha == 9982203b… exactly (no hidden fields).
2. Field-by-field local (sub_final @ /Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_final)
   vs remote (/tmp/modal_auth_eval_cpu/submission_dir): 10/10 runtime files BYTE-IDENTICAL
   (same relative paths + sha256), external_dependency_roots [] == [],
   upstream evaluate.py sha 7da71a84… identical both sides.
3. Sole content delta = repo_local_tac_import_manifest: local resolves 3 repo files for the
   FALLBACK import `tac.optimization.ddm_ix2_archive_container` (inflate_runner.py:84); remote
   records 0 files + unresolved because the tac package is a PythonPackage mount (sys.path),
   not under /workspace/pact/src. DECODE DOES NOT DEPEND ON IT: the module is VENDORED in the
   submission dir (ddm_ix2_archive_container.py, 28,5xx B) and the local import at
   inflate_runner.py:77 wins — the fallback line is a never-taken branch, statically scanned.
4. Current LOCAL tree sha = 083e09d5…, content sha = 8214d97e… (path-dependent hash by design:
   runtime_root_name sub_final vs submission_dir + out-of-repo absolute paths).

Conclusion: 9982203b… is the correct expected value for THIS archive on the Modal axis —
adopted only after mechanistic reproduction, never as expect-what-we-observed.
Attempt-3 remote cost: 20.7s CPU container (≈$0.00x), claim terminal-closed
failed_modal_cpu_auth_eval_no_score_claim.
