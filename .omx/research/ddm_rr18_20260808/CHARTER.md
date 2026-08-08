# ddm_rr18 — recursive adversarial review ROUND 18 (counter 1/3 after rr17 CLEAN; seal=3)

**Window under review (the ARM-CAP endpoint + ARM-VEH fire window, 08-07 22:00 -> 08-08 02:00):**
1. ARM-CAP n32 COMPLETE at step 6000 (SAFE_RUN ok exit=0, 5920s) + the two resume-path
   clearance fixes 710c4e131a (final-stage mem-probe sample) + 900cac49e0 (budget floor =
   executed window). Verify both fixes against the trainer source + probe v3 receipt
   3fbc65cb (were they CORRECT cures or symptom patches?).
2. THE ENDPOINT FACETS PASS (.omx/research/ddm_mx1g_20260807/endpoint_facets/): status passed,
   24 ckpt rows + K={2,4,8} tail rows, anchor abs_diff 0.0 at step 1500. **rr16's NOT-CLEAN
   demand was cache identity/hash/argv IN THE COMMITTED RECEIPTS — verify the endpoint facets
   receipts carry cache-bound provenance (cache_load block: paths+hashes+argv), the EXACT gap
   rr16 flagged on mx1t. If absent, that is a FINDING (counter resets).**
3. RR11-F1 REPLAY + cure: probe reauthors collapsed ARM-VEH input-cache tq1c->gt; corrected
   reauthor + fail-closed veh==cap guard 9fb26488c4. Verify the guard actually refuses (read
   the code; does Path.resolve() handle the SSD symlink cases?), verify the live ticket
   carries tq1c on all 4 veh keys, and adjudicate whether ARM-CAP's completed run was
   CONTAMINATED by any reauthor (it was not re-fired after the collapse — verify from safe_run
   receipts + stage-ckpt mtimes).
4. wc1 bench consumption: fp16-train 1.247x adopted for n120 ONLY, n32 arms lever-free
   (A/B symmetry). Verify the bench receipts support the sanity-delta-is-noise claim
   (cross-variant spread vs claimed 4.8e-7) and that no lever flag leaked into the ARM-VEH
   fire argv (read fire_argv_final.json in ddm_mx1g_20260807/fire_arm_veh/).
5. ARM-VEH fire custody: mem-probe receipt passed (peak 1.38GiB) -> fire-guard verdict ->
   fired argv with rss 45000/projected 21 SUBSTITUTED for REQUIRES_FRESH_MEM_PROBE
   placeholders. Adjudicate the substitution: is carrying ARM-CAP's derived values to ARM-VEH
   legitimate (same physics measured by the VEH probe) or a cross-arm constant transfer
   (cross-regime-constant-transfer genus)? If the latter, name the correct derivation.
6. The knee finding (steps 4250->6000 flat/regressive): verify from the receipts; adjudicate
   the derived n120 step-count recommendation (~4500) — is it licensed by ONE arm's curve or
   does it owe the ARM-VEH curve before n120 consumes it?

**Verdict:** CLEAN (counter 2/3) or NOT-CLEAN with findings (counter resets 0/3). Findings ->
.omx/research/ddm_rr18_20260808/ROUND18_FINDINGS.md. Assumption-challenge axis REQUIRED
(what shared assumption does this window operate within?). Fix trivial findings in-arm
(serializer, review x2); route structural ones to MAIN with named owners.

**Boundaries:** CPU-only, NO Metal, no scorer slot; read-only toward the LIVE ARM-VEH run dir.

**Discipline:** serializer + POST-EDIT --expected-content-sha256 per file; tags
[no-triality] [p0-ledger-ok]; review_tracker x2 per .py; NO Claude/AI attribution or
Co-Authored-By trailer. If serializer hits sandbox git-perms, write artifacts + say so.
