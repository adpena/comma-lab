# CHARTER_ADDENDUM — ddm_mx1d (from RR10-F2, HIGH, filed mid-flight 2026-08-07)
rr10 audited your in-progress guard: the entrypoint-side check as built TRUSTS schema/status
fields of the receipt JSON — a stale or hand-written {"status":"passed"} bypasses it.
REQUIRED before you land (RR10-F2 cure): the entrypoint validation must BIND the receipt to
this fire — (a) freshness window (receipt mtime within N hours, N stated + derived);
(b) host identity match (uname/hardware fingerprint recorded in receipt == current);
(c) config binding (pairs/caches/init shas in receipt == argv);
(d) budget binding (receipt's applied hard-cap == the cap this fire will apply);
on ANY mismatch REFUSE with the mismatched field named. Do NOT merely check status=="passed".
Also (RR10-F1 consumer): add a `review_interlock` field to the fire-guard verdict —
the guard REFUSES if a live review arm's charter names this fire class and has no landed
verdict (read .omx/state/codex_arm_queue state for live rr* arms + their prompts).
Full findings: .omx/research/ddm_rr10_20260807/ROUND10_FINDINGS.md
