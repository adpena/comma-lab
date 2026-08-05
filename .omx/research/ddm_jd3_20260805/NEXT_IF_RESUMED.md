# NEXT_IF_RESUMED ddm_jd3

Start state: build and v3 tickets exist; both local bounded smoke attempts were blocked before training by missing Metal access.

Fire order:

1. Re-run `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/jd3_ticket_v3_entry_ep1336.json` through `tools/launch_detached_process.py` on a Metal-access host.
2. After it exits, re-run `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/jd3_ticket_v3_refuse_final_ep1354.json` the same way.
3. Do not full FIRE from this receipt. MAIN owns full FIRE after smoke evidence exists.

Selection evidence required: real post-engagement realized gates, jd3 floor latch, rollback/retreat history if any, active stage EMA provenance, live-vs-EMA gate telemetry, and final `tr1_window_receipt.json`.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
