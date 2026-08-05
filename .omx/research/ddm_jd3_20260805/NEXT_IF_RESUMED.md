# NEXT_IF_RESUMED ddm_jd3

Start state: build and v3 tickets exist; both local bounded smoke attempts were blocked before training by missing Metal access.

Fire order:

1. Re-run `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/jd3_ticket_v3_entry_ep1336.json` through `tools/launch_detached_process.py` on a Metal-access host.
2. After it exits, re-run `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/jd3_ticket_v3_refuse_final_ep1354.json` the same way.
3. Before selecting a resume start, consume `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/chain_both_bases_sweep.json` if MAIN has landed it.
4. Do not full FIRE from this receipt. MAIN owns full FIRE after smoke evidence exists.

Selection evidence required: real post-engagement realized gates, jd3 floor latch, rollback/retreat history if any, active stage EMA provenance, live-vs-EMA gate telemetry, and final `tr1_window_receipt.json`.

v4 riders queued, not v3-stacked: SL2 solved-frame teacher distill, PE3 conditioning-only, and EN1 margin-weight at the next clean window boundary. Preserve single-variable discipline: v3 is only realized hold plus stage-scoped EMA, both cures for measured jd1 v2 defects.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
