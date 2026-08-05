# NEXT_IF_RESUMED

1. Treat rr1 cycle-2 round 3 as a clean pass; the cycle-2 counter is `1/3`. The next pass is cycle-2 round 4, and two more consecutive zero-finding rounds are still required before the JD4 review cycle seals.
2. Do not re-open RR1-C2-R2-F1 as a new issue: the raw SSD n600 receipt/variant remain append-only and stale, while the committed source class fix is real. Consume the existing n600 numbers only with `.omx/research/ddm_rr1_20260805/jd3_endpoint_n600_axis_correction.json`; future probes must use or copy the fixed committed `experiments/ddm_jd1_endpoint_verdict.py`, not `/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/jd3_endpoint_n600_both_bases.py`.
3. If reviewing the JD4 live run, start from `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd4_cont_mainlaunch/launch_manifest.json`, `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1406/telemetry.jsonl`, and `.omx/tmp/codex_runs/jd4_cont_ep1406.done`. In this round, `.done` was absent and process liveness could not be measured because `kill -0 90157` was sandbox-blocked with `operation not permitted`.
4. Re-verify that JD4 endpoint consumers wait for the done receipt and then run the both-bases n600 endpoint probe from the fixed committed instrument. Do not consume partial JD4 telemetry as an endpoint row.
5. Keep endpoint obligations unchanged: plateau policy Cases 0/A/B/C; both LIVE and EMA bases; dynamic-EMA A/B; no archive, byte-close, or contest claim until those artifacts exist.
6. Cycle 1 remains paused at `0/3` for the older JD3-chain scope. Do not confuse cycle-2 clean passes with sealing cycle 1.
7. Preserve all live SSD run dirs read-only. Any follow-on from the JD4 endpoint exits FIRED, FOLDED, or QUEUED-WITH-A-FIRE-ORDER.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
