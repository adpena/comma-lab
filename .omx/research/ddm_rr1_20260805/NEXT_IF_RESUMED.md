# NEXT_IF_RESUMED

1. Treat rr1 round 5 as clean-pass `1/3`. The next pass is round 6; it needs two more zero-finding recursive review rounds before the review arm can be sealed clean.
2. Keep RR1-R4-F1 endpoint-blocking for any full-window EMA-cured claim: the 60-epoch full continuation inherited the 8-epoch smoke EMA decay (`0.9966666667`, `U=1200`) instead of re-anchoring for the full window (`0.9995555556`, `U=9000`). Only consume it as a `smoke-EMA continuation` with live/EMA endpoint probes, or regenerate/restart with a fresh full-window reanchor.
3. Before merging DY1/#961, enforce declared-vs-resolved scope laws in production with geometry/hash awareness, not name-only inertness. Include a positive control that full-window declared EMA scope refuses when the only resolved row is the inherited smoke-scope row.
4. Treat `/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/tr1_jd3_v3_smoke_entry_ep1336` as mixed-scope: entry smoke plus full continuation. Use `smoke_snapshot_ep1344/` for two-smoke adjudication evidence; use the top-level telemetry/checkpoints only for the continuation.
5. For any future full continuation, create a unique trainer `--out-dir` or add an explicit manifest field proving same-dir continuation was intentional and safe for consumers.
6. If the full-v3 endpoint is harvested before round 6, verify the receipt labels inherited `jd1_realized_hold` and pose-controller state as inherited/static where applicable, and confirm no endpoint table silently calls them freshly full-window-derived.
7. Add peak RSS/VRAM or ru_maxrss capture to the governed launcher/manifest if future recursive-review rounds consume measured-runnability.
8. Preserve scratch probe variants or command manifests whenever committed-source physics is reused with CKPTS/path edits; the JD3 variants are now preserved.
9. In future smoke verdicts/tables, split `source_checkpoint_epoch` from `saved_reanchor_checkpoint_epoch`; do not use `smoke_start_*` tags alone for epoch-delta or slope derivations.
10. Continue endpoint governance from the live full-v3 window only after the owner harvests it; rr1 did not claim the scorer slot and did not inspect a completed endpoint.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
