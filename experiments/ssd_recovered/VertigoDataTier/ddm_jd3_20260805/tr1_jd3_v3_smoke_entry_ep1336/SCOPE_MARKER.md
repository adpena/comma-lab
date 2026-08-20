# SCOPE: entry_smoke_plus_full_continuation (RR1-R3-F1)
This directory holds TWO runs: (1) the jd3 v3 ENTRY SMOKE (epochs 1337-1344; its adjudication
evidence is preserved VERBATIM in smoke_snapshot_ep1344/ — use THAT for smoke claims) and
(2) the FULL v3 window continuation (epochs 1345-1406, fired 2026-08-05, launcher dir
full_v3_entry_cont_mainlaunch_r2). Top-level telemetry.jsonl is APPEND-MIXED across both;
tr1_window_receipt.json will reflect the FULL window at endpoint. Never cite top-level rows
as "smoke" evidence. Same-dir continuation was deliberate (in-place resume); future full
continuations get unique out-dirs (regenerator debt, with the argv-interpreter fix).
