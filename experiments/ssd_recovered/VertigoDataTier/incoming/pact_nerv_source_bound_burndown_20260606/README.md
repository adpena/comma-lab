# PACT-NeRV source-bound burndown package (2026-06-06)

Pinned repo ref reviewed: `69b4c523d2696978cae4423d95488d15d851e8cd` (`Align PR95 pose telemetry and EMA defaults`).

This package answers the partner-agent directive: no greenfield theory, only codebase-grounded findings with exact patch targets, failing tests, passing-test targets, smoke commands, value-per-byte accounting, and ranked burndown order.

## Executive verdict

**HiNeRV long training: blocked.**  
Minimum blocker: direct-live target-region min-ratio/class birth can remain zero while class/debt auxiliary actuators move. The existing loss has score-weighted unsolved argmax telemetry, but the actuator is still too indirect unless it is consumed by a scoped output-head/late-feature-grid birth step and a joint Seg/Pose trust region.

**SNeRV long training: blocked.**  
Minimum blocker: official MFU/HFR/TUB full source-forward closure is not proven. The current evidence says receiver primitives can decode, but `receiver_source_forward_replay_bound=false` and full trained TUB parity is missing.

**Shared long-run selection: blocked for PR95-source-faithfulness.**  
At ref `69b4c523d2696978cae4423d95488d15d851e8cd`, live-vs-EMA archive selection exports both candidates and checks SHA/bytes, but still selects by local proxy components + charged bytes. It does not require parse-back/replayed scorer components. PR95 selected by built archive parse-back. The minimum patch is an optional `archive_replay_components()` hook consumed only at archive-selection time, so MLX training velocity is preserved.

## Ranked burndown

1. `shared_parseback_selection` — patch `long_training_canonical.py` to select live/EMA by parse-back replay when adapter supplies it; add fail-closed mode for long runs.
2. `hinerv_target_region_birth_actuator` — add a scoped worst-region output-head + late feature-grid birth step using current score-weighted unsolved mass metrics.
3. `shared_seg_pose_trust_region` — add score-unit trust region using `100*Δd_seg`, `pose_marginal*Δd_pose`, and byte-price deltas.
4. `snerv_tub_full_source_forward_closure` — close unmapped temporal encoder and output2 decoder mapping; make parity proof real, not metadata-only.
5. `byte_value_ledger_required` — every actuator must emit `delta_score_per_byte` or explicit “no byte delta yet” blocker.
