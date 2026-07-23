---
title: DDM E1 runtime exporter Build 636
date_utc: 2026-07-23
lane_id: lane_ddm_e1_runtime_exporter_20260723
task: 636
research_only: true
execution_allowed: false
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
main_landing_review_required: true
---

# Outcome

Build one deterministic, fail-closed compiler from the sealed v15/J2 receiver
state to the contest packet:

`composed state -> archive.zip + inflate.py + inflate.sh -> camera raw`

The packet is a shipping proof, not a score or promotion claim. MAIN must
review the isolated landing before it can enter an exact-eval lane.

# Input identity

- v15 source archive: exactly 133,941 bytes and SHA-256
  `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`.
- J2 Lane seed: six canonical `DDLP1` records, compiled before any Lane
  coordinate trains.
- Receiver-effective lift dimension: **368**, not the stale J2 receipt's 706.
- State name: `v15_j2_lane_seed_theta0`.
- Pair count: 600; scorer grid: 384x512; camera: 874x1164; two frames/pair.

# Counted packet

The deterministic stored ZIP has exactly these ordered members:

1. `manifest.json`: canonical section identity, byte/hash custody, runtime
   contract, false-authority fields.
2. `base/chart.ddb`: Brotli-q11 semantic anchors, gradients, and residuals.
3. `semantic/composed.dds`: Brotli-q11 uint8 composed semantic plane after
   applying the complete 368-DOF state, canonical paint order, G1 replace
   semantics, Lane road-adjacency rule, and the current one-cell shared
   templates.

The manifest also declares five separate, typed, independently framed section
slots for future amplitude fields, per-element tolerance duals,
texture-quotient residual statistics, coder/probability parameters, and
realization-map metadata. Each slot has its own member prefix, schema, DDE1B
framing, and byte/SHA rate custody. All five are explicitly inactive in this
sealed state, so the packet pays no phantom no-op section bytes; activation
requires a matching receiver hook and new exact proof.

The manifest carries `ddm_composed_language.v1` plus ordered per-block version
stamps. Active L, D4, and D6 stamps name the exact input-member hashes,
zero-drift validity horizon, and fail-closed consumption policy; inactive D1,
D2, and D5 stamps refuse unstamped activation. The receiver verifies these
against the live extracted bytes at read time and repeats the accepted stamps
in its receipt.

The plane is a counted sufficient statistic, not hidden receiver code. It
removes inherited structured bytes that the G1 replace rule would otherwise
parse and override; therefore every exported payload byte reaches the camera
output. Generic parsing, integer chart expansion, palette placement, bicubic
camera realization, checkpointing, and raw assembly live only in `inflate.py`
under rule 118. The runtime has two dependencies: Torch and Brotli. It
contains no scorer, scorer weights, GT argmax table, per-video constant, or
encoded payload.

# Fail-closed receiver contract

- exact archive member order, canonical manifest bytes, section length and
  SHA-256;
- every section read once and every subrange contiguous, non-overlapping, and
  exactly consumed;
- decoded-length and decoded-SHA assertions for Brotli sections;
- no trailing bytes in any fixed-row or subcontainer grammar;
- every paid semantic section reaches the camera renderer;
- single-thread Torch CPU, no device fallback;
- storage preflight before bulk creation;
- write-once per-batch stage bytes and canonical checkpoint JSON;
- contiguous-prefix resume only;
- final raw assembled to a temporary file, fsynced, size/hash checked, then
  atomically replaced;
- final runtime receipt includes the section-consumption bijection, archive
  identity, raw identity, timing, and false-authority fields.

# Verification

The n600 proof exports twice and requires identical archive/runtime bytes,
inflates from a clean extracted directory, and compares the complete packaged
raw stream to the in-repo seeded v15/J2 receiver in canonical batch order.
Exact raw identity transfers the frozen-scorer row only after the source
scorer row is measured on that exact seeded state. A v15-only score may not be
borrowed if the Lane seed changes camera bytes.

Decode timing is single-thread CPU and must be below 1,800 seconds. Proof bulk
goes to the SSD waterfall. Batch stages remain preserved; cleanup may remove
only certified scratch.

The actual frozen upstream `evaluate.sh` and `evaluate.py` must also run on an
SSD-only copy of the exact packet. This rehearsal is interface and CPU-advisory
evidence, not contest-axis authority. It records upstream git/source hashes,
exact argv, logs, report fields, raw identity, and wall clock. Failed first
contacts remain durable evidence rather than being overwritten.

# Realization-map Jacobian

`ddm_semantic_paint_camera_uint8_jacobian_v1` derives the exact floor-index
camera preimage of every scorer-grid label and measures the active support of
each semantic paint coefficient on the named n600 state. A legal unit palette
coefficient perturbation changes exactly one uint8 output byte at every active
preimage pixel; label assignment support is 8 to 18 camera pixels across the
two frames per scorer cell. Its input hashes and exact-equality validity
horizon are part of the export and verification receipts.

# Recursive joint-refinement contract

Every packaged state records the fixed-budget block order
`L -> D2 -> D1 -> D4 -> D6 -> D5`, per-block byte custody, realized
`d_seg`, realized `d_pose`, total S, and an argmax-agreement proxy in the
verification receipt. The stop law is one complete joint cycle with no net
gain at constant archive bytes. Build #636 measures cycle 0 only and therefore
records `OPEN_ONE_MEASURED_EXPORT_CYCLE_ONLY`; it does not claim a fixed point
or family negative, and successor optimization cycles remain owed.

# Triality

- DSL/config: `ddm_runtime_exporter_config.v1`, CLI accepts only `--config`.
- DAG: the dated Build #636 findings/feed produced after the n600 proof.
- Equation: `ddm_runtime_export_identity_receiver_closed_v1`, the byte-level
  identity law between the in-repo composed receiver and packaged inflate,
  plus `ddm_semantic_paint_camera_uint8_jacobian_v1`.

# STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`;
`.omx/research/BUILD_SPEC_v10_compiler_receiver_20260718.md`;
`src/tac/witness_dsl/v10_production_receiver.py` (#402 exact-consumption,
storage, atomic/final-byte discipline);
`.omx/research/ddm_c1_composed_candidate_spec_603_613_20260723.md`;
`src/tac/optimization/direct_description_joint_descent.py`;
v14/v15 receipts; J2 receipt; lane registry;
subagent progress; per-arm and broadcast inboxes.
