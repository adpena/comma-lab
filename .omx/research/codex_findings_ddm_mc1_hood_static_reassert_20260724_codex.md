---
title: Codex findings - DDM MC1 static-hood reassert
date_utc: 2026-07-24
lane_id: lane_ddm_mc1_hood_static_reassert_20260724
research_only: true
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: MC1_MEASURED_INSTANCE_NOT_JOINT_POSITIVE
verdict_scope: "INSTANCE: base-byte reassert on three support encodings after the exact MENU1 winner; family and paradigm remain open"
pointer_moved: false
main_landing_review_required: true
---

# Verdict

Reject this measured MC1 instance. The best 139-byte single-static support
does what it was supposed to do on Seg: MyCar errors fall from `4,072,489` to
`2,553,139` and total errors fall from `8,318,787` to `6,571,730`. But it
destroys too much of MENU1's Pose gain: `d_pose 36.6181847780574 ->
64.85599367436599`. Joint advisory action rises from `26.28022355199344` to
`31.13027893413343` (`Delta S=+4.850055382139988`).

The negative is INSTANCE-scoped. It falsifies unconditional restoration of
base V19C hood bytes after this exact paint winner. It does not close a
one-time solved hood field, partial reassert, or PoseNet-stat-preserving
projection.

# What the measurement established

- Self-detection chooses class 4 from the V19C base argmax without a hard-coded
  class ID. The single-static support has mean per-frame IoU
  `0.9959164190251704`; the static-in-image premise is confirmed.
- The actual parse-backable static payload is 139 counted bytes. Per-frame
  storage is 58,026 bytes. The receiver-semantic support is zero-new-byte and
  needs no scorer at decode.
- Single-static is the strongest Seg formulation: it recovers `3,226,481`
  MyCar errors while introducing `1,707,131`, for net `1,519,350`.
- The same operation improves Road by `370,849` net and Undrivable by `20,689`,
  but worsens Lane by `10,784` and Movable by `153,047`.
- The Pose coupling is direct, not speculative: the reassert changes
  `45,397,200` official PoseNet preprocessed-input coordinates with L1
  `1,849,418,743.604599`. The hood's YUV6 statistics are part of the signal
  that produced the MENU1 pose gain.
- Reasserting V19C bytes onto identical V19C bytes is byte-identical on all
  600 pairs, so the V19C control stays at 37,237 MyCar errors; this operation
  cannot shrink the base's own bucket.

# Waterfill and first rung

No MC1 row is c1-waterfill eligible because all exact joint deltas are
positive. Same-pool alternatives compete and are never summed.

The next first rung is a solved static hood field or partial trust-region
reassert constrained to preserve official PoseNet input statistics while
recovering the measured MyCar mass. The acceptance test stays unchanged:
frame 0 byte-identical, exact n600 Seg/Pose, and negative joint `Delta S`.
The surviving rs1/#366 scope is `2,553,139` MyCar errors after the best
measured instance.

# Directive-consumption table

| authority | status | effect |
|---|---|---|
| delegated authority SHA `d0e234c87f09d3e2dcac850a1cac667214e216fb0153c003946f6e5aed26beed` | CONSUMED | Ran the one-mechanism n600 probe, priced three support partitions, retained joint Seg/Pose authority, and kept frame 0 immutable. |
| scorer-native doctrine points 1-8 | CONSUMED | Used frozen-scorer coordinates, exact receiver composition, non-additive pool law, n600-only evidence, and first-rung routing. |
| doctrine 9/9b upstream mine | CONSUMED | Reused MENU1's SHA-bound upstream modules/weights/evaluate composition and official PoseNet preprocessing; invented no scorer path. |
| `2026-07-19T19:42:07Z` reverse-waterfill directive | CONSUMED | Admitted no positive-Delta-S row; same-pool deltas were not summed. |
| `2026-07-19T19:48:01Z` Fisher/basis directive | CONSUMED-AS-BOUNDARY | This probe changed no residual basis; it used exact realized joint measurement and queued a Pose-stat-preserving formulation rather than a Fourier patch. |
| per-arm inbox | EMPTY | No superseding or stop directive was received. |
| fleet broadcast after 2026-07-23T00:00:00Z | EMPTY | No newer arm-specific directive was received. |

# Six-hook wire-in

- sensitivity map: per-class corrected/introduced rows and official Pose-input
  coupling are durable in the receipt; no global map mutation from a rejected
  research-only instance.
- Pareto constraint: exact joint `Delta S<0` is the binding admission law.
- bit allocator: c1 route records non-admission and the exact 139-byte price.
- cathedral/autopilot: not dispatchable because the pool is non-positive and
  authority forbids launch/promotion.
- continual learning: the empirical anchor is registered as
  `ddm_mc1_static_hood_reassert_joint_action_v1`.
- probe disambiguator: all three defensible support partitions were shipped
  and measured; no untested support-choice ambiguity was collapsed.

# STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, and `docs/operating_manual_craft_handoff.md`
- `.omx/research/ddm_scorer_native_doctrine_and_synthesis_20260723.md`
- MENU1 findings, DAG, directive table, equations memo, receipt, and SSD tree
- `src/tac/boundary_math/hood_static_component.py`
- canonical lane registry and current frontier pointer
- per-arm and fleet inboxes at every checkpoint

The first generated receipt (SHA `710b954617659ed6...`) used a descriptive
string in the `FREE` byte-count field. It was invalidated before clean review,
preserved under the SSD `invalidated_receipts/` directory, and replaced by the
typed receipt SHA `458043413339551fe785e605d54751c46fe0d8b24c7c4ee59a67426872e320e8`
with numeric `COUNTED/FREE/NULL` fields plus a separate `FREE_source`.

# MAIN review required

MAIN must review the entire base-to-branch diff and independently rederive all
hashes, support identity and rate partitions, frame-0 and outside-support
invariants, n600 transition arithmetic, official PoseNet coupling, joint
objective, verdict scope, canonical-equation registry row, SSD custody, and
pointer immobility. Do not merge or promote on this arm's assertion alone.
