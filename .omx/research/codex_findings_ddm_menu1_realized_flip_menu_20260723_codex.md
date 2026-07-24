---
title: Codex findings - DDM MENU1 realized-flip menu
date_utc: 2026-07-23
lane_id: lane_ddm_menu1_realized_flip_menu_20260723
research_only: true
execution_allowed: true
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: MENU1_MEASURED_BOX_NOT_REACHED
verdict_scope: "FORMULATION: exact V19C active-base menu, amplitude alternatives, fixed paint composition, and coarse top-cluster sidecar; families remain open"
pointer_moved: false
main_landing_review_required: true
---

# Finding

MENU1 compiled all 15,894 SN1-cluster/fix rows and closed five active V19C
measurement rows plus two named formulation bridges. The #613 box is not
reached.

The measured joint pool winner is the 974-byte frame1 local-statistics +
hard-placement + analytic-coverage arm:

```text
B = 138,801
errors = 8,318,787
d_seg = 0.07051923116048177
d_pose = 36.6181847780574
S = 26.28022355199344.
```

Its Pose improvement outweighs catastrophic Seg damage in the joint action.
The axis decomposition must therefore remain visible: this is not a Seg win.
The accepted endpoint is 8,181,948 errors and `0.06935923116048177` d_seg
above the box. MyCar binds with 4,072,489 errors.

# Premise falsifications and fixes

Round-1 review found two material defects before finalization:

1. The delegated `2,265,811-error V19C base` is the residual
   Road/Undrivable/MyCar bucket, not total d_seg errors. The pinned receipt
   yields 2,923,991 total errors. Code and receipts now type both.
2. The first runner applied `gt_f1`-fitted paint to both frames. Frame 0 has no
   corresponding RGB target and is part of PoseNet's two-frame input. The
   final runner preserves frame 0 byte-identically and applies all fitted
   paint to frame 1 only.

Invalidated receipts were preserved on SSD with explicit round names. They are
not in the final branch evidence.

# Measured price table

| row | pool | counted bytes | corrected | introduced | net correction | d_seg | d_pose | disposition |
|---|---|---:|---:|---:|---:|---:|---:|---|
| scalar gain/bias | paint amplitude | 12 | 136,350 | 95,984 | 40,366 | 0.02444479200575087 | 159.39533299820565 | same-pool dominated |
| temporal 16-knot affine | paint amplitude | 204 | 183,585 | 104,697 | 78,888 | 0.0241182369656033 | 150.74417503260625 | same-pool dominated |
| class x row-band statistics | paint amplitude | 974 | 909,949 | 24,889,951 | -23,980,002 | 0.22806797451443142 | 27.418160360123842 | same-pool dominated |
| statistics + hard + analytic | paint pipeline | 974 | 738,997 | 6,133,793 | -5,394,796 | 0.07051923116048177 | 36.6181847780574 | joint pool winner |
| coarse top-cluster prototype | semantic target | 88,568 incremental | 405,465 | 5,475,423 | -5,069,958 | 0.11349779764811198 | 34.606913333214735 | over budget and no joint gain |

The new sub-30-byte and temporal rungs are real improvements on both Seg and
Pose. Their marginal evidence is durable even though they lose the same-pool
joint competition.

The PT1 spectrum arm remains dominated by PT1 statistics under their shared
control: 186 B and 1,034,847 errors versus 30 B and 1,016,725 errors. Those
numbers are cross-control labels, not V19C prices.

# Targeted formulation verdict

The top SN1 row is Undrivable-to-Road, `ANNULUS_2_TO_5`, no-Lane-curve,
coarse-description, G3-tail. The measured sidecar represents only ordered
pair, boundary band, and pair support; it does not encode `d2`, historical
G3-tail, or semantic-history axes.

The generic Road prototype worsens the parent action:

```text
B_total = 227,369 > 200,000
S = 30.104108909525713 > 26.28022355199344
Delta errors_realized = -5,069,958.
```

Verdict scope is FORMULATION. The family and paradigm remain open. The next
first rung is a Fisher/margin-ranked, corrected-inner-Jacobian targeted
actuator with exact cluster-axis representation, not mask compression of this
failed prototype.

# Rate partition and external pools

Every compiled and measured row now has `COUNTED`, `FREE`, and `NULL`.
GT-fitted scalar, temporal, local, and target-mask values are COUNTED.
Deterministic decoder logic carries zero video-derived payload.

The pinned AT1X manifest, SHA
`251cc1e4268fb909a9f9a3ac2af845614c98aab15948f90d33d43a8c1542a1d9`,
reports zero amplitude factors and no through-R uint8 survival row. The
BN-expected-stat-to-camera-affine bridge is therefore blocked, not FREE.

The pinned PA1 receipt contributes three distinct cross-control
`pose_amplitude` rows. The frame0 scorer-stat row has strong PA1 evidence but
is `FREE_candidate`, pending receiver survival. No PA1 delta is added to the
V19C curve, and frame1-touching PA1 moves retain their measured Seg collateral.

# Reproducibility and custody

- final receipt:
  `.omx/research/ddm_menu1_realized_flip_menu_20260723T214943Z/ddm_menu1_realized_flip_menu_receipt.json`
- final receipt SHA:
  `2fc12eb505aa7de140b5e785e5fb528c349fd72b423a1821f23b70ea21d6f29d`
- compiled menu:
  `/Volumes/VertigoDataTier/pact/ddm_menu1_realized_flip_menu_20260723T214943Z_audit_ext_v2/compiled_realized_flip_menu.jsonl.gz`
- compiled menu: 440,953 B, SHA
  `e1d446376e4b29e54b061d6579bb2a56cbc5357f33dc8cdd371672f627b441f1`
- compiled cardinality: 15,894 unique rows and 2,649 unique clusters
- typed measurement config SHA:
  `5a8ce892bafbd4da710a3e5bfcb7ee35c09f1c7c208c665eadf864ed8c6d1db2`
- full evidence config SHA:
  `b5c9a4d084870263f5a0e9487b775008146e9a434b7b12928214dbe3f42709e1`
- all five candidate arms: 38 immutable n600 batch checkpoints each
- final cache/receiver/scorer execution: local macOS CPU only
- pointer: `0.1910828242 [contest-CPU]`, unmoved

Independent re-derivation verified row and cluster uniqueness, objective
arithmetic, transition conservation, byte-partition conservation, dual target
rejection, false-authority labels, and pointer immobility.

The first clean replay also exposed that the receipt embedded the volatile
live free-space byte count. That invalid receipt was preserved on the SSD by
its SHA, the immutable receipt now records only the checked lower bound plus
the check policy, and two subsequent executions produced the same receipt
SHA. A regression test varies live free space and requires identical preflight
receipt content.

Serializer preflight then correctly refused the compiled gzip as a gitignored
bulk artifact. The exact menu was moved losslessly into the SSD checkpoint
root, the superseded receipt was preserved there by SHA, and the final
repository receipt now binds the SSD path, byte count, and menu SHA.

The target lane is internally consistent at L1 with implementation,
strict-preflight, and three-clean-review gates. Global lane validation still
reports 110 pre-existing missing-evidence-path errors outside this lane; this
landing does not rewrite or conceal that registry debt.

# STORES CONSULTED

- `CLAUDE.md`
- `AGENTS.md`
- `docs/operating_manual_craft_handoff.md`
- the delegated authority file
- SHA-pinned SN1, PT1, E2, DR2B, C1, and V19C receipts
- AT1X SSD manifest
- PA1 MAIN receipt
- per-arm inbox through row 3
- fleet broadcast through row 57

# MAIN review required

MAIN must review the entire base-to-branch diff and independently rederive:

1. all content hashes and the V19C residual/total split;
2. exact receiver, frame1-only affine semantics, and frozen scorer paths;
3. payload parse-back and COUNTED/FREE/NULL classification;
4. 15,894-row completeness and stable identities;
5. same-pool competition, transition conservation, and objective arithmetic;
6. AT1X and PA1 bridge/cross-control handling;
7. verdict scope, MyCar routing, SSD custody, advisory labels, and pointer
   immobility.

Do not merge or promote on this arm's assertion alone.
