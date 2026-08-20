---
title: DDM CB1 per-class carrier byte-close findings
date_utc: 2026-07-25T22:13:59Z
lane_id: ddm_cb1_perclass_carrier_byteclose
research_only: true
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
exact_eval: false
paid_dispatch: false
pointer_moved: false
competitive_target: "official PR130 displayed 0.172"
main_landing_review_required: true
verdict: CB1_HAS_STRICT_NEGATIVE_JOINT_ROW
---

# Outcome

The per-class byte-close `NO_VERDICT` is closed for two exact carrier
instances on the current merged RG4 source-local PC1 base. The MyCar static
mask is a strict-negative joint row and is eligible for the c1/#613
waterfill. The inherited polished v13 Lane program is strongly uphill on this
base and is rejected from that waterfill.

These are real n600 rows through the emitted CB1/E4 runtime, composite
receiver, uint8 frames, and frozen CPU-torch scorers. They are advisory
measurements, not contest scores. No exact contest evaluation, paid dispatch,
promotion, campaign fire, or pointer mutation occurred. The competitive
target remains the official PR130 displayed `0.172`; the local
`0.1910828242` row is only a custody-specific baseline and is not called the
competitive frontier here.

# Exact measured rows

All deltas are candidate minus the fresh byte-closed control. Lower is better.
The advisory objective is
`100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`.

| carrier | archive B | delta B | d_seg | delta d_seg | d_pose | delta d_pose | delta joint S | c1 disposition |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| control | 131,301 | 0 | 0.061912604437934025 | 0 | 31.281041046492344 | 0 | 0 | parent |
| MyCar static mask | 131,620 | +319 | 0.06190208435058594 | -0.00001052008734808707 | 31.10158359200554 | -0.17945745448680483 | -0.0516456148508837 | ADMIT |
| polished v13 Lane band | 132,831 | +1,530 | 0.09850494384765625 | +0.03659233940972222 | 53.99936657109412 | +22.718325524601777 | +9.21156940553832 | REJECT |

The MyCar row buys `0.0516456148508837` advisory score units for 319
counted bytes, or `0.00016189847915637525` score units per byte. The Lane row
has negative value per byte and is not a waterfill input.

# Per-class Seg accounting

`delta errors realized = control errors - candidate errors`; positive is a
realized correction. `delta d_seg = candidate - control`; negative is better.

| carrier | class | sites | delta errors realized | delta d_seg |
|---|---|---:|---:|---:|
| MyCar static mask | Lane | 690,639 | -121 | +0.0001752000683424848 |
| MyCar static mask | Movable | 1,460,325 | -10 | +0.000006847790731567827 |
| MyCar static mask | MyCar | 29,993,509 | +5,286 | -0.00017623813205717986 |
| MyCar static mask | Road | 27,407,046 | -4,161 | +0.00015182227227260314 |
| MyCar static mask | Undrivable | 58,413,281 | +247 | -0.000004228490435248186 |
| polished v13 Lane band | Lane | 690,639 | -114,886 | +0.16634739712063762 |
| polished v13 Lane band | Movable | 1,460,325 | -86,426 | +0.05918271617619364 |
| polished v13 Lane band | MyCar | 29,993,509 | -3,935,657 | +0.13121695764240188 |
| polished v13 Lane band | Road | 27,407,046 | -157,449 | +0.0057448365650205335 |
| polished v13 Lane band | Undrivable | 58,413,281 | -22,190 | +0.00037987936339340354 |

The MyCar mask corrects 5,286 MyCar sites and 247 Undrivable sites while
introducing 4,161 Road, 121 Lane, and 10 Movable errors. Net Seg improvement
is 1,241 sites, and Pose also improves. The joint gate therefore admits the
row; the localized collateral is retained rather than hidden.

The Lane instance makes every class worse and adds 4,316,608 net Seg errors.
Its Pose regression is also large. The typed failure is:

- `primary_leg=pose_survival`
- `paint_leg=receiver-paint collision with this merged RG4 source-local base`
- `collateral_leg=all five target classes regress`
- `quantization_leg=E4 coder loss excluded`: packet parse-back, parent
  parse/reemit, source-local parse/reemit, and final raw identity all pass.

This negative has `verdict_scope=INSTANCE`: the exact v13 six-periodic-program
plus 130-drift-knot payload, rewrapped on this exact merged RG4 base. It does
not close the broader Lane carrier family. A source-local Lane receiver whose
paint is conditioned on the RG4 base and Pose survival remains open.

# Carrier and class-order custody

The MyCar support consumed the MC1 139-byte static-majority support but did
not hardcode class index 4. The measurement tool rederived the target from
the support's spatial/static class evidence under the canonical class order
and required a unique winner. The emitted static rule stores the detected
target explicitly, parses back exactly, and occupies 146 bytes before outer
compression. It addresses 50,350 support sites. The final CB1/E4 archive
marginal is +319 bytes.

The Lane carrier consumed, rather than rederived, the polished v13 payload:
six periodic programs in 90 bytes plus 130 drift knots in 1,962 bytes. Its
2,052-byte semantic payload compresses to a +1,530-byte final archive
marginal.

# Byte-close and runtime proof

The source is the current merged RG4 source-local PC1 archive:

- bytes: `139685`
- SHA-256:
  `d86710793f776cb28144d9c7d817a3c9965e8160a1fc001c87deee206f9ccbf6`
- settled source metrics:
  `d_seg=0.061912604437934025`,
  `d_pose=31.281041321337113`

CB1 extends the proven E4 framed exporter for the RG4 state without changing
the E4 coding contract: deterministic Brotli-Q11 with the declared
ImportError-only LZMA fallback, exact ZIP member closure, double compile
identity, and a self-contained emitted runtime. The runtime consumes exactly
`manifest.json` and `state/rg4.ddr4`, parses and reemits the RG4 state and its
parent worldsheet byte-identically, renders in 19 resumable stages, and
preserves all stage checkpoints.

| candidate | archive SHA-256 | final raw SHA-256 | raw bytes | decode seconds |
|---|---|---|---:|---:|
| control | `a08e8f629ebd58921187ade2178bf13ead9123d5caa24f08b1ecd9b1de4b3211` | `9b41b650f6be21e2ae9822c41c2429140531e8c4af4d96c284f7ebaca6bcc373` | 3,662,409,600 | 533.813196 |
| MyCar | `5e1441180f83a6d1d12dc72b574d6801f815c555ede3ca2f56ccb228bc45c3b3` | `a6cee0402433f079107e890a4570541de8ff5171f9ac8b1ae1a716e2d02c4302` | 3,662,409,600 | 488.526400 |
| Lane | `8f22f22eaece59bd7e559162d09b37595ae2f450046b8d919c81543439d026c4` | `26e0d27e15dc3752d5029fabaace1a51039b169c56eb8907a4d4c99d98861dcc` | 3,662,409,600 | 480.332087 |

All three decodes remain below the 30-minute budget. The scorer used the
SHA-pinned frozen `modules.py`, SegNet, and PoseNet files recorded in the
measurement receipt. It preserved 38 JSON and NPZ checkpoints per candidate
and replayed the first batch deterministically. Independent aggregation of
all 114 batch JSON receipts exactly reproduces all total and per-class error
counts; floating Pose sums agree to ordinary reduction-order precision.
Source rendering used the sealed batch-32 receiver geometry; scoring used the
typed MENU1 batch-16 frozen-scorer path. The fresh control reproduces the
settled source `d_seg` exactly.

The empty `source_archives.*.receiver_custody` maps in the receipt came from a
grammar-only inner-loop parse and are not cited as proof. The authority is the
later independent emitted-runtime render, exact raw identity, and frozen
scorer measurement.

# Recursive adversarial review

An initial finding-producing pass caught that the new class detector ranked
the MC1 evidence lexicographically instead of using MC1's exact
`bottom_share * static_iou` law. The detector was corrected, canonical class
coverage was made exact, and tied maxima now fail closed. That pass did not
advance the clean counter.

Three subsequent passes were clean:

1. **Call-site and grammar pass:** traced encode -> carrier compile -> RG4
   rewrap -> E4 frame -> emitted runtime -> camera receiver. Full carrier
   tests passed 26/26; runtime-exporter tests passed 28/28; focused CB1/tool
   tests passed 7/7.
2. **Custody and scored-quantity pass:** independently aggregated all 114
   scorer batch receipts, verified class conservation, and freshly hashed all
   three 3,662,409,600-byte outputs plus their final archives. Every value
   matched the receipt.
3. **Scope and false-authority pass:** rederived the objective signs,
   verified only MyCar is admitted, checked Lane's negative remains
   instance-scoped, and checked all exact-eval/pointer/dispatch flags remain
   false.

**Assumption challenge:** the shared assumption is that an inherited carrier
payload can be rewrapped unchanged after the current RG4 source-local base.
Violating it could unlock a better result: the Lane failure strongly motivates
a base-conditioned source-local Lane receiver, and the MyCar palette/support
could be jointly scorer-recursive rather than inherited. That possibility is
why Lane is not killed and why MyCar's result is not generalized beyond this
exact composition. The current task asked for measured rows of these polished
payloads, so fitting a new carrier family is a successor, not evidence that may
be imputed to this run.

# c1/#613 consumption surface

The measurement receipt carries two
`ddm_c1_bucket_attribution_row.v1` objects at JSONPath
`$.c1_bucket_attribution_rows`. Each includes parent/candidate IDs, exact
archive bytes, `delta_d_seg`, `delta_d_pose`, `delta_joint_s`, per-class
counts/deltas, mechanism metadata, evidence axis, verdict scope, and a
`waterfill_eligible` boolean.

Only `candidate_id=mycar_static_mask` has
`waterfill_eligible=true`. The Lane object is preserved as a typed rejected
row so later c1 work cannot accidentally re-admit it or generalize it into a
family death.

# Durable artifacts

- Typed config:
  `.omx/research/configs/ddm_cb1_perclass_carrier_byteclose_20260725.json`
- SHA-pinned measurement receipt:
  `.omx/research/ddm_cb1_perclass_carrier_byteclose_20260725/ddm_cb1_perclass_carrier_byteclose_receipt.json`
- Emitted runtime and shell:
  `.omx/research/ddm_cb1_perclass_carrier_byteclose_20260725/runtime/`
- Exact control, MyCar, and Lane packet archives:
  `.omx/research/ddm_cb1_perclass_carrier_byteclose_20260725/packets/`
- Resumable bulk/checkpoints:
  `/Volumes/VertigoDataTier/pact/ddm_cb1_perclass_carrier_byteclose_20260725T203310Z`
- DAG feed:
  `.omx/research/ddm_cb1_perclass_carrier_byteclose_DAG_FEED_20260725.md`
- Integrity manifest:
  `.omx/research/ddm_cb1_perclass_carrier_byteclose_SHA_RECEIPT_20260725.json`

# STORES CONSULTED

`PROGRAM.md`; `CLAUDE.md`; `AGENTS.md`;
`docs/operating_manual_craft_handoff.md`; delegated authority file and both
watched inboxes; `.omx/research/fable_eureka_hunt_tier_breakthrough_20260725.md`;
`.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`;
`.omx/research/SPEC_v8_perclass_decomposition_20260708.md`; Claude project
memory `per_class_carriers_culminated_in_v8_v9_witness_recall_dont_rederive_20260721.md`;
MC1 receipt and static support; v13 worldsheet and Lane phase-ablation
receipts; c1 composed candidate ledger; RG4 complete-run/source-local PC1
receipt and exact archive; current E4 exporter/receiver implementation; exact
n600 target cache; frozen scorer modules and weights.

# MAIN landing review required

Before merge or c1/#613 consumption, MAIN must independently:

1. reaggregate all 38 scorer batches per candidate and verify the per-class
   conservation and objective deltas;
2. verify the source/current-base identity and that neither PR110 nor another
   old-lineage vehicle entered the chain;
3. review class self-detection and the wildcard target encoding for absence of
   a fixed MyCar index;
4. verify the CB1/E4 member closure, double-compile identity, RG4 and parent
   parse/reemit equality, emitted-runtime independence, raw hashes, and
   under-30-minute decodes;
5. admit only the strict-negative MyCar row to c1 and preserve the Lane
   instance scope and typed rejection; and
6. preserve `score_claim=false`, pointer immobility, and the official `0.172`
   competitive-target correction.
