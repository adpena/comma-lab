# RZ1 Receipt - PE3 Grammar x Frozen-Head Regional Solve

## Answer

RZ1 stopped at gate (a). The n=3 smoke failed on the NA3-derived non-prefix stratified set `[104, 151, 464]`, so I did not fire n32 and did not fire the one allowed n600 batch.

The failure happened before any subtle byte trade: PE3's target labels worsened the selected cells even under the ideal label substitution, and the frozen-head regional prototype realization worsened both d_seg and pose. This is an INSTANCE/SMOKE negative for this PE3-as-target plus SQ1-style frozen-head median-regional-prototype realization. It is not the charter's FAMILY-level falsifier, because that pre-registered falsifier required n>=32.

## Measurement

Axis: `[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE`. Score claim: false. Promotion eligible: false.

Command:

```sh
.venv/bin/python experiments/ddm_rz1_pe3_head_solve.py --base-sub-dir /Volumes/VertigoDataTier/pact/ddm_pe4_20260805_r2/qo1_identity_pe4_extended_receiver --pe3-archive /Volumes/VertigoDataTier/pact/ddm_pe4_20260805_r2/sub_auto_pairbit_pe4_pe3_hybrid_75kb_receiver/archive.zip --gt-mkv upstream/videos/0.mkv --pairs-json .omx/research/ddm_na3_20260805/stratified_pose_selection_923.json --n 3 --out /Volumes/VertigoDataTier/pact/ddm_rz1_20260805/rz1_pe3_headsolve_smoke_n3.json --threads 8 --steps 25 --lr 4.0 --eval-every 5
```

Durable checkpoint:

| artifact | bytes | sha256 |
|---|---:|---|
| `/Volumes/VertigoDataTier/pact/ddm_rz1_20260805/rz1_pe3_headsolve_smoke_n3.json` | 40K | `d243ea6fa0d8f18f3684f28a4aa0d358e4c9c69082ad4321045380397f725b17` |
| `/Volumes/VertigoDataTier/pact/ddm_pe4_20260805_r2/qo1_identity_pe4_extended_receiver/inflated/0.raw` | 3,662,409,600 | `3ce7d269a7080a4024a576694cd0ddc697099c64cd02fdd2bb879339e4b03f31` |
| `/Volumes/VertigoDataTier/pact/ddm_pe4_20260805_r2/sub_auto_pairbit_pe4_pe3_hybrid_75kb_receiver/archive.zip` | 432,428 | `3f08c7fdd1c2746fa456ef8b6d8005e850d1a3acac5665a5d08b2ef17585b5e0` |

## PE3 Parse-Back

| quantity | value |
|---|---:|
| PE3 section bytes | 74,408 |
| PE3 section sha256 | `5cc024ad32df7fedb18afb75dbed6be9c1af948dac826a1736cb1084949855c2` |
| PE3 raw bytes | 169,975 |
| PE3 raw sha256 | `beecc444dac58e7b345df3783a8b38e20c8c74e8b011ac82bd4cb02c24e697a8` |
| receiver raster sha256 | `1661535005f09a8dcd864fb54d20d18be618455bb7cf0c5801fec3c4efe83818` |
| component records | 8,644 |
| depth-conditioned curve records | 750 |
| generator-bisector records | 7,894 |
| described scorer pixels, n600 | 540,058 |
| effective component/class prototype slots, n600 | 16,944 |

## Gate Table

Denominators: n=3 subset = 589,824 scorer pixels; n600 full = 117,964,800 scorer pixels. The subset was derived from NA3's n120 stratified selection with seed 20260805, derived seed 20260808, no prefix.

| row | flips | d_seg | net fixed vs base | d_pose | note |
|---|---:|---:|---:|---:|---|
| clean qo1 base on selected pairs | 2,919 | 0.0049489339 | 0 | 0.0005340956 | measured through frozen scorer |
| ideal PE3 target labels | 4,219 | 0.0071529812 | -1,300 | n/a | labels-only ceiling is already negative |
| dense frozen-head solve | 2,928 | 0.0049641927 | -9 | 0.0005326336 | upper-bound paint, not receiver-priced |
| regional PE3 prototype | 3,528 | 0.0059814453 | -609 | 0.0317154435 | receiver-priced prototype form |

Per-pair smoke rows:

| pair | base flips | ideal PE3 target flips | dense solve flips | regional prototype flips | regional d_pose |
|---:|---:|---:|---:|---:|---:|
| 104 | 935 | 1,334 | 936 | 1,174 | 0.0256015 |
| 151 | 1,166 | 1,678 | 1,168 | 1,266 | 0.0634930 |
| 464 | 818 | 1,207 | 824 | 1,088 | 0.00605185 |

## Bytes And Score Projection

The regional prototype payload used 92 effective component/class RGB slots on the n=3 subset. Exact n600 PE3 ownership implies 16,944 slots, so the raw prototype section is 50,832 B. The measured subset did not compress below raw: raw 276 B, zlib9 287 B, brotli q11 280 B, lzma-raw 276 B. Composed with the inherited PE3+OD9 projection, the projected regional form is 165,684 B.

| candidate | bytes used for projection | d_seg | d_pose | projected S | delta vs live |
|---|---:|---:|---:|---:|---:|
| regional PE3 prototype, raw slots | 165,684 | 0.0059814453 | 0.0317154435 | 1.2716313668 | +0.5176506371 |
| regional PE3 prototype, best measured compression | 165,684 | 0.0059814453 | 0.0317154435 | 1.2716313668 | +0.5176506371 |

This table is a subset projection, not an n600 score. It is sufficient only for the smoke stop: gate (a) failed before the n32 banking threshold.

## Recall Evidence

I searched beyond the charter seeds before and during implementation:

| scope searched | query or source | finding beyond seed | plan impact |
|---|---|---|---|
| governing docs | `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | RZ1 had the current scorer slot; protected files and index were off-limits; frontier line is the qo1 357,836 B advisory row. | Claimed the slot, avoided protected direct-description code, and stopped at n3 when the gate failed. |
| charter seeds | PK1, LP1 #934 commits `d4a4b3c541` and `db46820c7c`, OD7, ET1, RZ1 attack table | Direct paint/composite is negative; solve-from-frozen-head is the measured-positive alternative; regional law survives where per-pixel stamps die. | Used PE3 as WHERE/WHICH constraints and solved frozen-head paint before regional prototype reduction. |
| NA3 selection | `.omx/research/ddm_na3_20260805/stratified_pose_selection_923.json` | NA3 provides n120 stratified non-prefix seed IDs with pose-ratio match. | Derived n=3 smoke from NA3 blocks instead of using a prefix. |
| corpus search | `rg -n "ddm_rz1|PE3|frozen-head|head-solve|solve-from-frozen|regional" .omx/research/CANONICAL_RESEARCH_INDEX* .omx/state/canonical_equations_registry.jsonl .omx/research/sub015_DAG* .omx/research/ddm_*` | GC18 explicitly warned that PE3 band recall is not flips fixed and must be proved before TR1 inherits the target. LH1 independently repeated the #934 solve-from-frozen-head reading. NB1 warned not to over-consume prototype-grid negatives as solved-paint family negatives. | Added the labels-only target ceiling; kept the verdict scoped to INSTANCE/SMOKE; did not call this a family kill. |
| canonical equations registry | `.venv/bin/python tools/list_canonical_equations.py --json` plus narrowed grep | No settled canonical equation for PE3 head-solve conversion was found in the narrowed RZ1/PE3/head-solve surface. | Measured the conversion directly instead of importing a law. |

## Falsifier Verdict

The n>=32 FAMILY falsifier did not fire. The n=3 smoke gate failed first.

Verdict scope: INSTANCE/SMOKE for this exact implementation:

- PE3 target assignment over the clean qo1 base is negative on all three selected pairs.
- Dense frozen-head solve is neutral-negative on the same target, not a hidden positive.
- Regional component/class prototype reduction is worse than dense and wrecks pose on the subset.

Do not spend the n32 or n600 scorer budget on this exact target/realizer. A future re-open needs a positive label-ceiling audit first: if the grammar's WHERE/WHICH target labels do not reduce flips before realization, no frozen-head renderer can convert it into a positive row.

## NEXT_IF_RESUMED

```json
{
  "rz1_n32_gate_for_this_exact_target_realizer": {
    "disposition": "FOLDED",
    "reason": "n3 smoke failed before banking threshold; regional net_fixed=-609, label_ceiling_net_fixed=-1300, pose regressed to 0.0317154435"
  },
  "rz1_n600_gate": {
    "disposition": "FOLDED",
    "reason": "n600 allowed only after n32 composed delta-S < 0; n32 was not fired because n3 failed"
  },
  "target_label_ceiling_audit": {
    "disposition": "QUEUED-WITH-FIRE-ORDER",
    "fire_order": "Run a scorer-free or one-pass n32 NA3-derived label-ceiling audit before any new PE3 head-solve. Proceed only if ideal PE3 target labels have positive net_fixed on n>=32; otherwise route PE3 grammar to conditioning only."
  },
  "tr1_learned_carrier_primary": {
    "disposition": "QUEUED-WITH-FIRE-ORDER",
    "fire_order": "If the n32 target-label ceiling is nonpositive, demote grammar-to-target correction and use PE3 only as conditioning input for TR1 learned-carrier training."
  }
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
