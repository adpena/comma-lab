# OD3 Terminality Receipt - 2026-08-05

## Answer First

OD3 closed `OD1_BLOCKER_SEG_BASE_CAP_BOUND` for the OD2 n32 pair set on the `[macOS-CPU frozen-scorer advisory]` axis: 32/32 rows stopped by the canonical derived trajectory law before the derived ceiling, and 0/32 rows ended `safety_bound_REPORTED`.

| item | value |
|---|---:|
| pair set | OD2 seeded stratified-random n32 (`8, 32, 46, 57, 70, 107, 112, 119, 148, 154, 168, 198, 225, 234, 244, 251, 284, 328, 336, 349, 383, 399, 411, 423, 445, 465, 481, 516, 536, 561, 582, 583`) |
| required census: converged | 32/32 |
| required census: cap_best_at_derived_ceiling | 0/32 |
| required census: pre_plateau | 0/32 |
| required census: failed | 0/32 |
| raw stop token | `marginal_below_bar` 32/32 |
| derived ceiling | 100 steps |
| steps actually run | min 25, mean `54.218750`, max 75 |
| Stage-1 pooled eta | `0.604882865092900` |
| Stage-1 eta mean +/- sd | `0.600365026199096` +/- `0.094322785382384` |
| d_seg before -> terminal | `0.004331270853678` -> `0.003260135650635` |
| delta S_seg subset | `-0.107113520304362` |

Terminal Stage-2 k=4 frame_0 carriage preserved seg on 32/32 rows and repaired the Stage-1 pose transient. It beat the same-row baseline but missed the preregistered OD2 target.

| pose term | mean d_pose | S pose contribution |
|---|---:|---:|
| same-row baseline | `0.000801428562340` | `0.089522542543224` |
| terminal Stage-1 transient | `0.033106106524428` | `0.575379062222710` |
| terminal k=4 carriage | `0.000791809037082` | `0.088983652267268` |
| OD2 preregistered k=4 target | `0.000758869833362` | n/a |

Preregistered prediction verdict: `MISS_VS_OD2_TARGET_BUT_PASS_VS_SAME_ROW_BASELINE`. Delta vs OD2 target is `0.000032939203720` d_pose; delta vs same-row baseline is `-0.000009619525258` d_pose.

Composed OD3 subset projection, using OD2 arithmetic and the common-contract live own-vehicle frontier:

| component | delta S |
|---|---:|
| Stage-1 seg | `-0.107113520304362` |
| terminal k=4 pose vs baseline | `-0.000538890275956` |
| projected k=4 rate, 57,600 B | `0.038353475699837` |
| total projected delta vs current own vehicle | `-0.069298934880481` |
| projected advisory S | `0.684681794810640` |

This is not a pointer move: no receiver-closed archive was built, no n600 scorer job was run, and no `upstream/evaluate.py` contest-axis score was produced.

## Registered Prediction Fallback

Because k=4 missed the OD2 registered target, OD3 ran the addendum's quick capacity sweep on the four worst terminal k=4 rows: pairs `198,119,423,411`. This is an instance-scoped diagnostic panel, not a new n32 projection.

| panel arm | mean d_pose | B/pair | seg preserved | mean d_pose recovered vs k=4 | bytes per pose recovered |
|---|---:|---:|---:|---:|---:|
| k=4 | `0.002103622246068` | 96 | 4/4 | `0.000000000000000` | n/a |
| k=8 | `0.001832075824495` | 384 | 4/4 | `0.000271546421573` | 106.059 B/pair per 1e-4 d_pose |
| k=12 | `0.002074025076581` | 864 | 4/4 | `0.000029597169487` | 2594.843 B/pair per 1e-4 d_pose |

Capacity verdict: k=8 partially restores the worst-panel pose margin at +288 B/pair, while k=12 is worse than k=8 on this panel. k=8 is a follow-on candidate only if OD4's archive/rate accounting can make selective higher-k carriage byte-positive.

## Arithmetic

- Seg denominator: `32 * 384 * 512 = 6291456` pixels.
- Stage-1 flips: `27250` -> `20511`; described pixels `11141`.
- Eta pooled: `(flips_before - flips_after) / n_described = 0.604882865092900`.
- Pose contribution: `sqrt(10 * mean(d_pose))`, matching OD2's aggregate convention.
- Rate cost: `57,600 * 25 / 37,545,489 = 0.038353475699837` S.
- m96 carry-forward: the n32 subset is pose-easy at `0.42628664334579025x` population, while its seg ratio is matched at `1.0099888594483923x`; pose conclusions remain advisory pending n600.

## ST2 Prior Use

The ST2 targeter was checked as the iteration-budget prior, but its ranked rows did not overlap the OD2 n32 pair ids. `od3_st2_pair_order_od2_intersection.json` has `rows_sorted=[]` and lists all 32 OD2 pairs as missing ST2 rows. OD3 therefore used the charter-mandated OD2 pair set/order rather than fabricating a targeter ranking.

## Recall Evidence

Sources searched beyond the charter seeds:

- `rg -n "trajectory_derived_stopping_law_v1|marginal_below_bar|safety_bound_REPORTED" src/tac/optimization/trajectory_stopping.py .omx/state/canonical_equations_registry.jsonl .omx/research/ddm_tj1_20260805 -g '*.md' -g '*.json'`
- `rg -n "frame_0|k=4|C-PRIME|staged|pose-EASY|0.426|1.010" .omx/research/ddm_xa2_cross_application_20260804.md .omx/research/ddm_od1_20260805 .omx/research/ddm_od2_20260805 .omx/research/ddm_st2_20260805 -g '*.md' -g '*.json'`
- `rg -n "OD3|od3|ddm_od3|reserved scorer|scorer-slot|Stage-1 terminality" .omx/state/main_hot_state.md .omx/state/codex_arm_queue.jsonl .omx/research/CANONICAL_RESEARCH_INDEX* .omx/research/sub015_DAG_*`
- `.venv/bin/python tools/list_canonical_equations.py --json > /Volumes/VertigoDataTier/pact/ddm_od3_20260805/canonical_equations_snapshot.json`

Findings that changed the plan:

- `.omx/research/ddm_od3_20260805/CHARTER_ADDENDUM_PREREGISTERED_PREDICTION.md` registered the operator law that better-conditioned seg should improve pose. That changed OD3 from a k=4 safety re-proof into a hypothesis test and triggered the k=8/k=12 fallback panel after the k=4 miss.
- `trajectory_derived_stopping_law_v1` and `allocate_extra_compute` treat both `converged_projected` and `marginal_below_bar` as semantic stops, while keeping `safety_bound_REPORTED` separate. That made the OD3 stop census cap-clean even though every selected row stopped by the marginal bar rather than literal projected-zero tail.
- OD2's `PAIR_SELECTION.json` confirmed the m96 carry-forward: pose-easy `0.42628664334579025x`, seg-matched `1.0099888594483923x`.
- ST2 did not contain the OD2 pair ids in a consumable ranking table. Scoped negative: did not find usable ST2 row overlap for this pair set in `.omx/research/ddm_st2_20260805/ddm_st2_receipt.json` or the OD3 intersection sidecar.

## Evidence And SHA Table

| path | bytes | sha256 |
|---|---:|---|
| `.omx/tmp/codex_runs/od3_prompt.md` | 4673 | `62c135170736ee8c1f5f8242b8e659e1bbd2b75dcd1ece74b94ab9e2cccfe88c` |
| `.omx/tmp/codex_runs/_common_contract.md` | 4124 | `eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771` |
| `.omx/research/ddm_od2_20260805/OD2_STAGE12_RECEIPT.md` | 4940 | `60596412949f1eb88a2d75fb7d7858651f1454d982b35d9e11a793c7a77bb0d9` |
| `.omx/research/ddm_od2_20260805/NEXT_IF_RESUMED.md` | 2694 | `bb5242544526e0ad6f2f242c495202900766ca49e0ab4763c9b18eac0f87cdc1` |
| `.omx/research/ddm_od2_20260805/OD2_AGGREGATE.json` | 31831 | `43c97e844c23c00b5ad7367e147735587e00dec21b2f274ebfef7770b32a3ace` |
| `.omx/research/ddm_od2_20260805/PAIR_SELECTION.json` | 2388 | `0a8ac26a1cd39c7dc425dbb4922d0dda6f71227b205241d3d771ea9791c2d4f9` |
| `.omx/research/ddm_od2_20260805/od2_pairs_n32_seed20260805.npy` | 384 | `8b4aa8d47787757ca9a29cb1d176670ad2f39c15b7daf97f25615006c98a3f94` |
| `.omx/research/ddm_st2_20260805/ddm_st2_receipt.json` | 143947 | `b6e6a3cce7ca4fbc74430f9088accbdad888a600774a3331a7a808b50bb0e75c` |
| `.omx/research/ddm_st2_20260805/ST2_RECEIPT_20260805.md` | 5988 | `c5231d5866e8112f98988ce1c44620bbc840285fbd9851e19263f8f3547bcd93` |
| `.omx/research/ddm_od3_20260805/CHARTER_ADDENDUM_PREREGISTERED_PREDICTION.md` | 1004 | `aa1d63c1f9a7ba5af0ee6b417bf7f4392f6970e927fd19d84ede1f33d0a9c4c4` |
| `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_js1_n32_terminal_seg100_cprime_k4.json` | 312293 | `5f7f934e6bafa440572577509e0e733ab3c5e80940d8cf11178e91ecb93bd4ca` |
| `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_capacity_sweep_k8_worst4.json` | 40292 | `f16b2f62c531f2c1e694ba543a387c789192519eb70812eac9a7e543eade68c4` |
| `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_capacity_sweep_k12_worst4.json` | 40312 | `b9b5ea03810124d4fd99ca266f4427b57431bc768c250ab08251d540d8211581` |
| `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/canonical_equations_snapshot.json` | 3112071 | `8245131517b229bd2a589985608fabe329dae16f552afb6061717473a1ae2f68` |
| `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_st2_pair_order.json` | 6141 | `92e0f04decf4f6034f44edcccb94db4802dae0bcef98af8d98b24ae00bd18fff` |
| `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_st2_pair_order_od2_intersection.json` | 906 | `8b2d47c8261f3c248bab7200944ee61b9959da9cde5c10f7adb3b3b9fc4fb690` |
| `experiments/ddm_js1_staging_discriminator.py` | 33782 | `5cc6386d3ad222174b2aa7a4ee0ca4700c0966ee4a9856ee8b514d1f1019955e` |
| `src/tac/optimization/trajectory_stopping.py` | 19868 | `ec193396ba8d15f5c83ab0c7035e77f477991aece555a04354f7c6da5f3b005e` |

The JS1 telemetry patch that exposed `selected_curve` and `start_diagnostics` landed as commit `17e1c86982` before this receipt. The raw JS1 metadata still embeds an older own-vehicle frontier string; OD3 arithmetic uses the common-contract live frontier `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.

## Boundaries

- Axis: `[macOS-CPU frozen-scorer advisory]`, non-promotable.
- `score_claim=false`; `promotion_eligible=false`.
- No full n600 scorer job, no `upstream/evaluate.py`, no contest-axis claim.
- No receiver-closed archive build; OD4 owns archive construction and rate closure.
- Stage-1 representation bytes remain the open OD4 pricing question.
- k=4 coefficients are counted payload; DCT basis is rule-118 free code.
- Protected files were not edited.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
