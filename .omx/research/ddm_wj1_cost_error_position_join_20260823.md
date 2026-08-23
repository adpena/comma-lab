# ddm_wj1 — DX2's top-1% cost set contains 99.9269% of gross render-manufactured bit mass, but that mass is only 6.842 KB

**Disposition:** `MEASURED-STRONGLY-ASSOCIATED / TARGET-SET-EMITTED`,
`verdict_scope=INSTANCE:DX2_archive_976f706d_n600_BL1_cost_x_MST1_intermediate_observations`.
This is a deterministic scorer-free join over all **117,964,800** positions. The cost field is
BL1's shipped-law integer-frequency allocation. The render-manufactured field is MST1's retained
`L correct -> native-render-plus-frozen-head wrong` support. No scorer ran, no archive changed, no
shipping candidate was built, and the frontier did not move.

## Answer first

The pay-for-nothing prior is **confirmed as a position-membership association**, not as a byte win.
The 28,602 gross render-manufactured positions carry **54,774.684042 modeled bits = 6,846.835505 B**,
which is **6.017768% of the 113,777 B physical stream** and **16.155055% of the 42,382 B demand**.
The top 1% of cost positions contains **26,016 / 28,602 = 90.958674%** of those positions and
**54,734.651435 bits = 6,841.831429 B = 99.926914%** of their bit mass. Independence predicts only
**286.02 positions and 26.572293 B** in that cell. Observed mass is therefore **90.958674x by count**
and **257.479902x by modeled bits**. The registered `>=2x` and `>5,000 B` prediction holds; the
independence/`<1,000 B` falsifier does not fire.

The association survives class conditioning. At top 1%, within-class count enrichment is
Road **42.77x**, Lane **2.40x**, Undrivable **331.59x**, Movable **12.08x**, and MyCar **361.82x**.
Lane is the largest single class cell at **2,095.192458 B**, but it is only **30.61%** of the full
gross-manufactured mass; Road contributes 28.27%, Undrivable 21.86%, Movable 13.35%, and MyCar 5.90%.
The shared object is therefore not class alone on this manufactured support: exact position identity
still carries strong information inside every class. That differs from BL1's weaker final-error join
and is a real refinement, not a contradiction obtained by changing denominators.

The most plausible waste sub-cell is `render broke it, downstream restored it`. It contains
**11,685 positions and 2,252.103297 B** over all costs, or **5.313820% of demand**. Top 1% captures
**10,491 positions and 2,249.700879 B** of that repaired mass. The terminal-persistent side is larger:
**16,917 positions and 4,594.732208 B** over all costs. These are incumbent modeled-cost masses,
not removable bytes. LD1 commit `5e8d6011ba` already measured fixed-model coarsening at every rung
larger, so WJ1 claims **no actuator and no byte saving**. JF1 owns the model-refit mechanism.

## Membership correction: `+22,321` is not a position list

The charter inherited MST1's native stage net delta, **+22,321**, as though it named positions. It
does not. MST1's exact transition accounting is:

```text
28,602 gross correct->wrong transitions - 6,281 gross wrong->right transitions = +22,321 net errors
```

A net scalar has no membership mask and cannot be joined. The joinable manufactured support is the
retained **28,602-position** `gross_manufactured_native_render_head` mask: decoded label `L` equals
contest-CUDA DALI GT `G`, then the `[macOS-CPU advisory]` native-render-plus-frozen-SegNet-head
observation is wrong. This is the exact transition where transmitted correctness was lost.

The contingency complement is named **not render-manufactured**, not “render-correct.” It includes
native-head-correct positions, but also transmitted-label errors that remain wrong; calling that whole
complement render-correct would be false. This semantic correction changes no joint count or bit mass.

## Pin and count reproduction

BL1 commit `873947c665` and MST1 commit
`1c33f278920b91bf922e9620deb9ce20615135e8` were re-opened at source. Every payload was rehashed before
the join. No aggregate was substituted for either field.

| object | bytes | SHA-256 | reproduced use |
|---|---:|---|---|
| DX2 `archive.zip` | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | exact body pin |
| RC64 token stream | 113,777 | `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` | 910,216 physical bits |
| decoded field | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | TO2/BL1 field pin |
| per-position cost field | 943,718,400 | `99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86` | 117,964,800 little-endian float64 costs |
| DALI GT NPY | 117,964,928 | `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248` | class rows; contest-CUDA DALI lineage |

The physical/model reconciliation reproduced exactly: **910,216 physical bits** versus
**910,209.280609 modeled bits**, leaving BL1's explained **6.719391-bit** finite-interval/padding
residual. The modeled byte-equivalents in this memo divide the per-position field by eight; they are
not claims that those bytes can be removed independently.

| MST1 state, all / 117,964,800 | reproduced errors | adjacent net delta |
|---|---:|---:|
| decoded labels `L` | **9,182** | — |
| native render + frozen head | **31,503** | **+22,321** |
| float bilinear round trip + frozen head | **24,523** | **-6,980** |
| uint8 round trip + frozen head | **23,752** | **-771** |
| CPU-to-CUDA terminal/head, unseparated | **23,757** | **+5** |

The first three decision fields are MST1 `[macOS-CPU advisory]` intermediate observations against
contest-CUDA DALI GT. Final support is MS9/MST1
`[contest-CUDA T4 component-only exact field replay]`. The narrowest verdict scope therefore keeps
MST1's advisory-intermediate qualifier; the stage ordering is the robust fact, while exact joined
shares inherit that lineage.

## Global contingency: all four thresholds and all cells

For each threshold, `expensive` is the exact globally ranked set with threshold ties resolved in
`(frame,y,x)` raster order, matching BL1. Expected positions are
`positions(cost bucket) * positions(membership bucket) / 117,964,800`. Expected bit mass is
`modeled bits(cost bucket) * positions(membership bucket) / 117,964,800`. Thus the two observed/expected
columns state both the membership association and the delivered byte-mass association. Stream shares
use **113,777 B**; demand shares use **42,382 B**. A cell above 100% of demand only means its incumbent
modeled mass exceeds the campaign demand; it is not a savings claim.

| top set | cell | positions / 117,964,800 | modeled bits | byte-equiv | stream | demand | expected pos | expected byte mass | obs/exp pos | obs/exp bits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1% | expensive + render manufactured | 9,480 / 117,964,800 | 42,772.997135 | 5,346.624642 B | 4.699214% | 12.615319% | 28.602048 | 14.607196 B | 331.444792x | 366.026772x |
| 0.1% | expensive + not render manufactured | 108,485 / 117,964,800 | 439,189.076366 | 54,898.634546 B | 48.251083% | 129.532902% | 117,936.397952 | 60,230.651992 B | 0.919860x | 0.911473x |
| 0.1% | cheap + render manufactured | 19,122 / 117,964,800 | 12,001.686907 | 1,500.210863 B | 1.318554% | 3.539736% | 28,573.397952 | 12.979218 B | 0.669224x | 115.585615x |
| 0.1% | cheap + not render manufactured | 117,827,713 / 117,964,800 | 416,245.520201 | 52,030.690025 B | 45.730411% | 122.766009% | 117,818,261.602048 | 53,517.921671 B | 1.000080x | 0.972211x |
| 1% | expensive + render manufactured | 26,016 / 117,964,800 | 54,734.651435 | 6,841.831429 B | 6.013370% | 16.143248% | 286.020000 | 26.572293 B | 90.958674x | 257.479902x |
| 1% | expensive + not render manufactured | 1,153,632 / 117,964,800 | 822,013.897055 | 102,751.737132 B | 90.309761% | 242.441926% | 1,179,361.980000 | 109,566.996268 B | 0.978183x | 0.937798x |
| 1% | cheap + render manufactured | 2,586 / 117,964,800 | 40.032607 | 5.004076 B | 0.004398% | 0.011807% | 28,315.980000 | 1.014120 B | 0.091327x | 4.934401x |
| 1% | cheap + not render manufactured | 116,782,566 / 117,964,800 | 33,420.699512 | 4,177.587439 B | 3.671733% | 9.856985% | 116,756,836.020000 | 4,181.577395 B | 1.000220x | 0.999046x |
| 5% | expensive + render manufactured | 28,355 / 117,964,800 | 54,774.608957 | 6,846.826120 B | 6.017759% | 16.155033% | 1,430.100000 | 27.498811 B | 19.827285x | 248.986259x |
| 5% | expensive + not render manufactured | 5,869,885 / 117,964,800 | 852,544.253810 | 106,568.031726 B | 93.663949% | 251.446444% | 5,896,809.900000 | 113,387.359035 B | 0.995434x | 0.939858x |
| 5% | cheap + render manufactured | 247 / 117,964,800 | 0.075085 | 0.009386 B | 0.000008% | 0.000022% | 27,171.900000 | 0.087602 B | 0.009090x | 0.107140x |
| 5% | cheap + not render manufactured | 112,066,313 / 117,964,800 | 2,890.342757 | 361.292845 B | 0.317545% | 0.852468% | 112,039,388.100000 | 361.214628 B | 1.000240x | 1.000217x |
| 10% | expensive + render manufactured | 28,523 / 117,964,800 | 54,774.679645 | 6,846.834956 B | 6.017767% | 16.155054% | 2,860.200000 | 27.559069 B | 9.972380x | 248.442168x |
| 10% | expensive + not render manufactured | 11,767,957 / 117,964,800 | 854,532.394432 | 106,816.549304 B | 93.882375% | 252.032819% | 11,793,619.800000 | 113,635.825190 B | 0.997824x | 0.939990x |
| 10% | cheap + render manufactured | 79 / 117,964,800 | 0.004397 | 0.000550 B | 0.000000% | 0.000001% | 25,741.800000 | 0.027344 B | 0.003069x | 0.020101x |
| 10% | cheap + not render manufactured | 106,168,241 / 117,964,800 | 902.202135 | 112.775267 B | 0.099120% | 0.266092% | 106,142,578.200000 | 112.748473 B | 1.000242x | 1.000238x |

The count association weakens mechanically as the expensive set expands, yet the byte-mass ratio
stays near **248–366x** at all four thresholds because nearly every manufactured-position bit sits in
the cost tail. The top 0.1% already holds 5,346.625 B; top 1% is the practical knee at 6,841.831 B.

| top set | expensive positions | manufactured in expensive | rate in expensive | body rate | count enrichment | risk ratio vs cheap | joint bytes | joint demand share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1% | 117,965 | 9,480 | 8.036282% | 0.024246% | 331.444792x | 495.267437x | 5,346.624642 B | 12.615319% |
| 1% | 1,179,648 | 26,016 | 2.205404% | 0.024246% | 90.958674x | 995.972158x | 6,841.831429 B | 16.143248% |
| 5% | 5,898,240 | 28,355 | 0.480737% | 0.024246% | 19.827285x | 2,181.153846x | 6,846.826120 B | 16.155033% |
| 10% | 11,796,480 | 28,523 | 0.241792% | 0.024246% | 9.972380x | 3,249.455696x | 6,846.834956 B | 16.155054% |

## Per-class joint cells, Lane separate

Each expected baseline below is recomputed **inside that GT class**:
`class expensive count * class manufactured count / class positions` for positions, and
`class expensive bits * class manufactured count / class positions` for bit mass. This removes the
class-mix explanation BL1 warned about. Stream and demand shares keep the global denominators.

| GT class | top set | class expensive positions | class manufactured | joint positions | modeled bits | byte-equiv | stream | demand | expected pos | expected bytes | obs/exp pos | obs/exp bits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Road | 0.1% | 41,834 | 9,665 | 2,736 | 11,528.346226 | 1,441.043278 B | 1.266551% | 3.400130% | 14.752440 | 6.754446 B | 185.460846x | 213.347374x |
| Road | 1% | 564,488 | 9,665 | 8,513 | 15,465.441036 | 1,933.180130 B | 1.699096% | 4.561324% | 199.062373 | 14.927774 B | 42.765490x | 129.502240x |
| Road | 5% | 4,078,087 | 9,665 | 9,535 | 15,482.122896 | 1,935.265362 B | 1.700928% | 4.566244% | 1,438.106173 | 15.817960 B | 6.630248x | 122.346078x |
| Road | 10% | 8,854,300 | 9,665 | 9,629 | 15,482.163364 | 1,935.270420 B | 1.700933% | 4.566256% | 3,122.401137 | 15.888624 B | 3.083845x | 121.802267x |
| **Lane** | 0.1% | 40,787 | 6,367 | 2,618 | 14,191.783358 | 1,773.972920 B | 1.559167% | 4.185675% | 375.952697 | 222.170237 B | 6.963642x | 7.984746x |
| **Lane** | 1% | 271,112 | 6,367 | 6,005 | 16,761.539663 | 2,095.192458 B | 1.841490% | 4.943590% | 2,498.965050 | 348.363122 B | 2.402995x | 6.014392x |
| **Lane** | 5% | 587,045 | 6,367 | 6,362 | 16,767.685739 | 2,095.960717 B | 1.842166% | 4.945403% | 5,411.066045 | 351.906460 B | 1.175739x | 5.956017x |
| **Lane** | 10% | 672,975 | 6,367 | 6,367 | 16,767.687818 | 2,095.960977 B | 1.842166% | 4.945404% | 6,203.122711 | 351.948758 B | 1.026419x | 5.955302x |
| Undrivable | 0.1% | 13,036 | 8,088 | 2,249 | 8,007.152938 | 1,000.894117 B | 0.879698% | 2.361602% | 1.804993 | 0.776524 B | 1,245.988318x | 1,288.941586x |
| Undrivable | 1% | 162,090 | 8,088 | 7,442 | 11,966.064909 | 1,495.758114 B | 1.314640% | 3.529230% | 22.443333 | 1.723439 B | 331.590676x | 867.891593x |
| Undrivable | 5% | 674,966 | 8,088 | 8,006 | 11,975.904869 | 1,496.988109 B | 1.315721% | 3.532132% | 93.457257 | 1.782478 B | 85.664830x | 839.835545x |
| Undrivable | 10% | 1,516,550 | 8,088 | 8,052 | 11,975.923228 | 1,496.990404 B | 1.315723% | 3.532137% | 209.984804 | 1.787257 B | 38.345632x | 837.591005x |
| Movable | 0.1% | 13,310 | 3,160 | 1,284 | 6,015.941701 | 751.992713 B | 0.660936% | 1.774321% | 28.800331 | 15.089901 B | 44.582821x | 49.834172x |
| Movable | 1% | 113,360 | 3,160 | 2,962 | 7,311.506663 | 913.938333 B | 0.803272% | 2.156430% | 245.289670 | 24.649103 B | 12.075519x | 37.077955x |
| Movable | 5% | 299,624 | 3,160 | 3,150 | 7,314.950050 | 914.368756 B | 0.803650% | 2.157446% | 648.329853 | 25.122206 B | 4.858638x | 36.396834x |
| Movable | 10% | 441,560 | 3,160 | 3,158 | 7,314.954314 | 914.369289 B | 0.803650% | 2.157447% | 955.452600 | 25.135726 B | 3.305240x | 36.377278x |
| MyCar | 0.1% | 8,998 | 1,322 | 593 | 3,029.772913 | 378.721614 B | 0.332863% | 0.893591% | 0.396601 | 0.194215 B | 1,495.203679x | 1,950.014957x |
| MyCar | 1% | 68,598 | 1,322 | 1,094 | 3,230.099163 | 403.762395 B | 0.354872% | 0.952674% | 3.023568 | 0.248153 B | 361.824125x | 1,627.071726x |
| MyCar | 5% | 258,518 | 1,322 | 1,302 | 3,233.945403 | 404.243175 B | 0.355294% | 0.953809% | 11.394601 | 0.259944 B | 114.264638x | 1,555.118194x |
| MyCar | 10% | 311,095 | 1,322 | 1,317 | 3,233.950922 | 404.243865 B | 0.355295% | 0.953810% | 13.712018 | 0.260067 B | 96.047131x | 1,554.386181x |

Lane differs in shape: by top 10%, every one of its 6,367 manufactured positions is already inside
the cost set, so count enrichment approaches 1x, but their bit mass remains **5.96x** the within-class
independence baseline. The other classes retain large count and bit enrichments even at 10%.
Thus class concentration is real, but it does not explain away the position join.

## Downstream-repaired versus terminal-persistent

Within the 28,602 gross native breaks, `later repaired` means the terminal contest-CUDA field is
correct; `terminal persistent` means it remains a final manufactured error. These disjoint cells close
exactly: **11,685 + 16,917 = 28,602**. The first is the charter's sharpest futility candidate because
correct transmitted precision was broken at the advisory native observation and recovered by a free
downstream operation. The table prices membership only; it predicts no distortion change.

| top set | terminal disposition | positions / 28,602 | modeled bits | byte-equiv | stream | demand |
|---|---|---:|---:|---:|---:|---:|
| 0.1% | later repaired | 3,124 / 28,602 | 12,808.656145 | 1,601.082018 B | 1.407211% | 3.777741% |
| 0.1% | terminal persistent | 6,356 / 28,602 | 29,964.340990 | 3,745.542624 B | 3.292003% | 8.837579% |
| 1% | later repaired | 10,491 / 28,602 | 17,997.607031 | 2,249.700879 B | 1.977290% | 5.308152% |
| 1% | terminal persistent | 15,525 / 28,602 | 36,737.044404 | 4,592.130550 B | 4.036080% | 10.835096% |
| 5% | later repaired | 11,583 / 28,602 | 18,016.791715 | 2,252.098964 B | 1.979397% | 5.313810% |
| 5% | terminal persistent | 16,772 / 28,602 | 36,757.817241 | 4,594.727155 B | 4.038362% | 10.841223% |
| 10% | later repaired | 11,656 / 28,602 | 18,016.824811 | 2,252.103101 B | 1.979401% | 5.313820% |
| 10% | terminal persistent | 16,867 / 28,602 | 36,757.854834 | 4,594.731854 B | 4.038366% | 10.841234% |
| all costs | all later repaired | 11,685 / 28,602 | 18,016.826377 | 2,252.103297 B | 1.979401% | 5.313820% |
| all costs | all terminal persistent | 16,917 / 28,602 | 36,757.857665 | 4,594.732208 B | 4.038366% | 10.841235% |

The repaired set is nonempty and material, satisfying the charter's sharp sub-prediction, but it is
only 2.252 KB of incumbent modeled mass. Even a fictitious 100% harvest would close 5.31% of demand.
That is a useful priority signal for a refit, not an independent route to sub-0.12.

## Target payload and JF1 handoff

The direct consumer payload is the top-10%-union position list because it contains
**28,523 / 28,602** manufactured positions and **54,774.679645 / 54,774.684042 = 99.999992%** of
their modeled bit mass while carrying nested flags for all four thresholds. JF1 can filter the exact
top 0.1%, 1%, 5%, or 10% set and the repaired/persistent split without re-reading WJ1 aggregates.

| payload | positions | bytes | SHA-256 |
|---|---:|---:|---|
| structured raster-sorted position list, with cost, GT class, terminal disposition, and four threshold flags | 28,523 | 798,964 | `bb1c42698e38deb94d9bee8edbdf44261a40a95554defef38d6088730be5da7d` |
| top-0.1% joint packed mask | 9,480 | 14,745,600 | `fcf2c0eb2de46f86ed079ac33ae98b9c907f82b9b5c4efd10235c3c3e121c537` |
| top-1% joint packed mask | 26,016 | 14,745,600 | `591348816553bdd0bbeb8367058116023407804dd37587efb15811f059b67def` |
| top-5% joint packed mask | 28,355 | 14,745,600 | `73295e1d6a5eb1951671f147bbc3bdcdd725ff49fbadc1c81ff90e8a8d70beab` |
| top-10% joint packed mask | 28,523 | 14,745,600 | `4d4b7fe94007453f3b619fb97b8825ea256f2dd1ca59100f6d9fe3a2afac1fbb` |

The position-list schema is fixed-width NPY with `flat_index`, `(frame,y,x)`, `cost_bits`, `gt_class`,
`persistent_final`, and four uint8 membership flags. Its consumer receipt is
`.omx/tmp/arm_receipts_local/ddm_wj1_cost_error_position_join/measurement_v1/JF1_HANDOFF.json`,
SHA-256 `c5877eb47158fd9f1d248c7aaf304af0783f7ff5c5bc8bcee7af642599b71335`.

## Adjudication and boundaries

The pay-for-nothing hypothesis is **not independent and not class-only** on this DX2 instance.
Cost × gross-render-futility is a sharper target than cost alone: the top-1% intersection holds
6.842 KB instead of the 26.6 B independence baseline, and every class retains positive within-class
bit-mass enrichment. JF1 should therefore admit the emitted flags as a target-ordering input alongside
its existing LD1 rungs.

That statement ends at membership. It does not convert 6.847 KB into predicted savings, and it does
not convert 28,602 positions into predicted `d_seg`. RI1/NI1's amplification exponent 16.69 forbids
that count-to-distortion shortcut. The full cost field is an allocation reconciled to the incumbent
physical stream, not a lower bound. A high-cost position is not thereby droppable. Only a retained
field-coarsen-plus-model-refit re-encode can tell whether any of this mass is harvestable, and JF1 owns
that mechanism. The exact rate constant is cited, not re-derived: TX1 §0 gives
`25/37,545,489 = 6.658590e-07 S/B`.

LD1's six fixed-model rungs remain dead on rate: all grew by +21 to +1,528 B. TO2/EF1/XS1/MZ2 keep
the reorder, generic-estimator, coder, and storage-layout axes closed. WJ1 did not perturb or re-encode
anything and proposes none of those paths.

## Retention, reproducibility, and verification

The charter's explicit-opt-in tier was **local disk**, not either full SSD. Storage preflight measured
**507,100,200,960 B free** against a 3,000,000,000 B requirement. The durable store is
`.omx/tmp/arm_receipts_local/ddm_wj1_cost_error_position_join/measurement_v1/`. It contains **27
manifested artifacts totaling 1,283,879,716 B**, including a local byte-identical copy of the
943,718,400-byte cost field, DALI GT, all seven required MST1 masks, four full threshold masks, four
joint target masks, the structured position list, tables, source receipts, source snapshots, and the
JF1 handoff. `no_volume_writes=true`; both `/Volumes/*` trees were read only.

- `JOIN_RESULT.json`: 118,718 B, SHA-256
  `253511041b2b03209ee2dd138c4c1b753c4b95604784e730c01c11e321693ef0`.
- `MANIFEST.json`: SHA-256
  `4b009ec4f6bfb9842313460dda3678195c77bc833cc1001eb4875ae8154989ea`.
- `COMPLETED_VERIFICATION.json`: SHA-256
  `5bb5b1bb1d898134d2dc6347bd6a28d96d1929877721fadab9355b20c6bbee15`.
- Independent read-only replay receipt: SHA-256
  `c2f2013aaccfa87e11144f82b02dc4bc6381c6cc698ca168d4c73fb2ba91c8e6`; it recomputed the
  28,602-position support, 54,774.684042-bit mass, all threshold/class joint cells, and every target
  NPY column directly from retained fields.
- A deterministic resume preserved the exact `JOIN_RESULT.json` and position-list hashes.
- `ruff check`, `ruff format --check`, `python -m py_compile`, the tool self-test, and the bounded
  payload-retention audit passed with **1/1 Python file examined, zero unreadable files, zero findings**.
- Two genuine code-review passes covered field semantics/accounting and independent payload replay;
  both were recorded by `tools/review_tracker.py` after the transient live-database lock cleared.

No upstream file, shipped receiver, BL1/MST1/MS9/LD1/JF1 custody tree, jo1 r9 directory, staged index,
scorer lane, Modal resource, or Metal resource was modified.

## RECALL EVIDENCE

Before adjudication the recall searched the full `.omx/research` corpus and local arm receipts by
content, the canonical research indexes, `sub015_DAG_*` FEED blocks, `canonical_task_status.jsonl`,
`main_hot_state.md`, active lane claims, and `tools/list_canonical_equations.py --json`. Queries
included `per-position cost`, `bit allocation`, `render-manufactured`, `manufactured error`,
`stage split`, `cost futility`, `pay-for-nothing`, `field coarsen`, the exact archive/stream/decoded
SHAs, and `ddm_{bl1,mst1,ms9,ld1,jf1,wj1}`.

Beyond the charter seeds, the search confirmed the live JF1 mechanism ownership, the canonical atomic
flip/byte exchange law, and the current board's explicit no-paint/no-coder posture. It found no prior
current-DX2 per-position BL1×MST1 join in the bounded index/DAG/task scopes. The key plan change came
from reading MST1's actual masks rather than its headline: the charter's +22,321 net delta could not
serve as membership, forcing the exact 28,602-transition support and the honest
`not_render_manufactured` complement. The recall also kept LD1's fixed-model negative load-bearing,
so this arm emits a JF1 input rather than laundering target mass into a byte-win claim.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: `ddm_jf1_joint_field_model_refit`; consumer store:
  `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/wj1_target_consumer/`; fire trigger:
  WJ1 `COMPLETED_VERIFICATION.json` has `status=COMPLETE` and the consumer independently matches
  position-list SHA-256 `bb1c42698e38deb94d9bee8edbdf44261a40a95554defef38d6088730be5da7d`.
  Consume the nested cost×futility flags in a field-coarsen-plus-model-refit rung, retain every changed
  field/model/stream, and compare against JF1's null-refit control; do not treat WJ1's modeled mass as
  realized savings.

## LIVE-HYPOTHESES

- Cost×futility ordering will dominate cost-only ordering under an HPAC refit. It is plausible because
  the top-1% joint bit mass is 257.48x its independence baseline and the association survives every GT
  class; only JF1's real re-encode can validate it.
- The later-repaired subset may be the best first refit rung. It is plausible because 2,249.70 of its
  2,252.10 modeled bytes lie in the top 1%, giving a compact high-confidence target whose downstream
  terminal output is already correct; distortion survival remains unmeasured.
- Lane may need a class-aware cap rather than exclusive targeting. Lane holds the largest class mass
  (2.096 KB), but 69.39% of gross-manufactured mass lies outside Lane and Lane's top-10% count
  enrichment collapses near 1x; a Lane-only target would discard the stronger all-class position law.

## DEAD-ENDS

- The position-granularity independence hypothesis is closed on this DX2 instance: top-1% observed
  count and bit mass are 90.96x and 257.48x their independence baselines.
- The class-only explanation is closed on this manufactured support: every GT class retains positive
  within-class top-1% enrichment, including Lane at 2.40x by count and 6.01x by bit mass.
- Joining a fictitious 22,321-position “net error” set is closed: +22,321 is a difference of two gross
  transition counts and has no membership payload.
- Fixed-model harvesting is closed by LD1: all six coarsened rungs increased archive bytes. WJ1's
  target set is not an actuator.
- Coder swaps, reordering, and storage-layout attacks remain closed by TO2/EF1/XS1/MZ2 and were not
  reopened by this membership result.

OWN-VEHICLE FRONTIER: UNMOVED — DX2 remains S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600], archive SHA-256 976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674; WJ1 made no score claim.
