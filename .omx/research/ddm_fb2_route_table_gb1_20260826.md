# DDM FB2 — sub-0.12 route table on the exact GB1 body

**Date:** 2026-08-26  
**Arm:** `ddm_fb2_route_table_gb1`  
**Mode:** `$0`; `DERIVED-over-MEASURED`; no scorer, no payload materialization, no Modal  
**Authority object:** GB1 `archive.zip`, SHA-256
`ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`,
180,215 B, `d_seg=0.00020139`, `d_pose=0.00000637`,
`S=0.14811799921260607` `[contest-CUDA T4 n600]`.

## Verdict

The current GB1 body needs **42,229 B** of rate reduction at fixed distortion. The strict
integer cap is 137,986 B: 137,986 B scores `0.1199994414812099`; 137,987 B scores
`0.1200001073401630`.

The other demand reading changed materially when GB1 moved the object. At zero distortion,
the strict cap is 180,218 B. GB1 is already 3 B below it, so the corrected zero-distortion
demand is **0 B, with 3 B of integer slack**. FB1/DX2's approximately 150 B zero-distortion
shortfall must not be carried onto GB1. This does not make distortion-only work sufficient:
zeroing all distortion is a nonphysical joint limit, while every isolated current-body
distortion axis still misses 0.12.

The complete pair table has **8 ARITHMETICALLY-OPEN** cells and **13
CLOSED-BY-RECEIPT** cells. Across all 120 subsets of size at least two, monotonic extension
gives 90 arithmetically open subsets and 30 receipt-closed subsets. The surviving reference
routes are:

1. aligned-configuration trained renderer plus manufactured-seg repair (`R+M`), with the
   pose gate carried from step zero;
2. born-small plus a re-solved or learned implicit evaluator-cell carrier (`B+C`, with
   last-frame Seg action required rather than a pose-only frame-0 solve);
3. a genuinely trained smaller renderer on a changed born-small object (`R+B`).

Nothing measured here moved the exact frontier. This arm derived and routed; it did not
run a scorer or claim a new score.

While FB2 was being written, the already-issued W96A and BS3 harness arms reported. W96A
remains scientifically unmeasured and is blocked on retained-storage capacity plus the absent
exact expected-flip objective. BS3 advanced the born-small rate half from a hypothetical
101,128 B construction to a deterministic, receiver-parse-exact **101,150 B** retained body,
but it did not own the scorer and therefore measured no resolved-carrier distortion or score.
These concurrent receipts change dispositions and the born-small byte arithmetic, not the
open/closed topology of the table.

## Exact operating point and two-currency demand

The contest score is

```text
S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489.
```

On GB1:

| term | exact value |
|---|---:|
| `lambda_B = 25/N` | `6.6585895312217134793e-7 S/B` |
| segmentation | `0.020139` |
| pose | `0.007981227975693965854` |
| rate | `0.119997771236912109468` |
| total | `0.148117999212606075322` |
| gap to 0.12 | `0.028117999212606075322` |
| continuous byte-equivalent gap | `42,228.1612055564 B` |
| strict integer cut | **42,229 B** |

The GB1 block identity is the AR1B residue map with only the GB1 token-stream movement
applied:

| counted block | bytes | share |
|---|---:|---:|
| token stream | 113,624 | 63.05% |
| renderer | 30,856 | 17.12% |
| carrier | 22,010 | 12.21% |
| HPAC model | 13,515 | 7.50% |
| residual | 96 | 0.05% |
| ZIP + RX1 | 114 | 0.06% |
| **total** | **180,215** | **100%** |

This also corrects D3B's comparison basis. Its best 127,499 B token subsystem was +207 B
against the pre-GB1 127,292 B bar, but is **+360 B** against GB1's live
`113,624+13,515=127,139 B` subsystem. Its full 180,575 B archive likewise loses to GB1 by
360 B.

### Both demand readings

| reading | strict cap | GB1 relationship | conclusion |
|---|---:|---:|---|
| hold GB1 distortion fixed | 137,986 B | 42,229 B too large | a new rate representation is required unless distortion also falls |
| set both distortions to zero | 180,218 B | 3 B below cap | no additional bytes are required in the impossible zero-distortion limit |

The second row is a limiting surface, not a route. Any nonzero distortion immediately
spends its 3 B and then demands rate cuts at `1/lambda_B = 1,501,819.56 B/S`.

## Single-axis surface and the qualified two-axis law

The manufactured mass is `21,493/23,757 = 0.9047017721` of GB1's segmentation error.
Perfect repair of that measured mass leaves `d_seg=0.0000191921101082`. GB1 is rate-only,
so the same decoded pixels make the MST1 count transferable without an ancestor-vehicle
assumption.

| isolated perfected axis | resulting bytes | resulting S | miss to 0.12 | status |
|---|---:|---:|---:|---|
| renderer removed (`R`) | 149,359 | 0.1275722554 | +0.0075722554 | fails |
| all measured manufactured errors repaired (`M`) | 180,215 | 0.1298982102 | +0.0098982102 | fails |
| pose set to zero (`P`) | 180,215 | 0.1401367712 | +0.0201367712 | fails |
| carrier removed (`C`) | 158,205 | 0.1334624437 | +0.0134624437 | fails |
| measured current-body token change (`T`) | 180,215 or larger | >=0.1481179992 | >=+0.0281179992 | closed by JT23/JF2/LM1/D3B |
| born-small whole object (`B`) | representation-dependent | representation-dependent | not a current-body single axis | changed-object escape |
| decode-time compute (`D`) | no positive measured credit | unchanged or larger | no clearing row | measured forms closed/blocked |

Thus the inherited “at least two axes” law remains true for the separable current-body
constituents `R/M/P/C`: none clears alone. It is deliberately not stated as a theorem over
`T/B/D`. A magical deletion of the entire 127,139 B token subsystem would pass, but the
seven concordant current-body receipts say no measured token mechanism supplies that
deletion; `B` is itself a whole-object replacement; and `D` has no priced positive supplier.

## Renderer corner: exact `f` versus bytes

Let `f` be the fraction of the 21,493 manufactured errors repaired, while pose remains at
GB1. The table gives the largest strict byte cap and the renderer cut that would be needed
from 180,215 B.

| `f` | resulting `d_seg` | strict cap B | required cut | fraction of 30,856 B renderer |
|---:|---:|---:|---:|---:|
| 0.00 | 0.0002013900000000 | 137,986 | 42,229 | 136.858% |
| 0.25 | 0.0001558405275270 | 144,827 | 35,388 | 114.688% |
| 0.50 | 0.0001102910550541 | 151,668 | 28,547 | 92.517% |
| 0.75 | 0.0000647415825811 | 158,508 | 21,707 | 70.349% |
| 1.00 | 0.0000191921101082 | 165,349 | 14,866 | 48.179% |

Removing the whole renderer requires repairing more than `f=0.415606` of manufactured
mass, or 37.600% of all segmentation errors. At the generous half-renderer plus full-M
corner, `B=164,787` and `S=0.1196253383`, leaving **562.67 B** of continuous margin. GB1
therefore widens FB1's old approximately 410 B corner by about 153 B, exactly the GB1 rate
movement. At the full-renderer plus full-M corner, `S=0.1093524664`, leaving 15,990.67 B
of byte-equivalent margin.

## Complete two-axis table

Axis keys:

- `R`: renderer bytes, preserving the useful evaluator cells;
- `M`: repair of the measured manufactured-seg mass;
- `P`: pose driven to zero;
- `C`: carrier bytes or a carrier re-solve on a changed object;
- `T`: current categorical token subsystem;
- `B`: born-small whole-object representation;
- `D`: generic decode-time computation or constraint reconstruction.

“ARITHMETICALLY-OPEN” means only that a strict sub-0.12 point exists on the ideal surface
and no source receipt closes the named unmeasured member. “CLOSED-BY-RECEIPT” is scoped to
the named mechanism, never global nonexistence.

| pair | status | ideal arithmetic / source-scoped reason |
|---|---|---|
| `R+M` | **ARITHMETICALLY-OPEN** | full/full gives `S=0.1093524664`; half-R/full-M gives `S=0.1196253383`. S1E is only n60, one-seed, MPS-advisory OFF-config evidence against, not a population closure. |
| `R+P` | **ARITHMETICALLY-OPEN** | full-R/pose-zero gives `S=0.1195910274`, only 614.20 B margin. No admissible aligned-config W96 run with pose gate has measured this corner. |
| `R+C` | **ARITHMETICALLY-OPEN** | removing both counted blocks at current distortion gives `S=0.1129166998`, 10,637.84 B margin. PC2 closes frozen-carrier purchases, not a renderer trained to absorb carrier function. |
| `R+T` | **CLOSED-BY-RECEIPT** | current-body token/model directions are closed by JT23, JF2, LM1, and D3B; renderer-only movement leaves that exact field/model identity unchanged (RJ2). |
| `R+B` | **ARITHMETICALLY-OPEN** | the now-retained 101,150 B body at GB1 distortion would score `0.0954718611`; no genuinely trained smaller renderer has been measured on the exact born-small/GB1 object. OR1 already queues this reference form. |
| `R+D` | **CLOSED-BY-RECEIPT** | DC1S Family A is full-population closed and Family C is folded into B; Family B lacks a universal scorer-free cell certificate and has no priced packet. Generic compute alone supplies no renderer bytes or evaluator-cell correction. |
| `M+P` | **CLOSED-BY-RECEIPT** | even perfect measured-M plus pose zero at 180,215 B scores `0.1219169822`, missing by 2,878.96 B; arithmetic closes it before mechanism evidence. |
| `M+C` | **CLOSED-BY-RECEIPT** | the ideal surface passes (`S=0.1152426547`), but MSR1 closes zero-byte boundary movement to one net pixel and PC2 closes current frozen-carrier purchases. A changed object is classified under `B`. |
| `M+T` | **CLOSED-BY-RECEIPT** | JF2 directly measured joint field/model diagonals and found monotone pose death; MSR1 closes the address-free boundary move; JT23/LM1 close coder/model replacements on this field. |
| `M+B` | **ARITHMETICALLY-OPEN** | at the retained BS3 101,150 B body and perfect measured-M, the ideal score is `0.0772520721`. BO2 closes only the HG1 analytic-generator instance, not a learned implicit evaluator-cell vocabulary. |
| `M+D` | **CLOSED-BY-RECEIPT** | MSR1's zero-byte balanced flow repairs one net pixel; DC1S Family A loses 274,549 B and Family B is certificate-blocked. No measured decode-time member supplies the missing addressed correction. |
| `P+C` | **CLOSED-BY-RECEIPT** | ideal carrier deletion plus pose zero scores `0.1254812157`, still 8,231.80 B short. PC2 also shows every current carrier purchase is net negative. |
| `P+T` | **CLOSED-BY-RECEIPT** | JF2's three terminal diagonal byte winners carry 6.9x–344x null pose and >=93% of their damage in pose; JT23/LM1 leave no current token supplier. |
| `P+B` | **ARITHMETICALLY-OPEN** | at 101,150 B and pose zero, sub-0.12 permits `d_seg<0.0005264837`; BO2's HG1 instance has `d_seg=0.01294921`, but its scope explicitly does not close another born-small vocabulary. |
| `P+D` | **CLOSED-BY-RECEIPT** | measured decode-time forms do not alter GB1 pixels/pose, and pose zero alone still needs 30,243 B. DC1S supplies no positive rate credit. |
| `C+T` | **CLOSED-BY-RECEIPT** | PC2 drains the live carrier remainder and JT23/JF2/LM1/D3B drain the current token family. D3B is +360 B on GB1 and preserves the decoded field, so it does not reprice carrier value. |
| `C+B` | **ARITHMETICALLY-OPEN** | the retained BS3 body leaves 36,836.84 B of fixed-GB1-distortion headroom at 101,150 B. BO2 explicitly leaves a re-solved carrier and learned implicit evaluator-cell carrier unclosed. A frame-0 pose-only solve is insufficient because HG1's segmentation damage alone is fatal. |
| `C+D` | **CLOSED-BY-RECEIPT** | PC2 closes current carrier economics; DC1S's realized dictionary is larger, while its constraint-shipping member has no universal certificate or counted packet. |
| `T+B` | **CLOSED-BY-RECEIPT** | this double-counts the categorical object: born-small replaces the current token field rather than composing with it. BS2/BO2 and D3/D3B measured the available replacement forms; a new vocabulary is routed under `B`, not an additive `T+B` credit. |
| `T+D` | **CLOSED-BY-RECEIPT** | DC1S full-population Family A is 388,326 B versus the then 113,777 B stream; Family B is certificate-blocked; JT23 and LM1 close coder and learned-model substitutions. |
| `B+D` | **ARITHMETICALLY-OPEN** | generic decode-time structure can in principle expand the 101,150 B born-small packet without counted generic code. BO2 closes HG1 only; no receipt has tested a scorer-free implicit evaluator-cell solver on the changed object. |

### All higher-order combinations

There are 120 subsets of the seven axes with cardinality at least two. The exhaustive
classification is compact because score improvements are monotone:

- every superset containing one of `{R+M, R+P, R+C, R+B, M+B, P+B, C+B, B+D}` is
  **ARITHMETICALLY-OPEN**, subject to the same non-overlap and real-receiver proof. This is
  90 subsets total: 8 pairs and 82 higher-order subsets;
- the 30 remaining subsets are **CLOSED-BY-RECEIPT**. They are exactly all cardinality>=2
  subsets of `{M,P,C,T,D}` (26 subsets), plus `{R,T}`, `{R,D}`, `{R,T,D}`, and `{B,T}`.
  Their closures inherit MSR1/PC2/JF2/JT23/LM1/DC1S as applicable. `M+P` and `P+C` also
  fail the ideal arithmetic directly.

This is a classification of combinations, not an assertion that every open superset is
independently useful. Overlapping credits—especially `B` with `T`, or a carrier already
absorbed by `B`—must be remeasured as one exact archive and are never added twice.

## Open cells: unmeasured half and cheapest discriminating measurement

| open cell | unmeasured half | cheapest real discriminator | existing instrument | disposition |
|---|---|---|---|---|
| `R+M`, `R+P` | admissible trained W96 at the CE1-aligned configuration, >=2 seeds, pose gate from step zero | reproduce the S1E n60 evenly-strided screen through real R for <=65 epochs; advance to seeded n600-sampled only if movement is >=5x | `experiments/ddm_wd3_scorer_aware_width_distillation.py`; `tools/s1a_off_floor_adjudicator.py` | **QUEUED-WITH-A-FIRE-ORDER** as `ddm_w96a_aligned_config_renderer_window` |
| `R+C` | renderer trained to absorb carrier action, rather than deleting a frozen carrier | one short admissible W96 window with carrier conditioning ablated/absorbed; compare exact composed bytes and pose-gated n60 screen | same W96A instruments above; `src/tac/torch_vehicle/boundary_routing.py` only as existing routing substrate, not a proxy authority | **FOLDED** into W96A; no duplicate launch |
| `R+B` | genuinely trained smaller renderer on the exact changed object | require a retained <=137,986 B exact archive candidate before any scorer; then one seeded n>=32 screen through R | existing OR1 consumer plus `experiments/ddm_wd3_scorer_aware_width_distillation.py`; archive parse-back via the existing GB1 receiver | **QUEUED-WITH-A-FIRE-ORDER** as `ddm_or1_renderer_born_small` |
| `M+B`, `P+B`, `C+B`, `B+D` | re-solved carrier and learned implicit evaluator-cell vocabulary, not HG1's fixed analytic generator | BS3 has built the 101,150 B exact body; next apply the QS5 exact in-compile solve and compare GB1/BO2/resolved rows on the sealed seeded-random n32 set; learned member remains a labeled screen | `experiments/ddm_bs3_born_small_resolved_carrier.py`; retained `BODY_RESULT.json` and `FIRE_ORDER.json`; QS5/RJ2 reference paths pinned there | **QUEUED-WITH-A-FIRE-ORDER**, currently BLOCKED on scorer ownership |

No scorer or payload is fired by FB2. The successor charters own retention, resumability,
scorer-slot, and authority labels.

## Ranked top three successor outlines

The requested product cannot be made into a single authority-valued scalar without faking
comparability: “margin” is exact score arithmetic, whereas the evidence against is a mix of
n60 MPS screens, instance-scoped n600 rows, and formulation closures. The ranking therefore
keeps both factors explicit and uses the product only ordinally: more exact headroom and less
scope-matched adverse evidence rank higher. No advisory result is promoted.

| rank | route | exact ideal headroom | closest evidence against | product reading |
|---:|---|---:|---|---|
| 1 | aligned W96 `R+M` with pose-gated `R+P` rider | 0.01064753 S at full-R/full-M; 562.67 B at half-R/full-M | S1E best is +0.155413 S, but n60/one-seed/MPS-advisory/OFF-config; CE1 finds 92.7% surcharge configuration-associated (13.6x), while OA2 itself is rate-only and does not transfer that magnitude to W96 | best match to a narrow falsifier; strong adverse screen but precisely named missing configuration |
| 2 | born-small `B+C` with `M/P/D` implicit-cell riders | 0.02452814 S if the retained 101,150 B body preserved GB1 distortion; permits `d_seg<0.0004466714` with current pose | BO2 HG1 is +5.131079 distortion, 209.07x its budget; instance scope explicitly leaves the two proposed members open | larger arithmetic headroom, much stronger same-object evidence against; rate half is now exact, distortion half remains unmeasured |
| 3 | trained changed-object renderer `R+B` | same 36,836.84 B fixed-distortion headroom at retained BS3 size | BO2 is 97.25x exchange-rate hostile and W72 is 922x; no trained reference form exists | highest mechanism risk, but distinct from both HG1 and frozen-current-body renderer deletions |

### Successor 1 — aligned W96 window

- **Type:** bounded experiment, then typed GO/REFUTED screen.
- **Owner:** `MAIN renderer-training successor` / arm `ddm_w96a`.
- **Consumer store:** `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/`.
- **Current disposition:** **BLOCKED, QUEUED-WITH-A-FIRE-ORDER**. W96A reproduced 35 retained
  OFF-config rows across two seeds without running a scorer, but produced zero aligned rows.
- **Fire trigger:** APDataStore has at least 45,521,567,744 free bytes after certified
  cleanup/capacity expansion, and the exact expected-flip margin objective plus schedule and
  step-zero pose gate are implemented and reviewed. Then run >=2 seeds, <=65 epochs, with
  retained stage checkpoints. Advance from n60 only at >=5x movement; close at <2x across both.
- **Boundary:** OA2 measured HPAC rate alignment, not W96 distortion. CE1's 13.6x is a
  configuration hypothesis, not a transferred prediction.

### Successor 2 — born-small re-solved/implicit carrier

- **Type:** exact QS5-pattern carrier solve plus one labeled implicit evaluator-cell screen.
- **Owner:** `MAIN born-small carrier successor` / arm `ddm_bs3`.
- **Consumer store:** `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/`.
- **Current disposition:** **BLOCKED, QUEUED-WITH-A-FIRE-ORDER**. The deterministic 101,150 B
  body and repeat are retained and parse back exactly; resolved-carrier distortion and score are
  unmeasured because BS3 did not own the scorer lane.
- **Fire trigger:** MAIN grants this exact object scorer ownership and no full-n600 job is
  active; revalidate every source/payload SHA, then run the sealed uniform-random n32 exact solve
  against GB1 and BO2, preserving every candidate. Escalate only if pose damage falls >=10x and
  the last-frame segmentation budget survives.
- **Boundary:** a pure frame-0 re-solve cannot rescue BO2; even pose zero leaves HG1's
  segmentation far outside the budget. The live member must change last-frame Seg cells.

### Successor 3 — genuinely trained smaller renderer on changed object

- **Type:** byte gate first, then bounded scorer screen.
- **Owner:** `MAIN renderer-training successor` under existing task
  `ddm_or1_renderer_born_small`.
- **Consumer store:** `/Volumes/APDataStore/pact/ddm_or1_renderer_born_small/`.
- **Fire trigger:** produce a retained receiver-closed candidate <=137,986 B with no
  hidden video-derived code and a resumable trainer; only then spend a seeded n>=32 screen.
- **Boundary:** do not rerun HG1, W72, or an SVD renderer; the reference form must be
  trained on the changed object and carry pose from step zero.

## Receipt provenance

Arithmetic inputs and closure labels were taken from these source files, not headlines or
working memory. Content SHA-256 pins make later edits detectable.

| source | content SHA-256 | source commit | use |
|---|---|---|---|
| `ddm_fb1_sub012_feasibility_bound_20260823.md` | `ae8ae3afddb758610dbf23b74aefeb1e42225d565939826ef276fbb9b398010e` | `b9f56abf99fcefc4eb2ee131ff654438dc76768f` | DX2 derivation and section 9 retractions |
| `ddm_gb1_groupbin8_verdict_20260824.md` | `56866135b84f4e941df35b81dfa1bafbdf36e75a01ca50fe4530b6ba237f558a` | `884bb65f1ef9409cbeda1cb02ca2b268fe7ab3e6` | exact GB1 pointer, bytes, SHA, rate-only identity |
| `ddm_ar1b_archive_residue_purchase_20260822.md` | `388185a6c283359e12e282655fa2b05056d95b93400dcb8db32b7bd57181b190` | `e864cb4ab44ee32080e06601cc821eae2e4e7631` | counted block map |
| `ddm_mst1_manufactured_stage_split_20260822.md` | `a22b7825d1abe9d5be7d9acb2f0591d3b77770a11bb8e5d2752e0c71b86f9e11` | `1c33f278920b91bf922e9620deb9ce20615135e8` | 21,493/23,757 manufactured mass |
| `ddm_msr1_manufactured_seg_reduction_20260823.md` | `750e26e9be1c674e15288de75b30a13793679f1a463e5453253bdf28f175abad` | `7624816b02b7891030f6038647949c4ff245554c` | balanced-flow one-net-pixel ceiling and withdrawal of boundary re-aim |
| `ddm_jf2_terminal_diagonal_harvest_20260826.md` | `3fca9f4dc29d80cee3a2a822f92109764e03ec7ac84aa8bed333651119a0f81d` | `a6afeab2cde7f4000b62fefbb769cce692cbc916` | trained diagonal terminal and pose-dose closure |
| `ddm_s1e_off_floor_adjudicator_20260825.md` | `3b94c73fa345e0e278fdb793db377fe74c33e7597f61d4c4ab618e5def903d3a` | `70009a6d0729077d042c2762b459578493c6a725` | scoped W96 screen, not population negative |
| `ddm_bo2_born_small_distortion_row_20260824.md` | `9e7b71aa4b2d88d9ed73117591ee284beeaef911f8930bf0a806a5d8bfc22e84` | `6cb3714d1b849228a86fb18a7b36b98bde65a07e` | HG1 instance closure and scope exclusions |
| `ddm_bs2_born_small_carrier_20260824.md` | `8adfc42880fe8bc778c9b185a32421bd66e73c21d0477ed8b9084c284f7d9344` | `4292833f82ccd93a134df06ba1656f6749078d69` | 101,128 B body and 36,858 B fixed-distortion headroom |
| `ddm_ce1_allocation_ladder_verdict_20260817.md` | `b7a3814dbdaef665eeba7e8cc519ae90e39a5b8cc014e9af35bd55e0e6261f3e` | `77701c0445da029982e73456cddbe3898d869975` | 81.19% CE allocation and expected-flip 13.6x configuration signal |
| `ddm_oa2_frontier_objective_alignment_20260817.md` | `8db61c09f8fdf3a288ba8f7acf23d94962f8c26a3a05bc9855832222165a6e30` | `1f1bfa3173f4a3f89c8d5fb3b6aef17a7099e234` | HPAC rate-only alignment boundary; no W96 magnitude transfer |
| `ddm_tv2_evaluator_tolerance_curve_20260824.md` | `26a5a86c183595f4185d59f5050db558eeae80fbaef0ad5d92fe8bd0f3786245` | `3b5a2f55099a51cdca9e08a03466fea0a3760c33` | best tolerance row still >=33.7x exchange rate |
| `ddm_pc2_pose_carrier_live_remainder_20260826.md` | `12481889ca72df317c9779d003559af58f00d8aad1679c0b6baff268b9c08fc7` | `88ecc0a9b114ebc45527b040b94975ed80019354` | current carrier/pose remainder empty |
| `ddm_d3b_lossless_lane_factorization_20260826.md` | `bf2d7ca91fe690c5f73d2ae81e579a107b5391bcc7ae3469790e2a65b2f96868` | `738374ded243165e3b515a4a5d9fd01283796d17` | exact factorization +207 B pre-GB1, +360 B on GB1 |
| `ddm_d3c_class_pyramid_peel_order_20260826.md` | `6bac568b921bcecdd9ff46bcb0b4dfb8f97ad06ff6b456b1db890908aaff2fa1` | `607c965bec7fedf406e48d425af2ee99f5c3ed3f` | all 24 class-pyramid screen chains dominated |
| `ddm_or1_orthogonal_representation_regime_20260826.md` | `eb1b5ba696b7c4051389040e3f5624a7ad7dddd9bee89797b71cec0b59eedcd9` | `5599b0e28bfb0d1f411c2a39078ab62619bc1afa` | current remainder thin and changed-object queues |
| `ddm_jt23_coder_collection_compose_verdict_20260826.md` | `8ca118d129891978f7f89e34a020342eefe5e71e4a2907fb07251fc10e48387b` | `01f61ac77d721d9be281f3c7cd447ab93ded1f6a` | coder axis spent |
| `ddm_lm1_learned_model_falsifier_20260826.md` | `38529715330f9ce49e0cf7fb16130157d3cc6574a4cee249a114f1ef1f74c8c4` | `6feae0ac6bdfea2371e172b4fd479186d3496c3a` | learned HPAC model replacement closed |
| `ddm_dc1_decode_time_compute_20260821.md` | `f276d2aadcd8b63045c21cfaf344a777eecbef47e174f2b3d3393f19cd84a427` | `badc6e2e9b816c27a37f904066db85a4c5bbfc85` | decode-compute family definitions |
| `ddm_dc1s_sparse_grid_sweep_20260821.md` | `cb638dc09853559b46894c478cca4a49945d159eacf1af049027cea44ecb6b3e` | `955b7e426666b7ad2d3ecc6d325d0ef8b4b812b9` | full-pop sparse dictionary closure |
| `ddm_w96a_aligned_config_renderer_window_20260826.md` | `9c5e38199390d5c97071b43388c1fd0a7e28b5937ba4fb1c32a8c375408d9ed6` | `fc915c771f` | concurrent W96A storage/objective blockers and 35-row OFF replay |
| `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/BODY_RESULT.json` | `ea3ce5b18ec88d1451c5cd90cd49afc97ee1e52b67cebfe1524aa7abf49f84f3` | retained external receipt | concurrent 101,150 B deterministic exact body |
| `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/FIRE_ORDER.json` | `d684c9bc859f825e5d5341c822dcd8c989f91d3a8e7aef1a44316ced3b333db5` | retained external receipt | sealed n32 scorer fire order and reference instruments |

The canonical equation `score_marginal_lagrange_multipliers_v1` supplied only the symbolic
`lambda_rate=25/N`; it has no empirical anchors and is not treated as calibrated evidence.

## RECALL EVIDENCE

Searched the full `.omx/research/` corpus by content with the queries `gb1`, `groupbin8`,
`sub-0.12`, `renderer`, `born-small`, `manufactured seg`, `balanced flow`, `pose carrier`,
`sharp optimum`, `trained diagonal`, `class pyramid`, `decode-time`, `dictionary`, and
`learned model`; searched `CANONICAL_RESEARCH_INDEX_20260629.md`; searched the full
`sub015_DAG_*` FEED surface for `gb1|sub-0.12|born-small|renderer-corner|decode-time|aligned-config`;
queried the canonical equation registry for `score_marginal_lagrange_multipliers_v1`; and
searched the canonical task ledger plus keeper surfaces for the three successor routes.

Beyond the charter seeds, this found JT23's zero-byte coder closure, LM1's learned-model
closure, DC1S's full-pop dictionary loss, OR1's exact post-closure map and already-queued
renderer-born-small task, D3B's 153 B comparison-basis shift on GB1, and the already-issued
W96A/BS3 charters and keepers. During FB2, their harnesses then produced a W96A blocker receipt
and the BS3 101,150 B exact body/fire order. These changed the plan in six ways:

1. `T` is closed more strongly than the charter's JF2-only statement;
2. D3B is +360 B, not +207 B, against the current GB1 subsystem;
3. `R+B` is folded into the existing OR1 task instead of creating a duplicate;
4. W96A and BS3 are queued by their existing charters rather than improvised here;
5. W96A is now typed BLOCKED on exact objective plus storage, with the hypothesis unmeasured;
6. BS3 has discharged its rate/receiver half but remains BLOCKED on scorer ownership.

The canonical index is historical and contained no current GB1 route rows in the searched
scope. The current DAG supplied decode-time context but no later receipt that reverses the
source closures above. These are bounded absence statements, not global nonexistence.

## Ledger receipts

The successor arms registered their rows while FB2 was in progress. FB2 therefore appends
idempotent routing notes instead of duplicating tasks. All three added rows use actor `ddm_fb2`.

| ledger row | task | row SHA-256 |
|---:|---|---|
| 691 | `ddm_w96a_aligned_config_renderer_window` | `4d7a01b37929c4a5f0cc3fb698c43376ac6eb51c47fb5aca73a13007312e5a05` |
| 692 | `ddm_bs3_born_small_resolved_carrier` | `65f4391a1b8341f46a1e9966ac7a6d0fa775034e16f9503e46c6ed1dc2ef547d` |
| 693 | `ddm_or1_renderer_born_small` | `2f632fb36096689be661d2dbdf504e0e81b1398952eb356a352b3d3e17bf3cf6` |

- `ddm_w96a_aligned_config_renderer_window`: BLOCKED, QUEUED-WITH-A-FIRE-ORDER; owner `MAIN`;
  consumer `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/`; trigger: >=45,521,567,744
  free bytes plus reviewed exact expected-flip implementation, then two sequential retained seeds.
- `ddm_bs3_born_small_resolved_carrier`: BLOCKED, QUEUED-WITH-A-FIRE-ORDER; owner `ddm_bs3`
  with scorer step owned by `MAIN sole scorer-lane router`; consumer
  `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/`; trigger: exact-object scorer
  ownership, no active full-n600 job, and all BODY_RESULT identities revalidated.
- `ddm_or1_renderer_born_small`: QUEUED-WITH-A-FIRE-ORDER; owner `MAIN renderer-training
  successor`; consumer `/Volumes/APDataStore/pact/ddm_or1_renderer_born_small/`; trigger:
  receiver-closed <=137,986 B changed-object candidate with resumable training and no hidden
  video-derived code.

## GESTALT-DELTA

GB1 did not merely buy 153 B: it crossed the zero-distortion integer surface. The campaign
is no longer rate-infeasible at the impossible zero-distortion corner, but every real
current-body constituent remains individually insufficient or receipt-closed. Therefore the
shortest honest route is a tightly gated two-axis changed-object test, not another local
token/coder perturbation.

**Own-vehicle frontier: unchanged — S 0.14811799921260607 @ 180,215 B `[contest-CUDA T4 n600]`, GB1 archive SHA-256 `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`.**
