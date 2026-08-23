---
verdict_scope: INSTANCE:DX2_UNCHANGED_FIELD_GREEDY_CAUSAL_CORRECTOR_FAMILY_PRUNE
axis: macOS-CPU advisory / scorer-free exact RC64 byte measurement
score_claim: false
shipping_candidate: none
---

# DDM MP3 — DX2 HPAC member prune

## Result first

**Mandatory all-19 control deficit: 0 B.** Cold-starting the shipped 19-family online
corrector on the unchanged 600-frame field reproduced the incumbent **113,777 B** RC64 stream
byte-for-byte, SHA-256 `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5`.

The charter joined two different objects. The 19 pruneable members are generic causal receiver-code
families and own **0 separable counted archive bytes each**. The **13,515 B** AR1B span is a fixed
neural HPAC probability model, not storage partitioned among those families. It remains intact in
every row. The byte census nevertheless closes exactly: **19 × 0 B + 13,515 B identified fixed
neural object = 13,515 B, with 0 unexplained B**.

Real leave-one-out re-encoding found three non-positive single removals. The charter-specified greedy
MDL ladder reaches its instance-scoped optimum at **three families removed / 16 remaining**:
`surprise_fast256`, `shipped_fast256`, and `homog_surprise_fast256`. Its stream is **113,743 B**, a
**34 B** archive reduction at unchanged distortion and rate-component **ΔS =
−2.26392044062e-05** using the cited `25/37,545,489` exchange rate. The best fourth removal is
**+2 B marginally worse**, so it is the stopping row.

The measured 34 B supplies **34 / 42,382 = 0.0008022274 = 0.0802227%** of the fixed-distortion
sub-0.12 demand and leaves **42,348 B**. It removes no byte from the fixed 13,515 B neural object.
No shipping candidate or exact score row was built, and the frontier did not move.

## Member-byte census

Every number in this section is `[macOS-CPU advisory / scorer-free physical-byte census]` on the
DX2 `archive.zip` SHA `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`.

| Object | Separable counted B | Identity |
|---|---:|---|
| `shipped_joint` | 0 | generic causal receiver code |
| `temporal_spatial` | 0 | generic causal receiver code |
| `surprise_only` | 0 | generic causal receiver code |
| `spatial_surprise` | 0 | generic causal receiver code |
| `spatial_boundary` | 0 | generic causal receiver code |
| `run_surprise` | 0 | generic causal receiver code |
| `boundary_surprise` | 0 | generic causal receiver code |
| `temporal_surprise` | 0 | generic causal receiver code |
| `shipped_fast256` | 0 | generic causal receiver code |
| `shipped_fast4096` | 0 | generic causal receiver code |
| `surprise_fast256` | 0 | generic causal receiver code |
| `spatial4_surprise` | 0 | generic causal receiver code |
| `homog_surprise` | 0 | generic causal receiver code |
| `homog_boundary_surprise` | 0 | generic causal receiver code |
| `spatial4_boundary` | 0 | generic causal receiver code |
| `homog_spatial4` | 0 | generic causal receiver code |
| `spatial4_temporal` | 0 | generic causal receiver code |
| `homog_surprise_fast256` | 0 | generic causal receiver code |
| `spatial4_surprise_fast256` | 0 | generic causal receiver code |
| **19-family separable subtotal** | **0** | no learned family payload serialized |
| **fixed 64-channel IHS1 neural HPAC object** | **13,515** | separate object; AR1B physical span `[45,13560)` |
| **accounted total** | **13,515** | **0 unexplained B** |

The fixed physical span SHA is
`602115b323b0e403d08287af9b273a2d4fb23e026d83c1f6e4609ed77ef98f98`; it restores the canonical
17,952-byte IHS1 object SHA
`e8c0cfd73d3275adeff2897ea83efa9d045855c43fb3bb66ac037e5c84f2e6dd`. “Refit” on this member-count
surface therefore means the shipped receiver's real online fit: the surviving correctors start
cold, emit the unchanged neural prior for cold cells, and update only after each decoded symbol.
It does not mean retraining or shrinking the separate neural IHS1 model. JF1 owns that changed-field
neural-refit surface.

## Positive control

All denominators are the complete **600 × 384 × 512 = 117,964,800-symbol** unchanged DX2 field.

| Quantity | `[macOS-CPU advisory / scorer-free exact RC64 bytes]` |
|---|---:|
| incumbent stream | 113,777 B |
| all-19 cold-refit stream | 113,777 B |
| **control deficit** | **0 B** |
| incumbent / control stream identity | exact, `e2af55e641c…a4ac5` |
| reconstructed stored member | 180,268 B, `365f1b8d7046…31c7a` |
| independently decoded field | 117,964,800 B, `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |
| distortion change | exactly 0 by field identity |

## All 19 leave-one-out rows

Every row is a real full-field native RC64 re-encode after cold-refitting the surviving 18 online
families. Model bytes saved are zero because these members are free receiver code. The ZIP member is
stored and its non-token prefix is held fixed, so stream delta equals physical archive delta 1:1.
Every row was independently decoded and matched the full field SHA `cc10a7b09353…63efb`.

| Removed member | Model B saved | Stream B | Stream B added | Net archive ΔB | Rate-component ΔS | Field SHA comparison: decoded = source `cc10a7b…63efb` |
|---|---:|---:|---:|---:|---:|---|
| `surprise_fast256` | 0 | 113,759 | −18 | −18 | −1.19854611562e-05 | PASS |
| `shipped_fast256` | 0 | 113,765 | −12 | −12 | −7.99030743747e-06 | PASS |
| `homog_surprise_fast256` | 0 | 113,765 | −12 | −12 | −7.99030743747e-06 | PASS |
| `surprise_only` | 0 | 113,777 | +0 | +0 | +0 | PASS |
| `homog_surprise` | 0 | 113,778 | +1 | +1 | +6.65858953122e-07 | PASS |
| `homog_boundary_surprise` | 0 | 113,779 | +2 | +2 | +1.33171790624e-06 | PASS |
| `temporal_spatial` | 0 | 113,780 | +3 | +3 | +1.99757685937e-06 | PASS |
| `spatial_surprise` | 0 | 113,780 | +3 | +3 | +1.99757685937e-06 | PASS |
| `boundary_surprise` | 0 | 113,780 | +3 | +3 | +1.99757685937e-06 | PASS |
| `temporal_surprise` | 0 | 113,782 | +5 | +5 | +3.32929476561e-06 | PASS |
| `spatial4_temporal` | 0 | 113,782 | +5 | +5 | +3.32929476561e-06 | PASS |
| `spatial_boundary` | 0 | 113,788 | +11 | +11 | +7.32444848434e-06 | PASS |
| `shipped_fast4096` | 0 | 113,789 | +12 | +12 | +7.99030743747e-06 | PASS |
| `homog_spatial4` | 0 | 113,789 | +12 | +12 | +7.99030743747e-06 | PASS |
| `spatial4_boundary` | 0 | 113,790 | +13 | +13 | +8.65616639059e-06 | PASS |
| `spatial4_surprise` | 0 | 113,794 | +17 | +17 | +1.13196022031e-05 | PASS |
| `spatial4_surprise_fast256` | 0 | 113,800 | +23 | +23 | +1.53147559218e-05 | PASS |
| `shipped_joint` | 0 | 113,803 | +26 | +26 | +1.73123327812e-05 | PASS |
| `run_surprise` | 0 | 113,804 | +27 | +27 | +1.79781917343e-05 | PASS |

These ΔS entries are exact rate-component arithmetic at `6.658590e-07 S/B`, not exact-evaluator
score rows. No corresponding receiver-integrated `archive.zip` was built in this charter.

## Greedy MDL ladder and stopping row

At each rung, every one-member extension of the currently adopted removal set was physically
encoded; the smallest stream was independently decoded. The stopping criterion is the marginal
archive change from the prior adopted rung, not merely whether the row remains smaller than the
all-19 baseline.

| k removed | Removed jointly | Members left | Stream B | Total ΔB | Marginal ΔB | Total rate ΔS | Adopted | Field SHA comparison: decoded = source `cc10a7b…63efb` |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | `surprise_fast256` | 18 | 113,759 | −18 | −18 | −1.19854611562e-05 | YES | PASS |
| 2 | `shipped_fast256`, `surprise_fast256` | 17 | 113,748 | −29 | −11 | −1.93099096405e-05 | YES | PASS |
| **3** | `shipped_fast256`, `surprise_fast256`, `homog_surprise_fast256` | **16** | **113,743** | **−34** | **−5** | **−2.26392044062e-05** | **YES, k\*** | **PASS** |
| 4 | add `surprise_only` to k\* | 15 | 113,745 | −32 | **+2** | −2.13074864999e-05 | **NO, stop** | PASS |

This is a greedy-ladder instance verdict, not an exhaustive proof over all `2^19` subsets.

## Prediction adjudication

The prior-law prediction was conjunctive. Its directional clauses held: at least one single removal
was net-negative, and the greedy optimum removed at least two members. Its magnitude clause did not:
the prediction required **more than 1,000 B**, while the real optimum freed **34 B**. Therefore the
overall stated prediction is **falsified**, although the charter's narrower “all 19 leave-one-out
rows positive” falsifier did not trigger.

The mechanism behind the magnitude miss is now explicit. The naive **711 B/member** price assigned
shares of the fixed neural object to zero-stored receiver-code families. There are no such stored
shares to harvest. The only available gain on this surface is a small change in RC64 code length.

## Boundaries and disposition

- **Measured:** exact physical 13,515 B / 17,952 B neural-object custody; exact full-n600 native
  RC64 control, 19 leave-one-out streams, all greedy candidate streams through the k=4 stopping
  generation, and independent full-field identity for every verdict row.
- **Not measured:** d_seg, d_pose, a receiver-integrated candidate archive, clean-environment
  `inflate.sh`, contest-CPU/CUDA evaluation, or any exact score. No scorer, Modal, Metal, or local
  advisory fire occurred. The field identity makes distortion invariant conditional on future
  correct receiver integration; it is not a substitute for that integration proof.
- **Ownership:** the field was never changed; JF1's concurrent changed-field refit tree was neither
  read as an input nor modified. AR1B/EF1/OE1/AD2/BL1 trees were cited, not duplicated. `upstream/`
  and the sacred JO1 run were untouched.
- **No shipping candidate:** required by the charter. MP3's deliverable is the table and retained
  byte evidence.
- **Follow-on disposition:** `QUEUED-WITH-A-FIRE-ORDER`; owner=`MAIN`; consumer store=`the next
  governed DX2 receiver-code composition receipt, citing this memo and RESULT.json`; fire
  trigger=`MAIN next assembles a receiver-code rate candidate on the exact DX2 body and can integrate
  the measured 16-family SHIPPED_CONFIG, rebuild its RC64 suffix, pass clean-env full decode to
  cc10a7b…63efb, then run the authority evaluator under the single-flight lane`. It must not fire as
  a standalone 34-byte candidate.

## Custody and resumability

The charter-authorized **local explicit-opt-in tier** was used exclusively:

`/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_mp3_hpac_member_prune/`

No writes were made to either `/Volumes/*` tier. Inputs there were read-only. The retained receipt
contains 3,651 manifest-indexed files totaling **7,380,222,100 B** (6.9 GiB): pinned inputs, every
candidate stream and reconstructed stored member, all LOO decoded fields, selected ladder decoded
fields, and complete encoder/decoder/corrector checkpoints every 20 frames. Two external SIGTERM
interruptions resumed from those stage receipts without retyping or losing payloads.

| Receipt | SHA-256 |
|---|---|
| `RESULT.json` | `5fb042902bb1722402af467c9a3ad681725a618b310adc7f73b12c7b01c45d34` |
| `MANIFEST.json` | `3456c87dddfb18e79ce8e23c8403f9445b36faddb406a2d1591d2853aa9e63a3` |
| implementation measured by final receipt | `45fd33ae30b66d047a39306f6a7981a1726a2a82aca8a17586269e6035c2e691` |

## Verification

- `python -m py_compile`, the experiment self-test, and Ruff passed.
- The P0 measure-and-discard payload gate returned **0 findings / 1 scoped Python file**.
- Result audit found **19/19 LOO** and **4/4 ladder verdict rows** at the exact full field SHA.
- The implementation received two genuine `review_tracker.py` passes after its final edit.
- The manifest was regenerated after the final source hash and result adjudication were sealed.

## RECALL EVIDENCE

Recall covered the full required corpus before the premise was accepted or the experiment was
routed.

- Research/receipt content queries: `13,515`, `13515`, `19-member`, `families`, `zero-stored`,
  `generic causal`, `receiver code`, `IHS1`, `member count`, the archive/stream/field hashes, and
  `SHIPPED_CONFIG`. Searched `.omx/research/`, arm receipts, and the retained shipped receiver.
- Graph/index queries: `HPAC`, `corrector`, `member count`, and `FEED-fx2` across
  `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, `canonical_task_status.jsonl`, and the P0 ledger.
- Canonical-equation query: `hpac|corrector|rc64|member count` over
  `tools/list_canonical_equations.py --json`; the only match was the run-scoped
  `hpac_mc36_joint_descent_law_v1`, not a stored-byte attribution for these receiver families.
- Design/SPEC and task-ledger queries used the same terms and exact hashes. The bounded corpus search
  did not find a prior real removal-and-reencode table on this exact DX2 unchanged-field family set.

Beyond the charter seeds, `ddm_fx5_composed_rate_candidate_20260821.md` and DAG `FEED-fx2` state that
the 19 families are compiled generic receiver code at zero counted bytes. The actual shipped
`free_corrector.py` confirms that `SHIPPED_CONFIG["families"]` names code paths whose learned state is
cold-created and causally updated; it is not deserialized from the 13,515 B blob. AR1B's parser
custody separately identifies that blob as the fixed neural IHS1 model. This changed the plan from
an invalid neural per-member prune/trainer sweep to the real zero-stored corrector-family subset
experiment above. JF1's newly available receipt further confirmed that neural model refit is a
different, changed-field-owned surface and was not reusable here.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: `MAIN`; consumer store: the next governed DX2 receiver-code
  composition receipt plus this memo/`RESULT.json`; fire trigger: a next exact-DX2 composition is
  already being assembled, at which point integrate the measured 16-family k\* configuration,
  re-encode, clean-env decode to `cc10a7b…63efb`, and authority-evaluate under the single-flight lane.

## LIVE-HYPOTHESES

- The 34-byte k\* gain will compose without distortion with the next exact DX2 receiver-code
  candidate because it changes only free causal receiver code and the RC64 suffix, while every
  measured row reconstructed the unchanged field. This remains untested at archive/runtime closure.

## DEAD-ENDS

- Assigning roughly 711 B of the 13,515 B neural object to each corrector family is closed: all 19
  are zero-stored code, and the 13,515 B object remains separately identified and byte-custodied.
- The prior-law magnitude forecast of more than 1,000 B is closed on this greedy instance: the real
  optimum is 34 B, only 0.0802227% of the demand.
- Continuing the specified greedy removal ladder past k\*=3 is closed: the best fourth extension is
  +2 B marginally worse after real re-encode and exact decode.
- Adding another causal escape family remains closed by OE1; generic-estimator substitution,
  reorder, coder swap, and storage-layout changes remain closed by EF1/AD2/TO2/XS1/MZ2 and were not
  reopened here.
- Changed-field neural refit is not an MP3 continuation: JF1 owns and is already executing that
  distinct surface.

Own-vehicle frontier: **dx2 — S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]**, unchanged.
