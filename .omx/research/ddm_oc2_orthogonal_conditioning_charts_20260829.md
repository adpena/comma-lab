# ddm_oc2 — orthogonal conditioning charts on the lb1 body

**Verdict: `FAMILY_DRAINED_ON_LB1__MISS_RANK8_BANKED_2B__NO_FIRE`.** The prior-law
prediction was falsified in the tested scope. The two seed charts failed their frozen
orthogonality screen, and the one chart that was orthogonal in a real full-n600 comparison
saved only **2 B** after patch192. That is 28 B below the pre-registered 30 B solo-fire bar.
No chart was admitted, so no native port, decode-identity run, candidate seal, lane claim,
Modal dispatch, scorer load, or authority-score claim occurred.

All local numbers below are `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`, and
`promotable=false`. This arm was scorer-free and did not modify `upstream/`.

## 1. Source re-derivation and current-body headroom

The source receipt re-derives the mi1 indicator-ledger excess from its components:

| quantity | value | denominator / boundary |
|---|---:|---|
| indicator code length | 111,275.62229665744 B | all 117,964,800 n600 token positions |
| fitted-model entropy | 109,113.49589719012 B | same positions |
| realized excess | 2,162.126399467321 B | difference above; z = 11.8889805050 |
| current-body collections | 285 B | exact archive chain, not an entropy-ledger quantity |
| remaining conditioning-mass proxy | **1,877.126399467321 B** | 86.8185319753% of mi1 excess; 0.0001273007812 bit/position |

The current physical lineage closes as follows:

| collection | physical archive transition | bytes collected on current body |
|---|---:|---:|
| gb1 | 180,368 -> 180,215 B | 153 B |
| jt21 | 180,215 -> 180,192 B | 23 B |
| lb1 patch192 | 180,192 -> 180,083 B | 109 B |
| total | 180,368 -> 180,083 B | **285 B** |

jt22 measured 1 B in a separate dx2 mixer-context race. It is evidence that marginals
overlap, but it is not present in the lb1 physical lineage and therefore is not added to the
285 B subtraction. The 1,877.126 B result is a campaign routing proxy: it subtracts exact
archive bytes from an indicator-ledger excess and is not an information-theoretic hard bound.
Source: `measurement_v1/PREFLIGHT_AND_PREREGISTRATION.json` sha
`bd110a152ddfdef784e22347149b3bc28da0ae0c086431bbd1f6315197abcddc`.

## RECALL EVIDENCE

Before designing or adjudicating charts, I searched the full research corpus by content with
the query family `patch192|conditioning|groupbin8|scan.order|magnitude.rank|temporal.*class|causal.*class`,
then inspected the matching mi1/lb1/gb1/jt21/jt22/jt23/fs2 receipts. I also searched
`CANONICAL_RESEARCH_INDEX*`, the `sub015_DAG_*` FEED blocks, design/spec surfaces, and task-ledger
content for `1326`, `model axis`, `indicator excess`, and `orthogonal conditioning`; and ran
`tools/list_canonical_equations.py --json` before choosing a chart.

Beyond the charter's named seeds, the bounded recall found the current gb1 block-map/factorization
evidence in `ddm_or1`: the live body already ships `groupbin8_surprise`, so scan-order phase is
already a consumed axis rather than a new independent chart. No same-body surviving free chart
beyond the lb1 hypothesis was found in those scopes. No directly applicable canonical equation
licensed transferring an overlap number across substrates. This changed the plan by excluding a
scan-phase relabel and using pre-mixer runner-up mass rank as the distinct continuation after the
two seed charts closed. The exact task-1326 owning row was not found in the bounded task-ledger
scope; the charter and live queue birth remained the ownership source.

## 2. Frozen chart predictions and measured overlap

Predictions were written before each chart's outcome. patch192 was reproduced with the correct
receiver expression `(y//32)*16+(x//32)` and measured 206.368624 B on the same seeded random
two-fold indicator screen. The screen covered all 50,009,121 live positions out of
117,964,800 total positions; it was not a prefix.

| chart | distinct structural axis | predicted patch192 overlap | measured patch192 overlap | real n600 marginal after current body | disposition |
|---|---|---:|---:|---:|---|
| `temporal_transition30` | predicted class x previous decoded-frame class | 20% | **991.624%** | NOT RUN -- frozen screen closed it | `CLOSED_NONORTHOGONAL`; solo ledger +11.787 B became -105.099 B after patch192 |
| `causal_edge30` | predicted class x first available causal-neighbour class | 35% | undefined because solo gain was negative | NOT RUN -- frozen screen closed it | `CLOSED_NONORTHOGONAL`; solo ledger -7.323 B, conditional ledger +21.978 B |
| `miss_rank8` | runner-up probability as a share of total non-argmax mass | 15% | **0%** | **+2 B saved** | `BANKED_BELOW_SOLO_FIRE_BAR` |

The first two rows are an explicit scope reduction, not physical-byte claims. Their frozen
full-live-position screens failed the admission condition, so outcome-driven tuning and costly
physical encodes were forbidden. `CHART_SCREEN.json` sha
`7ca71f89c0c404638fc27966695acd7b4193c9b7ed3e1f1d955f163eecab90a5`.

The rank prediction was independently sealed at 17:31:36-0500, before either n600 receipt at
17:49:57-0500 and 17:50:31-0500. `RANK_PREREGISTRATION.json` sha
`c099122bb319fa2560ccbb88febb843c4a2c9b2a79192b9de4fc930c66fecfab`.

## 3. Real full-n600 joint price and retained payloads

The exact overlap for `miss_rank8` came from two real, resumable n600 encodes, not additive
ledger arithmetic:

| base | archive transition | token-stream transition | exact saving | candidate custody |
|---|---:|---:|---:|---|
| jt21 without patch192 | 180,192 -> 180,190 B | 113,601 -> 113,599 B | 2 B | `rank_solo/retained/candidate_oc2_rank_solo.zip`, sha `7f3a1a6f…e2fe8` |
| lb1 after patch192 | 180,083 -> **180,081 B** | 113,492 -> 113,490 B | **2 B** | `rank_after_patch/retained/candidate_oc2_rank_after.zip`, sha `67fb0b8e…80cab` |

Both encodes covered 600 frames, changed zero tokens, used the same proven RC64 inverse, and
retained the stream, per-frame ledger, checkpoints every 25 frames, receipt, build, and candidate
archive. The exact marginal-after-patch rate change is
`-2 * 25 / 37,545,489 = -1.3317179062443428e-6 S`, versus the admission threshold
`-30 * 25 / 37,545,489 = -1.997576859366514e-5 S`. The measured overlap is
`1 - 2/2 = 0%`; orthogonality survived, economic magnitude did not.

The adjudication receipt is `measurement_v1/RANK_ADJUDICATION.json` sha
`a4e22137b893647300569fd29ac53405456fba1bec8d76d94a29eec200e86ba3`.
The retained manifest covers 346 files and 25,130,549 B at
`measurement_v1/MANIFEST.json` sha
`67dab07caa886b9010560b91a680922faac900126347bcb33cb5b54f3a980f99`.

## 4. Admission, identity, and drained-family verdict

No set cleared 30 B. Therefore decode identity was **not run** and is **not claimed** for the
180,081 B Python-runtime candidate. This follows the charter's floor precisely: full native
generation-22 port and receiver identity are owed only for an admitted set. Generation-20 was
not used. No seal or dual-axis fire order was created, because sealing a 2 B bank as a solo row
would violate the frozen bar.

Typed negative: `FAMILY_CLOSED_ON_TESTED_LB1_BODY`. Scope is the free, decode-derived chart
formulation on the generation-22 lb1 body across temporal transition, local causal class edge,
and pre-mixer non-argmax mass rank, with scan phase already consumed by shipped groupbin8. It is
not a global claim that no conceivable conditioning model exists. Within this charter's current-
body family, however, the prior-law falsifier fired: no orthogonal candidate cleared 30 B, so the
remaining 1,877.126 B proxy is not reachable by retrying these charts. The model axis joins the
coder axis as closed on this body; a paid-table rescue remains priced out by mi1's 47.4x miss.

## 5. Component arithmetic and frontier boundary

The current own authority anchor remains gb1 contest-CUDA:
`S=0.14811799921260607 @ 180,215 B`. Holding its measured distortion components fixed, lb1 at
180,083 B projects to `0.14803010583079396`, and the unadmitted 180,081 B rank candidate projects
to `0.1480287741128877`. Those are same-distortion rate projections, not exact eval rows. No
authority evaluation was run, so the canonical pointer did not move and this arm did not achieve
the sub-0.12 goal.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER / BANKED:** owner `MAIN lossless-bank composer`; consumer store `/Volumes/APDataStore/pact/ddm_oc2_orthogonal_conditioning_charts/`; fire trigger: when MAIN assembles the next independently admitted lossless candidate on the same lb1-compatible body, include `miss_rank8` in that candidate's one real full-n600 joint re-encode, and only if the combined exact marginal clears the then-live bar perform the generation-22 native port, full decode identity, seal, lane claim, and authority fire.

## LIVE-HYPOTHESES

- The banked 2 B `miss_rank8` fact may survive a future independently admitted lossless composition because its real saving was exactly 2 B both before and after patch192, i.e. measured 0% overlap with patch192. It is plausible only as a rider; it is not worth another solo run.
- A genuinely new representation may reopen conditioning on a future body because mi1 still exposes a large fitted-model gap while the current body's adaptive free families show sharply shrinking marginals. This is body-change evidence, not licence to retry a chart or paid table here.

## DEAD-ENDS

- `temporal_transition30` is closed on this body: measured overlap with patch192 was 991.624%, and its conditional held-out marginal was -105.099 B.
- `causal_edge30` is closed on this body: its solo held-out gain was negative, so orthogonality was undefined; its +21.978 B conditional ledger signal was also below the 30 B bar.
- `miss_rank8` is closed as a solo-fire path on this body: real full-n600 marginal was only 2 B, missing the frozen bar by 28 B. Do not retune its bins after seeing this result.
- A new scan-phase relabel is closed as duplicate structure: the live body already ships the gb1 `groupbin8_surprise` scan-order chart.
- Additive bank arithmetic, coder/ZIP framing re-races, transmitted patch tables, the wrong `tile48*4+subtile4` patch index, paid-conditioning rescue, and generation-20 native reuse remain closed by the consumed lb1/jt23/mi1 negatives and were not retried.

**OWN-VEHICLE FRONTIER: S=0.14811799921260607 @ 180,215 B [contest-CUDA n600, gb1]; unchanged.**
