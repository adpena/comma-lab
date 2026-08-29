# DDM BZ2 — born-small capacity ceiling

**Task:** #1325  
**Date:** 2026-08-29  
**Disposition:** **PARTIAL-PROVED-AND-QUEUED**  
**Axis:** `[macOS-CPU scorer-free exact byte measurement; n600 DALI target fit]`  
**Score claim:** false

## Conclusion

The frozen HG1 born-small representation can be fit directly to the registered DALI GT partition and
packaged, with no residual corrections, into a deterministic **100,862 B** HG1 receiver-parseable
container. That is **37,124 B below** the 137,986 B fixed-GB1-distortion cap. The archive and repeat are
byte-identical (`sha256 773c7ae3e9e93dfda29fe18522f3ed8421a4c3b6ec4a52e68778054a30df54c9`),
and archive parse-back reproduces the fitted categorical field exactly.

This is **not** the requested realized capacity ceiling. The parse-backed field differs from DALI GT at
1,324,687 / 117,964,800 sites (**0.011229510837131076**), but categorical mismatch is not `d_seg`.
The inherited semantic renderer, exact R, uint8, and frozen scorer have not consumed these new fitted
bytes. FCD3 still owns the local scorer lane under the charter; BZ2 did not claim it. Therefore BZ2 has
proved byte feasibility and a native fit, while the representation's scorer-realized ceiling remains
**UNRESOLVED**.

The requested cross-object law is also **UNRESOLVED**. There are zero realized ceilings to fit: BZ2's
terminal is queued, and QBZ1's separate qbt/QBF1 training run has not published a scorer-realized
ceiling. I did not fit a law to native proxies or silently reuse QBZ1's object.

## Measured result

| quantity | result | meaning |
|---|---:|---|
| complete HG1 container | **100,862 B** | real ZIP bytes, inherited sections included |
| fixed-distortion cap delta | **−37,124 B** | byte-feasible only; no distortion credit |
| packet | 47,779 B | five real-coded stream bodies plus roster |
| inherited semantic + pose + compact sections | 52,962 B | byte-identical BS3 sections |
| ZIP/HG1 framing | 121 B | remainder after sections and packet |
| native full-n mismatch | **1,324,687 / 117,964,800 = 0.011229510837131076** | not `d_seg` |
| full-n parse-back | exact | zero residual corrections; fitted field SHA equals decoded field SHA |
| rate term at measured bytes | 0.06715986572980845 | `25*100862/37545489` |

The real-coder winners were LZMA2-extreme for Road/Undrivable (4,556 B), Lane (36,132 B), and
Movable (6,596 B), Brotli q11 for MyCar (95 B), and zlib-9 for the zero residual (22 B). Every coder
candidate and deterministic repeat is retained, not only the winners.

## Held-out controls and QBZ1 crosswalk

BZ2 uses QBZ1's exact `seed=20260829`, seeded 480/120 pair split, and modulus-5 spatial hash. A direct
code-level cross-check confirmed identical pair IDs and identical spatial masks for sampled pairs. The
represented objects remain separate: BZ2 fits HG1's horizon/lane/box/static-mask grammar; QBZ1 fits the
QBF1/qbt tensor schema.

| control | train | held out | interpretation |
|---|---:|---:|---|
| pair holdout | 1,062,697 / 94,371,840 = 0.011260742611355252 | 382,948 / 23,592,960 = **0.01623145209418403** | HG1 is address-bearing; unseen pairs filled from nearest training pair generalize materially worse |
| spatial holdout | 1,087,504 / 94,373,160 = 0.011523445861090166 | 274,278 / 23,591,640 = **0.011626067539179132** | within-frame nearest-observed completion is close to the observed-site result |

These are native categorical diagnostics only. They do not establish a scorer ceiling, a score, or a
portable law.

## BO2 arithmetic and corrected ranking

The retained BO2 report-precision components reproduce the charter arithmetic:

- fixed contest-CUDA GB1 distortion: `100*0.00020139 + sqrt(10*0.00000637)` =
  **0.028120227975693968**;
- strict fixed-distortion cap: **137,986 B**;
- ideal BO2 body: **101,128 B**, leaving **36,858 B**;
- distortion budget at that byte count: **0.02454278781296708 S**;
- matched local PyAV advisory damage: **5.131079311613639 S**;
- refusal: **209.06668593299145×**.

The numerator is a matched local PyAV advisory delta and the denominator is the contest-CUDA
fixed-distortion envelope. This is a robust refusal envelope, not a same-instrument score delta. The
retained physical BS3 container is 101,150 B, not the 101,128 B ideal arithmetic object; it leaves
36,836 B and has no matched n600 contest-CUDA distortion receipt.

BO2's exchange-ratio result remains **97.25× under convention A**, but its old “fourth-worst” statement
is withdrawn. It is **second-best of five measured regimes** under that convention. No successor should
reuse the old rank.

## RECALL EVIDENCE

Before implementation I searched 77 born-small/address-free research files plus the canonical research
index, the current DAG, the full `.omx/research` corpus, and the canonical-equation registry. Queries
included `born-small`, `born small`, `capacity ceiling`, `BO2`, `BS3`, `BS4Y`, `209x`, `97.25`,
`address-free`, `qbt`, and `qbz1`.

The closest prior work was:

- `ddm_bs4y_stage_1_4_execution_20260826.md` and
  `ddm_bs4y_stage40_adjudication_and_ap_reclaim_20260827.md`: the fresh carrier solve did not rescue
  the born-small instance; the retained stage-40 receipt measured 20 selected pairs and closes only
  that carrier formulation/instance.
- `ddm_bs3_route2_dead_adjudication_20260827.md`: the carrier route is dead, while the 36,858 B rate
  premise survives as arithmetic.
- `ddm_hg1_heterogeneous_analytic_generator_gate_20260823.md`: HG1's exact-residual horn is 460,408 B;
  deleting the residual exposes the born-small grammar but does not measure its realized distortion.
- `ddm_fb2_route_table_gb1_20260826.md`: explicitly keeps a genuinely trained changed-object renderer
  alive and distinguishes the retained 101,150 B body from the 101,128 B ideal.
- the concurrent QBZ1 executable: establishes the sister seed, 480/120 split, and spatial mask on the
  distinct QBF1/qbt object. BZ2 matched its protocol without consuming or duplicating its fit.

No prior artifact in the searched scope directly fit the complete frozen HG1 representation to the
registered DALI GT partition and retained a real-coded, zero-residual parse-back archive. BZ2 therefore
performed that missing measurement rather than repeating BO2 or BS4Y.

## Custody and verification

Durable root:
`/Volumes/APDataStore/pact/ddm_bz2_bornsmall_capacity_ceiling/`

Primary receipts:

- `RESULT.json` — typed result, source identities, controls, package, and unresolved law;
- `FULL_PACKAGE_RESULT.json` — every coder race and receiver parse-back;
- `BO2_REDERIVATION.json` — 209× arithmetic and ranking correction;
- `FIRE_ORDER.json` — owned scorer terminal and measured predecessor cost;
- `RETAINED_INVENTORY.json` — content-addressed retained inventory;
- `retained/full_package/archive.zip` and `archive.repeat.zip` — deterministic 100,862 B candidate;
- `retained/full_package/archive_parseback_tokens.u8` — 117,964,800 B fitted field,
  `sha256 968ffca296302616927de2fc35feedce64f1ddcd1efd61f970860ed1e9276b2f`.

All fitted targets, raw streams, coder candidates, repeats, packets, archives, and decoded fields are
retained. The run is stage-resumable and verifies checkpoint facts before reuse. Two independent source
review passes completed. `ruff`, `py_compile`, the DALI source preflight, the QBZ1 split/mask crosswalk,
archive parse-back, and an independent n600 confusion/mismatch recomputation passed.

The scorer-free stages measured 31.385 s (pair control), 29.707 s (spatial control), and 30.041 s
(full fit), at $0 local cost. The closest measured realization cost is BO2's retained CPU predecessor:
380.344 s inflate + 457.918 s evaluate = **839.510 s**, with 3,662,409,600 retained raw bytes and $0
cash cost. That is measured predecessor cost, not a promise that BZ2 will take identical time.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN scorer-lane router; consumer store:
  `/Volumes/APDataStore/pact/ddm_bz2_bornsmall_capacity_ceiling/realized_capacity_terminal/`; fire trigger:
  FCD3 releases the sole local scorer lane, MAIN records an active BZ2 claim, and the source/archive
  hashes revalidate. Render the retained fitted field through the inherited semantic renderer, exact R,
  and uint8; run frozen DALI-lineage SegNet/PoseNet at real n600 in resumable chunks; retain all scorer
  inputs/outputs; then type the BZ2 ceiling.
- **QUEUED-AFTER-TWO-TERMINALS** — owner: MAIN-designated BZ2/QBZ1 synthesis successor; consumer store:
  `/Volumes/APDataStore/pact/ddm_bz2_bornsmall_capacity_ceiling/cross_object_law/`; fire trigger: both BZ2
  and QBZ1 publish scorer-realized ceiling receipts on their exact fitted bytes. Fit at most the n=2
  cross-object law; otherwise publish `NO_PORTABLE_LAW`.

## LIVE-HYPOTHESES

- The 100,862 B direct-DALI fit may lower the realized born-small damage relative to BO2 because it
  changes 74,650 categorical sites and reduces native GT mismatch from 1.1301566% to 1.1229511%; this is
  plausible but untested until render/R/uint8 and frozen scoring consume the exact new archive.
- HG1 may have adequate within-frame capacity but weak pair-address generalization: the spatial holdout
  barely separates from observed sites, while the pair holdout is materially worse. The distinct
  controls make that mechanism plausible, but only one representation has been measured.
- A portable capacity predictor may depend on address freedom or decoder degrees of freedom, but n=0
  realized ceilings cannot identify it. BZ2 and QBZ1 can supply at most the two preregistered points.

## DEAD-ENDS

- Reusing BO2's 209× distortion as the new capacity ceiling is closed: BO2 scored a different fitted
  field and instrument axis; BZ2's exact new bytes have not passed through the scorer.
- Calling the 1.1229511% categorical mismatch `d_seg` is closed: it is before the semantic renderer,
  exact R, uint8, and SegNet.
- Fitting a cross-object law to BZ2/QBZ1 native proxies is closed: neither object has a realized ceiling,
  so such a law would be fabricated.
- Reopening the BS4Y carrier solve as the capacity test is closed at its instance/formulation scope: its
  fresh solve did not rescue the measured born-small carrier and did not fit the full representation to
  DALI GT.

`[contest-CUDA T4 n600] own-vehicle frontier: GB1 — S=0.14811799921260607, archive=180,215 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4; BZ2 did not move the pointer.`
