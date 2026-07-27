# G87 exact G78-to-G72 compile adapter and adversarial geometry closure

Date: 2026-07-27  
Axis: `[macOS-CPU advisory]` research-only compile materialization  
Pointer delta: **UNMOVED**  
Candidate/archive/score claim: **none**

## Outcome

G87 now strictly reopens the fresh full-n600, batch-16 G78 target/base-scorer
cache and maps all five chronological 120-pair stages into G72's real
role-aware V9 shearlet proposal derivation.  It materializes the complete
nonempty representable component universe with immutable, atomic, resumable
stage checkpoints:

- 75,882 exact 8-connected Road/Undrivable components;
- 303,528 proposals, four per component;
- 4,493,276 mismatch sites;
- maximum 248 components in any pair-role, proving the 4,096 representational
  cap nonbinding;
- no selection thresholds, selected-preimage operand, packet, archive, score,
  or promotion claim.

The receiver-facing dense scorer fields remain encoder-only and are explicitly
forbidden from candidate payloads.

## Adversarial finding: bounding-box contamination

The first r1 materialization exposed a real G72 geometry bug.  G72 correctly
labeled connected components, but then computed each component's centroid,
shear, missing/excess sign, and Fisher priority from every mismatch site inside
that component's bounding box.  A disconnected island inside a ring therefore
contributed to the ring's geometry and priority.

The full-n600 population audit measured:

| role | components | contaminated | foreign sites included |
|---|---:|---:|---:|
| Road | 64,253 | 6,975 | 694,771 |
| Undrivable | 11,629 | 571 | 18,360 |
| Total | 75,882 | 7,546 | 713,131 |

Contamination affected 9.944387338235681% of all components; the maximum
foreign inclusion in one proposal component was 2,063 sites.  The component
census itself remained correct, but r1 proposal geometry is invalid and is
therefore preserved only as a superseded false-authority artifact:

The independent exact r1-to-r2 proposal diff changed 30,184 of 303,528
proposals (the same 9.944387338235681%), exactly four proposals for each of the
7,546 contaminated components.  Road contributed 27,900 changes and
UndrivableBoundary 2,284.  Field-level impact was:

- Fisher priority: 30,184 proposals;
- center: 22,212;
- shear: 18,140;
- amplitude: 896;
- scale: 0.

Changed proposal counts by stage were 5,572 / 5,708 / 6,652 / 5,860 / 6,392.
This exact correspondence proves the fix changed only the contaminated
component families rather than perturbing the full proposal population.

- root:
  `/Volumes/VertigoDataTier/pact/taskspace_g87_g78_to_g72_complete_proposals_n600_20260727_r1`
- aggregate file SHA-256:
  `861c1d72170ec5866f2cc439e5d39c4ddee580d690a7e2ea0b5b995f94ddb7cd`
- aggregate self SHA-256:
  `0afabaefd638a6429728d64cea33c55047a47edf9ffe495d18e03f5f7e4ea697`
- promotion eligible: false
- payload selection allowed: false

## Structural correction and invariant

G72 now retains the connected-component label map and computes all geometry
only from `component_labels == component_id`.  A runtime invariant requires the
selected membership count to equal the independently counted component site
count before any proposal can be emitted.  The proposal-law identity was
versioned from:

`V9_FISHER_MARGIN_BOUNDARY_SHEARLET_ROLE_PRESERVING`

to:

`V9_FISHER_MARGIN_BOUNDARY_SHEARLET_ROLE_PRESERVING_EXACT_COMPONENT_MEMBERSHIP`

A ring-with-enclosed-island regression independently exposes the former fault:
the 40-site ring and disconnected 16-site island must produce a Fisher-priority
ratio of exactly 40/16, rather than the contaminated 56/16.

The fresh r2 aggregate records the post-fix live-data invariant across all
75,882 components:

`exact_component_membership_drives_geometry=true` and
`bbox_foreign_mismatch_sites_included=false`.

## Custody transition: why G78 r5 was rerun

G78 r4 had correctly sealed the G72 consumer source into its preflight.  Once
G72 changed, strict reopen failed with:

`sealed input changed: g72_consumer_contract`

No bypass or compatibility relaxation was used.  A fresh governed,
crash-resumable G78 r5 reran all 38 scorer batches under the corrected G72
source.  Its 5 stage receipts preserve all 10 stable dense-field/upstream
SHA-256 identities from r4; only the expected preflight and stage-receipt self
identities changed.  This is evidence that the scorer data stayed frozen while
consumer custody advanced.

- r5 preflight file SHA-256:
  `103a8c2f04d304406d655e19710c65b92078429a3553463e339464fd9babd14a`
- r5 preflight self SHA-256:
  `58bce5e2b20813e74f241e61f0f4bf4a114d629c23b149553297b2ab6807a1c8`
- r5 aggregate file SHA-256:
  `3ca3cf5503d5b5144f48e2e562e398780a74a91fcfbd59e60248849d279d6215`
- r5 aggregate self SHA-256:
  `22926b9689d5e493b3b8bb1e0a6dbfe1d26d6a6d56756e6426f591a4cf8dbf28`
- run: 515.49 s, 7,062 MiB peak RSS, 38 resumable batch checkpoints,
  5 immutable stage checkpoints.

## Fresh r2 proof

Root:
`/Volumes/VertigoDataTier/pact/taskspace_g87_g78_to_g72_complete_proposals_n600_20260727_r2`

- tree file bytes: 99,361,100
- aggregate bytes: 5,272
- aggregate file SHA-256:
  `27ffb9b9ddd6068828f36b5a572046aa92d22f1ffe70c0d2d2af696023f4e951`
- aggregate self SHA-256:
  `60cdf20839db7a1d63b390c106582b423e1c0befa076a0c9a4a57b2e63f7b168`
- G87 compile-input self SHA-256:
  `370898d6a62bb424ec9848e228929ac1d0fb93157ce85623e145dbbaa2facf4e`
- first materialization: 21.88 s, 1,465 MiB peak RSS
- strict resume replay: 6.41 s, 1,304 MiB peak RSS
- resume verdict: all six output file SHA-256 identities bit-identical

## Exact remaining boundary

G87 closes only G72's fresh target-margin custody and fresh V15 camera-through-R
base scorer cache blockers.  It does not close:

1. the G49 role-aware analytic wire;
2. V15 role-aware decoder proof;
3. fresh pose authority or final exact pose replay;
4. exact whole-object joint admission.

No payload selection should begin until the role-aware wire and decoder proof
exist.  Proposal priority is an ordering signal, not an admission rule; final
admission remains the nonlinear whole-object score/byte differential.
