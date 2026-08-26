The PC2 drain is complete: the GB1 pose/carrier axis has no untested live candidate. All named paths were closed, consumed, or superseded by D3B. No new payload, scorer run, score, or frontier move was produced.

The complete receipt is [ddm_pc2_pose_carrier_live_remainder_20260826.md](/Users/adpena/Projects/pact/.omx/research/ddm_pc2_pose_carrier_live_remainder_20260826.md).

Key results:

- Full bounded recall covered all named families and 124/124 matching `NEXT_IF_RESUMED` bullets.
- AP1 remains decisive: the cheapest carrier purchase returns 2,742 B but costs `+0.304506605 S`. Even removing the entire 22,010 B carrier for free cannot offset that damage.
- Eighteen terminal candidate exits are `completed` in the canonical ledger. The post-D3B re-accounting row is `pending` under MAIN.
- Ledger validation returned `{"rows":657,"status":"valid"}`; two reported malformed histories predate PC2.
- MAIN’s scorer lane was occupied, but no PC2 candidate qualified for scoring anyway.

The source repository denied Git-object writes before staging: `unable to create temporary file: Operation not permitted`. The main index remains untouched. A serializer-authored, transfer-ready commit was retained instead:

- Commit: `b9ae14c5bc62fa92910e8fbebe32abc30d1925cb`
- Memo SHA-256: `12481889ca72df317c9779d003559af58f00d8aad1679c0b6baff268b9c08fc7`
- Verified [Git bundle](/Volumes/APDataStore/pact/ddm_pc2_commit.uRh8we/ddm_pc2_b9ae14c5bc.bundle), 7,929 B, SHA-256 `0e3e455e6c1b8b0a76cc8a7610e2295d32b1a7bdd17a59d6ba800328db1e57b4`

`[contest-CUDA T4 n600] own-vehicle frontier: GB1 — S=0.14811799921260607, archive=180,215 B, d_seg=0.00020139, d_pose=6.37e-6; PC2 did not move the pointer.`

## NEXT_IF_RESUMED

- `QUEUED-LANDING` — owner: `MAIN/operator`; consumer store: `/Volumes/APDataStore/pact/ddm_pc2_commit.uRh8we/`; fire trigger: a checkout with Git-object write authority becomes available. Import/cherry-pick commit `b9ae14c5bc`.
- `QUEUED-AFTER-D3B` — owner: `ddm_pc2 successor under MAIN`; consumer store: `/Volumes/APDataStore/pact/ddm_pc2_pose_carrier_live_remainder/post_d3b_reaccount/`; fire trigger: D3B publishes a terminal receiver-closed archive with decoded-object equality, payload/archive hashes, repeat hash, and complete byte accounting.

## LIVE-HYPOTHESES

- D3B may change the complete-body byte distribution enough to make fresh pose/carrier accounting useful; this is plausible because it changes the representation while preserving task tokens losslessly.
- A fundamentally different jointly trained pose carrier may escape the AP1/RB1 and Film-W96 closures; it must be a new sufficient statistic, not another frozen-body refit.

## DEAD-ENDS

- AM1 delta2: PA2 measured it larger than delta1 and flat.
- PZ4A/PZ4R: metadata defeats precision savings; direct-v6 destroys pose.
- PK2/PK3: 432 gauge states produced at most 64 B, missing admission.
- BR1 rotation: numerically null; GN splice already consumed.
- Frozen-body coarsening, rank, refit, and waterfill: AP1/RB1 measured zero score-negative supply.
- S1 Film-W96 diagonal: its preregistered falsifier fired throughout.
- RC1/NI1, D3, and D3A: real rate savings do not survive evaluator distortion.
- SF1 format sum: the claimed saving combines mutually exclusive, undecodable formats.