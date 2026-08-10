# DDM PK3 rate-aware gauge preflight

**Trigger: NOT_MET.** All numbers are [macOS-CPU scorer-free byte/MSE projection]; `score_claim=false`.

| candidate | stage | carrier B | basis bits | coeff bits | archive B | saved B | receiver MSE | MSE pass | trigger |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `E4_decoder_dead_radial_2cbc11e6920828bfdbd4` | E4_decoder_dead_radial | 23053 | 104135 | 79069 | 190988 | +64 | 1.35873111526e-15 | PASS | NOT MET |
| `E4_decoder_dead_radial_6c1df4e58fce81d2bc32` | E4_decoder_dead_radial | 23053 | 104135 | 79069 | 190988 | +64 | 1.35873111526e-15 | PASS | NOT MET |
| `L1_givens_coarse_bee83fca78069fbcbb42` | L1_givens_coarse | 23053 | 104135 | 79068 | 191024 | +28 | 2.90136805489e-07 | PASS | NOT MET |
| `L1_givens_refine_d6102a97438707cb53b8` | L1_givens_refine | 23053 | 104135 | 79069 | 191024 | +28 | 7.44489289437e-08 | PASS | NOT MET |
| `L2_directed_shear_675d88e5cc48870fac18` | L2_directed_shear | 23053 | 104135 | 79069 | 191024 | +28 | 1.61217340064e-08 | PASS | NOT MET |
| `E3_exact_permutation_1f37f0d6813cd32dc0bf` | E3_exact_permutation | 23053 | 104135 | 79069 | 191028 | +24 | 3.53182133045e-16 | PASS | NOT MET |
| `L3_global_homotopy_451e4462a0e886206d19` | L3_global_homotopy | 23056 | 104135 | 79092 | 191028 | +24 | 2.66891872835e-05 | FAIL | NOT MET |
| `E1_exact_sign_3bd584ee7982d7080f69` | E1_exact_sign | 23053 | 104135 | 79069 | 191032 | +20 | 0 | PASS | NOT MET |
| `E2_exact_dc_shift_670c3d19ffc4f4479430` | E2_exact_dc_shift | 23070 | 104267 | 79069 | 191052 | +0 | 9.90103174398e-17 | PASS | NOT MET |
| `control_ddc185b00d850c5dc411` | control | 23054 | 104135 | 79076 | 191052 | +0 | 0 | PASS | NOT MET |

The best MSE-admissible gauge saved 64 full-archive bytes at directly measured receiver-product MSE 1.35873111526e-15. The byte bar was 2000 and the strict MSE bar was 2.5e-06.
All 432 unique materialized candidate rows are in `/Volumes/VertigoDataTier/pact/ddm_pk3_20260809/receiver_v7/PK3R_ROWS.jsonl` (SHA-256 `25859d13bbe7f691209a6bdc5c2c0bf110bc1c84a12983110de152a76f4ba2a0`); every row points to its retained CPR1 carrier and full archive. The search is a deterministic stagewise greedy chain, not an exhaustive Cartesian product across stages.

## Authority boundary

MEASURED: scorer-free full-population CPR1 arrays; actual Huffman/Rice streams; actual full XZ+ZIP archive bytes; parser parse-back; and direct chunked `C @ normalized_basis(B) / sqrt(12)` MSE over all 353,894,400 receiver-product values for every materialized candidate row.

NOT MEASURED: realized d_pose, d_seg, rendered uint8 frames, exact contest score, QAT response, or any retrained carrier. Exact receiver-product equality preserves downstream frames and d_pose by derivation; for nonzero MSE there is no measured or derived d_pose bound. The 2.5e-6 bar is only the chartered scorer-free fire gate.

## RECALL EVIDENCE

Bounded recall covered the canonical equations, canonical indexes, DAG, hot state, task ledgers, queue, PK2 receipts, intake codec/receiver/trainer, G110, QA70 v4d, and AA1/EH1 with the exact queries recorded in FINAL_RECEIPT. It found no direct CPR1 rate-aware-gauge equation. Receiver-source recall changed both the metric and transform law; storage-law recall added real full-archive arbitration. G110, QA70, and the canonical 417 B result are different objects, and the legacy-to-CPR1 3,328 B saving is already spent; no number transferred. Queue lines 106-108 are provenance duplicates of one stable action, so PK3 did not mint another. The live cx2 and tm1 stores were not touched.

The frozen raw CPR1 section is 23,054 B: 152 B fixed prefix, 104,135 Huffman bits (13,017 B), and 79,076 Rice bits (9,885 B). The charter's 23,384 B is a distinct measured full-archive leave-one-out marginal attributed to pose, not a raw-section size. The trigger remains an actual full-archive delta.

## Boundaries and disposition

The verdict is `INSTANCE × DECLARED_SEARCH_SURFACE`: the frozen PR130 CPR1 carrier and the bounded exact-affine, normalization-aware pairwise, and structured near-identity bank recorded in the receipt. It does not refute arbitrary GL(12), QAT, retraining, or new wires.
Because the trigger was NOT MET, the existing QAT action is `FOLDED` with reason `TRIGGER_FAILED`; no ticket was created, no scorer or training ran, and the pose-rate reopening is closed on this vehicle only for this pinned instance and declared search surface.

The exact frontier is unchanged: PR130 CPR1 `S=0.172141297491896447 @ 191,052 B [contest-CUDA, DALI GT, n600]`.

## LIVE-HYPOTHESES

- None authorized by this NOT_MET result. A broader correctly receiver-normalized GL(12) construction remains untested and mathematically plausible because PK2 tested uncorrected raw-space rotations, but PK3 does not promote it or leave a fire order.

## DEAD-ENDS

- The preserved `RAW_PRODUCT_V1` run: invalid for the trigger because it used reciprocal coefficient scaling and raw 24x32 `C @ B` MSE.
- The preserved receiver-v2 run: not used because NumPy matmul emitted overflow/invalid warnings in its screen; v3 remeasured the bank with explicit finite-checked contractions.
- The preserved receiver-v3 run: superseded because its candidate identities did not bind the runner source and its prose conflated raw CPR1 bytes with the distinct leave-one-out archive marginal.
- The preserved receiver-v4 run: aborted when its self-pin detected source drift before a verdict.
- The preserved receiver-v5 run: superseded because it directly measured only verdict-bearing rows.
- The preserved receiver-v6 run: all 432 carrier/archive pairs remain retained, but the source guard withheld a final receipt after review changed the runner.
- Reciprocal basis/coefficient scaling: receiver normalization cancels basis magnitude but leaves the coefficient rescaling active, so it is not a gauge.
- Generic-basis substitution and low-rank coefficient replacement: PK2 already measured them as destructive or larger; PK3 did not repeat them.
- Bare coder replacement on unchanged pose bytes: the prior real race was +4 B; PK3 changed the gauge and used the incumbent real coders instead.
- PK2's 64 raw-space random rotations: closed only as that uncorrected implementation; correctly normalized global mixing is not declared dead.
- The unchanged PR130 QAT runner: it is fidelity-only and lacks optimizer/scheduler/RNG/order/cursor resume state, so it cannot be fired as the required rate-aware resumable continuation.
- The declared receiver-v7 bank: 432 unique states, maximum full-archive saving 64 B, 1936 B short of the gate; the existing gauge-QAT action is folded.
