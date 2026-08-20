Built and landed the real residual-free PGQ1 receiver.

- Selected `target_quadratic_previous_f10_q20`.
- Archive: **183,137 B**, SHA-256 `c408adf9101bb19a363039a5e0f7185aabce8f31edb6787e2deaf6d0fe6738f4`.
- Exact rate change: **−4,089 B versus LC2**, but **+15,132 B versus the non-renderable 168,005 B envelope**.
- Full-600 coefficient surrogate: ordinary non-circular R² `0.4860878253721611`; 8/7,200 exact codes; zero endpoint crossings.
- Repeat archive is byte-identical. Public-runtime parse matches the direct parser, and a PGQ mutation changes both predicted coefficients and rendered frame bytes.
- CPR1 and the exact coefficient residual are absent; the counted LC2 basis remains.

Measured only on `[macOS-CPU scorer-free receiver build]`. No full rendered n600 video, `d_seg`, `d_pose`, advisory score, contest score, or timing verdict was produced.

PS135 still owns the active scorer dispatch and PID 43675 still holds the physical fleet lock. PZ4R was marked queued; no lane was claimed.

Artifacts:

- [Durable findings memo](/Users/adpena/Projects/pact/.omx/research/ddm_pz4r_pgq1_receiver_20260811.md)
- [Builder](/Users/adpena/Projects/pact/experiments/ddm_pz4r_pgq1_receiver.py)
- [Receiver](/Users/adpena/Projects/pact/experiments/ddm_pz4r_runtime/pose_gauge_receiver.py)
- Result: `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/PZ4R_RESULT.json`, SHA-256 `85c24a269b2380bb3ddf1d1a2ea39779d7541d0ea266f289b280d6dbe876c6c3`

Landed commits: implementation `c7b9387b9638a490194a0be121fdca0e624a0759`; memo `5c962530fe420a4fe69de645f03f78d12e23cf24`. Final validation: 9 tests passed, lint clean, completed-resume revalidation passed, staged index empty.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER; owner: `codex:ddm_pz4r` or next explicit scorer-lane claimant; consumer store: `selected_workload_root` recorded in `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/queued_full_n600_storage_plan.json`; fire trigger: PS135 has a retained terminal receipt and terminal dispatch row, the retained Vertigo→APDataStore preflight admits 5 GB plus an 8 GiB reserve, the queue reports scorer-free, PZ4R claims its lane, and `lockf` acquires the physical scorer lock.** Run the memo’s exact retained full-n600 decode and both-metric-axis evaluation.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER; owner: future PZ4R joint-receiver arm; consumer store: `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/joint_v1/`; fire trigger: direct-v6 measures `d_pose > 4e-5` or Seg collateral above `0.002` score units.** Train a counted, resumable output-conditioned renderer.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER; owner: future cp135/PR135 refit arm; consumer store: `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/cp135_refit/`; fire trigger: direct-v6 has a realized receipt and exact cp135/PR135 output and carrier banks are under custody.** Refit lineage-specific state; transfer apparatus only.

## LIVE-HYPOTHESES

- The 4,089-byte saving may outweigh moderate pose or Seg degradation; only the queued realized score can decide.
- Pose may survive despite coefficient R² 0.4861 because the true objective is PoseNet output after rendering, not coefficient parity.
- Joint conditioning may beat coefficient regression by directly selecting a useful frame preimage.
- A cp135-specific refit may combine this mechanism with the current rate frontier, but no LC2 weights or measurements transfer.

## DEAD-ENDS

- **FORMULATION:** PGQ1 or the 168,005-byte envelope alone cannot render; both lack the required basis/coefficients.
- **INSTANCE:** Retained PZ3 weights cannot be reused because their input object is official-DALI-GT PZ2, not LC2-decoded PGQ outputs.
- **FORMULATION:** Exact coefficient residuals are closed by PZ3’s 194,120-byte result.
- **FORMULATION:** Direct PoseNet-output-to-SE(3) warp control remains inadmissible.
- **FORMULATION:** Modular coefficient-error selection is closed because it hides renderer discontinuities.
- **INSTANCE:** The unversioned attempt and `direct_v2`–`direct_v5` are retained correction history and must not be scored.
- **INSTANCE:** Scoring now is closed while PS135 owns both the dispatch and physical lock.

Own-vehicle frontier unchanged: cp135 `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`; PZ4R moved no exact score and did not reach sub-0.15.