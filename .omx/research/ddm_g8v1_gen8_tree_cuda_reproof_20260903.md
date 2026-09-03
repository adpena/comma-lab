# ddm_g8v1 — gen-8 tree contest-CUDA identity re-proof: CONFIRMED (2026-09-03)

## Verdict

The generation-8 shipping tree (`submissions/semantic_joint_ctxmix/`, commit 5a26779938,
review-sealed 3/3) ran its own public entrypoint on a 1:1 Tesla T4 against the same
archive bytes as the pointer and reproduced the afr1 row EXACTLY. The pre-registered
identity falsifier (seal sha `efa0a839f01e10ecab58bc880bce794eb27032e1a84348a96a1a8e66447f80f7`)
did NOT fire. The score association for the gen-8 packet is now DIRECTLY MEASURED,
not bridged by local byte-identity.

## The row [contest-CUDA T4 n600]

- call `fc-01M1JJ0YK6YWVSTF4YRGZ81D4D`, lane `ddm_g8v1_gen8_tree_cuda_reproof_20260903`
- archive `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`, 180,002 B (sha verified in-run)
- Average PoseNet Distortion 0.00000637 — IDENTICAL to the afr1 base receipt
- Average SegNet Distortion 0.00020139 — IDENTICAL
- Submission file size 180,002 bytes — IDENTICAL
- Recomputed S from components = 0.14797617125559104 — IDENTICAL (report displays 0.15)
- Wall: inflate 532.33 s + evaluate 40.57 s ≈ 573 s on-GPU (gen-7 measured 578.9 + 42.7;
  the hardened wrapper is marginally FASTER; 3.1× inside the 1,800 s budget)
- Receipt: `experiments/results/ddm_g8v1_gen8_tree_cuda_reproof_20260903/t4_row/MODAL_REMOTE_RESULT.json`
- Spend: ~588 s Modal T4 ≈ $0.10–0.15 (fold into the #381 ledger; headroom now ~$1.25)

## Compliance consequence (the review's 4 REDs → 2)

- `submission_runtime_tree_matches_auth_eval` — CLOSED BY MEASUREMENT: this receipt binds
  the CURRENT tree (staged from commit 5a26779938 + the pinned archive) to the score.
- `auth_eval_raw_promotion_policy_blockers_absent` — SUPERSEDED: the fresh receipt is the
  packet's auth row; the stale-receipt flag no longer has an object. Adjudicated here in writing.
- `contest_cpu_auth_eval_exists` — RECORD-WITH-REASON stands (CPU measured over-budget,
  killed at 1,800 s; documented in the packet).
- `hosted_archive_manifest_supplied` — operator-only publish gate, by design.

Residual live-hypothesis count from the g8r review: ZERO. The packet is READY for the
operator's edit + final go-ahead. Nothing publishes without it.

## Provenance

Seal: `experiments/results/ddm_g8v1_gen8_tree_cuda_reproof_20260903/seal.json` (SEAL_VALID,
bound-base falsifier computed from `/Volumes/APDataStore/pact/ddm_pq12/afr1_authority_materialized/MODAL_REMOTE_RESULT.json`).
Fired via tools/fire_modal_auth_eval.py --seal (stage-0 validated); single-flight clear;
poller supervised (counter 725), harvested clean rc=0 in 589 s.
