# DAG FEED — ddm_p3v2 optimal-form pose re-solve (2026-07-29)

Pointer `0.1910828242 [contest-CPU]` UNMOVED. [macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE.
score_claim=false. [no-triality] [p0-ledger-ok]. Memo: `ddm_p3v2_optimal_form_pose_resolve_20260729.md`.

FEED-p3v2-a [MEASURED — the pre-registered rule: WALL REFUTED, vehicle = CANDIDATE_LINE]: the
free-frame_0 UNPRICED upper bound (work-res 192×256, STE-uint8, frozen CPU-torch PoseNet6, ~160 Adam
iters to convergence, n16) reaches mean d_pose **1.070e-4 → pose contribution 0.0327 ≤ the 0.05
binding threshold** (100% of pairs ≤1e-3; 93.75% ≤2.5e-4; max 4.6e-4). The photometric-wall N1=NO was
an ARTIFACT of P3's naive solve (zeros start d_pose~85 + rank-6 cosine basis + ~2-relin budget
truncation), NOT a vehicle property. **N1 re-decided YES.** Consumer: E2 node N1; the v10 pose-in-burn
charter (QA25) is NO LONGER forced by a confirmed wall — it becomes an OPTIMIZATION choice.
Receipt: `/Volumes/VertigoDataTier/pact/ddm_p3v2_20260729/p3v2_ladder_receipt_final.json`.

FEED-p3v2-b [MEASURED — S0: the rank-6 cosine basis is RANK_DEFICIENT]: run to convergence (~11
relins, n6) the P3 actuation plateaus at d_pose mean 15.29 / median 6.86 (traj 89.5→22.1→16.4→…→15.07)
— 5 orders above the free floor. The 38.06 "20.27 row" was dominantly a BASIS problem (generic cosine
cannot span the pose-Jacobian directions), AND budget-truncated (38→15 with more relins). #715's
covariance basis confirms the class failure: d_pose RISES 19.89→48 with rank. Generic bases are wrong.

FEED-p3v2-c [MEASURED — the cheap realizable carrier = the WARP BASE; the free win is
BASIS-ADVERSARIAL]: a ground-homography warp of frame_1 by the CARRIED 6-value pose target + a per-pair
s_t index (DECODER-REPRODUCIBLE, s_t stream = 194 B r7-SMEVR / n600) reaches d_pose n600 mean 0.393
(median 0.085; contribution 1.98) — beating 6-cosine (38), #715 (19.89), stored_f0 (8.06). BUT generic
DECODER-REPRODUCIBLE compression of the free residual over that base — 2D-DCT (low-freq + largest-mag),
low-rank SVD, AND the S2 LOTTO shared dictionary — ALL collapse back toward the warp-base class. The
pose-relevant frame_0 win is BASIS-ADVERSARIAL (needs the net's Jacobian, not a cheap generic basis);
confirms #249's expensive/adversarial finding on the tr1 vehicle. Verdict-scope FORMULATION.

FEED-p3v2-d [MEASURED — S2 LOTTO race (gc7r's flagged highest-leverage surface)]: the SHARED low-rank
frame_0 dictionary (counted once, n600-amortized) + per-pair coeffs BEATS per-pair rank-1 (R8: d_pose
0.308 @ 3,628 B/pair vs per-pair-r1 3.498 @ 2,660 B/pair) — the pose directions ARE partially shared
across pairs, a real finding. But LOTTO never approaches the free floor and is Pareto-DOMINATED by the
warp base (contribution 2.38 @ ~0 B vs LOTTO R8 1.75 @ 3,628 B). The frame_0-pixel LOTTO is closed at
FORMULATION; the JACOBIAN-aligned shared dictionary (store the pose-direction basis once) is the named
next rung, un-raced. gc7r LOTTO-audit row-3 dispositioned: RACED, dominated-for-cheap-realization.

FEED-p3v2-e [MEASURED — S3 composed-row point, n600]: replacing the P3 6-cosine pose member (38.06 @
7,295 B) with the warp-base carrier on n600 (warp mean d_pose **0.3931**, median 0.0848, s_t stream
**194 B** r7-SMEVR for all 600 pairs, pose target = already-carried sc1 t_p) moves the pb1 instrument
row pose 19.50954 → **1.9827** at slightly-LOWER rate (194 B replaces 7,295 B), seg untouched (0.38901,
frame_0 seg-free spot check CONFIRMED — SegNet argmax identical across frame_0). **Composed S ≈ 20.2746
→ 2.7431 (ΔS −17.53)** [macOS-CPU advisory; the row itself is instrument-side, QA02 evaluate fill
owed]. The remaining pose gap (~2.0 → the free 0.03 reach) is
the pose-field terminal-solve (sc1 e_p ~2 KB, joint-descent-trained) or v10 pose-in-burn conditioning
(#383) — pose entering the TRAINING loop, NOT a cheap frame_0-pixel carrier.

FEED-p3v2-f [APPARATUS — ledger + reaper]: ledger QA01 DUE → FIRED (this receipt); QA25 pose-in-burn
note updated (wall refuted ⇒ pose-in-burn is an optimization choice, not a forced head). Reaper note:
detached/tracked long runs die ~5 min in this session — the ladder + S3 were made RESUMABLE (per-pair
JSONL + npz cache) and run in sub-5-min foreground chunks; the finalizer
`experiments/ddm_p3v2_finalize_from_cache.py` aggregates the cache race-free.
