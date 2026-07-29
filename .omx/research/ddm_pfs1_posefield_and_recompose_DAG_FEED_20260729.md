# DAG FEED — ddm_pfs1 warp-base recompose + e_p pose-field production solve (2026-07-29)

FEED-pfs1-a [MEASURED — stale-frame confound caught+cured]: p3v2's s3 warp point (0.3931)
was solved on the ct1 FRAME_ROOT frames (state 2a2c0367, 07-25), which differ from the pb1
endpoint token-render at max_abs 255 — the s_t stream did NOT transfer. Re-solved on the
archive's OWN shipped f1: n600 mean d_pose 0.22155 → contribution 1.4884 (BETTER than the
stale 1.9827 — cleaner endpoint frames warp better). Consumer: any carrier solved on cached
frames must re-verify frame identity at CONSUMPTION (staleness confound class), and warp
carriers should always be solved on the receiver's own render.

FEED-pfs1-b [MEASURED — the second own-vehicle exact-protocol row]: recomposed archive
569,996 B (sha 624ffe57…; pose_warp.stp 6,864 B replaces the 6-cosine pose.tpgn 1,876 B)
through archive→inflate.sh→evaluate.py, full 600, locked env, rc=0: PoseNet 0.22144216 ·
SegNet 0.00389011 · S = 2.256641 [macOS-CPU advisory — real evaluator, real bytes] —
ΔS −18.018 vs the pb1 20.2746 row, entirely the pose axis. Drift: d_seg BIT-EQUAL to the
pb1 evaluator value (frame_0-seg-free law confirmed on deployed bytes); solve-instrument
vs live-GT d_pose |Δ| ≈ 1.0e-4. Modal Stage-B staging REPOINTED at this sha (QA17 surface;
operator-GO unchanged). Consumer: pn1 Stage-B; the honest-gap ledger.

FEED-pfs1-c [MEASURED — three warp-solver laws, each one aborted chunk]: (1) co-optimizing
s_t with the pose 6-vector walks the s_t·dim0 scale ridge to f16-fatal points (0.116
f64-solved → 10.22 f16-shipped); fix s_t. (2) raw t_p rotation dims through expmap are
POISON (74× worse start) — PoseNet dims 3–5 are not raw metric rotations; rotation must
start at ZERO and enter as GN-grown capacity. (3) the objective is f16-LATTICE-BOUND along
dim0 (spacing 0.03125 at |34|; a 0.011 rounding cost +10 d_pose at a ridge point) —
quantized-acceptance GN (accept only shipped-representable improvements) is the honest
solver. Consumer: every future pose-carrier solve on any vehicle.

FEED-pfs1-d [MEASURED — the e_p warp-pose ladder; FALSIFIER FIRED]: per-pair GN over the
FULL 6-DOF plane-induced homography (rotation ACTIVATED via s_r=1; the D1 receiver's s_r=0
left dims 3–5 inert) through the real receiver path: 0.221441 → 0.159509 (contribution
1.2630) at ~equal bytes; SVD-of-p_star rank rungs: rank-1 int8 1.4383 @ 702 B, rank-2
1.3779 @ 1,164 B, rank-4 1.3159 @ 2,004 B (int8 ≡ int16 to 4 decimals — coeff quantization
free). THE PRE-REGISTERED FALSIFIER FIRED: no ≤4KB warp+e_p point beats contribution 0.5
(best 1.3159; even unpriced 6-DOF reach 1.2630 = 25× the gc8 ≤0.05 drop-condition).
Residual TYPED: tail-concentrated — median pair solves to 0.0027, but 9.3% of pairs carry
74.6% of the mean (the 71–90 turn cluster; worst 4.83) and their structure is OFF the
ground-homography manifold (parallax/photometric). e_p field = rank-~2 (SVD 0.906/0.074),
translation-dominant — sc1's rank-1 law reproduced on a second base. Consumers:
**v10 SPEC row-12 — pose-in-burn returns to REQUIRED** (the p3v2 "optimization choice"
framing is superseded); gc8 extension-window pose axis STAYS (drop-condition not met);
QA25 carries the measured curve; the rank-1 702-B stream + 6dof f16 stream are grammar-v4
rate levers for the next recompose.

Pointer 0.1910828242 [contest-CPU] UNMOVED. All rows [macOS-CPU advisory]; score_claim=false.
