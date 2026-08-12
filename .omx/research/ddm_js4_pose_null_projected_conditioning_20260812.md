# ddm_js4 — fixed custody-PoseNet-null conditioning receipt (2026-08-12)

## Verdict

The bounded hidden-4 JS4 instance is **FOLDED**. It preserved a real robust SegNet overlap after fixed
pose-null projection, but it did not approach the T4 movement gate, its pose failure was already present
before uint8 rounding, and its current receiver representation requires 452,988,928 bytes of per-pair
projector bases. No long burn, n600 scorer job, candidate archive, or exact evaluation was launched.

Authority is `[macOS-CPU advisory, instrument floor 0.0131 S]`, seeded stratified-random n32. The
`projected_n600_*` fields below are weighted projections from the 32 strata, not an n600 scorer run.

| bounded result | measured value |
|---|---:|
| steps / batch / CPU threads / seed | 25 / 16 / 8 / 20260812 |
| baseline errors / candidate errors on n32 | 2,686 / 2,799 |
| all beneficial / harmful flips on n32 | 303 / 416 |
| robust beneficial / harmful flips on n32 | 64 / 48 |
| sample robust delta / projected-n600 robust delta | -16 / **-305** |
| T4 robust gate | ≤ -2,000 |
| selected parse-backed module | live int8, **744 B Brotli q11**, SHA `f5348670…` |
| first-order `||J_p c_proj||₂`, n32 mean / p95 / max | 4.434e-8 / 1.122e-7 / 1.805e-7 |
| pose delta from zero-correction CP135 rerender, stratified n32 | -9.548e-6 |
| continuous projected-correction pose delta, stratified n32 | **+8.836e-4** |
| uint8 projected-correction pose delta, stratified n32 | **+8.574e-4** |
| uint8 total pose delta vs custody, stratified n32 | **+8.479e-4** |
| uint8 total distribution p50 / p95 / max | 3.212e-4 / 3.418e-3 / 4.448e-3 |
| pairs at or above the 2e-6 guard | **26 / 32** |
| rounding increment, stratified n32 | **-2.619e-5** |
| current receiver payload lower bound | 452,989,672 B vs ≤1,500 B |

The first-order constraint worked numerically, but the finite correction left the tangent plane: only
3.077e-5 of raw correction energy was removed on average, yet continuous nonlinear pose damage was
about 442 times the 2e-6 guard. Rounding improved the stratified mean; it was not the cause of failure.

## Derivation and implementation

For each sampled pair, JS4 computes the six-row Jacobian of the frozen custody PoseNet output with
respect to the JS3 pre-R correction. The derivative path is

`c_theta -> bilinear 384x512-to-874x1164 -> uint8 STE -> bilinear PoseNet resize -> differentiable YUV6 -> PoseNet pose[:6]`.

The forward point is recentered exactly on the retained custody plane with a straight-through identity;
the CP135 zero-correction rerender offset is measured separately. All 32 Jacobians were rank 6 at the
canonical `1e-4 * sigma_max` threshold. Their effective dimensions ranged 1.0249–1.1314, median 1.0666,
matching the recalled rank-1-dominant structure.

Rather than form ill-conditioned normal equations directly, the canonical QR/SVD machinery produces an
orthonormal row basis `Q_p`. The differentiable batched projection is exactly

`c_proj = c_theta - Q_p^T (Q_p c_theta) = (I - J_p^T (J_p J_p^T)^+ J_p) c_theta`.

The basis is fixed for the run and routed by content-hashed semantic tokens. `lambda_pose=0` is deliberate:
JS3's inherited upstream preprocessing call is no-grad, so a nonzero reported scalar would not change the
gradient. Pose is still measured at the stage boundary and in the separate continuous/uint8 decomposition.

The landed runner adds nonblocking single-writer locking after the tool transport allowed an apparently
closed process to continue and a resume briefly overlapped it. The original in-memory writer produced the
promoted step-25 checkpoint and every final measurement. The duplicate was interrupted after the shared
step-20 authority. The measured runner bytes are retained separately; the landed source only changes
routing/locking after the measured mechanism.

## Falsifiers and dispositions

- **F1 did not fire (INSTANCE):** 64 robust beneficial pixels survived projection, against 48 robust
  harmful pixels; the weighted projected robust delta was -305. This establishes overlap for this module
  instance, not T4 sufficiency and not an n600 family verdict.
- **F2 did not fire as chartered (INSTANCE):** the total pose guard failed, but quantization alone did not.
  Continuous correction damage was +8.836e-4 and the rounding increment was -2.619e-5. The original
  auto-generated CVP follow-on is therefore superseded and **FOLDED** by the harvest adjudication.
- **F3 fired (INSTANCE, current representation):** 744 B of module plus 452,988,928 B of retained basis
  arrays gives a 452,989,672 B lower bound, and no complete contest receiver exists without those bases.
  This is not a lower bound on a future projector-free compilation.
- **MAIN burn: FOLDED.** Its command is sealed for provenance but must not fire from this result.
- **Curvature-aware trust region: QUEUED-WITH-A-FIRE-ORDER.** MAIN owns it; the consumer store and trigger
  are in `QUEUE_ANNEX.md` under the SSD run directory.
- **Projector-free compilation: QUEUED-WITH-A-FIRE-ORDER.** It fires only after a projected child both
  moves robust flips and passes pose below 2e-6.

## RECALL EVIDENCE

Searched the full `.omx/research/` corpus by content with `pose-null`, `PoseNet Jacobian`, `ker(J)`, `Q3`,
`#532`, `#714`, `#837`, `#889`, `uint8`, `CVP`, `tangent`, `curvature`, and `Gauss-Newton`; also searched
the canonical research indexes, `sub015_DAG_*` FEED blocks, task/queue ledgers, the lane registry, and ran
`tools/list_canonical_equations.py --json` before building.

Beyond the charter seeds, the search found four plan-changing facts:

1. `.omx/research/pose_crux_and_protection_20260610T195607Z.md` had already measured that a linear
   PoseNet-null is tangent-only and fails at finite width because of second-order curvature. This changed
   the smoke from a single post-uint8 number into first-order, continuous nonlinear, and rounding terms.
2. `.omx/research/ddm_sb1_20260804/sb1_receipt.md` classified its purported #837 support as incomplete
   custody (only an n2 `/tmp` smoke), so JS4 did not treat “seg-reachable” as settled.
3. `.omx/research/ddm_se2_20260804/SE2_SEG_SURVIVAL_Q3_RECEIPT_20260804.md` measured exact YUV6-Q3
   projection preserving pose but collapsing its prototype's Seg reach from 0.263238 to 0.017007. That
   result is a different exact-kernel formulation, but it made the JS4 F1 overlap measurement load-bearing.
4. `.omx/research/ddm_la1_20260805/RECEIPT.md` later refuted raw Q3 projection as a joint TR1 window
   constraint while explicitly leaving curvature-corrected conditioning open. This prevented a family
   claim and supplied the scoped curvature follow-on after JS4's continuous leakage fired.

The registry/index search found no stronger current-vehicle theorem that made the finite correction
Pose-invariant or made the per-pair bases receiver-free. The canonical score equation and authority rules
therefore remained unchanged; no score was computed from this advisory smoke.

## Verification and custody

- 30 focused JS4/JS3 tests pass; the projector, fixed-custody recentering, resume promotion, source/config
  pinning, complete-receiver fold, and single-writer refusal are covered.
- The relevant JS4 and JS3 suites pass Ruff and py_compile; the payload-discard detector reports zero
  findings. Both Python files received two post-fix review-tracker passes.
- All 32 Jacobians and all 32 bases are retained with per-file bytes and SHA-256. Each array is
  14,155,904 B; the projector manifest is 39,857 B, SHA `75655e17…`.
- Final result: `/Volumes/VertigoDataTier/pact/ddm_js4_20260812/FINAL_RESULT.json`, 43,423 B,
  SHA `21bb694d…`.
- Leakage receipt: `/Volumes/VertigoDataTier/pact/ddm_js4_20260812/projector_measurement/RESULT.json`,
  9,428 B, SHA `a805fc00…`.
- Harvest adjudication: `/Volumes/VertigoDataTier/pact/ddm_js4_20260812/HARVEST_ADJUDICATION.json`,
  4,432 B, SHA `76ef8830…`.
- Queue annex: `/Volumes/VertigoDataTier/pact/ddm_js4_20260812/QUEUE_ANNEX.md`, 994 B,
  SHA `bf1282cc…`.
- Exact measured source: `/Volumes/VertigoDataTier/pact/ddm_js4_20260812/source_custody/ddm_js4_pose_null_projected_conditioning.measured.py`,
  56,129 B, SHA `8989a846…`.

No exact row moved. Own-vehicle frontier remains **lc2 S=0.16959899569230852 @ 187,226 B
[contest-CUDA T4, n600]**.
