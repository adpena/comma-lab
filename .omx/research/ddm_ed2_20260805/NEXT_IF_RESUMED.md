# ddm_ed2 next if resumed

Current disposition: **FOLDED** for the qo1 IX2 `alpha=0.25` discrete
entropy-descent candidate. Do not rerun this archive or spend another n600
scorer on it.

Measured row:

- axis: `[macOS-CPU frozen-scorer advisory]`
- archive:
  `/Volumes/VertigoDataTier/pact/ddm_ed2_20260805/entropy_descent_a025/archive.zip`
- archive sha256:
  `4df0ec7cea34a2e57824b2e9d3e940c44a66ee4644cf71f7a81bd5b2c9f3f852`
- archive bytes: 350,130
- upstream report:
  `/Volumes/VertigoDataTier/pact/ddm_ed2_20260805/entropy_descent_a025/evaluate_report.txt`
- reported n600 components: `d_seg=0.00449912`, `d_pose=0.01071092`,
  rate term `0.00932549`, rounded final `1.01`
- exact SegNet decomposition:
  `.omx/research/ddm_ed2_20260805/ed2_seg_decomp.json`
- exact d_seg count: base 508,640 errors; candidate 530,739 errors;
  `delta_errors=+22,099`

Why folded:

- actual byte win is 7,706 B, worth only 6,052.902579 flip-equivalent errors at
  `W=1.27310821533 B/flip`
- rg5 fixed comparison was 10,441 B, worth 8,201.188143 flip-equivalent errors
- this candidate adds 22,099 net SegNet errors and therefore fails both bars
- pose bank is destroyed: reported `d_pose` rises from `0.00071459` to
  `0.01071092`

If the operator reopens #866, use this fire order instead of repeating ed2:

1. Wait for the jd5 pose-base boundary or another explicitly pose-safe base.
2. Byte-only screen a smaller alpha grid before any scorer spend, for example
   `alpha in {0.025, 0.05, 0.10}` on the exact live token surface.
3. Promote at most one alpha to a stratified n>=120 CPU scorer screen only if
   bytes move in the right direction and token edits are much smaller than ed2.
4. Spend n600 only if the n>=120 screen is plausibly below the live own-vehicle
   baseline after both d_seg and pose-bank accounting.

Do not cite this as a family kill. Scope is only:
`qo1 IX2 alpha=0.25 discrete entropy step on lattice-center tokens`.
