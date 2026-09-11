The shipped pose carrier (CPR1) stores an AR(1)+bias predictor whose residuals are Rice-coded. The shipped predictor was a
clipped fit: ten of its twelve biases sat at the ±16 clamp, so it was never rate-optimised. ddm_pc3 refit the predictor by an
exhaustive search over the closed legal schema at BIT-IDENTICAL codes (51,581 → 50,270 Rice bits); `decode_cap1` reconstructs the
byte-identical canonical CPR1 either way, so the decoded carrier, the token plane, and every rendered frame are unchanged. Archive
180,406 → 180,246 B (−160 B). The cold n600 public decode is byte-identical to move 44's retained raw across all 3,662,409,600 bytes,
so d_seg and d_pose are move 44's by construction; the only score change is the rate term, 25·(−160)/37,545,489 = −1.0654e-4.
The refit's fitted values live in the counted archive (rule 118: content, not code); the receiver is unchanged after pin
normalization, which is why the seal inherits move 44's t4_direct leg through the pr18 behavior digest (49 rows, equal).
