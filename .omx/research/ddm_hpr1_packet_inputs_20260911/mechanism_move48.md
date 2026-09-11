Move 46's lever re-applied on move 47's object: the HPAC prior's frame embedding (4,800 values) rounded to even for 2,310 of them,
on top of the RETRAINED prior that move 47 shipped. The prior is built from the same bytes at both ends before any symbol is decoded,
so the rounding conditions the tail's code length without touching a decoded symbol: `hpac` 12,262 → 11,629 B (−633), RLC1 stream
118,511 → 118,896 B (+385), archive 179,359 → 179,111 B (−248). Output-lossless by receipt: the decoded token field is byte-identical
to the shipped field (a92e7d90…) and the cold n600 public decode is byte-identical to the pointer's raw (2b762eba…; four archives now
decode to one raw), so d_seg and d_pose are unchanged and the score change is the rate term alone, 25·(−248)/37,545,489 = −1.6513e-4.
The rounding's value did not depend on the prior being stale (−245 B on the old prior, −248 B on the refit). Receiver unchanged
(behavior digest 9f6e7168… on moves 44–48); the move went through the first-measurement chain because the contract admits one
inherited row per measured leg and move 47 had consumed move 46's; its completion gives move 48 a measured T4 leg of its own. The
pre-fire risk gate ran in pr19's identity-class envelope (ceiling 1,232.42 s from moves 44/46's measured legs; stress 1,477.6 s ≤ 1,800).
