Round 2 of rp1's rate-directed token pre-distortion, re-based onto move 40's field and closed by a
carrier re-solve. The n600 K=192 sizing shards (4,503 argmax-neutral tokens, 15,552 first-order bits)
had been verified against sj1's pass-4 body — move 37's field — because the sizing script took its
ranking from a flag and its base field from a source constant. rp1 caught this before any heavy step
and re-verified the accepted set on move 40's field with the same cumulative bisection
(`experiments/ddm_rp1_rebase.py`, daadbf7a2): 4,142 of 4,503 tokens survived; the 307 control pairs
whose base plane was unchanged transferred 2,401 of 2,401 tokens (exactly 1.0), so the instrument did
not move and the 361 lost tokens belong to the base alone. Twin encodes of the full rebased field
showed the first-order promise (−1,756.6 B) inverting under the real coder (+620 B); the pass survived
only as a per-pair selection (146 saving pairs), priced by real encode, not by the ledger sum. The
carrier was re-solved on the resulting field and the pose read on the RESOLVED pose: the admission's
net is pose −1.74e-4 (4.649e-6 vs base 4.887e-6), seg exactly 0 over 117,964,800 cells (12,540 local
flips = 12,540 predicted), rate +5 B (+3.3e-6). Frame 0 was measured inside the admission (17
adoptions, all outside the kept set, −1.95e-6 = 0.1× bar) and NOT adopted. The receiver is move 40's,
byte-identical except archive.zip and the two pins; the decode-wall-clock leg is inherited from move
40's `t4_direct` measurement (990.054 s on Tesla T4).
