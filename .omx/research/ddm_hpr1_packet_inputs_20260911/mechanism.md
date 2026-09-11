The HPAC prior (the token tail's coder model, shipped in the archive's `hpac` member and materialized identically at both ends)
descends from cl2's move-26 fit; the token field moved at move 32, so for thirteen moves the coder's model was fit to a field that
no longer existed. ddm_hpr1 retrained the mixer in its SHIPPED geometry for 60 epochs (cl2's law), warm-started from the shipped
prior itself: the `hpac` member grows 11,911 → 12,262 B (+351 B; the refitted weights are video-derived and every one is COUNTED in
the archive; no shape bit ships — the field's geometry lives in receiver code and does not move), while the RLC1 stream it
conditions falls 119,749 → 118,511 B (−1,238 B): archive 179,359 B, −887 B against move 45 and −642 B against move 46 (whose
even-rounding of the same frame embedding this retrain replaces; the two edits do not add). Output-lossless by receipt: the decoded
token field is byte-identical to the shipped field (a92e7d90…) and the cold n600 public decode is byte-identical to the pointer's
raw (2b762eba…), so d_seg and d_pose are unchanged and the only score change is the rate term. Receiver unchanged (behavior digest
equal), so the seal inherits move 46's measured T4 leg on the normal path.
