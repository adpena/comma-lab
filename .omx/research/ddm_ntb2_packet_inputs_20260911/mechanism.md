The HPAC section (11,911 B on move 45) is the coder's PRIOR for the token tail: a small model whose bytes are shipped in the archive
and materialized identically at both ends before any symbol is decoded, so it conditions the tail's code length without touching
a single decoded symbol. ddm_ntb2 rounded 2,320 of the prior's 4,800 frame-embedding values to the nearest even value: the HPAC
member shrinks 603 B (11,911 → 11,308) while the tail it conditions grows 358 B (119,749 → 120,107 stream), net −245 B, archive
180,246 → 180,001 B. Output-lossless by construction and by receipt: the decoded token field is byte-identical to the shipped
field, and the cold n600 public decode is byte-identical to the pointer's own raw (2b762eba…, 600 pairs, cache disabled), so
d_seg and d_pose are move 45's exactly and the only score change is the rate term, 25·(−245)/37,545,489 = −1.6314e-4. The receiver
is unchanged (behavior digest 9f6e7168… equal to move 44/45's); the move went through the first-measurement chain only because an
inherited decode-wall-clock leg cannot be inherited a second time, and its completion gives move 46 a measured T4 leg of its own.
