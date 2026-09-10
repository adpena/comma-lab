# RLC1 native source audit

Research only; no score or seal claim. The candidate compiles `runtime/rlc1_geometry.c`
through its literal `inflate.sh` with the available C compiler. No binary is shipped.
The native module imports only integer, allocation and memory standard headers;
it has no file/network access, learned table, float operation or scorer dependency.
Its only data arguments are the counted 17-field config, previous decoded labels,
and newly observed groups. Public geometry, Q16, fixed indexing and safe arithmetic
limits are the remaining constants. Every numeric literal in the changed receiver
files is enumerated in `embedded_constants_audit.txt`.

The reference oracle is `experiments/ddm_rlc1_reference.py`. Native parity was tested
on 32 seeded random real frames with both full context arrays retained. The exact
signed-integer bounds and independent review are in `reference_review.md`. This
proves the tested geometry implementation; whole receiver identity and timing have
separate receipts. Normal submission declares CUDA/T4; advisory CPU testing does
not create a contest-CPU capability claim.
